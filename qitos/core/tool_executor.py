"""Batched Tool execution for the minimal agent loop.

Owns the proven execution invariants previously held by the Engine action
executor, now expressed over typed ``ToolCall`` / ``ToolResult``:

- every admitted call receives exactly one terminal ``ToolResult`` — unknown
  Tool, admission rejection, denial, timeout, cancellation and failure alike;
- duplicate ToolCall ids are rejected at batch admission, before any Tool
  side effect, so deterministic journal record ids stay unique;
- Tools run serially by default; in ``parallel`` mode argument validation,
  permission checks and ``before_tool_call`` hooks run sequentially in input
  order (Pi's preflight), and only prepared handler invocations share a
  bounded concurrent segment — exclusive calls act as barriers and results
  always commit in input order;
- each call gets one absolute monotonic deadline — the earlier of the run
  deadline and the Tool's ``timeout_s`` — propagated down to invocation,
  hooks and retry backoff;
- ``CancelToken`` mode ``"immediate"`` interrupts at the next safe point;
  ``"after_step"`` never interrupts an in-flight call — it is honored by the
  agent loop at the turn boundary instead;
- external task cancellation terminalizes every started or admitted call before
  re-raising ``asyncio.CancelledError`` when canonical appends settle; an append
  failure or unknown outcome stops the writer for close-and-replay recovery;
- a per-Tool ``RetryPolicy`` is the single retry owner;
- ``before_tool_call`` receives the validated arguments and may block;
  ``after_tool_call`` applies a field-level partial override. Hook
  contexts carry the assistant message, an immutable agent-context snapshot,
  ``is_error`` and the active cancel/deadline runtime (Pi parity); hooks are
  bounded by that runtime so a hung hook cannot block abort or the deadline.

The executor never raises for Tool-level failures. Persistence faults from
the transaction boundary (including ``JournalAppendCancelled``) and caller
cancellation propagate.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import inspect
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Literal, Mapping, Optional, Protocol, Sequence, Set, Tuple, Union

from ._freeze import thaw_deep
from .agent_events import (
    EventSink,
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolExecutionUpdate,
    emit_to,
)
from .cancellation import CancelSignalView, CancelToken
from .env import Env
from .journal import JournalAppendCancelled
from .journal import JournalCommitState
from .message import AssistantMessage, Message, ToolCall
from .tool import BaseTool, ToolValidationResult
from .tool_registry import ToolExposure
from .tool_result import ToolResult
from .tool_schema import tool_input_schema_errors

_FAILED_STATUSES = frozenset({"error", "timed_out"})

#: Sentinel distinguishing "override field not provided" from an explicit
#: ``None`` value in :class:`AfterToolCallOverride`.
UNSET: Any = object()


class ToolTransactionBoundary(Protocol):
    """Durable barriers around Tool admission and finalization."""

    async def tool_started(self, turn: int, call: ToolCall) -> None:
        """Record that the loop took responsibility for one Tool call."""
        ...

    async def tool_terminal(self, turn: int, call: ToolCall, result: ToolResult) -> None:
        """Record the unique terminal result of one Tool call."""
        ...


class _ToolEventSinkFault(Exception):
    """Keep observational listener faults out of Tool failure handling."""

    def __init__(self, cause: Exception) -> None:
        super().__init__(str(cause))
        self.cause = cause


class _ToolImmediateCancelled(Exception):
    """Internal signal that the run token interrupted a Tool invocation."""


class _ToolHandlerCancelled(Exception):
    """Internal signal that a handler raised CancelledError on its own."""


@dataclass(frozen=True, slots=True)
class AgentContextSnapshot:
    """Immutable view of the agent context at one turn boundary."""

    system_prompt: str = ""
    messages: Tuple[Message, ...] = ()
    tools: Optional[ToolExposure] = None
    env: Optional[Env] = None


@dataclass(frozen=True, slots=True)
class ToolHookRuntime:
    """Active cancellation/deadline view handed to Tool hooks (Pi's signal).

    Hooks receive the cooperative token and absolute deadline and must honor
    them. The executor bounds every hook await by this runtime and owns the
    cancelled hook task until it settles.
    """

    cancel_signal: Optional[CancelSignalView] = None
    deadline_monotonic: Optional[float] = None

    def remaining_seconds(self) -> Optional[float]:
        if self.deadline_monotonic is None:
            return None
        return max(0.0, self.deadline_monotonic - time.monotonic())


@dataclass(frozen=True, slots=True)
class BeforeToolCallContext:
    tool_call: ToolCall
    args: Mapping[str, Any]
    assistant_message: AssistantMessage
    agent_context: AgentContextSnapshot
    runtime: ToolHookRuntime
    turn: int
    run_id: str


@dataclass(frozen=True, slots=True)
class BeforeToolCallDecision:
    """Pi-aligned admission override for rejecting a call before execution."""

    block: bool = False
    reason: str = ""
    terminate: bool = False


@dataclass(frozen=True, slots=True)
class AfterToolCallContext:
    tool_call: ToolCall
    args: Mapping[str, Any]
    result: ToolResult
    is_error: bool
    assistant_message: AssistantMessage
    agent_context: AgentContextSnapshot
    runtime: ToolHookRuntime
    turn: int
    run_id: str


@dataclass(frozen=True, slots=True)
class AfterToolCallOverride:
    """Field-level partial override of an executed ToolResult (Pi parity).

    Fields left at ``UNSET`` keep the executed result's value; provided
    fields replace their counterpart in full (no deep merge). ``terminate``
    maps to the ``terminate`` metadata flag consumed by the loop's batch
    early-termination rule.
    """

    output: Any = UNSET
    model_output: Any = UNSET
    error: Any = UNSET
    status: Any = UNSET
    metadata: Any = UNSET
    terminate: Any = UNSET


BeforeToolCallHook = Callable[
    [BeforeToolCallContext],
    Union[BeforeToolCallDecision, None, Awaitable[Optional[BeforeToolCallDecision]]],
]
AfterToolCallHook = Callable[
    [AfterToolCallContext],
    Union[AfterToolCallOverride, None, Awaitable[Optional[AfterToolCallOverride]]],
]


@dataclass(frozen=True, slots=True)
class ToolExecutionConfig:
    """Immutable execution policy for one Tool batch."""

    mode: Literal["sequential", "parallel"] = "sequential"
    max_concurrency: int = 8
    fail_fast: bool = False
    deadline_monotonic: float | None = None
    cancel_token: CancelToken | None = None
    run_id: str = ""
    turn: int = 0
    env: Env | None = None
    before_tool_call: BeforeToolCallHook | None = None
    after_tool_call: AfterToolCallHook | None = None
    assistant_message: Optional[AssistantMessage] = None
    agent_context: Optional[AgentContextSnapshot] = None
    extra_runtime_context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in ("sequential", "parallel"):
            raise ValueError("mode must be 'sequential' or 'parallel'")
        if isinstance(self.max_concurrency, bool) or self.max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")


@dataclass(slots=True)
class _PreparedCall:
    """One call that passed sequential preflight and may be invoked."""

    index: int
    call: ToolCall
    tool: BaseTool
    effective_args: Dict[str, Any]
    runtime_context: Dict[str, Any]
    deadline: Optional[float]
    timeout_source: str
    ordering_meta: Dict[str, Any]
    start: float


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class ToolBatchExecutor:
    """Execute one assistant message's Tool calls against a frozen exposure."""

    def __init__(
        self,
        exposure: ToolExposure,
        config: ToolExecutionConfig,
        *,
        emit: EventSink | None = None,
        transaction: ToolTransactionBoundary | None = None,
    ) -> None:
        if not isinstance(exposure, ToolExposure):
            raise TypeError("ToolBatchExecutor requires a frozen ToolExposure")
        self._exposure = exposure
        self._config = config
        self._emit = emit
        self._transaction = transaction
        self._results: List[Optional[ToolResult]] = []
        self._started: Set[int] = set()
        self._committed: Set[int] = set()
        self._defer_terminal_commits = False
        self._journal_outcome_unknown = False
        self._handler_cancelled_error_seen = False
        self._events_failed = False
        #: Ordered terminal results of the last batch, including the results
        #: committed while an external cancellation was being drained. The
        #: loop reads them to keep the transcript paired when the batch task
        #: is cancelled.
        self.last_batch_results: Optional[List[ToolResult]] = None

    async def execute_batch(self, calls: Sequence[ToolCall]) -> List[ToolResult]:
        """Execute one batch and return terminal results in input order."""

        if not calls:
            return []
        call_ids = [call.id for call in calls]
        if len(set(call_ids)) != len(call_ids):
            raise ValueError(
                "ToolCall ids must be unique within one assistant message"
            )
        self._results = [None] * len(calls)
        self._started = set()
        self._committed = set()
        # Parallel handlers may finish in any order, but their canonical
        # Tool terminal records and end events belong to the assistant
        # message's input order. Defer every terminal candidate in a
        # parallel batch until the ordered finalization pass.
        self._defer_terminal_commits = (
            self._config.mode == "parallel" and len(calls) > 1
        )
        self._journal_outcome_unknown = False
        self._handler_cancelled_error_seen = False
        self._events_failed = False
        try:
            if self._immediate():
                for index, call in enumerate(calls):
                    self._results[index] = await self._prevented(
                        index, call, "cancel_token"
                    )
                return await self._final_results(calls)
            runnable = await self._admit(calls)
            if runnable:
                if len(runnable) == 1 or self._config.mode == "sequential":
                    await self._run_serial(runnable)
                else:
                    await self._run_segmented(runnable)
            if self._handler_cancelled_error_seen:
                # A swallowed CancelledError deep in the stack must still
                # surface once every admitted call reached a terminal state.
                raise asyncio.CancelledError()
            return await self._final_results(calls)
        except _ToolEventSinkFault as fault:
            # Event listeners are observational. Once one fails, stop before
            # further Tool side effects, terminalize every call without
            # dispatching more events, and only then surface the listener
            # fault to the loop/façade.
            for index, call in enumerate(calls):
                if self._results[index] is None:
                    self._results[index] = await self._prevented(
                        index, call, "event_sink_fault"
                    )
            self.last_batch_results = await self._final_results(calls)
            raise fault.cause
        except asyncio.CancelledError:
            if self._journal_outcome_unknown:
                # The canonical writer must be closed and replayed before
                # anyone can decide which Tool boundary exists. Do not append
                # a guessed counterpart or a Run terminal on this instance.
                self.last_batch_results = None
                raise
            # External task cancellation: every admitted call reaches a
            # terminal state before the cancellation propagates.
            for index, call in enumerate(calls):
                if self._results[index] is None:
                    self._results[index] = await self._prevented(
                        index, call, "caller_cancelled"
                    )
            self.last_batch_results = await self._final_results(calls)
            raise

    # ── batch admission ─────────────────────────────────────────────────

    async def _admit(self, calls: Sequence[ToolCall]) -> List[Tuple[int, ToolCall]]:
        """Return calls admitted by the already-validated batch identity."""

        return list(enumerate(calls))

    # ── batch strategies ────────────────────────────────────────────────

    async def _run_serial(self, runnable: List[Tuple[int, ToolCall]]) -> None:
        aborted: Optional[str] = None
        for index, call in runnable:
            if aborted is None and self._immediate():
                aborted = "cancel_token"
            if aborted is not None:
                self._results[index] = await self._prevented(index, call, aborted)
                continue
            outcome = await self._preflight(index, call)
            if isinstance(outcome, ToolResult):
                self._results[index] = outcome
            else:
                self._results[index] = await self._run_prepared(outcome)
            aborted = self._batch_abort_after(self._results[index]) or aborted

    def _batch_abort_after(self, result: Optional[ToolResult]) -> Optional[str]:
        if result is None:
            return None
        cancel_source = result.metadata.get("cancel_source")
        if isinstance(cancel_source, str) and cancel_source:
            return cancel_source
        if self._config.fail_fast and result.status in _FAILED_STATUSES:
            return "fail_fast"
        return None

    def _segment_prepared(self, prepared: List[_PreparedCall]) -> List[List[_PreparedCall]]:
        """Split into contiguous safe runs separated by exclusive barriers."""

        segments: List[List[_PreparedCall]] = []
        current: List[_PreparedCall] = []
        for item in prepared:
            if item.tool.spec.concurrency_safe is True and not item.tool.spec.needs_approval:
                current.append(item)
                continue
            if current:
                segments.append(current)
                current = []
            segments.append([item])
        if current:
            segments.append(current)
        return segments

    async def _run_segmented(self, runnable: List[Tuple[int, ToolCall]]) -> None:
        """Preflight sequentially in input order, then execute segments."""

        prepared: List[_PreparedCall] = []
        aborted: Optional[str] = None
        for index, call in runnable:
            if aborted is None and self._immediate():
                aborted = "cancel_token"
            if aborted is not None:
                self._results[index] = await self._prevented(index, call, aborted)
                continue
            outcome = await self._preflight(index, call)
            if isinstance(outcome, ToolResult):
                self._results[index] = outcome
                aborted = self._batch_abort_after(outcome) or aborted
            else:
                prepared.append(outcome)
        for segment in self._segment_prepared(prepared):
            if aborted is None and self._immediate():
                aborted = "cancel_token"
            if aborted is not None:
                for item in segment:
                    self._results[item.index] = await self._prevented(
                        item.index, item.call, aborted
                    )
                continue
            if len(segment) == 1:
                item = segment[0]
                self._results[item.index] = await self._run_prepared(item)
                aborted = self._batch_abort_after(self._results[item.index]) or aborted
            else:
                segment_abort = await self._run_segment_concurrently(segment)
                aborted = aborted or segment_abort
            if aborted is None and self._config.fail_fast:
                for item in segment:
                    result = self._results[item.index]
                    if result is not None and result.status in _FAILED_STATUSES:
                        aborted = "fail_fast"
                        break

    async def _run_segment_concurrently(
        self, segment: List[_PreparedCall]
    ) -> Optional[str]:
        """Run one safe segment in parallel, draining started calls to terminal."""

        max_concurrency = min(self._config.max_concurrency, len(segment))
        abort_reason: Optional[str] = None
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _run_one(item: _PreparedCall) -> ToolResult:
            try:
                async with semaphore:
                    if self._immediate():
                        return await self._prevented(item.index, item.call, "cancel_token")
                    return await self._run_prepared(item)
            except JournalAppendCancelled:
                raise
            except asyncio.CancelledError:
                # Sibling drain or external cancellation: this child settles
                # its own terminal state; the parent decides whether the
                # batch itself was cancelled.
                existing = self._results[item.index]
                if existing is not None:
                    return existing
                return await self._prevented(item.index, item.call, "parent_cancelled")

        tasks = {
            asyncio.create_task(
                _run_one(item), name=f"qitos-tool-{item.call.name}"
            ): item
            for item in segment
        }
        try:
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    item = tasks[task]
                    try:
                        self._results[item.index] = task.result()
                    except _ToolEventSinkFault:
                        raise
                    except JournalAppendCancelled:
                        raise
                    except asyncio.CancelledError:  # pragma: no cover - defensive
                        self._results[item.index] = await self._prevented(
                            item.index, item.call, abort_reason or "parent_cancelled"
                        )
                    except Exception as exc:  # pragma: no cover - defensive path
                        self._results[item.index] = await self._terminal_result(
                            item.index,
                            item.call,
                            self._missing_result(item.call, str(exc)),
                        )
                    result = self._results[item.index]
                    if (
                        abort_reason is None
                        and self._config.fail_fast
                        and result is not None
                        and result.status in _FAILED_STATUSES
                    ):
                        abort_reason = "fail_fast"
                if abort_reason is None and self._immediate():
                    abort_reason = "cancel_token"
                if abort_reason is not None and pending:
                    pending_tasks = list(pending)
                    for task in pending_tasks:
                        task.cancel()
                    drained = await asyncio.gather(*pending_tasks, return_exceptions=True)
                    for pending_task, outcome in zip(pending_tasks, drained):
                        item = tasks[pending_task]
                        if isinstance(outcome, ToolResult):
                            self._results[item.index] = outcome
                        elif isinstance(outcome, JournalAppendCancelled):
                            raise outcome
                        elif self._results[item.index] is None:
                            self._results[item.index] = await self._prevented(
                                item.index, item.call, abort_reason
                            )
                    pending.clear()
        except asyncio.CancelledError:
            abort_reason = (
                "cancel_token" if self._immediate() else "caller_cancelled"
            )
            pending_tasks = [task for task in tasks if not task.done()]
            for task in pending_tasks:
                task.cancel()
            if pending_tasks:
                drained = await asyncio.gather(*pending_tasks, return_exceptions=True)
                for pending_task, outcome in zip(pending_tasks, drained):
                    item = tasks[pending_task]
                    if isinstance(outcome, ToolResult):
                        self._results[item.index] = outcome
                    elif isinstance(outcome, JournalAppendCancelled):
                        raise outcome
                    elif self._results[item.index] is None:
                        self._results[item.index] = await self._prevented(
                            item.index, item.call, abort_reason
                        )
            for task, item in tasks.items():
                if (
                    self._results[item.index] is None
                    and task.done()
                    and not task.cancelled()
                ):
                    self._results[item.index] = task.result()
            raise
        finally:
            unfinished = [task for task in tasks if not task.done()]
            for task in unfinished:
                task.cancel()
            if unfinished:
                await asyncio.gather(*unfinished, return_exceptions=True)
        return abort_reason

    # ── sequential preflight ────────────────────────────────────────────

    async def _start_call(
        self, index: int, call: ToolCall, *, journal: bool = True
    ) -> None:
        if self._transaction is not None and journal:
            try:
                await self._transaction.tool_started(self._config.turn, call)
            except JournalAppendCancelled as exc:
                if exc.commit_state is JournalCommitState.COMMITTED:
                    self._started.add(index)
                elif exc.commit_state is JournalCommitState.UNKNOWN:
                    self._journal_outcome_unknown = True
                raise
            except asyncio.CancelledError:
                # A plain cancellation provides no committed admission
                # evidence. The cancellation drain may retry the admission as
                # an explicit cancelled call, but the handler never runs.
                raise
        self._started.add(index)
        await self._emit_event(
            ToolExecutionStart(
                tool_call_id=call.id, tool_name=call.name, args=call.arguments
            )
        )

    async def _commit_terminal(
        self, index: int, call: ToolCall, result: ToolResult, *, journal: bool = True
    ) -> ToolResult:
        """Freeze and record the unique terminal result through all barriers."""

        final = dataclasses.replace(result, call_id=call.id).frozen()
        if self._transaction is not None and journal:
            try:
                await self._transaction.tool_terminal(self._config.turn, call, final)
            except JournalAppendCancelled as exc:
                self._results[index] = final
                if exc.commit_state is JournalCommitState.COMMITTED:
                    self._committed.add(index)
                elif exc.commit_state is JournalCommitState.UNKNOWN:
                    self._journal_outcome_unknown = True
                raise
            except asyncio.CancelledError:
                # No durable outcome was provided. Preserve the already-run
                # handler's candidate, but do not infer terminal persistence.
                self._results[index] = final
                raise
        # The canonical result wins before an observational listener runs.
        # If that listener fails, batch recovery sees this slot as terminal
        # and only cancels calls that have not started yet.
        self._results[index] = final
        self._committed.add(index)
        await self._emit_event(
            ToolExecutionEnd(
                tool_call_id=call.id,
                tool_name=call.name,
                result=final,
                is_error=final.status != "success",
            )
        )
        return final

    async def _terminal_result(
        self, index: int, call: ToolCall, result: ToolResult, *, journal: bool = True
    ) -> ToolResult:
        """Return one terminal candidate or commit it immediately.

        Parallel batches keep handler execution concurrent while deferring the
        canonical transaction and end event to ``_final_results``. Serial
        batches retain their existing per-call durability boundary.
        """

        if not self._defer_terminal_commits:
            return await self._commit_terminal(
                index, call, result, journal=journal
            )
        final = dataclasses.replace(result, call_id=call.id).frozen()
        self._results[index] = final
        return final

    async def _emit_event(self, event: Any) -> None:
        if self._events_failed:
            return
        try:
            await emit_to(self._emit, event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._events_failed = True
            raise _ToolEventSinkFault(exc) from exc

    async def _preflight(
        self, index: int, call: ToolCall
    ) -> Union[_PreparedCall, ToolResult]:
        """Sequential admission: validation, permission and the before-hook.

        This phase never runs concurrently (Pi's parallel preflight order);
        only prepared handler invocations may overlap.
        """

        await self._start_call(index, call)
        start = time.monotonic()
        deadline, timeout_source = self._resolve_call_deadline(call, start)

        stopped = self._stop_result(call, start, attempts=0, deadline=deadline)
        if stopped is not None:
            return await self._terminal_result(index, call, stopped)

        if call.parse_error is not None:
            return await self._terminal_result(
                index,
                call,
                self._finish_result(
                    call,
                    status="error",
                    start=start,
                    attempts=0,
                    error=call.parse_error,
                    extra_metadata={
                        "error_category": "invalid_tool_arguments",
                        "recoverable": True,
                        "started": False,
                    },
                ),
            )

        tool = self._exposure.get(call.name)
        if tool is None:
            return await self._terminal_result(
                index,
                call,
                self._finish_result(
                    call,
                    status="error",
                    start=start,
                    attempts=0,
                    error=f"Unknown tool: {call.name}",
                    extra_metadata={
                        "error_category": "tool_not_found",
                        "available_tools": self._exposure.list_tools(),
                        "recoverable": True,
                        "started": False,
                    },
                ),
            )

        runtime_context = self._build_runtime_context(
            call, tool, deadline_monotonic=deadline
        )
        ordering_meta: Dict[str, Any] = {"started": False}
        if timeout_source != "none":
            ordering_meta["timeout_source"] = timeout_source

        stopped = self._stop_result(call, start, attempts=0, deadline=deadline)
        if stopped is not None:
            return await self._terminal_result(index, call, stopped)

        permission = tool.check_permissions(dict(call.arguments), runtime_context)
        if permission.decision == "deny":
            return await self._terminal_result(
                index,
                call,
                self._finish_result(
                    call,
                    status="denied",
                    start=start,
                    attempts=0,
                    error=permission.message or "Tool permission denied",
                    output={"message": permission.message, "scope": permission.scope},
                    extra_metadata={
                        **ordering_meta,
                        "error_category": "permission_denied",
                        "permission_scope": permission.scope,
                    },
                ),
            )
        if permission.decision == "ask":
            return await self._terminal_result(
                index,
                call,
                self._finish_result(
                    call,
                    status="needs_approval",
                    start=start,
                    attempts=0,
                    output={"message": permission.message, "scope": permission.scope},
                    extra_metadata={
                        **ordering_meta,
                        "error_category": "permission_ask",
                        "permission_scope": permission.scope,
                    },
                ),
            )

        effective_args = dict(
            thaw_deep(
                call.arguments
                if permission.updated_args is None
                else permission.updated_args
            )
        )
        validation = self._validate(tool, effective_args, runtime_context)
        if not validation.valid:
            return await self._terminal_result(
                index,
                call,
                self._finish_result(
                    call,
                    status="error",
                    start=start,
                    attempts=0,
                    error=validation.message or "tool input validation failed",
                    extra_metadata={
                        **ordering_meta,
                        "error_category": validation.code or "validation_error",
                        "started": False,
                    },
                ),
            )

        stopped = self._stop_result(call, start, attempts=0, deadline=deadline)
        if stopped is not None:
            return await self._terminal_result(index, call, stopped)

        if self._config.before_tool_call is not None:
            decision, hook_failure = await self._run_before_hook(
                call, effective_args
            )
            if hook_failure is not None:
                return await self._terminal_result(index, call, hook_failure)
            if decision is not None and decision.block:
                extra: Dict[str, Any] = {
                    **ordering_meta,
                    "error_category": "before_tool_call_blocked",
                    "started": False,
                }
                if decision.terminate:
                    extra["terminate"] = True
                return await self._terminal_result(
                    index,
                    call,
                    self._finish_result(
                        call,
                        status="denied",
                        start=start,
                        attempts=0,
                        error=decision.reason or "Tool execution was blocked",
                        extra_metadata=extra,
                    ),
                )
        stopped = self._stop_result(call, start, attempts=0, deadline=deadline)
        if stopped is not None:
            return await self._terminal_result(index, call, stopped)

        return _PreparedCall(
            index=index,
            call=call,
            tool=tool,
            effective_args=effective_args,
            runtime_context=runtime_context,
            deadline=deadline,
            timeout_source=timeout_source,
            ordering_meta=ordering_meta,
            start=start,
        )

    # ── prepared execution ──────────────────────────────────────────────

    async def _run_prepared(self, prepared: _PreparedCall) -> ToolResult:
        """Invoke the handler (with retries) and apply the after-hook."""

        call = prepared.call
        result = await self._invoke_with_retries(
            prepared.tool,
            call,
            prepared.effective_args,
            runtime_context=prepared.runtime_context,
            start=prepared.start,
            deadline=prepared.deadline,
            timeout_source=prepared.timeout_source,
            ordering_meta=prepared.ordering_meta,
        )

        if self._config.after_tool_call is not None:
            result = await self._run_after_hook(prepared, result)
        return await self._terminal_result(prepared.index, call, result)

    async def _run_before_hook(
        self, call: ToolCall, effective_args: Dict[str, Any]
    ) -> Tuple[Optional[BeforeToolCallDecision], Optional[ToolResult]]:
        """Run the bounded before-hook; return a decision or a terminal failure."""

        hook = self._config.before_tool_call
        assert hook is not None
        start = time.monotonic()
        status, value = await self._await_hook(
            hook(
                BeforeToolCallContext(
                    tool_call=call,
                    args=dict(effective_args),
                    assistant_message=self._assistant_message(call),
                    agent_context=self._agent_context(),
                    runtime=self._hook_runtime(),
                    turn=self._config.turn,
                    run_id=self._config.run_id,
                )
            )
        )
        if status == "cancelled":
            return None, self._finish_result(
                call,
                status="cancelled",
                start=start,
                attempts=0,
                error="tool call cancelled during before_tool_call",
                extra_metadata={
                    "error_category": "cancelled",
                    "cancel_source": "cancel_token",
                    "started": False,
                },
            )
        if status == "timed_out":
            return None, self._finish_result(
                call,
                status="timed_out",
                start=start,
                attempts=0,
                error="before_tool_call deadline expired",
                extra_metadata={"error_category": "timeout", "started": False},
            )
        if isinstance(value, BaseException):
            return None, self._finish_result(
                call,
                status="error",
                start=start,
                attempts=0,
                error=str(value),
                extra_metadata={
                    "error_category": "before_tool_call",
                    "started": False,
                },
            )
        if value is not None and not isinstance(value, BeforeToolCallDecision):
            return None, self._finish_result(
                call,
                status="error",
                start=start,
                attempts=0,
                error="before_tool_call must return a BeforeToolCallDecision or None",
                extra_metadata={
                    "error_category": "before_tool_call",
                    "started": False,
                },
            )
        return value, None

    async def _run_after_hook(
        self, prepared: _PreparedCall, result: ToolResult
    ) -> ToolResult:
        """Run the bounded after-hook and merge its field-level override."""

        hook = self._config.after_tool_call
        assert hook is not None
        call = prepared.call
        status, value = await self._await_hook(
            hook(
                AfterToolCallContext(
                    tool_call=call,
                    args=dict(prepared.effective_args),
                    result=result,
                    is_error=result.status != "success",
                    assistant_message=self._assistant_message(call),
                    agent_context=self._agent_context(),
                    runtime=self._hook_runtime(),
                    turn=self._config.turn,
                    run_id=self._config.run_id,
                )
            )
        )
        if status in ("cancelled", "timed_out"):
            # The Tool already executed; its real result stays the truth and
            # the lost override is recorded instead of hanging the run.
            metadata = dict(thaw_deep(result.metadata))
            metadata["after_tool_call"] = status
            return dataclasses.replace(result, metadata=metadata)
        if isinstance(value, BaseException):
            return self._finish_result(
                call,
                status="error",
                start=prepared.start,
                attempts=int(result.metadata.get("attempts", 0) or 0),
                error=str(value),
                extra_metadata={"error_category": "after_tool_call"},
            )
        if value is None:
            return result
        if not isinstance(value, AfterToolCallOverride):
            raise TypeError("after_tool_call must return an AfterToolCallOverride or None")
        return self._apply_override(result, value)

    @staticmethod
    def _apply_override(result: ToolResult, override: AfterToolCallOverride) -> ToolResult:
        changes: Dict[str, Any] = {}
        metadata: Dict[str, Any] = dict(thaw_deep(result.metadata))
        if override.output is not UNSET:
            changes["output"] = override.output
        if override.model_output is not UNSET:
            changes["model_output"] = override.model_output
        if override.error is not UNSET:
            changes["error"] = override.error
        if override.status is not UNSET:
            changes["status"] = override.status
        if override.metadata is not UNSET:
            if not isinstance(override.metadata, Mapping):
                raise TypeError("after_tool_call metadata override must be a mapping")
            metadata = dict(override.metadata)
        if override.terminate is not UNSET:
            metadata["terminate"] = bool(override.terminate)
        changes["metadata"] = metadata
        return dataclasses.replace(result, **changes)

    async def _await_hook(self, value: Any) -> Tuple[str, Any]:
        """Await one hook bounded by the cancel token and absolute deadline.

        Returns ``("done", value)``, ``("cancelled", None)`` or
        ``("timed_out", None)``. Hook exceptions arrive as ``("done", exc)``;
        caller cancellation of the batch task propagates.
        """

        if not inspect.isawaitable(value):
            return "done", value

        async def _await_value() -> Any:
            return await value

        hook_task = asyncio.create_task(_await_value(), name="qitos-tool-hook")
        tasks: Set[asyncio.Task[Any]] = {hook_task}
        watcher: Optional[asyncio.Task[bool]] = None
        token = self._config.cancel_token
        if token is not None:
            watcher = asyncio.create_task(token.wait_immediate())
            tasks.add(watcher)
        remaining = self._hook_runtime().remaining_seconds()
        try:
            done, _pending = await asyncio.wait(
                tasks, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError:
            hook_task.cancel()
            if watcher is not None:
                watcher.cancel()
            await asyncio.gather(
                *(task for task in (hook_task, watcher) if task is not None),
                return_exceptions=True,
            )
            raise
        if hook_task in done:
            if watcher is not None:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
            try:
                return "done", hook_task.result()
            except asyncio.CancelledError:
                return "cancelled", None
            except Exception as exc:
                return "done", exc
        outcome = "cancelled" if (watcher is not None and watcher in done) else "timed_out"
        hook_task.cancel()
        if watcher is not None and not watcher.done():
            watcher.cancel()
        await asyncio.gather(
            *(task for task in (hook_task, watcher) if task is not None),
            return_exceptions=True,
        )
        return outcome, None

    def _assistant_message(self, call: ToolCall) -> AssistantMessage:
        configured = self._config.assistant_message
        if configured is not None:
            return configured
        return AssistantMessage(tool_calls=(call,))

    def _agent_context(self) -> AgentContextSnapshot:
        configured = self._config.agent_context
        if configured is not None:
            return configured
        return AgentContextSnapshot(tools=self._exposure, env=self._config.env)

    def _hook_runtime(self) -> ToolHookRuntime:
        return ToolHookRuntime(
            cancel_signal=(
                None
                if self._config.cancel_token is None
                else self._config.cancel_token.signal
            ),
            deadline_monotonic=self._config.deadline_monotonic,
        )

    async def _invoke_with_retries(
        self,
        tool: BaseTool,
        call: ToolCall,
        effective_args: Dict[str, Any],
        *,
        runtime_context: Dict[str, Any],
        start: float,
        deadline: Optional[float],
        timeout_source: str,
        ordering_meta: Dict[str, Any],
    ) -> ToolResult:
        retry_policy = tool.spec.retry_policy
        max_attempts = retry_policy.max_attempts if retry_policy is not None else 1
        retryable_exceptions = (
            retry_policy.retryable_exceptions if retry_policy is not None else ()
        )
        attempts = 0
        last_error: Optional[str] = None
        while attempts < max_attempts:
            stopped = self._stop_result(
                call, start, attempts=attempts, deadline=deadline
            )
            if stopped is not None:
                return stopped
            attempts += 1
            ordering_meta["started"] = True
            self._set_progress_accepting(runtime_context, True)
            try:
                try:
                    output = await self._invoke_tool(
                        tool,
                        effective_args,
                        runtime_context=runtime_context,
                        timeout_s=self._remaining_seconds(deadline),
                    )
                finally:
                    # ToolExecutionUpdate events mirror Pi's tool progress
                    # callback. They are scheduled immediately by the
                    # callback and settle before the attempt can terminalize.
                    self._set_progress_accepting(runtime_context, False)
                    await self._drain_progress_updates(runtime_context)
                if self._immediate():
                    return self._finish_result(
                        call,
                        status="cancelled",
                        start=start,
                        attempts=attempts,
                        error="tool call cancelled",
                        extra_metadata={
                            **ordering_meta,
                            "error_category": "cancelled",
                            "cancel_source": "cancel_token",
                        },
                    )
                if output is None:
                    return self._finish_result(
                        call,
                        status="error",
                        start=start,
                        attempts=attempts,
                        error="Tool returned no result",
                        extra_metadata={
                            "error_category": "tool_result_missing",
                            "error_code": "TOOL_RESULT_MISSING",
                            "recoverable": True,
                            "started": True,
                        },
                    )
                reported = ToolResult.from_value(output)
                latency_ms = (time.monotonic() - start) * 1000
                reported_category = reported.metadata.get("error_category")
                metadata = {
                    **reported.metadata,
                    **ordering_meta,
                    "error_category": (
                        None
                        if reported.is_success
                        else reported_category
                        if isinstance(reported_category, str)
                        else f"tool_reported_{reported.status}"
                    ),
                    "latency_ms": latency_ms,
                    "attempts": attempts,
                    "progress_count": len(runtime_context["progress_events"]),
                    "artifacts": list(runtime_context["artifacts"]),
                }
                return ToolResult(
                    status=reported.status,
                    output=reported.output,
                    error=reported.error,
                    metadata=metadata,
                    artifacts=reported.artifacts,
                    model_output=reported.model_output,
                    usage=reported.usage,
                    added_tool_names=reported.added_tool_names,
                )
            except _ToolEventSinkFault:
                raise
            except JournalAppendCancelled:
                raise
            except _ToolImmediateCancelled:
                return self._finish_result(
                    call,
                    status="cancelled",
                    start=start,
                    attempts=attempts,
                    error="tool call cancelled",
                    extra_metadata={
                        **ordering_meta,
                        "error_category": "cancelled",
                        "cancel_source": "cancel_token",
                    },
                )
            except _ToolHandlerCancelled:
                # A handler may raise CancelledError without cancellation
                # having been requested on its owning executor task. Preserve
                # the terminal ToolResult, then re-raise from execute_batch.
                self._handler_cancelled_error_seen = True
                # The terminal cancelled result is committed by the caller;
                # batch-level cancellation then re-raises from execute_batch,
                # so the cancellation itself is never silently swallowed.
                return self._finish_result(
                    call,
                    status="cancelled",
                    start=start,
                    attempts=attempts,
                    error="tool call cancelled",
                    extra_metadata={
                        **ordering_meta,
                        "error_category": "cancelled",
                        "cancel_source": "handler_cancelled",
                    },
                )
            except asyncio.CancelledError:
                # Cancellation of the task that owns this invocation is not a
                # handler result. Let the serial/parallel batch owner drain and
                # terminalize every call before it re-raises.
                raise
            except asyncio.TimeoutError as exc:
                return self._finish_result(
                    call,
                    status="timed_out",
                    start=start,
                    attempts=attempts,
                    error=str(exc) or "tool call deadline expired",
                    extra_metadata={
                        **ordering_meta,
                        "error_category": "timeout",
                        "timeout_source": timeout_source,
                    },
                )
            except Exception as exc:
                last_error = str(exc)
                if not isinstance(exc, retryable_exceptions):
                    break
                if (
                    retry_policy is not None
                    and attempts < max_attempts
                    and retry_policy.backoff_factor > 0
                ):
                    delay = min(
                        retry_policy.backoff_factor * (2 ** (attempts - 1)),
                        retry_policy.max_backoff,
                    )
                    if retry_policy.jitter:
                        delay = delay * (0.5 + random.random())
                    remaining = self._remaining_seconds(deadline)
                    if remaining is not None and delay >= remaining:
                        return self._finish_result(
                            call,
                            status="timed_out",
                            start=start,
                            attempts=attempts,
                            error="tool call deadline expired before retry",
                            extra_metadata={
                                **ordering_meta,
                                "error_category": "timeout",
                                "timeout_source": timeout_source,
                            },
                        )
                    retry_cancel = await self._wait_for_retry(
                        delay, deadline_monotonic=deadline
                    )
                    if retry_cancel is not None:
                        return self._finish_result(
                            call,
                            status="cancelled",
                            start=start,
                            attempts=attempts,
                            error="tool call cancelled during retry backoff",
                            extra_metadata={
                                **ordering_meta,
                                "error_category": "cancelled",
                                "cancel_source": retry_cancel,
                            },
                        )

        if tool.spec.on_failure is not None:
            try:
                tool.spec.on_failure(
                    action={"name": call.name, "args": effective_args},
                    error=last_error,
                    attempts=attempts,
                )
            except Exception:
                pass  # on_failure must not raise
        return self._finish_result(
            call,
            status="error",
            start=start,
            attempts=attempts,
            error=last_error or "unknown tool execution error",
            extra_metadata={
                **ordering_meta,
                "error_category": "runtime_error",
                "progress_count": len(runtime_context["progress_events"]),
                "artifacts": list(runtime_context["artifacts"]),
            },
        )

    # ── admission helpers ───────────────────────────────────────────────

    def _validate(
        self,
        tool: BaseTool,
        args: Dict[str, Any],
        runtime_context: Dict[str, Any],
    ) -> ToolValidationResult:
        schema_errors = tool_input_schema_errors(
            tool.spec.input_schema or {}, dict(args)
        )
        if schema_errors:
            return ToolValidationResult.fail(
                "Tool arguments do not match input_schema:\n"
                + "\n".join(f"- {error}" for error in schema_errors),
                code="invalid_tool_arguments",
            )
        return tool.validate_input(dict(args), runtime_context=runtime_context)

    def _build_runtime_context(
        self,
        call: ToolCall,
        tool: BaseTool,
        *,
        deadline_monotonic: Optional[float],
    ) -> Dict[str, Any]:
        env = self._config.env
        progress_events: List[Dict[str, Any]] = []
        progress_updates: List[concurrent.futures.Future[None]] = []
        progress_lock = threading.Lock()
        progress_accepting = [False]
        loop = asyncio.get_running_loop()
        artifacts: List[Dict[str, Any]] = []

        def _emit_progress(payload: Dict[str, Any]) -> None:
            update = dict(payload)
            with progress_lock:
                if not progress_accepting[0]:
                    return
                progress_events.append(update)
                future = asyncio.run_coroutine_threadsafe(
                    self._emit_event(
                        ToolExecutionUpdate(
                            tool_call_id=call.id,
                            tool_name=call.name,
                            args=call.arguments,
                            partial_result=update,
                        )
                    ),
                    loop,
                )
                progress_updates.append(future)

        def _record_artifact(payload: Dict[str, Any]) -> None:
            artifacts.append(dict(payload))

        # Product context may add domain values (for example
        # ``permission_context`` or Subagent lineage), but it must never replace
        # the executor-owned authority, deadline, cancellation or callback
        # capabilities for this frozen turn.
        context: Dict[str, Any] = dict(self._config.extra_runtime_context)
        context.update({
            "env": env,
            "ops": self._resolve_ops(tool, env),
            "tool_registry": self._exposure,
            "progress_events": progress_events,
            "artifacts": artifacts,
            "emit_progress": _emit_progress,
            "record_artifact": _record_artifact,
            "run_id": self._config.run_id,
            "tool_call_id": call.id,
            "deadline_monotonic": deadline_monotonic,
            "remaining_seconds": lambda: self._remaining_seconds(deadline_monotonic),
            "agent_cancelled": self._immediate,
        })
        context["progress_updates"] = progress_updates
        context["progress_lock"] = progress_lock
        context["progress_accepting"] = progress_accepting
        return context

    @staticmethod
    def _set_progress_accepting(
        runtime_context: Dict[str, Any], accepting: bool
    ) -> None:
        lock = runtime_context.get("progress_lock")
        state = runtime_context.get("progress_accepting")
        if lock is None or not isinstance(state, list):
            return
        with lock:
            state[0] = accepting

    async def _drain_progress_updates(
        self, runtime_context: Dict[str, Any]
    ) -> None:
        updates = runtime_context.get("progress_updates")
        lock = runtime_context.get("progress_lock")
        if not isinstance(updates, list) or lock is None:
            return
        while True:
            with lock:
                pending = list(updates)
                updates.clear()
            if not pending:
                return
            await asyncio.gather(
                *(asyncio.wrap_future(future) for future in pending)
            )

    def _resolve_ops(self, tool: BaseTool, env: Optional[Env]) -> Dict[str, Any]:
        required = list(tool.spec.required_ops) + list(tool.spec.environment_ops)
        if not required:
            return {}
        if env is None:
            raise ValueError(
                f"Tool '{tool.name}' requires ops {required} but no env was provided"
            )
        out: Dict[str, Any] = {}
        for group in required:
            ops = env.get_ops(group)
            if ops is None:
                raise ValueError(
                    f"Env '{getattr(env, 'name', 'env')}' missing required ops group: {group}"
                )
            out[group] = ops
        return out

    # ── deadlines, cancellation, results ────────────────────────────────

    def _immediate(self) -> bool:
        """Immediate-cancel check; ``after_step`` never interrupts a call."""

        token = self._config.cancel_token
        return token is not None and token.immediate_requested

    def _resolve_call_deadline(
        self, call: ToolCall, started: float
    ) -> tuple[Optional[float], str]:
        tool = self._exposure.get(call.name)
        tool_timeout = (
            float(tool.spec.timeout_s)
            if tool is not None and tool.spec.timeout_s is not None
            else None
        )
        tool_deadline = started + tool_timeout if tool_timeout is not None else None
        run_deadline = self._config.deadline_monotonic
        if run_deadline is not None and (
            tool_deadline is None or run_deadline <= tool_deadline
        ):
            return float(run_deadline), "run_deadline"
        if tool_deadline is not None:
            return tool_deadline, "tool_spec"
        return None, "none"

    @staticmethod
    def _remaining_seconds(deadline_monotonic: Optional[float]) -> Optional[float]:
        if deadline_monotonic is None:
            return None
        return max(0.0, deadline_monotonic - time.monotonic())

    async def _invoke_tool(
        self,
        tool: BaseTool,
        args: Dict[str, Any],
        runtime_context: Optional[Dict[str, Any]],
        timeout_s: Optional[float],
    ) -> Any:
        if timeout_s is not None and timeout_s <= 0:
            raise asyncio.TimeoutError("tool call deadline expired")

        invocation = asyncio.create_task(
            tool.execute(args, runtime_context=runtime_context),
            name=f"qitos-tool-invoke-{tool.name}",
        )
        watcher: Optional[asyncio.Task[bool]] = None
        token = self._config.cancel_token
        waits: Set[asyncio.Task[Any]] = {invocation}
        if token is not None:
            watcher = asyncio.create_task(
                token.wait_immediate(), name=f"qitos-tool-cancel-{tool.name}"
            )
            waits.add(watcher)
        try:
            done, _pending = await asyncio.wait(
                waits,
                timeout=timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            invocation.cancel()
            if watcher is not None:
                watcher.cancel()
            await asyncio.gather(
                *[task for task in (invocation, watcher) if task is not None],
                return_exceptions=True,
            )
            raise

        if invocation in done:
            if watcher is not None:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
            try:
                return invocation.result()
            except asyncio.CancelledError as exc:
                raise _ToolHandlerCancelled from exc

        invocation.cancel()
        if watcher is not None and not watcher.done():
            watcher.cancel()
        await asyncio.gather(
            *[task for task in (invocation, watcher) if task is not None],
            return_exceptions=True,
        )
        if watcher is not None and watcher in done:
            raise _ToolImmediateCancelled
        raise asyncio.TimeoutError("tool call deadline expired")

    async def _wait_for_retry(
        self,
        delay: float,
        *,
        deadline_monotonic: Optional[float],
    ) -> str | None:
        """Wait out one retry backoff; cancellation propagates to the caller."""

        wake_at = time.monotonic() + max(0.0, delay)
        while True:
            if self._immediate():
                return "cancel_token"
            now = time.monotonic()
            if now >= wake_at:
                return None
            if deadline_monotonic is not None and now >= deadline_monotonic:
                return None
            remaining_delay = wake_at - now
            remaining_deadline = (
                None
                if deadline_monotonic is None
                else max(0.0, deadline_monotonic - now)
            )
            sleep_for = min(0.05, remaining_delay)
            if remaining_deadline is not None:
                sleep_for = min(sleep_for, remaining_deadline)
            if sleep_for <= 0:
                return None
            await asyncio.sleep(sleep_for)

    def _stop_result(
        self,
        call: ToolCall,
        start: float,
        *,
        attempts: int,
        deadline: Optional[float],
    ) -> Optional[ToolResult]:
        if self._immediate():
            return self._finish_result(
                call,
                status="cancelled",
                start=start,
                attempts=attempts,
                error="tool call cancelled",
                extra_metadata={
                    "error_category": "cancelled",
                    "cancel_source": "cancel_token",
                    "started": attempts > 0,
                },
            )
        remaining = self._remaining_seconds(deadline)
        if remaining is None or remaining > 0:
            return None
        return self._finish_result(
            call,
            status="timed_out",
            start=start,
            attempts=attempts,
            error="tool call deadline expired before admission",
            extra_metadata={
                "error_category": "timeout",
                "started": attempts > 0,
            },
        )

    async def _prevented(
        self, index: int, call: ToolCall, cancel_source: str
    ) -> ToolResult:
        """Commit a not-started call's cancelled terminal through all barriers."""

        if index not in self._started:
            await self._start_call(index, call)
        return await self._terminal_result(
            index, call, self._prevented_result(call, cancel_source)
        )

    def _prevented_result(self, call: ToolCall, cancel_source: str) -> ToolResult:
        """Terminal result for a call that was prevented from starting."""

        return ToolResult(
            status="cancelled",
            output=None,
            error=f"tool call cancelled: {cancel_source}",
            metadata={
                "tool_name": call.name,
                "error_category": "cancelled",
                "cancel_source": cancel_source,
                "started": False,
            },
        )

    def _missing_result(self, call: ToolCall, message: str) -> ToolResult:
        return ToolResult(
            status="error",
            output=(
                "[TOOL_RESULT_MISSING]\n\n"
                f"Tool: `{call.name}`\n"
                "Code: `TOOL_RESULT_MISSING`\n\n"
                "The executor did not produce a result. No success was inferred."
            ),
            error=message,
            metadata={
                "tool_name": call.name,
                "error_category": "executor_error",
                "error_code": "TOOL_RESULT_MISSING",
            },
        )

    async def _final_results(self, calls: Sequence[ToolCall]) -> List[ToolResult]:
        """Assemble ordered results, committing any defensively missing slot."""

        out: List[ToolResult] = []
        first_fault: Exception | None = None
        commit_blocked = False
        for index, call in enumerate(calls):
            result = self._results[index]
            if result is None:  # pragma: no cover - defensive path
                result = await self._terminal_result(
                    index, call, self._missing_result(call, "result_missing")
                )
            if index not in self._committed and not commit_blocked:
                try:
                    result = await self._commit_terminal(index, call, result)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # A handler may already have produced external side
                    # effects. Preserve its candidate, but never commit a
                    # later candidate across this missing canonical prefix
                    # and never re-execute a handler to manufacture a result.
                    if first_fault is None:
                        first_fault = exc
                    if index not in self._committed:
                        # Canonical terminals are an ordered prefix. Once one
                        # persistence boundary is missing, later candidates
                        # stay uncommitted for crash recovery to close in the
                        # same input order.
                        commit_blocked = True
            out.append(result)
        self.last_batch_results = out
        if first_fault is not None:
            raise first_fault
        return out

    def _finish_result(
        self,
        call: ToolCall,
        *,
        status: Any,
        start: float,
        attempts: int,
        output: Any = None,
        error: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        if output is None and status not in ("success", "partial"):
            code = str(
                (extra_metadata or {}).get("error_code") or "TOOL_EXECUTION_ERROR"
            )
            output = "\n".join(
                [
                    "[TOOL:error]",
                    "",
                    f"Tool: `{call.name}`",
                    f"Code: `{code}`",
                    "",
                    str(error or "The tool did not produce a result."),
                    "No success was inferred.",
                ]
            )
        metadata = {"tool_name": call.name, "attempts": attempts}
        metadata.update(extra_metadata or {})
        metadata.setdefault("latency_ms", (time.monotonic() - start) * 1000)
        return ToolResult(
            status=status,
            output=output,
            error=error,
            metadata=metadata,
        )


__all__ = [
    "AfterToolCallContext",
    "AfterToolCallHook",
    "AfterToolCallOverride",
    "AgentContextSnapshot",
    "BeforeToolCallContext",
    "BeforeToolCallDecision",
    "BeforeToolCallHook",
    "ToolBatchExecutor",
    "ToolExecutionConfig",
    "ToolHookRuntime",
    "ToolTransactionBoundary",
    "UNSET",
]
