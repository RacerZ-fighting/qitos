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
- external task cancellation terminalizes every started or admitted call
  (cancelled results, journal records and events) and only then re-raises
  ``asyncio.CancelledError``;
- a per-Tool ``RetryPolicy`` is the single retry owner;
- ``before_tool_call`` receives the validated arguments and may block, or
  return ``updated_args`` which are re-validated against the same schema and
  permission before execution (QitOS strengthening over Pi, whose hook only
  blocks); ``after_tool_call`` applies a field-level partial override. Hook
  contexts carry the assistant message, an immutable agent-context snapshot,
  ``is_error`` and the active cancel/deadline runtime (Pi parity); hooks are
  bounded by that runtime so a hung hook cannot block abort or the deadline.

The executor never raises for Tool-level failures. Persistence faults from
the transaction boundary (including ``JournalAppendCancelled``) and caller
cancellation propagate.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import random
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
from .cancellation import CancelToken
from .env import Env
from .journal import JournalAppendCancelled
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
    them. The executor bounds every hook await by this runtime; a hook that
    ignores cancellation is cancelled and abandoned, never awaited forever.
    """

    cancel_token: Optional[CancelToken] = None
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
    """Admission override; ``block`` rejects the call before execution.

    ``updated_args`` replace the validated arguments and are re-checked
    against the Tool's permission and input schema before execution; a failed
    re-check produces the corresponding denied/error terminal result.
    """

    block: bool = False
    reason: str = ""
    terminate: bool = False
    updated_args: Optional[Mapping[str, Any]] = None


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
        #: Ordered terminal results of the last batch, including the results
        #: committed while an external cancellation was being drained. The
        #: loop reads them to keep the transcript paired when the batch task
        #: is cancelled.
        self.last_batch_results: Optional[List[ToolResult]] = None

    async def execute_batch(self, calls: Sequence[ToolCall]) -> List[ToolResult]:
        """Execute one batch and return terminal results in input order."""

        if not calls:
            return []
        self._results = [None] * len(calls)
        self._started = set()
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
            if self._externally_cancelled():
                # A swallowed CancelledError deep in the stack must still
                # surface once every admitted call reached a terminal state.
                raise asyncio.CancelledError()
            return await self._final_results(calls)
        except asyncio.CancelledError:
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
        """Reject duplicate ToolCall ids before any Tool side effect.

        Journal record ids derive from the call id, so two calls sharing an
        id could otherwise both perform side effects and then conflict or
        fold inside the journal. Every duplicate occurrence is terminalized
        as an admission error without executing its handler; the journal
        records each duplicated id exactly once, under its first occurrence.
        """

        seen: Set[str] = set()
        duplicate_ids: Set[str] = set()
        for call in calls:
            if call.id in seen:
                duplicate_ids.add(call.id)
            seen.add(call.id)
        runnable: List[Tuple[int, ToolCall]] = []
        journaled_ids: Set[str] = set()
        for index, call in enumerate(calls):
            if call.id not in duplicate_ids:
                runnable.append((index, call))
                continue
            journaled = call.id not in journaled_ids
            journaled_ids.add(call.id)
            await self._start_call(index, call, journal=journaled)
            result = ToolResult(
                status="error",
                output=None,
                error=(
                    f'Duplicate tool call id "{call.id}" in one assistant '
                    "message; the call was not executed. Re-issue each tool "
                    "call with a unique id."
                ),
                metadata={
                    "tool_name": call.name,
                    "error_category": "duplicate_tool_call_id",
                    "recoverable": True,
                    "started": False,
                },
            )
            self._results[index] = await self._commit_terminal(
                index, call, result, journal=journaled
            )
        return runnable

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
        # A serial batch runs on the caller's task; an external cancellation
        # converted to a cancelled result below still aborts the batch here.
        if self._externally_cancelled():
            for index, call in runnable:
                if self._results[index] is None:
                    self._results[index] = await self._prevented(
                        index, call, "caller_cancelled"
                    )
            raise asyncio.CancelledError()

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
        if self._externally_cancelled():
            for index, call in runnable:
                if self._results[index] is None:
                    self._results[index] = await self._prevented(
                        index, call, "caller_cancelled"
                    )
            raise asyncio.CancelledError()

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
                    except JournalAppendCancelled:
                        raise
                    except asyncio.CancelledError:  # pragma: no cover - defensive
                        self._results[item.index] = await self._prevented(
                            item.index, item.call, abort_reason or "parent_cancelled"
                        )
                    except Exception as exc:  # pragma: no cover - defensive path
                        self._results[item.index] = await self._commit_terminal(
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
        await emit_to(
            self._emit,
            ToolExecutionStart(
                tool_call_id=call.id, tool_name=call.name, args=call.arguments
            ),
        )
        if self._transaction is not None and journal:
            try:
                await self._transaction.tool_started(self._config.turn, call)
            except asyncio.CancelledError:
                # Journal appends settle before surfacing cancellation, so
                # the started record is durable even though we re-raise.
                self._started.add(index)
                raise
        self._started.add(index)

    async def _commit_terminal(
        self, index: int, call: ToolCall, result: ToolResult, *, journal: bool = True
    ) -> ToolResult:
        """Freeze and record the unique terminal result through all barriers."""

        final = dataclasses.replace(result, call_id=call.id).frozen()
        if self._transaction is not None and journal:
            try:
                await self._transaction.tool_terminal(self._config.turn, call, final)
            except asyncio.CancelledError:
                # The append settled durably; mark the slot before the
                # cancellation propagates so nobody records a second terminal.
                self._results[index] = final
                raise
        await emit_to(
            self._emit,
            ToolExecutionEnd(
                tool_call_id=call.id,
                tool_name=call.name,
                result=final,
                is_error=final.status != "success",
            ),
        )
        return final

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
            return await self._commit_terminal(index, call, stopped)

        if call.parse_error is not None:
            return await self._commit_terminal(
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
            return await self._commit_terminal(
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
            return await self._commit_terminal(index, call, stopped)

        permission = tool.check_permissions(dict(call.arguments), runtime_context)
        if permission.decision == "deny":
            return await self._commit_terminal(
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
            return await self._commit_terminal(
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
            call.arguments
            if permission.updated_args is None
            else permission.updated_args
        )
        validation = self._validate(tool, effective_args, runtime_context)
        if not validation.valid:
            return await self._commit_terminal(
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
            return await self._commit_terminal(index, call, stopped)

        if self._config.before_tool_call is not None:
            decision, hook_failure = await self._run_before_hook(
                call, effective_args
            )
            if hook_failure is not None:
                return await self._commit_terminal(index, call, hook_failure)
            if decision is not None and decision.block:
                extra: Dict[str, Any] = {
                    **ordering_meta,
                    "error_category": "before_tool_call_blocked",
                    "started": False,
                }
                if decision.terminate:
                    extra["terminate"] = True
                return await self._commit_terminal(
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
            if decision is not None and decision.updated_args is not None:
                candidate = dict(decision.updated_args)
                recheck = tool.check_permissions(candidate, runtime_context)
                if recheck.decision == "deny":
                    return await self._commit_terminal(
                        index,
                        call,
                        self._finish_result(
                            call,
                            status="denied",
                            start=start,
                            attempts=0,
                            error=recheck.message or "Tool permission denied",
                            output={
                                "message": recheck.message,
                                "scope": recheck.scope,
                            },
                            extra_metadata={
                                **ordering_meta,
                                "error_category": "permission_denied",
                                "permission_scope": recheck.scope,
                            },
                        ),
                    )
                if recheck.decision == "ask":
                    return await self._commit_terminal(
                        index,
                        call,
                        self._finish_result(
                            call,
                            status="needs_approval",
                            start=start,
                            attempts=0,
                            output={
                                "message": recheck.message,
                                "scope": recheck.scope,
                            },
                            extra_metadata={
                                **ordering_meta,
                                "error_category": "permission_ask",
                                "permission_scope": recheck.scope,
                            },
                        ),
                    )
                candidate = dict(
                    candidate
                    if recheck.updated_args is None
                    else recheck.updated_args
                )
                revalidation = self._validate(tool, candidate, runtime_context)
                if not revalidation.valid:
                    return await self._commit_terminal(
                        index,
                        call,
                        self._finish_result(
                            call,
                            status="error",
                            start=start,
                            attempts=0,
                            error=(
                                "before_tool_call updated_args failed validation: "
                                + (revalidation.message or "invalid arguments")
                            ),
                            extra_metadata={
                                **ordering_meta,
                                "error_category": (
                                    revalidation.code or "validation_error"
                                ),
                                "started": False,
                            },
                        ),
                    )
                effective_args = candidate

        stopped = self._stop_result(call, start, attempts=0, deadline=deadline)
        if stopped is not None:
            return await self._commit_terminal(index, call, stopped)

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
        return await self._commit_terminal(prepared.index, call, result)

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
            self._abandon(hook_task, watcher)
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
        self._abandon(hook_task, watcher)
        return outcome, None

    @staticmethod
    def _abandon(*tasks: Optional[asyncio.Task[Any]]) -> None:
        """Leave cancelled hooks to finish on their own, consuming outcomes."""

        for task in tasks:
            if task is None or task.done():
                continue
            task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)

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
            cancel_token=self._config.cancel_token,
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
                    # callback; they drain after each attempt, success or not.
                    await self._drain_progress_updates(call, runtime_context)
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
                )
            except JournalAppendCancelled:
                raise
            except asyncio.CancelledError:
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
                        "cancel_source": (
                            "cancel_token"
                            if self._immediate()
                            else "caller_cancelled"
                        ),
                    },
                )
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
        progress_updates: List[Dict[str, Any]] = []
        artifacts: List[Dict[str, Any]] = []

        def _emit_progress(payload: Dict[str, Any]) -> None:
            progress_events.append(dict(payload))
            progress_updates.append(dict(payload))

        def _record_artifact(payload: Dict[str, Any]) -> None:
            artifacts.append(dict(payload))

        context: Dict[str, Any] = {
            "env": env,
            "ops": self._resolve_ops(tool, env),
            "tool_registry": self._exposure,
            "progress_events": progress_events,
            "artifacts": artifacts,
            "emit_progress": _emit_progress,
            "record_artifact": _record_artifact,
            "run_id": self._config.run_id,
            "deadline_monotonic": deadline_monotonic,
            "remaining_seconds": lambda: self._remaining_seconds(deadline_monotonic),
            "agent_cancelled": self._immediate,
        }
        context["progress_updates"] = progress_updates
        context.update(dict(self._config.extra_runtime_context))
        return context

    async def _drain_progress_updates(
        self, call: ToolCall, runtime_context: Dict[str, Any]
    ) -> None:
        updates = runtime_context.get("progress_updates")
        if not isinstance(updates, list):
            return
        while updates:
            payload = updates.pop(0)
            await emit_to(
                self._emit,
                ToolExecutionUpdate(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    args=call.arguments,
                    partial_result=payload,
                ),
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

    @staticmethod
    def _externally_cancelled() -> bool:
        task = asyncio.current_task()
        return task is not None and task.cancelling() > 0

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
        invocation = tool.execute(args, runtime_context=runtime_context)
        if timeout_s is None:
            return await invocation
        if timeout_s <= 0:
            raise asyncio.TimeoutError("tool call deadline expired")
        return await asyncio.wait_for(invocation, timeout=timeout_s)

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
        return await self._commit_terminal(
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
        for index, call in enumerate(calls):
            result = self._results[index]
            if result is None:  # pragma: no cover - defensive path
                result = await self._commit_terminal(
                    index, call, self._missing_result(call, "result_missing")
                )
            out.append(result)
        self.last_batch_results = out
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
