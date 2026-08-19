"""Structured async supervision for independently stateful Subagent runs."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from typing import Any

from ...core.budget import BudgetLedger, BudgetSnapshot
from ...core.subagent import (
    AgentConclusion,
    SubagentEngine,
    SubagentHandle,
    SubagentInvocation,
    SubagentInvocationCancelled,
    SubagentLaunchContext,
    SubagentLaunchRequest,
    SubagentPersistenceError,
    SubagentResult,
    SubagentRunLimitError,
    SubagentRuntimeContext,
    SubagentStatus,
)
from ...core.journal import (
    JournalAppendCancelled,
    JournalCommitError,
    JournalCommitState,
    JournalRecord,
    JournalRecordType,
    SessionJournal,
)
from ...core.message import AssistantMessage, ToolResultMessage
from ...core.model_response import ModelPricing
from ...core.runtime_input import (
    RuntimeInput,
    subagent_result_payload,
    subagent_terminal_runtime_input,
)
from ..journal import recover_run_outcome, recover_session
from .agent_engine import (
    subagent_budget_stop_reason,
    subagent_final_text,
    subagent_run_stats,
    subagent_stop_reason,
)
from .limits import SubagentRunLimiter, _SubagentRunLease

_logger = logging.getLogger(__name__)

SubagentInvocationFactory = Callable[
    [SubagentLaunchRequest, SubagentRuntimeContext], Awaitable[SubagentInvocation]
]
SubagentExecutionScope = Callable[
    [SubagentRuntimeContext],
    AbstractContextManager[Any] | AbstractAsyncContextManager[Any],
]
SubagentJournalFactory = Callable[[], SessionJournal]

_TOOL_RESULT_PREVIEW_MAX_ITEMS = 12
_TOOL_RESULT_PREVIEW_MAX_CHARS = 16_000

_MISSING_CONCLUSION_ERROR = (
    "Subagent ended without a final answer; its single same-context conclusion "
    "follow-up was exhausted or unavailable within the existing limits."
)


def _caused_by_file_not_found(error: BaseException) -> bool:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, FileNotFoundError):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


@dataclass(slots=True)
class _OwnedSubagent:
    handle: SubagentHandle
    subagent_run_id: str
    request: SubagentLaunchRequest
    background: bool
    launch_context: SubagentLaunchContext | None
    cancel_event: asyncio.Event
    terminal_event: asyncio.Event
    engine_ready: asyncio.Event
    journal: SessionJournal | None = None
    task: asyncio.Task[SubagentResult] | None = None
    engine: SubagentEngine | None = None
    result: SubagentResult | None = None
    run_lease: _SubagentRunLease | None = None


@dataclass(frozen=True, slots=True)
class _RecoveredSubagentStart:
    request: SubagentLaunchRequest
    subagent_run_id: str
    background: bool


class SubagentSupervisor:
    """Own subagent Engines, Tasks, terminal results, and parent delivery for one Run."""

    def __init__(
        self,
        *,
        invocation_factory: SubagentInvocationFactory,
        execution_scope: SubagentExecutionScope | None = None,
        max_concurrency: int = 4,
        run_limiter: SubagentRunLimiter | None = None,
        subagent_journal_factory: SubagentJournalFactory | None = None,
    ) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self._invocation_factory = invocation_factory
        self._execution_scope = execution_scope
        self._max_concurrency = max_concurrency
        if run_limiter is not None and not isinstance(run_limiter, SubagentRunLimiter):
            raise TypeError("run_limiter must be a SubagentRunLimiter or None")
        self._run_limiter = run_limiter
        if subagent_journal_factory is not None and not callable(subagent_journal_factory):
            raise TypeError("subagent_journal_factory must be callable or None")
        self._subagent_journal_factory = subagent_journal_factory
        self._limit = asyncio.Semaphore(max_concurrency)
        self._admission_lock = asyncio.Lock()
        self._subagents_started = 0
        self._closed = False
        self._subagents: dict[SubagentHandle, _OwnedSubagent] = {}
        self._recovered_runs: set[str] = set()
        self._lifecycle_locks: dict[str, asyncio.Lock] = {}

    async def launch(
        self,
        request: SubagentLaunchRequest,
        context: SubagentLaunchContext,
        *,
        background: bool,
    ) -> SubagentResult:
        """Launch one fresh subagent, returning a running or terminal projection."""

        if not isinstance(request, SubagentLaunchRequest):
            raise TypeError("request must be a SubagentLaunchRequest")
        if not isinstance(context, SubagentLaunchContext):
            raise TypeError("context must be a SubagentLaunchContext")
        normalized_parent = context.parent_run_id
        journal = context.journal
        lifecycle_lock = await self._lifecycle_lock(normalized_parent)
        async with lifecycle_lock:
            run_lease = (
                await self._run_limiter.reserve()
                if self._run_limiter is not None
                else None
            )
            try:
                async with self._admission_lock:
                    if self._closed:
                        raise RuntimeError("subagent supervisor is closed")
                    if (
                        context.max_subagents > 0
                        and self._subagents_started >= context.max_subagents
                    ):
                        raise RuntimeError(
                            "Run Subagent budget exhausted: "
                            f"max_subagents={context.max_subagents}."
                        )
                    self._subagents_started += 1
                    handle = SubagentHandle(
                        subagent_id=f"subagent-{uuid.uuid4().hex[:12]}",
                        parent_run_id=normalized_parent,
                    )
                    owned = _OwnedSubagent(
                        handle=handle,
                        subagent_run_id=f"run_{uuid.uuid4().hex[:12]}",
                        request=request,
                        background=background,
                        launch_context=context,
                        cancel_event=asyncio.Event(),
                        terminal_event=asyncio.Event(),
                        engine_ready=asyncio.Event(),
                        journal=journal,
                        run_lease=run_lease,
                    )
                    self._subagents[handle] = owned
                    owned.task = asyncio.current_task()
            except BaseException:
                if run_lease is not None:
                    await run_lease.rollback()
                raise

            try:
                await self._persist_started(owned)
            except JournalAppendCancelled as cancellation:
                if cancellation.commit_state is JournalCommitState.NOT_COMMITTED:
                    await self._discard_unstarted(owned)
                    raise

                if cancellation.commit_state is JournalCommitState.COMMITTED:
                    if run_lease is not None:
                        run_lease.commit()
                    cancelled_result = self._cancelled_result(
                        owned,
                        error="Subagent launch was cancelled after durable admission.",
                    )
                    try:
                        await self._store_terminal(owned, cancelled_result)
                    except asyncio.CancelledError:
                        # Recovery terminalizes the durable started record if a
                        # second cancellation prevents this terminal append.
                        pass
                    finally:
                        owned.task = None
                        owned.launch_context = None
                        await self._release_run_lease(owned)
                else:
                    await self._retain_failed_admission(
                        owned,
                        cause=cancellation.commit_error,
                    )
                raise
            except JournalCommitError as commit_error:
                if commit_error.commit_state is JournalCommitState.NOT_COMMITTED:
                    await self._discard_unstarted(owned)
                else:
                    await self._retain_failed_admission(
                        owned,
                        cause=commit_error,
                    )
                raise SubagentPersistenceError(
                    "failed to persist subagent.started; subagent was not executed"
                ) from commit_error
            except BaseException:
                await self._discard_unstarted(owned)
                raise
            if run_lease is not None:
                run_lease.commit()

            async with self._admission_lock:
                if self._closed:
                    cancelled = True
                elif background:
                    owned.task = asyncio.create_task(
                        self._supervise_background(owned),
                        name=f"qitos-{handle.subagent_id}",
                    )
                    return self._current_result(owned)
                else:
                    cancelled = False
            if cancelled:
                result = self._cancelled_result(
                    owned,
                    error="Subagent supervisor closed before subagent start.",
                )
                await self._store_terminal(owned, result)
                owned.task = None
                owned.launch_context = None
                await self._release_run_lease(owned)
                return self._current_result(owned)

        try:
            result = await self._run_request_with_limit(owned)
        except asyncio.CancelledError:
            self._cancel_engine(owned)
            await self._store_terminal(owned, self._cancelled_result(owned))
            raise
        else:
            await self._store_terminal(owned, result)
            return self._current_result(owned)
        finally:
            owned.task = None
            owned.engine = None
            owned.launch_context = None
            await self._release_run_lease(owned)

    def result(self, handle: SubagentHandle) -> SubagentResult | None:
        """Return one owned subagent's immutable current state."""

        owned = self._owned(handle)
        return None if owned is None else self._current_result(owned)

    async def wait(
        self,
        handle: SubagentHandle,
        *,
        timeout_seconds: float | None = None,
    ) -> SubagentResult | None:
        """Wait for terminal state without cancelling the subagent on timeout."""

        owned = self._owned(handle)
        if owned is None:
            return None
        current = self._current_result(owned)
        if current.ready:
            return current
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative or None")
        try:
            if timeout_seconds is None:
                await owned.terminal_event.wait()
            else:
                await asyncio.wait_for(
                    owned.terminal_event.wait(),
                    timeout=timeout_seconds,
                )
        except asyncio.TimeoutError:
            return self._current_result(owned)
        return self._current_result(owned)

    async def recover(
        self,
        *,
        parent_run_id: str,
        journal: SessionJournal,
        budget_ledger: BudgetLedger | None = None,
    ) -> tuple[SubagentResult, ...]:
        """Recover terminal facts and close interrupted subagents without replay."""

        normalized_parent = str(parent_run_id or "").strip()
        if not normalized_parent:
            raise ValueError("parent_run_id must be a non-empty string")
        if budget_ledger is not None and not isinstance(budget_ledger, BudgetLedger):
            raise TypeError("budget_ledger must be a BudgetLedger or None")
        if normalized_parent in self._recovered_runs:
            return self._results_for_parent(normalized_parent)
        lifecycle_lock = await self._lifecycle_lock(normalized_parent)
        async with lifecycle_lock:
            if normalized_parent in self._recovered_runs:
                return self._results_for_parent(normalized_parent)
            async with self._admission_lock:
                if any(
                    owned.handle.parent_run_id == normalized_parent
                    and owned.task is not None
                    and not owned.task.done()
                    for owned in self._subagents.values()
                ):
                    raise RuntimeError(
                        "cannot recover Subagent state while owned tasks are active"
                    )
            return await self._recover_once(
                parent_run_id=normalized_parent,
                journal=journal,
                budget_ledger=budget_ledger,
            )

    async def _recover_once(
        self,
        *,
        parent_run_id: str,
        journal: SessionJournal,
        budget_ledger: BudgetLedger | None,
    ) -> tuple[SubagentResult, ...]:
        """Recover one parent while its lifecycle admission is serialized."""

        records = await journal.replay()
        try:
            started, terminal = self._decode_lifecycle(
                records,
                parent_run_id=parent_run_id,
            )
        except (TypeError, ValueError) as exc:
            raise SubagentPersistenceError(
                "subagent lifecycle journal records are invalid"
            ) from exc

        launch_ids = {f"{handle.parent_run_id}:{handle.subagent_id}" for handle in started}
        if self._subagent_journal_factory is not None:
            subagent_run_ids: list[str] = []
            for handle, start in started.items():
                terminal_result = terminal.get(handle)
                resolved_run_id = (
                    terminal_result.subagent_run_id
                    if terminal_result is not None and terminal_result.subagent_run_id
                    else start.subagent_run_id
                )
                if resolved_run_id:
                    subagent_run_ids.append(resolved_run_id)
            try:
                launch_ids.update(
                    await self._descendant_launch_ids(
                        parent_run_id=parent_run_id,
                        subagent_run_ids=tuple(subagent_run_ids),
                    )
                )
            except asyncio.CancelledError:
                raise
            except SubagentPersistenceError:
                raise
            except Exception as exc:
                raise SubagentPersistenceError(
                    "failed to restore descendant Subagent launch history"
                ) from exc

        recovered = {
            handle: (
                result
                if result.subagent_run_id
                else replace(result, subagent_run_id=started[handle].subagent_run_id)
            )
            for handle, result in terminal.items()
        }
        for handle, start in started.items():
            if handle in recovered:
                continue
            result = await self._recover_interrupted_result(
                handle,
                start,
                root_budget=(
                    budget_ledger.snapshot_after_origin(start.subagent_run_id)
                    if budget_ledger is not None and start.subagent_run_id
                    else None
                ),
            )
            try:
                await journal.append(
                    JournalRecordType.SUBAGENT_TERMINAL,
                    result.to_dict(),
                    record_id=self._terminal_record_id(handle),
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise SubagentPersistenceError(
                    "failed to persist interrupted subagent terminal record"
                ) from exc
            recovered[handle] = result
        if self._run_limiter is not None:
            try:
                await self._run_limiter.restore_started(launch_ids)
            except SubagentRunLimitError as exc:
                raise SubagentPersistenceError(
                    "restored Subagent launch history exceeds the configured Run limit"
                ) from exc

        async with self._admission_lock:
            for handle, result in recovered.items():
                owned = self._subagents.get(handle)
                if owned is None:
                    start = started[handle]
                    owned = _OwnedSubagent(
                        handle=handle,
                        subagent_run_id=result.subagent_run_id or start.subagent_run_id,
                        request=result.request,
                        background=start.background,
                        launch_context=None,
                        cancel_event=asyncio.Event(),
                        terminal_event=asyncio.Event(),
                        engine_ready=asyncio.Event(),
                    )
                    self._subagents[handle] = owned
                self._set_terminal(owned, result)
            self._subagents_started = max(self._subagents_started, len(started))
            self._recovered_runs.add(parent_run_id)
        return self._results_for_parent(parent_run_id)

    async def interrupt(
        self,
        handle: SubagentHandle,
        *,
        wait_seconds: float = 5.0,
    ) -> SubagentResult | None:
        """Cancel one active subagent and wait a bounded time for its cleanup."""

        if wait_seconds < 0:
            raise ValueError("wait_seconds must be non-negative")
        owned = self._owned(handle)
        if owned is None:
            return None
        current = self._current_result(owned)
        if current.ready:
            return current
        self.request_interrupt(handle)
        task = owned.task
        if task is not None:
            if wait_seconds > 0:
                done, _ = await asyncio.wait((task,), timeout=wait_seconds)
                if done:
                    await asyncio.gather(*done, return_exceptions=True)
            else:
                # Give cancellation before the Task's first step one turn to
                # settle. Such a Task never enters the supervisor coroutine
                # and therefore cannot terminalize itself.
                await asyncio.sleep(0)
            await self._terminalize_cancelled_task(owned, task)
        return self._current_result(owned)

    async def message(
        self,
        handle: SubagentHandle,
        content: str,
        *,
        timeout_seconds: float | None = None,
    ) -> tuple[bool, SubagentResult | None]:
        """Post one parent message to an active Subagent's durable mailbox."""

        text = str(content or "").strip()
        if not text:
            raise ValueError("content must be a non-empty string")
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative or None")
        deadline = (
            None
            if timeout_seconds is None
            else time.monotonic() + timeout_seconds
        )
        owned = self._owned(handle)
        if owned is None:
            return False, None
        current = self._current_result(owned)
        if current.ready:
            return False, current
        try:
            await self._wait_until_ready_or_terminal(
                owned,
                timeout_seconds=(
                    None
                    if deadline is None
                    else max(0.0, deadline - time.monotonic())
                ),
            )
        except TimeoutError:
            return False, self._current_result(owned)
        current = self._current_result(owned)
        if current.ready or owned.engine is None:
            return False, current
        post_runtime_event = getattr(owned.engine, "apost_runtime_event", None)
        if not callable(post_runtime_event):
            raise RuntimeError("subagent Engine does not expose an async runtime mailbox")
        subagent_run_id = self._subagent_run_id(owned)
        if not subagent_run_id:
            return False, current
        event = RuntimeInput(
            event_id=f"{owned.handle.subagent_id}:parent:{uuid.uuid4().hex}",
            kind="agent.parent.message",
            correlation_id=owned.handle.subagent_id,
            source="qitos.parent",
            payload={"content": text},
        )
        post = post_runtime_event(event, run_id=subagent_run_id)
        try:
            if deadline is None:
                accepted = await post
            else:
                accepted = await asyncio.wait_for(
                    post,
                    timeout=max(0.0, deadline - time.monotonic()),
                )
        except asyncio.TimeoutError:
            return False, self._current_result(owned)
        return bool(accepted), self._current_result(owned)

    def request_interrupt(self, handle: SubagentHandle) -> bool:
        """Signal one active subagent without waiting for terminal cleanup."""

        owned = self._owned(handle)
        if owned is None or self._current_result(owned).ready:
            return False
        if owned.cancel_event.is_set():
            return True
        owned.result = SubagentResult(
            handle=owned.handle,
            request=owned.request,
            status=SubagentStatus.CANCEL_REQUESTED,
            subagent_run_id=self._subagent_run_id(owned),
        )
        owned.cancel_event.set()
        self._cancel_engine(owned)
        if owned.task is not None:
            owned.task.cancel()
        return True

    @property
    def active_count(self) -> int:
        """Return subagents whose execution has not reached terminal state."""

        return sum(
            1
            for owned in self._subagents.values()
            if not self._current_result(owned).ready
        )

    def snapshot_events(self) -> list[RuntimeInput]:
        """Project bounded completed Tool evidence from active subagent Engines."""

        events: list[RuntimeInput] = []
        for owned in self._subagents.values():
            if self._current_result(owned).ready or owned.engine is None:
                continue
            events.append(
                RuntimeInput(
                    event_id=f"{owned.handle.subagent_id}:conclude-snapshot",
                    kind="agent.subagent.snapshot",
                    correlation_id=owned.handle.subagent_id,
                    source="qitos.agent",
                    payload={
                        "handle": owned.handle.to_dict(),
                        "subagent_id": owned.handle.subagent_id,
                        "status": "running",
                        "subagent_status": SubagentStatus.RUNNING.value,
                        "agent_type": owned.request.agent_type,
                        "name": owned.request.name,
                        "description": owned.request.description,
                        "output": self._tool_result_preview(owned.engine),
                        "steps": int(getattr(owned.engine, "step_count", 0) or 0),
                        "total_tokens": int(
                            getattr(owned.engine, "token_usage", 0) or 0
                        ),
                        "total_cost_usd": float(
                            getattr(owned.engine, "cost_usage_usd", 0.0) or 0.0
                        ),
                        "usage_complete": bool(
                            getattr(owned.engine, "usage_complete", False)
                        ),
                        "cost_complete": bool(
                            getattr(owned.engine, "cost_complete", False)
                        ),
                        "run_id": self._subagent_run_id(owned),
                    },
                )
            )
        return events

    def setup(self, *, reset_run_limiter: bool = False) -> None:
        """Open this supervisor for a fresh owner Run.

        A shared recursive ``SubagentRunLimiter`` has one root lifecycle owner.
        Nested supervisors therefore leave its cumulative Run-tree accounting
        intact unless their composition owner explicitly starts a new root Run.
        """

        if any(
            owned.task is not None and not owned.task.done()
            for owned in self._subagents.values()
        ):
            raise RuntimeError("cannot reopen a subagent supervisor with owned tasks")
        if reset_run_limiter and self._run_limiter is not None:
            self._run_limiter.reset_for_new_run()
        self._limit = asyncio.Semaphore(self._max_concurrency)
        self._admission_lock = asyncio.Lock()
        self._subagents_started = 0
        self._closed = False
        self._subagents.clear()
        self._recovered_runs.clear()
        self._lifecycle_locks.clear()

    async def aclose(self, *, wait_seconds: float = 5.0) -> int:
        """Request cancellation and wait a bounded time for owned Tasks."""

        if wait_seconds < 0:
            raise ValueError("wait_seconds must be non-negative")
        async with self._admission_lock:
            self._closed = True
            owned_tasks = [
                owned
                for owned in self._subagents.values()
                if owned.task is not None and not owned.task.done()
            ]
        for owned in owned_tasks:
            requested = self.request_interrupt(owned.handle)
            if not requested and owned.task is not None and not owned.task.done():
                # A terminal result may still own bounded parent delivery.
                # Cancelling that delivery cannot interrupt Subagent cleanup,
                # which has already completed before terminal publication.
                owned.task.cancel()

        tasks = [owned.task for owned in owned_tasks if owned.task is not None]
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=wait_seconds)
            if done:
                await asyncio.gather(*done, return_exceptions=True)
            if pending:
                # A second Task.cancel() can interrupt the SubagentInvocation's
                # own async cleanup. Keep the Tasks registered so a later
                # close/wait can drain them without losing ownership.
                await asyncio.sleep(0)
        for owned in owned_tasks:
            task = owned.task
            if task is not None:
                await self._terminalize_cancelled_task(owned, task)
        return sum(1 for task in tasks if not task.done())

    async def _supervise_background(self, owned: _OwnedSubagent) -> SubagentResult:
        try:
            result = await self._run_request_with_limit(owned)
        except asyncio.CancelledError:
            result = self._cancelled_result(owned)
        except Exception as exc:  # pragma: no cover - defensive boundary
            result = self._failed_result(owned, exc)
        try:
            persisted = await self._store_terminal(owned, result)
            result = self._current_result(owned)
            # A durable terminal result no longer consumes execution capacity.
            # Parent mailbox delivery remains owned, but it must not block a
            # later Subagent from entering the root Run's active limit.
            await self._release_run_lease(owned)
            if persisted:
                await self._post_completion_event(owned, result)
            return result
        except asyncio.CancelledError:
            raise
        finally:
            owned.task = None
            owned.engine = None
            owned.launch_context = None
            await self._release_run_lease(owned)
        return result

    async def _run_request_with_limit(self, owned: _OwnedSubagent) -> SubagentResult:
        """Apply concurrency and the parent's absolute deadline to all run work."""

        launch_context = owned.launch_context
        if launch_context is None:
            raise RuntimeError("Subagent launch context was released before execution")
        deadline = launch_context.deadline_monotonic
        started = time.monotonic()
        if deadline is not None and started >= deadline:
            return self._budget_exhausted_result(
                owned,
                error="Subagent deadline expired before execution admission.",
                elapsed_seconds=0.0,
            )

        async def _run_admitted() -> SubagentResult:
            async with self._limit:
                if deadline is not None and time.monotonic() >= deadline:
                    return self._budget_exhausted_result(
                        owned,
                        error="Subagent deadline expired before execution.",
                        elapsed_seconds=max(0.0, time.monotonic() - started),
                    )
                return await self._run_request(owned)

        if deadline is None:
            return await _run_admitted()

        async def _wait_for_deadline() -> None:
            await asyncio.sleep(max(0.0, deadline - time.monotonic()))

        request_task = asyncio.create_task(
            _run_admitted(),
            name=f"qitos-{owned.handle.subagent_id}-deadline-request",
        )
        deadline_task = asyncio.create_task(
            _wait_for_deadline(),
            name=f"qitos-{owned.handle.subagent_id}-deadline-timer",
        )
        try:
            completed, _pending = await asyncio.wait(
                (request_task, deadline_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            self._cancel_engine(owned)
            await self._cancel_and_drain_tasks(request_task, deadline_task)
            raise
        if request_task in completed:
            caller_cancelled = await self._cancel_and_drain_tasks(deadline_task)
            if caller_cancelled:
                self._cancel_engine(owned)
                raise asyncio.CancelledError
            return await request_task

        self._cancel_engine(owned)
        caller_cancelled = await self._cancel_and_drain_tasks(
            request_task,
            deadline_task,
        )
        if caller_cancelled:
            raise asyncio.CancelledError
        return self._budget_exhausted_result(
            owned,
            error="Subagent deadline expired.",
            elapsed_seconds=max(0.0, time.monotonic() - started),
        )

    @staticmethod
    async def _cancel_and_drain_tasks(*tasks: asyncio.Task[Any]) -> bool:
        """Cancel owned tasks, await settlement, and report caller cancellation."""

        for task in tasks:
            if not task.done():
                task.cancel()
        waiter = asyncio.gather(*tasks, return_exceptions=True)
        caller_cancelled = False
        while not waiter.done():
            try:
                await asyncio.shield(waiter)
            except asyncio.CancelledError:
                caller_cancelled = True
        waiter.result()
        return caller_cancelled

    async def _run_request(self, owned: _OwnedSubagent) -> SubagentResult:
        started = time.monotonic()
        launch_context = owned.launch_context
        if launch_context is None:
            raise RuntimeError("Subagent launch context was released before execution")
        scoped_context = SubagentRuntimeContext(
            launch=launch_context,
            handle=owned.handle,
            subagent_run_id=owned.subagent_run_id,
            cancellation_requested=owned.cancel_event.is_set,
        )
        scope = (
            self._execution_scope(scoped_context)
            if self._execution_scope is not None
            else nullcontext()
        )

        async def _run_in_scope() -> SubagentResult:
            if owned.cancel_event.is_set():
                raise RuntimeError("Subagent was cancelled before it started")
            budget_error = self._budget_admission_error(launch_context)
            if budget_error is not None:
                return self._budget_exhausted_result(owned, error=budget_error)
            return await self._run_invocation(owned, scoped_context)

        try:
            if isinstance(scope, AbstractAsyncContextManager):
                async with scope:
                    result = await _run_in_scope()
            else:
                with scope:
                    result = await _run_in_scope()
        except SubagentInvocationCancelled as exc:
            result = self._cancelled_result(owned, error=str(exc))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if owned.cancel_event.is_set():
                result = self._cancelled_result(owned, error=str(exc))
            else:
                result = self._failed_result(owned, exc)
        return replace(
            result,
            elapsed_seconds=max(0.0, time.monotonic() - started),
        )

    async def _run_invocation(
        self,
        owned: _OwnedSubagent,
        runtime_context: SubagentRuntimeContext,
    ) -> SubagentResult:
        invocation = await self._invocation_factory(owned.request, runtime_context)
        if not isinstance(invocation, SubagentInvocation):
            raise TypeError("invocation_factory must resolve to SubagentInvocation")
        owned.engine = invocation.engine
        conclusion: AgentConclusion | None = None
        try:
            if self._closed or owned.cancel_event.is_set():
                invocation.engine.cancel("immediate")
                raise RuntimeError("subagent supervisor closed before subagent start")
            owned.engine_ready.set()
            run_kwargs = dict(invocation.run_kwargs)
            requested_run_id = run_kwargs.get("run_id")
            if requested_run_id is not None and requested_run_id != owned.subagent_run_id:
                raise ValueError(
                    "Subagent invocation Run id conflicts with its durable launch"
                )
            run_kwargs["run_id"] = owned.subagent_run_id
            engine_result = await invocation.engine.arun(
                invocation.task,
                **run_kwargs,
            )
            if invocation.conclusion_factory is not None:
                conclusion = await invocation.conclusion_factory(engine_result)
                if not isinstance(conclusion, AgentConclusion):
                    raise TypeError(
                        "Subagent conclusion_factory must return AgentConclusion"
                    )
        finally:
            await self._cleanup_invocation(invocation)
        state = engine_result.state
        raw_stop_reason = state.stop_reason or ""
        stop_reason = str(getattr(raw_stop_reason, "value", raw_stop_reason))
        summary = str(state.final_result or "").strip()
        status = self._subagent_status(stop_reason)
        status_detail = stop_reason or "unknown stop reason"
        # The model's final natural-language answer is canonical. A product
        # conclusion factory may attach stable evidence/resource references
        # and other typed transport fields, but it must never author or
        # replace what the Subagent told its parent.
        if conclusion is not None:
            conclusion = replace(conclusion, summary=summary)
        error: str | None = None
        if status_detail == "missing_conclusion":
            error = _MISSING_CONCLUSION_ERROR
        elif status is SubagentStatus.COMPLETED and not summary:
            status = SubagentStatus.FAILED
            status_detail = "missing_conclusion"
            error = _MISSING_CONCLUSION_ERROR
        return SubagentResult(
            handle=owned.handle,
            request=owned.request,
            status=status,
            conclusion=(
                conclusion
                if conclusion is not None
                else AgentConclusion(
                    summary=summary,
                    failure_paths=(
                        (status_detail,) if status is SubagentStatus.FAILED else ()
                    ),
                    unknowns=(
                        (status_detail,)
                        if status is SubagentStatus.BUDGET_EXHAUSTED
                        else ()
                    ),
                )
            ),
            subagent_run_id=owned.subagent_run_id,
            error=error,
            steps=int(engine_result.step_count or 0),
            total_tokens=int(
                getattr(
                    engine_result,
                    "local_total_tokens",
                    getattr(engine_result, "total_tokens", 0),
                )
                or 0
            ),
            total_cost_usd=float(
                getattr(
                    engine_result,
                    "local_total_cost_usd",
                    getattr(engine_result, "total_cost_usd", 0.0),
                )
                or 0.0
            ),
            usage_complete=bool(getattr(engine_result, "local_usage_complete", False)),
            cost_complete=bool(getattr(engine_result, "local_cost_complete", False)),
        )

    def _set_terminal(self, owned: _OwnedSubagent, result: SubagentResult) -> None:
        if owned.terminal_event.is_set():
            if owned.result != result:
                raise SubagentPersistenceError(
                    "subagent terminal result conflicts with its existing terminal state"
                )
            return
        owned.result = result
        owned.terminal_event.set()

    @staticmethod
    async def _wait_until_ready_or_terminal(
        owned: _OwnedSubagent,
        *,
        timeout_seconds: float | None,
    ) -> None:
        engine_wait = asyncio.create_task(
            owned.engine_ready.wait(),
            name=f"qitos-{owned.handle.subagent_id}-engine-ready",
        )
        terminal_wait = asyncio.create_task(
            owned.terminal_event.wait(),
            name=f"qitos-{owned.handle.subagent_id}-terminal-wait",
        )
        waits = (engine_wait, terminal_wait)
        try:
            completed, _ = await asyncio.wait(
                waits,
                timeout=timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not completed:
                raise TimeoutError("Subagent did not become ready before its deadline")
        finally:
            for task in waits:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*waits, return_exceptions=True)

    async def _persist_started(self, owned: _OwnedSubagent) -> None:
        if owned.journal is None:
            return
        try:
            await owned.journal.append(
                JournalRecordType.SUBAGENT_STARTED,
                {
                    "handle": owned.handle.to_dict(),
                    "request": owned.request.to_dict(),
                    "background": owned.background,
                    "subagent_run_id": owned.subagent_run_id,
                },
                record_id=(
                    f"{owned.handle.parent_run_id}:subagent:"
                    f"{owned.handle.subagent_id}:started"
                ),
            )
        except asyncio.CancelledError:
            raise
        except JournalCommitError:
            raise
        except Exception as exc:
            raise SubagentPersistenceError(
                "failed to persist subagent.started; subagent was not executed"
            ) from exc

    async def _discard_unstarted(self, owned: _OwnedSubagent) -> None:
        """Release local admission after a launch failed before persistence."""

        async with self._admission_lock:
            self._subagents.pop(owned.handle, None)
            self._subagents_started -= 1
        if owned.run_lease is not None:
            await owned.run_lease.rollback()

    async def _store_terminal(
        self,
        owned: _OwnedSubagent,
        result: SubagentResult,
    ) -> bool:
        if owned.journal is not None:
            try:
                await owned.journal.append(
                    JournalRecordType.SUBAGENT_TERMINAL,
                    result.to_dict(),
                    record_id=self._terminal_record_id(owned.handle),
                )
            except JournalAppendCancelled as cancellation:
                if cancellation.commit_state is JournalCommitState.COMMITTED:
                    self._set_terminal(owned, result)
                elif cancellation.commit_state is JournalCommitState.UNKNOWN:
                    self._set_terminal(
                        owned,
                        self._persistence_unknown_result(
                            result,
                            cause=cancellation.commit_error,
                        ),
                    )
                else:
                    self._set_terminal(
                        owned,
                        self._persistence_failed_result(
                            result,
                            cause=cancellation.commit_error,
                        ),
                    )
                raise
            except asyncio.CancelledError:
                self._set_terminal(owned, self._persistence_failed_result(result))
                raise
            except JournalCommitError as commit_error:
                if commit_error.commit_state is JournalCommitState.COMMITTED:
                    self._set_terminal(owned, result)
                    return True
                if commit_error.commit_state is JournalCommitState.UNKNOWN:
                    self._set_terminal(
                        owned,
                        self._persistence_unknown_result(
                            result,
                            cause=commit_error,
                        ),
                    )
                    return False
                self._set_terminal(
                    owned,
                    self._persistence_failed_result(result, cause=commit_error),
                )
                return False
            except Exception as exc:
                self._set_terminal(
                    owned,
                    self._persistence_failed_result(result, cause=exc),
                )
                return False
        self._set_terminal(owned, result)
        return True

    async def _retain_failed_admission(
        self,
        owned: _OwnedSubagent,
        *,
        cause: BaseException | None,
    ) -> None:
        if owned.run_lease is not None:
            owned.run_lease.commit()
        result = self._cancelled_result(
            owned,
            error=(
                "Subagent launch was not executed because its durable admission "
                "could not be confirmed."
            ),
        )
        self._set_terminal(
            owned,
            self._persistence_failed_result(result, cause=cause),
        )
        owned.task = None
        owned.launch_context = None
        await self._release_run_lease(owned)

    async def _terminalize_cancelled_task(
        self,
        owned: _OwnedSubagent,
        task: asyncio.Task[SubagentResult],
    ) -> None:
        """Persist cancellation for a Task that never entered its supervisor."""

        if not task.done() or owned.terminal_event.is_set():
            return
        await self._store_terminal(owned, self._cancelled_result(owned))
        await self._release_run_lease(owned)
        owned.task = None
        owned.engine = None
        owned.launch_context = None

    @staticmethod
    async def _cleanup_invocation(invocation: SubagentInvocation) -> None:
        errors: list[BaseException] = []
        cancellation: asyncio.CancelledError | None = None
        first_close_error: BaseException | None = None
        for attempt in range(2):
            try:
                await invocation.engine.aclose()
            except asyncio.CancelledError as exc:
                cancellation = exc
                break
            except BaseException as exc:
                if attempt == 0:
                    first_close_error = exc
                    continue
                if first_close_error is not None:
                    errors.append(first_close_error)
                errors.append(exc)
                break
            else:
                first_close_error = None
                break
        if invocation.cleanup is not None:
            try:
                await invocation.cleanup()
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc
            except BaseException as exc:
                errors.append(exc)
        if cancellation is not None:
            if errors:
                raise cancellation from errors[-1]
            raise cancellation
        if errors:
            if len(errors) > 1:
                raise errors[-1] from errors[0]
            raise errors[0]

    @staticmethod
    async def _release_run_lease(owned: _OwnedSubagent) -> None:
        lease = owned.run_lease
        if lease is None:
            return
        await lease.release()
        owned.run_lease = None

    def _current_result(self, owned: _OwnedSubagent) -> SubagentResult:
        if owned.result is not None:
            return owned.result
        return SubagentResult(
            handle=owned.handle,
            request=owned.request,
            status=(
                SubagentStatus.CANCEL_REQUESTED
                if owned.cancel_event.is_set()
                else (
                    SubagentStatus.RUNNING
                    if owned.task is not None
                    else SubagentStatus.PENDING
                )
            ),
            subagent_run_id=owned.subagent_run_id,
        )

    @staticmethod
    def result_payload(result: SubagentResult) -> dict[str, Any]:
        """Project typed subagent state without overloading Tool execution status."""

        return subagent_result_payload(result)

    async def _post_completion_event(
        self,
        owned: _OwnedSubagent,
        result: SubagentResult,
    ) -> None:
        launch_context = owned.launch_context
        if launch_context is None or launch_context.post_runtime_event is None:
            return
        try:
            await launch_context.post_runtime_event(
                subagent_terminal_runtime_input(result)
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            # The terminal result remains queryable when the parent has closed or
            # durable mailbox acceptance fails.
            return

    def _owned(self, handle: SubagentHandle) -> _OwnedSubagent | None:
        if not isinstance(handle, SubagentHandle):
            raise TypeError("handle must be a SubagentHandle")
        return self._subagents.get(handle)

    def _results_for_parent(self, parent_run_id: str) -> tuple[SubagentResult, ...]:
        return tuple(
            self._current_result(owned)
            for owned in self._subagents.values()
            if owned.handle.parent_run_id == parent_run_id
        )

    @staticmethod
    def _decode_lifecycle(
        records: Sequence[JournalRecord],
        *,
        parent_run_id: str,
    ) -> tuple[
        dict[SubagentHandle, _RecoveredSubagentStart],
        dict[SubagentHandle, SubagentResult],
    ]:
        started: dict[SubagentHandle, _RecoveredSubagentStart] = {}
        terminal: dict[SubagentHandle, SubagentResult] = {}
        for record in records:
            if not isinstance(record, JournalRecord):
                raise TypeError("journal replay must contain JournalRecord values")
            if record.run_id != parent_run_id:
                continue
            if record.type is JournalRecordType.SUBAGENT_STARTED:
                payload = dict(record.payload)
                if "child_run_id" in payload:
                    if "subagent_run_id" in payload:
                        raise ValueError(
                            "subagent.started contains conflicting Run id fields"
                        )
                    payload["subagent_run_id"] = payload.pop("child_run_id")
                if set(payload) not in (
                    {"handle", "request"},
                    {"handle", "request", "subagent_run_id"},
                    {"handle", "request", "background"},
                    {"handle", "request", "background", "subagent_run_id"},
                ):
                    raise ValueError("subagent.started fields are invalid")
                raw_handle = payload.get("handle")
                raw_request = payload.get("request")
                raw_subagent_run_id = payload.get("subagent_run_id", "")
                raw_background = payload.get("background", False)
                if not isinstance(raw_handle, Mapping) or not isinstance(
                    raw_request, Mapping
                ):
                    raise ValueError("subagent.started payload is invalid")
                if not isinstance(raw_subagent_run_id, str) or (
                    raw_subagent_run_id and raw_subagent_run_id != raw_subagent_run_id.strip()
                ):
                    raise ValueError("subagent.started subagent_run_id is invalid")
                if not isinstance(raw_background, bool):
                    raise ValueError("subagent.started background flag is invalid")
                handle = SubagentHandle.from_dict(raw_handle)
                request = SubagentLaunchRequest.from_dict(raw_request)
                if handle.parent_run_id != parent_run_id:
                    raise ValueError("subagent.started parent is inconsistent")
                if handle in terminal:
                    raise ValueError("subagent.started occurs after subagent.terminal")
                recovered_start = _RecoveredSubagentStart(
                    request=request,
                    subagent_run_id=raw_subagent_run_id,
                    background=raw_background,
                )
                previous_start = started.get(handle)
                if previous_start is not None and previous_start != recovered_start:
                    raise ValueError("subagent.started conflicts with an earlier record")
                started[handle] = recovered_start
            elif record.type is JournalRecordType.SUBAGENT_TERMINAL:
                result = SubagentResult.from_dict(record.payload)
                if result.handle.parent_run_id != parent_run_id:
                    raise ValueError("subagent.terminal parent is inconsistent")
                if not result.ready:
                    raise ValueError("subagent.terminal contains a live status")
                recovered_start = started.get(result.handle)
                if recovered_start is None:
                    raise ValueError("subagent.terminal has no subagent.started record")
                if result.request != recovered_start.request:
                    raise ValueError("subagent.terminal request is inconsistent")
                started_run_id = recovered_start.subagent_run_id
                if (
                    started_run_id
                    and result.subagent_run_id
                    and result.subagent_run_id != started_run_id
                ):
                    raise ValueError("subagent.terminal Run id is inconsistent")
                previous_result = terminal.get(result.handle)
                if previous_result is not None and previous_result != result:
                    raise ValueError("subagent.terminal conflicts with an earlier record")
                terminal[result.handle] = result
        return started, terminal

    async def _recover_interrupted_result(
        self,
        handle: SubagentHandle,
        start: _RecoveredSubagentStart,
        *,
        root_budget: BudgetSnapshot | None,
    ) -> SubagentResult:
        """Rebuild one started Subagent's terminal fact from its own run journal.

        A subagent that reached a durable run terminal record (loop taxonomy)
        before the parent died is projected from that journal; a subagent without
        a terminal record is closed as interrupted without replaying its Agent.
        """

        if start.subagent_run_id and self._subagent_journal_factory is not None:
            records = await self._replay_subagent_journal(start.subagent_run_id)
            if records is not None:
                try:
                    outcome = recover_run_outcome(records)
                except ValueError as exc:
                    raise SubagentPersistenceError(
                        "subagent run journal is not a recoverable loop journal"
                    ) from exc
                if outcome is not None:
                    return self._result_from_outcome(
                        handle=handle,
                        start=start,
                        outcome=outcome,
                        records=records,
                        root_budget=root_budget,
                    )
        return SubagentResult(
            handle=handle,
            request=start.request,
            status=SubagentStatus.INTERRUPTED,
            conclusion=AgentConclusion(
                failure_paths=(
                    "The parent process exited before the subagent terminal record.",
                )
            ),
            subagent_run_id=start.subagent_run_id,
            error="Subagent side effects may be incomplete; the Agent was not replayed.",
        )

    def _result_from_outcome(
        self,
        *,
        handle: SubagentHandle,
        start: _RecoveredSubagentStart,
        outcome: Any,
        records: Sequence[JournalRecord],
        root_budget: BudgetSnapshot | None,
    ) -> SubagentResult:
        model_pricing = self._model_pricing(records)
        stats = subagent_run_stats(
            outcome.messages,
            model_pricing=model_pricing,
        )
        budget_reason = (
            subagent_budget_stop_reason(stats, start.request.budget, root_budget)
            if outcome.status.value == "completed"
            else None
        )
        stop_reason = budget_reason or subagent_stop_reason(
            outcome.status,
            outcome.error,
        )
        status = self._subagent_status(stop_reason)
        status_detail = stop_reason or "unknown stop reason"
        summary = subagent_final_text(outcome.messages)
        error = outcome.error
        if status is SubagentStatus.COMPLETED and not summary:
            status = SubagentStatus.FAILED
            status_detail = "missing_conclusion"
            error = _MISSING_CONCLUSION_ERROR
        return SubagentResult(
            handle=handle,
            request=start.request,
            status=status,
            conclusion=AgentConclusion(
                summary=summary,
                failure_paths=(status_detail,) if status is SubagentStatus.FAILED else (),
                unknowns=(
                    (status_detail,) if status is SubagentStatus.BUDGET_EXHAUSTED else ()
                ),
            ),
            subagent_run_id=start.subagent_run_id,
            error=error,
            steps=stats.steps,
            total_tokens=stats.total_tokens,
            total_cost_usd=stats.total_cost_usd,
            usage_complete=stats.usage_complete,
            cost_complete=stats.cost_complete,
        )

    @staticmethod
    def _model_pricing(records: Sequence[JournalRecord]) -> ModelPricing | None:
        starts = [
            record
            for record in records
            if record.type is JournalRecordType.RUN_STARTED
        ]
        if not starts:
            return None
        raw = starts[0].payload.get("model_pricing")
        if raw is None:
            return None
        if not isinstance(raw, Mapping) or set(raw) != {
            "input_usd_per_million",
            "output_usd_per_million",
            "cache_read_usd_per_million",
            "cache_write_usd_per_million",
        }:
            raise SubagentPersistenceError("subagent model pricing metadata is invalid")
        try:
            return ModelPricing(**dict(raw))
        except (TypeError, ValueError) as exc:
            raise SubagentPersistenceError(
                "subagent model pricing metadata is invalid"
            ) from exc

    async def _descendant_launch_ids(
        self,
        *,
        parent_run_id: str,
        subagent_run_ids: tuple[str, ...],
    ) -> set[str]:
        # Do not deduplicate here: two lifecycle records claiming the same
        # descendant Run are corrupt lineage and must be rejected explicitly.
        pending = list(subagent_run_ids)
        visited = {parent_run_id}
        launch_ids: set[str] = set()
        while pending:
            run_id = pending.pop(0)
            if run_id in visited:
                raise SubagentPersistenceError(
                    "subagent journal lineage contains a cycle or duplicate Run"
                )
            visited.add(run_id)
            records = await self._replay_subagent_journal(run_id)
            if records is None:
                # A durable launch may fail before its Engine creates a journal.
                continue
            try:
                started, terminal = self._decode_lifecycle(
                    records,
                    parent_run_id=run_id,
                )
            except (TypeError, ValueError) as exc:
                raise SubagentPersistenceError(
                    "descendant subagent lifecycle journal records are invalid"
                ) from exc
            for handle, start in started.items():
                launch_ids.add(f"{handle.parent_run_id}:{handle.subagent_id}")
                terminal_result = terminal.get(handle)
                resolved_run_id = (
                    terminal_result.subagent_run_id
                    if terminal_result is not None and terminal_result.subagent_run_id
                    else start.subagent_run_id
                )
                if resolved_run_id:
                    pending.append(resolved_run_id)
        return launch_ids

    async def _replay_subagent_journal(
        self,
        run_id: str,
    ) -> tuple[JournalRecord, ...] | None:
        factory = self._subagent_journal_factory
        if factory is None:
            return None
        journal = factory()
        if not isinstance(journal, SessionJournal):
            raise TypeError("subagent_journal_factory must return a SessionJournal")

        records: tuple[JournalRecord, ...] | None = None
        missing = False
        failure: BaseException | None = None
        try:
            await journal.open(run_id)
            records = await journal.replay()
        except BaseException as exc:
            if _caused_by_file_not_found(exc):
                missing = True
            else:
                failure = exc
        try:
            await journal.close()
        except asyncio.CancelledError as exc:
            if not isinstance(failure, asyncio.CancelledError):
                raise exc from failure
            raise
        except BaseException as exc:
            if failure is None:
                failure = exc
            else:
                _logger.warning(
                    "Subagent journal cleanup also failed for Run %s",
                    run_id,
                    exc_info=exc,
                )
        if failure is not None:
            raise failure
        if missing:
            return None
        if records is None:
            raise RuntimeError("subagent journal replay returned no records")
        return records

    async def _lifecycle_lock(self, parent_run_id: str) -> asyncio.Lock:
        async with self._admission_lock:
            lifecycle_lock = self._lifecycle_locks.get(parent_run_id)
            if lifecycle_lock is None:
                lifecycle_lock = asyncio.Lock()
                self._lifecycle_locks[parent_run_id] = lifecycle_lock
            return lifecycle_lock

    @staticmethod
    def _terminal_record_id(handle: SubagentHandle) -> str:
        return f"{handle.parent_run_id}:subagent:{handle.subagent_id}:terminal"

    @staticmethod
    def _persistence_failed_result(
        result: SubagentResult,
        *,
        cause: BaseException | None = None,
    ) -> SubagentResult:
        error = (
            "Subagent reached terminal state but its terminal record was not persisted."
        )
        if cause is not None and str(cause):
            error = f"{error} {cause}"
        return replace(
            result,
            status=SubagentStatus.FAILED,
            conclusion=replace(
                result.conclusion,
                failure_paths=result.conclusion.failure_paths + (error,),
            ),
            error=error,
        )

    @staticmethod
    def _persistence_unknown_result(
        result: SubagentResult,
        *,
        cause: BaseException | None = None,
    ) -> SubagentResult:
        error = (
            "Subagent reached terminal state but its durable terminal outcome is "
            "unknown; close and reopen the Journal before continuing."
        )
        if cause is not None and str(cause):
            error = f"{error} {cause}"
        return replace(
            result,
            status=SubagentStatus.UNKNOWN,
            conclusion=replace(
                result.conclusion,
                unknowns=result.conclusion.unknowns + (error,),
            ),
            error=error,
        )

    @staticmethod
    def _cancel_engine(owned: _OwnedSubagent) -> None:
        if owned.engine is not None:
            owned.engine.cancel("immediate")

    def _cancelled_result(
        self,
        owned: _OwnedSubagent,
        *,
        error: str = "Subagent was cancelled.",
    ) -> SubagentResult:
        error = error or "Subagent was cancelled."
        tokens, cost, usage_complete, cost_complete = self._engine_usage(owned)
        return SubagentResult(
            handle=owned.handle,
            request=owned.request,
            status=SubagentStatus.CANCELLED,
            conclusion=AgentConclusion(
                summary=self._engine_committed_final_text(owned),
                failure_paths=(error,),
            ),
            subagent_run_id=self._subagent_run_id(owned),
            error=error,
            total_tokens=tokens,
            total_cost_usd=cost,
            usage_complete=usage_complete,
            cost_complete=cost_complete,
        )

    def _budget_exhausted_result(
        self,
        owned: _OwnedSubagent,
        *,
        error: str,
        elapsed_seconds: float = 0.0,
    ) -> SubagentResult:
        tokens, cost, usage_complete, cost_complete = self._engine_usage(owned)
        return SubagentResult(
            handle=owned.handle,
            request=owned.request,
            status=SubagentStatus.BUDGET_EXHAUSTED,
            conclusion=AgentConclusion(
                summary=self._engine_committed_final_text(owned),
                unknowns=(error,),
            ),
            subagent_run_id=self._subagent_run_id(owned),
            error=error,
            total_tokens=tokens,
            total_cost_usd=cost,
            usage_complete=usage_complete,
            cost_complete=cost_complete,
            elapsed_seconds=elapsed_seconds,
        )

    @staticmethod
    def _budget_admission_error(context: SubagentLaunchContext) -> str | None:
        ledger = context.budget_ledger
        if ledger is None:
            return None
        snapshot = ledger.snapshot()
        if snapshot.steps_exhausted:
            return "Root step budget is exhausted."
        if snapshot.max_tokens is not None:
            if not snapshot.usage_complete:
                return "Root token usage is incomplete; Subagent admission is closed."
            if snapshot.tokens_exhausted:
                return "Root token budget is exhausted."
        if snapshot.max_cost_usd is not None:
            if not snapshot.cost_complete:
                return "Root cost usage is incomplete; Subagent admission is closed."
            if snapshot.cost_exhausted:
                return "Root cost budget is exhausted."
        return None

    def _failed_result(self, owned: _OwnedSubagent, exc: Exception) -> SubagentResult:
        error = str(exc) or type(exc).__name__
        tokens, cost, usage_complete, cost_complete = self._engine_usage(owned)
        return SubagentResult(
            handle=owned.handle,
            request=owned.request,
            status=SubagentStatus.FAILED,
            conclusion=AgentConclusion(
                summary=self._engine_committed_final_text(owned),
                failure_paths=(error,),
            ),
            subagent_run_id=self._subagent_run_id(owned),
            error=error,
            total_tokens=tokens,
            total_cost_usd=cost,
            usage_complete=usage_complete,
            cost_complete=cost_complete,
        )

    @staticmethod
    def _engine_usage(owned: _OwnedSubagent) -> tuple[int, float, bool, bool]:
        engine = owned.engine
        if engine is None:
            return (0, 0.0, False, False)
        return (
            int(getattr(engine, "token_usage", 0) or 0),
            float(getattr(engine, "cost_usage_usd", 0.0) or 0.0),
            bool(getattr(engine, "usage_complete", False)),
            bool(getattr(engine, "cost_complete", False)),
        )

    @staticmethod
    def _engine_committed_final_text(owned: _OwnedSubagent) -> str:
        engine = owned.engine
        if engine is None:
            return ""
        value = getattr(engine, "committed_final_text", "")
        return value.strip() if isinstance(value, str) else ""

    @staticmethod
    def _subagent_run_id(owned: _OwnedSubagent) -> str:
        engine = owned.engine
        return str(engine.active_run_id if engine is not None else owned.subagent_run_id)

    @staticmethod
    def _tool_result_preview(source: Any) -> str:
        items: list[str] = []
        step = 0
        for message in list(getattr(source, "messages", ()) or ()):
            if isinstance(message, AssistantMessage):
                step += 1
                continue
            if not isinstance(message, ToolResultMessage):
                continue
            result = message.result
            text = (
                str(result.error or "")
                if result.output is None and result.error
                else result.text
            ).strip()
            if text:
                items.append(f"[step {step} {message.tool_name}] {text}")
        if not items:
            return ""
        rendered = "\n".join(items[-_TOOL_RESULT_PREVIEW_MAX_ITEMS:])
        if len(rendered) <= _TOOL_RESULT_PREVIEW_MAX_CHARS:
            return rendered
        marker = "\n...[earlier subagent evidence clipped]...\n"
        tail_size = _TOOL_RESULT_PREVIEW_MAX_CHARS - len(marker)
        return marker + rendered[-tail_size:]

    @staticmethod
    def _subagent_status(stop_reason: str) -> SubagentStatus:
        if stop_reason in {"completed", "final", "success"}:
            return SubagentStatus.COMPLETED
        if stop_reason == "blocked":
            return SubagentStatus.BLOCKED
        if stop_reason == "max_steps" or stop_reason == "context_overflow":
            return SubagentStatus.BUDGET_EXHAUSTED
        if stop_reason.startswith("budget_"):
            return SubagentStatus.BUDGET_EXHAUSTED
        if stop_reason.startswith("cancelled"):
            return SubagentStatus.CANCELLED
        if stop_reason.startswith("interrupt"):
            return SubagentStatus.INTERRUPTED
        return SubagentStatus.FAILED
