"""Structured async supervision for independently stateful child Agent runs."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from typing import Any

from ...core.child import (
    AgentConclusion,
    ChildEngine,
    ChildHandle,
    ChildInvocation,
    ChildInvocationCancelled,
    ChildLaunchContext,
    ChildLaunchRequest,
    ChildPersistenceError,
    ChildResult,
    ChildRunLimitError,
    ChildRuntimeContext,
    ChildStatus,
)
from ...core.journal import (
    JournalAppendCancelled,
    JournalCommitError,
    JournalCommitState,
    JournalRecord,
    JournalRecordType,
    SessionJournal,
)
from ...core.runtime_input import (
    RuntimeInput,
    child_result_payload,
    child_terminal_runtime_input,
)
from ...core.tool_result import ToolResult
from .limits import ChildRunLimiter, _ChildRunLease

_logger = logging.getLogger(__name__)

ChildInvocationFactory = Callable[
    [ChildLaunchRequest, ChildRuntimeContext], Awaitable[ChildInvocation]
]
ChildExecutionScope = Callable[
    [ChildRuntimeContext],
    AbstractContextManager[Any] | AbstractAsyncContextManager[Any],
]
ChildJournalFactory = Callable[[], SessionJournal]

_PARTIAL_RESULT_MAX_ITEMS = 12
_PARTIAL_RESULT_MAX_CHARS = 16_000


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
class _OwnedChild:
    handle: ChildHandle
    child_run_id: str
    request: ChildLaunchRequest
    background: bool
    launch_context: ChildLaunchContext | None
    cancel_event: asyncio.Event
    terminal_event: asyncio.Event
    engine_ready: asyncio.Event
    journal: SessionJournal | None = None
    task: asyncio.Task[ChildResult] | None = None
    engine: ChildEngine | None = None
    result: ChildResult | None = None
    run_lease: _ChildRunLease | None = None


@dataclass(frozen=True, slots=True)
class _RecoveredChildStart:
    request: ChildLaunchRequest
    child_run_id: str
    background: bool


class ChildSupervisor:
    """Own child Engines, Tasks, terminal results, and parent delivery for one Run."""

    def __init__(
        self,
        *,
        invocation_factory: ChildInvocationFactory,
        execution_scope: ChildExecutionScope | None = None,
        max_concurrency: int = 4,
        run_limiter: ChildRunLimiter | None = None,
        child_journal_factory: ChildJournalFactory | None = None,
    ) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self._invocation_factory = invocation_factory
        self._execution_scope = execution_scope
        self._max_concurrency = max_concurrency
        if run_limiter is not None and not isinstance(run_limiter, ChildRunLimiter):
            raise TypeError("run_limiter must be a ChildRunLimiter or None")
        self._run_limiter = run_limiter
        if child_journal_factory is not None and not callable(child_journal_factory):
            raise TypeError("child_journal_factory must be callable or None")
        self._child_journal_factory = child_journal_factory
        self._limit = asyncio.Semaphore(max_concurrency)
        self._admission_lock = asyncio.Lock()
        self._children_started = 0
        self._closed = False
        self._children: dict[ChildHandle, _OwnedChild] = {}
        self._recovered_runs: set[str] = set()
        self._lifecycle_locks: dict[str, asyncio.Lock] = {}

    async def launch(
        self,
        request: ChildLaunchRequest,
        context: ChildLaunchContext,
        *,
        background: bool,
    ) -> ChildResult:
        """Launch one fresh child, returning a running or terminal projection."""

        if not isinstance(request, ChildLaunchRequest):
            raise TypeError("request must be a ChildLaunchRequest")
        if not isinstance(context, ChildLaunchContext):
            raise TypeError("context must be a ChildLaunchContext")
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
                        raise RuntimeError("child supervisor is closed")
                    if (
                        context.max_children > 0
                        and self._children_started >= context.max_children
                    ):
                        raise RuntimeError(
                            "Run child-agent budget exhausted: "
                            f"max_children={context.max_children}."
                        )
                    self._children_started += 1
                    handle = ChildHandle(
                        child_id=f"child-{uuid.uuid4().hex[:12]}",
                        parent_run_id=normalized_parent,
                    )
                    owned = _OwnedChild(
                        handle=handle,
                        child_run_id=f"run_{uuid.uuid4().hex[:12]}",
                        request=request,
                        background=background,
                        launch_context=context,
                        cancel_event=asyncio.Event(),
                        terminal_event=asyncio.Event(),
                        engine_ready=asyncio.Event(),
                        journal=journal,
                        run_lease=run_lease,
                    )
                    self._children[handle] = owned
                    owned.task = asyncio.current_task()
            except BaseException:
                if run_lease is not None:
                    await run_lease.rollback()
                raise

            try:
                await self._persist_started(owned)
            except JournalAppendCancelled as cancellation:
                if cancellation.commit_state is JournalCommitState.NOT_COMMITTED:
                    async with self._admission_lock:
                        self._children.pop(handle, None)
                        self._children_started -= 1
                    if run_lease is not None:
                        await run_lease.rollback()
                    raise

                if cancellation.commit_state is JournalCommitState.COMMITTED:
                    if run_lease is not None:
                        run_lease.commit()
                    cancelled_result = self._cancelled_result(
                        owned,
                        error="Child launch was cancelled after durable admission.",
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
                    async with self._admission_lock:
                        self._children.pop(handle, None)
                        self._children_started -= 1
                    if run_lease is not None:
                        await run_lease.rollback()
                else:
                    await self._retain_failed_admission(
                        owned,
                        cause=commit_error,
                    )
                raise ChildPersistenceError(
                    "failed to persist child.started; child was not executed"
                ) from commit_error
            except BaseException:
                async with self._admission_lock:
                    self._children.pop(handle, None)
                    self._children_started -= 1
                if run_lease is not None:
                    await run_lease.rollback()
                raise
            if run_lease is not None:
                run_lease.commit()

            async with self._admission_lock:
                if self._closed:
                    cancelled = True
                elif background:
                    owned.task = asyncio.create_task(
                        self._supervise_background(owned),
                        name=f"qitos-{handle.child_id}",
                    )
                    return self._current_result(owned)
                else:
                    cancelled = False
            if cancelled:
                result = self._cancelled_result(
                    owned,
                    error="Child supervisor closed before child start.",
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

    def result(self, handle: ChildHandle) -> ChildResult | None:
        """Return one owned child's immutable current state."""

        owned = self._owned(handle)
        return None if owned is None else self._current_result(owned)

    async def wait(
        self,
        handle: ChildHandle,
        *,
        timeout_seconds: float | None = None,
    ) -> ChildResult | None:
        """Wait for terminal state without cancelling the child on timeout."""

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
                async with asyncio.timeout(timeout_seconds):
                    await owned.terminal_event.wait()
        except TimeoutError:
            return self._current_result(owned)
        return self._current_result(owned)

    async def recover(
        self,
        *,
        parent_run_id: str,
        journal: SessionJournal,
    ) -> tuple[ChildResult, ...]:
        """Recover terminal facts and close interrupted children without replay."""

        normalized_parent = str(parent_run_id or "").strip()
        if not normalized_parent:
            raise ValueError("parent_run_id must be a non-empty string")
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
                    for owned in self._children.values()
                ):
                    raise RuntimeError(
                        "cannot recover Child state while owned tasks are active"
                    )
            return await self._recover_once(
                parent_run_id=normalized_parent,
                journal=journal,
            )

    async def _recover_once(
        self,
        *,
        parent_run_id: str,
        journal: SessionJournal,
    ) -> tuple[ChildResult, ...]:
        """Recover one parent while its lifecycle admission is serialized."""

        records = await journal.replay()
        try:
            started, terminal = self._decode_lifecycle(
                records,
                parent_run_id=parent_run_id,
            )
        except (TypeError, ValueError) as exc:
            raise ChildPersistenceError(
                "child lifecycle journal records are invalid"
            ) from exc

        launch_ids = {f"{handle.parent_run_id}:{handle.child_id}" for handle in started}
        if self._child_journal_factory is not None:
            child_run_ids: list[str] = []
            for handle, start in started.items():
                terminal_result = terminal.get(handle)
                resolved_run_id = (
                    terminal_result.child_run_id
                    if terminal_result is not None and terminal_result.child_run_id
                    else start.child_run_id
                )
                if resolved_run_id:
                    child_run_ids.append(resolved_run_id)
            try:
                launch_ids.update(
                    await self._descendant_launch_ids(
                        parent_run_id=parent_run_id,
                        child_run_ids=tuple(child_run_ids),
                    )
                )
            except asyncio.CancelledError:
                raise
            except ChildPersistenceError:
                raise
            except Exception as exc:
                raise ChildPersistenceError(
                    "failed to restore descendant Child launch history"
                ) from exc

        recovered = {
            handle: (
                result
                if result.child_run_id
                else replace(result, child_run_id=started[handle].child_run_id)
            )
            for handle, result in terminal.items()
        }
        for handle, start in started.items():
            if handle in recovered:
                continue
            result = ChildResult(
                handle=handle,
                request=start.request,
                status=ChildStatus.INTERRUPTED,
                conclusion=AgentConclusion(
                    failure_paths=(
                        "The parent process exited before the child terminal record.",
                    )
                ),
                child_run_id=start.child_run_id,
                error="Child side effects may be incomplete; the Engine was not replayed.",
            )
            try:
                await journal.append(
                    JournalRecordType.CHILD_TERMINAL,
                    result.to_dict(),
                    record_id=self._terminal_record_id(handle),
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise ChildPersistenceError(
                    "failed to persist interrupted child terminal record"
                ) from exc
            recovered[handle] = result
        if self._run_limiter is not None:
            try:
                await self._run_limiter.restore_started(launch_ids)
            except ChildRunLimitError as exc:
                raise ChildPersistenceError(
                    "restored Child launch history exceeds the configured Run limit"
                ) from exc

        async with self._admission_lock:
            for handle, result in recovered.items():
                owned = self._children.get(handle)
                if owned is None:
                    start = started[handle]
                    owned = _OwnedChild(
                        handle=handle,
                        child_run_id=result.child_run_id or start.child_run_id,
                        request=result.request,
                        background=start.background,
                        launch_context=None,
                        cancel_event=asyncio.Event(),
                        terminal_event=asyncio.Event(),
                        engine_ready=asyncio.Event(),
                    )
                    self._children[handle] = owned
                self._set_terminal(owned, result)
            self._children_started = max(self._children_started, len(started))
            self._recovered_runs.add(parent_run_id)
        return self._results_for_parent(parent_run_id)

    async def interrupt(
        self,
        handle: ChildHandle,
        *,
        wait_seconds: float = 5.0,
    ) -> ChildResult | None:
        """Cancel one active child and wait a bounded time for its cleanup."""

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
        handle: ChildHandle,
        content: str,
        *,
        timeout_seconds: float | None = None,
    ) -> tuple[bool, ChildResult | None]:
        """Post one parent message to an active Child's durable mailbox."""

        text = str(content or "").strip()
        if not text:
            raise ValueError("content must be a non-empty string")
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative or None")
        owned = self._owned(handle)
        if owned is None:
            return False, None
        current = self._current_result(owned)
        if current.ready:
            return False, current
        try:
            await self._wait_until_ready_or_terminal(
                owned,
                timeout_seconds=timeout_seconds,
            )
        except TimeoutError:
            return False, self._current_result(owned)
        current = self._current_result(owned)
        if current.ready or owned.engine is None:
            return False, current
        post_runtime_event = getattr(owned.engine, "apost_runtime_event", None)
        if not callable(post_runtime_event):
            raise RuntimeError("child Engine does not expose an async runtime mailbox")
        child_run_id = self._child_run_id(owned)
        if not child_run_id:
            return False, current
        event = RuntimeInput(
            event_id=f"{owned.handle.child_id}:parent:{uuid.uuid4().hex}",
            kind="agent.parent.message",
            correlation_id=owned.handle.child_id,
            source="qitos.parent",
            payload={"content": text},
        )
        accepted = await post_runtime_event(event, run_id=child_run_id)
        return bool(accepted), self._current_result(owned)

    def request_interrupt(self, handle: ChildHandle) -> bool:
        """Signal one active child without waiting for terminal cleanup."""

        owned = self._owned(handle)
        if owned is None or self._current_result(owned).ready:
            return False
        if owned.cancel_event.is_set():
            return True
        owned.result = ChildResult(
            handle=owned.handle,
            request=owned.request,
            status=ChildStatus.CANCEL_REQUESTED,
            child_run_id=self._child_run_id(owned),
        )
        owned.cancel_event.set()
        self._cancel_engine(owned)
        if owned.task is not None:
            owned.task.cancel()
        return True

    @property
    def active_count(self) -> int:
        """Return children whose execution has not reached terminal state."""

        return sum(
            1
            for owned in self._children.values()
            if not self._current_result(owned).ready
        )

    def snapshot_events(self) -> list[RuntimeInput]:
        """Project bounded completed Tool evidence from active child Engines."""

        events: list[RuntimeInput] = []
        for owned in self._children.values():
            if self._current_result(owned).ready or owned.engine is None:
                continue
            records = list(getattr(owned.engine, "records", []) or [])
            events.append(
                RuntimeInput(
                    event_id=f"{owned.handle.child_id}:conclude-snapshot",
                    kind="agent.child.snapshot",
                    correlation_id=owned.handle.child_id,
                    source="qitos.agent",
                    payload={
                        "handle": owned.handle.to_dict(),
                        "child_id": owned.handle.child_id,
                        "status": "running",
                        "child_status": ChildStatus.RUNNING.value,
                        "agent_type": owned.request.agent_type,
                        "name": owned.request.name,
                        "description": owned.request.description,
                        "output": self._partial_result(owned.engine),
                        "steps": len(records),
                        "total_tokens": int(
                            getattr(owned.engine, "_token_usage", 0) or 0
                        ),
                        "total_cost_usd": float(
                            getattr(owned.engine, "_cost_usage_usd", 0.0) or 0.0
                        ),
                        "usage_complete": bool(
                            getattr(owned.engine, "_usage_complete", False)
                        ),
                        "cost_complete": bool(
                            getattr(owned.engine, "_cost_complete", False)
                        ),
                        "run_id": self._child_run_id(owned),
                    },
                )
            )
        return events

    def setup(self) -> None:
        """Open this supervisor for a fresh owner Run."""

        if any(
            owned.task is not None and not owned.task.done()
            for owned in self._children.values()
        ):
            raise RuntimeError("cannot reopen a child supervisor with owned tasks")
        self._limit = asyncio.Semaphore(self._max_concurrency)
        self._admission_lock = asyncio.Lock()
        self._children_started = 0
        self._closed = False
        self._children.clear()
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
                for owned in self._children.values()
                if owned.task is not None and not owned.task.done()
            ]
        for owned in owned_tasks:
            requested = self.request_interrupt(owned.handle)
            if not requested and owned.task is not None and not owned.task.done():
                # A terminal result may still own bounded parent delivery.
                # Cancelling that delivery cannot interrupt Child cleanup,
                # which has already completed before terminal publication.
                owned.task.cancel()

        tasks = [owned.task for owned in owned_tasks if owned.task is not None]
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=wait_seconds)
            if done:
                await asyncio.gather(*done, return_exceptions=True)
            if pending:
                # A second Task.cancel() can interrupt the ChildInvocation's
                # own async cleanup. Keep the Tasks registered so a later
                # close/wait can drain them without losing ownership.
                await asyncio.sleep(0)
        for owned in owned_tasks:
            task = owned.task
            if task is not None:
                await self._terminalize_cancelled_task(owned, task)
        return sum(1 for task in tasks if not task.done())

    async def _supervise_background(self, owned: _OwnedChild) -> ChildResult:
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
            # later Child from entering the root Run's active limit.
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

    async def _run_request_with_limit(self, owned: _OwnedChild) -> ChildResult:
        """Apply the supervisor limit uniformly to foreground and background runs."""

        async with self._limit:
            return await self._run_request(owned)

    async def _run_request(self, owned: _OwnedChild) -> ChildResult:
        started = time.monotonic()
        launch_context = owned.launch_context
        if launch_context is None:
            raise RuntimeError("Child launch context was released before execution")
        scoped_context = ChildRuntimeContext(
            launch=launch_context,
            handle=owned.handle,
            child_run_id=owned.child_run_id,
            cancellation_requested=owned.cancel_event.is_set,
        )
        scope = (
            self._execution_scope(scoped_context)
            if self._execution_scope is not None
            else nullcontext()
        )

        async def _run_in_scope() -> ChildResult:
            if owned.cancel_event.is_set():
                raise RuntimeError("Child agent was cancelled before it started")
            return await self._run_invocation(owned, scoped_context)

        try:
            if isinstance(scope, AbstractAsyncContextManager):
                async with scope:
                    result = await _run_in_scope()
            else:
                with scope:
                    result = await _run_in_scope()
        except ChildInvocationCancelled as exc:
            result = self._cancelled_result(owned, error=str(exc))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if owned.cancel_event.is_set():
                result = self._cancelled_result(owned, error=str(exc))
            elif isinstance(exc, TimeoutError):
                error = str(exc) or "Child deadline expired."
                result = ChildResult(
                    handle=owned.handle,
                    request=owned.request,
                    status=ChildStatus.BUDGET_EXHAUSTED,
                    conclusion=AgentConclusion(unknowns=(error,)),
                    error=error,
                    child_run_id=self._child_run_id(owned),
                )
            else:
                result = self._failed_result(owned, exc)
        return replace(
            result,
            elapsed_seconds=max(0.0, time.monotonic() - started),
        )

    async def _run_invocation(
        self,
        owned: _OwnedChild,
        runtime_context: ChildRuntimeContext,
    ) -> ChildResult:
        invocation = await self._invocation_factory(owned.request, runtime_context)
        if not isinstance(invocation, ChildInvocation):
            raise TypeError("invocation_factory must resolve to ChildInvocation")
        owned.engine = invocation.engine
        try:
            if self._closed or owned.cancel_event.is_set():
                invocation.engine.cancel("immediate")
                raise RuntimeError("child supervisor closed before child start")
            owned.engine_ready.set()
            run_kwargs = dict(invocation.run_kwargs)
            requested_run_id = run_kwargs.get("run_id")
            if requested_run_id is not None and requested_run_id != owned.child_run_id:
                raise ValueError(
                    "Child invocation Run id conflicts with its durable launch"
                )
            run_kwargs["run_id"] = owned.child_run_id
            engine_result = await invocation.engine.arun(
                invocation.task,
                **run_kwargs,
            )
        finally:
            await self._cleanup_invocation(invocation)
        state = engine_result.state
        raw_stop_reason = state.stop_reason or ""
        stop_reason = str(getattr(raw_stop_reason, "value", raw_stop_reason))
        summary = str(state.final_result or "") or self._partial_result(engine_result)
        status = self._child_status(stop_reason)
        status_detail = stop_reason or "unknown stop reason"
        return ChildResult(
            handle=owned.handle,
            request=owned.request,
            status=status,
            conclusion=AgentConclusion(
                summary=summary,
                failure_paths=(status_detail,) if status is ChildStatus.FAILED else (),
                unknowns=(
                    (status_detail,) if status is ChildStatus.BUDGET_EXHAUSTED else ()
                ),
            ),
            child_run_id=owned.child_run_id,
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

    def _set_terminal(self, owned: _OwnedChild, result: ChildResult) -> None:
        if owned.terminal_event.is_set():
            if owned.result != result:
                raise ChildPersistenceError(
                    "child terminal result conflicts with its existing terminal state"
                )
            return
        owned.result = result
        owned.terminal_event.set()

    @staticmethod
    async def _wait_until_ready_or_terminal(
        owned: _OwnedChild,
        *,
        timeout_seconds: float | None,
    ) -> None:
        engine_wait = asyncio.create_task(
            owned.engine_ready.wait(),
            name=f"qitos-{owned.handle.child_id}-engine-ready",
        )
        terminal_wait = asyncio.create_task(
            owned.terminal_event.wait(),
            name=f"qitos-{owned.handle.child_id}-terminal-wait",
        )
        waits = (engine_wait, terminal_wait)
        try:
            async with asyncio.timeout(timeout_seconds):
                await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in waits:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*waits, return_exceptions=True)

    async def _persist_started(self, owned: _OwnedChild) -> None:
        if owned.journal is None:
            return
        try:
            await owned.journal.append(
                JournalRecordType.CHILD_STARTED,
                {
                    "handle": owned.handle.to_dict(),
                    "request": owned.request.to_dict(),
                    "background": owned.background,
                    "child_run_id": owned.child_run_id,
                },
                record_id=(
                    f"{owned.handle.parent_run_id}:child:"
                    f"{owned.handle.child_id}:started"
                ),
            )
        except asyncio.CancelledError:
            raise
        except JournalCommitError:
            raise
        except Exception as exc:
            raise ChildPersistenceError(
                "failed to persist child.started; child was not executed"
            ) from exc

    async def _store_terminal(
        self,
        owned: _OwnedChild,
        result: ChildResult,
    ) -> bool:
        if owned.journal is not None:
            try:
                await owned.journal.append(
                    JournalRecordType.CHILD_TERMINAL,
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
        owned: _OwnedChild,
        *,
        cause: BaseException | None,
    ) -> None:
        if owned.run_lease is not None:
            owned.run_lease.commit()
        result = self._cancelled_result(
            owned,
            error=(
                "Child launch was not executed because its durable admission "
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
        owned: _OwnedChild,
        task: asyncio.Task[ChildResult],
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
    async def _cleanup_invocation(invocation: ChildInvocation) -> None:
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
    async def _release_run_lease(owned: _OwnedChild) -> None:
        lease = owned.run_lease
        if lease is None:
            return
        await lease.release()
        owned.run_lease = None

    def _current_result(self, owned: _OwnedChild) -> ChildResult:
        if owned.result is not None:
            return owned.result
        return ChildResult(
            handle=owned.handle,
            request=owned.request,
            status=(
                ChildStatus.CANCEL_REQUESTED
                if owned.cancel_event.is_set()
                else (
                    ChildStatus.RUNNING
                    if owned.task is not None
                    else ChildStatus.PENDING
                )
            ),
            child_run_id=owned.child_run_id,
        )

    @staticmethod
    def result_payload(result: ChildResult) -> dict[str, Any]:
        """Project typed child state without overloading Tool execution status."""

        return child_result_payload(result)

    async def _post_completion_event(
        self,
        owned: _OwnedChild,
        result: ChildResult,
    ) -> None:
        launch_context = owned.launch_context
        if launch_context is None or launch_context.post_runtime_event is None:
            return
        try:
            await launch_context.post_runtime_event(
                child_terminal_runtime_input(result)
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            # The terminal result remains queryable when the parent has closed or
            # durable mailbox acceptance fails.
            return

    def _owned(self, handle: ChildHandle) -> _OwnedChild | None:
        if not isinstance(handle, ChildHandle):
            raise TypeError("handle must be a ChildHandle")
        return self._children.get(handle)

    def _results_for_parent(self, parent_run_id: str) -> tuple[ChildResult, ...]:
        return tuple(
            self._current_result(owned)
            for owned in self._children.values()
            if owned.handle.parent_run_id == parent_run_id
        )

    @staticmethod
    def _decode_lifecycle(
        records: Sequence[JournalRecord],
        *,
        parent_run_id: str,
    ) -> tuple[
        dict[ChildHandle, _RecoveredChildStart],
        dict[ChildHandle, ChildResult],
    ]:
        started: dict[ChildHandle, _RecoveredChildStart] = {}
        terminal: dict[ChildHandle, ChildResult] = {}
        for record in records:
            if not isinstance(record, JournalRecord):
                raise TypeError("journal replay must contain JournalRecord values")
            if record.run_id != parent_run_id:
                continue
            if record.type is JournalRecordType.CHILD_STARTED:
                if set(record.payload) not in (
                    {"handle", "request"},
                    {"handle", "request", "child_run_id"},
                    {"handle", "request", "background"},
                    {"handle", "request", "background", "child_run_id"},
                ):
                    raise ValueError("child.started fields are invalid")
                raw_handle = record.payload.get("handle")
                raw_request = record.payload.get("request")
                raw_child_run_id = record.payload.get("child_run_id", "")
                raw_background = record.payload.get("background", False)
                if not isinstance(raw_handle, Mapping) or not isinstance(
                    raw_request, Mapping
                ):
                    raise ValueError("child.started payload is invalid")
                if not isinstance(raw_child_run_id, str) or (
                    raw_child_run_id and raw_child_run_id != raw_child_run_id.strip()
                ):
                    raise ValueError("child.started child_run_id is invalid")
                if not isinstance(raw_background, bool):
                    raise ValueError("child.started background flag is invalid")
                handle = ChildHandle.from_dict(raw_handle)
                request = ChildLaunchRequest.from_dict(raw_request)
                if handle.parent_run_id != parent_run_id:
                    raise ValueError("child.started parent is inconsistent")
                if handle in terminal:
                    raise ValueError("child.started occurs after child.terminal")
                recovered_start = _RecoveredChildStart(
                    request=request,
                    child_run_id=raw_child_run_id,
                    background=raw_background,
                )
                previous_start = started.get(handle)
                if previous_start is not None and previous_start != recovered_start:
                    raise ValueError("child.started conflicts with an earlier record")
                started[handle] = recovered_start
            elif record.type is JournalRecordType.CHILD_TERMINAL:
                result = ChildResult.from_dict(record.payload)
                if result.handle.parent_run_id != parent_run_id:
                    raise ValueError("child.terminal parent is inconsistent")
                if not result.ready:
                    raise ValueError("child.terminal contains a live status")
                recovered_start = started.get(result.handle)
                if recovered_start is None:
                    raise ValueError("child.terminal has no child.started record")
                if result.request != recovered_start.request:
                    raise ValueError("child.terminal request is inconsistent")
                started_run_id = recovered_start.child_run_id
                if (
                    started_run_id
                    and result.child_run_id
                    and result.child_run_id != started_run_id
                ):
                    raise ValueError("child.terminal Run id is inconsistent")
                previous_result = terminal.get(result.handle)
                if previous_result is not None and previous_result != result:
                    raise ValueError("child.terminal conflicts with an earlier record")
                terminal[result.handle] = result
        return started, terminal

    async def _descendant_launch_ids(
        self,
        *,
        parent_run_id: str,
        child_run_ids: tuple[str, ...],
    ) -> set[str]:
        # Do not deduplicate here: two lifecycle records claiming the same
        # descendant Run are corrupt lineage and must be rejected explicitly.
        pending = list(child_run_ids)
        visited = {parent_run_id}
        launch_ids: set[str] = set()
        while pending:
            run_id = pending.pop(0)
            if run_id in visited:
                raise ChildPersistenceError(
                    "child journal lineage contains a cycle or duplicate Run"
                )
            visited.add(run_id)
            records = await self._replay_child_journal(run_id)
            if records is None:
                # A durable launch may fail before its Engine creates a journal.
                continue
            try:
                started, terminal = self._decode_lifecycle(
                    records,
                    parent_run_id=run_id,
                )
            except (TypeError, ValueError) as exc:
                raise ChildPersistenceError(
                    "descendant child lifecycle journal records are invalid"
                ) from exc
            for handle, start in started.items():
                launch_ids.add(f"{handle.parent_run_id}:{handle.child_id}")
                terminal_result = terminal.get(handle)
                resolved_run_id = (
                    terminal_result.child_run_id
                    if terminal_result is not None and terminal_result.child_run_id
                    else start.child_run_id
                )
                if resolved_run_id:
                    pending.append(resolved_run_id)
        return launch_ids

    async def _replay_child_journal(
        self,
        run_id: str,
    ) -> tuple[JournalRecord, ...] | None:
        factory = self._child_journal_factory
        if factory is None:
            return None
        journal = factory()
        if not isinstance(journal, SessionJournal):
            raise TypeError("child_journal_factory must return a SessionJournal")

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
                    "Child journal cleanup also failed for Run %s",
                    run_id,
                    exc_info=exc,
                )
        if failure is not None:
            raise failure
        if missing:
            return None
        if records is None:
            raise RuntimeError("child journal replay returned no records")
        return records

    async def _lifecycle_lock(self, parent_run_id: str) -> asyncio.Lock:
        async with self._admission_lock:
            lifecycle_lock = self._lifecycle_locks.get(parent_run_id)
            if lifecycle_lock is None:
                lifecycle_lock = asyncio.Lock()
                self._lifecycle_locks[parent_run_id] = lifecycle_lock
            return lifecycle_lock

    @staticmethod
    def _terminal_record_id(handle: ChildHandle) -> str:
        return f"{handle.parent_run_id}:child:{handle.child_id}:terminal"

    @staticmethod
    def _persistence_failed_result(
        result: ChildResult,
        *,
        cause: BaseException | None = None,
    ) -> ChildResult:
        error = (
            "Child reached terminal state but its terminal record was not persisted."
        )
        if cause is not None and str(cause):
            error = f"{error} {cause}"
        return replace(
            result,
            status=ChildStatus.FAILED,
            conclusion=replace(
                result.conclusion,
                failure_paths=result.conclusion.failure_paths + (error,),
            ),
            error=error,
        )

    @staticmethod
    def _persistence_unknown_result(
        result: ChildResult,
        *,
        cause: BaseException | None = None,
    ) -> ChildResult:
        error = (
            "Child reached terminal state but its durable terminal outcome is "
            "unknown; close and reopen the Journal before continuing."
        )
        if cause is not None and str(cause):
            error = f"{error} {cause}"
        return replace(
            result,
            status=ChildStatus.UNKNOWN,
            conclusion=replace(
                result.conclusion,
                unknowns=result.conclusion.unknowns + (error,),
            ),
            error=error,
        )

    @staticmethod
    def _cancel_engine(owned: _OwnedChild) -> None:
        if owned.engine is not None:
            owned.engine.cancel("immediate")

    def _cancelled_result(
        self,
        owned: _OwnedChild,
        *,
        error: str = "Child agent was cancelled.",
    ) -> ChildResult:
        error = error or "Child agent was cancelled."
        tokens, cost, usage_complete, cost_complete = self._engine_usage(owned)
        return ChildResult(
            handle=owned.handle,
            request=owned.request,
            status=ChildStatus.CANCELLED,
            conclusion=AgentConclusion(failure_paths=(error,)),
            child_run_id=self._child_run_id(owned),
            error=error,
            total_tokens=tokens,
            total_cost_usd=cost,
            usage_complete=usage_complete,
            cost_complete=cost_complete,
        )

    def _failed_result(self, owned: _OwnedChild, exc: Exception) -> ChildResult:
        error = str(exc) or type(exc).__name__
        tokens, cost, usage_complete, cost_complete = self._engine_usage(owned)
        return ChildResult(
            handle=owned.handle,
            request=owned.request,
            status=ChildStatus.FAILED,
            conclusion=AgentConclusion(failure_paths=(error,)),
            child_run_id=self._child_run_id(owned),
            error=error,
            total_tokens=tokens,
            total_cost_usd=cost,
            usage_complete=usage_complete,
            cost_complete=cost_complete,
        )

    @staticmethod
    def _engine_usage(owned: _OwnedChild) -> tuple[int, float, bool, bool]:
        engine = owned.engine
        if engine is None:
            return (0, 0.0, False, False)
        return (
            int(getattr(engine, "_token_usage", 0) or 0),
            float(getattr(engine, "_cost_usage_usd", 0.0) or 0.0),
            bool(getattr(engine, "_usage_complete", False)),
            bool(getattr(engine, "_cost_complete", False)),
        )

    @staticmethod
    def _child_run_id(owned: _OwnedChild) -> str:
        engine = owned.engine
        return str(engine.active_run_id if engine is not None else owned.child_run_id)

    @staticmethod
    def _partial_result(engine_result: Any) -> str:
        items: list[str] = []
        for record in list(getattr(engine_result, "records", []) or []):
            step_id = int(getattr(record, "step_id", 0) or 0)
            invocations = list(getattr(record, "tool_invocations", []) or [])
            results = list(getattr(record, "action_results", []) or [])
            for index, raw_result in enumerate(results):
                invocation = invocations[index] if index < len(invocations) else {}
                tool_name = (
                    str(invocation.get("tool_name", "") or "tool")
                    if isinstance(invocation, dict)
                    else "tool"
                )
                result = ToolResult.from_value(raw_result)
                text = (
                    str(result.error or "")
                    if result.output is None and result.error
                    else result.text
                ).strip()
                if text:
                    items.append(f"[step {step_id} {tool_name}] {text}")
        if not items:
            return ""
        rendered = "\n".join(items[-_PARTIAL_RESULT_MAX_ITEMS:])
        if len(rendered) <= _PARTIAL_RESULT_MAX_CHARS:
            return rendered
        marker = "\n...[earlier child evidence clipped]...\n"
        tail_size = _PARTIAL_RESULT_MAX_CHARS - len(marker)
        return marker + rendered[-tail_size:]

    @staticmethod
    def _child_status(stop_reason: str) -> ChildStatus:
        if stop_reason in {"completed", "final", "success"}:
            return ChildStatus.COMPLETED
        if stop_reason == "blocked":
            return ChildStatus.BLOCKED
        if stop_reason == "max_steps" or stop_reason == "context_overflow":
            return ChildStatus.BUDGET_EXHAUSTED
        if stop_reason.startswith("budget_"):
            return ChildStatus.BUDGET_EXHAUSTED
        if stop_reason.startswith("cancelled"):
            return ChildStatus.CANCELLED
        if stop_reason.startswith("interrupt"):
            return ChildStatus.INTERRUPTED
        return ChildStatus.FAILED
