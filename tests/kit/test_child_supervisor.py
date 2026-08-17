"""Structured-concurrency tests for the Run-owned child supervisor."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
import time
from types import SimpleNamespace
from typing import Any

import pytest

from qitos.core.child import (
    AgentConclusion,
    ChildHandle,
    ChildInvocation,
    ChildInvocationCancelled,
    ChildLaunchContext,
    ChildLaunchRequest,
    ChildPostRuntimeEvent,
    ChildPersistenceError,
    ChildResult,
    ChildRuntimeContext,
    ChildStatus,
)
from qitos.core.journal import (
    JournalAppendCancelled,
    JournalCommitError,
    JournalCommitState,
    JournalError,
    JournalPosition,
    JournalRecordType,
    SessionJournal,
)
from qitos.kit.child import ChildRunLimiter, ChildSupervisor
from qitos.kit.journal import JsonlSessionJournal


def _request(task: str = "inspect") -> ChildLaunchRequest:
    return ChildLaunchRequest(task=task, description=f"{task} task")


def _context(
    *,
    journal: SessionJournal | None = None,
    post_runtime_event: ChildPostRuntimeEvent | None = None,
    deadline_monotonic: float | None = None,
) -> ChildLaunchContext:
    return ChildLaunchContext(
        parent_run_id="parent-run",
        journal=journal,
        post_runtime_event=post_runtime_event,
        deadline_monotonic=deadline_monotonic,
    )


async def _set_event(event: asyncio.Event) -> None:
    event.set()


async def _ready_invocation(**kwargs: Any) -> ChildInvocation:
    return ChildInvocation(**kwargs)


class _ClosableEngine:
    async def aclose(self) -> None:
        return None


class _CompletingEngine(_ClosableEngine):
    active_run_id = "child-run"

    async def arun(self, task: str, **kwargs: object) -> object:
        run_id = kwargs.pop("run_id")
        assert isinstance(run_id, str)
        assert kwargs == {}
        return SimpleNamespace(
            state=SimpleNamespace(final_result=f"done:{task}", stop_reason="completed"),
            records=[],
            step_count=1,
            total_tokens=2,
            run_id=run_id,
        )

    def cancel(self, mode: str) -> None:
        _ = mode


class _FailingChildJournal(JsonlSessionJournal):
    def __init__(self, *args: object, fail_type: JournalRecordType) -> None:
        super().__init__(*args)
        self._fail_type = fail_type

    async def append(
        self,
        record_type: JournalRecordType,
        payload,
        *,
        record_id: str,
    ):
        if record_type is self._fail_type:
            raise JournalError(f"injected {record_type.value} failure")
        return await super().append(record_type, payload, record_id=record_id)


class _CommittedCancellationJournal(JsonlSessionJournal):
    async def append(
        self,
        record_type: JournalRecordType,
        payload,
        *,
        record_id: str,
    ):
        position = await super().append(
            record_type,
            payload,
            record_id=record_id,
        )
        if record_type is JournalRecordType.CHILD_STARTED:
            raise JournalAppendCancelled(position)
        return position


class _UnknownCancellationJournal(JsonlSessionJournal):
    async def append(
        self,
        record_type: JournalRecordType,
        payload,
        *,
        record_id: str,
    ):
        if record_type is JournalRecordType.CHILD_STARTED:
            raise JournalAppendCancelled(
                None,
                commit_state=JournalCommitState.UNKNOWN,
                pending_position=JournalPosition(
                    run_id=self.run_id,
                    seq=2,
                    record_id=record_id,
                ),
                commit_error=OSError("injected uncertain commit"),
            )
        return await super().append(record_type, payload, record_id=record_id)


class _CommitErrorJournal(JsonlSessionJournal):
    def __init__(
        self,
        *args: object,
        fail_type: JournalRecordType,
        commit_state: JournalCommitState,
    ) -> None:
        super().__init__(*args)
        self._fail_type = fail_type
        self._commit_state = commit_state

    async def append(
        self,
        record_type: JournalRecordType,
        payload,
        *,
        record_id: str,
    ):
        if record_type is self._fail_type:
            raise JournalCommitError(
                JournalPosition(self.run_id, 2, record_id),
                self._commit_state,
                cause=OSError("injected commit failure"),
            )
        return await super().append(record_type, payload, record_id=record_id)


@pytest.mark.asyncio
async def test_parent_deadline_bounds_waiting_invocation_factory(
    monkeypatch,
) -> None:
    started = asyncio.Event()
    never = asyncio.Event()

    def unsupported_timeout_at(*_args, **_kwargs):
        raise AssertionError("the Child deadline path requires Python 3.10 support")

    monkeypatch.setattr(asyncio, "timeout_at", unsupported_timeout_at, raising=False)

    async def build(_request, _context):
        started.set()
        await never.wait()
        raise AssertionError("unreachable")

    supervisor = ChildSupervisor(invocation_factory=build)
    result = await supervisor.launch(
        _request(),
        _context(deadline_monotonic=time.monotonic() + 0.02),
        background=False,
    )

    assert started.is_set()
    assert result.status is ChildStatus.BUDGET_EXHAUSTED
    assert result.ready is True
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_parent_deadline_drains_engine_and_invocation_cleanup() -> None:
    started = asyncio.Event()
    engine_cancelled = asyncio.Event()
    engine_settled = asyncio.Event()
    invocation_cleaned = asyncio.Event()

    class BlockingEngine(_ClosableEngine):
        active_run_id = "child-run"

        async def arun(self, task: str, **kwargs: object) -> object:
            _ = task, kwargs
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                engine_settled.set()

        def cancel(self, mode: str) -> None:
            assert mode == "immediate"
            engine_cancelled.set()

    async def cleanup() -> None:
        invocation_cleaned.set()

    supervisor = ChildSupervisor(
        invocation_factory=lambda request, _context: _ready_invocation(
            engine=BlockingEngine(),
            task=request.task,
            cleanup=cleanup,
        )
    )
    result = await supervisor.launch(
        _request(),
        _context(deadline_monotonic=time.monotonic() + 0.02),
        background=False,
    )

    assert started.is_set()
    assert result.status is ChildStatus.BUDGET_EXHAUSTED
    assert engine_cancelled.is_set()
    assert engine_settled.is_set()
    assert invocation_cleaned.is_set()
    assert not {
        task.get_name()
        for task in asyncio.all_tasks()
        if not task.done() and "deadline-" in task.get_name()
    }
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_parent_deadline_bounds_waiting_supervisor_slot() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    factory_calls = 0

    class BlockingEngine(_ClosableEngine):
        active_run_id = "child-run"

        async def arun(self, task: str, **kwargs: object) -> object:
            _ = task, kwargs
            started.set()
            await release.wait()
            return SimpleNamespace(
                state=SimpleNamespace(final_result="done", stop_reason="completed"),
                records=[],
                step_count=1,
                total_tokens=0,
                run_id="child-run",
            )

        def cancel(self, mode: str) -> None:
            _ = mode
            release.set()

    async def build(request, _context):
        nonlocal factory_calls
        factory_calls += 1
        return ChildInvocation(engine=BlockingEngine(), task=request.task)

    supervisor = ChildSupervisor(invocation_factory=build, max_concurrency=1)
    first = await supervisor.launch(_request("first"), _context(), background=True)
    await asyncio.wait_for(started.wait(), timeout=1)

    second = await supervisor.launch(
        _request("second"),
        _context(deadline_monotonic=time.monotonic() + 0.02),
        background=False,
    )

    assert first.status is ChildStatus.RUNNING
    assert second.status is ChildStatus.BUDGET_EXHAUSTED
    assert factory_calls == 1
    release.set()
    terminal = await supervisor.wait(first.handle, timeout_seconds=1)
    assert terminal is not None and terminal.status is ChildStatus.COMPLETED
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_factory_timeout_fault_is_not_misreported_as_parent_deadline() -> None:
    async def build(_request, _context):
        raise TimeoutError("factory fault")

    supervisor = ChildSupervisor(invocation_factory=build)
    result = await supervisor.launch(
        _request(),
        _context(deadline_monotonic=time.monotonic() + 1.0),
        background=False,
    )

    assert result.status is ChildStatus.FAILED
    assert result.error == "factory fault"
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_foreground_children_share_supervisor_concurrency_limit() -> None:
    started: asyncio.Queue[str] = asyncio.Queue()
    release_first = asyncio.Event()
    active = 0
    peak = 0

    class Engine(_ClosableEngine):
        active_run_id = "child-run"

        async def arun(self, task: str, **kwargs: object) -> object:
            nonlocal active, peak
            run_id = kwargs.pop("run_id")
            assert isinstance(run_id, str)
            assert kwargs == {}
            active += 1
            peak = max(peak, active)
            await started.put(task)
            try:
                if task == "one":
                    await release_first.wait()
            finally:
                active -= 1
            return SimpleNamespace(
                state=SimpleNamespace(final_result=task, stop_reason="completed"),
                records=[],
                step_count=1,
                total_tokens=1,
                run_id=run_id,
            )

    supervisor = ChildSupervisor(
        invocation_factory=lambda request, _context: _ready_invocation(
            engine=Engine(),
            task=request.task,
        ),
        max_concurrency=1,
    )
    first = asyncio.create_task(
        supervisor.launch(_request("one"), _context(), background=False)
    )
    assert await asyncio.wait_for(started.get(), timeout=1) == "one"
    second = asyncio.create_task(
        supervisor.launch(_request("two"), _context(), background=False)
    )
    await asyncio.sleep(0)

    assert started.empty()
    assert peak == 1

    release_first.set()
    first_result, second_result = await asyncio.gather(first, second)

    assert first_result.status is ChildStatus.COMPLETED
    assert second_result.status is ChildStatus.COMPLETED
    assert await asyncio.wait_for(started.get(), timeout=1) == "two"
    assert peak == 1


@pytest.mark.asyncio
async def test_foreground_child_local_cancellation_does_not_cancel_parent() -> None:
    async def cancelled_factory(
        request: ChildLaunchRequest,
        _context: ChildRuntimeContext,
    ) -> ChildInvocation:
        _ = request
        raise ChildInvocationCancelled("Child construction cleanup was cancelled")

    supervisor = ChildSupervisor(invocation_factory=cancelled_factory)

    result = await supervisor.launch(
        _request(),
        _context(),
        background=False,
    )

    assert result.status is ChildStatus.CANCELLED
    assert result.error == "Child construction cleanup was cancelled"
    assert result.child_run_id
    assert await supervisor.aclose() == 0


@pytest.mark.asyncio
async def test_foreground_child_preserves_real_caller_cancellation() -> None:
    factory_started = asyncio.Event()
    factory_settled = asyncio.Event()

    async def waiting_factory(
        request: ChildLaunchRequest,
        _context: ChildRuntimeContext,
    ) -> ChildInvocation:
        _ = request
        factory_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            factory_settled.set()

    supervisor = ChildSupervisor(invocation_factory=waiting_factory)
    launch_task = asyncio.create_task(
        supervisor.launch(
            _request(),
            _context(deadline_monotonic=time.monotonic() + 10),
            background=False,
        )
    )
    await factory_started.wait()
    launch_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await launch_task

    assert factory_settled.is_set()
    assert not {
        task.get_name()
        for task in asyncio.all_tasks()
        if not task.done() and "deadline-" in task.get_name()
    }
    assert await supervisor.aclose() == 0


@pytest.mark.asyncio
async def test_wait_timeout_does_not_cancel_child() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class Engine(_ClosableEngine):
        active_run_id = "child-run"

        async def arun(self, task: str, **kwargs: object) -> object:
            assert task == "inspect"
            run_id = kwargs.pop("run_id")
            assert isinstance(run_id, str)
            assert kwargs == {}
            started.set()
            await release.wait()
            return SimpleNamespace(
                state=SimpleNamespace(final_result="done", stop_reason="completed"),
                records=[],
                step_count=1,
                total_tokens=2,
                run_id=run_id,
            )

        def cancel(self, mode: str) -> None:
            assert mode == "immediate"
            release.set()

    supervisor = ChildSupervisor(
        invocation_factory=lambda request, _context: _ready_invocation(
            engine=Engine(),
            task=request.task,
        )
    )
    launched = await supervisor.launch(
        _request(),
        _context(),
        background=True,
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    waiting = await supervisor.wait(launched.handle, timeout_seconds=0)

    assert waiting is not None
    assert waiting.status is ChildStatus.RUNNING
    assert supervisor.active_count == 1

    release.set()
    terminal = await supervisor.wait(launched.handle, timeout_seconds=1)
    assert terminal is not None
    assert terminal.status is ChildStatus.COMPLETED
    assert await supervisor.aclose(wait_seconds=1) == 0


@pytest.mark.asyncio
async def test_terminal_delivery_remains_owned_until_close() -> None:
    delivery_started = asyncio.Event()
    delivery_cancelled = asyncio.Event()

    class Engine(_ClosableEngine):
        active_run_id = "child-run"

        async def arun(self, task: str, **kwargs: object) -> object:
            _ = task, kwargs
            return SimpleNamespace(
                state=SimpleNamespace(final_result="done", stop_reason="completed"),
                records=[],
                step_count=1,
                total_tokens=2,
                run_id=self.active_run_id,
            )

        def cancel(self, mode: str) -> None:
            _ = mode

    async def post_runtime_event(_event: object) -> bool:
        delivery_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            delivery_cancelled.set()
            raise
        return True  # pragma: no cover

    supervisor = ChildSupervisor(
        invocation_factory=lambda request, _context: _ready_invocation(
            engine=Engine(),
            task=request.task,
        )
    )
    launched = await supervisor.launch(
        _request(),
        _context(post_runtime_event=post_runtime_event),
        background=True,
    )
    await asyncio.wait_for(delivery_started.wait(), timeout=1)

    terminal = supervisor.result(launched.handle)
    assert terminal is not None
    assert terminal.status is ChildStatus.COMPLETED
    assert supervisor.active_count == 0
    assert await supervisor.aclose(wait_seconds=1) == 0
    assert delivery_cancelled.is_set()
    assert supervisor.result(launched.handle) == terminal


@pytest.mark.asyncio
async def test_interrupt_waits_for_started_child_cleanup() -> None:
    started = asyncio.Event()
    cleaned = asyncio.Event()
    resource_closed = asyncio.Event()

    class Engine(_ClosableEngine):
        active_run_id = "child-run"

        async def arun(self, task: str, **kwargs: object) -> object:
            _ = task, kwargs
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        def cancel(self, mode: str) -> None:
            assert mode == "immediate"

    supervisor = ChildSupervisor(
        invocation_factory=lambda request, _context: _ready_invocation(
            engine=Engine(),
            task=request.task,
            cleanup=lambda: _set_event(resource_closed),
        )
    )
    launched = await supervisor.launch(
        _request(),
        _context(),
        background=True,
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    terminal = await supervisor.interrupt(launched.handle, wait_seconds=1)

    assert terminal is not None
    assert terminal.status is ChildStatus.CANCELLED
    assert cleaned.is_set()
    assert resource_closed.is_set()
    assert supervisor.active_count == 0
    assert await supervisor.aclose(wait_seconds=0) == 0


@pytest.mark.asyncio
async def test_invocation_cleanup_failure_is_a_terminal_child_failure() -> None:
    async def fail_cleanup() -> None:
        raise RuntimeError("cleanup failed")

    supervisor = ChildSupervisor(
        invocation_factory=lambda request, _context: _ready_invocation(
            engine=_CompletingEngine(),
            task=request.task,
            cleanup=fail_cleanup,
        )
    )

    result = await supervisor.launch(
        _request(),
        _context(),
        background=False,
    )

    assert result.status is ChildStatus.FAILED
    assert result.error == "cleanup failed"
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_invocation_factory_can_finish_async_resource_construction() -> None:
    factory_finished = asyncio.Event()

    async def invocation_factory(
        request: ChildLaunchRequest,
        _context: ChildRuntimeContext,
    ) -> ChildInvocation:
        await asyncio.sleep(0)
        factory_finished.set()
        return ChildInvocation(engine=_CompletingEngine(), task=request.task)

    supervisor = ChildSupervisor(invocation_factory=invocation_factory)

    result = await supervisor.launch(
        _request(),
        _context(),
        background=False,
    )

    assert factory_finished.is_set()
    assert result.status is ChildStatus.COMPLETED
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_engine_cleanup_retries_one_incomplete_close() -> None:
    class Engine(_CompletingEngine):
        def __init__(self) -> None:
            self.close_calls = 0

        async def aclose(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("cleanup remains incomplete")

    engine = Engine()
    supervisor = ChildSupervisor(
        invocation_factory=lambda request, _context: _ready_invocation(
            engine=engine,
            task=request.task,
        )
    )

    result = await supervisor.launch(
        _request(),
        _context(),
        background=False,
    )

    assert result.status is ChildStatus.COMPLETED
    assert engine.close_calls == 2
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_invocation_cannot_override_durable_child_run_id() -> None:
    supervisor = ChildSupervisor(
        invocation_factory=lambda request, _context: _ready_invocation(
            engine=_CompletingEngine(),
            task=request.task,
            run_kwargs={"run_id": "conflicting-run"},
        )
    )

    result = await supervisor.launch(
        _request(),
        _context(),
        background=False,
    )

    assert result.status is ChildStatus.FAILED
    assert "conflicts with its durable launch" in str(result.error)
    assert result.child_run_id
    assert result.child_run_id != "conflicting-run"
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_interrupt_cancels_async_invocation_construction() -> None:
    factory_started = asyncio.Event()
    factory_cancelled = asyncio.Event()

    async def invocation_factory(
        request: ChildLaunchRequest,
        _context: ChildRuntimeContext,
    ) -> ChildInvocation:
        _ = request
        factory_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            factory_cancelled.set()
            raise
        raise AssertionError("unreachable")  # pragma: no cover

    supervisor = ChildSupervisor(invocation_factory=invocation_factory)
    launched = await supervisor.launch(
        _request(),
        _context(),
        background=True,
    )
    await asyncio.wait_for(factory_started.wait(), timeout=1)

    terminal = await supervisor.interrupt(launched.handle, wait_seconds=1)

    assert terminal is not None
    assert terminal.status is ChildStatus.CANCELLED
    assert factory_cancelled.is_set()
    assert supervisor.active_count == 0
    assert await supervisor.aclose(wait_seconds=0) == 0


@pytest.mark.asyncio
async def test_invocation_returned_after_interrupt_is_cleaned_without_starting() -> (
    None
):
    factory_started = asyncio.Event()
    release_factory = asyncio.Event()
    cleanup_called = asyncio.Event()
    engine_cancelled = asyncio.Event()
    engine_closed = asyncio.Event()

    class Engine(_ClosableEngine):
        active_run_id = ""

        async def arun(self, task: str, **kwargs: object) -> object:
            _ = task, kwargs
            raise AssertionError("cancelled invocation must not start")

        def cancel(self, mode: str) -> None:
            assert mode == "immediate"
            engine_cancelled.set()

        async def aclose(self) -> None:
            engine_closed.set()

    async def invocation_factory(
        request: ChildLaunchRequest,
        _context: ChildRuntimeContext,
    ) -> ChildInvocation:
        factory_started.set()
        try:
            await release_factory.wait()
        except asyncio.CancelledError:
            # A factory that finishes an atomic acquisition despite cancellation
            # still transfers the returned invocation to the supervisor.
            pass
        return ChildInvocation(
            engine=Engine(),
            task=request.task,
            cleanup=lambda: _set_event(cleanup_called),
        )

    supervisor = ChildSupervisor(invocation_factory=invocation_factory)
    launched = await supervisor.launch(
        _request(),
        _context(),
        background=True,
    )
    await asyncio.wait_for(factory_started.wait(), timeout=1)

    terminal = await supervisor.interrupt(launched.handle, wait_seconds=1)

    assert terminal is not None
    assert terminal.status is ChildStatus.CANCELLED
    assert engine_cancelled.is_set()
    assert engine_closed.is_set()
    assert cleanup_called.is_set()
    assert supervisor.active_count == 0
    assert await supervisor.aclose(wait_seconds=0) == 0


@pytest.mark.asyncio
async def test_close_terminalizes_child_cancelled_before_task_start() -> None:
    factory_called = False

    def invocation_factory(request, _context):
        nonlocal factory_called
        factory_called = True
        return _ready_invocation(engine=object(), task=request.task)

    supervisor = ChildSupervisor(invocation_factory=invocation_factory)
    launched = await supervisor.launch(
        _request(),
        _context(),
        background=True,
    )

    assert await supervisor.aclose(wait_seconds=0) == 0
    terminal = supervisor.result(launched.handle)
    assert terminal is not None
    assert terminal.status is ChildStatus.CANCELLED
    assert factory_called is False


@pytest.mark.asyncio
async def test_immediate_interrupt_persists_child_cancelled_before_task_start(
    tmp_path,
) -> None:
    factory_called = False
    journal = JsonlSessionJournal(tmp_path / "journal")
    await journal.create("parent-run", {})

    def invocation_factory(request, _context):
        nonlocal factory_called
        factory_called = True
        return _ready_invocation(engine=object(), task=request.task)

    supervisor = ChildSupervisor(invocation_factory=invocation_factory)
    launched = await supervisor.launch(
        _request(),
        _context(journal=journal),
        background=True,
    )

    terminal = await supervisor.interrupt(launched.handle, wait_seconds=0)

    assert terminal is not None
    assert terminal.status is ChildStatus.CANCELLED
    assert supervisor.active_count == 0
    assert factory_called is False
    records = await journal.replay()
    assert [
        record.type
        for record in records
        if record.type
        in {JournalRecordType.CHILD_STARTED, JournalRecordType.CHILD_TERMINAL}
    ] == [JournalRecordType.CHILD_STARTED, JournalRecordType.CHILD_TERMINAL]
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_repeated_interrupt_does_not_cancel_invocation_cleanup() -> None:
    started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_cancelled = asyncio.Event()

    class Engine(_ClosableEngine):
        active_run_id = "child-run"

        async def arun(self, task: str, **kwargs: object) -> object:
            _ = task, kwargs
            started.set()
            await asyncio.Event().wait()

        def cancel(self, mode: str) -> None:
            assert mode == "immediate"

    async def cleanup() -> None:
        cleanup_started.set()
        try:
            await release_cleanup.wait()
        except asyncio.CancelledError:
            cleanup_cancelled.set()
            raise

    supervisor = ChildSupervisor(
        invocation_factory=lambda request, _context: _ready_invocation(
            engine=Engine(),
            task=request.task,
            cleanup=cleanup,
        )
    )
    launched = await supervisor.launch(
        _request(),
        _context(),
        background=True,
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    assert supervisor.request_interrupt(launched.handle) is True
    await asyncio.wait_for(cleanup_started.wait(), timeout=1)
    assert supervisor.request_interrupt(launched.handle) is True
    assert await supervisor.aclose(wait_seconds=0) == 1
    assert cleanup_cancelled.is_set() is False

    release_cleanup.set()
    terminal = await supervisor.wait(launched.handle, timeout_seconds=1)
    assert terminal is not None
    assert terminal.status is ChildStatus.CANCELLED
    assert cleanup_cancelled.is_set() is False
    assert await supervisor.aclose(wait_seconds=1) == 0


@pytest.mark.asyncio
async def test_child_lifecycle_journals_started_before_terminal(tmp_path) -> None:
    journal = JsonlSessionJournal(tmp_path / "journal")
    await journal.create("parent-run", {})
    supervisor = ChildSupervisor(
        invocation_factory=lambda request, _context: _ready_invocation(
            engine=_CompletingEngine(),
            task=request.task,
        )
    )

    result = await supervisor.launch(
        _request(),
        _context(journal=journal),
        background=False,
    )

    child_records = [
        record
        for record in await journal.replay()
        if record.type
        in {JournalRecordType.CHILD_STARTED, JournalRecordType.CHILD_TERMINAL}
    ]
    assert [record.type for record in child_records] == [
        JournalRecordType.CHILD_STARTED,
        JournalRecordType.CHILD_TERMINAL,
    ]
    assert child_records[0].payload["handle"] == result.handle.to_dict()
    assert child_records[0].payload["background"] is False
    assert child_records[1].payload == result.to_dict()
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_invocation_factory_receives_persisted_child_handle(tmp_path) -> None:
    journal = JsonlSessionJournal(tmp_path / "journal")
    await journal.create("parent-run", {})
    observed_handle = None

    def invocation_factory(
        request: ChildLaunchRequest,
        runtime_context: ChildRuntimeContext,
    ) -> Awaitable[ChildInvocation]:
        nonlocal observed_handle
        observed_handle = runtime_context.handle
        return _ready_invocation(engine=_CompletingEngine(), task=request.task)

    supervisor = ChildSupervisor(invocation_factory=invocation_factory)
    result = await supervisor.launch(
        _request(),
        _context(journal=journal),
        background=False,
    )

    assert observed_handle == result.handle
    records = await journal.replay()
    started = next(
        record for record in records if record.type is JournalRecordType.CHILD_STARTED
    )
    assert observed_handle == ChildHandle.from_dict(started.payload["handle"])
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_started_record_failure_never_constructs_child(tmp_path) -> None:
    factory_called = False
    journal = _FailingChildJournal(
        tmp_path / "journal",
        fail_type=JournalRecordType.CHILD_STARTED,
    )
    await journal.create("parent-run", {})

    def invocation_factory(request, _context):
        nonlocal factory_called
        factory_called = True
        return _ready_invocation(engine=_CompletingEngine(), task=request.task)

    supervisor = ChildSupervisor(invocation_factory=invocation_factory)

    with pytest.raises(ChildPersistenceError, match="was not executed"):
        await supervisor.launch(
            _request(),
            _context(journal=journal),
            background=False,
        )

    assert factory_called is False
    assert supervisor.active_count == 0
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_cancelled_committed_start_keeps_budget_and_persists_terminal(
    tmp_path,
) -> None:
    factory_called = False
    journal = _CommittedCancellationJournal(tmp_path / "journal")
    await journal.create("parent-run", {})
    limiter = ChildRunLimiter(max_active_children=1, max_children=1)

    def invocation_factory(request, _context):
        nonlocal factory_called
        factory_called = True
        return _ready_invocation(engine=_CompletingEngine(), task=request.task)

    supervisor = ChildSupervisor(
        invocation_factory=invocation_factory,
        run_limiter=limiter,
    )

    with pytest.raises(JournalAppendCancelled) as cancelled:
        await supervisor.launch(
            _request(),
            _context(journal=journal),
            background=False,
        )

    assert cancelled.value.committed_position is not None
    assert factory_called is False
    assert limiter.children_started == 1
    assert limiter.active_children == 0
    lifecycle = [
        record
        for record in await journal.replay()
        if record.type
        in {JournalRecordType.CHILD_STARTED, JournalRecordType.CHILD_TERMINAL}
    ]
    assert [record.type for record in lifecycle] == [
        JournalRecordType.CHILD_STARTED,
        JournalRecordType.CHILD_TERMINAL,
    ]
    assert lifecycle[-1].payload["status"] == ChildStatus.CANCELLED.value
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_cancelled_unknown_start_never_rolls_back_durable_budget(
    tmp_path,
) -> None:
    factory_called = False
    journal = _UnknownCancellationJournal(tmp_path / "journal")
    await journal.create("parent-run", {})
    limiter = ChildRunLimiter(max_active_children=1, max_children=1)

    def invocation_factory(request, _context):
        nonlocal factory_called
        factory_called = True
        return _ready_invocation(engine=_CompletingEngine(), task=request.task)

    supervisor = ChildSupervisor(
        invocation_factory=invocation_factory,
        run_limiter=limiter,
    )

    with pytest.raises(JournalAppendCancelled) as cancelled:
        await supervisor.launch(
            _request(),
            _context(journal=journal),
            background=False,
        )

    assert cancelled.value.commit_state is JournalCommitState.UNKNOWN
    assert factory_called is False
    assert limiter.children_started == 1
    assert limiter.active_children == 0
    assert supervisor.active_count == 0
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_unknown_start_commit_error_never_rolls_back_durable_budget(
    tmp_path,
) -> None:
    journal = _CommitErrorJournal(
        tmp_path / "journal",
        fail_type=JournalRecordType.CHILD_STARTED,
        commit_state=JournalCommitState.UNKNOWN,
    )
    await journal.create("parent-run", {})
    limiter = ChildRunLimiter(max_active_children=1, max_children=1)
    supervisor = ChildSupervisor(
        invocation_factory=lambda request, _context: _ready_invocation(
            engine=_CompletingEngine(),
            task=request.task,
        ),
        run_limiter=limiter,
    )

    with pytest.raises(ChildPersistenceError, match="was not executed"):
        await supervisor.launch(
            _request(),
            _context(journal=journal),
            background=False,
        )

    assert limiter.children_started == 1
    assert limiter.active_children == 0
    assert supervisor.active_count == 0
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_committed_terminal_error_preserves_terminal_result(tmp_path) -> None:
    journal = _CommitErrorJournal(
        tmp_path / "journal",
        fail_type=JournalRecordType.CHILD_TERMINAL,
        commit_state=JournalCommitState.COMMITTED,
    )
    await journal.create("parent-run", {})
    supervisor = ChildSupervisor(
        invocation_factory=lambda request, _context: _ready_invocation(
            engine=_CompletingEngine(),
            task=request.task,
        )
    )

    result = await supervisor.launch(
        _request(),
        _context(journal=journal),
        background=False,
    )

    assert result.status is ChildStatus.COMPLETED
    assert result.conclusion.summary == "done:inspect"
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_unknown_terminal_commit_error_reports_unknown_result(tmp_path) -> None:
    journal = _CommitErrorJournal(
        tmp_path / "journal",
        fail_type=JournalRecordType.CHILD_TERMINAL,
        commit_state=JournalCommitState.UNKNOWN,
    )
    await journal.create("parent-run", {})
    supervisor = ChildSupervisor(
        invocation_factory=lambda request, _context: _ready_invocation(
            engine=_CompletingEngine(),
            task=request.task,
        )
    )

    result = await supervisor.launch(
        _request(),
        _context(journal=journal),
        background=False,
    )

    assert result.status is ChildStatus.UNKNOWN
    assert "durable terminal outcome is unknown" in str(result.error)
    assert result.conclusion.summary == "done:inspect"
    assert result.conclusion.unknowns
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_terminal_record_failure_is_visible_and_not_delivered(tmp_path) -> None:
    delivered = False
    journal = _FailingChildJournal(
        tmp_path / "journal",
        fail_type=JournalRecordType.CHILD_TERMINAL,
    )
    await journal.create("parent-run", {})

    async def post_runtime_event(_event: object) -> bool:
        nonlocal delivered
        delivered = True
        return True

    supervisor = ChildSupervisor(
        invocation_factory=lambda request, _context: _ready_invocation(
            engine=_CompletingEngine(),
            task=request.task,
        )
    )
    launched = await supervisor.launch(
        _request(),
        _context(
            journal=journal,
            post_runtime_event=post_runtime_event,
        ),
        background=True,
    )
    terminal = await supervisor.wait(launched.handle, timeout_seconds=1)

    assert terminal is not None
    assert terminal.status is ChildStatus.FAILED
    assert "not persisted" in str(terminal.error)
    assert terminal.conclusion.summary == "done:inspect"
    assert delivered is False
    started = next(
        record
        for record in await journal.replay()
        if record.type is JournalRecordType.CHILD_STARTED
    )
    assert started.payload["background"] is True
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_recovery_terminalizes_started_child_without_replay(tmp_path) -> None:
    journal = JsonlSessionJournal(tmp_path / "journal")
    await journal.create("parent-run", {})
    request = _request()

    def invocation_factory(_request, _context):
        raise AssertionError("recovery replayed the child")

    supervisor = ChildSupervisor(invocation_factory=invocation_factory)

    handle = ChildHandle(child_id="child-interrupted", parent_run_id="parent-run")
    await journal.append(
        JournalRecordType.CHILD_STARTED,
        {"handle": handle.to_dict(), "request": request.to_dict()},
        record_id="parent-run:child:child-interrupted:started",
    )

    recovered = await supervisor.recover(parent_run_id="parent-run", journal=journal)

    assert len(recovered) == 1
    assert recovered[0].status is ChildStatus.INTERRUPTED
    assert recovered[0].handle == handle
    terminal_records = [
        record
        for record in await journal.replay()
        if record.type is JournalRecordType.CHILD_TERMINAL
    ]
    assert len(terminal_records) == 1
    assert ChildStatus(terminal_records[0].payload["status"]) is ChildStatus.INTERRUPTED
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_recovery_rejects_conflicting_child_started_records(tmp_path) -> None:
    journal = JsonlSessionJournal(tmp_path / "journal")
    await journal.create("parent-run", {})
    handle = ChildHandle(child_id="child-conflict", parent_run_id="parent-run")
    for index, request in enumerate((_request("first"), _request("second"))):
        await journal.append(
            JournalRecordType.CHILD_STARTED,
            {"handle": handle.to_dict(), "request": request.to_dict()},
            record_id=f"parent-run:child:conflicting-start:{index}",
        )

    supervisor = ChildSupervisor(
        invocation_factory=lambda _request, _context: _ready_invocation(
            engine=_CompletingEngine(),
            task="unused",
        )
    )

    with pytest.raises(ChildPersistenceError, match="lifecycle journal"):
        await supervisor.recover(parent_run_id="parent-run", journal=journal)

    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_recovery_rejects_terminal_run_id_that_conflicts_with_start(
    tmp_path,
) -> None:
    journal = JsonlSessionJournal(tmp_path / "journal")
    await journal.create("parent-run", {})
    request = _request()
    handle = ChildHandle(child_id="child-lineage", parent_run_id="parent-run")
    await journal.append(
        JournalRecordType.CHILD_STARTED,
        {
            "handle": handle.to_dict(),
            "request": request.to_dict(),
            "child_run_id": "child-run-started",
        },
        record_id="parent-run:child:child-lineage:started",
    )
    await journal.append(
        JournalRecordType.CHILD_TERMINAL,
        ChildResult(
            handle=handle,
            request=request,
            status=ChildStatus.COMPLETED,
            child_run_id="child-run-terminal",
        ).to_dict(),
        record_id="parent-run:child:child-lineage:terminal",
    )
    supervisor = ChildSupervisor(
        invocation_factory=lambda _request, _context: _ready_invocation(
            engine=_CompletingEngine(),
            task="unused",
        )
    )

    with pytest.raises(ChildPersistenceError, match="lifecycle journal"):
        await supervisor.recover(parent_run_id="parent-run", journal=journal)

    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_concurrent_recovery_is_single_flight(tmp_path) -> None:
    replay_entered = asyncio.Event()
    release_replay = asyncio.Event()

    class Journal(JsonlSessionJournal):
        def __init__(self, *args: object) -> None:
            super().__init__(*args)
            self.replay_calls = 0

        async def replay(self):
            self.replay_calls += 1
            replay_entered.set()
            await release_replay.wait()
            return await super().replay()

    journal = Journal(tmp_path / "journal")
    await journal.create("parent-run", {})
    supervisor = ChildSupervisor(
        invocation_factory=lambda _request, _context: _ready_invocation(
            engine=_CompletingEngine(),
            task="unused",
        )
    )

    first = asyncio.create_task(
        supervisor.recover(parent_run_id="parent-run", journal=journal)
    )
    await asyncio.wait_for(replay_entered.wait(), timeout=1)
    second = asyncio.create_task(
        supervisor.recover(parent_run_id="parent-run", journal=journal)
    )
    release_replay.set()
    first_result, second_result = await asyncio.gather(first, second)

    assert first_result == second_result == ()
    assert journal.replay_calls == 1
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_failed_interrupted_terminal_does_not_commit_restored_budget(
    tmp_path,
) -> None:
    journal = _FailingChildJournal(
        tmp_path / "journal",
        fail_type=JournalRecordType.CHILD_TERMINAL,
    )
    await journal.create("parent-run", {})
    request = _request()
    handle = ChildHandle(child_id="child-interrupted", parent_run_id="parent-run")
    await journal.append(
        JournalRecordType.CHILD_STARTED,
        {"handle": handle.to_dict(), "request": request.to_dict()},
        record_id="parent-run:child:child-interrupted:started",
    )
    limiter = ChildRunLimiter(max_active_children=1, max_children=1)
    supervisor = ChildSupervisor(
        invocation_factory=lambda _request, _context: _ready_invocation(
            engine=_CompletingEngine(),
            task="unused",
        ),
        run_limiter=limiter,
    )

    with pytest.raises(ChildPersistenceError, match="interrupted child terminal"):
        await supervisor.recover(parent_run_id="parent-run", journal=journal)

    assert limiter.children_started == 0
    assert supervisor.result(handle) is None
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_recovery_rejects_orphan_and_conflicting_terminal_records(
    tmp_path,
) -> None:
    request = _request()
    handle = ChildHandle(child_id="child-terminal", parent_run_id="parent-run")
    terminal = ChildResult(
        handle=handle,
        request=request,
        status=ChildStatus.COMPLETED,
        conclusion=AgentConclusion(summary="done"),
    )

    orphan_journal = JsonlSessionJournal(tmp_path / "orphan")
    await orphan_journal.create("parent-run", {})
    await orphan_journal.append(
        JournalRecordType.CHILD_TERMINAL,
        terminal.to_dict(),
        record_id="parent-run:child:orphan-terminal",
    )
    orphan_supervisor = ChildSupervisor(
        invocation_factory=lambda _request, _context: _ready_invocation(
            engine=_CompletingEngine(),
            task="unused",
        )
    )

    with pytest.raises(ChildPersistenceError, match="lifecycle journal"):
        await orphan_supervisor.recover(
            parent_run_id="parent-run",
            journal=orphan_journal,
        )

    await orphan_supervisor.aclose()
    await orphan_journal.close()

    conflict_journal = JsonlSessionJournal(tmp_path / "conflict")
    await conflict_journal.create("parent-run", {})
    await conflict_journal.append(
        JournalRecordType.CHILD_STARTED,
        {"handle": handle.to_dict(), "request": request.to_dict()},
        record_id="parent-run:child:started",
    )
    await conflict_journal.append(
        JournalRecordType.CHILD_TERMINAL,
        terminal.to_dict(),
        record_id="parent-run:child:terminal:first",
    )
    await conflict_journal.append(
        JournalRecordType.CHILD_TERMINAL,
        ChildResult(
            handle=handle,
            request=request,
            status=ChildStatus.FAILED,
            error="different",
        ).to_dict(),
        record_id="parent-run:child:terminal:second",
    )
    conflict_supervisor = ChildSupervisor(
        invocation_factory=lambda _request, _context: _ready_invocation(
            engine=_CompletingEngine(),
            task="unused",
        )
    )

    with pytest.raises(ChildPersistenceError, match="lifecycle journal"):
        await conflict_supervisor.recover(
            parent_run_id="parent-run",
            journal=conflict_journal,
        )

    await conflict_supervisor.aclose()
    await conflict_journal.close()


@pytest.mark.asyncio
async def test_fork_does_not_inherit_authority_over_parent_child(tmp_path) -> None:
    parent = JsonlSessionJournal(tmp_path / "journal")
    await parent.create("parent-run", {})
    request = _request()
    handle = ChildHandle(child_id="child-parent", parent_run_id="parent-run")
    await parent.append(
        JournalRecordType.CHILD_STARTED,
        {"handle": handle.to_dict(), "request": request.to_dict()},
        record_id="parent-run:child:child-parent:started",
    )
    position = await parent.append(
        JournalRecordType.STEP_COMMITTED,
        {
            "step_id": 0,
            "consumed_terminal_ids": [],
            "state_delta": [],
            "before_digest": "same",
            "after_digest": "same",
        },
        record_id="parent-run:step:0",
    )
    child_journal = await parent.fork(position, "fork-run")

    def invocation_factory(_request, _context):
        raise AssertionError("fork recovery replayed the parent child")

    supervisor = ChildSupervisor(invocation_factory=invocation_factory)

    recovered = await supervisor.recover(
        parent_run_id="fork-run",
        journal=child_journal,
    )

    assert recovered == ()
    assert supervisor.result(handle) is None
    assert not any(
        record.type is JournalRecordType.CHILD_TERMINAL
        for record in await child_journal.replay()
    )
    await supervisor.aclose()
    await child_journal.close()
    await parent.close()
