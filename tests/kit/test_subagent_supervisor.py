"""Structured-concurrency tests for the Run-owned subagent supervisor."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import replace
import time
from types import SimpleNamespace
from typing import Any

import pytest

from qitos.core.subagent import (
    AgentConclusion,
    SubagentHandle,
    SubagentInvocation,
    SubagentInvocationCancelled,
    SubagentLaunchContext,
    SubagentLaunchRequest,
    SubagentPostRuntimeEvent,
    SubagentPersistenceError,
    SubagentResult,
    SubagentRuntimeContext,
    SubagentStatus,
)
from qitos.core.journal import (
    JournalAppendCancelled,
    JournalCommitError,
    JournalCommitState,
    JournalError,
    JournalPosition,
    JournalRecordRef,
    JournalRecordType,
    SessionJournal,
)
from qitos.core.plan import Plan, PlanContractError, PlanNode, PlanStatus
from qitos.core.task import Task
from qitos.kit.subagent import SubagentRunLimiter, SubagentSupervisor
from qitos.kit.journal import JsonlSessionJournal, recover_session
from qitos.kit.journal.turn_recorder import encode_plan_updated, encode_task_created


def _request(task: str = "inspect") -> SubagentLaunchRequest:
    return SubagentLaunchRequest(task=task, description=f"{task} task")


def _context(
    *,
    journal: SessionJournal | None = None,
    post_runtime_event: SubagentPostRuntimeEvent | None = None,
    deadline_monotonic: float | None = None,
) -> SubagentLaunchContext:
    return SubagentLaunchContext(
        parent_run_id="parent-run",
        journal=journal,
        post_runtime_event=post_runtime_event,
        deadline_monotonic=deadline_monotonic,
    )


async def _set_event(event: asyncio.Event) -> None:
    event.set()


async def _ready_invocation(**kwargs: Any) -> SubagentInvocation:
    return SubagentInvocation(**kwargs)


async def _seed_parent_plan(
    journal: SessionJournal,
    node_ids: tuple[str, ...] = ("delegate",),
) -> None:
    await journal.append(
        JournalRecordType.TASK_CREATED,
        encode_task_created(Task(task_id="parent-task", objective="Parent work")),
        record_id="parent-run:task:parent-task:created",
    )
    await journal.append(
        JournalRecordType.PLAN_UPDATED,
        encode_plan_updated(
            "parent-task",
            Plan(
                tuple(
                    PlanNode(node_id, f"Delegate {node_id}")
                    for node_id in node_ids
                )
            )
        ),
        record_id="parent-run:plan:initial",
    )


class _ClosableEngine:
    async def aclose(self) -> None:
        return None


class _CompletingEngine(_ClosableEngine):
    active_run_id = "subagent-run"

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


class _FailingSubagentJournal(JsonlSessionJournal):
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
        if record_type is JournalRecordType.SUBAGENT_STARTED:
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
        if record_type is JournalRecordType.SUBAGENT_STARTED:
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
        raise AssertionError("the Subagent deadline path requires Python 3.10 support")

    monkeypatch.setattr(asyncio, "timeout_at", unsupported_timeout_at, raising=False)

    async def build(_request, _context):
        started.set()
        await never.wait()
        raise AssertionError("unreachable")

    supervisor = SubagentSupervisor(invocation_factory=build)
    result = await supervisor.launch(
        _request(),
        _context(deadline_monotonic=time.monotonic() + 0.02),
        background=False,
    )

    assert started.is_set()
    assert result.status is SubagentStatus.BUDGET_EXHAUSTED
    assert result.ready is True
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_parent_deadline_drains_engine_and_invocation_cleanup() -> None:
    started = asyncio.Event()
    engine_cancelled = asyncio.Event()
    engine_settled = asyncio.Event()
    invocation_cleaned = asyncio.Event()

    class BlockingEngine(_ClosableEngine):
        active_run_id = "subagent-run"

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

    supervisor = SubagentSupervisor(
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
    assert result.status is SubagentStatus.BUDGET_EXHAUSTED
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
        active_run_id = "subagent-run"

        async def arun(self, task: str, **kwargs: object) -> object:
            _ = task, kwargs
            started.set()
            await release.wait()
            return SimpleNamespace(
                state=SimpleNamespace(final_result="done", stop_reason="completed"),
                records=[],
                step_count=1,
                total_tokens=0,
                run_id="subagent-run",
            )

        def cancel(self, mode: str) -> None:
            _ = mode
            release.set()

    async def build(request, _context):
        nonlocal factory_calls
        factory_calls += 1
        return SubagentInvocation(engine=BlockingEngine(), task=request.task)

    supervisor = SubagentSupervisor(invocation_factory=build, max_concurrency=1)
    first = await supervisor.launch(_request("first"), _context(), background=True)
    await asyncio.wait_for(started.wait(), timeout=1)

    second = await supervisor.launch(
        _request("second"),
        _context(deadline_monotonic=time.monotonic() + 0.02),
        background=False,
    )

    assert first.status is SubagentStatus.RUNNING
    assert second.status is SubagentStatus.BUDGET_EXHAUSTED
    assert factory_calls == 1
    release.set()
    terminal = await supervisor.wait(first.handle, timeout_seconds=1)
    assert terminal is not None and terminal.status is SubagentStatus.COMPLETED
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_factory_timeout_fault_is_not_misreported_as_parent_deadline() -> None:
    async def build(_request, _context):
        raise TimeoutError("factory fault")

    supervisor = SubagentSupervisor(invocation_factory=build)
    result = await supervisor.launch(
        _request(),
        _context(deadline_monotonic=time.monotonic() + 1.0),
        background=False,
    )

    assert result.status is SubagentStatus.FAILED
    assert result.error == "factory fault"
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_foreground_subagents_share_supervisor_concurrency_limit() -> None:
    started: asyncio.Queue[str] = asyncio.Queue()
    release_first = asyncio.Event()
    active = 0
    peak = 0

    class Engine(_ClosableEngine):
        active_run_id = "subagent-run"

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

    supervisor = SubagentSupervisor(
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

    assert first_result.status is SubagentStatus.COMPLETED
    assert second_result.status is SubagentStatus.COMPLETED
    assert await asyncio.wait_for(started.get(), timeout=1) == "two"
    assert peak == 1


@pytest.mark.asyncio
async def test_foreground_subagent_local_cancellation_does_not_cancel_parent() -> None:
    async def cancelled_factory(
        request: SubagentLaunchRequest,
        _context: SubagentRuntimeContext,
    ) -> SubagentInvocation:
        _ = request
        raise SubagentInvocationCancelled("Subagent construction cleanup was cancelled")

    supervisor = SubagentSupervisor(invocation_factory=cancelled_factory)

    result = await supervisor.launch(
        _request(),
        _context(),
        background=False,
    )

    assert result.status is SubagentStatus.CANCELLED
    assert result.error == "Subagent construction cleanup was cancelled"
    assert result.subagent_run_id
    assert await supervisor.aclose() == 0


@pytest.mark.asyncio
async def test_foreground_subagent_preserves_real_caller_cancellation() -> None:
    factory_started = asyncio.Event()
    factory_settled = asyncio.Event()

    async def waiting_factory(
        request: SubagentLaunchRequest,
        _context: SubagentRuntimeContext,
    ) -> SubagentInvocation:
        _ = request
        factory_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            factory_settled.set()

    supervisor = SubagentSupervisor(invocation_factory=waiting_factory)
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
async def test_wait_timeout_does_not_cancel_subagent() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class Engine(_ClosableEngine):
        active_run_id = "subagent-run"

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

    supervisor = SubagentSupervisor(
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
    assert waiting.status is SubagentStatus.RUNNING
    assert supervisor.active_count == 1

    release.set()
    terminal = await supervisor.wait(launched.handle, timeout_seconds=1)
    assert terminal is not None
    assert terminal.status is SubagentStatus.COMPLETED
    assert await supervisor.aclose(wait_seconds=1) == 0


@pytest.mark.asyncio
async def test_terminal_delivery_remains_owned_until_close() -> None:
    delivery_started = asyncio.Event()
    delivery_cancelled = asyncio.Event()

    class Engine(_ClosableEngine):
        active_run_id = "subagent-run"

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

    supervisor = SubagentSupervisor(
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
    assert terminal.status is SubagentStatus.COMPLETED
    assert supervisor.active_count == 0
    assert await supervisor.aclose(wait_seconds=1) == 0
    assert delivery_cancelled.is_set()
    assert supervisor.result(launched.handle) == terminal


@pytest.mark.asyncio
async def test_interrupt_waits_for_started_subagent_cleanup() -> None:
    started = asyncio.Event()
    cleaned = asyncio.Event()
    resource_closed = asyncio.Event()

    class Engine(_ClosableEngine):
        active_run_id = "subagent-run"

        async def arun(self, task: str, **kwargs: object) -> object:
            _ = task, kwargs
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        def cancel(self, mode: str) -> None:
            assert mode == "immediate"

    supervisor = SubagentSupervisor(
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
    assert terminal.status is SubagentStatus.CANCELLED
    assert cleaned.is_set()
    assert resource_closed.is_set()
    assert supervisor.active_count == 0
    assert await supervisor.aclose(wait_seconds=0) == 0


@pytest.mark.asyncio
async def test_invocation_cleanup_failure_is_a_terminal_subagent_failure() -> None:
    async def fail_cleanup() -> None:
        raise RuntimeError("cleanup failed")

    supervisor = SubagentSupervisor(
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

    assert result.status is SubagentStatus.FAILED
    assert result.error == "cleanup failed"
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_invocation_factory_can_finish_async_resource_construction() -> None:
    factory_finished = asyncio.Event()

    async def invocation_factory(
        request: SubagentLaunchRequest,
        _context: SubagentRuntimeContext,
    ) -> SubagentInvocation:
        await asyncio.sleep(0)
        factory_finished.set()
        return SubagentInvocation(engine=_CompletingEngine(), task=request.task)

    supervisor = SubagentSupervisor(invocation_factory=invocation_factory)

    result = await supervisor.launch(
        _request(),
        _context(),
        background=False,
    )

    assert factory_finished.is_set()
    assert result.status is SubagentStatus.COMPLETED
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
    supervisor = SubagentSupervisor(
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

    assert result.status is SubagentStatus.COMPLETED
    assert engine.close_calls == 2
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_invocation_cannot_override_durable_subagent_run_id() -> None:
    supervisor = SubagentSupervisor(
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

    assert result.status is SubagentStatus.FAILED
    assert "conflicts with its durable launch" in str(result.error)
    assert result.subagent_run_id
    assert result.subagent_run_id != "conflicting-run"
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_interrupt_cancels_async_invocation_construction() -> None:
    factory_started = asyncio.Event()
    factory_cancelled = asyncio.Event()

    async def invocation_factory(
        request: SubagentLaunchRequest,
        _context: SubagentRuntimeContext,
    ) -> SubagentInvocation:
        _ = request
        factory_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            factory_cancelled.set()
            raise
        raise AssertionError("unreachable")  # pragma: no cover

    supervisor = SubagentSupervisor(invocation_factory=invocation_factory)
    launched = await supervisor.launch(
        _request(),
        _context(),
        background=True,
    )
    await asyncio.wait_for(factory_started.wait(), timeout=1)

    terminal = await supervisor.interrupt(launched.handle, wait_seconds=1)

    assert terminal is not None
    assert terminal.status is SubagentStatus.CANCELLED
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
        request: SubagentLaunchRequest,
        _context: SubagentRuntimeContext,
    ) -> SubagentInvocation:
        factory_started.set()
        try:
            await release_factory.wait()
        except asyncio.CancelledError:
            # A factory that finishes an atomic acquisition despite cancellation
            # still transfers the returned invocation to the supervisor.
            pass
        return SubagentInvocation(
            engine=Engine(),
            task=request.task,
            cleanup=lambda: _set_event(cleanup_called),
        )

    supervisor = SubagentSupervisor(invocation_factory=invocation_factory)
    launched = await supervisor.launch(
        _request(),
        _context(),
        background=True,
    )
    await asyncio.wait_for(factory_started.wait(), timeout=1)

    terminal = await supervisor.interrupt(launched.handle, wait_seconds=1)

    assert terminal is not None
    assert terminal.status is SubagentStatus.CANCELLED
    assert engine_cancelled.is_set()
    assert engine_closed.is_set()
    assert cleanup_called.is_set()
    assert supervisor.active_count == 0
    assert await supervisor.aclose(wait_seconds=0) == 0


@pytest.mark.asyncio
async def test_close_terminalizes_subagent_cancelled_before_task_start() -> None:
    factory_called = False

    def invocation_factory(request, _context):
        nonlocal factory_called
        factory_called = True
        return _ready_invocation(engine=object(), task=request.task)

    supervisor = SubagentSupervisor(invocation_factory=invocation_factory)
    launched = await supervisor.launch(
        _request(),
        _context(),
        background=True,
    )

    assert await supervisor.aclose(wait_seconds=0) == 0
    terminal = supervisor.result(launched.handle)
    assert terminal is not None
    assert terminal.status is SubagentStatus.CANCELLED
    assert factory_called is False


@pytest.mark.asyncio
async def test_immediate_interrupt_persists_subagent_cancelled_before_task_start(
    tmp_path,
) -> None:
    factory_called = False
    journal = JsonlSessionJournal(tmp_path / "journal")
    await journal.create("parent-run", {})

    def invocation_factory(request, _context):
        nonlocal factory_called
        factory_called = True
        return _ready_invocation(engine=object(), task=request.task)

    supervisor = SubagentSupervisor(invocation_factory=invocation_factory)
    launched = await supervisor.launch(
        _request(),
        _context(journal=journal),
        background=True,
    )

    terminal = await supervisor.interrupt(launched.handle, wait_seconds=0)

    assert terminal is not None
    assert terminal.status is SubagentStatus.CANCELLED
    assert supervisor.active_count == 0
    assert factory_called is False
    records = await journal.replay()
    assert [
        record.type
        for record in records
        if record.type
        in {JournalRecordType.SUBAGENT_STARTED, JournalRecordType.SUBAGENT_TERMINAL}
    ] == [JournalRecordType.SUBAGENT_STARTED, JournalRecordType.SUBAGENT_TERMINAL]
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_repeated_interrupt_does_not_cancel_invocation_cleanup() -> None:
    started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_cancelled = asyncio.Event()

    class Engine(_ClosableEngine):
        active_run_id = "subagent-run"

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

    supervisor = SubagentSupervisor(
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
    assert terminal.status is SubagentStatus.CANCELLED
    assert cleanup_cancelled.is_set() is False
    assert await supervisor.aclose(wait_seconds=1) == 0


@pytest.mark.asyncio
async def test_subagent_lifecycle_journals_started_before_terminal(tmp_path) -> None:
    journal = JsonlSessionJournal(tmp_path / "journal")
    await journal.create("parent-run", {})
    supervisor = SubagentSupervisor(
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

    subagent_records = [
        record
        for record in await journal.replay()
        if record.type
        in {JournalRecordType.SUBAGENT_STARTED, JournalRecordType.SUBAGENT_TERMINAL}
    ]
    assert [record.type for record in subagent_records] == [
        JournalRecordType.SUBAGENT_STARTED,
        JournalRecordType.SUBAGENT_TERMINAL,
    ]
    assert subagent_records[0].payload["handle"] == result.handle.to_dict()
    assert subagent_records[0].payload["background"] is False
    assert subagent_records[1].payload == result.to_dict()
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_invocation_factory_receives_persisted_subagent_handle(tmp_path) -> None:
    journal = JsonlSessionJournal(tmp_path / "journal")
    await journal.create("parent-run", {})
    observed_handle = None

    def invocation_factory(
        request: SubagentLaunchRequest,
        runtime_context: SubagentRuntimeContext,
    ) -> Awaitable[SubagentInvocation]:
        nonlocal observed_handle
        observed_handle = runtime_context.handle
        return _ready_invocation(engine=_CompletingEngine(), task=request.task)

    supervisor = SubagentSupervisor(invocation_factory=invocation_factory)
    result = await supervisor.launch(
        _request(),
        _context(journal=journal),
        background=False,
    )

    assert observed_handle == result.handle
    records = await journal.replay()
    started = next(
        record for record in records if record.type is JournalRecordType.SUBAGENT_STARTED
    )
    assert observed_handle == SubagentHandle.from_dict(started.payload["handle"])
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_plan_assignment_commits_before_subagent_started(tmp_path) -> None:
    journal = JsonlSessionJournal(tmp_path / "journal-assignment")
    await journal.create("parent-run", {})
    await _seed_parent_plan(journal)
    request = SubagentLaunchRequest(
        task="inspect",
        description="inspect task",
        parent_task_id="parent-task",
        plan_assignment="delegate",
    )
    supervisor = SubagentSupervisor(
        invocation_factory=lambda request, _context: _ready_invocation(
            engine=_CompletingEngine(), task=request.task
        )
    )

    result = await supervisor.launch(
        request,
        _context(journal=journal),
        background=False,
    )
    records = await journal.replay()
    assignment_index = next(
        index
        for index, record in enumerate(records)
        if record.type is JournalRecordType.PLAN_UPDATED
        and record.record_id.endswith(result.handle.subagent_id)
    )
    started_index = next(
        index
        for index, record in enumerate(records)
        if record.type is JournalRecordType.SUBAGENT_STARTED
    )
    recovered = recover_session(records)

    assert assignment_index < started_index
    assert recovered.plan is not None
    assigned = recovered.plan.node("delegate")
    assert assigned.status is PlanStatus.IN_PROGRESS
    assert assigned.owner == result.handle
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_parent_plan_requires_explicit_subagent_assignment(tmp_path) -> None:
    journal = JsonlSessionJournal(tmp_path / "journal-missing-assignment")
    await journal.create("parent-run", {})
    await _seed_parent_plan(journal)
    factory_called = False

    def invocation_factory(request, _context):
        nonlocal factory_called
        factory_called = True
        return _ready_invocation(engine=_CompletingEngine(), task=request.task)

    supervisor = SubagentSupervisor(invocation_factory=invocation_factory)
    with pytest.raises(PlanContractError, match="requires an explicit"):
        await supervisor.launch(
            SubagentLaunchRequest(
                task="inspect",
                description="inspect task",
                parent_task_id="parent-task",
            ),
            _context(journal=journal),
            background=False,
        )

    assert factory_called is False
    assert all(
        record.type is not JournalRecordType.SUBAGENT_STARTED
        for record in await journal.replay()
    )

    result = await supervisor.launch(
        SubagentLaunchRequest(
            task="inspect",
            description="inspect task",
            parent_task_id="parent-task",
            plan_assignment="delegate",
        ),
        _context(journal=journal),
        background=False,
    )
    assert result.status is SubagentStatus.COMPLETED
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_parent_task_without_plan_allows_unassigned_subagent(tmp_path) -> None:
    journal = JsonlSessionJournal(tmp_path / "journal-no-parent-plan")
    await journal.create("parent-run", {})
    await journal.append(
        JournalRecordType.TASK_CREATED,
        encode_task_created(Task(task_id="parent-task", objective="Parent work")),
        record_id="parent-run:task:parent-task:created",
    )
    supervisor = SubagentSupervisor(
        invocation_factory=lambda request, _context: _ready_invocation(
            engine=_CompletingEngine(), task=request.task
        )
    )

    result = await supervisor.launch(
        SubagentLaunchRequest(
            task="inspect",
            description="inspect task",
            parent_task_id="parent-task",
        ),
        _context(journal=journal),
        background=False,
    )

    assert result.status is SubagentStatus.COMPLETED
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_concurrent_subagent_assignments_preserve_independent_owners(
    tmp_path,
) -> None:
    journal = JsonlSessionJournal(tmp_path / "journal-concurrent-assignment")
    await journal.create("parent-run", {})
    await _seed_parent_plan(journal, ("first", "second"))
    supervisor = SubagentSupervisor(
        invocation_factory=lambda request, _context: _ready_invocation(
            engine=_CompletingEngine(), task=request.task
        )
    )

    results = await asyncio.gather(
        *(
            supervisor.launch(
                SubagentLaunchRequest(
                    task=node_id,
                    description=f"{node_id} task",
                    parent_task_id="parent-task",
                    plan_assignment=node_id,
                ),
                _context(journal=journal),
                background=True,
            )
            for node_id in ("first", "second")
        )
    )
    recovered = recover_session(await journal.replay())

    assert recovered.plan is not None
    assert {
        recovered.plan.node(node_id).owner for node_id in ("first", "second")
    } == {result.handle for result in results}
    await asyncio.gather(*(supervisor.wait(result.handle) for result in results))
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_unknown_plan_assignment_never_constructs_subagent(tmp_path) -> None:
    journal = JsonlSessionJournal(tmp_path / "journal-invalid-assignment")
    await journal.create("parent-run", {})
    await _seed_parent_plan(journal)
    factory_called = False

    def invocation_factory(request, _context):
        nonlocal factory_called
        factory_called = True
        return _ready_invocation(engine=_CompletingEngine(), task=request.task)

    supervisor = SubagentSupervisor(invocation_factory=invocation_factory)
    with pytest.raises(ValueError, match="Unknown Plan node"):
        await supervisor.launch(
            SubagentLaunchRequest(
                task="inspect",
                description="inspect task",
                parent_task_id="parent-task",
                plan_assignment="missing",
            ),
            _context(journal=journal),
            background=False,
        )

    assert factory_called is False
    assert all(
        record.type is not JournalRecordType.SUBAGENT_STARTED
        for record in await journal.replay()
    )
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_subagent_started_failure_releases_plan_assignment(tmp_path) -> None:
    journal = _FailingSubagentJournal(
        tmp_path / "journal-release-assignment",
        fail_type=JournalRecordType.SUBAGENT_STARTED,
    )
    await journal.create("parent-run", {})
    await _seed_parent_plan(journal)
    supervisor = SubagentSupervisor(
        invocation_factory=lambda request, _context: _ready_invocation(
            engine=_CompletingEngine(), task=request.task
        )
    )

    with pytest.raises(SubagentPersistenceError, match="was not executed"):
        await supervisor.launch(
            SubagentLaunchRequest(
                task="inspect",
                description="inspect task",
                parent_task_id="parent-task",
                plan_assignment="delegate",
            ),
            _context(journal=journal),
            background=False,
        )
    recovered = recover_session(await journal.replay())

    assert recovered.plan is not None
    released = recovered.plan.node("delegate")
    assert released.status is PlanStatus.PENDING
    assert released.owner is None
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_started_record_failure_never_constructs_subagent(tmp_path) -> None:
    factory_called = False
    journal = _FailingSubagentJournal(
        tmp_path / "journal",
        fail_type=JournalRecordType.SUBAGENT_STARTED,
    )
    await journal.create("parent-run", {})

    def invocation_factory(request, _context):
        nonlocal factory_called
        factory_called = True
        return _ready_invocation(engine=_CompletingEngine(), task=request.task)

    supervisor = SubagentSupervisor(invocation_factory=invocation_factory)

    with pytest.raises(SubagentPersistenceError, match="was not executed"):
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
    limiter = SubagentRunLimiter(max_active_subagents=1, max_subagents=1)

    def invocation_factory(request, _context):
        nonlocal factory_called
        factory_called = True
        return _ready_invocation(engine=_CompletingEngine(), task=request.task)

    supervisor = SubagentSupervisor(
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
    assert limiter.subagents_started == 1
    assert limiter.active_subagents == 0
    lifecycle = [
        record
        for record in await journal.replay()
        if record.type
        in {JournalRecordType.SUBAGENT_STARTED, JournalRecordType.SUBAGENT_TERMINAL}
    ]
    assert [record.type for record in lifecycle] == [
        JournalRecordType.SUBAGENT_STARTED,
        JournalRecordType.SUBAGENT_TERMINAL,
    ]
    assert lifecycle[-1].payload["status"] == SubagentStatus.CANCELLED.value
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_cancelled_unknown_start_never_rolls_back_durable_budget(
    tmp_path,
) -> None:
    factory_called = False
    journal = _UnknownCancellationJournal(tmp_path / "journal")
    await journal.create("parent-run", {})
    limiter = SubagentRunLimiter(max_active_subagents=1, max_subagents=1)

    def invocation_factory(request, _context):
        nonlocal factory_called
        factory_called = True
        return _ready_invocation(engine=_CompletingEngine(), task=request.task)

    supervisor = SubagentSupervisor(
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
    assert limiter.subagents_started == 1
    assert limiter.active_subagents == 0
    assert supervisor.active_count == 0
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_unknown_start_commit_error_never_rolls_back_durable_budget(
    tmp_path,
) -> None:
    journal = _CommitErrorJournal(
        tmp_path / "journal",
        fail_type=JournalRecordType.SUBAGENT_STARTED,
        commit_state=JournalCommitState.UNKNOWN,
    )
    await journal.create("parent-run", {})
    limiter = SubagentRunLimiter(max_active_subagents=1, max_subagents=1)
    supervisor = SubagentSupervisor(
        invocation_factory=lambda request, _context: _ready_invocation(
            engine=_CompletingEngine(),
            task=request.task,
        ),
        run_limiter=limiter,
    )

    with pytest.raises(SubagentPersistenceError, match="was not executed"):
        await supervisor.launch(
            _request(),
            _context(journal=journal),
            background=False,
        )

    assert limiter.subagents_started == 1
    assert limiter.active_subagents == 0
    assert supervisor.active_count == 0
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_committed_terminal_error_preserves_terminal_result(tmp_path) -> None:
    journal = _CommitErrorJournal(
        tmp_path / "journal",
        fail_type=JournalRecordType.SUBAGENT_TERMINAL,
        commit_state=JournalCommitState.COMMITTED,
    )
    await journal.create("parent-run", {})
    supervisor = SubagentSupervisor(
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

    assert result.status is SubagentStatus.COMPLETED
    assert result.conclusion.summary == "done:inspect"
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_invocation_projects_typed_conclusion_before_cleanup(tmp_path) -> None:
    journal = JsonlSessionJournal(tmp_path / "journal")
    await journal.create("parent-run", {})
    projected = AgentConclusion(
        summary="committed product conclusion",
        evidence=(JournalRecordRef("subagent-run", "subagent-run:tool:terminal"),),
        next_steps=("Reuse the recorded foothold",),
    )
    cleaned = False

    async def build(request, _context):
        async def conclusion_factory(result):
            assert result.state.final_result == "done:inspect"
            assert cleaned is False
            return projected

        async def cleanup() -> None:
            nonlocal cleaned
            cleaned = True

        return SubagentInvocation(
            engine=_CompletingEngine(),
            task=request.task,
            cleanup=cleanup,
            conclusion_factory=conclusion_factory,
        )

    supervisor = SubagentSupervisor(invocation_factory=build)
    result = await supervisor.launch(
        _request(),
        _context(journal=journal),
        background=False,
    )

    assert cleaned is True
    expected = replace(projected, summary="done:inspect")
    assert result.conclusion == expected
    terminal = next(
        record
        for record in await journal.replay()
        if record.type is JournalRecordType.SUBAGENT_TERMINAL
    )
    assert SubagentResult.from_dict(terminal.payload).conclusion == expected
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_typed_projection_cannot_replace_a_missing_model_conclusion() -> None:
    class EmptyConclusionEngine(_CompletingEngine):
        async def arun(self, task: str, **kwargs: object) -> object:
            result = await super().arun(task, **kwargs)
            result.state.final_result = ""
            return result

    async def build(request, _context):
        async def conclusion_factory(_result):
            return AgentConclusion(
                summary="fabricated summary",
                next_steps=("Keep the typed projection",),
            )

        return SubagentInvocation(
            engine=EmptyConclusionEngine(),
            task=request.task,
            conclusion_factory=conclusion_factory,
        )

    supervisor = SubagentSupervisor(invocation_factory=build)

    result = await supervisor.launch(_request(), _context(), background=False)

    assert result.status is SubagentStatus.FAILED
    assert result.conclusion.summary == ""
    assert result.conclusion.next_steps == ("Keep the typed projection",)
    assert result.error is not None
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_unknown_terminal_commit_error_reports_unknown_result(tmp_path) -> None:
    journal = _CommitErrorJournal(
        tmp_path / "journal",
        fail_type=JournalRecordType.SUBAGENT_TERMINAL,
        commit_state=JournalCommitState.UNKNOWN,
    )
    await journal.create("parent-run", {})
    supervisor = SubagentSupervisor(
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

    assert result.status is SubagentStatus.UNKNOWN
    assert "durable terminal outcome is unknown" in str(result.error)
    assert result.conclusion.summary == "done:inspect"
    assert result.conclusion.unknowns
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_terminal_record_failure_is_visible_and_not_delivered(tmp_path) -> None:
    delivered = False
    journal = _FailingSubagentJournal(
        tmp_path / "journal",
        fail_type=JournalRecordType.SUBAGENT_TERMINAL,
    )
    await journal.create("parent-run", {})

    async def post_runtime_event(_event: object) -> bool:
        nonlocal delivered
        delivered = True
        return True

    supervisor = SubagentSupervisor(
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
    assert terminal.status is SubagentStatus.FAILED
    assert "not persisted" in str(terminal.error)
    assert terminal.conclusion.summary == "done:inspect"
    assert delivered is False
    started = next(
        record
        for record in await journal.replay()
        if record.type is JournalRecordType.SUBAGENT_STARTED
    )
    assert started.payload["background"] is True
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_recovery_terminalizes_started_subagent_without_replay(tmp_path) -> None:
    journal = JsonlSessionJournal(tmp_path / "journal")
    await journal.create("parent-run", {})
    request = _request()

    def invocation_factory(_request, _context):
        raise AssertionError("recovery replayed the subagent")

    supervisor = SubagentSupervisor(invocation_factory=invocation_factory)

    handle = SubagentHandle(subagent_id="subagent-interrupted", parent_run_id="parent-run")
    await journal.append(
        JournalRecordType.SUBAGENT_STARTED,
        {"handle": handle.to_dict(), "request": request.to_dict()},
        record_id="parent-run:subagent:subagent-interrupted:started",
    )

    recovered = await supervisor.recover(parent_run_id="parent-run", journal=journal)

    assert len(recovered) == 1
    assert recovered[0].status is SubagentStatus.INTERRUPTED
    assert recovered[0].handle == handle
    terminal_records = [
        record
        for record in await journal.replay()
        if record.type is JournalRecordType.SUBAGENT_TERMINAL
    ]
    assert len(terminal_records) == 1
    assert SubagentStatus(terminal_records[0].payload["status"]) is SubagentStatus.INTERRUPTED
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_recovery_rejects_conflicting_subagent_started_records(tmp_path) -> None:
    journal = JsonlSessionJournal(tmp_path / "journal")
    await journal.create("parent-run", {})
    handle = SubagentHandle(subagent_id="subagent-conflict", parent_run_id="parent-run")
    for index, request in enumerate((_request("first"), _request("second"))):
        await journal.append(
            JournalRecordType.SUBAGENT_STARTED,
            {"handle": handle.to_dict(), "request": request.to_dict()},
            record_id=f"parent-run:subagent:conflicting-start:{index}",
        )

    supervisor = SubagentSupervisor(
        invocation_factory=lambda _request, _context: _ready_invocation(
            engine=_CompletingEngine(),
            task="unused",
        )
    )

    with pytest.raises(SubagentPersistenceError, match="lifecycle journal"):
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
    handle = SubagentHandle(subagent_id="subagent-lineage", parent_run_id="parent-run")
    await journal.append(
        JournalRecordType.SUBAGENT_STARTED,
        {
            "handle": handle.to_dict(),
            "request": request.to_dict(),
            "subagent_run_id": "subagent-run-started",
        },
        record_id="parent-run:subagent:subagent-lineage:started",
    )
    await journal.append(
        JournalRecordType.SUBAGENT_TERMINAL,
        SubagentResult(
            handle=handle,
            request=request,
            status=SubagentStatus.COMPLETED,
            subagent_run_id="subagent-run-terminal",
        ).to_dict(),
        record_id="parent-run:subagent:subagent-lineage:terminal",
    )
    supervisor = SubagentSupervisor(
        invocation_factory=lambda _request, _context: _ready_invocation(
            engine=_CompletingEngine(),
            task="unused",
        )
    )

    with pytest.raises(SubagentPersistenceError, match="lifecycle journal"):
        await supervisor.recover(parent_run_id="parent-run", journal=journal)

    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_recovery_decodes_legacy_child_lifecycle_records(tmp_path) -> None:
    journal = JsonlSessionJournal(tmp_path / "journal")
    await journal.create("parent-run", {})
    request = _request()
    request_payload = request.to_dict()
    request_payload["budget"]["max_children"] = request_payload["budget"].pop(
        "max_subagents"
    )
    handle_payload = {"child_id": "child-legacy", "parent_run_id": "parent-run"}
    await journal.append(
        JournalRecordType.SUBAGENT_STARTED,
        {
            "handle": handle_payload,
            "request": request_payload,
            "background": True,
            "child_run_id": "run-child-legacy",
        },
        record_id="parent-run:child:child-legacy:started",
    )
    result_payload = SubagentResult(
        handle=SubagentHandle("child-legacy", "parent-run"),
        request=request,
        status=SubagentStatus.COMPLETED,
        conclusion=AgentConclusion(summary="legacy conclusion"),
        subagent_run_id="run-child-legacy",
    ).to_dict()
    result_payload["handle"]["child_id"] = result_payload["handle"].pop(
        "subagent_id"
    )
    result_payload["request"] = request_payload
    result_payload["child_run_id"] = result_payload.pop("subagent_run_id")
    await journal.append(
        JournalRecordType.SUBAGENT_TERMINAL,
        result_payload,
        record_id="parent-run:child:child-legacy:terminal",
    )
    supervisor = SubagentSupervisor(
        invocation_factory=lambda _request, _context: _ready_invocation(
            engine=_CompletingEngine(),
            task="unused",
        )
    )

    recovered = await supervisor.recover(
        parent_run_id="parent-run",
        journal=journal,
    )

    assert len(recovered) == 1
    assert recovered[0].handle.subagent_id == "child-legacy"
    assert recovered[0].subagent_run_id == "run-child-legacy"
    assert recovered[0].conclusion.summary == "legacy conclusion"
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
    supervisor = SubagentSupervisor(
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
    journal = _FailingSubagentJournal(
        tmp_path / "journal",
        fail_type=JournalRecordType.SUBAGENT_TERMINAL,
    )
    await journal.create("parent-run", {})
    request = _request()
    handle = SubagentHandle(subagent_id="subagent-interrupted", parent_run_id="parent-run")
    await journal.append(
        JournalRecordType.SUBAGENT_STARTED,
        {"handle": handle.to_dict(), "request": request.to_dict()},
        record_id="parent-run:subagent:subagent-interrupted:started",
    )
    limiter = SubagentRunLimiter(max_active_subagents=1, max_subagents=1)
    supervisor = SubagentSupervisor(
        invocation_factory=lambda _request, _context: _ready_invocation(
            engine=_CompletingEngine(),
            task="unused",
        ),
        run_limiter=limiter,
    )

    with pytest.raises(SubagentPersistenceError, match="interrupted subagent terminal"):
        await supervisor.recover(parent_run_id="parent-run", journal=journal)

    assert limiter.subagents_started == 0
    assert supervisor.result(handle) is None
    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_recovery_rejects_orphan_and_conflicting_terminal_records(
    tmp_path,
) -> None:
    request = _request()
    handle = SubagentHandle(subagent_id="subagent-terminal", parent_run_id="parent-run")
    terminal = SubagentResult(
        handle=handle,
        request=request,
        status=SubagentStatus.COMPLETED,
        conclusion=AgentConclusion(summary="done"),
    )

    orphan_journal = JsonlSessionJournal(tmp_path / "orphan")
    await orphan_journal.create("parent-run", {})
    await orphan_journal.append(
        JournalRecordType.SUBAGENT_TERMINAL,
        terminal.to_dict(),
        record_id="parent-run:subagent:orphan-terminal",
    )
    orphan_supervisor = SubagentSupervisor(
        invocation_factory=lambda _request, _context: _ready_invocation(
            engine=_CompletingEngine(),
            task="unused",
        )
    )

    with pytest.raises(SubagentPersistenceError, match="lifecycle journal"):
        await orphan_supervisor.recover(
            parent_run_id="parent-run",
            journal=orphan_journal,
        )

    await orphan_supervisor.aclose()
    await orphan_journal.close()

    conflict_journal = JsonlSessionJournal(tmp_path / "conflict")
    await conflict_journal.create("parent-run", {})
    await conflict_journal.append(
        JournalRecordType.SUBAGENT_STARTED,
        {"handle": handle.to_dict(), "request": request.to_dict()},
        record_id="parent-run:subagent:started",
    )
    await conflict_journal.append(
        JournalRecordType.SUBAGENT_TERMINAL,
        terminal.to_dict(),
        record_id="parent-run:subagent:terminal:first",
    )
    await conflict_journal.append(
        JournalRecordType.SUBAGENT_TERMINAL,
        SubagentResult(
            handle=handle,
            request=request,
            status=SubagentStatus.FAILED,
            error="different",
        ).to_dict(),
        record_id="parent-run:subagent:terminal:second",
    )
    conflict_supervisor = SubagentSupervisor(
        invocation_factory=lambda _request, _context: _ready_invocation(
            engine=_CompletingEngine(),
            task="unused",
        )
    )

    with pytest.raises(SubagentPersistenceError, match="lifecycle journal"):
        await conflict_supervisor.recover(
            parent_run_id="parent-run",
            journal=conflict_journal,
        )

    await conflict_supervisor.aclose()
    await conflict_journal.close()


@pytest.mark.asyncio
async def test_fork_does_not_inherit_authority_over_parent_subagent(tmp_path) -> None:
    parent = JsonlSessionJournal(tmp_path / "journal")
    await parent.create("parent-run", {})
    request = _request()
    handle = SubagentHandle(subagent_id="subagent-parent", parent_run_id="parent-run")
    await parent.append(
        JournalRecordType.SUBAGENT_STARTED,
        {"handle": handle.to_dict(), "request": request.to_dict()},
        record_id="parent-run:subagent:subagent-parent:started",
    )
    position = await parent.append(
        JournalRecordType.STEP_COMMITTED,
        {
            "turn": 0,
            "transcript_record_ids": [],
            "tool_terminal_record_ids": [],
        },
        record_id="parent-run:turn:0:committed",
    )
    subagent_journal = await parent.fork(position, "fork-run")

    def invocation_factory(_request, _context):
        raise AssertionError("fork recovery replayed the parent subagent")

    supervisor = SubagentSupervisor(invocation_factory=invocation_factory)

    recovered = await supervisor.recover(
        parent_run_id="fork-run",
        journal=subagent_journal,
    )

    assert recovered == ()
    assert supervisor.result(handle) is None
    assert not any(
        record.type is JournalRecordType.SUBAGENT_TERMINAL
        for record in await subagent_journal.replay()
    )
    await supervisor.aclose()
    await subagent_journal.close()
    await parent.close()
