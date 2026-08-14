"""Structured-concurrency tests for the Run-owned child supervisor."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from qitos.core.child import (
    ChildHandle,
    ChildInvocation,
    ChildLaunchRequest,
    ChildPersistenceError,
    ChildStatus,
)
from qitos.core.journal import JournalError, JournalRecordType
from qitos.kit.child import ChildSupervisor
from qitos.kit.journal import JsonlSessionJournal


def _request(task: str = "inspect") -> ChildLaunchRequest:
    return ChildLaunchRequest(task=task, description=f"{task} task")


async def _set_event(event: asyncio.Event) -> None:
    event.set()


class _CompletingEngine:
    active_run_id = "child-run"

    async def arun(self, task: str, **kwargs: object) -> object:
        assert kwargs == {}
        return SimpleNamespace(
            state=SimpleNamespace(final_result=f"done:{task}", stop_reason="completed"),
            records=[],
            step_count=1,
            total_tokens=2,
            run_id=self.active_run_id,
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


@pytest.mark.asyncio
async def test_wait_timeout_does_not_cancel_child() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class Engine:
        active_run_id = "child-run"

        async def arun(self, task: str, **kwargs: object) -> object:
            assert task == "inspect"
            assert kwargs == {}
            started.set()
            await release.wait()
            return SimpleNamespace(
                state=SimpleNamespace(final_result="done", stop_reason="completed"),
                records=[],
                step_count=1,
                total_tokens=2,
                run_id=self.active_run_id,
            )

        def cancel(self, mode: str) -> None:
            assert mode == "immediate"
            release.set()

    supervisor = ChildSupervisor(
        invocation_factory=lambda request, _context: ChildInvocation(
            engine=Engine(),
            task=request.task,
        )
    )
    launched = await supervisor.launch(
        _request(),
        {},
        parent_run_id="parent-run",
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

    class Engine:
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
        invocation_factory=lambda request, _context: ChildInvocation(
            engine=Engine(),
            task=request.task,
        )
    )
    launched = await supervisor.launch(
        _request(),
        {"post_runtime_event": post_runtime_event},
        parent_run_id="parent-run",
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

    class Engine:
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
        invocation_factory=lambda request, _context: ChildInvocation(
            engine=Engine(),
            task=request.task,
            cleanup=lambda: _set_event(resource_closed),
        )
    )
    launched = await supervisor.launch(
        _request(),
        {},
        parent_run_id="parent-run",
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
        invocation_factory=lambda request, _context: ChildInvocation(
            engine=_CompletingEngine(),
            task=request.task,
            cleanup=fail_cleanup,
        )
    )

    result = await supervisor.launch(
        _request(),
        {},
        parent_run_id="parent-run",
        background=False,
    )

    assert result.status is ChildStatus.FAILED
    assert result.error == "cleanup failed"
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_invocation_factory_can_finish_async_resource_construction() -> None:
    factory_finished = asyncio.Event()

    async def invocation_factory(request, _context):
        await asyncio.sleep(0)
        factory_finished.set()
        return ChildInvocation(engine=_CompletingEngine(), task=request.task)

    supervisor = ChildSupervisor(invocation_factory=invocation_factory)

    result = await supervisor.launch(
        _request(),
        {},
        parent_run_id="parent-run",
        background=False,
    )

    assert factory_finished.is_set()
    assert result.status is ChildStatus.COMPLETED
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_close_terminalizes_child_cancelled_before_task_start() -> None:
    factory_called = False

    def invocation_factory(request, _context):
        nonlocal factory_called
        factory_called = True
        return ChildInvocation(engine=object(), task=request.task)

    supervisor = ChildSupervisor(invocation_factory=invocation_factory)
    launched = await supervisor.launch(
        _request(),
        {},
        parent_run_id="parent-run",
        background=True,
    )

    assert await supervisor.aclose(wait_seconds=0) == 0
    terminal = supervisor.result(launched.handle)
    assert terminal is not None
    assert terminal.status is ChildStatus.CANCELLED
    assert factory_called is False


@pytest.mark.asyncio
async def test_child_lifecycle_journals_started_before_terminal(tmp_path) -> None:
    journal = JsonlSessionJournal(tmp_path / "journal")
    await journal.create("parent-run", {})
    supervisor = ChildSupervisor(
        invocation_factory=lambda request, _context: ChildInvocation(
            engine=_CompletingEngine(),
            task=request.task,
        )
    )

    result = await supervisor.launch(
        _request(),
        {"journal": journal},
        parent_run_id="parent-run",
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

    def invocation_factory(request, runtime_context):
        nonlocal observed_handle
        observed_handle = runtime_context["child_handle"]
        return ChildInvocation(engine=_CompletingEngine(), task=request.task)

    supervisor = ChildSupervisor(invocation_factory=invocation_factory)
    result = await supervisor.launch(
        _request(),
        {"journal": journal},
        parent_run_id="parent-run",
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
        return ChildInvocation(engine=_CompletingEngine(), task=request.task)

    supervisor = ChildSupervisor(invocation_factory=invocation_factory)

    with pytest.raises(ChildPersistenceError, match="was not executed"):
        await supervisor.launch(
            _request(),
            {"journal": journal},
            parent_run_id="parent-run",
            background=False,
        )

    assert factory_called is False
    assert supervisor.active_count == 0
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
        invocation_factory=lambda request, _context: ChildInvocation(
            engine=_CompletingEngine(),
            task=request.task,
        )
    )
    launched = await supervisor.launch(
        _request(),
        {"journal": journal, "post_runtime_event": post_runtime_event},
        parent_run_id="parent-run",
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
