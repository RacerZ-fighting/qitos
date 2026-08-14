"""Behavior tests for one root Run's recursive Child admission limits."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from qitos.core.child import (
    ChildHandle,
    ChildInvocation,
    ChildLaunchContext,
    ChildLaunchRequest,
    ChildPostRuntimeEvent,
    ChildPersistenceError,
    ChildResult,
    ChildRunLimitError,
    ChildRuntimeContext,
    ChildStatus,
)
from qitos.core.journal import JournalRecordType, SessionJournal
from qitos.kit.child import ChildRunLimiter, ChildSupervisor
from qitos.kit.journal import JsonlSessionJournal


def _request(task: str) -> ChildLaunchRequest:
    return ChildLaunchRequest(task=task, description=f"Run {task}")


def _context(
    parent_run_id: str,
    *,
    journal: SessionJournal | None = None,
    post_runtime_event: ChildPostRuntimeEvent | None = None,
) -> ChildLaunchContext:
    return ChildLaunchContext(
        parent_run_id=parent_run_id,
        journal=journal,
        post_runtime_event=post_runtime_event,
    )


async def _ready_invocation(**kwargs: Any) -> ChildInvocation:
    return ChildInvocation(**kwargs)


class _Engine:
    active_run_id = "child-run"

    def __init__(
        self,
        *,
        started: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
    ) -> None:
        self._started = started
        self._release = release

    async def arun(self, task: str, **kwargs: Any) -> Any:
        run_id = kwargs.pop("run_id")
        assert isinstance(run_id, str)
        assert kwargs == {}
        if self._started is not None:
            self._started.set()
        if self._release is not None:
            await self._release.wait()
        return SimpleNamespace(
            state=SimpleNamespace(
                final_result=f"completed:{task}",
                stop_reason="completed",
            ),
            records=[],
            step_count=1,
            total_tokens=1,
            run_id=run_id,
        )

    async def aclose(self) -> None:
        return None

    def cancel(self, mode: str) -> None:
        assert mode == "immediate"
        if self._release is not None:
            self._release.set()


def _supervisor(
    limiter: ChildRunLimiter,
    engine_factory: Any,
) -> ChildSupervisor:
    async def build(
        request: ChildLaunchRequest,
        _context: ChildRuntimeContext,
    ) -> ChildInvocation:
        return ChildInvocation(engine=engine_factory(), task=request.task)

    return ChildSupervisor(invocation_factory=build, run_limiter=limiter)


@pytest.mark.asyncio
async def test_recursive_supervisors_share_cumulative_child_budget() -> None:
    limiter = ChildRunLimiter(max_active_children=2, max_children=2)
    first = _supervisor(limiter, _Engine)
    nested = _supervisor(limiter, _Engine)

    one = await first.launch(
        _request("one"),
        _context("root-run"),
        background=False,
    )
    two = await nested.launch(
        _request("two"),
        _context("child-run"),
        background=False,
    )

    assert one.status is ChildStatus.COMPLETED
    assert two.status is ChildStatus.COMPLETED
    assert limiter.children_started == 2
    assert limiter.active_children == 0
    with pytest.raises(ChildRunLimitError, match="max_children=2"):
        await first.launch(
            _request("three"),
            _context("root-run"),
            background=False,
        )

    await first.aclose()
    await nested.aclose()


@pytest.mark.asyncio
async def test_active_limit_rejects_without_consuming_cumulative_budget() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    limiter = ChildRunLimiter(max_active_children=1, max_children=3)
    first = _supervisor(
        limiter,
        lambda: _Engine(started=started, release=release),
    )
    nested = _supervisor(limiter, _Engine)
    running = await first.launch(
        _request("blocking"),
        _context("root-run"),
        background=True,
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    with pytest.raises(ChildRunLimitError, match="max_active_children=1"):
        await nested.launch(
            _request("nested"),
            _context("child-run"),
            background=False,
        )
    assert limiter.children_started == 1
    assert limiter.active_children == 1

    release.set()
    terminal = await first.wait(running.handle, timeout_seconds=1)
    assert terminal is not None and terminal.status is ChildStatus.COMPLETED
    assert limiter.active_children == 0

    admitted = await nested.launch(
        _request("nested"),
        _context("child-run"),
        background=False,
    )
    assert admitted.status is ChildStatus.COMPLETED
    assert limiter.children_started == 2
    assert limiter.active_children == 0

    await first.aclose()
    await nested.aclose()


@pytest.mark.asyncio
async def test_terminal_child_releases_active_slot_before_parent_delivery() -> None:
    delivery_started = asyncio.Event()
    release_delivery = asyncio.Event()
    limiter = ChildRunLimiter(max_active_children=1, max_children=2)
    supervisor = _supervisor(limiter, _Engine)

    async def post_runtime_event(_event: object) -> bool:
        delivery_started.set()
        await release_delivery.wait()
        return True

    first = await supervisor.launch(
        _request("first"),
        _context("root-run", post_runtime_event=post_runtime_event),
        background=True,
    )
    await asyncio.wait_for(delivery_started.wait(), timeout=1)

    first_result = supervisor.result(first.handle)
    assert first_result is not None
    assert first_result.status is ChildStatus.COMPLETED
    assert limiter.active_children == 0

    second = await supervisor.launch(
        _request("second"),
        _context("root-run"),
        background=False,
    )
    assert second.status is ChildStatus.COMPLETED
    assert limiter.children_started == 2
    assert limiter.active_children == 0

    release_delivery.set()
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_provisional_admission_can_be_rolled_back() -> None:
    limiter = ChildRunLimiter(max_active_children=1, max_children=1)
    lease = await limiter.reserve()

    assert limiter.active_children == 1
    assert limiter.children_started == 1
    await lease.rollback()
    await lease.rollback()

    assert limiter.active_children == 0
    assert limiter.children_started == 0
    replacement = await limiter.reserve()
    replacement.commit()
    await replacement.release()
    assert limiter.children_started == 1


@pytest.mark.asyncio
async def test_recovery_restores_durable_launches_into_shared_budget(
    tmp_path: Path,
) -> None:
    journal = JsonlSessionJournal(tmp_path / "journal")
    await journal.create("root-run", {})
    original = ChildSupervisor(
        invocation_factory=(
            lambda request, _context: _ready_invocation(
                engine=_Engine(),
                task=request.task,
            )
        )
    )
    await original.launch(
        _request("persisted"),
        _context("root-run", journal=journal),
        background=False,
    )
    await original.aclose()

    limiter = ChildRunLimiter(max_active_children=1, max_children=1)
    recovered = _supervisor(limiter, _Engine)
    results = await recovered.recover(parent_run_id="root-run", journal=journal)

    assert len(results) == 1
    assert limiter.children_started == 1
    with pytest.raises(ChildRunLimitError, match="max_children=1"):
        await recovered.launch(
            _request("over-budget"),
            _context("root-run", journal=journal),
            background=False,
        )

    await recovered.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_launch_persists_run_id_before_invocation_factory(
    tmp_path: Path,
) -> None:
    journal = JsonlSessionJournal(tmp_path / "journal")
    await journal.create("root-run", {})
    observed_run_ids: list[str] = []

    async def build(
        request: ChildLaunchRequest,
        context: ChildRuntimeContext,
    ) -> ChildInvocation:
        child_run_id = context.child_run_id
        observed_run_ids.append(child_run_id)
        engine = _Engine()
        engine.active_run_id = child_run_id
        return ChildInvocation(engine=engine, task=request.task)

    supervisor = ChildSupervisor(invocation_factory=build)
    result = await supervisor.launch(
        _request("persist locator"),
        _context("root-run", journal=journal),
        background=False,
    )

    started = next(
        record
        for record in await journal.replay()
        if record.type is JournalRecordType.CHILD_STARTED
    )
    assert observed_run_ids == [started.payload["child_run_id"]]
    assert result.child_run_id == started.payload["child_run_id"]

    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_recovery_restores_nested_journal_launches_into_root_budget(
    tmp_path: Path,
) -> None:
    journal_root = tmp_path / "journal"
    root_journal = JsonlSessionJournal(journal_root)
    await root_journal.create("root-run", {})
    child_request = _request("child")
    child_handle = ChildHandle(
        child_id="child-direct",
        parent_run_id="root-run",
    )
    await root_journal.append(
        JournalRecordType.CHILD_STARTED,
        {
            "handle": child_handle.to_dict(),
            "request": child_request.to_dict(),
            "child_run_id": "child-run",
        },
        record_id="root-run:child:child-direct:started",
    )
    await root_journal.append(
        JournalRecordType.CHILD_TERMINAL,
        ChildResult(
            handle=child_handle,
            request=child_request,
            status=ChildStatus.COMPLETED,
            child_run_id="child-run",
        ).to_dict(),
        record_id="root-run:child:child-direct:terminal",
    )

    child_journal = JsonlSessionJournal(journal_root)
    await child_journal.create("child-run", {})
    grandchild_request = _request("grandchild")
    grandchild_handle = ChildHandle(
        child_id="child-nested",
        parent_run_id="child-run",
    )
    await child_journal.append(
        JournalRecordType.CHILD_STARTED,
        {
            "handle": grandchild_handle.to_dict(),
            "request": grandchild_request.to_dict(),
            "child_run_id": "grandchild-run",
        },
        record_id="child-run:child:child-nested:started",
    )
    await child_journal.append(
        JournalRecordType.CHILD_TERMINAL,
        ChildResult(
            handle=grandchild_handle,
            request=grandchild_request,
            status=ChildStatus.COMPLETED,
            child_run_id="grandchild-run",
        ).to_dict(),
        record_id="child-run:child:child-nested:terminal",
    )
    await child_journal.close()

    limiter = ChildRunLimiter(max_active_children=2, max_children=2)
    recovered = ChildSupervisor(
        invocation_factory=lambda request, _context: _ready_invocation(
            engine=_Engine(),
            task=request.task,
        ),
        run_limiter=limiter,
        child_journal_factory=lambda: JsonlSessionJournal(journal_root),
    )

    await recovered.recover(parent_run_id="root-run", journal=root_journal)

    assert limiter.children_started == 2
    assert limiter.active_children == 0
    with pytest.raises(ChildRunLimitError, match="max_children=2"):
        await recovered.launch(
            _request("over budget after resume"),
            _context("root-run", journal=root_journal),
            background=False,
        )

    await recovered.aclose()
    await root_journal.close()


@pytest.mark.asyncio
async def test_recovery_rejects_duplicate_descendant_run_identity(
    tmp_path: Path,
) -> None:
    journal_root = tmp_path / "journal"
    root_journal = JsonlSessionJournal(journal_root)
    await root_journal.create("root-run", {})
    for index in range(2):
        request = _request(f"child-{index}")
        handle = ChildHandle(
            child_id=f"child-{index}",
            parent_run_id="root-run",
        )
        await root_journal.append(
            JournalRecordType.CHILD_STARTED,
            {
                "handle": handle.to_dict(),
                "request": request.to_dict(),
                "child_run_id": "shared-child-run",
            },
            record_id=f"root-run:child:child-{index}:started",
        )

    supervisor = ChildSupervisor(
        invocation_factory=lambda request, _context: _ready_invocation(
            engine=_Engine(),
            task=request.task,
        ),
        child_journal_factory=lambda: JsonlSessionJournal(journal_root),
    )

    with pytest.raises(ChildPersistenceError, match="cycle or duplicate Run"):
        await supervisor.recover(
            parent_run_id="root-run",
            journal=root_journal,
        )

    assert supervisor.active_count == 0
    await supervisor.aclose()
    await root_journal.close()


@pytest.mark.asyncio
async def test_recovery_rejects_history_over_configured_child_limit(
    tmp_path: Path,
) -> None:
    root_journal = JsonlSessionJournal(tmp_path / "journal")
    await root_journal.create("root-run", {})
    for index in range(2):
        request = _request(f"child-{index}")
        handle = ChildHandle(
            child_id=f"child-{index}",
            parent_run_id="root-run",
        )
        await root_journal.append(
            JournalRecordType.CHILD_STARTED,
            {"handle": handle.to_dict(), "request": request.to_dict()},
            record_id=f"root-run:child:child-{index}:started",
        )

    limiter = ChildRunLimiter(max_active_children=1, max_children=1)
    supervisor = ChildSupervisor(
        invocation_factory=lambda request, _context: _ready_invocation(
            engine=_Engine(),
            task=request.task,
        ),
        run_limiter=limiter,
    )

    with pytest.raises(ChildPersistenceError, match="configured Run limit"):
        await supervisor.recover(
            parent_run_id="root-run",
            journal=root_journal,
        )

    assert limiter.children_started == 0
    assert supervisor.active_count == 0
    await supervisor.aclose()
    await root_journal.close()
