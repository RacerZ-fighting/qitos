"""Behavior tests for one root Run's recursive Subagent admission limits."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from qitos.core.subagent import (
    SubagentHandle,
    SubagentInvocation,
    SubagentLaunchContext,
    SubagentLaunchRequest,
    SubagentPostRuntimeEvent,
    SubagentPersistenceError,
    SubagentResult,
    SubagentRunLimitError,
    SubagentRuntimeContext,
    SubagentStatus,
)
from qitos.core.journal import JournalRecordType, SessionJournal
from qitos.kit.subagent import SubagentRunLimiter, SubagentSupervisor
from qitos.kit.journal import JsonlSessionJournal
from qitos.kit.tool.subagent import SubagentTool


def _request(task: str) -> SubagentLaunchRequest:
    return SubagentLaunchRequest(task=task, description=f"Run {task}")


def _context(
    parent_run_id: str,
    *,
    journal: SessionJournal | None = None,
    post_runtime_event: SubagentPostRuntimeEvent | None = None,
) -> SubagentLaunchContext:
    return SubagentLaunchContext(
        parent_run_id=parent_run_id,
        journal=journal,
        post_runtime_event=post_runtime_event,
    )


async def _ready_invocation(**kwargs: Any) -> SubagentInvocation:
    return SubagentInvocation(**kwargs)


class _Engine:
    active_run_id = "subagent-run"

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
    limiter: SubagentRunLimiter,
    engine_factory: Any,
) -> SubagentSupervisor:
    async def build(
        request: SubagentLaunchRequest,
        _context: SubagentRuntimeContext,
    ) -> SubagentInvocation:
        return SubagentInvocation(engine=engine_factory(), task=request.task)

    return SubagentSupervisor(invocation_factory=build, run_limiter=limiter)


@pytest.mark.asyncio
async def test_recursive_supervisors_share_cumulative_subagent_budget() -> None:
    limiter = SubagentRunLimiter(max_active_subagents=2, max_subagents=2)
    first = _supervisor(limiter, _Engine)
    nested = _supervisor(limiter, _Engine)

    one = await first.launch(
        _request("one"),
        _context("root-run"),
        background=False,
    )
    two = await nested.launch(
        _request("two"),
        _context("subagent-run"),
        background=False,
    )

    assert one.status is SubagentStatus.COMPLETED
    assert two.status is SubagentStatus.COMPLETED
    assert limiter.subagents_started == 2
    assert limiter.active_subagents == 0
    with pytest.raises(SubagentRunLimitError, match="max_subagents=2"):
        await first.launch(
            _request("three"),
            _context("root-run"),
            background=False,
        )

    await first.aclose()
    await nested.aclose()


@pytest.mark.asyncio
async def test_terminal_subagents_release_default_active_slots_for_later_launches() -> (
    None
):
    started = [asyncio.Event() for _ in range(4)]
    releases = [asyncio.Event() for _ in range(4)]
    engine_index = 0

    def engine_factory() -> _Engine:
        nonlocal engine_index
        index = engine_index
        engine_index += 1
        if index < len(started):
            return _Engine(started=started[index], release=releases[index])
        return _Engine()

    limiter = SubagentRunLimiter(max_active_subagents=4)
    supervisor = _supervisor(limiter, engine_factory)
    running = [
        await supervisor.launch(
            _request(f"blocking-{index}"),
            _context("root-run"),
            background=True,
        )
        for index in range(4)
    ]
    await asyncio.gather(
        *(asyncio.wait_for(event.wait(), timeout=1) for event in started)
    )

    with pytest.raises(SubagentRunLimitError, match="max_active_subagents=4"):
        await supervisor.launch(
            _request("fifth while full"),
            _context("root-run"),
            background=False,
        )

    releases[0].set()
    first = await supervisor.wait(running[0].handle, timeout_seconds=1)
    assert first is not None and first.status is SubagentStatus.COMPLETED
    fifth = await supervisor.launch(
        _request("fifth after terminal"),
        _context("root-run"),
        background=False,
    )

    assert fifth.status is SubagentStatus.COMPLETED
    assert limiter.max_subagents is None
    assert limiter.subagents_started == 5
    assert limiter.active_subagents == 3

    for release in releases[1:]:
        release.set()
    await asyncio.gather(
        *(
            supervisor.wait(result.handle, timeout_seconds=1)
            for result in running[1:]
        )
    )
    assert limiter.active_subagents == 0
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_failed_subagent_releases_unlimited_active_slot() -> None:
    attempts = 0

    async def build(
        request: SubagentLaunchRequest,
        _context: SubagentRuntimeContext,
    ) -> SubagentInvocation:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("fixture construction failed")
        return SubagentInvocation(engine=_Engine(), task=request.task)

    limiter = SubagentRunLimiter(max_active_subagents=1)
    supervisor = SubagentSupervisor(invocation_factory=build, run_limiter=limiter)

    failed = await supervisor.launch(
        _request("failed"),
        _context("root-run"),
        background=False,
    )
    replacement = await supervisor.launch(
        _request("replacement"),
        _context("root-run"),
        background=False,
    )

    assert failed.status is SubagentStatus.FAILED
    assert replacement.status is SubagentStatus.COMPLETED
    assert limiter.subagents_started == 2
    assert limiter.active_subagents == 0
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_interrupted_subagent_releases_unlimited_active_slot() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    attempts = 0

    def engine_factory() -> _Engine:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return _Engine(started=started, release=release)
        return _Engine()

    limiter = SubagentRunLimiter(max_active_subagents=1)
    supervisor = _supervisor(limiter, engine_factory)
    running = await supervisor.launch(
        _request("interrupt"),
        _context("root-run"),
        background=True,
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    interrupted = await supervisor.interrupt(running.handle, wait_seconds=1)
    replacement = await supervisor.launch(
        _request("replacement"),
        _context("root-run"),
        background=False,
    )

    assert interrupted is not None
    assert interrupted.status is SubagentStatus.CANCELLED
    assert replacement.status is SubagentStatus.COMPLETED
    assert limiter.subagents_started == 2
    assert limiter.active_subagents == 0
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_nested_supervisor_setup_does_not_reset_shared_run_budget() -> None:
    limiter = SubagentRunLimiter(max_active_subagents=1, max_subagents=1)
    root = _supervisor(limiter, _Engine)
    nested = _supervisor(limiter, _Engine)

    first = await root.launch(
        _request("one"),
        _context("root-run"),
        background=False,
    )
    nested.setup()

    assert first.status is SubagentStatus.COMPLETED
    assert limiter.subagents_started == 1
    with pytest.raises(SubagentRunLimitError, match="max_subagents=1"):
        await nested.launch(
            _request("nested"),
            _context("subagent-run"),
            background=False,
        )

    await root.aclose()
    await nested.aclose()


@pytest.mark.asyncio
async def test_subagent_tool_setup_starts_fresh_root_run_limit_generation() -> None:
    limiter = SubagentRunLimiter(max_active_subagents=1, max_subagents=1)

    async def build(
        request: SubagentLaunchRequest,
        _context: SubagentRuntimeContext,
    ) -> SubagentInvocation:
        return SubagentInvocation(engine=_Engine(), task=request.task)

    tool = SubagentTool(invocation_factory=build, run_limiter=limiter)
    first = await tool.execute(
        {
            "description": "first",
            "prompt": "one",
            "success_criteria": ["Complete one"],
        },
        runtime_context={"run_id": "root-one"},
    )
    await tool.aclose()

    tool.setup()
    second = await tool.execute(
        {
            "description": "second",
            "prompt": "two",
            "success_criteria": ["Complete two"],
        },
        runtime_context={"run_id": "root-two"},
    )

    assert first.output["subagent_status"] == SubagentStatus.COMPLETED.value
    assert second.output["subagent_status"] == SubagentStatus.COMPLETED.value
    assert limiter.subagents_started == 1
    await tool.aclose()


@pytest.mark.asyncio
async def test_nested_subagent_tool_setup_preserves_active_shared_run_limit() -> None:
    limiter = SubagentRunLimiter(max_active_subagents=2, max_subagents=2)
    lease = await limiter.reserve()

    async def build(
        request: SubagentLaunchRequest,
        _context: SubagentRuntimeContext,
    ) -> SubagentInvocation:
        return SubagentInvocation(engine=_Engine(), task=request.task)

    nested = SubagentTool(
        invocation_factory=build,
        run_limiter=limiter,
        owns_run_limiter=False,
    )
    nested.setup()

    assert limiter.active_subagents == 1
    assert limiter.subagents_started == 1
    await lease.rollback()
    await nested.aclose()


@pytest.mark.asyncio
async def test_active_limit_rejects_without_consuming_cumulative_budget() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    limiter = SubagentRunLimiter(max_active_subagents=1, max_subagents=3)
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

    with pytest.raises(SubagentRunLimitError, match="max_active_subagents=1"):
        await nested.launch(
            _request("nested"),
            _context("subagent-run"),
            background=False,
        )
    assert limiter.subagents_started == 1
    assert limiter.active_subagents == 1

    release.set()
    terminal = await first.wait(running.handle, timeout_seconds=1)
    assert terminal is not None and terminal.status is SubagentStatus.COMPLETED
    assert limiter.active_subagents == 0

    admitted = await nested.launch(
        _request("nested"),
        _context("subagent-run"),
        background=False,
    )
    assert admitted.status is SubagentStatus.COMPLETED
    assert limiter.subagents_started == 2
    assert limiter.active_subagents == 0

    await first.aclose()
    await nested.aclose()


@pytest.mark.asyncio
async def test_terminal_subagent_releases_active_slot_before_parent_delivery() -> None:
    delivery_started = asyncio.Event()
    release_delivery = asyncio.Event()
    limiter = SubagentRunLimiter(max_active_subagents=1, max_subagents=2)
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
    assert first_result.status is SubagentStatus.COMPLETED
    assert limiter.active_subagents == 0

    second = await supervisor.launch(
        _request("second"),
        _context("root-run"),
        background=False,
    )
    assert second.status is SubagentStatus.COMPLETED
    assert limiter.subagents_started == 2
    assert limiter.active_subagents == 0

    release_delivery.set()
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_provisional_admission_can_be_rolled_back() -> None:
    limiter = SubagentRunLimiter(max_active_subagents=1, max_subagents=1)
    lease = await limiter.reserve()

    assert limiter.active_subagents == 1
    assert limiter.subagents_started == 1
    with pytest.raises(RuntimeError, match="active subagents"):
        limiter.reset_for_new_run()
    await lease.rollback()
    await lease.rollback()

    assert limiter.active_subagents == 0
    assert limiter.subagents_started == 0
    replacement = await limiter.reserve()
    replacement.commit()
    await replacement.release()
    assert limiter.subagents_started == 1


@pytest.mark.asyncio
async def test_recovery_restores_durable_launches_into_shared_budget(
    tmp_path: Path,
) -> None:
    journal = JsonlSessionJournal(tmp_path / "journal")
    await journal.create("root-run", {})
    original = SubagentSupervisor(
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

    limiter = SubagentRunLimiter(max_active_subagents=1, max_subagents=1)
    recovered = _supervisor(limiter, _Engine)
    results = await recovered.recover(parent_run_id="root-run", journal=journal)

    assert len(results) == 1
    assert limiter.subagents_started == 1
    with pytest.raises(SubagentRunLimitError, match="max_subagents=1"):
        await recovered.launch(
            _request("over-budget"),
            _context("root-run", journal=journal),
            background=False,
        )

    await recovered.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_recovery_retains_unlimited_terminal_launch_history_without_active_slots(
    tmp_path: Path,
) -> None:
    journal = JsonlSessionJournal(tmp_path / "journal")
    await journal.create("root-run", {})
    original = SubagentSupervisor(
        invocation_factory=(
            lambda request, _context: _ready_invocation(
                engine=_Engine(),
                task=request.task,
            )
        )
    )
    for index in range(5):
        result = await original.launch(
            _request(f"persisted-{index}"),
            _context("root-run", journal=journal),
            background=False,
        )
        assert result.status is SubagentStatus.COMPLETED
    await original.aclose()

    limiter = SubagentRunLimiter(max_active_subagents=4)
    recovered = _supervisor(limiter, _Engine)
    results = await recovered.recover(parent_run_id="root-run", journal=journal)

    assert len(results) == 5
    assert limiter.subagents_started == 5
    assert limiter.active_subagents == 0
    later = await recovered.launch(
        _request("later"),
        _context("root-run", journal=journal),
        background=False,
    )
    assert later.status is SubagentStatus.COMPLETED
    assert limiter.subagents_started == 6

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
        request: SubagentLaunchRequest,
        context: SubagentRuntimeContext,
    ) -> SubagentInvocation:
        subagent_run_id = context.subagent_run_id
        observed_run_ids.append(subagent_run_id)
        engine = _Engine()
        engine.active_run_id = subagent_run_id
        return SubagentInvocation(engine=engine, task=request.task)

    supervisor = SubagentSupervisor(invocation_factory=build)
    result = await supervisor.launch(
        _request("persist locator"),
        _context("root-run", journal=journal),
        background=False,
    )

    started = next(
        record
        for record in await journal.replay()
        if record.type is JournalRecordType.SUBAGENT_STARTED
    )
    assert observed_run_ids == [started.payload["subagent_run_id"]]
    assert result.subagent_run_id == started.payload["subagent_run_id"]

    await supervisor.aclose()
    await journal.close()


@pytest.mark.asyncio
async def test_recovery_restores_nested_journal_launches_into_root_budget(
    tmp_path: Path,
) -> None:
    journal_root = tmp_path / "journal"
    root_journal = JsonlSessionJournal(journal_root)
    await root_journal.create("root-run", {})
    subagent_request = _request("subagent")
    subagent_handle = SubagentHandle(
        subagent_id="subagent-direct",
        parent_run_id="root-run",
    )
    await root_journal.append(
        JournalRecordType.SUBAGENT_STARTED,
        {
            "handle": subagent_handle.to_dict(),
            "request": subagent_request.to_dict(),
            "subagent_run_id": "subagent-run",
        },
        record_id="root-run:subagent:subagent-direct:started",
    )
    await root_journal.append(
        JournalRecordType.SUBAGENT_TERMINAL,
        SubagentResult(
            handle=subagent_handle,
            request=subagent_request,
            status=SubagentStatus.COMPLETED,
            subagent_run_id="subagent-run",
        ).to_dict(),
        record_id="root-run:subagent:subagent-direct:terminal",
    )

    subagent_journal = JsonlSessionJournal(journal_root)
    await subagent_journal.create("subagent-run", {})
    grandsubagent_request = _request("grandsubagent")
    grandsubagent_handle = SubagentHandle(
        subagent_id="subagent-nested",
        parent_run_id="subagent-run",
    )
    await subagent_journal.append(
        JournalRecordType.SUBAGENT_STARTED,
        {
            "handle": grandsubagent_handle.to_dict(),
            "request": grandsubagent_request.to_dict(),
            "subagent_run_id": "grandsubagent-run",
        },
        record_id="subagent-run:subagent:subagent-nested:started",
    )
    await subagent_journal.append(
        JournalRecordType.SUBAGENT_TERMINAL,
        SubagentResult(
            handle=grandsubagent_handle,
            request=grandsubagent_request,
            status=SubagentStatus.COMPLETED,
            subagent_run_id="grandsubagent-run",
        ).to_dict(),
        record_id="subagent-run:subagent:subagent-nested:terminal",
    )
    await subagent_journal.close()

    limiter = SubagentRunLimiter(max_active_subagents=2, max_subagents=2)
    recovered = SubagentSupervisor(
        invocation_factory=lambda request, _context: _ready_invocation(
            engine=_Engine(),
            task=request.task,
        ),
        run_limiter=limiter,
        subagent_journal_factory=lambda: JsonlSessionJournal(journal_root),
    )

    await recovered.recover(parent_run_id="root-run", journal=root_journal)

    assert limiter.subagents_started == 2
    assert limiter.active_subagents == 0
    with pytest.raises(SubagentRunLimitError, match="max_subagents=2"):
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
        request = _request(f"subagent-{index}")
        handle = SubagentHandle(
            subagent_id=f"subagent-{index}",
            parent_run_id="root-run",
        )
        await root_journal.append(
            JournalRecordType.SUBAGENT_STARTED,
            {
                "handle": handle.to_dict(),
                "request": request.to_dict(),
                "subagent_run_id": "shared-subagent-run",
            },
            record_id=f"root-run:subagent:subagent-{index}:started",
        )

    supervisor = SubagentSupervisor(
        invocation_factory=lambda request, _context: _ready_invocation(
            engine=_Engine(),
            task=request.task,
        ),
        subagent_journal_factory=lambda: JsonlSessionJournal(journal_root),
    )

    with pytest.raises(SubagentPersistenceError, match="cycle or duplicate Run"):
        await supervisor.recover(
            parent_run_id="root-run",
            journal=root_journal,
        )

    assert supervisor.active_count == 0
    await supervisor.aclose()
    await root_journal.close()


@pytest.mark.asyncio
async def test_recovery_rejects_history_over_configured_subagent_limit(
    tmp_path: Path,
) -> None:
    root_journal = JsonlSessionJournal(tmp_path / "journal")
    await root_journal.create("root-run", {})
    for index in range(2):
        request = _request(f"subagent-{index}")
        handle = SubagentHandle(
            subagent_id=f"subagent-{index}",
            parent_run_id="root-run",
        )
        await root_journal.append(
            JournalRecordType.SUBAGENT_STARTED,
            {"handle": handle.to_dict(), "request": request.to_dict()},
            record_id=f"root-run:subagent:subagent-{index}:started",
        )

    limiter = SubagentRunLimiter(max_active_subagents=1, max_subagents=1)
    supervisor = SubagentSupervisor(
        invocation_factory=lambda request, _context: _ready_invocation(
            engine=_Engine(),
            task=request.task,
        ),
        run_limiter=limiter,
    )

    with pytest.raises(SubagentPersistenceError, match="configured Run limit"):
        await supervisor.recover(
            parent_run_id="root-run",
            journal=root_journal,
        )

    assert limiter.subagents_started == 0
    assert supervisor.active_count == 0
    await supervisor.aclose()
    await root_journal.close()
