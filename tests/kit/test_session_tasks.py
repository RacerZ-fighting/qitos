"""SessionRun task lifecycle: start, transitions, follow-up, resume."""

from __future__ import annotations

import pytest

from qitos.core.agent import AgentRunRejected
from qitos.core.agent_loop import AgentRunStatus
from qitos.core.budget import BudgetLedger
from qitos.core.journal import JournalRecordType
from qitos.core.task import Task, TaskBlocker, TaskStatus
from qitos.kit.journal import InMemoryJournalStore, recover_session
from qitos.kit.journal.turn_recorder import decode_task_created
from qitos.kit.session import (
    SessionHarness,
    SessionRun,
    TaskTransitionRejected,
)

from tests.core.agent_fakes import ScriptedModel, failed_events, text_events


def _harness() -> SessionHarness:
    return SessionHarness(InMemoryJournalStore())


def _task(task_id: str, **kwargs: object) -> Task:
    return Task(task_id=task_id, objective=f"{task_id} objective", **kwargs)  # type: ignore[arg-type]


async def _started_run(*turns: str, task: Task | None = None) -> SessionRun:
    harness = _harness()
    model = ScriptedModel([text_events(text) for text in turns])
    return await harness.start(model=model, task=task)


async def _types(session_run: SessionRun) -> list[JournalRecordType]:
    records = await session_run.journal.replay()
    return [record.type for record in records]


@pytest.mark.asyncio
async def test_start_commits_task_before_input() -> None:
    task = _task("root-1")
    session_run = await _started_run("done", task=task)
    records = await session_run.journal.replay()
    created = [
        record
        for record in records
        if record.type is JournalRecordType.TASK_CREATED
    ]
    assert len(created) == 1
    assert decode_task_created(created[0].payload) == task
    assert session_run.task == task
    lifecycle = session_run.task_lifecycle
    assert lifecycle is not None
    assert lifecycle.status is TaskStatus.ACTIVE

    result = await session_run.prompt("begin")
    assert result.status is AgentRunStatus.COMPLETED
    ordering = await _types(session_run)
    assert ordering.index(JournalRecordType.TASK_CREATED) < ordering.index(
        JournalRecordType.INPUT_ACCEPTED
    )
    await session_run.close()


@pytest.mark.asyncio
async def test_run_terminal_does_not_transition_task() -> None:
    session_run = await _started_run("done", task=_task("root-1"))
    result = await session_run.prompt("begin")
    assert result.status is AgentRunStatus.COMPLETED
    lifecycle = session_run.task_lifecycle
    assert lifecycle is not None
    assert lifecycle.status is TaskStatus.ACTIVE
    assert JournalRecordType.TASK_TRANSITION not in await _types(session_run)
    await session_run.close()


@pytest.mark.asyncio
async def test_transition_is_durable_before_return_and_terminal_once() -> None:
    session_run = await _started_run("done", task=_task("root-1"))
    first_run_id = session_run.run_id
    await session_run.prompt("begin")
    lifecycle = await session_run.complete_task("criteria met")
    assert not isinstance(lifecycle, TaskTransitionRejected)
    assert lifecycle.status is TaskStatus.COMPLETED
    assert lifecycle.terminal_reason == "criteria met"
    # The settled leg kept its run terminal last; the transition committed
    # into the advanced leg ahead of its input.
    assert session_run.run_id != first_run_id
    records = await session_run.journal.replay()
    recovered = recover_session(records)
    folded = recovered.tasks["root-1"].lifecycle
    assert folded.status is TaskStatus.COMPLETED
    assert recovered.unfinished_root is None

    second = await session_run.complete_task("again")
    assert isinstance(second, TaskTransitionRejected)
    assert second.reason == "terminal"
    await session_run.close()


@pytest.mark.asyncio
async def test_transition_rejections_are_typed() -> None:
    taskless = await _started_run("done")
    rejected = await taskless.complete_task("nothing to complete")
    assert isinstance(rejected, TaskTransitionRejected)
    assert rejected.reason == "unknown"
    await taskless.close()

    session_run = await _started_run("done", task=_task("root-1"))
    invalid = await session_run.unblock_task()
    assert isinstance(invalid, TaskTransitionRejected)
    assert invalid.reason == "invalid"
    blocker = TaskBlocker(awaiting="input", detail="needs scope approval")
    blocked = await session_run.block_task(blocker)
    assert not isinstance(blocked, TaskTransitionRejected)
    again = await session_run.block_task(blocker)
    assert isinstance(again, TaskTransitionRejected)
    assert again.reason == "invalid"
    await session_run.close()


@pytest.mark.asyncio
async def test_blocked_returns_to_active_only_via_unblock() -> None:
    session_run = await _started_run("done", task=_task("root-1"))
    blocker = TaskBlocker(awaiting="external", detail="awaiting target window")
    blocked = await session_run.block_task(blocker)
    assert not isinstance(blocked, TaskTransitionRejected)
    assert blocked.status is TaskStatus.BLOCKED
    assert blocked.blocker == blocker
    records = await session_run.journal.replay()
    assert recover_session(records).tasks["root-1"].lifecycle.status is (
        TaskStatus.BLOCKED
    )

    unblocked = await session_run.unblock_task()
    assert not isinstance(unblocked, TaskTransitionRejected)
    assert unblocked.status is TaskStatus.ACTIVE
    assert unblocked.blocker is None
    await session_run.close()


@pytest.mark.asyncio
async def test_terminal_transition_allowed_from_blocked() -> None:
    session_run = await _started_run("done", task=_task("root-1"))
    await session_run.block_task(
        TaskBlocker(awaiting="input", detail="awaiting operator")
    )
    cancelled = await session_run.cancel_task("engagement aborted")
    assert not isinstance(cancelled, TaskTransitionRejected)
    assert cancelled.status is TaskStatus.CANCELLED
    await session_run.close()


@pytest.mark.asyncio
async def test_transition_carries_budget_usage_snapshot() -> None:
    harness = _harness()
    model = ScriptedModel([text_events("done", usage={"total_tokens": 120})])
    session_run = await harness.start(
        model=model,
        task=_task("root-1"),
        budget_ledger=BudgetLedger(max_tokens=10000),
    )
    await session_run.prompt("begin")
    lifecycle = await session_run.complete_task("done")
    assert not isinstance(lifecycle, TaskTransitionRejected)
    assert lifecycle.usage is not None
    assert lifecycle.usage.total_tokens == 120
    await session_run.close()


@pytest.mark.asyncio
async def test_resume_restores_projection_and_continues_transitions() -> None:
    harness = _harness()
    model = ScriptedModel([text_events("done")])
    session_run = await harness.start(model=model, task=_task("root-1"))
    await session_run.prompt("begin")
    await session_run.block_task(
        TaskBlocker(awaiting="input", detail="needs operator decision")
    )
    run_id = session_run.run_id
    await session_run.close()

    resumed = await harness.resume(run_id, model=ScriptedModel([text_events("x")]))
    assert isinstance(resumed, SessionRun)
    lifecycle = resumed.task_lifecycle
    assert lifecycle is not None
    assert lifecycle.status is TaskStatus.BLOCKED
    unblocked = await resumed.unblock_task()
    assert not isinstance(unblocked, TaskTransitionRejected)
    assert unblocked.status is TaskStatus.ACTIVE
    records = await resumed.journal.replay()
    assert recover_session(records).tasks["root-1"].lifecycle.status is (
        TaskStatus.ACTIVE
    )
    await resumed.close()


@pytest.mark.asyncio
async def test_start_follow_up_commits_new_task_before_prompting() -> None:
    harness = _harness()
    model = ScriptedModel([text_events("first"), text_events("second")])
    session_run = await harness.start(model=model, task=_task("task-1"))
    await session_run.prompt("one")
    await session_run.complete_task("criteria met")
    follow_up_task = _task("task-2")
    result = await session_run.start_follow_up(follow_up_task, "two")
    assert not isinstance(result, TaskTransitionRejected)
    assert result.status is AgentRunStatus.COMPLETED
    assert session_run.task == follow_up_task

    records = await session_run.journal.replay()
    ordering = [record.type for record in records]
    assert ordering.index(JournalRecordType.TASK_CREATED) < ordering.index(
        JournalRecordType.INPUT_ACCEPTED
    )
    recovered = recover_session(records)
    assert recovered.tasks["task-1"].lifecycle.status is TaskStatus.COMPLETED
    follow_up = recovered.unfinished_root
    assert follow_up is not None
    assert follow_up.definition.task_id == "task-2"
    await session_run.close()


@pytest.mark.asyncio
async def test_start_follow_up_validates_current_task_state() -> None:
    session_run = await _started_run("done", task=_task("task-1"))
    unfinished = await session_run.start_follow_up(_task("task-2"), "two")
    assert isinstance(unfinished, TaskTransitionRejected)
    assert unfinished.reason == "invalid"
    await session_run.close()

    taskless = await _started_run("done")
    unknown = await taskless.start_follow_up(_task("task-2"), "two")
    assert isinstance(unknown, TaskTransitionRejected)
    assert unknown.reason == "unknown"
    await taskless.close()


@pytest.mark.asyncio
async def test_start_follow_up_in_place_when_leg_never_ran() -> None:
    harness = _harness()
    model = ScriptedModel([text_events("done")])
    session_run = await harness.start(model=model, task=_task("task-1"))
    run_id = session_run.run_id
    failed = await session_run.fail_task("objective abandoned")
    assert not isinstance(failed, TaskTransitionRejected)
    result = await session_run.start_follow_up(_task("task-2"), "go")
    assert not isinstance(result, TaskTransitionRejected)
    assert result.status is AgentRunStatus.COMPLETED
    assert session_run.run_id == run_id
    records = await session_run.journal.replay()
    recovered = recover_session(records)
    assert recovered.tasks["task-1"].lifecycle.status is TaskStatus.FAILED
    follow_up = recovered.unfinished_root
    assert follow_up is not None
    assert follow_up.definition.task_id == "task-2"
    await session_run.close()


@pytest.mark.asyncio
async def test_transition_survives_leg_advance_carry_forward() -> None:
    harness = _harness()
    model = ScriptedModel([text_events("first"), text_events("second")])
    session_run = await harness.start(model=model, task=_task("root-1"))
    await session_run.prompt("one")
    blocked = await session_run.block_task(
        TaskBlocker(awaiting="input", detail="needs operator decision")
    )
    assert not isinstance(blocked, TaskTransitionRejected)
    # The block transition committed into a leg with no own turn commit;
    # the next leg advance must carry it past the fork boundary.
    result = await session_run.prompt("two")
    assert result.status is AgentRunStatus.COMPLETED
    lifecycle = session_run.task_lifecycle
    assert lifecycle is not None
    assert lifecycle.status is TaskStatus.BLOCKED
    records = await session_run.journal.replay()
    assert recover_session(records).tasks["root-1"].lifecycle.status is (
        TaskStatus.BLOCKED
    )
    await session_run.close()


@pytest.mark.asyncio
async def test_follow_up_task_survives_failed_first_leg() -> None:
    harness = _harness()
    model = ScriptedModel(
        [text_events("first"), failed_events("boom"), text_events("third")]
    )
    session_run = await harness.start(model=model, task=_task("task-1"))
    await session_run.prompt("one")
    await session_run.complete_task("criteria met")
    follow_up = await session_run.start_follow_up(_task("task-2"), "two")
    assert not isinstance(follow_up, TaskTransitionRejected)
    assert follow_up.status is AgentRunStatus.FAILED
    # The follow-up leg failed before its first own commit; the next
    # advance must carry the new task.created past the fork boundary.
    result = await session_run.prompt("three")
    assert not isinstance(result, AgentRunRejected)
    assert result.status is AgentRunStatus.COMPLETED
    records = await session_run.journal.replay()
    recovered = recover_session(records)
    assert recovered.tasks["task-1"].lifecycle.status is TaskStatus.COMPLETED
    unfinished = recovered.unfinished_root
    assert unfinished is not None
    assert unfinished.definition.task_id == "task-2"
    await session_run.close()


@pytest.mark.asyncio
async def test_prompt_rejects_after_task_terminal() -> None:
    session_run = await _started_run("done", task=_task("root-1"))
    await session_run.prompt("begin")
    await session_run.fail_task("unrecoverable")
    rejected = await session_run.prompt("more work")
    assert isinstance(rejected, AgentRunRejected)
    assert rejected.reason == "task_terminal"
    await session_run.close()
