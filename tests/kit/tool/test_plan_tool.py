from __future__ import annotations

import pytest

from qitos.core import Plan, PlanContractError, PlanStatus
from qitos.core.journal import JournalRecordType
from qitos.core.task import Task
from qitos.core.tool_registry import ToolRegistry
from qitos.kit.journal import (
    InMemoryJournalStore,
    InMemorySessionJournal,
    recover_session,
)
from qitos.kit.journal.turn_recorder import encode_task_created
from qitos.kit.session import SessionHarness
from qitos.kit.tool.planning import UpdatePlanTool

from tests.core.agent_fakes import (
    ScriptedModel,
    text_events,
    tool_call_wire,
    tool_events,
)


async def _journal() -> InMemorySessionJournal:
    journal = InMemorySessionJournal(InMemoryJournalStore())
    await journal.create("run-plan-tool", {"purpose": "test"})
    await journal.append(
        JournalRecordType.TASK_CREATED,
        encode_task_created(Task(task_id="task-plan", objective="Plan work")),
        record_id="run-plan-tool:task:task-plan:created",
    )
    return journal


def _item(step: str, *, status: str = "pending") -> dict[str, str]:
    return {"step": step, "status": status}


@pytest.mark.asyncio
async def test_update_plan_tool_commits_whole_checklist_to_the_journal() -> None:
    journal = await _journal()
    arguments = {
        "plan": [
            _item("Collect evidence", status="completed"),
            _item("Verify result"),
        ],
        "explanation": "Evidence collected",
    }

    result = await UpdatePlanTool().execute(
        arguments,
        runtime_context={
            "journal": journal,
            "tool_call_id": "call-plan",
            "task_id": "task-plan",
        },
    )
    recovered = recover_session(await journal.replay())

    assert result["plan"] == arguments["plan"]
    assert recovered.plan is not None
    assert tuple(item.step for item in recovered.plan.items) == (
        "Collect evidence",
        "Verify result",
    )
    assert any(
        record.type is JournalRecordType.PLAN_UPDATED
        for record in await journal.replay()
    )


@pytest.mark.asyncio
async def test_update_plan_tool_commits_inside_the_agent_tool_transaction(
    tmp_path,
) -> None:
    arguments = {"plan": [_item("Inspect target")]}
    model = ScriptedModel(
        [
            tool_events([tool_call_wire("plan-call", "update_plan", arguments)]),
            text_events("done"),
        ]
    )
    session_run = await SessionHarness(tmp_path / "journals").start(
        model=model,
        tool_registry=ToolRegistry().register(UpdatePlanTool()),
        task=Task(task_id="task-plan", objective="Plan the work"),
    )

    await session_run.prompt("start")
    records = await session_run.journal.replay()
    ordering = [record.type for record in records]

    assert session_run.plan is not None
    assert session_run.plan.items[0].status is PlanStatus.PENDING
    assert ordering.index(JournalRecordType.TOOL_STARTED) < ordering.index(
        JournalRecordType.PLAN_UPDATED
    )
    assert ordering.index(JournalRecordType.PLAN_UPDATED) < ordering.index(
        JournalRecordType.TOOL_TERMINAL
    )
    await session_run.close()


@pytest.mark.asyncio
async def test_terminal_follow_up_starts_without_the_previous_tasks_plan(
    tmp_path,
) -> None:
    arguments = {"plan": [_item("Inspect target")]}
    model = ScriptedModel(
        [
            tool_events([tool_call_wire("plan-call", "update_plan", arguments)]),
            text_events("first done"),
            text_events("follow-up done"),
        ]
    )
    session_run = await SessionHarness(tmp_path / "follow-up-journals").start(
        model=model,
        tool_registry=ToolRegistry().register(UpdatePlanTool()),
        task=Task(task_id="task-first", objective="First work"),
    )
    await session_run.prompt("start")
    assert session_run.plan is not None
    await session_run.complete_task("verified")

    await session_run.start_follow_up(
        Task(task_id="task-second", objective="Second work"),
        "continue",
    )
    recovered = recover_session(await session_run.journal.replay())

    assert session_run.task is not None
    assert session_run.task.task_id == "task-second"
    assert session_run.plan is None
    assert recovered.plans["task-first"].items[0].status is PlanStatus.PENDING
    assert "task-second" not in recovered.plans
    await session_run.close()


@pytest.mark.asyncio
async def test_update_plan_tool_allows_rewriting_or_clearing_current_steps() -> None:
    journal = await _journal()
    tool = UpdatePlanTool()
    await tool.execute(
        {"plan": [_item("Initial approach", status="in_progress")]},
        runtime_context={
            "journal": journal,
            "tool_call_id": "first",
            "task_id": "task-plan",
        },
    )

    result = await tool.execute(
        {"plan": []},
        runtime_context={
            "journal": journal,
            "tool_call_id": "second",
            "task_id": "task-plan",
        },
    )

    assert result["plan"] == []
    assert recover_session(await journal.replay()).plan == Plan()


@pytest.mark.asyncio
async def test_update_plan_tool_requires_a_durable_session_boundary() -> None:
    with pytest.raises(PlanContractError):
        await UpdatePlanTool().execute({"plan": [_item("A")]})


def test_update_plan_tool_exposes_strict_checklist_schema() -> None:
    schema = UpdatePlanTool().spec.input_schema
    item_schema = schema["properties"]["plan"]["items"]

    assert schema["additionalProperties"] is False
    assert item_schema["additionalProperties"] is False
    assert set(item_schema["required"]) == {"step", "status"}
    assert set(item_schema["properties"]) == {"step", "status"}
