from __future__ import annotations

import pytest

from qitos.core import ChildHandle, PlanContractError, PlanStatus
from qitos.core.journal import JournalRecordType
from qitos.core.task import Task
from qitos.core.tool_registry import ToolRegistry
from qitos.kit.journal import (
    InMemoryJournalStore,
    InMemorySessionJournal,
    recover_session,
)
from qitos.kit.journal.turn_recorder import encode_task_created
from qitos.kit.tool.planning import UpdatePlanTool
from qitos.kit.session import SessionHarness

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


def _node(
    node_id: str,
    *,
    status: str = "pending",
    dependencies: list[str] | None = None,
    owner: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "node_id": node_id,
        "description": f"Work on {node_id}",
        "status": status,
        "dependencies": dependencies or [],
        "owner": owner,
    }


@pytest.mark.asyncio
async def test_update_plan_tool_commits_whole_graph_to_the_journal() -> None:
    journal = await _journal()
    arguments = {
        "plan": [
            _node("collect", status="completed"),
            _node("verify", dependencies=["collect"]),
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
    assert recovered.plan.ready_node_ids == ("verify",)
    assert any(
        record.type is JournalRecordType.PLAN_UPDATED
        for record in await journal.replay()
    )


@pytest.mark.asyncio
async def test_update_plan_tool_commits_inside_the_agent_tool_transaction(
    tmp_path,
) -> None:
    arguments = {"plan": [_node("inspect")]}
    model = ScriptedModel(
        [
            tool_events(
                [tool_call_wire("plan-call", "update_plan", arguments)]
            ),
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
    assert session_run.plan.node("inspect").status is PlanStatus.PENDING
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
    arguments = {"plan": [_node("inspect")]}
    model = ScriptedModel(
        [
            tool_events(
                [tool_call_wire("plan-call", "update_plan", arguments)]
            ),
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
    assert recovered.plans["task-first"].node("inspect").status is PlanStatus.PENDING
    assert "task-second" not in recovered.plans
    await session_run.close()


@pytest.mark.asyncio
async def test_update_plan_tool_rejects_history_rewrite() -> None:
    journal = await _journal()
    tool = UpdatePlanTool()
    await tool.execute(
        {"plan": [_node("keep")]},
        runtime_context={
            "journal": journal,
            "tool_call_id": "first",
            "task_id": "task-plan",
        },
    )

    with pytest.raises(PlanContractError):
        await tool.execute(
            {"plan": []},
            runtime_context={
                "journal": journal,
                "tool_call_id": "second",
                "task_id": "task-plan",
            },
        )


@pytest.mark.asyncio
async def test_update_plan_tool_cannot_invent_a_child_owner() -> None:
    journal = await _journal()
    owner = ChildHandle("child-a", "run-plan-tool")

    with pytest.raises(PlanContractError):
        await UpdatePlanTool().execute(
            {
                "plan": [
                    _node(
                        "delegated",
                        status=PlanStatus.IN_PROGRESS.value,
                        owner=owner.to_dict(),
                    )
                ]
            },
            runtime_context={
                "journal": journal,
                "tool_call_id": "forged",
                "task_id": "task-plan",
            },
        )


@pytest.mark.asyncio
async def test_update_plan_tool_requires_a_durable_session_boundary() -> None:
    with pytest.raises(PlanContractError):
        await UpdatePlanTool().execute({"plan": [_node("a")]})


def test_update_plan_tool_exposes_strict_graph_schema() -> None:
    schema = UpdatePlanTool().spec.input_schema
    node_schema = schema["properties"]["plan"]["items"]

    assert schema["additionalProperties"] is False
    assert node_schema["additionalProperties"] is False
    assert "dependencies" in node_schema["required"]
    assert "owner" in node_schema["properties"]
