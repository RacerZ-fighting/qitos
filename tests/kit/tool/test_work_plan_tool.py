from __future__ import annotations

import pytest

from qitos.core import WorkPlanContractError, WorkPlanState, parse_work_plan_update
from qitos.kit.tool.planning import UpdateWorkPlanTool


@pytest.mark.asyncio
async def test_update_work_plan_tool_returns_normalized_intent_without_mutating_state() -> None:
    class _State:
        work_plan = WorkPlanState()

    arguments = {
        "plan": [
            {"step": "Collect evidence", "status": "completed"},
            {"step": "Verify conclusion", "status": "in_progress"},
        ],
        "explanation": "Evidence collected",
    }
    state = _State()
    tool = UpdateWorkPlanTool()

    result = await tool.execute(arguments, runtime_context={"state": state})

    assert result["plan"] == arguments["plan"]
    assert result["explanation"] == arguments["explanation"]
    assert state.work_plan == WorkPlanState()
    assert parse_work_plan_update(result).plan.items


@pytest.mark.asyncio
async def test_update_work_plan_tool_rejects_invalid_replacement() -> None:
    tool = UpdateWorkPlanTool()

    with pytest.raises(WorkPlanContractError):
        await tool.execute(
            {
                "plan": [
                    {"step": "first", "status": "in_progress"},
                    {"step": "second", "status": "in_progress"},
                ]
            }
        )


def test_update_work_plan_tool_exposes_strict_schema() -> None:
    schema = UpdateWorkPlanTool().spec.input_schema

    assert schema["additionalProperties"] is False
    assert schema["properties"]["plan"]["items"]["additionalProperties"] is False
