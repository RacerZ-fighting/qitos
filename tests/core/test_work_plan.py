from __future__ import annotations

from types import SimpleNamespace

import pytest

from qitos.core import (
    ToolResult,
    WorkPlanContractError,
    WorkPlanItem,
    WorkPlanState,
    WorkPlanStatus,
    parse_work_plan_update,
    reduce_work_plan,
    render_work_plan_markdown,
    work_plan_state_from_dict,
    work_plan_state_to_dict,
)


def _plan() -> WorkPlanState:
    return WorkPlanState(
        (
            WorkPlanItem("Inspect the target", WorkPlanStatus.COMPLETED),
            WorkPlanItem("Validate the result", WorkPlanStatus.IN_PROGRESS),
        )
    )


def test_work_plan_round_trips_through_durable_dict() -> None:
    plan = _plan()

    restored = work_plan_state_from_dict(work_plan_state_to_dict(plan))

    assert restored == plan


def test_work_plan_rejects_ambiguous_checklists() -> None:
    with pytest.raises(WorkPlanContractError):
        WorkPlanState(
            (
                WorkPlanItem("same", WorkPlanStatus.PENDING),
                WorkPlanItem("same", WorkPlanStatus.COMPLETED),
            )
        )

    with pytest.raises(WorkPlanContractError):
        WorkPlanState(
            (
                WorkPlanItem("first", WorkPlanStatus.IN_PROGRESS),
                WorkPlanItem("second", WorkPlanStatus.IN_PROGRESS),
            )
        )


def test_work_plan_reducer_applies_only_successful_updates_in_call_order() -> None:
    first = {"plan": [{"step": "first", "status": "in_progress"}]}
    rejected = {"plan": [{"step": "ignored", "status": "completed"}]}
    last = {"plan": [{"step": "last", "status": "completed"}]}

    reduced = reduce_work_plan(
        WorkPlanState(),
        [
            SimpleNamespace(name="update_plan", args=first),
            SimpleNamespace(name="update_plan", args=rejected),
            SimpleNamespace(name="unrelated", args={}),
            SimpleNamespace(name="update_plan", args=last),
        ],
        [
            ToolResult(output={}),
            ToolResult(status="error", error="rejected"),
            ToolResult(output={}),
            ToolResult(output={}),
        ],
    )

    assert reduced == parse_work_plan_update(last).plan


def test_work_plan_markdown_is_a_deterministic_read_only_projection() -> None:
    plan = _plan()

    first = render_work_plan_markdown(plan)
    second = render_work_plan_markdown(
        work_plan_state_from_dict(work_plan_state_to_dict(plan))
    )

    assert first == second
    assert first is not None
    assert all(item.step in first for item in plan.items)
    assert render_work_plan_markdown(WorkPlanState()) is None
