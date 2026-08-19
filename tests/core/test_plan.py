from __future__ import annotations

from types import SimpleNamespace

import pytest

from qitos.core import (
    Plan,
    PlanContractError,
    PlanItem,
    PlanStatus,
    ToolResult,
    parse_plan_update,
    plan_from_dict,
    plan_to_dict,
    reduce_plan,
    render_plan_markdown,
)


def test_plan_round_trips_as_an_ordered_checklist() -> None:
    plan = Plan(
        (
            PlanItem("Inspect target", PlanStatus.COMPLETED),
            PlanItem("Verify result", PlanStatus.IN_PROGRESS),
        )
    )

    restored = plan_from_dict(plan_to_dict(plan))

    assert restored == plan
    assert tuple(item.step for item in restored.items) == (
        "Inspect target",
        "Verify result",
    )


def test_plan_allows_at_most_one_in_progress_item() -> None:
    with pytest.raises(PlanContractError, match="at most one"):
        Plan(
            (
                PlanItem("First", PlanStatus.IN_PROGRESS),
                PlanItem("Second", PlanStatus.IN_PROGRESS),
            )
        )


def test_model_plan_shape_rejects_retired_graph_fields() -> None:
    with pytest.raises(PlanContractError, match="only step and status"):
        parse_plan_update(
            {
                "plan": [
                    {
                        "step": "Inspect target",
                        "status": "pending",
                        "dependencies": [],
                    }
                ]
            }
        )


def test_plan_reducer_accepts_free_replacement_in_call_order() -> None:
    first = {
        "plan": [{"step": "Initial approach", "status": "in_progress"}]
    }
    last = {
        "plan": [
            {"step": "Revised approach", "status": "completed"},
            {"step": "Report", "status": "pending"},
        ]
    }

    reduced = reduce_plan(
        None,
        [
            SimpleNamespace(name="update_plan", args=first),
            SimpleNamespace(name="update_plan", args=last),
        ],
        [ToolResult(output={}), ToolResult(output={})],
    )

    assert reduced == parse_plan_update(last).plan


def test_plan_markdown_preserves_checklist_order() -> None:
    plan = Plan(
        (
            PlanItem("First", PlanStatus.COMPLETED),
            PlanItem("Second", PlanStatus.PENDING),
        )
    )

    rendered = render_plan_markdown(plan)

    assert rendered is not None
    assert rendered.index("First") < rendered.index("Second")
    assert render_plan_markdown(Plan()) is None


def test_legacy_graph_snapshot_migrates_without_retired_workflow_state() -> None:
    restored = plan_from_dict(
        {
            "nodes": [
                {
                    "node_id": "first",
                    "description": "First",
                    "status": "cancelled",
                    "dependencies": [],
                    "owner": None,
                },
                {
                    "node_id": "second",
                    "description": "Second",
                    "status": "blocked",
                    "dependencies": ["first"],
                    "owner": None,
                },
            ]
        }
    )

    assert restored == Plan(
        (
            PlanItem("First", PlanStatus.COMPLETED),
            PlanItem("Second", PlanStatus.PENDING),
        )
    )
    assert plan_to_dict(restored) == {
        "items": [
            {"step": "First", "status": "completed"},
            {"step": "Second", "status": "pending"},
        ]
    }
