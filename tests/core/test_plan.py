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


def test_a_checklist_may_carry_a_step_per_concurrent_line_of_work() -> None:
    # An Agent driving several Subagents at once has that many steps underway,
    # and a checklist that can show only one of them describes a run that is
    # not happening.
    statuses = (
        PlanStatus.IN_PROGRESS,
        PlanStatus.IN_PROGRESS,
        PlanStatus.COMPLETED,
        PlanStatus.IN_PROGRESS,
    )
    update = parse_plan_update(
        {
            "plan": [
                {"step": f"Front {index}", "status": status.value}
                for index, status in enumerate(statuses)
            ]
        }
    )

    assert tuple(item.status for item in update.plan.items) == statuses
    assert plan_from_dict(plan_to_dict(update.plan)) == update.plan
    rendered = render_plan_markdown(update.plan)
    assert rendered is not None
    assert rendered.count("[~]") == 3


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


def test_plan_snapshot_decoding_rejects_an_unrecognised_shape() -> None:
    """Only the current durable shape decodes; anything else fails closed.

    A payload the decoder does not recognise must raise rather than yield an
    empty Plan, so a snapshot written by another shape cannot silently erase a
    recovered checklist.
    """

    with pytest.raises(PlanContractError):
        plan_from_dict({"nodes": [{"description": "First", "status": "pending"}]})
    with pytest.raises(PlanContractError):
        plan_from_dict({"items": [], "extra": []})
