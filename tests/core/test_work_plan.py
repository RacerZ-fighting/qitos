from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import pytest

from qitos.checkpoint import Checkpoint, CheckpointId
from qitos.core import (
    Action,
    StateSchema,
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


def test_work_plan_round_trips_through_checkpoint_json() -> None:
    @dataclass
    class _State(StateSchema):
        work_plan: WorkPlanState = field(default_factory=WorkPlanState)

        def to_dict(self) -> Dict[str, Any]:
            payload = super().to_dict()
            payload["work_plan"] = work_plan_state_to_dict(self.work_plan)
            return payload

        @classmethod
        def from_dict(cls, payload: Dict[str, Any], strict: bool = True) -> _State:
            owned = dict(payload)
            owned["work_plan"] = work_plan_state_from_dict(owned["work_plan"])
            return super().from_dict(owned, strict=strict)

    state = _State(task="authorized task", work_plan=_plan())
    checkpoint = Checkpoint(
        id=CheckpointId("checkpoint-1"),
        thread_id="thread-1",
        step=1,
        state_data=state.to_dict(),
    )

    restored_checkpoint = Checkpoint.from_dict(checkpoint.to_dict())
    restored_state = _State.from_dict(restored_checkpoint.state_data)

    assert restored_state.work_plan == state.work_plan


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
            Action(name="update_plan", args=first),
            Action(name="update_plan", args=rejected),
            Action(name="unrelated", args={}),
            Action(name="update_plan", args=last),
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
