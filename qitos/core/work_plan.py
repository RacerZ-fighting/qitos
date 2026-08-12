"""Typed lightweight work-plan state for long-running agents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from .action import Action
from .tool_result import ToolResult

MAX_WORK_PLAN_ITEMS = 64
MAX_WORK_PLAN_STEP_CHARS = 512
MAX_WORK_PLAN_EXPLANATION_CHARS = 2_000
UPDATE_PLAN_TOOL_NAME = "update_plan"


class WorkPlanContractError(ValueError):
    """Raised when a plan update cannot become canonical Agent state."""


class WorkPlanStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


def _normalized_text(
    value: str,
    field: str,
    limit: int,
    *,
    single_line: bool = True,
) -> str:
    if not isinstance(value, str):
        raise WorkPlanContractError(f"{field} must be text")
    normalized = value.strip()
    if not normalized:
        raise WorkPlanContractError(f"{field} must be non-empty text")
    if len(normalized) > limit:
        raise WorkPlanContractError(
            f"{field} must be at most {limit} characters"
        )
    if single_line and any(
        ord(character) < 32 or ord(character) == 127 for character in normalized
    ):
        raise WorkPlanContractError(f"{field} must be a single printable line")
    return normalized


@dataclass(frozen=True, slots=True)
class WorkPlanItem:
    step: str
    status: WorkPlanStatus

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "step",
            _normalized_text(
                self.step,
                "WorkPlan step",
                MAX_WORK_PLAN_STEP_CHARS,
            ),
        )
        if not isinstance(self.status, WorkPlanStatus):
            raise WorkPlanContractError("WorkPlan status is invalid")


@dataclass(frozen=True, slots=True)
class WorkPlanState:
    """Ordered canonical checklist; user-facing files remain projections."""

    items: tuple[WorkPlanItem, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple):
            raise WorkPlanContractError("WorkPlan items must be an immutable tuple")
        if len(self.items) > MAX_WORK_PLAN_ITEMS:
            raise WorkPlanContractError(
                f"WorkPlan must contain at most {MAX_WORK_PLAN_ITEMS} items"
            )
        if any(not isinstance(item, WorkPlanItem) for item in self.items):
            raise WorkPlanContractError("WorkPlan contains an invalid item")
        steps = tuple(item.step for item in self.items)
        if len(steps) != len(set(steps)):
            raise WorkPlanContractError("WorkPlan steps must be unique")
        active = sum(
            item.status is WorkPlanStatus.IN_PROGRESS for item in self.items
        )
        if active > 1:
            raise WorkPlanContractError(
                "WorkPlan allows at most one in-progress item"
            )


@dataclass(frozen=True, slots=True)
class WorkPlanUpdate:
    plan: WorkPlanState
    explanation: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plan, WorkPlanState):
            raise WorkPlanContractError("WorkPlan update state is invalid")
        if self.explanation is not None:
            object.__setattr__(
                self,
                "explanation",
                _normalized_text(
                    self.explanation,
                    "WorkPlan explanation",
                    MAX_WORK_PLAN_EXPLANATION_CHARS,
                    single_line=False,
                ),
            )


def parse_work_plan_update(arguments: Mapping[str, object]) -> WorkPlanUpdate:
    unknown = set(arguments) - {"plan", "explanation"}
    if unknown:
        fields = ", ".join(sorted(unknown))
        raise WorkPlanContractError(f"Unknown WorkPlan fields: {fields}")
    raw_plan = arguments.get("plan")
    if not isinstance(raw_plan, list):
        raise WorkPlanContractError("WorkPlan must be a list")
    items: list[WorkPlanItem] = []
    for raw_item in raw_plan:
        if not isinstance(raw_item, Mapping) or set(raw_item) != {"step", "status"}:
            raise WorkPlanContractError(
                "Each WorkPlan item requires step and status"
            )
        step = raw_item.get("step")
        status = raw_item.get("status")
        if not isinstance(step, str) or not isinstance(status, str):
            raise WorkPlanContractError("WorkPlan step and status must be text")
        try:
            plan_status = WorkPlanStatus(status)
        except ValueError as exc:
            raise WorkPlanContractError(
                f"Invalid WorkPlan status: {status}"
            ) from exc
        items.append(WorkPlanItem(step, plan_status))
    explanation = arguments.get("explanation")
    if explanation is not None and not isinstance(explanation, str):
        raise WorkPlanContractError("WorkPlan explanation must be text")
    return WorkPlanUpdate(WorkPlanState(tuple(items)), explanation)


def work_plan_state_to_dict(state: WorkPlanState) -> dict[str, object]:
    return {
        "items": [
            {"step": item.step, "status": item.status.value}
            for item in state.items
        ]
    }


def work_plan_state_from_dict(payload: Mapping[str, object]) -> WorkPlanState:
    if set(payload) != {"items"}:
        raise WorkPlanContractError("WorkPlan state requires only items")
    return parse_work_plan_update({"plan": payload["items"]}).plan


def reduce_work_plan(
    current: WorkPlanState,
    actions: Sequence[Action],
    results: Sequence[ToolResult],
) -> WorkPlanState:
    """Fold successful update_plan calls into a new WorkPlan value."""
    if len(actions) != len(results):
        raise ValueError("actions and results must have the same length")
    reduced = current
    for action, raw_result in zip(actions, results):
        result = ToolResult.from_value(raw_result)
        if action.name == UPDATE_PLAN_TOOL_NAME and result.is_success:
            reduced = parse_work_plan_update(dict(action.args or {})).plan
    return reduced


_MARKERS = {
    WorkPlanStatus.PENDING: "[ ]",
    WorkPlanStatus.IN_PROGRESS: "[~]",
    WorkPlanStatus.COMPLETED: "[x]",
}


def render_work_plan_markdown(state: WorkPlanState) -> str | None:
    """Render one stable TODO snapshot, or None for an empty plan."""
    if not state.items:
        return None
    lines = ["# TODO", ""]
    lines.extend(f"- {_MARKERS[item.status]} {item.step}" for item in state.items)
    return "\n".join(lines)


__all__ = [
    "MAX_WORK_PLAN_EXPLANATION_CHARS",
    "MAX_WORK_PLAN_ITEMS",
    "MAX_WORK_PLAN_STEP_CHARS",
    "UPDATE_PLAN_TOOL_NAME",
    "WorkPlanContractError",
    "WorkPlanItem",
    "WorkPlanState",
    "WorkPlanStatus",
    "WorkPlanUpdate",
    "parse_work_plan_update",
    "reduce_work_plan",
    "render_work_plan_markdown",
    "work_plan_state_from_dict",
    "work_plan_state_to_dict",
]
