"""Small replaceable checklist used to project current task progress."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from .tool_result import ToolResult

MAX_PLAN_ITEMS = 64
MAX_PLAN_STEP_CHARS = 512
MAX_PLAN_EXPLANATION_CHARS = 2_000
UPDATE_PLAN_TOOL_NAME = "update_plan"


class PlanContractError(ValueError):
    """Raised when a Plan value violates the public checklist contract."""


class PlanStatus(str, Enum):
    """Display status for one checklist item."""

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
        raise PlanContractError(f"{field} must be text")
    normalized = value.strip()
    if not normalized:
        raise PlanContractError(f"{field} must be non-empty text")
    if len(normalized) > limit:
        raise PlanContractError(f"{field} must be at most {limit} characters")
    if single_line and any(
        ord(character) < 32 or ord(character) == 127 for character in normalized
    ):
        raise PlanContractError(f"{field} must be a single printable line")
    return normalized


@dataclass(frozen=True, slots=True)
class PlanItem:
    """One freely replaceable progress step."""

    step: str
    status: PlanStatus = PlanStatus.PENDING

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "step",
            _normalized_text(self.step, "Plan step", MAX_PLAN_STEP_CHARS),
        )
        if not isinstance(self.status, PlanStatus):
            raise PlanContractError("Plan status is invalid")


@dataclass(frozen=True, slots=True)
class Plan:
    """Current progress checklist; history remains in the append-only Journal."""

    items: tuple[PlanItem, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple):
            raise PlanContractError("Plan items must be an immutable tuple")
        if len(self.items) > MAX_PLAN_ITEMS:
            raise PlanContractError(
                f"Plan must contain at most {MAX_PLAN_ITEMS} items"
            )
        if any(not isinstance(item, PlanItem) for item in self.items):
            raise PlanContractError("Plan contains an invalid item")
        in_progress = sum(
            item.status is PlanStatus.IN_PROGRESS for item in self.items
        )
        if in_progress > 1:
            raise PlanContractError("Plan allows at most one in-progress item")


@dataclass(frozen=True, slots=True)
class PlanUpdate:
    plan: Plan
    explanation: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plan, Plan):
            raise PlanContractError("Plan update value is invalid")
        if self.explanation is not None:
            object.__setattr__(
                self,
                "explanation",
                _normalized_text(
                    self.explanation,
                    "Plan explanation",
                    MAX_PLAN_EXPLANATION_CHARS,
                    single_line=False,
                ),
            )


def parse_plan_update(arguments: Mapping[str, object]) -> PlanUpdate:
    """Parse the strict model-facing whole-checklist replacement shape."""

    unknown = set(arguments) - {"plan", "explanation"}
    if unknown:
        fields = ", ".join(sorted(unknown))
        raise PlanContractError(f"Unknown Plan fields: {fields}")
    raw_plan = arguments.get("plan")
    if not isinstance(raw_plan, list):
        raise PlanContractError("Plan must be a list")
    items: list[PlanItem] = []
    for raw_item in raw_plan:
        if not isinstance(raw_item, Mapping) or set(raw_item) != {"step", "status"}:
            raise PlanContractError("Each Plan item requires only step and status")
        try:
            status = PlanStatus(str(raw_item["status"]))
        except ValueError as exc:
            raise PlanContractError(
                f"Invalid Plan status: {raw_item['status']}"
            ) from exc
        items.append(PlanItem(step=raw_item["step"], status=status))
    explanation = arguments.get("explanation")
    if explanation is not None and not isinstance(explanation, str):
        raise PlanContractError("Plan explanation must be text")
    return PlanUpdate(Plan(tuple(items)), explanation)


def plan_to_dict(plan: Plan) -> dict[str, object]:
    """Encode one Plan with the current durable shape."""

    if not isinstance(plan, Plan):
        raise TypeError("plan must be a Plan")
    return {
        "items": [
            {"step": item.step, "status": item.status.value}
            for item in plan.items
        ]
    }


def _legacy_status(value: object) -> PlanStatus:
    if value in {"completed", "cancelled"}:
        return PlanStatus.COMPLETED
    if value == "in_progress":
        return PlanStatus.IN_PROGRESS
    if value in {"pending", "failed", "blocked"}:
        return PlanStatus.PENDING
    raise PlanContractError(f"Invalid legacy Plan status: {value}")


def _legacy_plan_from_dict(payload: Mapping[str, object]) -> Plan:
    """Fold the retired graph snapshot into a display-only checklist."""

    raw_nodes = payload.get("nodes")
    durable_fields = {
        "node_id",
        "description",
        "status",
        "dependencies",
        "owner",
    }
    if not isinstance(raw_nodes, list) or any(
        not isinstance(node, Mapping) or set(node) != durable_fields
        for node in raw_nodes
    ):
        raise PlanContractError("Legacy durable Plan nodes have invalid fields")
    items: list[PlanItem] = []
    active_seen = False
    for raw_node in raw_nodes:
        status = _legacy_status(raw_node["status"])
        if status is PlanStatus.IN_PROGRESS:
            if active_seen:
                status = PlanStatus.PENDING
            active_seen = True
        items.append(PlanItem(step=raw_node["description"], status=status))
    return Plan(tuple(items))


def plan_from_dict(payload: Mapping[str, object]) -> Plan:
    """Decode the current shape or one retired dependency-graph snapshot."""

    if set(payload) == {"nodes"}:
        return _legacy_plan_from_dict(payload)
    if set(payload) != {"items"}:
        raise PlanContractError("Plan state requires only items")
    raw_items = payload["items"]
    if not isinstance(raw_items, list):
        raise PlanContractError("Durable Plan items must be a list")
    return parse_plan_update({"plan": raw_items}).plan


def _action_name(action: object) -> str:
    name = getattr(action, "name", None)
    if isinstance(name, str) and name:
        return name
    if isinstance(action, Mapping):
        return str(action.get("name", ""))
    return ""


def _action_args(action: object) -> Mapping[str, object]:
    for field in ("args", "arguments"):
        value = getattr(action, field, None)
        if isinstance(value, Mapping):
            return value
    if isinstance(action, Mapping):
        raw = action.get("args") or action.get("arguments") or {}
        if isinstance(raw, Mapping):
            return raw
    return {}


def reduce_plan(
    current: Plan | None,
    actions: Sequence[object],
    results: Sequence[ToolResult],
) -> Plan | None:
    """Fold successful whole-checklist replacements in call order."""

    if len(actions) != len(results):
        raise ValueError("actions and results must have the same length")
    reduced = current
    for action, raw_result in zip(actions, results):
        result = ToolResult.from_value(raw_result)
        if _action_name(action) == UPDATE_PLAN_TOOL_NAME and result.is_success:
            reduced = parse_plan_update(dict(_action_args(action))).plan
    return reduced


_MARKERS = {
    PlanStatus.PENDING: "[ ]",
    PlanStatus.IN_PROGRESS: "[~]",
    PlanStatus.COMPLETED: "[x]",
}


def render_plan_markdown(plan: Plan) -> str | None:
    """Render a deterministic TODO projection, or None for an empty Plan."""

    if not isinstance(plan, Plan):
        raise TypeError("plan must be a Plan")
    if not plan.items:
        return None
    lines = ["# TODO", ""]
    lines.extend(f"- {_MARKERS[item.status]} {item.step}" for item in plan.items)
    return "\n".join(lines)


__all__ = [
    "MAX_PLAN_EXPLANATION_CHARS",
    "MAX_PLAN_ITEMS",
    "MAX_PLAN_STEP_CHARS",
    "UPDATE_PLAN_TOOL_NAME",
    "Plan",
    "PlanContractError",
    "PlanItem",
    "PlanStatus",
    "PlanUpdate",
    "parse_plan_update",
    "plan_from_dict",
    "plan_to_dict",
    "reduce_plan",
    "render_plan_markdown",
]
