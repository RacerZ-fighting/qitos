"""Goal-bearing Task contract for QitOS Sessions.

A Task is the durable goal of one Root or Child execution. The immutable
definition carries the objective, success criteria, constraints, stable
resource/context references, budget and creation provenance; the durable
lifecycle (``active | blocked | completed | failed | cancelled``, usage,
typed blocker or terminal reason) is folded from ``task.created`` /
``task.transition`` journal records. Benchmark resources, environment
probing, evaluation metrics and free-form metadata do not belong here; they
stay at application boundaries and reference the Task by id.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal, Optional

from .model_response import ModelUsage


class TaskStatus(str, Enum):
    """Durable lifecycle state of one Task."""

    ACTIVE = "active"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        """Completed, failed and cancelled commit exactly once."""

        return self in _TERMINAL_STATUSES


_TERMINAL_STATUSES = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
)

_TASK_REFERENCE_KINDS = frozenset({"file", "dir", "url", "artifact", "image"})

_TASK_BUDGET_FIELDS = {
    "max_steps",
    "max_runtime_seconds",
    "max_tokens",
    "max_cost_usd",
    "max_tool_concurrency",
    "max_children",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _optional_text(value: Any, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string or None")


@dataclass(frozen=True, slots=True)
class TaskReference:
    """Stable typed reference to Task context; never probed at runtime."""

    kind: str
    uri: str
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind not in _TASK_REFERENCE_KINDS:
            raise ValueError(f"unsupported TaskReference.kind: {self.kind!r}")
        if not isinstance(self.uri, str) or not self.uri.strip():
            raise ValueError("TaskReference.uri must be a non-empty string")
        if not isinstance(self.description, str):
            raise TypeError("TaskReference.description must be a string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "uri": self.uri,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskReference":
        if set(value) != {"kind", "uri", "description"}:
            raise ValueError("TaskReference fields are invalid")
        return cls(
            kind=value["kind"],
            uri=value["uri"],
            description=value["description"],
        )


@dataclass(frozen=True, slots=True)
class TaskBudget:
    """Task-level budget contract."""

    max_steps: Optional[int] = None
    max_runtime_seconds: Optional[float] = None
    max_tokens: Optional[int] = None
    max_cost_usd: Optional[float] = None
    max_tool_concurrency: Optional[int] = None
    max_children: Optional[int] = None

    def __post_init__(self) -> None:
        for name in (
            "max_steps",
            "max_tokens",
            "max_tool_concurrency",
            "max_children",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer or None")
            if value <= 0:
                raise ValueError(f"{name} must be positive or None")
        for name in ("max_runtime_seconds", "max_cost_usd"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number or None")
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be positive or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "max_runtime_seconds": self.max_runtime_seconds,
            "max_tokens": self.max_tokens,
            "max_cost_usd": self.max_cost_usd,
            "max_tool_concurrency": self.max_tool_concurrency,
            "max_children": self.max_children,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskBudget":
        if set(value) != _TASK_BUDGET_FIELDS:
            raise ValueError("TaskBudget fields are invalid")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class TaskBlocker:
    """Durable blocker on one Task.

    A blocked Task is resumable only after explicit caller input or an
    observed external-state change, delivered through an explicit unblock
    transition.
    """

    awaiting: Literal["input", "external"]
    detail: str

    def __post_init__(self) -> None:
        if self.awaiting not in ("input", "external"):
            raise ValueError("TaskBlocker.awaiting must be 'input' or 'external'")
        if not isinstance(self.detail, str) or not self.detail.strip():
            raise ValueError("TaskBlocker.detail must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {"awaiting": self.awaiting, "detail": self.detail}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskBlocker":
        if set(value) != {"awaiting", "detail"}:
            raise ValueError("TaskBlocker fields are invalid")
        return cls(awaiting=value["awaiting"], detail=value["detail"])


@dataclass(frozen=True, slots=True)
class TaskLifecycle:
    """Folded lifecycle projection of one Task.

    Invariants: ``blocker`` is present exactly while the Task is BLOCKED and
    ``terminal_reason`` exactly at a terminal status; a usage snapshot is
    allowed at any status.
    """

    status: TaskStatus
    usage: ModelUsage | None = None
    blocker: TaskBlocker | None = None
    terminal_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, TaskStatus):
            raise TypeError("TaskLifecycle.status must be a TaskStatus")
        if self.blocker is not None and not isinstance(self.blocker, TaskBlocker):
            raise TypeError("TaskLifecycle.blocker must be a TaskBlocker or None")
        if (self.blocker is not None) is not (self.status is TaskStatus.BLOCKED):
            raise ValueError("blocker is present exactly while the Task is blocked")
        if (self.terminal_reason is not None) is not self.status.terminal:
            raise ValueError(
                "terminal_reason is present exactly at a terminal status"
            )
        if self.terminal_reason is not None and (
            not isinstance(self.terminal_reason, str)
            or not self.terminal_reason.strip()
        ):
            raise ValueError("TaskLifecycle.terminal_reason must be non-empty")
        if self.usage is not None and not isinstance(self.usage, ModelUsage):
            raise TypeError("TaskLifecycle.usage must be a ModelUsage or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "usage": self.usage.to_dict() if self.usage is not None else None,
            "blocker": (
                self.blocker.to_dict() if self.blocker is not None else None
            ),
            "terminal_reason": self.terminal_reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskLifecycle":
        if set(value) != {"status", "usage", "blocker", "terminal_reason"}:
            raise ValueError("TaskLifecycle fields are invalid")
        raw_usage = value["usage"]
        raw_blocker = value["blocker"]
        if raw_usage is not None and not isinstance(raw_usage, Mapping):
            raise TypeError("TaskLifecycle.usage must be an object or None")
        if raw_blocker is not None and not isinstance(raw_blocker, Mapping):
            raise TypeError("TaskLifecycle.blocker must be an object or None")
        return cls(
            status=TaskStatus(value["status"]),
            usage=(
                ModelUsage.from_mapping(raw_usage) if raw_usage is not None else None
            ),
            blocker=(
                TaskBlocker.from_dict(raw_blocker)
                if raw_blocker is not None
                else None
            ),
            terminal_reason=value["terminal_reason"],
        )


@dataclass(frozen=True, slots=True)
class Task:
    """Immutable goal definition committed as one ``task.created`` record."""

    task_id: str
    objective: str
    parent_task_id: str | None = None
    success_criteria: tuple[str, ...] = ()
    constraints: Mapping[str, str] = field(default_factory=dict)
    references: tuple[TaskReference, ...] = ()
    budget: TaskBudget = field(default_factory=TaskBudget)
    created_at: str = field(default_factory=_utc_now_iso)
    created_by_run_id: str | None = None
    plan_assignment: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("Task.task_id must be a non-empty string")
        if not isinstance(self.objective, str) or not self.objective.strip():
            raise ValueError("Task.objective must be a non-empty string")
        _optional_text(self.parent_task_id, "Task.parent_task_id")
        _optional_text(self.created_by_run_id, "Task.created_by_run_id")
        _optional_text(self.plan_assignment, "Task.plan_assignment")
        if not isinstance(self.success_criteria, tuple) or any(
            not isinstance(item, str) or not item.strip()
            for item in self.success_criteria
        ):
            raise TypeError(
                "Task.success_criteria must be a tuple of non-empty strings"
            )
        if not isinstance(self.constraints, Mapping) or any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in self.constraints.items()
        ):
            raise TypeError("Task.constraints must map strings to strings")
        object.__setattr__(
            self, "constraints", MappingProxyType(dict(self.constraints))
        )
        if not isinstance(self.references, tuple) or any(
            not isinstance(item, TaskReference) for item in self.references
        ):
            raise TypeError(
                "Task.references must be a tuple of TaskReference values"
            )
        if not isinstance(self.budget, TaskBudget):
            raise TypeError("Task.budget must be a TaskBudget")
        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ValueError("Task.created_at must be an ISO-8601 UTC timestamp")
        try:
            parsed = datetime.fromisoformat(self.created_at)
        except ValueError as exc:
            raise ValueError(
                "Task.created_at must be an ISO-8601 UTC timestamp"
            ) from exc
        if parsed.tzinfo is None:
            raise ValueError("Task.created_at must carry a UTC offset")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "parent_task_id": self.parent_task_id,
            "objective": self.objective,
            "success_criteria": list(self.success_criteria),
            "constraints": dict(self.constraints),
            "references": [item.to_dict() for item in self.references],
            "budget": self.budget.to_dict(),
            "created_at": self.created_at,
            "created_by_run_id": self.created_by_run_id,
            "plan_assignment": self.plan_assignment,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Task":
        expected = {
            "task_id",
            "parent_task_id",
            "objective",
            "success_criteria",
            "constraints",
            "references",
            "budget",
            "created_at",
            "created_by_run_id",
            "plan_assignment",
        }
        if set(value) != expected:
            raise ValueError("Task fields are invalid")
        raw_criteria = value["success_criteria"]
        raw_constraints = value["constraints"]
        raw_references = value["references"]
        raw_budget = value["budget"]
        if not isinstance(raw_criteria, list):
            raise TypeError("Task.success_criteria must be an array")
        if not isinstance(raw_constraints, Mapping):
            raise TypeError("Task.constraints must be an object")
        if not isinstance(raw_references, list) or any(
            not isinstance(item, Mapping) for item in raw_references
        ):
            raise TypeError("Task.references must be an array of objects")
        if not isinstance(raw_budget, Mapping):
            raise TypeError("Task.budget must be an object")
        return cls(
            task_id=value["task_id"],
            parent_task_id=value["parent_task_id"],
            objective=value["objective"],
            success_criteria=tuple(raw_criteria),
            constraints=dict(raw_constraints),
            references=tuple(
                TaskReference.from_dict(item) for item in raw_references
            ),
            budget=TaskBudget.from_dict(raw_budget),
            created_at=value["created_at"],
            created_by_run_id=value["created_by_run_id"],
            plan_assignment=value["plan_assignment"],
        )


def validate_task_transition(
    from_status: TaskStatus, to_status: TaskStatus
) -> None:
    """Raise ``ValueError`` when one lifecycle move is not a legal transition.

    An active Task may block or terminate; a blocked Task may terminate or
    return to active only through an explicit unblock; terminal statuses are
    final.
    """

    if not isinstance(from_status, TaskStatus) or not isinstance(
        to_status, TaskStatus
    ):
        raise TypeError("task transitions use TaskStatus values")
    if from_status.terminal:
        raise ValueError(f"terminal task status {from_status.value!r} is final")
    if to_status is TaskStatus.ACTIVE:
        if from_status is not TaskStatus.BLOCKED:
            raise ValueError("only a blocked task can return to active")
    elif to_status is TaskStatus.BLOCKED:
        if from_status is not TaskStatus.ACTIVE:
            raise ValueError("only an active task can block")


__all__ = [
    "Task",
    "TaskBlocker",
    "TaskBudget",
    "TaskLifecycle",
    "TaskReference",
    "TaskStatus",
    "validate_task_transition",
]
