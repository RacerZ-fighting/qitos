"""Typed contracts for independently stateful child Agent runs."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol

from .journal import JournalRecordRef
from .task import TaskBudget

DEFAULT_CHILD_MAX_STEPS = 200
ChildInvocationCleanup = Callable[[], Awaitable[None]]


class ChildPersistenceError(RuntimeError):
    """Raised when a Child lifecycle fact cannot be durably recorded."""


class ChildStatus(str, Enum):
    """Stable lifecycle state for one child handle."""

    PENDING = "pending"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    BUDGET_EXHAUSTED = "budget_exhausted"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    UNKNOWN = "unknown"

    @property
    def terminal(self) -> bool:
        return self in {
            ChildStatus.COMPLETED,
            ChildStatus.BLOCKED,
            ChildStatus.BUDGET_EXHAUSTED,
            ChildStatus.FAILED,
            ChildStatus.CANCELLED,
            ChildStatus.INTERRUPTED,
            ChildStatus.UNKNOWN,
        }


@dataclass(frozen=True, slots=True)
class ChildHandle:
    """Opaque child identity scoped to its exact parent Run."""

    child_id: str
    parent_run_id: str

    def __post_init__(self) -> None:
        for name in ("child_id", "parent_run_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"ChildHandle.{name} must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        return {
            "child_id": self.child_id,
            "parent_run_id": self.parent_run_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ChildHandle":
        if set(value) != {"child_id", "parent_run_id"}:
            raise ValueError("ChildHandle fields are invalid")
        return cls(
            child_id=value["child_id"],
            parent_run_id=value["parent_run_id"],
        )


@dataclass(frozen=True, slots=True)
class ChildLaunchRequest:
    """One bounded child assignment without a live Agent object."""

    task: str
    description: str
    name: str = ""
    agent_type: str = "general-purpose"
    context: str = ""
    profile: str = "default"
    allowed_tool_groups: tuple[str, ...] = ()
    working_directory: str | None = None
    budget: TaskBudget = field(
        default_factory=lambda: TaskBudget(max_steps=DEFAULT_CHILD_MAX_STEPS)
    )

    def __post_init__(self) -> None:
        for name in ("task", "description", "agent_type", "profile"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"ChildLaunchRequest.{name} must be a non-empty string"
                )
        for name in ("name", "context"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"ChildLaunchRequest.{name} must be a string")
        if not isinstance(self.allowed_tool_groups, tuple) or any(
            not isinstance(group, str) or not group.strip()
            for group in self.allowed_tool_groups
        ):
            raise TypeError(
                "ChildLaunchRequest.allowed_tool_groups must contain non-empty strings"
            )
        normalized_groups = tuple(
            dict.fromkeys(group.strip() for group in self.allowed_tool_groups)
        )
        object.__setattr__(self, "allowed_tool_groups", normalized_groups)
        if self.working_directory is not None and (
            not isinstance(self.working_directory, str)
            or not self.working_directory.strip()
        ):
            raise TypeError(
                "ChildLaunchRequest.working_directory must be a non-empty string or None"
            )
        if not isinstance(self.budget, TaskBudget):
            raise TypeError("ChildLaunchRequest.budget must be a TaskBudget")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "description": self.description,
            "name": self.name,
            "agent_type": self.agent_type,
            "context": self.context,
            "profile": self.profile,
            "allowed_tool_groups": list(self.allowed_tool_groups),
            "working_directory": self.working_directory,
            "budget": _budget_to_dict(self.budget),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ChildLaunchRequest":
        expected = {
            "task",
            "description",
            "name",
            "agent_type",
            "context",
            "profile",
            "allowed_tool_groups",
            "working_directory",
            "budget",
        }
        if set(value) != expected:
            raise ValueError("ChildLaunchRequest fields are invalid")
        raw_groups = value["allowed_tool_groups"]
        raw_budget = value["budget"]
        if not isinstance(raw_groups, list):
            raise TypeError("allowed_tool_groups must be an array")
        if not isinstance(raw_budget, Mapping):
            raise TypeError("budget must be an object")
        return cls(
            task=value["task"],
            description=value["description"],
            name=value["name"],
            agent_type=value["agent_type"],
            context=value["context"],
            profile=value["profile"],
            allowed_tool_groups=tuple(raw_groups),
            working_directory=value["working_directory"],
            budget=_budget_from_dict(raw_budget),
        )


class ChildStateView(Protocol):
    @property
    def final_result(self) -> Any:
        ...

    @property
    def stop_reason(self) -> Any:
        ...


class ChildRunResult(Protocol):
    step_count: int
    total_tokens: int
    total_cost_usd: float
    local_total_tokens: int
    local_total_cost_usd: float
    usage_complete: bool
    cost_complete: bool
    local_usage_complete: bool
    local_cost_complete: bool
    run_id: str

    @property
    def state(self) -> ChildStateView:
        """Return the Child's terminal state through the minimal read view."""
        ...

    @property
    def records(self) -> Sequence[Any]:
        """Return the completed steps as a covariant read-only view."""
        ...


class ChildEngine(Protocol):
    """Minimal async Engine surface owned by a child supervisor."""

    @property
    def active_run_id(self) -> str:
        ...

    async def arun(self, task: str, **kwargs: Any) -> ChildRunResult:
        ...

    def cancel(self, mode: str) -> None:
        ...


@dataclass(frozen=True, slots=True)
class ChildInvocation:
    """A fresh child Engine and its exact immutable run arguments."""

    engine: ChildEngine
    task: str
    run_kwargs: Mapping[str, Any] = field(default_factory=dict)
    cleanup: ChildInvocationCleanup | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("ChildInvocation.task must be a non-empty string")
        if not isinstance(self.run_kwargs, Mapping):
            raise TypeError("ChildInvocation.run_kwargs must be a mapping")
        if self.cleanup is not None and not callable(self.cleanup):
            raise TypeError("ChildInvocation.cleanup must be async callable or None")
        object.__setattr__(
            self,
            "run_kwargs",
            MappingProxyType(dict(self.run_kwargs)),
        )


@dataclass(frozen=True, slots=True)
class AgentConclusion:
    """Bounded child conclusion; Journal and Artifacts remain evidence truth."""

    summary: str = ""
    evidence: tuple[JournalRecordRef, ...] = ()
    failure_paths: tuple[str, ...] = ()
    unknowns: tuple[str, ...] = ()
    next_steps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.summary, str):
            raise TypeError("AgentConclusion.summary must be a string")
        if not isinstance(self.evidence, tuple) or any(
            not isinstance(item, JournalRecordRef) for item in self.evidence
        ):
            raise TypeError(
                "AgentConclusion.evidence must contain JournalRecordRef values"
            )
        for name in ("failure_paths", "unknowns", "next_steps"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(
                not isinstance(item, str) or not item.strip() for item in values
            ):
                raise TypeError(
                    f"AgentConclusion.{name} must contain non-empty strings"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "evidence": [item.to_dict() for item in self.evidence],
            "failure_paths": list(self.failure_paths),
            "unknowns": list(self.unknowns),
            "next_steps": list(self.next_steps),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AgentConclusion":
        expected = {
            "summary",
            "evidence",
            "failure_paths",
            "unknowns",
            "next_steps",
        }
        if set(value) != expected:
            raise ValueError("AgentConclusion fields are invalid")
        raw_evidence = value["evidence"]
        if not isinstance(raw_evidence, list) or any(
            not isinstance(item, Mapping) for item in raw_evidence
        ):
            raise TypeError("AgentConclusion.evidence must be an array of objects")
        return cls(
            summary=value["summary"],
            evidence=tuple(JournalRecordRef.from_dict(item) for item in raw_evidence),
            failure_paths=_string_tuple(value["failure_paths"], "failure_paths"),
            unknowns=_string_tuple(value["unknowns"], "unknowns"),
            next_steps=_string_tuple(value["next_steps"], "next_steps"),
        )


@dataclass(frozen=True, slots=True)
class ChildResult:
    """Terminal or current projection for one child handle."""

    handle: ChildHandle
    request: ChildLaunchRequest
    status: ChildStatus
    conclusion: AgentConclusion = field(default_factory=AgentConclusion)
    child_run_id: str = ""
    error: str | None = None
    steps: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    usage_complete: bool = False
    cost_complete: bool = False
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.handle, ChildHandle):
            raise TypeError("ChildResult.handle must be a ChildHandle")
        if not isinstance(self.request, ChildLaunchRequest):
            raise TypeError("ChildResult.request must be a ChildLaunchRequest")
        if not isinstance(self.status, ChildStatus):
            raise TypeError("ChildResult.status must be a ChildStatus")
        if not isinstance(self.conclusion, AgentConclusion):
            raise TypeError("ChildResult.conclusion must be an AgentConclusion")
        if not isinstance(self.child_run_id, str):
            raise TypeError("ChildResult.child_run_id must be a string")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("ChildResult.error must be a string or None")
        for name in ("steps", "total_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"ChildResult.{name} must be non-negative")
        if (
            isinstance(self.total_cost_usd, bool)
            or not isinstance(self.total_cost_usd, (int, float))
            or not math.isfinite(float(self.total_cost_usd))
            or self.total_cost_usd < 0
        ):
            raise ValueError(
                "ChildResult.total_cost_usd must be finite and non-negative"
            )
        for name in ("usage_complete", "cost_complete"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"ChildResult.{name} must be a boolean")
        if (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, (int, float))
            or not math.isfinite(float(self.elapsed_seconds))
            or self.elapsed_seconds < 0
        ):
            raise ValueError("ChildResult.elapsed_seconds must be finite and non-negative")

    @property
    def ready(self) -> bool:
        return self.status.terminal

    @property
    def succeeded(self) -> bool:
        return self.status is ChildStatus.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        return {
            "handle": self.handle.to_dict(),
            "request": self.request.to_dict(),
            "status": self.status.value,
            "ready": self.ready,
            "conclusion": self.conclusion.to_dict(),
            "child_run_id": self.child_run_id,
            "error": self.error,
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "total_cost_usd": float(self.total_cost_usd),
            "usage_complete": self.usage_complete,
            "cost_complete": self.cost_complete,
            "elapsed_seconds": float(self.elapsed_seconds),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ChildResult":
        legacy = {
            "handle",
            "request",
            "status",
            "ready",
            "conclusion",
            "child_run_id",
            "error",
            "steps",
            "total_tokens",
            "elapsed_seconds",
        }
        expected = legacy | {
            "total_cost_usd",
            "usage_complete",
            "cost_complete",
        }
        if set(value) not in (legacy, expected):
            raise ValueError("ChildResult fields are invalid")
        raw_handle = value["handle"]
        raw_request = value["request"]
        raw_conclusion = value["conclusion"]
        if not isinstance(raw_handle, Mapping):
            raise TypeError("ChildResult.handle must be an object")
        if not isinstance(raw_conclusion, Mapping):
            raise TypeError("ChildResult.conclusion must be an object")
        if not isinstance(raw_request, Mapping):
            raise TypeError("ChildResult.request must be an object")
        result = cls(
            handle=ChildHandle.from_dict(raw_handle),
            request=ChildLaunchRequest.from_dict(raw_request),
            status=ChildStatus(value["status"]),
            conclusion=AgentConclusion.from_dict(raw_conclusion),
            child_run_id=value["child_run_id"],
            error=value["error"],
            steps=value["steps"],
            total_tokens=value["total_tokens"],
            total_cost_usd=value.get("total_cost_usd", 0.0),
            usage_complete=value.get("usage_complete", False),
            cost_complete=value.get("cost_complete", False),
            elapsed_seconds=value["elapsed_seconds"],
        )
        if value["ready"] is not result.ready:
            raise ValueError("ChildResult.ready does not match status")
        return result


def _budget_to_dict(budget: TaskBudget) -> dict[str, Any]:
    return {
        "max_steps": budget.max_steps,
        "max_runtime_seconds": budget.max_runtime_seconds,
        "max_tokens": budget.max_tokens,
        "max_cost_usd": budget.max_cost_usd,
        "max_tool_concurrency": budget.max_tool_concurrency,
        "max_children": budget.max_children,
    }


def _budget_from_dict(value: Mapping[str, Any]) -> TaskBudget:
    expected = {
        "max_steps",
        "max_runtime_seconds",
        "max_tokens",
        "max_cost_usd",
        "max_tool_concurrency",
        "max_children",
    }
    if set(value) != expected:
        raise ValueError("TaskBudget fields are invalid")
    return TaskBudget(**dict(value))


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"AgentConclusion.{name} must be an array")
    return tuple(value)


__all__ = [
    "DEFAULT_CHILD_MAX_STEPS",
    "AgentConclusion",
    "ChildEngine",
    "ChildHandle",
    "ChildInvocation",
    "ChildInvocationCleanup",
    "ChildLaunchRequest",
    "ChildPersistenceError",
    "ChildResult",
    "ChildRunResult",
    "ChildStateView",
    "ChildStatus",
]
