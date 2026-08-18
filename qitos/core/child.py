"""Typed contracts for independently stateful child Agent runs."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol

from .budget import BudgetLedger
from .journal import JournalRecordRef, SessionJournal
from .runtime_input import RuntimeInput
from .task import TaskBudget, TaskReference
from .tool import ToolPermissionContext
from .tool_registry import ToolExposure

DEFAULT_CHILD_MAX_STEPS = 200
ChildInvocationCleanup = Callable[[], Awaitable[None]]
ChildPostRuntimeEvent = Callable[[RuntimeInput], Awaitable[bool]]
ChildCancellationCheck = Callable[[], bool]


class ChildPersistenceError(RuntimeError):
    """Raised when a Child lifecycle fact cannot be durably recorded."""


class ChildInvocationCancelled(RuntimeError):
    """Signal that one Child invocation ended without cancelling its caller."""


class ChildRunLimitError(RuntimeError):
    """Raised when a root Run's shared Child limit rejects a launch."""


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
    success_criteria: tuple[str, ...] = ()
    constraints: Mapping[str, str] = field(default_factory=dict)
    references: tuple[TaskReference, ...] = ()
    permission_context: ToolPermissionContext | None = None
    profile: str = "default"
    allowed_tool_groups: tuple[str, ...] = ()
    working_directory: str | None = None
    budget: TaskBudget = field(
        default_factory=lambda: TaskBudget(max_steps=DEFAULT_CHILD_MAX_STEPS)
    )
    parent_task_id: str | None = None
    plan_assignment: str | None = None

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
        if not isinstance(self.success_criteria, tuple) or any(
            not isinstance(item, str) or not item.strip()
            for item in self.success_criteria
        ):
            raise TypeError(
                "ChildLaunchRequest.success_criteria must contain non-empty strings"
            )
        if not isinstance(self.constraints, Mapping) or any(
            not isinstance(key, str)
            or not isinstance(item, str)
            for key, item in self.constraints.items()
        ):
            raise TypeError(
                "ChildLaunchRequest.constraints must map strings to strings"
            )
        object.__setattr__(
            self,
            "constraints",
            MappingProxyType(dict(self.constraints)),
        )
        if not isinstance(self.references, tuple) or any(
            not isinstance(item, TaskReference) for item in self.references
        ):
            raise TypeError(
                "ChildLaunchRequest.references must contain TaskReference values"
            )
        if self.permission_context is not None:
            if not isinstance(self.permission_context, ToolPermissionContext):
                raise TypeError(
                    "ChildLaunchRequest.permission_context must be a "
                    "ToolPermissionContext or None"
                )
            object.__setattr__(
                self,
                "permission_context",
                ToolPermissionContext.from_dict(self.permission_context.to_dict()),
            )
        for name in ("parent_task_id", "plan_assignment"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(
                    f"ChildLaunchRequest.{name} must be a non-empty string or None"
                )
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
            "success_criteria": list(self.success_criteria),
            "constraints": dict(self.constraints),
            "references": [item.to_dict() for item in self.references],
            "permission_context": (
                None
                if self.permission_context is None
                else self.permission_context.to_dict()
            ),
            "profile": self.profile,
            "allowed_tool_groups": list(self.allowed_tool_groups),
            "working_directory": self.working_directory,
            "budget": _budget_to_dict(self.budget),
            "parent_task_id": self.parent_task_id,
            "plan_assignment": self.plan_assignment,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ChildLaunchRequest":
        expected = {
            "task",
            "description",
            "name",
            "agent_type",
            "context",
            "success_criteria",
            "constraints",
            "references",
            "permission_context",
            "profile",
            "allowed_tool_groups",
            "working_directory",
            "budget",
            "parent_task_id",
            "plan_assignment",
        }
        if set(value) != expected:
            raise ValueError("ChildLaunchRequest fields are invalid")
        raw_groups = value["allowed_tool_groups"]
        raw_budget = value["budget"]
        raw_criteria = value["success_criteria"]
        raw_constraints = value["constraints"]
        raw_references = value["references"]
        raw_permission = value["permission_context"]
        if not isinstance(raw_groups, list):
            raise TypeError("allowed_tool_groups must be an array")
        if not isinstance(raw_budget, Mapping):
            raise TypeError("budget must be an object")
        if not isinstance(raw_criteria, list):
            raise TypeError("success_criteria must be an array")
        if not isinstance(raw_constraints, Mapping):
            raise TypeError("constraints must be an object")
        if not isinstance(raw_references, list) or any(
            not isinstance(item, Mapping) for item in raw_references
        ):
            raise TypeError("references must be an array of objects")
        if raw_permission is not None and not isinstance(raw_permission, Mapping):
            raise TypeError("permission_context must be an object or null")
        return cls(
            task=value["task"],
            description=value["description"],
            name=value["name"],
            agent_type=value["agent_type"],
            context=value["context"],
            success_criteria=tuple(raw_criteria),
            constraints=dict(raw_constraints),
            references=tuple(
                TaskReference.from_dict(item) for item in raw_references
            ),
            permission_context=(
                None
                if raw_permission is None
                else ToolPermissionContext.from_dict(dict(raw_permission))
            ),
            profile=value["profile"],
            allowed_tool_groups=tuple(raw_groups),
            working_directory=value["working_directory"],
            budget=_budget_from_dict(raw_budget),
            parent_task_id=value["parent_task_id"],
            plan_assignment=value["plan_assignment"],
        )


@dataclass(frozen=True, slots=True)
class ChildLaunchContext:
    """Validated parent-Run values retained while one Child is supervised."""

    parent_run_id: str
    delegate_depth: int = 0
    max_children: int = 0
    deadline_monotonic: float | None = None
    budget_ledger: BudgetLedger | None = None
    journal: SessionJournal | None = None
    post_runtime_event: ChildPostRuntimeEvent | None = None
    parent_tool_authority: ToolExposure | None = None
    parent_permission_context: ToolPermissionContext | None = None
    parent_history: tuple[object, ...] = ()
    parent_history_snapshot: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.parent_run_id, str) or not self.parent_run_id.strip():
            raise ValueError("ChildLaunchContext.parent_run_id must be non-empty")
        for name, value in (
            ("delegate_depth", self.delegate_depth),
            ("max_children", self.max_children),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"ChildLaunchContext.{name} must be non-negative")
        deadline = self.deadline_monotonic
        if deadline is not None and (
            isinstance(deadline, bool)
            or not isinstance(deadline, (int, float))
            or not math.isfinite(float(deadline))
            or deadline < 0
        ):
            raise ValueError(
                "ChildLaunchContext.deadline_monotonic must be finite and non-negative"
            )
        if deadline is not None:
            object.__setattr__(self, "deadline_monotonic", float(deadline))
        if self.budget_ledger is not None and not isinstance(
            self.budget_ledger, BudgetLedger
        ):
            raise TypeError(
                "ChildLaunchContext.budget_ledger must be a BudgetLedger or None"
            )
        if self.journal is not None and not isinstance(self.journal, SessionJournal):
            raise TypeError(
                "ChildLaunchContext.journal must implement SessionJournal or be None"
            )
        if self.post_runtime_event is not None and not callable(
            self.post_runtime_event
        ):
            raise TypeError(
                "ChildLaunchContext.post_runtime_event must be callable or None"
            )
        if self.parent_tool_authority is not None and not isinstance(
            self.parent_tool_authority, ToolExposure
        ):
            raise TypeError(
                "ChildLaunchContext.parent_tool_authority must be a "
                "ToolExposure or None"
            )
        if self.parent_permission_context is not None:
            if not isinstance(self.parent_permission_context, ToolPermissionContext):
                raise TypeError(
                    "ChildLaunchContext.parent_permission_context must be a "
                    "ToolPermissionContext or None"
                )
            # The parent Tool turn is immutable. Retain a value snapshot rather
            # than a mutable policy object that another turn can widen after
            # Child admission.
            object.__setattr__(
                self,
                "parent_permission_context",
                ToolPermissionContext.from_dict(
                    self.parent_permission_context.to_dict()
                ),
            )
        if not isinstance(self.parent_history, tuple):
            raise TypeError("ChildLaunchContext.parent_history must be a tuple")


@dataclass(frozen=True, slots=True)
class ChildRuntimeContext:
    """Immutable invocation context created after durable Child admission."""

    launch: ChildLaunchContext
    handle: ChildHandle
    child_run_id: str
    cancellation_requested: ChildCancellationCheck

    def __post_init__(self) -> None:
        if not isinstance(self.launch, ChildLaunchContext):
            raise TypeError("ChildRuntimeContext.launch must be a ChildLaunchContext")
        if not isinstance(self.handle, ChildHandle):
            raise TypeError("ChildRuntimeContext.handle must be a ChildHandle")
        if not isinstance(self.child_run_id, str) or not self.child_run_id.strip():
            raise ValueError("ChildRuntimeContext.child_run_id must be non-empty")
        if not callable(self.cancellation_requested):
            raise TypeError(
                "ChildRuntimeContext.cancellation_requested must be callable"
            )

    @property
    def parent_run_id(self) -> str:
        return self.launch.parent_run_id

    @property
    def delegate_depth(self) -> int:
        return self.launch.delegate_depth

    @property
    def deadline_monotonic(self) -> float | None:
        return self.launch.deadline_monotonic

    @property
    def budget_ledger(self) -> BudgetLedger | None:
        return self.launch.budget_ledger

    @property
    def parent_tool_authority(self) -> ToolExposure | None:
        return self.launch.parent_tool_authority

    @property
    def parent_permission_context(self) -> ToolPermissionContext | None:
        return self.launch.parent_permission_context

    @property
    def parent_history(self) -> tuple[object, ...]:
        return self.launch.parent_history

    @property
    def parent_history_snapshot(self) -> object | None:
        return self.launch.parent_history_snapshot


class ChildStateView(Protocol):
    @property
    def final_result(self) -> Any:
        raise NotImplementedError

    @property
    def stop_reason(self) -> Any:
        raise NotImplementedError


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
        raise NotImplementedError

    async def arun(self, task: str, **kwargs: Any) -> ChildRunResult:
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release an idle Engine, including one that never entered ``arun``."""
        ...

    def cancel(self, mode: str) -> None:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ChildInvocation:
    """A fresh child Engine and its exact immutable run arguments."""

    engine: ChildEngine
    task: str
    run_kwargs: Mapping[str, Any] = field(default_factory=dict)
    cleanup: ChildInvocationCleanup | None = None
    conclusion_factory: (
        Callable[[ChildRunResult], Awaitable["AgentConclusion"]] | None
    ) = None

    def __post_init__(self) -> None:
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("ChildInvocation.task must be a non-empty string")
        if not isinstance(self.run_kwargs, Mapping):
            raise TypeError("ChildInvocation.run_kwargs must be a mapping")
        if self.cleanup is not None and not callable(self.cleanup):
            raise TypeError("ChildInvocation.cleanup must be async callable or None")
        if self.conclusion_factory is not None and not callable(
            self.conclusion_factory
        ):
            raise TypeError(
                "ChildInvocation.conclusion_factory must be async callable or None"
            )
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
    resource_refs: tuple[str, ...] = ()
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
        for name in ("resource_refs", "failure_paths", "unknowns", "next_steps"):
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
            "resource_refs": list(self.resource_refs),
            "failure_paths": list(self.failure_paths),
            "unknowns": list(self.unknowns),
            "next_steps": list(self.next_steps),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AgentConclusion":
        expected = {
            "summary",
            "evidence",
            "resource_refs",
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
            resource_refs=_string_tuple(value["resource_refs"], "resource_refs"),
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
            raise ValueError(
                "ChildResult.elapsed_seconds must be finite and non-negative"
            )

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
    return budget.to_dict()


def _budget_from_dict(value: Mapping[str, Any]) -> TaskBudget:
    return TaskBudget.from_dict(value)


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"AgentConclusion.{name} must be an array")
    return tuple(value)


__all__ = [
    "DEFAULT_CHILD_MAX_STEPS",
    "AgentConclusion",
    "ChildEngine",
    "ChildCancellationCheck",
    "ChildHandle",
    "ChildInvocation",
    "ChildInvocationCancelled",
    "ChildInvocationCleanup",
    "ChildLaunchContext",
    "ChildLaunchRequest",
    "ChildPostRuntimeEvent",
    "ChildPersistenceError",
    "ChildRunLimitError",
    "ChildResult",
    "ChildRunResult",
    "ChildRuntimeContext",
    "ChildStateView",
    "ChildStatus",
]
