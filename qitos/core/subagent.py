"""Typed contracts for independently stateful Subagent runs."""

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

DEFAULT_SUBAGENT_MAX_STEPS = 200
SubagentInvocationCleanup = Callable[[], Awaitable[None]]
SubagentPostRuntimeEvent = Callable[[RuntimeInput], Awaitable[bool]]
SubagentCancellationCheck = Callable[[], bool]


class SubagentPersistenceError(RuntimeError):
    """Raised when a Subagent lifecycle fact cannot be durably recorded."""


class SubagentInvocationCancelled(RuntimeError):
    """Signal that one Subagent invocation ended without cancelling its caller."""


class SubagentRunLimitError(RuntimeError):
    """Raised when a root Run's shared Subagent limit rejects a launch."""


class SubagentStatus(str, Enum):
    """Stable lifecycle state for one subagent handle."""

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
            SubagentStatus.COMPLETED,
            SubagentStatus.BLOCKED,
            SubagentStatus.BUDGET_EXHAUSTED,
            SubagentStatus.FAILED,
            SubagentStatus.CANCELLED,
            SubagentStatus.INTERRUPTED,
            SubagentStatus.UNKNOWN,
        }


@dataclass(frozen=True, slots=True)
class SubagentHandle:
    """Opaque subagent identity scoped to its exact parent Run."""

    subagent_id: str
    parent_run_id: str

    def __post_init__(self) -> None:
        for name in ("subagent_id", "parent_run_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"SubagentHandle.{name} must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        return {
            "subagent_id": self.subagent_id,
            "parent_run_id": self.parent_run_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SubagentHandle":
        fields = set(value)
        if fields == {"subagent_id", "parent_run_id"}:
            subagent_id = value["subagent_id"]
        elif fields == {"child_id", "parent_run_id"}:
            subagent_id = value["child_id"]
        else:
            raise ValueError("SubagentHandle fields are invalid")
        return cls(
            subagent_id=subagent_id,
            parent_run_id=value["parent_run_id"],
        )


@dataclass(frozen=True, slots=True)
class SubagentLaunchRequest:
    """One bounded subagent assignment without a live Agent object."""

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
        default_factory=lambda: TaskBudget(max_steps=DEFAULT_SUBAGENT_MAX_STEPS)
    )
    parent_task_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("task", "description", "agent_type", "profile"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"SubagentLaunchRequest.{name} must be a non-empty string"
                )
        for name in ("name", "context"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"SubagentLaunchRequest.{name} must be a string")
        if not isinstance(self.success_criteria, tuple) or any(
            not isinstance(item, str) or not item.strip()
            for item in self.success_criteria
        ):
            raise TypeError(
                "SubagentLaunchRequest.success_criteria must contain non-empty strings"
            )
        if not isinstance(self.constraints, Mapping) or any(
            not isinstance(key, str)
            or not isinstance(item, str)
            for key, item in self.constraints.items()
        ):
            raise TypeError(
                "SubagentLaunchRequest.constraints must map strings to strings"
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
                "SubagentLaunchRequest.references must contain TaskReference values"
            )
        if self.permission_context is not None:
            if not isinstance(self.permission_context, ToolPermissionContext):
                raise TypeError(
                    "SubagentLaunchRequest.permission_context must be a "
                    "ToolPermissionContext or None"
                )
            object.__setattr__(
                self,
                "permission_context",
                ToolPermissionContext.from_dict(self.permission_context.to_dict()),
            )
        if self.parent_task_id is not None and (
            not isinstance(self.parent_task_id, str)
            or not self.parent_task_id.strip()
        ):
            raise ValueError(
                "SubagentLaunchRequest.parent_task_id must be a non-empty string or None"
            )
        if not isinstance(self.allowed_tool_groups, tuple) or any(
            not isinstance(group, str) or not group.strip()
            for group in self.allowed_tool_groups
        ):
            raise TypeError(
                "SubagentLaunchRequest.allowed_tool_groups must contain non-empty strings"
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
                "SubagentLaunchRequest.working_directory must be a non-empty string or None"
            )
        if not isinstance(self.budget, TaskBudget):
            raise TypeError("SubagentLaunchRequest.budget must be a TaskBudget")

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
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SubagentLaunchRequest":
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
        }
        if set(value) not in (expected, expected | {"plan_assignment"}):
            raise ValueError("SubagentLaunchRequest fields are invalid")
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
        )


@dataclass(frozen=True, slots=True)
class SubagentLaunchContext:
    """Validated parent-Run values retained while one Subagent is supervised."""

    parent_run_id: str
    delegate_depth: int = 0
    max_subagents: int = 0
    deadline_monotonic: float | None = None
    budget_ledger: BudgetLedger | None = None
    journal: SessionJournal | None = None
    post_runtime_event: SubagentPostRuntimeEvent | None = None
    parent_tool_authority: ToolExposure | None = None
    parent_permission_context: ToolPermissionContext | None = None
    parent_history: tuple[object, ...] = ()
    parent_history_snapshot: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.parent_run_id, str) or not self.parent_run_id.strip():
            raise ValueError("SubagentLaunchContext.parent_run_id must be non-empty")
        for name, value in (
            ("delegate_depth", self.delegate_depth),
            ("max_subagents", self.max_subagents),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"SubagentLaunchContext.{name} must be non-negative")
        deadline = self.deadline_monotonic
        if deadline is not None and (
            isinstance(deadline, bool)
            or not isinstance(deadline, (int, float))
            or not math.isfinite(float(deadline))
            or deadline < 0
        ):
            raise ValueError(
                "SubagentLaunchContext.deadline_monotonic must be finite and non-negative"
            )
        if deadline is not None:
            object.__setattr__(self, "deadline_monotonic", float(deadline))
        if self.budget_ledger is not None and not isinstance(
            self.budget_ledger, BudgetLedger
        ):
            raise TypeError(
                "SubagentLaunchContext.budget_ledger must be a BudgetLedger or None"
            )
        if self.journal is not None and not isinstance(self.journal, SessionJournal):
            raise TypeError(
                "SubagentLaunchContext.journal must implement SessionJournal or be None"
            )
        if self.post_runtime_event is not None and not callable(
            self.post_runtime_event
        ):
            raise TypeError(
                "SubagentLaunchContext.post_runtime_event must be callable or None"
            )
        if self.parent_tool_authority is not None and not isinstance(
            self.parent_tool_authority, ToolExposure
        ):
            raise TypeError(
                "SubagentLaunchContext.parent_tool_authority must be a "
                "ToolExposure or None"
            )
        if self.parent_permission_context is not None:
            if not isinstance(self.parent_permission_context, ToolPermissionContext):
                raise TypeError(
                    "SubagentLaunchContext.parent_permission_context must be a "
                    "ToolPermissionContext or None"
                )
            # The parent Tool turn is immutable. Retain a value snapshot rather
            # than a mutable policy object that another turn can widen after
            # Subagent admission.
            object.__setattr__(
                self,
                "parent_permission_context",
                ToolPermissionContext.from_dict(
                    self.parent_permission_context.to_dict()
                ),
            )
        if not isinstance(self.parent_history, tuple):
            raise TypeError("SubagentLaunchContext.parent_history must be a tuple")


@dataclass(frozen=True, slots=True)
class SubagentRuntimeContext:
    """Immutable invocation context created after durable Subagent admission."""

    launch: SubagentLaunchContext
    handle: SubagentHandle
    subagent_run_id: str
    cancellation_requested: SubagentCancellationCheck

    def __post_init__(self) -> None:
        if not isinstance(self.launch, SubagentLaunchContext):
            raise TypeError("SubagentRuntimeContext.launch must be a SubagentLaunchContext")
        if not isinstance(self.handle, SubagentHandle):
            raise TypeError("SubagentRuntimeContext.handle must be a SubagentHandle")
        if not isinstance(self.subagent_run_id, str) or not self.subagent_run_id.strip():
            raise ValueError("SubagentRuntimeContext.subagent_run_id must be non-empty")
        if not callable(self.cancellation_requested):
            raise TypeError(
                "SubagentRuntimeContext.cancellation_requested must be callable"
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


class SubagentStateView(Protocol):
    @property
    def final_result(self) -> Any:
        raise NotImplementedError

    @property
    def stop_reason(self) -> Any:
        raise NotImplementedError


class SubagentRunResult(Protocol):
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
    def state(self) -> SubagentStateView:
        """Return the Subagent's terminal state through the minimal read view."""
        ...

    @property
    def records(self) -> Sequence[Any]:
        """Return the completed steps as a covariant read-only view."""
        ...


class SubagentEngine(Protocol):
    """Minimal async Engine surface owned by a subagent supervisor."""

    @property
    def active_run_id(self) -> str:
        raise NotImplementedError

    async def arun(self, task: str, **kwargs: Any) -> SubagentRunResult:
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release an idle Engine, including one that never entered ``arun``."""
        ...

    def cancel(self, mode: str) -> None:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class SubagentInvocation:
    """A fresh subagent Engine and its exact immutable run arguments."""

    engine: SubagentEngine
    task: str
    run_kwargs: Mapping[str, Any] = field(default_factory=dict)
    cleanup: SubagentInvocationCleanup | None = None
    conclusion_factory: (
        Callable[[SubagentRunResult], Awaitable["AgentConclusion"]] | None
    ) = None

    def __post_init__(self) -> None:
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("SubagentInvocation.task must be a non-empty string")
        if not isinstance(self.run_kwargs, Mapping):
            raise TypeError("SubagentInvocation.run_kwargs must be a mapping")
        if self.cleanup is not None and not callable(self.cleanup):
            raise TypeError("SubagentInvocation.cleanup must be async callable or None")
        if self.conclusion_factory is not None and not callable(
            self.conclusion_factory
        ):
            raise TypeError(
                "SubagentInvocation.conclusion_factory must be async callable or None"
            )
        object.__setattr__(
            self,
            "run_kwargs",
            MappingProxyType(dict(self.run_kwargs)),
        )


@dataclass(frozen=True, slots=True)
class AgentConclusion:
    """Bounded subagent conclusion; Journal and Artifacts remain evidence truth."""

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
class SubagentResult:
    """Terminal or current projection for one subagent handle."""

    handle: SubagentHandle
    request: SubagentLaunchRequest
    status: SubagentStatus
    conclusion: AgentConclusion = field(default_factory=AgentConclusion)
    subagent_run_id: str = ""
    error: str | None = None
    steps: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    usage_complete: bool = False
    cost_complete: bool = False
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.handle, SubagentHandle):
            raise TypeError("SubagentResult.handle must be a SubagentHandle")
        if not isinstance(self.request, SubagentLaunchRequest):
            raise TypeError("SubagentResult.request must be a SubagentLaunchRequest")
        if not isinstance(self.status, SubagentStatus):
            raise TypeError("SubagentResult.status must be a SubagentStatus")
        if not isinstance(self.conclusion, AgentConclusion):
            raise TypeError("SubagentResult.conclusion must be an AgentConclusion")
        if not isinstance(self.subagent_run_id, str):
            raise TypeError("SubagentResult.subagent_run_id must be a string")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("SubagentResult.error must be a string or None")
        for name in ("steps", "total_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"SubagentResult.{name} must be non-negative")
        if (
            isinstance(self.total_cost_usd, bool)
            or not isinstance(self.total_cost_usd, (int, float))
            or not math.isfinite(float(self.total_cost_usd))
            or self.total_cost_usd < 0
        ):
            raise ValueError(
                "SubagentResult.total_cost_usd must be finite and non-negative"
            )
        for name in ("usage_complete", "cost_complete"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"SubagentResult.{name} must be a boolean")
        if (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, (int, float))
            or not math.isfinite(float(self.elapsed_seconds))
            or self.elapsed_seconds < 0
        ):
            raise ValueError(
                "SubagentResult.elapsed_seconds must be finite and non-negative"
            )

    @property
    def ready(self) -> bool:
        return self.status.terminal

    @property
    def succeeded(self) -> bool:
        return self.status is SubagentStatus.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        return {
            "handle": self.handle.to_dict(),
            "request": self.request.to_dict(),
            "status": self.status.value,
            "ready": self.ready,
            "conclusion": self.conclusion.to_dict(),
            "subagent_run_id": self.subagent_run_id,
            "error": self.error,
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "total_cost_usd": float(self.total_cost_usd),
            "usage_complete": self.usage_complete,
            "cost_complete": self.cost_complete,
            "elapsed_seconds": float(self.elapsed_seconds),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SubagentResult":
        base_fields = {
            "handle",
            "request",
            "status",
            "ready",
            "conclusion",
            "subagent_run_id",
            "error",
            "steps",
            "total_tokens",
            "elapsed_seconds",
        }
        expected = base_fields | {
            "total_cost_usd",
            "usage_complete",
            "cost_complete",
        }
        raw_value = dict(value)
        fields = set(raw_value)
        legacy_base_fields = base_fields.difference({"subagent_run_id"}) | {
            "child_run_id"
        }
        legacy_expected = expected.difference({"subagent_run_id"}) | {
            "child_run_id"
        }
        if fields in (legacy_base_fields, legacy_expected):
            raw_value["subagent_run_id"] = raw_value.pop("child_run_id")
            fields = set(raw_value)
        if fields not in (base_fields, expected):
            raise ValueError("SubagentResult fields are invalid")
        raw_handle = raw_value["handle"]
        raw_request = raw_value["request"]
        raw_conclusion = raw_value["conclusion"]
        if not isinstance(raw_handle, Mapping):
            raise TypeError("SubagentResult.handle must be an object")
        if not isinstance(raw_conclusion, Mapping):
            raise TypeError("SubagentResult.conclusion must be an object")
        if not isinstance(raw_request, Mapping):
            raise TypeError("SubagentResult.request must be an object")
        result = cls(
            handle=SubagentHandle.from_dict(raw_handle),
            request=SubagentLaunchRequest.from_dict(raw_request),
            status=SubagentStatus(raw_value["status"]),
            conclusion=AgentConclusion.from_dict(raw_conclusion),
            subagent_run_id=raw_value["subagent_run_id"],
            error=raw_value["error"],
            steps=raw_value["steps"],
            total_tokens=raw_value["total_tokens"],
            total_cost_usd=raw_value.get("total_cost_usd", 0.0),
            usage_complete=raw_value.get("usage_complete", False),
            cost_complete=raw_value.get("cost_complete", False),
            elapsed_seconds=raw_value["elapsed_seconds"],
        )
        if raw_value["ready"] is not result.ready:
            raise ValueError("SubagentResult.ready does not match status")
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
    "DEFAULT_SUBAGENT_MAX_STEPS",
    "AgentConclusion",
    "SubagentEngine",
    "SubagentCancellationCheck",
    "SubagentHandle",
    "SubagentInvocation",
    "SubagentInvocationCancelled",
    "SubagentInvocationCleanup",
    "SubagentLaunchContext",
    "SubagentLaunchRequest",
    "SubagentPostRuntimeEvent",
    "SubagentPersistenceError",
    "SubagentRunLimitError",
    "SubagentResult",
    "SubagentRunResult",
    "SubagentRuntimeContext",
    "SubagentStateView",
    "SubagentStatus",
]
