"""Action protocol for QitOS kernel."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, Optional


class ActionStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    RUNNING = "running"
    ERROR = "error"
    SKIPPED = "skipped"
    DENIED = "denied"
    NEEDS_INPUT = "needs_input"
    NEEDS_APPROVAL = "needs_approval"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


@dataclass
class Action:
    """Normalized action contract emitted by policy and consumed by executor."""

    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    action_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Action":
        return cls(
            name=payload.get("name", ""),
            args=payload.get("args", {}),
            action_id=payload.get("action_id"),
            metadata=payload.get("metadata", {}),
        )


@dataclass
class ActionResult:
    """Standardized action execution result."""

    name: str
    status: ActionStatus
    output: Any = None
    error: Optional[str] = None
    action_id: Optional[str] = None
    attempts: int = 1
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionExecutionPolicy:
    """Executor policy for action batches."""

    mode: str = "serial"  # serial | parallel
    fail_fast: bool = False
    max_concurrency: int = 4
    # ``None`` preserves explicit ``ToolSpec.concurrency_safe`` declarations.
    # A caller may restrict parallel execution to a smaller set without
    # changing the registered tool metadata.
    parallel_tool_names: FrozenSet[str] | None = None
