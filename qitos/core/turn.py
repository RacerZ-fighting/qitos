"""Immutable inputs captured once for a QitOS model turn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .history import HistorySnapshot
from .env import RuntimeCapabilitySnapshot
from .model_capabilities import ModelCapabilities
from .tool_registry import ToolExposure

if TYPE_CHECKING:
    from ..models.base import Model
    from ..protocols import ModelProtocol
    from .model_response import ModelPricing


@dataclass(frozen=True, slots=True)
class TurnBudgetSnapshot:
    """One turn's immutable view of Run limits and accumulated usage."""

    step: int
    max_steps: int
    used_tokens: int
    remaining_tokens: int | None
    used_cost_usd: float
    remaining_cost_usd: float | None
    model_pricing: ModelPricing | None
    deadline_monotonic: float | None
    max_tool_concurrency: int
    max_children: int

    @property
    def remaining_steps(self) -> int:
        return max(0, self.max_steps - self.step)


@dataclass(frozen=True, slots=True)
class TurnRuntimeCapabilities:
    """Provider and runtime facts that cannot change during one turn."""

    model: ModelCapabilities
    runtime: RuntimeCapabilitySnapshot | None = None
    mailbox: bool = True
    child_agents: bool = False

    @property
    def environment_ops(self) -> tuple[str, ...]:
        """Operation groups verified by the initialized backend."""

        if self.runtime is None:
            return ()
        return self.runtime.operation_groups


@dataclass(frozen=True, slots=True)
class TurnSnapshot:
    """Exact model, history, tools, capabilities, and budget for one turn."""

    run_id: str
    step_id: int
    model: Model | None
    protocol: ModelProtocol | Any
    protocol_source: str
    history: HistorySnapshot
    tools: ToolExposure
    capabilities: TurnRuntimeCapabilities
    budget: TurnBudgetSnapshot
    terminal_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.step_id < 0:
            raise ValueError("step_id must be non-negative")
        if self.terminal_reason is not None and not self.terminal_reason.strip():
            raise ValueError("terminal_reason must be non-empty when provided")

    @property
    def is_terminal_synthesis(self) -> bool:
        """Return whether this turn can only form a terminal conclusion."""

        return self.terminal_reason is not None


__all__ = ["TurnBudgetSnapshot", "TurnRuntimeCapabilities", "TurnSnapshot"]
