"""Agent specification and registry for multi-agent delegation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, TypeVar

if TYPE_CHECKING:
    from .agent_module import AgentModule


SourceStateT = TypeVar("SourceStateT")
TargetStateT = TypeVar("TargetStateT")


class ContextStrategy(str, Enum):
    """Controls what context a sub-agent receives from its parent."""

    FULL = "full"
    SUMMARY = "summary"
    ISOLATED = "isolated"


@dataclass
class HandoffContext:
    """Context package passed between agents during delegation."""

    strategy: ContextStrategy = ContextStrategy.SUMMARY
    payload: Dict[str, Any] = field(default_factory=dict)
    shared_state_fields: List[str] = field(default_factory=list)
    max_history_rounds: Optional[int] = None


@dataclass
class StateAdapter(ABC, Generic[SourceStateT, TargetStateT]):
    """Converts state between agents with different StateT types during handoff."""

    @abstractmethod
    def adapt(self, source: SourceStateT) -> TargetStateT:
        """Convert source state to the target state type."""


@dataclass
class AgentSpec:
    """Describes an agent available for delegation."""

    name: str
    description: str
    agent: AgentModule
    context_strategy: ContextStrategy = ContextStrategy.SUMMARY
    max_steps_override: Optional[int] = None
    shared_env: bool = True
    state_adapter: Optional[StateAdapter] = None
    handoff_context: Optional[HandoffContext] = None
    model_override: Optional[str] = None
    tools_override: Optional[Any] = None  # ToolRegistry instance
    tool_name: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("AgentSpec.name must be non-empty")
        if self.tool_name is not None:
            normalized_tool_name = str(self.tool_name).strip()
            if not normalized_tool_name:
                raise ValueError("AgentSpec.tool_name must be non-empty when set")
            self.tool_name = normalized_tool_name


class AgentRegistry:
    """Manages agents available for delegation within a run."""

    def __init__(self) -> None:
        self._specs: Dict[str, AgentSpec] = {}

    def register(self, spec: AgentSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Agent '{spec.name}' is already registered")
        self._specs[spec.name] = spec

    def resolve(self, name: str) -> AgentSpec:
        if name not in self._specs:
            raise KeyError(f"Agent '{name}' not found in registry")
        return self._specs[name]

    def list_available(self) -> List[AgentSpec]:
        return list(self._specs.values())
