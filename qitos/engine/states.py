"""FSM state and event model for the canonical QitOS engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any, Dict, List, Optional

from ..core.model_request import ModelRequest


class RuntimePhase(str, Enum):
    INIT = "INIT"
    DECIDE = "DECIDE"
    ACT = "ACT"
    CRITIC = "CRITIC"
    REDUCE = "REDUCE"
    CHECK_STOP = "CHECK_STOP"
    END = "END"
    DECIDE_ERROR = "DECIDE_ERROR"
    ACT_ERROR = "ACT_ERROR"
    RECOVER = "RECOVER"
    DELEGATE_START = "DELEGATE_START"
    DELEGATE_END = "DELEGATE_END"
    HANDOFF_START = "HANDOFF_START"
    HANDOFF_END = "HANDOFF_END"
    INTERRUPT = "INTERRUPT"
    FANOUT_START = "FANOUT_START"
    FANOUT_END = "FANOUT_END"
    COMPACT = "COMPACT"
    SESSION_START = "SESSION_START"
    SESSION_END = "SESSION_END"


@dataclass
class RuntimeBudget:
    max_steps: int = 10  # Default matches Engine's safe step limit
    max_runtime_seconds: Optional[float] = None
    max_tokens: Optional[int] = None
    max_cost_usd: Optional[float] = None
    max_tool_concurrency: int = 4
    max_children: int = 4
    deadline_monotonic: Optional[float] = None

    def __post_init__(self) -> None:
        for name in ("max_steps", "max_tool_concurrency", "max_children"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_tokens is not None:
            if not isinstance(self.max_tokens, int) or isinstance(
                self.max_tokens, bool
            ):
                raise TypeError("max_tokens must be an integer or None")
            if self.max_tokens <= 0:
                raise ValueError("max_tokens must be positive")
        for name in ("max_runtime_seconds", "max_cost_usd", "deadline_monotonic"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number or None")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            if name != "deadline_monotonic" and float(value) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass
class ContextConfig:
    """Engine-side context-length policy.

    Three thresholds act on one request, in this order:

    * ``warning_ratio`` (0.75) — occupancy at which a ``warning`` context event
      is emitted. Observability only; nothing is reduced.
    * ``compact_ratio`` (0.80) — fraction of the provider-safe total input
      budget at which the history strategy must compact. System and current
      user tokens count toward this threshold.
    * overflow (1.0) — exceeding ``available_input_budget`` raises
      ``ContextOverflowError`` when ``strict_overflow`` is set.

    The compaction threshold is also exposed as ``soft_input_target`` in
    telemetry. There is no second sliding-window target: all normal reduction
    goes through the configured transaction-aware history strategy.
    """

    enabled: bool = True
    warning_ratio: float = 0.75
    compact_ratio: float = 0.80
    safety_reserve_tokens: Optional[int] = None
    safety_reserve_ratio: float = 0.05
    min_safety_reserve_tokens: int = 1024
    default_context_window: int = 128000
    tool_result_max_chars: int = 50000
    tool_result_per_message_max_chars: int = 200000
    conversation_max_rounds: int = 10
    reactive_compact: bool = True
    # Repeated-call protection is generic policy and may be disabled explicitly
    # by a product configuration when its investigation semantics require it.
    tool_call_loop_detection_enabled: bool = True
    loop_max_repeats: int = 3
    max_handoffs: int = 10
    strict_overflow: bool = True
    show_ui: bool = True


@dataclass
class ContextTelemetry:
    context_window: Optional[int] = None
    available_input_budget: Optional[int] = None
    hard_input_budget: Optional[int] = None
    soft_input_target: Optional[int] = None
    input_budget_source: str = "unresolved"
    system_prompt_tokens: int = 0
    history_tokens: int = 0
    prepared_tokens: int = 0
    message_injection_tokens: int = 0
    user_content_block_tokens: int = 0
    request_overhead_tokens: int = 0
    input_tokens_total: int = 0
    output_tokens: int = 0
    provider_prompt_tokens: Optional[int] = None
    provider_completion_tokens: Optional[int] = None
    provider_total_tokens: Optional[int] = None
    planned_prompt_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    cache_write_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    usage_source: str = "absent"
    meter_source: str = "local_estimate"
    meter_status: str = "not_configured"
    meter_error: str = ""
    token_estimate_error: Optional[int] = None
    occupancy_ratio: float = 0.0
    warning_threshold_ratio: float = 0.75
    counting_mode: str = "disabled"
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    tokens_total: int = 0
    peak_input_tokens: int = 0
    peak_occupancy_ratio: float = 0.0
    history_message_count: int = 0
    compact_events: List[Dict[str, Any]] = field(default_factory=list)
    reserve_tokens: int = 0
    max_output_tokens: int = 0
    configured_max_output_tokens: int = 0
    history_budget: Optional[int] = None


@dataclass
class RuntimeEvent:
    step_id: int
    phase: RuntimePhase
    ok: bool = True
    payload: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class StepRecord:
    step_id: int
    transaction_id: str = ""
    phase_events: List[RuntimeEvent] = field(default_factory=list)
    observation: Any = None
    decision: Any = None
    model_request: ModelRequest | None = None
    model_response: Dict[str, Any] = field(default_factory=dict)
    actions: List[Any] = field(default_factory=list)
    action_results: List[Any] = field(default_factory=list)
    tool_invocations: List[Any] = field(default_factory=list)
    # Effective action-execution policy, concurrency peak and segment count
    # for this step's action batch (issue #35).
    action_execution: Dict[str, Any] = field(default_factory=dict)
    critic_outputs: List[Any] = field(default_factory=list)
    state_diff: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    prompt_metadata: Dict[str, Any] = field(default_factory=dict)
    protocol_id: Optional[str] = None
    parser_selected: Optional[str] = None
    parser_fallback_used: bool = False
    parser_attempts: List[Dict[str, Any]] = field(default_factory=list)
    parser_diagnostics: Dict[str, Any] = field(default_factory=dict)
    parser_contract: Optional[str] = None
    parser_salvage_applied: bool = False
    decision_source: Optional[str] = None
    agent_id: Optional[str] = None
    native_tool_call_used: bool = False
    native_tool_call_fallback_reason: Optional[str] = None
    # Parser-derived actions are mirrored as OpenAI-compatible tool calls so
    # their results can remain in the same durable conversation chain.
    history_tool_calls_pending: bool = False
    visual_assets: List[Dict[str, Any]] = field(default_factory=list)
    observation_modalities: List[str] = field(default_factory=list)
    visual_asset_count: int = 0
    has_screenshot: bool = False
    has_dom: bool = False
    has_accessibility_tree: bool = False
    model_input_modalities: List[str] = field(default_factory=list)
    model_input_visual_count: int = 0


@dataclass
class CriticTrace:
    """Structured record of a single critic evaluation within a run."""

    step_id: int
    critic_name: str
    action: str  # "continue" | "stop" | "retry"
    reason: str = ""
    score: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    instruction_patch: Optional[str] = None
    state_patch: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "step_id": self.step_id,
            "critic_name": self.critic_name,
            "action": self.action,
            "reason": self.reason,
            "score": self.score,
        }
        if self.details:
            d["details"] = self.details
        if self.instruction_patch is not None:
            d["instruction_patch"] = self.instruction_patch
        if self.state_patch is not None:
            d["state_patch"] = self.state_patch
        return d


@dataclass
class HandoffTrace:
    """Structured record of an agent handoff within a run."""

    step_id: int
    from_agent: str
    to_agent: str
    context_strategy: str = ""
    messages_passed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "context_strategy": self.context_strategy,
            "messages_passed": self.messages_passed,
        }


@dataclass(frozen=True)
class EngineConfig:
    """Serializable snapshot of Engine configuration."""

    agent_name: str = ""
    model_id: str = ""
    budget_max_steps: int = 10
    budget_max_runtime_seconds: Optional[float] = None
    budget_max_tokens: Optional[int] = None
    critic_names: List[str] = field(default_factory=list)
    stop_criteria_names: List[str] = field(default_factory=list)
    has_checkpoint_store: bool = False
    has_tracing_provider: bool = False
    protocol_id: Optional[str] = None
    delegate_depth: int = 0
    has_shared_memory: bool = False
    has_env: bool = False
    tool_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepResult:
    """Result of a single Engine step for interactive REPLs.

    Provides all data an external driver (REPL, debugger, etc.) needs
    to display output, handle permissions, and decide whether to continue.
    """

    step_id: int
    decision: Any  # Decision[ActionT]
    record: StepRecord
    observation: Any
    action_results: List[Any] = field(default_factory=list)
    stop: bool = False
    stop_reason: Optional[Any] = None  # StopReason
    error: Optional[Exception] = None
    recovered: bool = False
    interrupt_info: Optional[Any] = None  # InterruptInfo (avoids circular import)
