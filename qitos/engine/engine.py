"""Canonical Engine for AgentModule execution."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar
from uuid import uuid4

_logger = logging.getLogger("qitos.engine")

from ..checkpoint.store import (
    Checkpoint,
    CheckpointConfig,
    CheckpointId,
    CheckpointMetadata,
    CheckpointStore,
)
from ..core.agent_module import AgentModule, CanonicalActionResult
from ..core.action import Action, ActionExecutionPolicy
from ..core.artifact import ArtifactStore
from ..core.decision import Decision
from ..core.errors import ErrorCategory, StopReason
from ..core.env import Env, EnvObservation, EnvStepResult
from ..core.history import (
    History,
    HistoryMessage,
    HistoryPolicy,
    HistorySnapshot,
)
from ..core.journal import (
    JournalError,
    JournalPosition,
    JournalRecordType,
    SessionJournal,
)
from ..core.memory import Memory, MemoryRecord
from ..core.model_capabilities import ModelCapabilities
from ..core.model_response import (
    ModelPricing,
    ModelUsage,
    ModelUsageSource,
    normalize_model_usage,
)
from ..core.model_request import ModelContinuation
from ..core.runtime_input import RuntimeInput
from ..core.spec import ExperimentSpec, RunSpec
from ..core.state import StateSchema
from ..core.task import Task, TaskResult, TaskValidationIssue
from ..core.tool_result import ToolResult
from ..core.tool_registry import ToolExposure
from ..core.turn import TurnBudgetSnapshot, TurnRuntimeCapabilities, TurnSnapshot
from ..trace import TraceWriter
from ..protocols import get_protocol, infer_protocol_from_parser
from ..models.profile_registry import infer_default_protocol, infer_model_profile
from ._action_runtime import _ActionRuntime
from ._context_runtime import _ContextRuntime
from ._control_runtime import _ControlRuntime
from ._decision_runtime import _DecisionRuntime
from ._env_runtime import _EnvRuntime
from ._loop_detector import ToolCallLoopDetector
from ._model_runtime import _ModelRuntime
from ._handoff_runtime import _HandoffRuntime
from ._journal_runtime import _JournalRuntime, history_message_to_dict
from ._trace_runtime import _TraceRuntime
from ._turn_runtime import _TurnRuntime
from .action_executor import ActionExecutor
from .cancellation import CancelMode, CancelToken
from .branching import BranchSelector, FirstCandidateSelector
from ._runtime_inbox import RuntimeWaitOutcome, _RuntimeInbox
from .critic import Critic
from .hooks import EngineHook, HookContext
from .parser import Parser
from .recovery import RecoveryPolicy, build_failure_report
from .search import Search
from .states import (
    ContextConfig,
    CriticTrace,
    EngineConfig,
    HandoffTrace,
    RuntimeBudget,
    RuntimeEvent,
    RuntimePhase,
    StepRecord,
    StepResult,
)
from .stop_criteria import FinalResultCriteria, StopCriteria
from .validation import StateValidationGate


StateT = TypeVar("StateT", bound=StateSchema)
ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")

RecoveryHandler = Callable[[StateT, RuntimePhase, Exception], None]


@dataclass
class StepSummary:
    step_id: int
    tool_name: str
    status: str
    latency_ms: float = 0.0
    error: Optional[str] = None
    result_preview: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "result_preview": self.result_preview,
        }


@dataclass
class EngineResult(Generic[StateT]):
    state: StateT
    records: List[StepRecord]
    events: List[RuntimeEvent]
    step_count: int
    task_result: Optional[TaskResult] = None
    runtime_seconds: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    run_id: str = ""
    critic_traces: List[CriticTrace] = field(default_factory=list)
    handoff_traces: List[HandoffTrace] = field(default_factory=list)
    _cancel_token: Optional[CancelToken] = None

    def cancel(self, mode: str = "immediate") -> None:
        """Request cancellation of the running Engine.

        Parameters
        ----------
        mode : str
            ``"immediate"`` — stop as soon as possible (may be mid-step).
            ``"after_step"`` — wait for the current step to complete first.
        """
        if self._cancel_token is None:
            return
        self._cancel_token.request_cancel(mode)

    @property
    def tool_calls_by_name(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for record in self.records:
            for inv in list(getattr(record, "tool_invocations", []) or []):
                if not isinstance(inv, dict):
                    continue
                name = str(inv.get("tool_name", "") or "").strip()
                if not name:
                    continue
                counts[name] = counts.get(name, 0) + 1
        return counts

    @property
    def success_rate(self) -> float:
        total = 0
        success = 0
        for record in self.records:
            for item in list(getattr(record, "action_results", []) or []):
                total += 1
                if ToolResult.from_value(item).is_success:
                    success += 1
        if total <= 0:
            return 0.0
        return float(success) / float(total)

    @property
    def step_summaries(self) -> List[StepSummary]:
        items: List[StepSummary] = []
        for record in self.records:
            invocations = list(getattr(record, "tool_invocations", []) or [])
            action_results = list(getattr(record, "action_results", []) or [])
            for idx, invocation in enumerate(invocations):
                tool_name = ""
                latency_ms = 0.0
                if isinstance(invocation, dict):
                    tool_name = str(invocation.get("tool_name", "") or "")
                    latency = invocation.get("latency_ms")
                    if isinstance(latency, (int, float)):
                        latency_ms = float(latency)
                tool_result = (
                    ToolResult.from_value(action_results[idx])
                    if idx < len(action_results)
                    else ToolResult(status="error", error="missing_action_result")
                )
                preview = tool_result.text
                items.append(
                    StepSummary(
                        step_id=record.step_id,
                        tool_name=tool_name,
                        status=tool_result.status,
                        latency_ms=latency_ms,
                        error=tool_result.error,
                        result_preview=preview[:200],
                    )
                )
        return items

    def to_dict(self) -> Dict[str, Any]:
        task_result_dict: Any = None
        if self.task_result is not None:
            if hasattr(self.task_result, "to_dict"):
                task_result_dict = self.task_result.to_dict()
            else:
                task_result_dict = self.task_result
        return {
            "step_count": self.step_count,
            "runtime_seconds": self.runtime_seconds,
            "total_tokens": self.total_tokens,
            "tool_calls_by_name": self.tool_calls_by_name,
            "success_rate": self.success_rate,
            "step_summaries": [item.to_dict() for item in self.step_summaries],
            "critic_traces": [ct.to_dict() for ct in self.critic_traces],
            "handoff_traces": [ht.to_dict() for ht in self.handoff_traces],
            "task_result": task_result_dict,
            "state": (
                self.state.to_dict() if hasattr(self.state, "to_dict") else self.state
            ),
        }


class Engine(Generic[StateT, ObservationT, ActionT]):
    """Single execution kernel for all AgentModule workflows."""

    def __init__(
        self,
        agent: AgentModule[StateT, ObservationT, ActionT],
        agent_registry: Optional[Any] = None,
        budget: Optional[RuntimeBudget] = None,
        delegate_depth: int = 0,
        shared_memory: Any = None,
        validation_gate: Optional[StateValidationGate] = None,
        recovery_handler: Optional[RecoveryHandler] = None,
        recovery_policy: Optional[RecoveryPolicy] = None,
        trace_writer: Optional[TraceWriter] = None,
        parser: Optional[Parser[ActionT]] = None,
        protocol: Any = None,
        stop_criteria: Optional[List[StopCriteria]] = None,
        branch_selector: Optional[BranchSelector[StateT, ObservationT, ActionT]] = None,
        search: Optional[Search[StateT, ObservationT, ActionT]] = None,
        critics: Optional[List[Critic]] = None,
        env: Optional[Env] = None,
        history_policy: Optional[HistoryPolicy] = None,
        hooks: Optional[List[EngineHook]] = None,
        render_hooks: Optional[List[Any]] = None,
        context_config: Optional[ContextConfig | Dict[str, Any]] = None,
        cache_backend: Optional[Any] = None,
        checkpoint_store: Optional[CheckpointStore] = None,
        artifact_store: Optional[ArtifactStore] = None,
        permission_pipeline: Optional[Any] = None,
        read_before_write_enforcer: Optional[Any] = None,
        permission_interaction_callback: Optional[Any] = None,
        loop_detector: Optional[ToolCallLoopDetector] = None,
        tracing_provider: Optional[Any] = None,
        auto_approve: bool = False,
        action_execution_policy: Optional[ActionExecutionPolicy] = None,
        model_pricing: ModelPricing | None = None,
        journal: SessionJournal | None = None,
        state_snapshot_interval: int = 16,
    ):
        if journal is not None and checkpoint_store is not None:
            raise ValueError(
                "journal and checkpoint_store are alternative persistence owners"
            )
        self.agent = agent
        self.agent_registry = agent_registry
        self._delegate_depth = delegate_depth
        self._shared_memory = shared_memory
        self.tool_registry = agent.tool_registry
        # Ensure Engine always has a ToolRegistry — agents without tools still
        # need one for handoff/permission tools registered by the Engine itself.
        if self.tool_registry is None:
            from ..core.tool_registry import ToolRegistry as _TR

            self.tool_registry = _TR()
        self.budget = budget or RuntimeBudget()
        if model_pricing is not None and not isinstance(model_pricing, ModelPricing):
            raise TypeError("model_pricing must be a ModelPricing")
        configured_pricing = model_pricing
        if configured_pricing is None:
            candidate_pricing = getattr(getattr(agent, "llm", None), "pricing", None)
            if isinstance(candidate_pricing, ModelPricing):
                configured_pricing = candidate_pricing
        if self.budget.max_cost_usd is not None and configured_pricing is None:
            raise ValueError("max_cost_usd requires explicit model_pricing")
        self.model_pricing = configured_pricing
        self._base_budget = RuntimeBudget(
            max_steps=self.budget.max_steps,
            max_runtime_seconds=self.budget.max_runtime_seconds,
            max_tokens=self.budget.max_tokens,
            max_cost_usd=self.budget.max_cost_usd,
            max_tool_concurrency=self.budget.max_tool_concurrency,
            max_children=self.budget.max_children,
            deadline_monotonic=self.budget.deadline_monotonic,
        )
        self.validation_gate = validation_gate or StateValidationGate()
        self.recovery_handler = recovery_handler
        self.recovery_policy = recovery_policy or RecoveryPolicy()
        self.trace_writer = trace_writer
        self.run_spec: RunSpec | None = None
        self.experiment_spec: ExperimentSpec | None = None
        self.parser = parser
        self.protocol = protocol
        self._resolved_protocol: Any = None
        self._resolved_protocol_source: str = ""
        self.branch_selector = branch_selector or FirstCandidateSelector()
        self.search = search
        self.critics = critics or []
        self.env = env
        self.history_policy = history_policy or HistoryPolicy()
        self.context_config = (
            context_config
            if isinstance(context_config, ContextConfig)
            else ContextConfig(**dict(context_config or {}))
        )
        self.hooks: List[Any] = list(hooks or [])
        if render_hooks:
            self.hooks.extend(render_hooks)
        if stop_criteria is None:
            self._uses_default_stop_criteria = True
            self.stop_criteria: List[StopCriteria] = [FinalResultCriteria()]
        else:
            self._uses_default_stop_criteria = False
            self.stop_criteria = list(stop_criteria)

        # Wire permission pipeline and RBW enforcer: explicit params > agent attrs
        resolved_pipeline = permission_pipeline or getattr(
            agent, "permission_pipeline", None
        )
        resolved_rbw = read_before_write_enforcer or getattr(
            agent, "_rbw_enforcer", None
        )

        self.auto_approve = auto_approve
        # Action execution policy is public API and must survive executor
        # rebuilds (handoff, resume). Stored alongside the other executor
        # dependencies so _build_action_executor() is the single source of truth.
        self._action_execution_policy = action_execution_policy
        self._permission_pipeline = resolved_pipeline
        self._rbw_enforcer = resolved_rbw
        self._permission_interaction_callback = permission_interaction_callback
        self.executor = self._build_action_executor(self.tool_registry)
        self.events: List[RuntimeEvent] = []
        self.records: List[StepRecord] = []
        self._active_state: Optional[StateT] = None
        self._active_task: str = ""
        self._active_task_obj: Optional[Task] = None
        self._last_env_observation: Optional[EnvObservation] = None
        self._last_env_result: Optional[EnvStepResult] = None
        self._token_usage: int = 0
        self._cost_usage_usd: float = 0.0
        self._model_continuation: ModelContinuation | None = None
        self._active_run_id: str = ""
        self._runtime_deadline_monotonic: Optional[float] = None
        from ..kit.history import WindowHistory

        self._runtime_history: History = WindowHistory(window_size=24)
        self._tool_loop_detector = (
            loop_detector
            if self.context_config.tool_call_loop_detection_enabled
            else None
        )
        if (
            self._tool_loop_detector is None
            and self.context_config.tool_call_loop_detection_enabled
        ):
            self._tool_loop_detector = ToolCallLoopDetector(
                max_repeats=max(1, int(self.context_config.loop_max_repeats))
            )
        self._last_system_prompt: str = ""
        self._critic_modified_prompt: Optional[str] = None
        self._critic_instruction_patch: Optional[str] = None
        self._last_prompt_metadata: Dict[str, Any] = {}
        self._last_context_telemetry: Dict[str, Any] = {}
        self._last_runtime_error: Optional[Dict[str, Any]] = None
        self._model_runtime: _ModelRuntime[StateT, ObservationT, ActionT] = (
            _ModelRuntime(self)
        )
        self._decision_runtime: _DecisionRuntime[StateT, ObservationT, ActionT] = (
            _DecisionRuntime(self, self._model_runtime)
        )
        self._action_runtime: _ActionRuntime[StateT, ActionT] = _ActionRuntime(self)
        self._journal_runtime: _JournalRuntime[StateT, ActionT] = _JournalRuntime(self)
        self._env_runtime: _EnvRuntime[StateT, ObservationT, ActionT] = _EnvRuntime(
            self
        )
        self._control_runtime: _ControlRuntime[StateT, ObservationT, ActionT] = (
            _ControlRuntime(self)
        )
        self._turn_runtime: _TurnRuntime[StateT, ObservationT, ActionT] = (
            _TurnRuntime(self)
        )
        self._trace_runtime: _TraceRuntime[StateT] = _TraceRuntime(self)
        self._handoff_runtime: _HandoffRuntime[StateT, ObservationT, ActionT] = (
            _HandoffRuntime(self)
        )
        self._handoff_history: list[str] = []  # tracks agent names for loop detection
        # NOTE (v0.6): Handoff Decision-mode handling is stable for v0.6.
        # Changes to the Engine loop for full handoff context strategies,
        # shared memory, and canonical multi-agent templates are deferred to v0.7.
        # See docs/internal/plans/v0.7_handoff_scope.md for details.
        self.stream_callback: Optional[Any] = (
            None  # Callable[[str], None] for streaming
        )
        self._context_runtime = _ContextRuntime(self)
        self._context_runtime.apply_config(self.context_config)

        # LLM Cache: auto-wrap agent.llm with CachedModel if backend provided
        self.cache_backend = cache_backend
        if (
            self.cache_backend is not None
            and getattr(self.agent, "llm", None) is not None
        ):
            from ..cache import CachedModel

            if not isinstance(self.agent.llm, CachedModel):
                self.agent.llm = CachedModel(self.agent.llm, self.cache_backend)

        self._checkpoint_store = checkpoint_store
        self._last_checkpoint_id: Optional[CheckpointId] = None
        self.artifact_store = artifact_store
        self.journal = journal
        if state_snapshot_interval <= 0:
            raise ValueError("state_snapshot_interval must be positive")
        self._state_snapshot_interval = int(state_snapshot_interval)
        self._journal_pending_history: list[dict[str, Any]] = []
        self._journal_terminal_record_ids: dict[str, list[str]] = {}
        self._canonical_action_results: list[CanonicalActionResult] = []
        self._last_journal_position: JournalPosition | None = None

        self._tracing_provider = tracing_provider

        # Handoff tools: auto-register if agent declares handoff_targets
        self._handoff_tools: List[Any] = []
        if getattr(agent, "handoff_targets", None) and self.tool_registry is not None:
            self._register_handoff_tools()

        # Cancellation token — shared with EngineResult for external cancel
        self._cancel_token = CancelToken()
        self._runtime_inbox = _RuntimeInbox()
        self._active_async_task: Optional[asyncio.Task[Any]] = None
        self._active_async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._connected_mcp_servers: List[Any] = []
        self._mcp_tool_names: List[str] = []

    def _build_action_executor(
        self,
        tool_registry: Any,
        *,
        turn: TurnSnapshot | None = None,
    ) -> Optional[ActionExecutor]:
        """Construct an ActionExecutor carrying every engine-level dependency.

        Single source of truth for executor construction so that rebuilds
        (handoff, resume) cannot silently drop the execution policy, the
        permission pipeline or the cancellation token.
        """
        if tool_registry is None:
            return None
        configured_policy = self._action_execution_policy or ActionExecutionPolicy()
        max_tool_concurrency = (
            turn.budget.max_tool_concurrency
            if turn is not None
            else self.budget.max_tool_concurrency
        )
        policy = ActionExecutionPolicy(
            mode=configured_policy.mode,
            fail_fast=configured_policy.fail_fast,
            max_concurrency=min(
                int(configured_policy.max_concurrency),
                int(max_tool_concurrency),
            ),
            parallel_tool_names=configured_policy.parallel_tool_names,
        )
        return ActionExecutor(
            tool_registry=tool_registry,
            policy=policy,
            trace_writer=self.trace_writer,
            delegate_depth=self._delegate_depth,
            shared_memory=self._shared_memory,
            engine=self,
            permission_pipeline=self._permission_pipeline,
            read_before_write_enforcer=self._rbw_enforcer,
            permission_interaction_callback=self._permission_interaction_callback,
            auto_approve=self.auto_approve,
            turn_budget=turn.budget if turn is not None else None,
        )

    def _capture_turn(
        self,
        state: StateT,
        step_id: int,
    ) -> TurnSnapshot:
        """Capture the exact immutable inputs shared by model and tools."""

        exposure = self.agent.build_tool_exposure(state, self.tool_registry)
        if not isinstance(exposure, ToolExposure):
            raise TypeError("Agent.build_tool_exposure() must return ToolExposure")
        model = getattr(self.agent, "llm", None)
        model_capabilities = getattr(model, "capabilities", ModelCapabilities())
        if not isinstance(model_capabilities, ModelCapabilities):
            model_capabilities = ModelCapabilities()
        environment_ops: set[str] = set()
        for name in exposure.list_tools():
            description = exposure.describe_tool(name)
            environment_ops.update(description.get("required_ops") or [])
            environment_ops.update(description.get("environment_ops") or [])
        remaining_tokens = (
            None
            if self.budget.max_tokens is None
            else max(0, int(self.budget.max_tokens) - int(self._token_usage))
        )
        remaining_cost = (
            None
            if self.budget.max_cost_usd is None
            else max(0.0, float(self.budget.max_cost_usd) - self._cost_usage_usd)
        )
        protocol = self.resolve_protocol()
        return TurnSnapshot(
            run_id=self._active_run_id,
            step_id=step_id,
            model=model,
            protocol=protocol,
            protocol_source=self._resolved_protocol_source,
            history=self._history().snapshot(),
            tools=exposure,
            capabilities=TurnRuntimeCapabilities(
                model=model_capabilities,
                environment_ops=tuple(sorted(environment_ops)),
                mailbox=True,
                child_agents="Agent" in exposure.list_tools(),
            ),
            budget=TurnBudgetSnapshot(
                step=step_id,
                max_steps=int(self.budget.max_steps),
                used_tokens=int(self._token_usage),
                remaining_tokens=remaining_tokens,
                used_cost_usd=float(self._cost_usage_usd),
                remaining_cost_usd=remaining_cost,
                model_pricing=self.model_pricing,
                deadline_monotonic=self._runtime_deadline_monotonic,
                max_tool_concurrency=int(self.budget.max_tool_concurrency),
                max_children=int(self.budget.max_children),
            ),
        )

    def cancel(self, mode: str = "immediate") -> None:
        """Request cancellation of an in-flight run.

        Thread-safe and idempotent, and usable once a run has already started
        (unlike ``EngineResult.cancel()``, which is only reachable after the
        run returns).

        Parameters
        ----------
        mode : str
            ``"immediate"`` — stop as soon as possible (may be mid-step).
            ``"after_step"`` — wait for the current step to complete first.
        """
        self._cancel_token.request_cancel(mode)
        self._runtime_inbox.wake()
        if mode == "immediate":
            loop = self._active_async_loop
            task = self._active_async_task
            if loop is not None and task is not None and not task.done():
                try:
                    loop.call_soon_threadsafe(task.cancel)
                except RuntimeError:
                    pass

    @property
    def runtime_deadline_monotonic(self) -> Optional[float]:
        """Return the effective absolute deadline for the active run."""

        return self._runtime_deadline_monotonic

    def remaining_runtime_seconds(self) -> Optional[float]:
        """Return live time remaining before the active run deadline."""

        deadline = self._runtime_deadline_monotonic
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())

    def _record_model_cost(
        self,
        usage: ModelUsage | Mapping[str, Any] | None,
        *,
        pricing: ModelPricing | None,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        if pricing is None:
            return 0.0
        normalized = normalize_model_usage(usage) or ModelUsage(
            input_tokens=max(0, int(input_tokens)),
            output_tokens=max(0, int(output_tokens)),
            total_tokens=max(0, int(input_tokens)) + max(0, int(output_tokens)),
            source=ModelUsageSource.ESTIMATE,
        )
        transaction_cost = pricing.cost_usd(normalized)
        self._cost_usage_usd += transaction_cost
        return transaction_cost

    def _activate_runtime_budget(self, started_at: float) -> None:
        deadlines: List[float] = []
        if self.budget.deadline_monotonic is not None:
            deadlines.append(float(self.budget.deadline_monotonic))
        if self.budget.max_runtime_seconds is not None:
            deadlines.append(started_at + float(self.budget.max_runtime_seconds))
        self._runtime_deadline_monotonic = min(deadlines) if deadlines else None

    @property
    def active_run_id(self) -> str:
        """Return the current run id, or an empty string outside a run."""

        return self._active_run_id

    def post_runtime_event(self, event: RuntimeInput, *, run_id: str) -> bool:
        """Post one idempotent event to the exact active run."""

        if not isinstance(event, RuntimeInput):
            raise TypeError("event must be a RuntimeInput")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        return self._runtime_inbox.post(run_id, event)

    def resolve_protocol(self) -> Any:
        """Resolve the protocol from the configuration visible to this turn."""

        explicit = self.protocol
        if explicit is not None:
            self._resolved_protocol = get_protocol(explicit)
            self._resolved_protocol_source = "run_protocol"
            return self._resolved_protocol
        agent_protocol = getattr(self.agent, "model_protocol", None)
        if agent_protocol is not None:
            self._resolved_protocol = get_protocol(agent_protocol)
            self._resolved_protocol_source = "agent_model_protocol"
            return self._resolved_protocol
        parser = self.parser or getattr(self.agent, "model_parser", None)
        if parser is not None:
            inferred = infer_protocol_from_parser(parser)
            if inferred is not None:
                self._resolved_protocol = inferred
                self._resolved_protocol_source = "parser_inferred"
                return self._resolved_protocol
        llm = getattr(self.agent, "llm", None)
        model_protocol = get_protocol(getattr(llm, "qitos_protocol", None))
        if model_protocol is not None:
            self._resolved_protocol = model_protocol
            self._resolved_protocol_source = "model_qitos_protocol"
            return self._resolved_protocol
        harness_metadata = getattr(llm, "qitos_harness_metadata", {}) or {}
        metadata_protocol = get_protocol(
            harness_metadata.get("protocol")
            if isinstance(harness_metadata, Mapping)
            else None
        )
        if metadata_protocol is not None:
            self._resolved_protocol = metadata_protocol
            self._resolved_protocol_source = "model_harness_metadata"
            return self._resolved_protocol
        model_name = getattr(llm, "model", None) or getattr(llm, "model_name", None)
        default_protocol = infer_default_protocol(model_name, fallback="react_text_v1")
        self._resolved_protocol = get_protocol(default_protocol)
        self._resolved_protocol_source = (
            "model_profile"
            if infer_model_profile(model_name) is not None
            else "framework_default"
        )
        return self._resolved_protocol

    def register_hook(self, hook: Any) -> None:
        """Register one runtime hook instance."""
        self.hooks.append(hook)

    def unregister_hook(self, hook: Any) -> None:
        """Unregister one runtime hook instance if present."""
        self.hooks = [h for h in self.hooks if h is not hook]

    def clear_hooks(self) -> None:
        """Remove all runtime hooks."""
        self.hooks = []

    # ------------------------------------------------------------------
    # Public step-by-step API for interactive REPLs and external drivers
    # ------------------------------------------------------------------

    def init_session(
        self,
        task: str,
        *,
        history_snapshot: HistorySnapshot | None = None,
        **kwargs: Any,
    ) -> tuple[StateT, ObservationT]:
        """Initialize a new session for step-by-step execution.

        Sets up Engine run state, creates initial state and observation.
        Returns (state, observation) ready for the first ``step()`` call.
        """
        self._reset_run_state()
        memory = self._memory()
        if memory is not None:
            try:
                memory.reset()
            except Exception as exc:
                _logger.debug("Failed to reset memory: %s", exc)
        self._reset_history(history_snapshot)
        if hasattr(self.recovery_policy, "reset"):
            try:
                self.recovery_policy.reset()
            except Exception as exc:
                _logger.debug("Failed to reset recovery_policy: %s", exc)
        self._active_run_id = f"run_{uuid4().hex[:12]}"
        self._last_system_prompt = ""
        self._last_prompt_metadata = {}
        self._token_usage = 0
        self._last_context_telemetry = {}
        self._context_runtime.reset()
        self._resolved_protocol = self.resolve_protocol()

        task_obj, task_text = self._normalize_task(task)
        self._apply_task_budget(task_obj)
        self.budget.__post_init__()
        if self.budget.max_cost_usd is not None and self.model_pricing is None:
            raise ValueError("max_cost_usd requires explicit model_pricing")
        self.executor = self._build_action_executor(self.tool_registry)
        started_at = time.monotonic()
        self._activate_runtime_budget(started_at)

        state = self.agent.init_state(task_text, **kwargs)
        self._memory_append("task", {"objective": task_text}, 0)
        self._active_task = task_text
        self._active_task_obj = task_obj
        self._active_state = state

        self._setup_toolsets(
            {
                "state": state,
                "trace_writer": self.trace_writer,
                "task": task_obj or task_text,
            }
        )

        observation = self._build_initial_observation(
            state, step_id=0, started_at=started_at
        )
        return state, observation

    async def astep(
        self,
        state: StateT,
        observation: ObservationT,
    ) -> StepResult:
        """Execute one canonical immutable turn on the caller's event loop.

        Interactive callers own step advancement; full runs use the same turn
        runtime with transactional advancement and persistence enabled.
        """

        self._drain_runtime_events(state.current_step)
        execution = await self._turn_runtime.execute(
            state,
            observation,
            task=self._active_task or state.task,
            started_at=time.monotonic(),
            step_id=state.current_step,
            managed_run=False,
        )
        return execution.step

    def step(
        self,
        state: StateT,
        observation: ObservationT,
    ) -> StepResult:
        """Run one step from a synchronous application boundary."""

        return self._run_sync(self.astep(state, observation), operation="step")

    def advance_step(self, state: StateT) -> None:
        """Advance the state step counter after a completed step."""
        state.advance_step()

    def append_user_message(self, content: str, step_id: int) -> None:
        """Append a user message to the conversation history."""
        self._history_append("user", content, step_id, metadata={"source": "user"})

    def submit_turn(
        self, state: StateT, user_message: str
    ) -> tuple[StateT, ObservationT]:
        """Submit a user message and build the initial observation for the next turn.

        This is the public API for multi-turn REPLs. It wraps
        ``append_user_message()`` + ``_build_initial_observation()`` so
        callers don't need to reach into private methods.

        Returns (state, observation) ready for the next ``step()`` call.
        """
        step_id = state.current_step
        self.append_user_message(user_message, step_id)
        observation = self._build_initial_observation(state, step_id, time.monotonic())
        return state, observation

    async def aexecute_actions(
        self, state: StateT, decision: Decision[ActionT], record: StepRecord
    ) -> List[Any]:
        """Execute a decision's actions without blocking the event loop.

        Useful for REPLs that want to handle DECIDE themselves but delegate
        ACT execution to the engine.
        """
        turn = self._capture_turn(state, record.step_id)
        return await self._run_act(state, decision, record, turn)

    def execute_actions(
        self, state: StateT, decision: Decision[ActionT], record: StepRecord
    ) -> List[Any]:
        """Execute actions from a synchronous application boundary."""

        return self._run_sync(
            self.aexecute_actions(state, decision, record),
            operation="execute_actions",
        )

    def rebuild_observation(self, state: StateT) -> ObservationT:
        """Build a fresh observation for the current state.

        Useful after error recovery or parser repair when the REPL needs
        to continue the loop without submitting a new user message.
        """
        return self._build_initial_observation(
            state, state.current_step, time.monotonic()
        )

    def budget_exhausted(self, state: StateT) -> bool:
        """Check if the runtime budget has been exhausted."""
        return self._budget_exhausted(state.current_step, state)

    @property
    def current_state(self) -> Optional[StateT]:
        """Return the active state, if any."""
        return self._active_state

    @property
    def checkpoint_store(self) -> Optional[CheckpointStore]:
        """Return the configured CheckpointStore, if any."""
        return self._checkpoint_store

    @property
    def last_checkpoint_id(self) -> Optional[CheckpointId]:
        """Return the latest successfully committed checkpoint id, if any."""
        return self._last_checkpoint_id

    @property
    def last_journal_position(self) -> JournalPosition | None:
        """Return the latest durably appended journal position, if configured."""

        return self._last_journal_position

    @property
    def tracing_provider(self) -> Any:
        """Return the configured TracingProvider, if any."""
        return self._tracing_provider

    async def arun(
        self,
        task: str | Task,
        *,
        history_snapshot: HistorySnapshot | None = None,
        **kwargs: Any,
    ) -> EngineResult[StateT]:
        """Run the canonical Engine loop on the caller's event loop."""

        current_task = asyncio.current_task()
        if current_task is None:
            raise RuntimeError("Engine.arun() requires a running event loop")
        if self._active_async_task is not None and not self._active_async_task.done():
            raise RuntimeError("Engine already has an active run")
        self._active_async_task = current_task
        self._active_async_loop = asyncio.get_running_loop()
        try:
            result = await self._arun_impl(
                task,
                history_snapshot=history_snapshot,
                **kwargs,
            )
        except BaseException:
            await self._close_journal_after_error()
            raise
        else:
            await self._close_journal()
            return result
        finally:
            if self._active_async_task is current_task:
                self._active_async_task = None
                self._active_async_loop = None

    async def _arun_impl(
        self,
        task: str | Task,
        *,
        history_snapshot: HistorySnapshot | None = None,
        **kwargs: Any,
    ) -> EngineResult[StateT]:
        # Check for resume-from-checkpoint internal kwargs
        _resume_state = kwargs.pop("_resume_state", None)
        _resume_step = kwargs.pop("_resume_step", None)
        _resume_run_id = kwargs.pop("_resume_run_id", None)
        _resume_checkpoint_id = kwargs.pop("_resume_checkpoint_id", None)
        _resume_journal = bool(kwargs.pop("_resume_journal", False))
        _resume_canonical_results = tuple(
            kwargs.pop("_resume_canonical_results", ())
        )
        _resume_usage = dict(kwargs.pop("_resume_usage", {}) or {})
        _resume_continuation = kwargs.pop("_resume_continuation", None)

        self._reset_run_state()
        self._canonical_action_results = list(_resume_canonical_results)
        memory = self._memory()
        if memory is not None:
            try:
                memory.reset()
            except Exception as exc:
                _logger.debug("Failed to reset memory: %s", exc)
        self._reset_history(history_snapshot)
        if hasattr(self.recovery_policy, "reset"):
            try:
                self.recovery_policy.reset()
            except Exception as exc:
                _logger.debug("Failed to reset recovery_policy: %s", exc)
        self._active_run_id = str(_resume_run_id or "").strip() or (
            str(getattr(self.trace_writer, "run_id", "")).strip()
            if self.trace_writer is not None
            else ""
        ) or f"run_{uuid4().hex[:12]}"
        self._last_checkpoint_id = _resume_checkpoint_id
        self._last_system_prompt = ""
        self._last_prompt_metadata = {}
        task_obj, task_text = self._normalize_task(task)
        self._apply_task_budget(task_obj)
        started_at = time.monotonic()
        self._activate_runtime_budget(started_at)
        self._token_usage = int(_resume_usage.get("total_tokens", 0) or 0)
        self._cost_usage_usd = float(_resume_usage.get("cost_usd", 0.0) or 0.0)
        self._model_continuation = (
            _resume_continuation
            if isinstance(_resume_continuation, ModelContinuation)
            else None
        )
        self._last_context_telemetry = {}
        self._context_runtime.reset()
        self._context_runtime.restore_usage(
            prompt_tokens=int(_resume_usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(_resume_usage.get("completion_tokens", 0) or 0),
            total_tokens=self._token_usage,
        )
        self._resolved_protocol = self.resolve_protocol()
        # Record multi-agent topology in trace metadata
        if self.trace_writer is not None and self.agent_registry is not None:
            self.trace_writer.metadata["agent_topology"] = {
                "type": "multi_agent",
                "agents": [s.name for s in self.agent_registry.list_available()],
            }
            self.trace_writer.metadata["agent_name"] = self.agent.name

        # State initialization: fresh or resumed
        if _resume_state is not None:
            state = _resume_state
        else:
            try:
                state = self.agent.init_state(task_text, **kwargs)
            except Exception as exc:
                self._report_runtime_exception("INIT_STATE", 0, exc, emit=False)
                raise
        self._memory_append(
            "task",
            {
                "objective": task_text,
                "task_id": task_obj.id if task_obj is not None else None,
            },
            0,
        )
        self._active_task = task_text
        self._active_task_obj = task_obj
        self._active_state = state
        self._hydrate_trace_metadata(task_obj=task_obj, task_text=task_text)

        try:
            await self._asetup_toolsets(
                {
                    "state": state,
                    "trace_writer": self.trace_writer,
                    "task": task_obj or task_text,
                }
            )
        except Exception as exc:
            self._report_runtime_exception("SETUP_TOOLSETS", 0, exc, emit=False)
            raise
        try:
            self._setup_env(task_obj=task_obj, state=state, kwargs=kwargs)
        except Exception as exc:
            self._report_runtime_exception("SETUP_ENV", 0, exc, emit=False)
            raise
        if self.journal is not None and not _resume_journal:
            await self._journal_initialize(task_obj, task_text, state)
            self._active_state = type(state).from_dict(state.to_dict())
        elif self.journal is not None:
            self._active_state = type(state).from_dict(state.to_dict())
        self._runtime_inbox.open(self._active_run_id)
        harness_diagnostics = self._harness_mismatch_diagnostics()
        self._emit(
            0,
            RuntimePhase.INIT,
            payload={
                "task": task_text,
                "task_id": task_obj.id if task_obj is not None else None,
                "task_meta": self._task_meta(task_obj),
                "run_meta": self._run_meta(),
                "env": self._env_identity(),
                "harness_diagnostics": harness_diagnostics,
            },
        )
        if harness_diagnostics.get("mismatch"):
            self._emit(
                0,
                RuntimePhase.INIT,
                payload={"stage": "harness_mismatch", **harness_diagnostics},
            )
        self._notify_run_start(task_text, state)
        try:
            preflight_issues = self._preflight_validate(
                task_obj=task_obj, workspace=kwargs.get("workspace")
            )
        except BaseException:
            self._runtime_inbox.close(self._active_run_id)
            raise
        if preflight_issues:
            has_task_issue = any(
                not issue.code.startswith("ENV_") for issue in preflight_issues
            )
            stop_reason = (
                StopReason.TASK_VALIDATION_FAILED
                if has_task_issue
                else StopReason.ENV_CAPABILITY_MISMATCH
            )
            state.set_stop(stop_reason)
            state.final_result = "Preflight validation failed."
            self._emit(
                0,
                RuntimePhase.END,
                ok=False,
                payload={
                    "stop_reason": state.stop_reason,
                    "error_category": (
                        ErrorCategory.TASK.value
                        if has_task_issue
                        else ErrorCategory.ENV.value
                    ),
                    "issues": [self._task_issue_to_dict(x) for x in preflight_issues],
                },
            )
            result = EngineResult(
                state=state,
                records=self.records,
                events=self.events,
                step_count=0,
                task_result=self._build_task_result(
                    state, task_obj=task_obj, started_at=started_at
                ),
                runtime_seconds=time.monotonic() - started_at,
                total_tokens=int(self._token_usage),
                total_cost_usd=float(self._cost_usage_usd),
                run_id=self._active_run_id,
                _cancel_token=self._cancel_token,
            )
            if self.journal is not None:
                await self._journal_finish_run(state)
            self._runtime_inbox.close(self._active_run_id)
            self._notify_run_end(result)
            self._clear_active_context()
            await self._teardown_env()
            await self._ateardown_toolsets(
                {
                    "state": state,
                    "trace_writer": self.trace_writer,
                    "task": task_obj or task_text,
                }
            )
            return result

        step_id = _resume_step if _resume_step is not None else 0
        cancelled = False
        propagate_cancel = False
        journal_interrupted = False
        try:
            # MCP discovery happens after preflight but before the first model
            # turn. Empty configuration creates no thread or connection.
            if getattr(self.agent, "mcp_servers", None):
                await self._connect_mcp_servers()
            current_observation = self._build_initial_observation(
                state, step_id, started_at
            )
            if _resume_state is None and _resume_step is None:
                if self.journal is None:
                    await self._save_checkpoint(
                        step_id,
                        state,
                        task_text,
                        source="input",
                    )
            while True:
                if (
                    self._cancel_token.is_cancel_requested
                    and self._cancel_token.mode == CancelMode.IMMEDIATE
                ):
                    state.set_stop(StopReason.CANCELLED_IMMEDIATE)
                    self._emit(
                        step_id,
                        RuntimePhase.END,
                        ok=False,
                        payload={"stop_reason": state.stop_reason},
                    )
                    await self._journal_interrupt_run(
                        step_id=step_id,
                        reason=StopReason.CANCELLED_IMMEDIATE.value,
                    )
                    journal_interrupted = True
                    break

                if self._budget_exhausted(step_id, state):
                    self._emit(
                        step_id,
                        RuntimePhase.END,
                        ok=False,
                        payload={"stop_reason": state.stop_reason},
                    )
                    break

                # External input is accepted only here. The turn runtime then
                # captures one immutable provider/tool/config view and commits
                # one complete transaction before control returns.
                self._drain_runtime_events(step_id)
                execution = await self._turn_runtime.execute(
                    state,
                    current_observation,
                    task=task_text,
                    started_at=started_at,
                    step_id=step_id,
                )
                state = execution.state
                current_observation = execution.observation
                step_id = execution.next_step_id
                journal_interrupted = (
                    journal_interrupted or execution.journal_interrupted
                )

                if execution.stop:
                    break

                if (
                    execution.check_after_step_cancel
                    and self._cancel_token.is_cancel_requested
                    and self._cancel_token.mode == CancelMode.AFTER_STEP
                ):
                    await self._journal_interrupt_run(
                        step_id=execution.step.step_id,
                        reason="cancelled_after_step",
                    )
                    journal_interrupted = True
                    self._emit(
                        execution.step.step_id,
                        RuntimePhase.END,
                        ok=False,
                        payload={"stop_reason": "cancelled_after_step"},
                    )
                    break
        except asyncio.CancelledError:
            cancelled = True
            propagate_cancel = not self._cancel_token.is_cancel_requested
            self._cancel_token.request_cancel("immediate")
            if state.stop_reason is None:
                state.set_stop(StopReason.CANCELLED_IMMEDIATE)
            self._emit(
                step_id,
                RuntimePhase.END,
                ok=False,
                payload={"stop_reason": state.stop_reason},
            )
            await self._journal_interrupt_run(
                step_id=step_id,
                reason=StopReason.CANCELLED_IMMEDIATE.value,
            )
            journal_interrupted = True
        finally:
            self._runtime_inbox.close(self._active_run_id)
            await self._teardown_env()
            await self._ateardown_toolsets(
                {
                    "state": state,
                    "trace_writer": self.trace_writer,
                    "task": task_obj or task_text,
                }
            )
            # Checkpoint on cancellation (immediate mode)
            if (
                self._cancel_token.is_cancel_requested
                and self._checkpoint_store is not None
            ):
                try:
                    await self._save_checkpoint(
                        step_id,
                        state,
                        task_text,
                        source="cancellation",
                    )
                except Exception as exc:
                    _logger.warning(
                        "Checkpoint save failed during cancellation: %s", exc
                    )
            # Cleanup MCP servers
            await self._cleanup_mcp_servers()

        if cancelled and propagate_cancel:
            self._clear_active_context()
            raise asyncio.CancelledError

        if self.trace_writer is not None:
            task_result = self._build_task_result(
                state, task_obj=task_obj, started_at=started_at
            )
            self.trace_writer.finalize(
                status=self._trace_status(state.stop_reason),
                summary={
                    "stop_reason": state.stop_reason,
                    "final_result": state.final_result,
                    "steps": len(self.records),
                    "token_usage": self._context_runtime.tokens_total,
                    "latency_seconds": task_result.metrics.get("elapsed_seconds", 0.0),
                    "cost_usd": task_result.metrics.get("cost_usd", 0.0),
                    "context": self._context_runtime.run_summary(),
                    "parser": self._trace_runtime.parser_summary(),
                    "task_meta": self._task_meta(task_obj),
                    "task_result": task_result.to_dict(),
                    "run_meta": self._run_meta(),
                    "failure_report": build_failure_report(
                        self.recovery_policy, state.stop_reason
                    ),
                    "last_error": self._last_runtime_error,
                },
            )

        # Extract structured traces from records and events.
        _critic_traces = self._extract_critic_traces()
        _handoff_traces = self._extract_handoff_traces()

        if self.journal is not None and not journal_interrupted:
            await self._journal_finish_run(state)

        result = EngineResult(
            state=state,
            records=self.records,
            events=self.events,
            step_count=len(self.records),
            task_result=self._build_task_result(
                state, task_obj=task_obj, started_at=started_at
            ),
            runtime_seconds=time.monotonic() - started_at,
            total_tokens=int(self._token_usage),
            total_cost_usd=float(self._cost_usage_usd),
            run_id=self._active_run_id,
            critic_traces=_critic_traces,
            handoff_traces=_handoff_traces,
            _cancel_token=self._cancel_token,
        )
        self._notify_run_end(result)
        self._clear_active_context()
        return result

    def run(
        self,
        task: str | Task,
        *,
        history_snapshot: HistorySnapshot | None = None,
        **kwargs: Any,
    ) -> EngineResult[StateT]:
        """Run from a synchronous application boundary.

        Async applications must await :meth:`arun`; a nested event loop is
        never created.
        """

        return self._run_sync(
            self.arun(task, history_snapshot=history_snapshot, **kwargs),
            operation="run",
        )

    @staticmethod
    def _run_sync(awaitable: Any, *, operation: str) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable)
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        raise RuntimeError(
            f"Engine.{operation}() cannot run inside an active event loop; "
            f"await Engine.a{operation}() instead"
        )

    async def arun_stream(
        self,
        task: str | Task,
        *,
        transformers: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Run on the current event loop and yield structured Engine events."""

        from .events import EngineEvent, EngineEventHook, EngineEventType, EventStream
        from .stream.transformer import TransformerChain

        stream = EventStream()
        hook = EngineEventHook(stream)
        self.hooks.append(hook)
        chain = TransformerChain(transformers) if transformers else None
        if chain is not None:
            chain.on_run_start()

        async def execute() -> EngineResult[StateT]:
            try:
                return await self.arun(task, **kwargs)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                stream.emit(
                    EngineEvent(
                        event_type=EngineEventType.RUN_END,
                        ok=False,
                        payload={
                            "stop_reason": "error",
                            "error_type": type(exc).__name__,
                        },
                        error=str(exc),
                    )
                )
                raise
            finally:
                stream.close()

        run_task = asyncio.create_task(execute(), name="qitos-engine-run")
        exhausted = False
        try:
            async for event in stream:
                if chain is None:
                    yield event
                    continue
                for output in await chain.aprocess(event):
                    yield output
            exhausted = True
        finally:
            self.hooks = [existing for existing in self.hooks if existing is not hook]
            if chain is not None:
                chain.on_run_end()
            if not exhausted and not run_task.done():
                run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                if exhausted:
                    raise

    async def arun_stream_tokens(
        self,
        task: str | Task,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Yield the same event stream, including token-level model events."""

        async for event in self.arun_stream(task, **kwargs):
            yield event

    def _apply_task_budget(self, task_obj: Optional[Task]) -> None:
        self.budget.max_steps = self._base_budget.max_steps
        self.budget.max_runtime_seconds = self._base_budget.max_runtime_seconds
        self.budget.max_tokens = self._base_budget.max_tokens
        self.budget.max_cost_usd = self._base_budget.max_cost_usd
        self.budget.max_tool_concurrency = self._base_budget.max_tool_concurrency
        self.budget.max_children = self._base_budget.max_children
        self.budget.deadline_monotonic = self._base_budget.deadline_monotonic
        if task_obj is not None:
            budget = task_obj.budget
            if budget.max_steps is not None:
                self.budget.max_steps = min(
                    self.budget.max_steps, int(budget.max_steps)
                )
            if budget.max_runtime_seconds is not None:
                requested_runtime = float(budget.max_runtime_seconds)
                self.budget.max_runtime_seconds = (
                    requested_runtime
                    if self.budget.max_runtime_seconds is None
                    else min(self.budget.max_runtime_seconds, requested_runtime)
                )
            if budget.max_tokens is not None:
                requested_tokens = int(budget.max_tokens)
                self.budget.max_tokens = (
                    requested_tokens
                    if self.budget.max_tokens is None
                    else min(self.budget.max_tokens, requested_tokens)
                )
            if budget.max_cost_usd is not None:
                requested_cost = float(budget.max_cost_usd)
                self.budget.max_cost_usd = (
                    requested_cost
                    if self.budget.max_cost_usd is None
                    else min(self.budget.max_cost_usd, requested_cost)
                )
            if budget.max_tool_concurrency is not None:
                self.budget.max_tool_concurrency = min(
                    self.budget.max_tool_concurrency,
                    int(budget.max_tool_concurrency),
                )
            if budget.max_children is not None:
                self.budget.max_children = min(
                    self.budget.max_children,
                    int(budget.max_children),
                )
        if self._uses_default_stop_criteria:
            self.stop_criteria = [FinalResultCriteria()]

    # -- Configuration export --------------------------------------------------

    def export_config(self) -> EngineConfig:
        """Return a serializable snapshot of this Engine's configuration."""
        return EngineConfig(
            agent_name=getattr(self.agent, "name", "") or "",
            model_id=getattr(self, "_resolved_model_id", "") or "",
            budget_max_steps=self.budget.max_steps,
            budget_max_runtime_seconds=self.budget.max_runtime_seconds,
            budget_max_tokens=self.budget.max_tokens,
            critic_names=[type(c).__name__ for c in self.critics],
            stop_criteria_names=[type(s).__name__ for s in self.stop_criteria],
            has_checkpoint_store=self._checkpoint_store is not None,
            has_tracing_provider=self._tracing_provider is not None,
            protocol_id=getattr(self, "_resolved_protocol_id", None),
            delegate_depth=self._delegate_depth,
            has_shared_memory=self._shared_memory is not None,
            has_env=self.env is not None,
            tool_count=len(self.tool_registry) if self.tool_registry else 0,
        )

    # -- Trace extraction helpers ----------------------------------------------

    def _extract_critic_traces(self) -> List[CriticTrace]:
        """Extract structured CriticTrace entries from step records."""
        traces: List[CriticTrace] = []
        for record in self.records:
            for output in list(getattr(record, "critic_outputs", []) or []):
                if not isinstance(output, dict):
                    continue
                traces.append(
                    CriticTrace(
                        step_id=record.step_id,
                        critic_name=str(output.get("critic_name", "unknown")),
                        action=str(output.get("action", "continue")),
                        reason=str(output.get("reason", "")),
                        score=float(output.get("score", 1.0)),
                        details=output.get("details", {}),
                        instruction_patch=output.get("instruction_patch"),
                        state_patch=output.get("state_patch"),
                    )
                )
        return traces

    def _extract_handoff_traces(self) -> List[HandoffTrace]:
        """Extract structured HandoffTrace entries from runtime events."""
        traces: List[HandoffTrace] = []
        for event in self.events:
            if event.phase != RuntimePhase.HANDOFF_START:
                continue
            payload = event.payload or {}
            traces.append(
                HandoffTrace(
                    step_id=event.step_id,
                    from_agent=str(payload.get("from", "")),
                    to_agent=str(payload.get("to", "")),
                    context_strategy=str(payload.get("context_strategy", "")),
                    messages_passed=int(payload.get("messages_passed", 0)),
                )
            )
        return traces

    def _build_env_view(
        self, state: StateT, step_id: int, started_at: float
    ) -> Dict[str, Any]:
        return self._env_runtime.build_env_view(state, step_id, started_at)

    def _build_initial_observation(
        self, state: StateT, step_id: int, started_at: float
    ) -> ObservationT:
        return self._env_runtime.build_initial_observation(state, step_id, started_at)

    def _build_observation_after_action(
        self,
        state: StateT,
        step_id: int,
        started_at: float,
        decision: Decision[ActionT],
        action_results: List[Any],
    ) -> ObservationT:
        return self._env_runtime.build_observation_after_action(
            state, step_id, started_at, decision, action_results
        )

    async def _run_decide(
        self,
        state: StateT,
        observation: ObservationT,
        record: StepRecord,
        turn: TurnSnapshot,
    ) -> Decision[ActionT]:
        # Propagate streaming callback to model runtime
        self._model_runtime.stream_callback = self.stream_callback
        try:
            return await self._model_runtime.run_decide(
                state,
                observation,
                record,
                turn,
            )
        finally:
            self._model_runtime.stream_callback = None

    def _select_branch(
        self,
        state: StateT,
        observation: ObservationT,
        branch_decision: Decision[ActionT],
    ) -> Decision[ActionT]:
        return self._decision_runtime.select_branch(
            state, observation, branch_decision
        )

    async def _run_act(
        self,
        state: StateT,
        decision: Decision[ActionT],
        record: StepRecord,
        turn: TurnSnapshot,
    ) -> List[Any]:
        return await self._action_runtime.run_act(state, decision, record, turn)

    def _run_reduce(
        self,
        state: StateT,
        observation: ObservationT,
        decision: Decision[ActionT],
        record: StepRecord,
    ) -> None:
        self._control_runtime.run_reduce(state, observation, decision, record)

    def _apply_critics(self, state: StateT, record: StepRecord) -> Any:
        return self._control_runtime.apply_critics(state, record)

    def _apply_critic_patches(
        self, state: StateT, critic_result: Dict[str, Any]
    ) -> None:
        """Apply modified_prompt, instruction_patch, and state_patch from critic retry."""
        # Store patches so they can be picked up by the next decide() call
        modified_prompt = critic_result.get("modified_prompt")
        instruction_patch = critic_result.get("instruction_patch")
        state_patch = critic_result.get("state_patch")

        if modified_prompt is not None:
            self._critic_modified_prompt = modified_prompt
        if instruction_patch is not None:
            self._critic_instruction_patch = instruction_patch
        if state_patch is not None:
            for key, value in state_patch.items():
                if hasattr(state, key):
                    setattr(state, key, value)

    def _run_check_stop(
        self,
        state: StateT,
        decision: Decision[ActionT],
        step_id: int,
        started_at: float,
    ) -> bool:
        return self._control_runtime.run_check_stop(
            state, decision, step_id, started_at
        )

    def _finish_check_stop(
        self,
        step_id: int,
        state: StateT,
        decision: Decision[ActionT],
        stop: bool,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._control_runtime._finish_check_stop(
            step_id, state, decision, stop, extra_payload
        )

    def _should_stop_by_criteria(
        self, state: StateT, step_id: int, elapsed_seconds: float
    ) -> tuple[bool, Optional[StopReason], Optional[str]]:
        return self._control_runtime.should_stop_by_criteria(
            state, step_id, elapsed_seconds
        )

    def _budget_exhausted(self, step_id: int, state: StateT) -> bool:
        return self._control_runtime.budget_exhausted(step_id, state)

    def _drain_runtime_events(self, step_id: int) -> None:
        events = self._runtime_inbox.drain(self._active_run_id)
        if not events:
            return
        payload = {"runtime_events": [event.to_dict() for event in events]}
        self._history_append(
            "user",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            step_id,
            metadata={"source": "runtime", "event_count": len(events)},
        )
        self._emit(
            step_id,
            RuntimePhase.DECIDE,
            payload={"stage": "runtime_input", **payload},
        )

    async def _wait_for_runtime_event(self) -> RuntimeWaitOutcome:
        timeout_seconds = self.remaining_runtime_seconds()
        return await self._runtime_inbox.wait(
            self._active_run_id,
            timeout_seconds=timeout_seconds,
            cancelled=lambda: self._cancel_token.is_cancel_requested,
        )

    def _normalize_decision(self, raw_decision: Any, step: int) -> Decision[ActionT]:
        return self._decision_runtime.normalize_decision(raw_decision, step)

    def _compute_state_diff(
        self, before: Dict[str, Any], after: Dict[str, Any]
    ) -> Dict[str, Any]:
        diff: Dict[str, Any] = {}
        all_keys = set(before.keys()) | set(after.keys())
        for key in all_keys:
            b = before.get(key)
            a = after.get(key)
            if b != a:
                diff[key] = {"before": b, "after": a}
        return diff

    def _report_runtime_exception(
        self,
        phase: RuntimePhase | str,
        step_id: int,
        exc: Exception,
        *,
        emit: bool = True,
    ) -> None:
        """Make handled runtime exceptions visible in stderr, traces, and a file.

        Python normally prints only uncaught exceptions. Recovery deliberately
        catches exceptions, so without this explicit report a run can end with
        ``unrecoverable_error`` and no useful diagnostics in redirected logs.
        """
        phase_name = getattr(phase, "value", str(phase))
        traceback_text = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        payload = {
            "phase": phase_name,
            "step_id": int(step_id),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback_text,
        }
        self._last_runtime_error = payload

        print(
            f"[QitOS] runtime exception phase={phase_name} step={step_id} "
            f"type={type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        print(traceback_text, file=sys.stderr, end="", flush=True)

        error_log = os.environ.get("QITOS_ERROR_LOG", "").strip()
        if not error_log:
            trace_dir = os.environ.get("QITOS_TRACE_DIR", "").strip()
            if trace_dir:
                error_log = str(Path(trace_dir) / "step_error.log")
        if error_log:
            try:
                path = Path(error_log)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        f"\n{'=' * 60}\nQitOS RUNTIME EXCEPTION "
                        f"phase={phase_name} step={step_id}\n"
                    )
                    stream.write(traceback_text)
                    stream.flush()
            except Exception as log_exc:
                _logger.warning(
                    "Failed to write QitOS error log %s: %s", error_log, log_exc
                )

        if emit:
            try:
                self._emit(
                    int(step_id),
                    RuntimePhase.RECOVER,
                    ok=False,
                    payload=payload,
                    error=str(exc),
                )
            except Exception as emit_exc:
                _logger.warning(
                    "Failed to emit QitOS recovery diagnostic: %s", emit_exc
                )

    def _recover(self, state: StateT, phase: RuntimePhase, exc: Exception) -> bool:
        step_id = int(getattr(state, "current_step", len(self.records) - 1) or 0)
        self._report_runtime_exception(phase, step_id, exc)
        return self._control_runtime.recover(state, phase, exc)

    def _emit(
        self,
        step_id: int,
        phase: RuntimePhase,
        ok: bool = True,
        payload: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        self._trace_runtime.emit(step_id, phase, ok=ok, payload=payload, error=error)

    def _write_trace_step(self, step: StepRecord) -> None:
        self._trace_runtime.write_trace_step(step)

    def _finalize_step(self, record: StepRecord, state: StateT) -> None:
        self._trace_runtime.finalize_step(record, state)

    async def _journal_initialize(
        self,
        task_obj: Task | None,
        task_text: str,
        state: StateT,
    ) -> None:
        await self._journal_runtime.initialize(task_obj, task_text, state)

    async def _journal_model_completed(
        self,
        record: StepRecord,
        decision: Decision[ActionT],
    ) -> None:
        await self._journal_runtime.model_completed(record, decision)

    async def _journal_tool_starts(
        self,
        indexed_actions: List[tuple[int, Action]],
        record: StepRecord,
    ) -> None:
        await self._journal_runtime.tool_starts(indexed_actions, record)

    async def _finalize_action_results(
        self,
        state: StateT,
        actions: List[Action],
        results: List[ToolResult],
        *,
        record: StepRecord,
    ) -> List[ToolResult]:
        return await self._journal_runtime.finalize_action_results(
            state,
            actions,
            results,
            record=record,
        )

    def _reduce_action_results(
        self,
        state: StateT,
        actions: List[Action],
        results: List[ToolResult],
        step_id: int,
    ) -> None:
        self._journal_runtime.reduce_action_results(state, actions, results, step_id)

    async def _journal_commit_step(
        self,
        record: StepRecord,
        *,
        before: Dict[str, Any],
        state: StateT,
        terminal: bool,
    ) -> None:
        await self._journal_runtime.commit_step(
            record,
            before=before,
            state=state,
            terminal=terminal,
        )

    async def _journal_snapshot_state(
        self,
        state: StateT,
        *,
        step_id: int,
        reason: str,
        record_id: str,
    ) -> None:
        await self._journal_runtime.snapshot_state(
            state,
            step_id=step_id,
            reason=reason,
            record_id=record_id,
        )

    async def _journal_interrupt_run(self, *, step_id: int, reason: str) -> None:
        await self._journal_runtime.interrupt_run(step_id=step_id, reason=reason)

    async def _journal_finish_run(self, state: StateT) -> None:
        await self._journal_runtime.finish_run(state)

    async def _save_checkpoint(
        self,
        step_id: int,
        state: StateT,
        task_text: str,
        source: str = "loop",
    ) -> None:
        """Persist one recoverable safe-boundary snapshot before returning."""
        if self._checkpoint_store is None:
            return

        task_data = (
            self._active_task_obj.to_dict()
            if isinstance(self._active_task_obj, Task)
            else None
        )
        checkpoint = Checkpoint(
            id=CheckpointId(uuid4().hex),
            thread_id=self._active_run_id,
            step=step_id,
            state_data=state.to_dict(),
            task_text=task_text,
            task_data=task_data,
            history=self._history().snapshot(),
            parent_id=self._last_checkpoint_id,
            parent_thread_id=(
                self._active_run_id if self._last_checkpoint_id is not None else None
            ),
        )

        metadata: CheckpointMetadata = {
            "source": source,
            "step": step_id,
            "run_id": self._active_run_id,
        }

        config = CheckpointConfig(thread_id=self._active_run_id)
        await self._checkpoint_store.put(config, checkpoint, metadata)
        self._last_checkpoint_id = checkpoint.id

    async def aresume_from_journal(self, run_id: str) -> EngineResult[StateT]:
        """Resume one Run from its canonical journal."""

        try:
            result = await self._aresume_from_journal_impl(run_id)
        except BaseException:
            await self._close_journal_after_error()
            raise
        else:
            await self._close_journal()
            return result

    async def _aresume_from_journal_impl(self, run_id: str) -> EngineResult[StateT]:
        """Resume implementation with lifecycle ownership held by the caller."""

        journal = self.journal
        if journal is None:
            raise RuntimeError("No journal configured; cannot resume.")
        if journal.run_id:
            if journal.run_id != run_id:
                raise JournalError("journal is already open for another Run")
        else:
            await journal.open(run_id)
        replay = self._journal_runtime.replay(await journal.replay())
        for recovery in replay["recovered_terminals"]:
            self._last_journal_position = await journal.append(
                JournalRecordType.TOOL_TERMINAL,
                recovery["payload"],
                record_id=recovery["record_id"],
            )
        if replay["recovered_terminals"]:
            replay = self._journal_runtime.replay(await journal.replay())
        task_text = replay["task"]
        task_data = replay["task_data"]
        task: str | Task = (
            Task.from_dict(task_data) if isinstance(task_data, dict) else task_text
        )
        state_type = type(self.agent.init_state(task_text))
        state = self.agent.restore_state(state_type.from_dict(replay["state"]))
        history = HistorySnapshot.from_messages(replay["history"])
        for recovery in replay["recovered_steps"]:
            self._last_journal_position = await journal.append(
                JournalRecordType.STEP_COMMITTED,
                recovery["payload"],
                record_id=recovery["record_id"],
            )
        if state.stop_reason and not replay["completed"]:
            self._active_run_id = run_id
            if not replay["terminal_snapshot_current"]:
                await self._journal_runtime.snapshot_state(
                    state,
                    step_id=int(state.current_step),
                    reason="terminal",
                    record_id=f"{run_id}:snapshot:terminal",
                )
            await self._journal_finish_run(state)
            replay["completed"] = True
        if replay["completed"]:
            self._active_run_id = run_id
            self._active_task = task_text
            self._active_task_obj = task if isinstance(task, Task) else None
            self._active_state = state
            self._token_usage = int(replay["usage"]["total_tokens"])
            self._cost_usage_usd = float(replay["usage"]["cost_usd"])
            self._context_runtime.restore_usage(
                prompt_tokens=int(replay["usage"]["prompt_tokens"]),
                completion_tokens=int(replay["usage"]["completion_tokens"]),
                total_tokens=self._token_usage,
            )
            replayed_records = await journal.replay()
            self._last_journal_position = replayed_records[-1].position
            self._reset_history(history)
            self._hydrate_trace_metadata(
                task_obj=self._active_task_obj,
                task_text=task_text,
            )
            self._notify_run_start(task_text, state)
            result = EngineResult(
                state=state,
                records=replay["records"],
                events=[],
                step_count=len(replay["records"]),
                total_tokens=self._token_usage,
                total_cost_usd=self._cost_usage_usd,
                run_id=run_id,
            )
            self._notify_run_end(result)
            if self.trace_writer is not None:
                self.trace_writer.finalize(
                    status=self._trace_status(state.stop_reason),
                    summary={
                        "stop_reason": state.stop_reason,
                        "final_result": state.final_result,
                        "steps": 0,
                        "task_meta": self._task_meta(self._active_task_obj),
                        "run_meta": self._run_meta(),
                        "failure_report": build_failure_report(
                            self.recovery_policy, state.stop_reason
                        ),
                    },
                )
            self._clear_active_context()
            return result
        return await self.arun(
            task,
            history_snapshot=history,
            _resume_state=state,
            _resume_step=state.current_step,
            _resume_run_id=run_id,
            _resume_journal=True,
            _resume_canonical_results=replay["canonical_results"],
            _resume_usage=replay["usage"],
            _resume_continuation=replay["continuation"],
        )

    def resume_from_journal(self, run_id: str) -> EngineResult[StateT]:
        """Resume a journal Run from a synchronous application boundary."""

        return self._run_sync(
            self.aresume_from_journal(run_id),
            operation="resume_from_journal",
        )

    async def afork_journal(
        self,
        run_id: str,
        position: JournalPosition,
        *,
        new_run_id: str | None = None,
    ) -> SessionJournal:
        """Fork a Run at one committed journal boundary."""

        journal = self.journal
        if journal is None:
            raise RuntimeError("No journal configured; cannot fork.")
        try:
            if journal.run_id:
                if journal.run_id != run_id:
                    raise JournalError("journal is already open for another Run")
            else:
                await journal.open(run_id)
            child_run_id = str(new_run_id or f"run_{uuid4().hex[:12]}")
            child = await journal.fork(position, child_run_id)
        except BaseException:
            await self._close_journal_after_error()
            raise
        else:
            try:
                await journal.close()
            except BaseException:
                try:
                    await child.close()
                except BaseException as exc:
                    _logger.error(
                        "Forked child Journal close failed while preserving the "
                        "source close error: %s",
                        exc,
                        exc_info=True,
                    )
                raise
            return child

    async def _close_journal(self) -> None:
        journal = self.journal
        if journal is not None:
            await journal.close()

    async def _close_journal_after_error(self) -> None:
        try:
            await self._close_journal()
        except Exception as exc:
            _logger.warning("Journal close failed while handling an error: %s", exc)

    async def aresume_from_checkpoint(
        self,
        config: CheckpointConfig,
    ) -> EngineResult:
        """Resume a run from a saved checkpoint.

        Args:
            config: CheckpointConfig pointing to the checkpoint to resume from.

        Returns:
            EngineResult from the resumed run.
        """
        if self._checkpoint_store is None:
            raise RuntimeError("No checkpoint_store configured; cannot resume.")

        tuple_ = await self._checkpoint_store.get_tuple(config)
        if tuple_ is None:
            raise ValueError(f"Checkpoint not found: {config}")

        checkpoint = tuple_.checkpoint
        task: str | Task = (
            Task.from_dict(checkpoint.task_data)
            if checkpoint.task_data is not None
            else checkpoint.task_text
        )
        task_text = task.objective if isinstance(task, Task) else task
        state_type: type[StateSchema] = (
            type(self._active_state)
            if self._active_state is not None
            else type(self.agent.init_state(task_text))
        )
        state = state_type.from_dict(checkpoint.state_data)
        self._last_checkpoint_id = checkpoint.id
        if state.stop_reason:
            if self.trace_writer is not None:
                task_obj = task if isinstance(task, Task) else None
                self._hydrate_trace_metadata(task_obj=task_obj, task_text=task_text)
                self.trace_writer.finalize(
                    status=self._trace_status(state.stop_reason),
                    summary={
                        "stop_reason": state.stop_reason,
                        "final_result": state.final_result,
                        "steps": 0,
                        "task_meta": self._task_meta(task_obj),
                        "run_meta": self._run_meta(),
                        "failure_report": build_failure_report(
                            self.recovery_policy, state.stop_reason
                        ),
                    },
                )
            return EngineResult(
                state=state,
                records=[],
                events=[],
                step_count=0,
                run_id=checkpoint.thread_id,
            )
        resume_step = checkpoint.step + 1
        if tuple_.metadata.get("source") == "input":
            resume_step = checkpoint.step

        return await self.arun(
            task,
            history_snapshot=checkpoint.history,
            _resume_state=state,
            _resume_step=resume_step,
            _resume_run_id=checkpoint.thread_id,
            _resume_checkpoint_id=checkpoint.id,
        )

    def resume_from_checkpoint(
        self,
        config: CheckpointConfig,
    ) -> EngineResult:
        """Resume from a synchronous application boundary."""

        return self._run_sync(
            self.aresume_from_checkpoint(config),
            operation="resume_from_checkpoint",
        )

    async def aresume(
        self,
        checkpoint_id: CheckpointId,
        resume_value: Any = None,
        resume_values: Optional[Dict[str, Any]] = None,
    ) -> EngineResult:
        """Resume an interrupted run.

        Args:
            checkpoint_id: The checkpoint to resume from.
            resume_value: Value to pass to the first ``interrupt()`` call.
            resume_values: Dict mapping interrupt IDs to values for
                multiple interrupts.

        Returns:
            EngineResult from the resumed run.
        """
        from .interrupt import _set_resume_values

        # Prepare resume values
        values: Dict[str, Any] = dict(resume_values or {})
        if resume_value is not None and not values:
            # Default: map to the first interrupt
            values["int_1"] = resume_value

        _set_resume_values(values)

        config = CheckpointConfig(
            thread_id=self._active_run_id,
            checkpoint_id=checkpoint_id,
        )
        return await self.aresume_from_checkpoint(config)

    def resume(
        self,
        checkpoint_id: CheckpointId,
        resume_value: Any = None,
        resume_values: Optional[Dict[str, Any]] = None,
    ) -> EngineResult:
        """Resume an interrupt from a synchronous application boundary."""

        return self._run_sync(
            self.aresume(
                checkpoint_id,
                resume_value=resume_value,
                resume_values=resume_values,
            ),
            operation="resume",
        )

    async def _save_interrupt_checkpoint(
        self,
        step_id: int,
        state: StateT,
        interrupt_exc: Any,
    ) -> CheckpointId:
        """Save a checkpoint when an interrupt fires.  Returns the checkpoint ID."""
        from .interrupt import EngineInterrupt

        if self._checkpoint_store is None:
            # No store configured — generate a transient ID
            return CheckpointId(uuid4().hex)

        await self._save_checkpoint(
            step_id, state, self._active_task, source="interrupt"
        )
        # Update the interrupt exception with the checkpoint ID
        if (
            isinstance(interrupt_exc, EngineInterrupt)
            and self._last_checkpoint_id is not None
        ):
            interrupt_exc.checkpoint_id = self._last_checkpoint_id
            return self._last_checkpoint_id
        return CheckpointId(uuid4().hex)

    def _memory_append(
        self,
        role: str,
        content: Any,
        step_id: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        memory = self._memory()
        if memory is None:
            return
        memory.append(
            MemoryRecord(
                role=role, content=content, step_id=step_id, metadata=metadata or {}
            )
        )

    def _memory(self) -> Optional[Memory]:
        mem = getattr(self.agent, "memory", None)
        return mem if isinstance(mem, Memory) else None

    def _history(self) -> History:
        hist = getattr(self.agent, "history", None)
        if isinstance(hist, History):
            return hist
        if getattr(self.agent, "llm", None) is not None and self.context_config.enabled:
            try:
                from ..kit.history import CompactHistory

                if not isinstance(self._runtime_history, CompactHistory):
                    self._runtime_history = CompactHistory(
                        llm=getattr(self.agent, "llm", None),
                        max_tokens=max(
                            1024,
                            int(
                                (
                                    self._context_runtime.resolve_request_budget(
                                        getattr(self.agent, "llm", None)
                                    ).get("available_input_budget")
                                    or 16000
                                )
                            ),
                        ),
                    )
            except Exception as exc:
                _logger.debug("Failed to append context telemetry to history: %s", exc)
        return self._runtime_history

    def _reset_history(self, snapshot: HistorySnapshot | None) -> None:
        history = self._history()
        try:
            history.reset()
        except Exception:
            if snapshot is not None:
                raise
            _logger.debug("Failed to reset history", exc_info=True)
            return
        if snapshot is None:
            return
        try:
            history.restore(snapshot)
        except NotImplementedError as exc:
            raise TypeError(
                f"{type(history).__name__} cannot restore a history snapshot"
            ) from exc

    def _history_append(
        self,
        role: str,
        content: Any,
        step_id: int,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        reasoning_content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
        name: Optional[str] = None,
        native_items: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        history = self._history()
        message = HistoryMessage(
            role=role,
            content=content,
            step_id=step_id,
            reasoning_content=reasoning_content,
            metadata=metadata or {},
            tool_calls=[
                dict(x) for x in list(tool_calls or []) if isinstance(x, dict)
            ],
            tool_call_id=tool_call_id,
            name=name,
            native_items=[
                dict(x) for x in list(native_items or []) if isinstance(x, dict)
            ],
        )
        history.append(message)
        if self.journal is not None:
            self._journal_pending_history.append(history_message_to_dict(message))

    def _normalize_history_messages(self, payload: Any) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if not isinstance(payload, list):
            return messages
        for item in payload:
            if isinstance(item, HistoryMessage):
                role = str(item.role).strip()
                if not role:
                    continue
                message: Dict[str, Any] = {"role": role, "content": item.content}
                message["_step_id"] = int(item.step_id)
                if item.reasoning_content:
                    message["reasoning_content"] = item.reasoning_content
                if item.metadata:
                    message["_metadata"] = dict(item.metadata)
                if item.tool_calls:
                    message["tool_calls"] = [dict(x) for x in item.tool_calls]
                if item.tool_call_id:
                    message["tool_call_id"] = str(item.tool_call_id)
                if item.name:
                    message["name"] = str(item.name)
                if item.native_items:
                    message["native_items"] = [dict(x) for x in item.native_items]

                if role not in {"assistant", "tool"}:
                    content = str(item.content or "")
                    if not content:
                        continue
                    message["content"] = content
                elif (
                    message.get("content") in (None, "")
                    and not message.get("tool_calls")
                    and not message.get("tool_call_id")
                    and not message.get("native_items")
                ):
                    continue
                messages.append(message)
                continue
            if isinstance(item, dict):
                role = str(item.get("role", "")).strip()
                if not role:
                    continue
                payload_message: Dict[str, Any] = {
                    "role": role,
                    "content": item.get("content"),
                }
                reasoning_content = item.get("reasoning_content")
                if isinstance(reasoning_content, str) and reasoning_content:
                    payload_message["reasoning_content"] = reasoning_content
                step_value = item.get("step_id")
                if step_value is not None:
                    try:
                        payload_message["_step_id"] = int(step_value)
                    except Exception as exc:
                        _logger.debug("Failed to parse step_id: %s", exc)
                item_metadata = item.get("metadata")
                if isinstance(item_metadata, dict) and item_metadata:
                    payload_message["_metadata"] = dict(item_metadata)
                tool_calls = item.get("tool_calls")
                if isinstance(tool_calls, list):
                    payload_message["tool_calls"] = [
                        dict(x) for x in tool_calls if isinstance(x, dict)
                    ]
                if item.get("tool_call_id") not in (None, ""):
                    payload_message["tool_call_id"] = str(item.get("tool_call_id"))
                if item.get("name") not in (None, ""):
                    payload_message["name"] = str(item.get("name"))
                native_items = item.get("native_items")
                if isinstance(native_items, list):
                    payload_message["native_items"] = [
                        dict(x) for x in native_items if isinstance(x, dict)
                    ]

                if role not in {"assistant", "tool"}:
                    content = str(payload_message.get("content") or "")
                    if not content:
                        continue
                    payload_message["content"] = content
                elif (
                    payload_message.get("content") in (None, "")
                    and not payload_message.get("tool_calls")
                    and not payload_message.get("tool_call_id")
                    and not payload_message.get("native_items")
                ):
                    continue
                messages.append(payload_message)
        return messages

    def _hook_context(self, **kwargs: Any) -> HookContext:
        return HookContext(task=self._active_task, **kwargs)

    def _infer_failed_phase(self, record: StepRecord) -> RuntimePhase:
        return self._control_runtime.infer_failed_phase(record)

    def _normalize_task(self, task: str | Task) -> tuple[Optional[Task], str]:
        if isinstance(task, Task):
            return task, task.objective
        return None, str(task)

    def _preflight_validate(
        self, task_obj: Optional[Task], workspace: Any = None
    ) -> List[TaskValidationIssue]:
        issues: List[TaskValidationIssue] = []
        if task_obj is not None:
            try:
                issues.extend(
                    task_obj.validate_structured(
                        workspace=str(workspace) if workspace else None
                    )
                )
            except Exception as exc:
                issues.append(
                    TaskValidationIssue(
                        code="TASK_VALIDATION_EXCEPTION",
                        message=str(exc),
                        field="task",
                    )
                )

        for issue in self._validate_env_capabilities():
            issues.append(
                TaskValidationIssue(
                    code=str(issue.get("code", "ENV_CAPABILITY_ERROR")),
                    message=str(
                        issue.get("message", "Environment capability mismatch")
                    ),
                    field=str(issue.get("field", "env")),
                    details=(
                        issue.get("details", {})
                        if isinstance(issue.get("details", {}), dict)
                        else {}
                    ),
                )
            )
        health = self._validate_env_health()
        if health is not None:
            issues.append(
                TaskValidationIssue(
                    code=str(health.get("code", "ENV_HEALTH_CHECK_FAILED")),
                    message=str(
                        health.get("message", "Environment health check failed")
                    ),
                    field=str(health.get("field", "env")),
                    details=(
                        health.get("details", {})
                        if isinstance(health.get("details", {}), dict)
                        else {}
                    ),
                )
            )
        return issues

    def _validate_env_capabilities(self) -> List[Dict[str, Any]]:
        return self._env_runtime.validate_env_capabilities()

    def _collect_required_ops(self) -> set[str]:
        return self._env_runtime.collect_required_ops()

    def _validate_env_health(self) -> Optional[Dict[str, Any]]:
        return self._env_runtime.validate_env_health()

    def _setup_env(
        self, task_obj: Optional[Task], state: StateT, kwargs: Dict[str, Any]
    ) -> None:
        self._env_runtime.setup_env(task_obj, state, kwargs)

    def _build_env_from_spec(
        self, env_spec: Any, fallback_workspace: Any = None
    ) -> Optional[Env]:
        return self._env_runtime.build_env_from_spec(env_spec, fallback_workspace)

    async def _teardown_env(self) -> None:
        await self._env_runtime.teardown_env()

    def _run_env_step(
        self,
        decision: Decision[ActionT],
        action_results: List[Any],
        *,
        state: StateT | None = None,
    ) -> Optional[EnvStepResult]:
        return self._env_runtime.run_env_step(decision, action_results, state=state)

    def _env_payload(self) -> Dict[str, Any]:
        return self._env_runtime.env_payload()

    def _env_identity(self) -> Dict[str, Any]:
        return self._env_runtime.env_identity()

    def _env_observation_to_dict(
        self, observation: Optional[EnvObservation]
    ) -> Optional[Dict[str, Any]]:
        return self._env_runtime.env_observation_to_dict(observation)

    def _env_step_result_to_dict(
        self, result: Optional[EnvStepResult]
    ) -> Optional[Dict[str, Any]]:
        return self._env_runtime.env_step_result_to_dict(result)

    def _setup_toolsets(self, context: Dict[str, Any]) -> None:
        if not hasattr(self.tool_registry, "setup"):
            return
        self._write_lifecycle_event("toolset_setup_start", context)
        try:
            self.tool_registry.setup(context)
            self._write_lifecycle_event("toolset_setup_end", context)
        except Exception as exc:
            self._write_lifecycle_event(
                "toolset_setup_error", context, ok=False, error=str(exc)
            )

    async def _asetup_toolsets(self, context: Dict[str, Any]) -> None:
        if not hasattr(self.tool_registry, "asetup"):
            self._setup_toolsets(context)
            return
        self._write_lifecycle_event("toolset_setup_start", context)
        try:
            await self.tool_registry.asetup(context)
            self._write_lifecycle_event("toolset_setup_end", context)
        except Exception as exc:
            self._write_lifecycle_event(
                "toolset_setup_error", context, ok=False, error=str(exc)
            )
            raise

    def _teardown_toolsets(self, context: Dict[str, Any]) -> None:
        if not hasattr(self.tool_registry, "teardown"):
            return
        self._write_lifecycle_event("toolset_teardown_start", context)
        try:
            self.tool_registry.teardown(context)
            self._write_lifecycle_event("toolset_teardown_end", context)
        except Exception as exc:
            self._write_lifecycle_event(
                "toolset_teardown_error", context, ok=False, error=str(exc)
            )

    async def _ateardown_toolsets(self, context: Dict[str, Any]) -> None:
        if not hasattr(self.tool_registry, "ateardown"):
            self._teardown_toolsets(context)
            return
        self._write_lifecycle_event("toolset_teardown_start", context)
        try:
            await self.tool_registry.ateardown(context)
            self._write_lifecycle_event("toolset_teardown_end", context)
        except Exception as exc:
            self._write_lifecycle_event(
                "toolset_teardown_error", context, ok=False, error=str(exc)
            )

    def _write_lifecycle_event(
        self,
        phase: str,
        payload: Dict[str, Any],
        ok: bool = True,
        error: Optional[str] = None,
    ) -> None:
        self._trace_runtime.write_lifecycle_event(phase, payload, ok=ok, error=error)

    def _estimate_tokens(self, payload: Any) -> int:
        text = payload if isinstance(payload, str) else repr(payload)
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _task_meta(self, task_obj: Optional[Task]) -> Optional[Dict[str, Any]]:
        return self._trace_runtime.task_meta(task_obj)

    def _task_issue_to_dict(self, issue: TaskValidationIssue) -> Dict[str, Any]:
        return self._trace_runtime.task_issue_to_dict(issue)

    def _hydrate_trace_metadata(self, task_obj: Optional[Task], task_text: str) -> None:
        self._trace_runtime.hydrate_trace_metadata(task_obj, task_text)

    def _run_meta(self) -> Dict[str, Any]:
        return self._trace_runtime.run_meta()

    def _build_task_result(
        self, state: StateT, task_obj: Optional[Task], started_at: float
    ) -> TaskResult:
        return self._trace_runtime.build_task_result(state, task_obj, started_at)

    def _sanitize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._trace_runtime.sanitize_payload(payload)

    @staticmethod
    def _trace_status(stop_reason: str | None) -> str:
        if stop_reason == StopReason.UNRECOVERABLE_ERROR.value:
            return "failed"
        if stop_reason == StopReason.CANCELLED_IMMEDIATE.value:
            return "stopped"
        return "completed"

    def _notify_event(self, event: RuntimeEvent, state: StateT) -> None:
        self._trace_runtime.notify_event(event, state)

    def _notify_run_start(self, task: str, state: StateT) -> None:
        self._trace_runtime.notify_run_start(task, state)

    def _notify_run_end(self, result: EngineResult[StateT]) -> None:
        self._trace_runtime.notify_run_end(result)

    def _dispatch_hook(self, method_name: str, ctx: HookContext) -> None:
        self._trace_runtime.dispatch_hook(method_name, ctx)

    def _inject_hook_payload(self, method_name: str, ctx: HookContext) -> None:
        self._trace_runtime.inject_hook_payload(method_name, ctx)

    def _reset_run_state(self) -> None:
        self._runtime_inbox.close()
        self._trace_runtime.reset_run_state()
        self._last_runtime_error = None
        self._resolved_protocol = None
        self._resolved_protocol_source = ""
        self._last_prompt_metadata = {}
        self._runtime_deadline_monotonic = None
        self._cost_usage_usd = 0.0
        self._model_continuation = None
        self._last_checkpoint_id = None
        self._canonical_action_results = []
        self._journal_pending_history = []
        self._journal_terminal_record_ids = {}
        self._last_journal_position = None
        if self._tool_loop_detector is not None:
            self._tool_loop_detector.reset()
        self._handoff_history = []
        self._critic_modified_prompt = None
        self._critic_instruction_patch = None
        self._cancel_token.clear()

    # -- MCP server lifecycle helpers --

    async def _connect_mcp_servers(self) -> None:
        """Connect all configured MCP servers and bridge their tools."""
        from ..mcp.bridge import mcp_server_to_function_tools

        servers = list(getattr(self.agent, "mcp_servers", None) or [])
        if not servers:
            return

        used_names = set(
            self.tool_registry.list_tools() if self.tool_registry is not None else []
        )

        for server in servers:
            registered_names: List[str] = []

            async def _rollback_setup() -> None:
                for name in reversed(registered_names):
                    try:
                        self.tool_registry.unregister(name)
                    except Exception as unregister_exc:
                        _logger.warning(
                            "MCP tool rollback failed for %s: %s",
                            name,
                            unregister_exc,
                        )
                if hasattr(server, "cleanup"):
                    try:
                        await server.cleanup()
                    except Exception as cleanup_exc:
                        _logger.warning(
                            "MCP server cleanup after setup failure failed: %s",
                            cleanup_exc,
                        )

            try:
                if hasattr(server, "connect"):
                    await server.connect()
                # Bridge MCP tools into the engine's tool registry
                if self.tool_registry is not None:
                    tools = await mcp_server_to_function_tools(
                        server,
                        name_prefix=f"mcp__{server.name}",
                        used_names=used_names,
                    )
                    for tool in tools:
                        if hasattr(self.tool_registry, "register"):
                            self.tool_registry.register(tool)
                            registered_names.append(tool.name)
                self._connected_mcp_servers.append(server)
                self._mcp_tool_names.extend(registered_names)
            except asyncio.CancelledError:
                await _rollback_setup()
                raise
            except Exception as exc:
                await _rollback_setup()
                # One optional server must not prevent the remaining run.
                _logger.warning(
                    "MCP server %s could not be exposed: %s",
                    getattr(server, "name", "<unknown>"),
                    exc,
                )

    async def _cleanup_mcp_servers(self) -> None:
        """Remove run-scoped MCP tools and close transports on this loop."""
        for name in reversed(self._mcp_tool_names):
            try:
                if self.tool_registry is not None:
                    self.tool_registry.unregister(name)
            except Exception as exc:
                _logger.warning("MCP tool cleanup failed for %s: %s", name, exc)
        self._mcp_tool_names = []

        for server in self._connected_mcp_servers:
            try:
                if hasattr(server, "cleanup"):
                    await server.cleanup()
            except Exception as exc:
                _logger.warning("MCP server cleanup failed: %s", exc)
        self._connected_mcp_servers = []

    # -- Handoff tool helpers --

    def _register_handoff_tools(self) -> None:
        """Register HandoffTool for each declared handoff target."""
        from ..kit.tool.handoff_tool import HandoffTool

        targets = self.agent.handoff_targets or []
        for target_name in targets:
            # Resolve description from agent registry if available
            description = ""
            if self.agent_registry is not None:
                try:
                    spec = self.agent_registry.resolve(target_name)
                    description = getattr(spec, "description", "") or ""
                except Exception as exc:
                    _logger.debug(
                        "Failed to resolve handoff target %s: %s", target_name, exc
                    )

            tool = HandoffTool(
                target_name=target_name,
                target_description=description,
            )
            if hasattr(self.tool_registry, "register"):
                self.tool_registry.register(tool)
            self._handoff_tools.append(tool)

    def _intercept_handoff_action(self, action: Any) -> Any | None:
        """Check if an action is a handoff tool call. Return Decision.handoff() or None."""
        if not action.name.startswith("transfer_to_"):
            return None

        from ..core.decision import Decision

        target = action.name.replace("transfer_to_", "", 1)
        rationale = ""
        if isinstance(action.args, dict):
            rationale = action.args.get("rationale", "")

        return Decision.handoff(target=target, rationale=rationale)

    def _harness_mismatch_diagnostics(self) -> Dict[str, Any]:
        llm = getattr(self.agent, "llm", None)
        metadata = dict(getattr(llm, "qitos_harness_metadata", {}) or {})
        expected_protocol = str(metadata.get("protocol") or "").strip()
        expected_parser = str(metadata.get("parser") or "").strip()
        active_protocol = getattr(self.resolve_protocol(), "id", None)
        parser = self.parser or getattr(self.agent, "model_parser", None)
        active_parser = parser.__class__.__name__ if parser is not None else None
        mismatch_fields: List[str] = []
        if (
            expected_protocol
            and active_protocol
            and expected_protocol != active_protocol
        ):
            mismatch_fields.append("protocol")
        if expected_parser and active_parser and expected_parser != active_parser:
            mismatch_fields.append("parser")
        return {
            "mismatch": bool(mismatch_fields),
            "mismatch_fields": mismatch_fields,
            "expected_protocol": expected_protocol or None,
            "active_protocol": active_protocol,
            "expected_parser": expected_parser or None,
            "active_parser": active_parser,
            "model_name": getattr(llm, "model", None)
            or getattr(llm, "model_name", None),
        }

    def _clear_active_context(self) -> None:
        self._trace_runtime.clear_active_context()


__all__ = ["Engine", "EngineResult"]
