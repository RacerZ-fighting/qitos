"""Private control-flow helpers for Engine."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, TypeVar, cast

from ..core.decision import Decision
from ..core.completion import CompletionDisposition
from ..core.errors import (
    ErrorCategory,
    ModelRequestCancelled,
    ModelRequestDeadlineExceeded,
    QitosRuntimeError,
    RuntimeErrorInfo,
    StopReason,
)
from ..core.state import StateSchema
from ._context_runtime import (
    ContextCompactionRequired,
    ContextOverflowError,
    DecisionContextConfigurationError,
)
from .critic_result import CriticResult
from .states import RuntimePhase, StepRecord

if TYPE_CHECKING:
    from .engine import Engine


StateT = TypeVar("StateT", bound=StateSchema)
ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


class _ControlRuntime(Generic[StateT, ObservationT, ActionT]):
    def __init__(self, engine: Engine[StateT, ObservationT, ActionT]):
        self.engine = engine

    def run_reduce(
        self,
        state: StateT,
        observation: ObservationT,
        decision: Decision[ActionT],
        record: StepRecord,
    ) -> None:
        engine = self.engine
        engine._dispatch_hook(
            "on_before_reduce",
            engine._hook_context(
                step_id=record.step_id,
                phase=RuntimePhase.REDUCE,
                state=state,
                observation=observation,
                decision=decision,
                action_results=(record.action_results if record is not None else []),
                record=record,
            ),
        )
        engine._emit(record.step_id, RuntimePhase.REDUCE, payload={"stage": "start"})
        before = state.to_dict()
        new_state = engine.agent.reduce(state, observation, decision)
        if new_state is not state:
            state.reduce_update(new_state.__dict__)
        after = state.to_dict()
        engine._memory_append("next_state", after, record.step_id)
        record.state_diff = engine._compute_state_diff(before, after)
        engine._emit(
            record.step_id,
            RuntimePhase.REDUCE,
            payload={"stage": "state_reduced", "state_diff": record.state_diff},
        )
        engine._dispatch_hook(
            "on_after_reduce",
            engine._hook_context(
                step_id=record.step_id,
                phase=RuntimePhase.REDUCE,
                state=state,
                observation=observation,
                decision=decision,
                action_results=(record.action_results if record is not None else []),
                record=record,
                payload={"state_diff": record.state_diff},
            ),
        )

    def run_handoff(
        self,
        state: StateT,
        decision: Decision[ActionT],
        record: StepRecord,
    ) -> ObservationT:
        """Apply the optional legacy handoff policy outside the turn loop."""

        engine = self.engine
        if engine.agent_registry is None:
            raise ValueError("handoff requires agent_registry on Engine")
        target_name = str(decision.meta.get("handoff_target", ""))
        if target_name in engine._handoff_history:
            raise QitosRuntimeError(
                RuntimeErrorInfo(
                    category=ErrorCategory.SYSTEM,
                    message=(
                        f"Handoff loop detected: agent '{target_name}' already visited "
                        f"in this run (history: {' -> '.join(engine._handoff_history)})"
                    ),
                    phase="handoff",
                    step_id=record.step_id,
                    recoverable=False,
                )
            )
        max_handoffs = engine.context_config.max_handoffs
        if len(engine._handoff_history) >= max_handoffs:
            raise QitosRuntimeError(
                RuntimeErrorInfo(
                    category=ErrorCategory.SYSTEM,
                    message=f"Maximum handoff count ({max_handoffs}) exceeded",
                    phase="handoff",
                    step_id=record.step_id,
                    recoverable=False,
                )
            )

        engine._handoff_history.append(engine.agent.name)
        handoff = engine._handoff_runtime.execute_handoff(state, decision, record)
        state.metadata["last_handoff"] = {
            "from": handoff.from_agent,
            "to": handoff.to_agent,
        }
        if engine.trace_writer is not None:
            count = engine.trace_writer.metadata.get("handoff_count", 0) or 0
            engine.trace_writer.metadata["handoff_count"] = count + 1
        return cast(
            ObservationT,
            {
                "action_results": [
                    {
                        "handoff": True,
                        "from": handoff.from_agent,
                        "to": handoff.to_agent,
                    }
                ]
            },
        )

    def apply_critics(self, state: StateT, record: StepRecord) -> Dict[str, Any]:
        """Evaluate all critics and return a result dict with action + optional patches.

        Returns dict with keys:
        - action: "continue" | "stop" | "retry"
        - modified_prompt: str | None
        - instruction_patch: str | None
        - state_patch: dict | None
        - reason: str | None
        """
        engine = self.engine
        empty = {
            "action": "continue",
            "modified_prompt": None,
            "instruction_patch": None,
            "state_patch": None,
            "reason": None,
        }
        if not engine.critics:
            return empty
        engine._dispatch_hook(
            "on_before_critic",
            engine._hook_context(
                step_id=record.step_id,
                phase=RuntimePhase.CRITIC,
                state=state,
                decision=record.decision,
                action_results=record.action_results,
                record=record,
            ),
        )
        engine._emit(
            record.step_id,
            RuntimePhase.CRITIC,
            payload={"stage": "start", "critic_count": len(engine.critics)},
        )
        outputs: List[Dict[str, Any]] = []
        for critic in engine.critics:
            out = critic.evaluate(state, record.decision, record.action_results)
            # Normalize to CriticResult
            if isinstance(out, CriticResult):
                result = out
            elif isinstance(out, dict):
                result = CriticResult.from_dict(out)
            else:
                result = CriticResult(action="continue", reason="invalid_critic_output")
            d = result.to_dict()
            d["critic_name"] = type(critic).__name__
            outputs.append(d)
        record.critic_outputs = outputs
        engine._emit(
            record.step_id,
            RuntimePhase.CRITIC,
            payload={"stage": "outputs", "critic_outputs": outputs},
        )
        for output in outputs:
            action = str(output.get("action", "continue"))
            if action == "stop":
                engine._emit(
                    record.step_id,
                    RuntimePhase.CRITIC,
                    payload={"stage": "stop", "reason": output.get("reason")},
                )
                engine._dispatch_hook(
                    "on_after_critic",
                    engine._hook_context(
                        step_id=record.step_id,
                        phase=RuntimePhase.CRITIC,
                        state=state,
                        decision=record.decision,
                        action_results=record.action_results,
                        record=record,
                        payload={"critic_outputs": outputs, "result": "stop"},
                    ),
                )
                return {
                    "action": "stop",
                    "modified_prompt": None,
                    "instruction_patch": None,
                    "state_patch": None,
                    "reason": output.get("reason"),
                }
            if action == "retry":
                engine._emit(
                    record.step_id,
                    RuntimePhase.CRITIC,
                    payload={"stage": "retry", "reason": output.get("reason")},
                )
                engine._dispatch_hook(
                    "on_after_critic",
                    engine._hook_context(
                        step_id=record.step_id,
                        phase=RuntimePhase.CRITIC,
                        state=state,
                        decision=record.decision,
                        action_results=record.action_results,
                        record=record,
                        payload={"critic_outputs": outputs, "result": "retry"},
                    ),
                )
                return {
                    "action": "retry",
                    "modified_prompt": output.get("modified_prompt"),
                    "instruction_patch": output.get("instruction_patch"),
                    "state_patch": output.get("state_patch"),
                    "reason": output.get("reason"),
                }
        engine._emit(record.step_id, RuntimePhase.CRITIC, payload={"stage": "pass"})
        engine._dispatch_hook(
            "on_after_critic",
            engine._hook_context(
                step_id=record.step_id,
                phase=RuntimePhase.CRITIC,
                state=state,
                decision=record.decision,
                action_results=record.action_results,
                record=record,
                payload={"critic_outputs": outputs, "result": "continue"},
            ),
        )
        return empty

    def run_check_stop(
        self,
        state: StateT,
        decision: Decision[ActionT],
        step_id: int,
        started_at: float,
    ) -> bool:
        engine = self.engine
        engine._dispatch_hook(
            "on_before_check_stop",
            engine._hook_context(
                step_id=step_id,
                phase=RuntimePhase.CHECK_STOP,
                state=state,
                decision=decision,
            ),
        )
        engine._emit(
            state.current_step, RuntimePhase.CHECK_STOP, payload={"stage": "start"}
        )

        if decision.mode == "final":
            if engine._runtime_inbox.has_events(engine.active_run_id):
                state.final_result = None
                if state.stop_reason in {
                    StopReason.FINAL.value,
                    StopReason.COMPLETED.value,
                    StopReason.BLOCKED.value,
                }:
                    state.stop_reason = None
                self._finish_check_stop(
                    step_id=step_id,
                    state=state,
                    decision=decision,
                    stop=False,
                    extra_payload={
                        "completion_disposition": "continue",
                        "completion_reason": "runtime_input_pending",
                    },
                )
                return False
            assessment = engine.agent.assess_completion(state, decision)
            disposition = assessment.disposition
            if disposition is CompletionDisposition.CONTINUE:
                state.final_result = None
                if state.stop_reason in {
                    StopReason.FINAL.value,
                    StopReason.COMPLETED.value,
                    StopReason.BLOCKED.value,
                }:
                    state.stop_reason = None
                engine._history_append(
                    "user",
                    assessment.feedback,
                    step_id,
                    metadata={
                        "source": "completion_assessment",
                        "reason": assessment.reason,
                    },
                )
                self._finish_check_stop(
                    step_id=step_id,
                    state=state,
                    decision=decision,
                    stop=False,
                    extra_payload={
                        "completion_disposition": disposition.value,
                        "completion_reason": assessment.reason,
                    },
                )
                return False
            stop_reason = (
                StopReason.BLOCKED
                if disposition is CompletionDisposition.BLOCKED
                else StopReason.COMPLETED
            )
            state.set_stop(stop_reason, decision.final_answer)
            self._finish_check_stop(
                step_id=step_id,
                state=state,
                decision=decision,
                stop=True,
                extra_payload={
                    "completion_disposition": disposition.value,
                    "completion_reason": assessment.reason,
                },
            )
            return True
        if engine.agent.should_stop(state):
            if state.stop_reason is None:
                state.set_stop(StopReason.AGENT_CONDITION)
            self._finish_check_stop(
                step_id=step_id, state=state, decision=decision, stop=True
            )
            return True
        if engine.env is not None and engine.env.is_terminal(
            state=state, last_result=engine._last_env_result
        ):
            if state.stop_reason is None:
                state.set_stop(StopReason.ENV_TERMINAL)
            self._finish_check_stop(
                step_id=step_id,
                state=state,
                decision=decision,
                stop=True,
                extra_payload={"env_terminal": True},
            )
            return True

        elapsed = time.monotonic() - started_at
        should_stop, reason, detail = self.should_stop_by_criteria(
            state, step_id, elapsed
        )
        if should_stop:
            if state.stop_reason is None:
                state.set_stop(reason or StopReason.UNRECOVERABLE_ERROR)
            self._finish_check_stop(
                step_id=step_id,
                state=state,
                decision=decision,
                stop=True,
                extra_payload={"stop_detail": detail},
            )
            return True

        self._finish_check_stop(
            step_id=step_id, state=state, decision=decision, stop=False
        )
        return False

    def _finish_check_stop(
        self,
        step_id: int,
        state: StateT,
        decision: Decision[ActionT],
        stop: bool,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        engine = self.engine
        if stop:
            payload: Dict[str, Any] = {
                "stage": "stop",
                "stop_reason": state.stop_reason,
                "final_result": state.final_result,
            }
            if extra_payload:
                payload.update(extra_payload)
            engine._emit(state.current_step, RuntimePhase.CHECK_STOP, payload=payload)
        else:
            payload = {"stage": "continue"}
            if extra_payload:
                payload.update(extra_payload)
            engine._emit(
                state.current_step,
                RuntimePhase.CHECK_STOP,
                payload=payload,
            )
        engine._dispatch_hook(
            "on_after_check_stop",
            engine._hook_context(
                step_id=step_id,
                phase=RuntimePhase.CHECK_STOP,
                state=state,
                decision=decision,
                stop_reason=state.stop_reason if stop else None,
                payload={"result": "stop" if stop else "continue"},
            ),
        )

    def should_stop_by_criteria(
        self,
        state: StateT,
        step_id: int,
        elapsed_seconds: float,
    ) -> tuple[bool, Optional[StopReason], Optional[str]]:
        engine = self.engine
        run_budget = engine._run_budget_snapshot()
        for criteria in engine.stop_criteria:
            hit, reason, detail = criteria.should_stop(
                state,
                step_id,
                runtime_info={
                    "elapsed_seconds": elapsed_seconds,
                    "total_tokens": int(run_budget.total_tokens),
                    "total_cost_usd": float(run_budget.total_cost_usd),
                    "local_total_tokens": int(engine._token_usage),
                    "local_total_cost_usd": float(engine._cost_usage_usd),
                    "budget_max_steps": engine.budget.max_steps,
                    "budget_max_runtime_seconds": engine.budget.max_runtime_seconds,
                    "budget_max_tokens": engine.budget.max_tokens,
                    "budget_max_cost_usd": engine.budget.max_cost_usd,
                },
            )
            if hit:
                return True, reason, detail
        return False, None, None

    def budget_exhausted(self, step_id: int, state: StateT) -> bool:
        engine = self.engine
        run_budget = engine._run_budget_snapshot()
        if step_id >= engine.budget.max_steps:
            state.set_stop(StopReason.BUDGET_STEPS)
            return True
        remaining = engine.remaining_runtime_seconds()
        if remaining is not None and remaining <= 0:
            state.set_stop(StopReason.BUDGET_TIME)
            return True
        if engine.budget.max_tokens is not None and engine._token_usage >= int(
            engine.budget.max_tokens
        ):
            state.set_stop(StopReason.BUDGET_TOKENS)
            return True
        if run_budget.tokens_exhausted:
            state.set_stop(StopReason.BUDGET_TOKENS)
            return True
        if (
            engine.budget.max_cost_usd is not None
            and engine._cost_usage_usd >= float(engine.budget.max_cost_usd)
        ):
            state.set_stop(StopReason.BUDGET_COST)
            return True
        if run_budget.cost_exhausted:
            state.set_stop(StopReason.BUDGET_COST)
            return True
        return False

    def recover(self, state: StateT, phase: RuntimePhase, exc: Exception) -> bool:
        engine = self.engine
        step_id = state.current_step
        engine._dispatch_hook(
            "on_recover",
            engine._hook_context(
                step_id=step_id,
                phase=phase,
                state=state,
                error=exc,
                stop_reason=state.stop_reason,
            ),
        )
        if phase == RuntimePhase.DECIDE:
            engine._emit(step_id, RuntimePhase.DECIDE_ERROR, ok=False, error=str(exc))
        elif phase == RuntimePhase.ACT:
            engine._emit(step_id, RuntimePhase.ACT_ERROR, ok=False, error=str(exc))
        engine._emit(step_id, RuntimePhase.RECOVER, ok=False, error=str(exc))

        if engine.recovery_handler is not None:
            engine.recovery_handler(state, phase, exc)

        if isinstance(exc, DecisionContextConfigurationError):
            state.set_stop(StopReason.INFRASTRUCTURE_INVALID)
            return False

        if isinstance(exc, ModelRequestCancelled):
            state.set_stop(StopReason.CANCELLED_IMMEDIATE)
            return False

        if isinstance(exc, ModelRequestDeadlineExceeded):
            state.set_stop(StopReason.BUDGET_TIME)
            return False

        if isinstance(exc, ContextOverflowError) or self._is_api_context_overflow(exc):
            # Provider overflow is a bounded last-resort signal, not permission
            # to mutate or arbitrarily slice canonical history. Each retry asks
            # the normal transaction-aware projection for a smaller budget.
            retry = engine._context_runtime.begin_reactive_compaction()
            if retry is not None:
                engine._emit(
                    step_id,
                    RuntimePhase.COMPACT,
                    payload={
                        "stage": "context_history",
                        "context": {
                            "stage": "reactive_compact_retry",
                            "reason": (
                                "force_threshold_reached"
                                if isinstance(exc, ContextCompactionRequired)
                                else "provider_context_overflow"
                            ),
                            "canonical_history_mutated": False,
                            **retry,
                        },
                    },
                )
                return True
            state.set_stop(StopReason.CONTEXT_OVERFLOW)
            return False

        decision = engine.recovery_policy.handle(state, phase.value, step_id, exc)
        if decision.stop_reason:
            state.set_stop(decision.stop_reason)
        # Apply recovery patches if provided
        if decision.state_patch:
            for key, value in decision.state_patch.items():
                if hasattr(state, key):
                    setattr(state, key, value)
        if decision.instruction_patch:
            engine._critic_instruction_patch = decision.instruction_patch
        if not decision.continue_run and state.stop_reason is None:
            state.set_stop(StopReason.UNRECOVERABLE_ERROR)
        return decision.continue_run

    def infer_failed_phase(self, record: StepRecord) -> RuntimePhase:
        if not record.phase_events:
            return RuntimePhase.RECOVER
        latest = record.phase_events[-1].phase
        if latest == RuntimePhase.DECIDE:
            return RuntimePhase.DECIDE
        if latest == RuntimePhase.ACT:
            return RuntimePhase.ACT
        return RuntimePhase.RECOVER

    @staticmethod
    def _is_api_context_overflow(exc: Exception) -> bool:
        """Check if an exception is an API-level context overflow error."""
        msg = str(exc).lower()
        return any(
            kw in msg
            for kw in (
                "context_length_exceeded",
                "context length",
                "prompt too long",
                "maximum context",
                "too many tokens",
                "reduce the length",
                "input is too long",
            )
        )
