"""One immutable, transactional Agent turn.

The full-run loop and the public step API share this executor so decide, tool,
reducer, completion, and persistence ordering cannot drift.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast
from uuid import uuid4

from ..core.decision import Decision
from ..core.errors import StopReason
from ..core.journal import JournalError
from ..core.state import StateSchema
from .hooks import HookContext
from .interrupt import EngineInterrupt, InterruptInfo, _reset_interrupt_context
from .states import RuntimePhase, StepRecord, StepResult

if TYPE_CHECKING:
    from .engine import Engine


StateT = TypeVar("StateT", bound=StateSchema)
ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


@dataclass(slots=True)
class TurnExecution(Generic[StateT, ObservationT]):
    """Internal control result returned to the small run loop."""

    state: StateT
    observation: ObservationT
    next_step_id: int
    step: StepResult
    stop: bool = False
    journal_interrupted: bool = False
    check_after_step_cancel: bool = False


class _TurnRuntime(Generic[StateT, ObservationT, ActionT]):
    """Execute exactly one captured turn and publish one safe boundary."""

    def __init__(self, engine: Engine[StateT, ObservationT, ActionT]) -> None:
        self.engine = engine

    async def execute(
        self,
        state: StateT,
        observation: ObservationT,
        *,
        task: str,
        started_at: float,
        step_id: int | None = None,
        managed_run: bool = True,
    ) -> TurnExecution[StateT, ObservationT]:
        engine = self.engine
        current_step = state.current_step if step_id is None else step_id
        _reset_interrupt_context()

        # Capture once after the caller's safe-point drain. Provider projection
        # and tool dispatch must consume this exact immutable view.
        turn = engine._capture_turn(state, current_step)
        engine.validation_gate.before_phase(state, RuntimePhase.DECIDE.value)

        record = StepRecord(step_id=current_step, agent_id=engine.agent.name)
        record.transaction_id = f"step-{current_step}-{uuid4().hex[:12]}"
        engine.records.append(record)
        step_before = state.to_dict()
        if managed_run and engine.journal is not None:
            state = cast(StateT, type(state).from_dict(step_before))
            step_before = state.to_dict()

        engine._dispatch_hook(
            "on_before_step",
            HookContext(
                task=task,
                step_id=current_step,
                phase=RuntimePhase.DECIDE,
                state=state,
                observation=observation,
                record=record,
            ),
        )

        try:
            decision = await engine._run_decide(state, observation, record, turn)
            if managed_run:
                await engine._journal_model_completed(record, decision)
        except EngineInterrupt as exc:
            if not managed_run:
                return await self._interrupt(
                    state,
                    observation,
                    record,
                    exc,
                    decision=None,
                )
            return await self._recover_decide_error(
                state,
                observation,
                record,
                exc,
                task=task,
                started_at=started_at,
                managed_run=managed_run,
            )
        except Exception as exc:
            return await self._recover_decide_error(
                state,
                observation,
                record,
                exc,
                task=task,
                started_at=started_at,
                managed_run=managed_run,
            )

        if decision.mode == "handoff":
            next_observation = engine._control_runtime.run_handoff(
                state, decision, record
            )
            return await self._commit_continue(
                state,
                next_observation,
                record,
                decision,
                step_before=step_before,
                task=task,
                phase=RuntimePhase.HANDOFF_END,
                managed_run=managed_run,
            )

        if self._is_runtime_wait(decision):
            return await self._run_runtime_wait(
                state,
                observation,
                record,
                decision,
                step_before=step_before,
                task=task,
                started_at=started_at,
                managed_run=managed_run,
            )

        if self._is_plain_wait(decision):
            next_observation = engine._build_initial_observation(
                state, current_step + 1, started_at
            )
            return await self._commit_continue(
                state,
                next_observation,
                record,
                decision,
                step_before=step_before,
                task=task,
                phase=RuntimePhase.CHECK_STOP,
                managed_run=managed_run,
            )

        try:
            action_results: list[Any]
            if self._is_final_like(decision):
                action_results = []
            else:
                action_results = await engine._run_act(state, decision, record, turn)

            next_observation = engine._build_observation_after_action(
                state=state,
                step_id=current_step,
                started_at=started_at,
                decision=decision,
                action_results=action_results,
            )
            record.observation = next_observation
            engine._memory_append("observation", next_observation, record.step_id)
            engine._run_reduce(state, next_observation, decision, record)
        except EngineInterrupt as exc:
            if not managed_run:
                return await self._interrupt(
                    state,
                    cast(ObservationT, record.observation or observation),
                    record,
                    exc,
                    decision=decision,
                )
            return await self._recover_execution_error(
                state,
                record,
                decision,
                exc,
                task=task,
                started_at=started_at,
                step_before=step_before,
                managed_run=managed_run,
            )
        except Exception as exc:
            return await self._recover_execution_error(
                state,
                record,
                decision,
                exc,
                task=task,
                started_at=started_at,
                step_before=step_before,
                managed_run=managed_run,
            )

        critic = engine._apply_critics(state, record)
        if isinstance(critic, str):
            critic = {
                "action": critic,
                "modified_prompt": None,
                "instruction_patch": None,
                "state_patch": None,
            }
        if critic["action"] == "stop":
            state.set_stop(StopReason.CRITIC_STOP)
            engine._finalize_step(record, state)
            if managed_run and engine.journal is not None:
                await engine._journal_commit_step(
                    record,
                    before=step_before,
                    state=state,
                    terminal=True,
                )
            self._after_step_hook(state, record, task=task, phase=RuntimePhase.CRITIC)
            engine._emit(
                current_step,
                RuntimePhase.END,
                payload={"stop_reason": state.stop_reason},
            )
            return self._execution(
                state,
                next_observation,
                record,
                decision,
                stop=True,
                next_step_id=current_step,
            )

        if critic["action"] == "retry":
            engine._apply_critic_patches(state, critic)
            return await self._commit_continue(
                state,
                next_observation,
                record,
                decision,
                step_before=step_before,
                task=task,
                phase=RuntimePhase.CRITIC,
                managed_run=managed_run,
            )

        # Linearize stop/commit with mailbox acceptance. An event accepted before
        # this boundary defers a final answer to the next turn; a post that loses
        # the race observes the sealed inbox and is rejected.
        async with engine._runtime_input_post_lock:
            stop = engine._run_check_stop(
                state,
                record.decision,
                current_step,
                started_at,
            )
            engine.validation_gate.after_phase(state, RuntimePhase.CHECK_STOP.value)
            engine._finalize_step(record, state)
            if not stop and managed_run:
                state.advance_step()
            if managed_run:
                if engine.journal is not None:
                    await engine._journal_commit_step(
                        record,
                        before=step_before,
                        state=state,
                        terminal=stop,
                    )
                else:
                    await engine._save_checkpoint(current_step, state, task)
            if stop:
                engine._runtime_inbox.close(engine.active_run_id)
        self._after_step_hook(state, record, task=task, phase=RuntimePhase.CHECK_STOP)

        if stop:
            engine._emit(
                current_step,
                RuntimePhase.END,
                payload={"stop_reason": state.stop_reason},
            )
        return self._execution(
            state,
            next_observation,
            record,
            decision,
            stop=stop,
            next_step_id=current_step if stop else current_step + 1,
            check_after_step_cancel=not stop,
        )

    async def execute_terminal(
        self,
        state: StateT,
        observation: ObservationT,
        *,
        task: str,
        stop_reason: StopReason,
        step_id: int,
        use_model: bool,
    ) -> TurnExecution[StateT, ObservationT]:
        """Commit the reserved tool-free terminal step exactly once."""

        engine = self.engine
        _reset_interrupt_context()
        turn = engine._capture_turn(
            state,
            step_id,
            terminal_reason=stop_reason.value,
        )
        engine.validation_gate.before_phase(state, RuntimePhase.DECIDE.value)
        record = StepRecord(step_id=step_id, agent_id=engine.agent.name)
        record.transaction_id = f"step-{step_id}-{uuid4().hex[:12]}"
        engine.records.append(record)
        step_before = state.to_dict()
        if engine.journal is not None:
            state = cast(StateT, type(state).from_dict(step_before))
            step_before = state.to_dict()
        engine._dispatch_hook(
            "on_before_step",
            HookContext(
                task=task,
                step_id=step_id,
                phase=RuntimePhase.DECIDE,
                state=state,
                observation=observation,
                record=record,
                payload={"terminal_synthesis": True},
            ),
        )

        decision: Decision[ActionT] | None = None
        failure: Exception | None = None
        attempted_model = bool(use_model and turn.model is not None)
        if attempted_model:
            engine._model_continuation = None
            try:
                decision = await engine._run_terminal_synthesis(
                    state,
                    observation,
                    record,
                    turn,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failure = exc
                engine._model_continuation = None

        if decision is None:
            if not attempted_model:
                engine._dispatch_hook(
                    "on_before_decide",
                    engine._hook_context(
                        step_id=step_id,
                        phase=RuntimePhase.DECIDE,
                        state=state,
                        observation=observation,
                        record=record,
                        payload={"terminal_synthesis": True, "model_skipped": True},
                    ),
                )
            answer = str(
                engine.agent.build_terminal_fallback(state, stop_reason.value) or ""
            ).strip()
            if not answer:
                answer = (
                    f"Run stopped because {stop_reason.value}; no final model "
                    "conclusion was available."
                )
            decision = Decision.final(
                answer,
                meta={
                    "terminal_synthesis": True,
                    "terminal_reason": stop_reason.value,
                    "fallback": True,
                    "model_attempted": attempted_model,
                    "error_type": type(failure).__name__ if failure else None,
                },
            )
            record.decision = decision
            record.actions = []
            fallback_metadata = {
                "reason": stop_reason.value,
                "model_attempted": attempted_model,
                "error_type": type(failure).__name__ if failure else None,
                "error": str(failure) if failure else None,
            }
            record.prompt_metadata["terminal_synthesis"] = fallback_metadata
            if not record.model_response:
                record.model_response = {
                    "text": answer,
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                    "usage_complete": not attempted_model,
                    "cost_complete": not attempted_model,
                }
            record.model_response["terminal_fallback"] = fallback_metadata
            engine._history_append(
                "assistant",
                answer,
                step_id,
                metadata={"source": "terminal_fallback"},
            )
            engine._memory_append("decision", decision, step_id)
            engine._emit(
                step_id,
                RuntimePhase.DECIDE,
                ok=failure is None,
                payload={
                    "stage": "decision_ready",
                    "mode": "final",
                    "final_answer": answer,
                    "terminal_synthesis": True,
                    "fallback": True,
                    "error_type": type(failure).__name__ if failure else None,
                },
                error=str(failure) if failure else None,
            )
            engine._dispatch_hook(
                "on_after_decide",
                engine._hook_context(
                    step_id=step_id,
                    phase=RuntimePhase.DECIDE,
                    state=state,
                    observation=observation,
                    decision=decision,
                    model_response=dict(record.model_response),
                    record=record,
                    payload={
                        "model_response": dict(record.model_response),
                        "terminal_synthesis": True,
                        "fallback": True,
                    },
                ),
            )

        if engine.journal is not None:
            await engine._journal_model_completed(record, decision)
        state.set_stop(stop_reason, decision.final_answer)
        engine.validation_gate.after_phase(state, RuntimePhase.CHECK_STOP.value)
        engine._finalize_step(record, state)
        async with engine._runtime_input_post_lock:
            if engine.journal is not None:
                await engine._journal_commit_step(
                    record,
                    before=step_before,
                    state=state,
                    terminal=True,
                )
            else:
                await engine._save_checkpoint(step_id, state, task)
            engine._runtime_inbox.close(engine.active_run_id)
        self._after_step_hook(
            state,
            record,
            task=task,
            phase=RuntimePhase.CHECK_STOP,
            include_stop_reason=True,
        )
        engine._emit(
            step_id,
            RuntimePhase.END,
            payload={
                "stop_reason": state.stop_reason,
                "terminal_synthesis": True,
                "fallback": bool(decision.meta.get("fallback")),
            },
        )
        return self._execution(
            state,
            observation,
            record,
            decision,
            stop=True,
            next_step_id=step_id,
        )

    async def _recover_decide_error(
        self,
        state: StateT,
        observation: ObservationT,
        record: StepRecord,
        exc: Exception,
        *,
        task: str,
        started_at: float,
        managed_run: bool,
    ) -> TurnExecution[StateT, ObservationT]:
        engine = self.engine
        if isinstance(exc, JournalError):
            raise exc
        failed_phase = engine._infer_failed_phase(record)
        if not engine._recover(state, failed_phase, exc):
            return await self._stop_after_error(
                state,
                record,
                exc,
                task=task,
                observation=observation,
                managed_run=managed_run,
            )

        engine._finalize_step(record, state)
        next_observation = engine._build_initial_observation(
            state, record.step_id + 1, started_at
        )
        if managed_run:
            state.advance_step()
        if managed_run:
            await engine._journal_snapshot_state(
                state,
                step_id=record.step_id,
                reason="recovery",
                record_id=f"{record.transaction_id}:recovery",
            )
        self._after_step_hook(
            state,
            record,
            task=task,
            phase=RuntimePhase.RECOVER,
            include_stop_reason=True,
        )
        execution = self._execution(
            state,
            next_observation,
            record,
            None,
            next_step_id=record.step_id + 1,
        )
        execution.step.recovered = True
        return execution

    async def _recover_execution_error(
        self,
        state: StateT,
        record: StepRecord,
        decision: Decision[ActionT],
        exc: Exception,
        *,
        task: str,
        started_at: float,
        step_before: dict[str, Any],
        managed_run: bool,
    ) -> TurnExecution[StateT, ObservationT]:
        engine = self.engine
        if managed_run and engine.journal is not None:
            raise exc
        failed_phase = engine._infer_failed_phase(record)
        if not engine._recover(state, failed_phase, exc):
            return await self._stop_after_error(
                state,
                record,
                exc,
                task=task,
                observation=cast(ObservationT, record.observation),
                decision=decision,
                managed_run=managed_run,
            )

        engine._finalize_step(record, state)
        next_observation = engine._build_initial_observation(
            state, record.step_id + 1, started_at
        )
        if managed_run:
            state.advance_step()
        if managed_run and engine.journal is not None:
            await engine._journal_commit_step(
                record,
                before=step_before,
                state=state,
                terminal=False,
            )
        self._after_step_hook(
            state,
            record,
            task=task,
            phase=RuntimePhase.RECOVER,
            include_stop_reason=True,
        )
        execution = self._execution(
            state,
            next_observation,
            record,
            decision,
            next_step_id=record.step_id + 1,
        )
        execution.step.recovered = True
        return execution

    async def _stop_after_error(
        self,
        state: StateT,
        record: StepRecord,
        exc: Exception,
        *,
        task: str,
        observation: ObservationT,
        managed_run: bool,
        decision: Decision[ActionT] | None = None,
    ) -> TurnExecution[StateT, ObservationT]:
        engine = self.engine
        engine._finalize_step(record, state)
        engine._emit(
            record.step_id,
            RuntimePhase.END,
            ok=False,
            payload={"stop_reason": state.stop_reason},
        )
        if managed_run:
            await engine._journal_interrupt_run(
                step_id=record.step_id,
                reason=state.stop_reason or "unrecoverable_error",
            )
        execution = self._execution(
            state,
            observation,
            record,
            decision,
            stop=True,
            next_step_id=record.step_id,
            journal_interrupted=managed_run,
        )
        execution.step.error = exc
        execution.step.stop_reason = StopReason.UNRECOVERABLE_ERROR
        return execution

    async def _interrupt(
        self,
        state: StateT,
        observation: ObservationT,
        record: StepRecord,
        exc: EngineInterrupt,
        *,
        decision: Decision[ActionT] | None,
    ) -> TurnExecution[StateT, ObservationT]:
        checkpoint_id = await self.engine._save_interrupt_checkpoint(
            record.step_id, state, exc
        )
        info = InterruptInfo(
            interrupt_id=exc.interrupt_id,
            checkpoint_id=checkpoint_id,
            value=exc.value,
        )
        self.engine._emit(
            record.step_id,
            RuntimePhase.INTERRUPT,
            ok=True,
            payload={"interrupt_id": exc.interrupt_id},
        )
        self.engine._finalize_step(record, state)
        execution = self._execution(
            state,
            observation,
            record,
            decision,
            stop=True,
            next_step_id=record.step_id,
        )
        execution.step.stop_reason = StopReason.INTERRUPT
        execution.step.interrupt_info = info
        return execution

    async def _run_runtime_wait(
        self,
        state: StateT,
        observation: ObservationT,
        record: StepRecord,
        decision: Decision[ActionT],
        *,
        step_before: dict[str, Any],
        task: str,
        started_at: float,
        managed_run: bool,
    ) -> TurnExecution[StateT, ObservationT]:
        engine = self.engine
        engine._emit(
            record.step_id,
            RuntimePhase.DECIDE,
            payload={"stage": "runtime_wait_start"},
        )
        wait_outcome = await engine._wait_for_runtime_event()
        engine._emit(
            record.step_id,
            RuntimePhase.DECIDE,
            payload={"stage": "runtime_wait_end", "outcome": wait_outcome},
        )
        engine._finalize_step(record, state)

        if wait_outcome == "event":
            next_observation = engine._build_initial_observation(
                state, record.step_id + 1, started_at
            )
            if managed_run:
                state.advance_step()
            if managed_run and engine.journal is not None:
                await engine._journal_commit_step(
                    record,
                    before=step_before,
                    state=state,
                    terminal=False,
                )
            self._after_step_hook(
                state, record, task=task, phase=RuntimePhase.CHECK_STOP
            )
            return self._execution(
                state,
                next_observation,
                record,
                decision,
                next_step_id=record.step_id + 1,
            )

        if wait_outcome in {"cancelled", "timeout"}:
            if managed_run and engine.journal is not None:
                await engine._journal_commit_step(
                    record,
                    before=step_before,
                    state=state,
                    terminal=False,
                )
            self._after_step_hook(
                state, record, task=task, phase=RuntimePhase.CHECK_STOP
            )
            return self._execution(
                state,
                observation,
                record,
                decision,
                next_step_id=record.step_id,
            )
        raise RuntimeError("runtime inbox closed while Engine was waiting")

    async def _commit_continue(
        self,
        state: StateT,
        observation: ObservationT,
        record: StepRecord,
        decision: Decision[ActionT],
        *,
        step_before: dict[str, Any],
        task: str,
        phase: RuntimePhase,
        managed_run: bool,
    ) -> TurnExecution[StateT, ObservationT]:
        engine = self.engine
        engine._finalize_step(record, state)
        if managed_run:
            state.advance_step()
        if managed_run and engine.journal is not None:
            await engine._journal_commit_step(
                record,
                before=step_before,
                state=state,
                terminal=False,
            )
        self._after_step_hook(state, record, task=task, phase=phase)
        return self._execution(
            state,
            observation,
            record,
            decision,
            next_step_id=record.step_id + 1,
        )

    def _after_step_hook(
        self,
        state: StateT,
        record: StepRecord,
        *,
        task: str,
        phase: RuntimePhase,
        include_stop_reason: bool = False,
    ) -> None:
        self.engine._dispatch_hook(
            "on_after_step",
            HookContext(
                task=task,
                step_id=record.step_id,
                phase=phase,
                state=state,
                record=record,
                stop_reason=state.stop_reason if include_stop_reason else None,
                journal_position=self.engine._last_journal_position,
            ),
        )

    def _execution(
        self,
        state: StateT,
        observation: ObservationT,
        record: StepRecord,
        decision: Decision[ActionT] | None,
        *,
        next_step_id: int,
        stop: bool = False,
        journal_interrupted: bool = False,
        check_after_step_cancel: bool = False,
    ) -> TurnExecution[StateT, ObservationT]:
        stop_reason: StopReason | None = None
        if stop:
            try:
                stop_reason = StopReason(state.stop_reason)
            except (TypeError, ValueError):
                stop_reason = StopReason.UNRECOVERABLE_ERROR
        return TurnExecution(
            state=state,
            observation=observation,
            next_step_id=next_step_id,
            step=StepResult(
                step_id=record.step_id,
                decision=decision,
                record=record,
                observation=observation,
                action_results=list(record.action_results),
                stop=stop,
                stop_reason=stop_reason,
            ),
            stop=stop,
            journal_interrupted=journal_interrupted,
            check_after_step_cancel=check_after_step_cancel,
        )

    @staticmethod
    def _is_runtime_wait(decision: Decision[Any]) -> bool:
        return (
            decision.mode == "wait"
            and decision.meta.get("runtime_wait") is True
            and not bool(decision.meta.get("task_complete_requested"))
            and not bool(decision.meta.get("parser_error"))
        )

    @staticmethod
    def _is_plain_wait(decision: Decision[Any]) -> bool:
        return (
            decision.mode == "wait"
            and not bool(decision.meta.get("task_complete_requested"))
            and not bool(decision.meta.get("parser_error"))
        )

    @staticmethod
    def _is_final_like(decision: Decision[Any]) -> bool:
        return decision.mode == "final" or (
            decision.mode == "wait"
            and (
                bool(decision.meta.get("task_complete_requested"))
                or bool(decision.meta.get("parser_error"))
            )
        )


__all__ = ["TurnExecution"]
