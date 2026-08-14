"""Private canonical Run journal state machine for Engine."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, Generic, Mapping, TypeVar
from uuid import uuid4

from ..core.action import Action
from ..core.agent_module import ActionResultContext, CanonicalActionResult
from ..core.child import ChildHandle, ChildResult
from ..core.decision import Decision, DecisionMode
from ..core.completion import CompletionDisposition
from ..core.errors import StopReason
from ..core.history import HistoryMessage
from ..core.journal import (
    JournalError,
    JournalRecord,
    JournalRecordRef,
    JournalRecordType,
    resolve_inherited_record,
)
from ..core.model_request import ModelContinuation, ModelRequest
from ..core.process import ProcessSnapshot
from ..core.runtime_input import (
    RuntimeInput,
    child_terminal_runtime_input,
    process_terminal_runtime_input,
)
from ..core.state import StateSchema
from ..core.state_delta import apply_state_delta, build_state_delta, state_digest
from ..core.task import Task
from ..core.tool_result import ToolResult
from .states import StepRecord

if TYPE_CHECKING:
    from .engine import Engine


StateT = TypeVar("StateT", bound=StateSchema)
ActionT = TypeVar("ActionT")


class _JournalRuntime(Generic[StateT, ActionT]):
    """Own journal append ordering, state commits, and deterministic replay."""

    def __init__(self, engine: Engine[StateT, Any, ActionT]) -> None:
        self.engine = engine

    async def initialize(
        self,
        task_obj: Task | None,
        task_text: str,
        state: StateT,
    ) -> None:
        engine = self.engine
        journal = engine.journal
        if journal is None:
            return
        engine._last_journal_position = await journal.create(
            engine._active_run_id,
            {
                "agent": engine.agent.name,
                "lineage_id": engine._active_lineage_id,
            },
        )
        engine._prepare_owned_budget_ledger(await journal.replay())
        engine._last_journal_position = await journal.append(
            JournalRecordType.INPUT_ACCEPTED,
            {
                "task": task_text,
                "task_data": task_obj.to_dict() if task_obj is not None else None,
            },
            record_id=f"{engine._active_run_id}:input",
        )
        await self.snapshot_state(
            state,
            step_id=int(state.current_step),
            reason="initial",
            record_id=f"{engine._active_run_id}:snapshot:initial",
        )

    async def model_completed(
        self,
        record: StepRecord,
        decision: Decision[ActionT],
    ) -> None:
        engine = self.engine
        journal = engine.journal
        if journal is None:
            engine._pending_runtime_input_ids.clear()
            return
        history_append = list(engine._journal_pending_history)
        runtime_input_ids = list(engine._pending_runtime_input_ids)
        engine._last_journal_position = await journal.append(
            JournalRecordType.MODEL_COMPLETED,
            {
                "step_id": record.step_id,
                "transaction_id": record.transaction_id,
                "model_request": (
                    record.model_request.to_dict()
                    if record.model_request is not None
                    else None
                ),
                "model_response": dict(record.model_response),
                "prompt_metadata": dict(record.prompt_metadata),
                "decision": decision_to_dict(decision),
                "history_append": history_append,
                "runtime_input_ids": runtime_input_ids,
            },
            record_id=f"{record.transaction_id}:model",
        )
        del engine._journal_pending_history[: len(history_append)]
        del engine._pending_runtime_input_ids[: len(runtime_input_ids)]

    async def tool_starts(
        self,
        indexed_actions: list[tuple[int, Action]],
        record: StepRecord,
    ) -> None:
        engine = self.engine
        journal = engine.journal
        if journal is None:
            return
        for index, action in indexed_actions:
            engine._last_journal_position = await journal.append(
                JournalRecordType.TOOL_STARTED,
                {
                    "step_id": record.step_id,
                    "transaction_id": record.transaction_id,
                    "action_index": index,
                    "action": action_to_dict(action),
                },
                record_id=f"{record.transaction_id}:tool:{index}:started",
            )

    async def finalize_action_results(
        self,
        state: StateT,
        actions: list[Action],
        results: list[ToolResult],
        *,
        record: StepRecord,
    ) -> list[ToolResult]:
        engine = self.engine
        finalized: list[ToolResult] = []
        terminal_ids: list[str] = []
        for index, (action, raw_result) in enumerate(
            zip(actions, results, strict=True)
        ):
            terminal: JournalRecordRef | None = None
            expected_call_id = _action_call_id(action, record.step_id, index)
            input_result = ToolResult.from_value(raw_result)
            input_result.call_id = expected_call_id
            result = ToolResult.from_value(
                engine.agent.finalize_action_result(
                    state,
                    action,
                    input_result,
                    step_id=record.step_id,
                    context=ActionResultContext(
                        prior_results=tuple(engine._canonical_action_results)
                    ),
                )
            )
            result.call_id = expected_call_id
            if engine.journal is not None:
                record_id = f"{record.transaction_id}:tool:{index}:terminal"
                engine._last_journal_position = await engine.journal.append(
                    JournalRecordType.TOOL_TERMINAL,
                    {
                        "step_id": record.step_id,
                        "transaction_id": record.transaction_id,
                        "action_index": index,
                        "action": action_to_dict(action),
                        "result": result.to_dict(),
                    },
                    record_id=record_id,
                )
                terminal_ids.append(record_id)
                terminal = JournalRecordRef(engine.journal.run_id, record_id)
            finalized.append(result)
            engine._canonical_action_results.append(
                CanonicalActionResult(record.step_id, action, result, terminal)
            )
            self.reduce_action_results(
                state,
                [action],
                [result],
                record.step_id,
            )
        if engine.journal is not None:
            engine._journal_terminal_record_ids[record.transaction_id] = terminal_ids
        return finalized

    def reduce_action_results(
        self,
        state: StateT,
        actions: list[Action],
        results: list[ToolResult],
        step_id: int,
    ) -> None:
        for action, result in zip(actions, results, strict=True):
            next_state = self.engine.agent.reduce_action_result(
                state,
                action,
                result,
                step_id=step_id,
            )
            if next_state is not state:
                state.reduce_update(next_state.to_dict())

    async def commit_step(
        self,
        record: StepRecord,
        *,
        before: Dict[str, Any],
        state: StateT,
        terminal: bool,
    ) -> None:
        engine = self.engine
        journal = engine.journal
        if journal is None:
            return
        after = state.to_dict()
        history_append = list(engine._journal_pending_history)
        engine._last_journal_position = await journal.append(
            JournalRecordType.STEP_COMMITTED,
            {
                "step_id": record.step_id,
                "transaction_id": record.transaction_id,
                "terminal_record_ids": list(
                    engine._journal_terminal_record_ids.get(record.transaction_id, [])
                ),
                "before_digest": state_digest(before),
                "after_digest": state_digest(after),
                "state_delta": build_state_delta(before, after),
                "history_append": history_append,
            },
            record_id=f"{record.transaction_id}:committed",
        )
        del engine._journal_pending_history[: len(history_append)]
        engine._active_state = type(state).from_dict(after)
        should_snapshot = terminal or (
            state.current_step > 0
            and state.current_step % engine._state_snapshot_interval == 0
        )
        if should_snapshot:
            await self.snapshot_state(
                state,
                step_id=record.step_id,
                reason="terminal" if terminal else "interval",
                record_id=f"{record.transaction_id}:snapshot",
            )

    async def snapshot_state(
        self,
        state: StateT,
        *,
        step_id: int,
        reason: str,
        record_id: str,
    ) -> None:
        engine = self.engine
        journal = engine.journal
        if journal is None:
            return
        payload = state.to_dict()
        engine._last_journal_position = await journal.append(
            JournalRecordType.STATE_SNAPSHOT,
            {
                "step_id": step_id,
                "state": payload,
                "state_digest": state_digest(payload),
                "reason": reason,
            },
            record_id=record_id,
        )
        engine._active_state = type(state).from_dict(payload)

    async def interrupt_run(self, *, step_id: int, reason: str) -> None:
        engine = self.engine
        committed = engine._active_state
        if engine.journal is None or committed is None:
            return
        committed_state = committed.to_dict()
        engine._last_journal_position = await engine.journal.append(
            JournalRecordType.RUN_INTERRUPTED,
            {
                "step_id": step_id,
                "reason": reason,
                "state_digest": state_digest(committed_state),
            },
            record_id=f"{engine._active_run_id}:interrupted:{uuid4().hex[:12]}",
        )

    async def finish_run(self, state: StateT) -> None:
        engine = self.engine
        journal = engine.journal
        if journal is None:
            return
        state_payload = state.to_dict()
        committed_payload = (
            engine._active_state.to_dict() if engine._active_state is not None else None
        )
        if committed_payload != state_payload:
            await self.snapshot_state(
                state,
                step_id=int(state.current_step),
                reason="terminal",
                record_id=f"{engine._active_run_id}:snapshot:terminal",
            )
        engine._last_journal_position = await journal.append(
            JournalRecordType.RUN_COMPLETED,
            {
                "state_digest": state_digest(state_payload),
                "stop_reason": state.stop_reason,
                "final_result": state.final_result,
            },
            record_id=f"{engine._active_run_id}:complete",
        )

    def replay(self, records: tuple[JournalRecord, ...]) -> Dict[str, Any]:
        engine = self.engine
        effective = effective_journal_records(records)
        pending_runtime_inputs = recover_pending_runtime_inputs(records)
        task = ""
        task_data: Dict[str, Any] | None = None
        state_data: Dict[str, Any] | None = None
        history: list[HistoryMessage] = []
        step_records: dict[str, StepRecord] = {}
        transactions: dict[str, dict[str, Any]] = {}
        completed = False
        recovered_terminals: list[dict[str, Any]] = []
        recovered_steps: list[dict[str, Any]] = []
        canonical_results: list[CanonicalActionResult] = []
        terminal_snapshot_digest = ""
        usage_prompt_tokens = 0
        usage_completion_tokens = 0
        usage_total_tokens = 0
        usage_cost_usd = 0.0
        usage_complete = True
        cost_complete = True
        latest_continuation: ModelContinuation | None = None
        for journal_record in effective:
            payload = journal_record.payload
            if journal_record.type is JournalRecordType.INPUT_ACCEPTED:
                task = str(payload.get("task") or "")
                raw_task = payload.get("task_data")
                task_data = dict(raw_task) if isinstance(raw_task, Mapping) else None
                continue
            if journal_record.type is JournalRecordType.STATE_SNAPSHOT:
                raw_state = payload.get("state")
                if not isinstance(raw_state, dict):
                    raise JournalError("state.snapshot is missing state")
                expected = str(payload.get("state_digest") or "")
                if expected != state_digest(raw_state):
                    raise JournalError("state.snapshot digest does not match")
                state_data = dict(raw_state)
                if payload.get("reason") == "terminal":
                    terminal_snapshot_digest = expected
                continue
            if journal_record.type is JournalRecordType.MODEL_COMPLETED:
                transaction_id = str(payload.get("transaction_id") or "")
                step_id = _non_negative_int(
                    payload.get("step_id"), "model.completed step_id"
                )
                if not transaction_id:
                    raise JournalError("model.completed is missing transaction_id")
                record = step_records.setdefault(
                    transaction_id,
                    StepRecord(
                        step_id=step_id,
                        transaction_id=transaction_id,
                        agent_id=engine.agent.name,
                    ),
                )
                raw_response = payload.get("model_response")
                record.model_response = (
                    dict(raw_response) if isinstance(raw_response, Mapping) else {}
                )
                raw_request = payload.get("model_request")
                record.model_request = (
                    ModelRequest.from_dict(raw_request)
                    if isinstance(raw_request, Mapping)
                    else None
                )
                raw_usage = record.model_response.get("usage")
                record_usage_complete = bool(
                    record.model_response.get("usage_complete", False)
                )
                record_cost_complete = bool(
                    record.model_response.get("cost_complete", False)
                )
                usage_complete = usage_complete and record_usage_complete
                cost_complete = cost_complete and record_cost_complete
                if isinstance(raw_usage, Mapping):
                    prompt_tokens = raw_usage.get(
                        "prompt_tokens", raw_usage.get("input_tokens", 0)
                    )
                    completion_tokens = raw_usage.get(
                        "completion_tokens", raw_usage.get("output_tokens", 0)
                    )
                    total_tokens = raw_usage.get("total_tokens")
                    prompt_value = (
                        int(prompt_tokens)
                        if isinstance(prompt_tokens, int)
                        and not isinstance(prompt_tokens, bool)
                        else 0
                    )
                    completion_value = (
                        int(completion_tokens)
                        if isinstance(completion_tokens, int)
                        and not isinstance(completion_tokens, bool)
                        else 0
                    )
                    total_value = (
                        int(total_tokens)
                        if isinstance(total_tokens, int)
                        and not isinstance(total_tokens, bool)
                        else prompt_value + completion_value
                    )
                    usage_prompt_tokens += max(0, prompt_value)
                    usage_completion_tokens += max(0, completion_value)
                    usage_total_tokens += max(0, total_value)
                raw_cost = record.model_response.get("cost_usd", 0.0)
                if isinstance(raw_cost, (int, float)) and not isinstance(
                    raw_cost, bool
                ):
                    usage_cost_usd += max(0.0, float(raw_cost))
                raw_prompt_metadata = payload.get("prompt_metadata")
                record.prompt_metadata = (
                    dict(raw_prompt_metadata)
                    if isinstance(raw_prompt_metadata, Mapping)
                    else {}
                )
                raw_decision = payload.get("decision")
                if not isinstance(raw_decision, Mapping):
                    raise JournalError("model.completed is missing decision")
                record.decision = decision_from_dict(raw_decision)
                record.actions = list(record.decision.actions)
                if transaction_id in transactions:
                    raise JournalError("model.completed transaction is duplicated")
                raw_continuation = record.model_response.get("continuation")
                try:
                    continuation = (
                        ModelContinuation.from_dict(raw_continuation)
                        if isinstance(raw_continuation, Mapping)
                        else None
                    )
                except (TypeError, ValueError) as exc:
                    raise JournalError(
                        "model.completed has an invalid model continuation"
                    ) from exc
                transactions[transaction_id] = {
                    "step_id": step_id,
                    "decision": record.decision,
                    "continuation": continuation,
                    "started": set(),
                    "terminals": {},
                    "terminal_record_ids": {},
                    "committed": False,
                }
                history.extend(history_append_from_payload(payload))
                continue
            if journal_record.type is JournalRecordType.TOOL_STARTED:
                transaction_id = str(payload.get("transaction_id") or "")
                if not transaction_id:
                    raise JournalError("tool.started payload is invalid")
                action_index = _non_negative_int(
                    payload.get("action_index"), "tool.started action_index"
                )
                transaction = transactions.get(transaction_id)
                if transaction is None:
                    raise JournalError("tool.started has no model transaction")
                if _non_negative_int(
                    payload.get("step_id"), "tool.started step_id"
                ) != int(transaction["step_id"]):
                    raise JournalError(
                        "tool.started step does not match its transaction"
                    )
                actions = list(transaction["decision"].actions)
                if action_index >= len(actions):
                    raise JournalError("tool.started action_index is out of range")
                action_payload = payload.get("action")
                if not isinstance(action_payload, Mapping):
                    raise JournalError("tool.started is missing action")
                started_action = action_from_dict(action_payload)
                if action_to_dict(started_action) != action_to_dict(
                    actions[action_index]
                ):
                    raise JournalError(
                        "tool.started action does not match model decision"
                    )
                if action_index in transaction["started"]:
                    raise JournalError("tool.started action is duplicated")
                transaction["started"].add(action_index)
                continue
            if journal_record.type is JournalRecordType.TOOL_TERMINAL:
                transaction_id = str(payload.get("transaction_id") or "")
                step_id = _non_negative_int(
                    payload.get("step_id"), "tool.terminal step_id"
                )
                result = payload.get("result")
                if not transaction_id or not isinstance(result, Mapping):
                    raise JournalError("tool.terminal payload is invalid")
                record = step_records.setdefault(
                    transaction_id,
                    StepRecord(
                        step_id=step_id,
                        transaction_id=transaction_id,
                        agent_id=engine.agent.name,
                    ),
                )
                terminal_result = ToolResult.from_value(dict(result))
                if terminal_result.to_dict() != dict(result):
                    raise JournalError("tool.terminal result is not canonical")
                record.action_results.append(terminal_result)
                transaction = transactions.get(transaction_id)
                if transaction is None:
                    raise JournalError("tool.terminal has no model transaction")
                action_index = _non_negative_int(
                    payload.get("action_index"), "tool.terminal action_index"
                )
                if step_id != int(transaction["step_id"]):
                    raise JournalError(
                        "tool.terminal step does not match its transaction"
                    )
                actions = list(transaction["decision"].actions)
                if action_index >= len(actions):
                    raise JournalError("tool.terminal action_index is out of range")
                if action_index != len(transaction["terminals"]):
                    raise JournalError("tool.terminal actions are out of order")
                if action_index in transaction["terminals"]:
                    raise JournalError("tool.terminal action is duplicated")
                transaction["terminals"][action_index] = terminal_result
                action_payload = payload.get("action")
                if not isinstance(action_payload, Mapping):
                    raise JournalError("tool.terminal is missing action")
                terminal_action = action_from_dict(action_payload)
                if action_to_dict(terminal_action) != action_to_dict(
                    actions[action_index]
                ):
                    raise JournalError(
                        "tool.terminal action does not match model decision"
                    )
                expected_call_id = _action_call_id(
                    terminal_action,
                    step_id,
                    action_index,
                )
                if terminal_result.call_id is None:
                    terminal_result.call_id = expected_call_id
                elif terminal_result.call_id != expected_call_id:
                    raise JournalError("tool.terminal call_id does not match action")
                transaction["terminal_record_ids"][
                    action_index
                ] = journal_record.record_id
                canonical_results.append(
                    CanonicalActionResult(
                        step_id,
                        terminal_action,
                        terminal_result,
                        JournalRecordRef(
                            journal_record.run_id,
                            journal_record.record_id,
                        ),
                    )
                )
                continue
            if journal_record.type is JournalRecordType.STEP_COMMITTED:
                if state_data is None:
                    raise JournalError("step.committed has no base state")
                before_digest = str(payload.get("before_digest") or "")
                if state_digest(state_data) != before_digest:
                    raise JournalError("step.committed before digest does not match")
                raw_delta = payload.get("state_delta")
                if not isinstance(raw_delta, list):
                    raise JournalError("step.committed is missing state_delta")
                patched = apply_state_delta(state_data, raw_delta)
                if not isinstance(patched, dict):
                    raise JournalError("step.committed produced a non-object state")
                if state_digest(patched) != str(payload.get("after_digest") or ""):
                    raise JournalError("step.committed after digest does not match")
                state_data = patched
                transaction_id = str(payload.get("transaction_id") or "")
                transaction = transactions.get(transaction_id)
                if transaction is None:
                    raise JournalError("step.committed has no model transaction")
                if transaction["committed"]:
                    raise JournalError("step.committed transaction is duplicated")
                if _non_negative_int(
                    payload.get("step_id"), "step.committed step_id"
                ) != int(transaction["step_id"]):
                    raise JournalError(
                        "step.committed step does not match its transaction"
                    )
                actions = list(transaction["decision"].actions)
                committed_terminals = transaction["terminals"]
                if set(committed_terminals) != set(range(len(actions))):
                    raise JournalError(
                        "step.committed does not contain one terminal per action"
                    )
                raw_terminal_ids = payload.get("terminal_record_ids")
                if not isinstance(raw_terminal_ids, list) or any(
                    not isinstance(record_id, str) or not record_id
                    for record_id in raw_terminal_ids
                ):
                    raise JournalError("step.committed terminal_record_ids are invalid")
                expected_terminal_ids = [
                    transaction["terminal_record_ids"][index]
                    for index in range(len(actions))
                ]
                if raw_terminal_ids != expected_terminal_ids:
                    raise JournalError(
                        "step.committed terminal_record_ids do not match its actions"
                    )
                transaction["committed"] = True
                latest_continuation = transaction["continuation"]
                history.extend(history_append_from_payload(payload))
                continue
            if journal_record.type is JournalRecordType.RUN_INTERRUPTED:
                if state_data is None or state_digest(state_data) != str(
                    payload.get("state_digest") or ""
                ):
                    raise JournalError("run.interrupted state digest does not match")
                continue
            if journal_record.type is JournalRecordType.RUN_COMPLETED:
                completed = True
                if state_data is None or state_digest(state_data) != str(
                    payload.get("state_digest") or ""
                ):
                    raise JournalError("run.completed state digest does not match")
        if not task or state_data is None:
            raise JournalError("journal has no recoverable input and state")
        for transaction_id, transaction in transactions.items():
            if transaction["committed"]:
                continue
            started = set(transaction["started"])
            terminals: dict[int, ToolResult] = transaction["terminals"]
            actions = list(transaction["decision"].actions)
            for index, action in enumerate(actions):
                if index in terminals:
                    continue
                was_started = index in started
                result = ToolResult(
                    call_id=_action_call_id(action, int(transaction["step_id"]), index),
                    status="error" if was_started else "cancelled",
                    output=None,
                    error=(
                        "tool execution was interrupted after durable start; "
                        "side effects are unknown"
                        if was_started
                        else "tool execution did not start before interruption"
                    ),
                    metadata={
                        "recovered": True,
                        "side_effect": "unknown" if was_started else "none",
                    },
                )
                recovered_terminals.append(
                    {
                        "record_id": f"{transaction_id}:tool:{index}:terminal",
                        "payload": {
                            "step_id": int(transaction["step_id"]),
                            "transaction_id": transaction_id,
                            "action_index": index,
                            "action": action_to_dict(action),
                            "result": result.to_dict(),
                            "recovered": True,
                        },
                    }
                )
        if recovered_terminals:
            return {
                "task": task,
                "task_data": task_data,
                "state": state_data,
                "history": history,
                "records": list(step_records.values()),
                "completed": completed,
                "recovered_terminals": recovered_terminals,
                "recovered_steps": [],
                "canonical_results": canonical_results,
                "terminal_snapshot_current": False,
                "continuation": latest_continuation,
                "pending_runtime_inputs": pending_runtime_inputs,
                "usage": {
                    "prompt_tokens": usage_prompt_tokens,
                    "completion_tokens": usage_completion_tokens,
                    "total_tokens": usage_total_tokens,
                    "cost_usd": usage_cost_usd,
                    "usage_complete": usage_complete,
                    "cost_complete": cost_complete,
                },
            }
        for transaction_id, transaction in transactions.items():
            if transaction["committed"]:
                continue
            terminal_results: dict[int, ToolResult] = transaction["terminals"]
            decision = transaction["decision"]
            actions = list(decision.actions)
            if len(terminal_results) != len(actions):
                raise JournalError(
                    "journal contains an incomplete terminal action batch"
                )
            before = dict(state_data)
            ordered_results = [terminal_results[index] for index in range(len(actions))]
            working = type(engine.agent.init_state(task)).from_dict(before)
            self.reduce_action_results(
                working,
                actions,
                ordered_results,
                int(transaction["step_id"]),
            )
            observation = self.build_replayed_observation(
                working,
                decision,
                ordered_results,
                int(transaction["step_id"]),
                task,
            )
            next_state = engine.agent.reduce(working, observation, decision)
            if next_state is not working:
                working.reduce_update(next_state.to_dict())
            completion_history: list[HistoryMessage] = []
            if decision.mode == "final":
                if pending_runtime_inputs:
                    working.final_result = None
                    if working.stop_reason in {
                        StopReason.FINAL.value,
                        StopReason.COMPLETED.value,
                        StopReason.BLOCKED.value,
                    }:
                        working.stop_reason = None
                else:
                    assessment = engine.agent.assess_completion(working, decision)
                    if assessment.disposition is CompletionDisposition.CONTINUE:
                        working.final_result = None
                        working.stop_reason = None
                        completion_history.append(
                            HistoryMessage(
                                role="user",
                                step_id=int(transaction["step_id"]),
                                content=assessment.feedback,
                                metadata={
                                    "source": "completion_assessment",
                                    "reason": assessment.reason,
                                },
                            )
                        )
                    elif assessment.disposition is CompletionDisposition.BLOCKED:
                        working.set_stop(StopReason.BLOCKED, decision.final_answer)
                    else:
                        working.set_stop(StopReason.COMPLETED, decision.final_answer)
            if not working.final_result and not working.stop_reason:
                working.advance_step()
            state_data = working.to_dict()
            if state_data.get("final_result") and not state_data.get("stop_reason"):
                state_data["stop_reason"] = StopReason.COMPLETED.value
            history_append = [
                tool_result_history_message(
                    action,
                    result,
                    int(transaction["step_id"]),
                )
                for action, result in zip(actions, ordered_results, strict=True)
            ]
            history_append.extend(completion_history)
            history.extend(history_append)
            recovered_steps.append(
                {
                    "record_id": f"{transaction_id}:committed",
                    "payload": {
                        "step_id": int(transaction["step_id"]),
                        "transaction_id": transaction_id,
                        "terminal_record_ids": [
                            f"{transaction_id}:tool:{index}:terminal"
                            for index in range(len(actions))
                        ],
                        "before_digest": state_digest(before),
                        "after_digest": state_digest(state_data),
                        "state_delta": build_state_delta(before, state_data),
                        "history_append": [
                            history_message_to_dict(item) for item in history_append
                        ],
                        "recovered": True,
                    },
                }
            )
            latest_continuation = transaction["continuation"]
        return {
            "task": task,
            "task_data": task_data,
            "state": state_data,
            "history": history,
            "records": list(step_records.values()),
            "completed": completed,
            "recovered_terminals": [],
            "recovered_steps": recovered_steps,
            "canonical_results": canonical_results,
            "terminal_snapshot_current": (
                bool(terminal_snapshot_digest)
                and terminal_snapshot_digest == state_digest(state_data)
            ),
            "continuation": latest_continuation,
            "pending_runtime_inputs": pending_runtime_inputs,
            "usage": {
                "prompt_tokens": usage_prompt_tokens,
                "completion_tokens": usage_completion_tokens,
                "total_tokens": usage_total_tokens,
                "cost_usd": usage_cost_usd,
                "usage_complete": usage_complete,
                "cost_complete": cost_complete,
            },
        }

    @staticmethod
    def build_replayed_observation(
        state: StateT,
        decision: Decision[Action],
        results: list[ToolResult],
        step_id: int,
        task: str,
    ) -> Any:
        from ..core.observation import Observation

        return Observation(
            task=task,
            step_id=step_id,
            state=state.to_dict(),
            decision=decision_to_dict(decision),
            action_results=results,
        )


def action_to_dict(action: Action) -> Dict[str, Any]:
    return {
        "name": action.name,
        "args": dict(action.args or {}),
        "action_id": action.action_id,
        "metadata": dict(action.metadata or {}),
    }


def _non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise JournalError(f"{field} is invalid")
    return value


def _has_exact_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    field: str,
) -> None:
    if set(payload) != expected:
        raise JournalError(f"{field} fields are invalid")


def action_from_dict(payload: Mapping[str, Any]) -> Action:
    _has_exact_keys(
        payload,
        {"name", "args", "action_id", "metadata"},
        "action",
    )
    name = payload["name"]
    args = payload["args"]
    action_id = payload["action_id"]
    metadata = payload["metadata"]
    if not isinstance(name, str) or not name.strip():
        raise JournalError("action name is invalid")
    if not isinstance(args, Mapping):
        raise JournalError("action args are invalid")
    if action_id is not None and not isinstance(action_id, str):
        raise JournalError("action action_id is invalid")
    if not isinstance(metadata, Mapping):
        raise JournalError("action metadata is invalid")
    try:
        action = Action(
            name=name,
            args=dict(args),
            action_id=action_id,
            metadata=dict(metadata),
        )
    except (TypeError, ValueError) as exc:
        raise JournalError("action is invalid") from exc
    if action_to_dict(action) != dict(payload):
        raise JournalError("action is not canonical")
    return action


def decision_to_dict(decision: Decision[Any]) -> Dict[str, Any]:
    return {
        "mode": decision.mode,
        "actions": [
            action_to_dict(
                action if isinstance(action, Action) else Action.from_dict(dict(action))
            )
            for action in decision.actions
        ],
        "final_answer": decision.final_answer,
        "rationale": decision.rationale,
        "meta": dict(decision.meta),
        "candidates": [decision_to_dict(item) for item in decision.candidates],
    }


def decision_from_dict(payload: Mapping[str, Any]) -> Decision[Action]:
    _has_exact_keys(
        payload,
        {"mode", "actions", "final_answer", "rationale", "meta", "candidates"},
        "decision",
    )
    mode = payload["mode"]
    raw_actions = payload["actions"]
    final_answer = payload["final_answer"]
    rationale = payload["rationale"]
    meta = payload["meta"]
    raw_candidates = payload["candidates"]
    if mode == "act":
        decision_mode: DecisionMode = "act"
    elif mode == "final":
        decision_mode = "final"
    elif mode == "wait":
        decision_mode = "wait"
    elif mode == "branch":
        decision_mode = "branch"
    elif mode == "handoff":
        decision_mode = "handoff"
    else:
        raise JournalError("decision mode is invalid")
    if not isinstance(raw_actions, list) or not all(
        isinstance(item, Mapping) for item in raw_actions
    ):
        raise JournalError("decision actions are invalid")
    if final_answer is not None and not isinstance(final_answer, str):
        raise JournalError("decision final_answer is invalid")
    if rationale is not None and not isinstance(rationale, str):
        raise JournalError("decision rationale is invalid")
    if not isinstance(meta, Mapping):
        raise JournalError("decision meta is invalid")
    if not isinstance(raw_candidates, list) or not all(
        isinstance(item, Mapping) for item in raw_candidates
    ):
        raise JournalError("decision candidates are invalid")
    try:
        decision = Decision(
            mode=decision_mode,
            actions=[action_from_dict(item) for item in raw_actions],
            final_answer=final_answer,
            rationale=rationale,
            meta=dict(meta),
            candidates=[decision_from_dict(item) for item in raw_candidates],
        )
        decision.validate()
    except (TypeError, ValueError) as exc:
        raise JournalError("decision is invalid") from exc
    if decision_to_dict(decision) != dict(payload):
        raise JournalError("decision is not canonical")
    return decision


def history_message_to_dict(message: HistoryMessage) -> Dict[str, Any]:
    return {
        "role": message.role,
        "step_id": message.step_id,
        "content": message.content,
        "reasoning_content": message.reasoning_content,
        "tool_calls": [dict(item) for item in message.tool_calls],
        "tool_call_id": message.tool_call_id,
        "name": message.name,
        "metadata": dict(message.metadata),
        "native_items": [dict(item) for item in message.native_items],
    }


def history_message_from_dict(payload: Mapping[str, Any]) -> HistoryMessage:
    _has_exact_keys(
        payload,
        {
            "role",
            "step_id",
            "content",
            "reasoning_content",
            "tool_calls",
            "tool_call_id",
            "name",
            "metadata",
            "native_items",
        },
        "history message",
    )
    role = payload["role"]
    reasoning_content = payload["reasoning_content"]
    raw_tool_calls = payload["tool_calls"]
    tool_call_id = payload["tool_call_id"]
    name = payload["name"]
    metadata = payload["metadata"]
    raw_native_items = payload["native_items"]
    if not isinstance(role, str) or not role:
        raise JournalError("history message role is invalid")
    step_id = _non_negative_int(payload["step_id"], "history message step_id")
    for value, field in (
        (reasoning_content, "reasoning_content"),
        (tool_call_id, "tool_call_id"),
        (name, "name"),
    ):
        if value is not None and not isinstance(value, str):
            raise JournalError(f"history message {field} is invalid")
    if not isinstance(raw_tool_calls, list) or not all(
        isinstance(item, Mapping) for item in raw_tool_calls
    ):
        raise JournalError("history message tool_calls are invalid")
    if not isinstance(metadata, Mapping):
        raise JournalError("history message metadata is invalid")
    if not isinstance(raw_native_items, list) or not all(
        isinstance(item, Mapping) for item in raw_native_items
    ):
        raise JournalError("history message native_items are invalid")
    message = HistoryMessage(
        role=role,
        step_id=step_id,
        content=payload["content"],
        reasoning_content=reasoning_content,
        tool_calls=[dict(item) for item in raw_tool_calls],
        tool_call_id=tool_call_id,
        name=name,
        metadata=dict(metadata),
        native_items=[dict(item) for item in raw_native_items],
    )
    if history_message_to_dict(message) != dict(payload):
        raise JournalError("history message is not canonical")
    return message


def history_append_from_payload(payload: Mapping[str, Any]) -> list[HistoryMessage]:
    raw_messages = payload.get("history_append", [])
    if not isinstance(raw_messages, list):
        raise JournalError("history_append must be an array")
    if not all(isinstance(item, Mapping) for item in raw_messages):
        raise JournalError("history_append entries must be objects")
    return [history_message_from_dict(item) for item in raw_messages]


def tool_result_history_message(
    action: Action,
    result: ToolResult,
    step_id: int,
) -> HistoryMessage:
    model_output = result.model_visible_output
    if isinstance(model_output, str):
        content = model_output
    else:
        try:
            content = json.dumps(model_output, ensure_ascii=False, default=str)
        except (TypeError, ValueError, OverflowError):
            content = str(model_output)
    return HistoryMessage(
        role="tool",
        step_id=step_id,
        content=content,
        tool_call_id=result.call_id or action.action_id,
        name=action.name,
        metadata={"source": "journal_recovery", "tool_name": action.name},
    )


def recover_pending_runtime_inputs(
    records: tuple[JournalRecord, ...],
) -> tuple[RuntimeInput, ...]:
    """Return local accepted events that no completed model turn consumed."""

    posted: dict[str, RuntimeInput] = {}
    consumed: set[str] = set()
    background_children = _background_child_handles(records)
    for record in records:
        # A fork receives an independent mailbox. Inherited parent events and
        # their consumption markers never become child input.
        if record.type is JournalRecordType.CHILD_TERMINAL:
            try:
                result = ChildResult.from_dict(record.payload)
                if result.handle.parent_run_id != record.run_id:
                    raise ValueError("child.terminal parent is inconsistent")
                if result.handle not in background_children:
                    continue
                event = child_terminal_runtime_input(result)
            except (TypeError, ValueError) as exc:
                raise JournalError(
                    "child.terminal event projection is invalid"
                ) from exc
            _add_runtime_input(posted, event)
            continue
        if record.type is JournalRecordType.PROCESS_TERMINAL:
            try:
                snapshot = ProcessSnapshot.from_dict(record.payload)
                if snapshot.handle.owner_run_id != record.run_id:
                    raise ValueError("process.terminal owner is inconsistent")
                event = process_terminal_runtime_input(snapshot)
            except (TypeError, ValueError) as exc:
                raise JournalError(
                    "process.terminal event projection is invalid"
                ) from exc
            _add_runtime_input(posted, event)
            continue
        if record.type is JournalRecordType.RUNTIME_INPUT_POSTED:
            raw_event = record.payload.get("event")
            if not isinstance(raw_event, Mapping):
                raise JournalError("runtime_input.posted is missing event")
            try:
                event = RuntimeInput.from_dict(raw_event)
            except (TypeError, ValueError) as exc:
                raise JournalError("runtime_input.posted event is invalid") from exc
            _add_runtime_input(posted, event)
            continue
        if record.type is not JournalRecordType.MODEL_COMPLETED:
            continue
        raw_ids = record.payload.get("runtime_input_ids", [])
        if not isinstance(raw_ids, list) or not all(
            isinstance(event_id, str) and event_id for event_id in raw_ids
        ):
            raise JournalError("model.completed runtime_input_ids are invalid")
        consumed.update(raw_ids)
    return tuple(
        event for event_id, event in posted.items() if event_id not in consumed
    )


def _add_runtime_input(posted: dict[str, RuntimeInput], event: RuntimeInput) -> None:
    previous = posted.setdefault(event.event_id, event)
    if previous != event:
        raise JournalError("runtime input id was reused with different content")


def _background_child_handles(
    records: tuple[JournalRecord, ...],
) -> set[ChildHandle]:
    modes: dict[ChildHandle, bool] = {}
    for record in records:
        if record.type is not JournalRecordType.CHILD_STARTED:
            continue
        raw_background = record.payload.get("background")
        if raw_background is None:
            continue
        raw_handle = record.payload.get("handle")
        if not isinstance(raw_background, bool) or not isinstance(raw_handle, Mapping):
            raise JournalError("child.started delivery policy is invalid")
        try:
            handle = ChildHandle.from_dict(raw_handle)
        except (TypeError, ValueError) as exc:
            raise JournalError("child.started delivery policy is invalid") from exc
        if handle.parent_run_id != record.run_id:
            raise JournalError("child.started parent is inconsistent")
        previous = modes.setdefault(handle, raw_background)
        if previous is not raw_background:
            raise JournalError("child.started delivery policy changed")
    return {handle for handle, background in modes.items() if background}


def _action_call_id(action: Action, step_id: int, action_index: int) -> str:
    return (
        str(action.action_id)
        if action.action_id not in (None, "")
        else f"call_{step_id}_{action_index}"
    )


def effective_journal_records(
    records: tuple[JournalRecord, ...],
) -> tuple[JournalRecord, ...]:
    return tuple(resolve_inherited_record(record) for record in records)


__all__ = ["_JournalRuntime", "history_message_to_dict"]
