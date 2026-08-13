"""Private canonical Run journal state machine for Engine."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, Generic, Mapping, TypeVar, cast
from uuid import uuid4

from ..core.action import Action
from ..core.agent_module import ActionResultContext, CanonicalActionResult
from ..core.decision import Decision
from ..core.errors import StopReason
from ..core.history import HistoryMessage
from ..core.journal import JournalError, JournalRecord, JournalRecordType
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
            {"agent": engine.agent.name},
        )
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
            return
        history_append = list(engine._journal_pending_history)
        engine._last_journal_position = await journal.append(
            JournalRecordType.MODEL_COMPLETED,
            {
                "step_id": record.step_id,
                "transaction_id": record.transaction_id,
                "model_response": dict(record.model_response),
                "decision": decision_to_dict(decision),
                "history_append": history_append,
            },
            record_id=f"{record.transaction_id}:model",
        )
        del engine._journal_pending_history[: len(history_append)]

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
            result = ToolResult.from_value(
                engine.agent.finalize_action_result(
                    state,
                    action,
                    ToolResult.from_value(raw_result),
                    step_id=record.step_id,
                    context=ActionResultContext(
                        prior_results=tuple(engine._canonical_action_results)
                    ),
                )
            )
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
            finalized.append(result)
            engine._canonical_action_results.append(
                CanonicalActionResult(record.step_id, action, result)
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
            engine._active_state.to_dict()
            if engine._active_state is not None
            else None
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
                step_id = int(payload.get("step_id") or 0)
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
                raw_decision = payload.get("decision")
                if not isinstance(raw_decision, Mapping):
                    raise JournalError("model.completed is missing decision")
                record.decision = decision_from_dict(raw_decision)
                record.actions = list(record.decision.actions)
                transactions.setdefault(
                    transaction_id,
                    {
                        "step_id": step_id,
                        "decision": record.decision,
                        "started": set(),
                        "terminals": {},
                        "committed": False,
                    },
                )
                history.extend(history_append_from_payload(payload))
                continue
            if journal_record.type is JournalRecordType.TOOL_STARTED:
                transaction_id = str(payload.get("transaction_id") or "")
                action_index = payload.get("action_index")
                if (
                    not transaction_id
                    or isinstance(action_index, bool)
                    or not isinstance(action_index, int)
                ):
                    raise JournalError("tool.started payload is invalid")
                transaction = transactions.get(transaction_id)
                if transaction is None:
                    raise JournalError("tool.started has no model transaction")
                transaction["started"].add(action_index)
                continue
            if journal_record.type is JournalRecordType.TOOL_TERMINAL:
                transaction_id = str(payload.get("transaction_id") or "")
                step_id = int(payload.get("step_id") or 0)
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
                record.action_results.append(terminal_result)
                transaction = transactions.get(transaction_id)
                action_index = payload.get("action_index")
                if (
                    transaction is None
                    or isinstance(action_index, bool)
                    or not isinstance(action_index, int)
                ):
                    raise JournalError("tool.terminal has no model transaction")
                transaction["terminals"][action_index] = terminal_result
                action_payload = payload.get("action")
                if not isinstance(action_payload, Mapping):
                    raise JournalError("tool.terminal is missing action")
                canonical_results.append(
                    CanonicalActionResult(
                        step_id,
                        Action.from_dict(dict(action_payload)),
                        terminal_result,
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
                transaction["committed"] = True
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
            ordered_results = [
                terminal_results[index] for index in range(len(actions))
            ]
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
            if not working.final_result and not working.stop_reason:
                working.advance_step()
            state_data = working.to_dict()
            if state_data.get("final_result") and not state_data.get("stop_reason"):
                state_data["stop_reason"] = StopReason.FINAL.value
            history_append = [
                tool_result_history_message(
                    action,
                    result,
                    int(transaction["step_id"]),
                )
                for action, result in zip(actions, ordered_results, strict=True)
            ]
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


def decision_to_dict(decision: Decision[Any]) -> Dict[str, Any]:
    return {
        "mode": decision.mode,
        "actions": [
            action_to_dict(
                action
                if isinstance(action, Action)
                else Action.from_dict(dict(action))
            )
            for action in decision.actions
        ],
        "final_answer": decision.final_answer,
        "rationale": decision.rationale,
        "meta": dict(decision.meta),
        "candidates": [decision_to_dict(item) for item in decision.candidates],
    }


def decision_from_dict(payload: Mapping[str, Any]) -> Decision[Action]:
    decision = Decision(
        mode=cast(Any, str(payload.get("mode") or "")),
        actions=[
            Action.from_dict(dict(item))
            for item in list(payload.get("actions") or [])
            if isinstance(item, Mapping)
        ],
        final_answer=(
            str(payload["final_answer"])
            if payload.get("final_answer") is not None
            else None
        ),
        rationale=(
            str(payload["rationale"])
            if payload.get("rationale") is not None
            else None
        ),
        meta=(
            dict(payload.get("meta") or {})
            if isinstance(payload.get("meta"), Mapping)
            else {}
        ),
        candidates=[
            decision_from_dict(item)
            for item in list(payload.get("candidates") or [])
            if isinstance(item, Mapping)
        ],
    )
    decision.validate()
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
    return HistoryMessage(
        role=str(payload.get("role") or ""),
        step_id=int(payload.get("step_id") or 0),
        content=payload.get("content"),
        reasoning_content=(
            str(payload["reasoning_content"])
            if payload.get("reasoning_content") is not None
            else None
        ),
        tool_calls=[
            dict(item)
            for item in list(payload.get("tool_calls") or [])
            if isinstance(item, Mapping)
        ],
        tool_call_id=(
            str(payload["tool_call_id"])
            if payload.get("tool_call_id") is not None
            else None
        ),
        name=str(payload["name"]) if payload.get("name") is not None else None,
        metadata=(
            dict(payload.get("metadata") or {})
            if isinstance(payload.get("metadata"), Mapping)
            else {}
        ),
        native_items=[
            dict(item)
            for item in list(payload.get("native_items") or [])
            if isinstance(item, Mapping)
        ],
    )


def history_append_from_payload(payload: Mapping[str, Any]) -> list[HistoryMessage]:
    raw_messages = payload.get("history_append") or []
    if not isinstance(raw_messages, list):
        raise JournalError("history_append must be an array")
    return [
        history_message_from_dict(item)
        for item in raw_messages
        if isinstance(item, Mapping)
    ]


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
        tool_call_id=action.action_id,
        name=action.name,
        metadata={"source": "journal_recovery", "tool_name": action.name},
    )


def effective_journal_records(
    records: tuple[JournalRecord, ...],
) -> tuple[JournalRecord, ...]:
    effective: list[JournalRecord] = []
    for record in records:
        if record.type is not JournalRecordType.INHERITED:
            effective.append(record)
            continue
        raw_record = record.payload.get("record")
        if not isinstance(raw_record, Mapping):
            raise JournalError("journal.inherited is missing its origin record")
        effective.append(JournalRecord.from_dict(raw_record))
    return tuple(effective)


__all__ = ["_JournalRuntime", "history_message_to_dict"]
