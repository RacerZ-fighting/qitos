"""SessionJournal-backed transaction boundary for the minimal agent loop.

The loop only knows the ``TurnTransactionBoundary`` protocol; this recorder
maps its barriers onto canonical journal records. Transcript entries
(``transcript.message``) own message content exactly once; operation records
reference them by record id. Record ids are deterministic: one model
transaction and one turn commit per turn, one terminal record per Tool call,
one run terminal per run, one configuration diff per changed dimension per
turn, and a per-turn sequence number for transcript entries.

Payload schemas for the loop path are owned here and decode fail closed with
exact key sets; records written by the retired Engine path (``step_id`` /
``action_index`` / embedded ``result`` or ``messages``) are rejected instead
of being guessed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ...core.agent_loop import (
    AgentLoopResult,
    AgentRunStatus,
    TurnConfigSnapshot,
    TurnTransactionBoundary,
)
from ...core.budget import BudgetLedger
from ...core.journal import (
    JournalPosition,
    JournalRecordType,
    SessionJournal,
)
from ...core.message import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolResultMessage,
    message_from_dict,
    message_to_dict,
)
from ...core.model_request import ModelRequest
from ...core.model_response import ModelPricing, ModelUsage
from ...core.thinking import ThinkingLevel
from ...core.tool_result import ToolResult


# ── payload codecs (exact key sets, fail closed) ──────────────────────────


def _decode_turn(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("turn must be a non-negative integer")
    return value


def _decode_record_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be non-empty text")
    return value


def _decode_record_id_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{field} must be a list of non-empty record ids")
    return tuple(value)


def encode_transcript_message(message: Message) -> dict[str, Any]:
    return {"message": message_to_dict(message)}


def decode_transcript_message(payload: Mapping[str, Any]) -> Message:
    if set(payload) != {"message"}:
        raise ValueError("transcript.message fields are invalid")
    raw = payload["message"]
    if not isinstance(raw, Mapping):
        raise ValueError("transcript.message message must be a mapping")
    try:
        return message_from_dict(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("transcript.message message is not decodable") from exc


def encode_model_completed(
    turn: int, request: ModelRequest, message_record_id: str
) -> dict[str, Any]:
    return {
        "turn": turn,
        "request": request.to_dict(),
        "message_record_id": message_record_id,
    }


def decode_model_completed(
    payload: Mapping[str, Any],
) -> tuple[int, ModelRequest, str]:
    if set(payload) != {"turn", "request", "message_record_id"}:
        raise ValueError("model.completed fields are invalid")
    turn = _decode_turn(payload["turn"])
    raw_request = payload["request"]
    if not isinstance(raw_request, Mapping):
        raise ValueError("model.completed request must be a mapping")
    try:
        request = ModelRequest.from_dict(raw_request)
    except (TypeError, ValueError) as exc:
        raise ValueError("model.completed request is not decodable") from exc
    return (
        turn,
        request,
        _decode_record_id(payload["message_record_id"], "message_record_id"),
    )


def encode_tool_started(turn: int, call: ToolCall) -> dict[str, Any]:
    return {"turn": turn, "call": call.to_dict()}


def decode_tool_started(payload: Mapping[str, Any]) -> tuple[int, ToolCall]:
    if set(payload) != {"turn", "call"}:
        raise ValueError("tool.started fields are invalid")
    raw_call = payload["call"]
    if not isinstance(raw_call, Mapping):
        raise ValueError("tool.started call must be a mapping")
    try:
        call = ToolCall.from_dict(raw_call)
    except (TypeError, ValueError) as exc:
        raise ValueError("tool.started call is not decodable") from exc
    return _decode_turn(payload["turn"]), call


def encode_tool_terminal(
    turn: int, call: ToolCall, message_record_id: str
) -> dict[str, Any]:
    return {
        "turn": turn,
        "call_id": call.id,
        "call": call.to_dict(),
        "message_record_id": message_record_id,
    }


def decode_tool_terminal(payload: Mapping[str, Any]) -> tuple[int, ToolCall, str]:
    if set(payload) != {"turn", "call_id", "call", "message_record_id"}:
        raise ValueError("tool.terminal fields are invalid")
    turn = _decode_turn(payload["turn"])
    call_id = _decode_record_id(payload["call_id"], "call_id")
    raw_call = payload["call"]
    if not isinstance(raw_call, Mapping):
        raise ValueError("tool.terminal call must be a mapping")
    try:
        call = ToolCall.from_dict(raw_call)
    except (TypeError, ValueError) as exc:
        raise ValueError("tool.terminal call is not decodable") from exc
    if call.id != call_id:
        raise ValueError("tool.terminal call does not match its call_id")
    return (
        turn,
        call,
        _decode_record_id(payload["message_record_id"], "message_record_id"),
    )


def encode_step_committed(
    turn: int,
    transcript_record_ids: Sequence[str],
    tool_terminal_record_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "turn": turn,
        "transcript_record_ids": list(transcript_record_ids),
        "tool_terminal_record_ids": list(tool_terminal_record_ids),
    }


def decode_step_committed(
    payload: Mapping[str, Any],
) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    if set(payload) != {"turn", "transcript_record_ids", "tool_terminal_record_ids"}:
        raise ValueError("step.committed fields are invalid")
    return (
        _decode_turn(payload["turn"]),
        _decode_record_id_list(
            payload["transcript_record_ids"], "transcript_record_ids"
        ),
        _decode_record_id_list(
            payload["tool_terminal_record_ids"], "tool_terminal_record_ids"
        ),
    )


def encode_run_terminal(status: AgentRunStatus, error: str | None) -> dict[str, Any]:
    return {"status": status.value, "error": error}


def decode_run_terminal(
    record_type: JournalRecordType, payload: Mapping[str, Any]
) -> tuple[AgentRunStatus, str | None]:
    if record_type not in (
        JournalRecordType.RUN_COMPLETED,
        JournalRecordType.RUN_INTERRUPTED,
    ):
        raise ValueError("record type is not a run terminal")
    if set(payload) != {"status", "error"}:
        raise ValueError("run terminal fields are invalid")
    try:
        status = AgentRunStatus(str(payload["status"]))
    except ValueError as exc:
        raise ValueError("run terminal status is not a loop run status") from exc
    if record_type is JournalRecordType.RUN_INTERRUPTED:
        if status is not AgentRunStatus.ABORTED:
            raise ValueError("run.interrupted must carry an aborted status")
    elif status is AgentRunStatus.ABORTED:
        raise ValueError("run.completed cannot carry an aborted status")
    error = payload["error"]
    if error is not None and not isinstance(error, str):
        raise ValueError("run terminal error must be text or None")
    return status, error


def decode_input_accepted(payload: Mapping[str, Any]) -> tuple[str, ...]:
    if set(payload) != {"transcript_record_ids"}:
        raise ValueError("input.accepted fields are invalid")
    return _decode_record_id_list(
        payload["transcript_record_ids"], "transcript_record_ids"
    )


def encode_model_change(identity: tuple[str, str, str]) -> dict[str, Any]:
    provider, model, api = identity
    return {"provider": provider, "model": model, "api": api}


def decode_model_change(payload: Mapping[str, Any]) -> tuple[str, str, str]:
    if set(payload) != {"provider", "model", "api"}:
        raise ValueError("model.change fields are invalid")
    identity: list[str] = []
    for field in ("provider", "model", "api"):
        identity.append(_decode_record_id(payload[field], f"model.change {field}"))
    return identity[0], identity[1], identity[2]


def encode_thinking_change(level: ThinkingLevel) -> dict[str, Any]:
    return {"level": level.value}


def decode_thinking_change(payload: Mapping[str, Any]) -> ThinkingLevel:
    if set(payload) != {"level"}:
        raise ValueError("thinking.change fields are invalid")
    raw = payload["level"]
    if not isinstance(raw, str):
        raise ValueError("thinking.change level must be text")
    try:
        return ThinkingLevel(raw)
    except ValueError as exc:
        raise ValueError("thinking.change level is not a ThinkingLevel") from exc


def encode_tools_change(tool_names: Sequence[str]) -> dict[str, Any]:
    return {"active_tool_names": list(tool_names)}


def decode_tools_change(payload: Mapping[str, Any]) -> tuple[str, ...]:
    if set(payload) != {"active_tool_names"}:
        raise ValueError("tools.change fields are invalid")
    names = _decode_record_id_list(
        payload["active_tool_names"], "active_tool_names"
    )
    if len(names) != len(set(names)):
        raise ValueError("tools.change active_tool_names must be unique")
    return names


def encode_compaction(
    summary: str,
    first_kept_transcript_id: str,
    tokens_before: int,
    usage: ModelUsage | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "summary": summary,
        "first_kept_transcript_id": first_kept_transcript_id,
        "tokens_before": tokens_before,
    }
    if usage is not None:
        payload["usage"] = usage.to_dict()
    return payload


def decode_compaction(
    payload: Mapping[str, Any],
) -> tuple[str, str, int, ModelUsage | None]:
    if not {"summary", "first_kept_transcript_id", "tokens_before"}.issubset(
        payload
    ) or not set(payload).issubset(
        {"summary", "first_kept_transcript_id", "tokens_before", "usage"}
    ):
        raise ValueError("compaction fields are invalid")
    summary = payload["summary"]
    if not isinstance(summary, str):
        raise ValueError("compaction summary must be text")
    first_kept = _decode_record_id(
        payload["first_kept_transcript_id"], "first_kept_transcript_id"
    )
    tokens_before = payload["tokens_before"]
    if (
        isinstance(tokens_before, bool)
        or not isinstance(tokens_before, int)
        or tokens_before < 0
    ):
        raise ValueError("compaction tokens_before must be a non-negative integer")
    raw_usage = payload.get("usage")
    usage: ModelUsage | None = None
    if raw_usage is not None:
        if not isinstance(raw_usage, Mapping):
            raise ValueError("compaction usage must be a mapping or absent")
        try:
            usage = ModelUsage.from_mapping(raw_usage)
        except (TypeError, ValueError) as exc:
            raise ValueError("compaction usage is not decodable") from exc
    return summary, first_kept, tokens_before, usage


def encode_runtime_input_consumed(event_id: str) -> dict[str, Any]:
    return {"event_id": event_id}


def decode_runtime_input_consumed(payload: Mapping[str, Any]) -> str:
    if set(payload) != {"event_id"}:
        raise ValueError("runtime_input.consumed fields are invalid")
    return _decode_record_id(payload["event_id"], "event_id")


# ── recorder ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RecoveredRecorderState:
    """Seeding state that lets a resumed recorder continue one journal.

    ``next_turn`` continues turn numbering. The configuration triple carries
    the last journaled freeze facts (``None`` where the lineage never
    journaled that dimension) so the recorder writes diffs only instead of
    rewriting history. ``recorded_message_count`` is the number of transcript
    entries replayed through the lineage, kept for continuation accounting.
    ``input_accepted`` marks that this run already committed its initial
    input: later prompt batches (steering converted by the façade) stay
    durable through their turn commits and must not write a second
    ``input.accepted`` record.
    """

    next_turn: int
    model_identity: tuple[str, str, str] | None
    thinking_level: ThinkingLevel | None
    active_tool_names: tuple[str, ...] | None
    recorded_message_count: int
    input_accepted: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.next_turn, bool) or not isinstance(
            self.next_turn, int
        ) or self.next_turn < 0:
            raise ValueError("next_turn must be a non-negative integer")
        if self.model_identity is not None and (
            not isinstance(self.model_identity, tuple)
            or len(self.model_identity) != 3
            or not all(isinstance(item, str) and item for item in self.model_identity)
        ):
            raise TypeError("model_identity must be a (provider, model, api) tuple")
        if self.thinking_level is not None and not isinstance(
            self.thinking_level, ThinkingLevel
        ):
            raise TypeError("thinking_level must be a ThinkingLevel or None")
        if self.active_tool_names is not None and (
            not isinstance(self.active_tool_names, tuple)
            or not all(
                isinstance(name, str) and name for name in self.active_tool_names
            )
        ):
            raise TypeError("active_tool_names must be a tuple of non-empty strings")
        if isinstance(self.recorded_message_count, bool) or not isinstance(
            self.recorded_message_count, int
        ) or self.recorded_message_count < 0:
            raise ValueError("recorded_message_count must be a non-negative integer")
        if not isinstance(self.input_accepted, bool):
            raise TypeError("input_accepted must be a boolean")


class JournalTurnTransaction(TurnTransactionBoundary):
    """Record loop transaction barriers into one run's SessionJournal.

    Message content enters the journal exactly once as a ``transcript.message``
    record; every operation record references it by record id. Tool-result
    transcript messages are constructed deterministically from the call and
    its terminal result, so ``turn_committed`` only writes transcript entries
    for messages no earlier barrier recorded (steering and follow-up user
    messages, assistant messages that never reached model admission).
    """

    def __init__(
        self,
        journal: SessionJournal,
        *,
        recovered: RecoveredRecorderState | None = None,
        budget_ledger: BudgetLedger | None = None,
        model_pricing: ModelPricing | None = None,
    ) -> None:
        if recovered is not None and not isinstance(
            recovered, RecoveredRecorderState
        ):
            raise TypeError("recovered must be a RecoveredRecorderState or None")
        if budget_ledger is not None and not isinstance(budget_ledger, BudgetLedger):
            raise TypeError("budget_ledger must be a BudgetLedger or None")
        if model_pricing is not None and not isinstance(model_pricing, ModelPricing):
            raise TypeError("model_pricing must be a ModelPricing or None")
        self._journal = journal
        seed = recovered or RecoveredRecorderState(0, None, None, None, 0)
        self._turn = seed.next_turn
        self._transcript_seq = 0
        self._last_model_identity = seed.model_identity
        self._last_thinking_level = seed.thinking_level
        self._last_tool_names = seed.active_tool_names
        self._recorded_message_count = seed.recorded_message_count
        # Recorded messages stay referenced for the recorder's lifetime so an
        # identity lookup can never hit a reused object address.
        self._recorded_messages: list[Message] = []
        self._message_record_ids: dict[int, str] = {}
        self._tool_message_record_ids: dict[tuple[int, str], str] = {}
        self._tool_terminal_record_ids: dict[tuple[int, str], str] = {}
        self._budget_ledger = budget_ledger
        self._model_pricing = model_pricing
        self._input_accepted = seed.input_accepted

    @property
    def journal(self) -> SessionJournal:
        return self._journal

    @property
    def recorded_message_count(self) -> int:
        """Transcript entries journaled through this recorder (seed included)."""

        return self._recorded_message_count

    @classmethod
    async def create(
        cls,
        journal: SessionJournal,
        run_id: str,
        metadata: Mapping[str, Any],
        **kwargs: Any,
    ) -> "JournalTurnTransaction":
        """Create the run journal and return its transaction recorder."""

        await journal.create(run_id, metadata)
        return cls(journal, **kwargs)

    def _require_turn(self, turn: int) -> None:
        if isinstance(turn, bool) or not isinstance(turn, int) or turn < 0:
            raise ValueError("turn must be a non-negative integer")
        if turn < self._turn:
            raise ValueError("transaction turns must not regress")
        if turn != self._turn:
            self._turn = turn
            self._transcript_seq = 0

    async def _append_transcript(self, message: Message) -> str:
        record_id = (
            f"{self._journal.run_id}:turn:{self._turn}"
            f":transcript:{self._transcript_seq}"
        )
        await self._journal.append(
            JournalRecordType.TRANSCRIPT_MESSAGE,
            encode_transcript_message(message),
            record_id=record_id,
        )
        self._transcript_seq += 1
        self._recorded_message_count += 1
        self._recorded_messages.append(message)
        self._message_record_ids[id(message)] = record_id
        if isinstance(message, ToolResultMessage):
            self._tool_message_record_ids[(self._turn, message.tool_call_id)] = (
                record_id
            )
        return record_id

    async def input_accepted(self, prompts: tuple[Message, ...]) -> JournalPosition | None:
        """Commit the run's prompt transcript entries before model side effects.

        One Run accepts its initial input exactly once. When a recovered or
        steering-driven run already has its ``input.accepted`` record, the
        batch is left to its turn commit (the durable path every steered
        message takes) and no second input record is written; the return is
        then ``None``.
        """

        if not prompts:
            raise ValueError("input_accepted requires at least one prompt message")
        if self._input_accepted:
            return None
        record_ids = [await self._append_transcript(prompt) for prompt in prompts]
        position = await self._journal.append(
            JournalRecordType.INPUT_ACCEPTED,
            {"transcript_record_ids": record_ids},
            record_id=f"{self._journal.run_id}:input",
        )
        self._input_accepted = True
        return position

    async def turn_frozen(
        self, turn: int, config: TurnConfigSnapshot
    ) -> tuple[JournalPosition, ...]:
        """Journal the per-turn freeze diffs against the last recorded config."""

        if not isinstance(config, TurnConfigSnapshot):
            raise TypeError("config must be a TurnConfigSnapshot")
        self._require_turn(turn)
        run_id = self._journal.run_id
        positions: list[JournalPosition] = []
        identity = (config.provider, config.model, config.api)
        if identity != self._last_model_identity:
            positions.append(
                await self._journal.append(
                    JournalRecordType.MODEL_CHANGE,
                    encode_model_change(identity),
                    record_id=f"{run_id}:turn:{turn}:config:model",
                )
            )
            self._last_model_identity = identity
        level = (
            config.thinking_level
            if config.thinking_level is not None
            else ThinkingLevel.OFF
        )
        if self._last_thinking_level is None or level != self._last_thinking_level:
            positions.append(
                await self._journal.append(
                    JournalRecordType.THINKING_CHANGE,
                    encode_thinking_change(level),
                    record_id=f"{run_id}:turn:{turn}:config:thinking",
                )
            )
            self._last_thinking_level = level
        if config.tool_names != self._last_tool_names:
            positions.append(
                await self._journal.append(
                    JournalRecordType.TOOLS_CHANGE,
                    encode_tools_change(config.tool_names),
                    record_id=f"{run_id}:turn:{turn}:config:tools",
                )
            )
            self._last_tool_names = config.tool_names
        return tuple(positions)

    async def model_terminal(
        self, turn: int, request: ModelRequest, message: AssistantMessage
    ) -> JournalPosition:
        if not isinstance(message, AssistantMessage):
            raise TypeError("message must be an AssistantMessage")
        self._require_turn(turn)
        message_record_id = await self._append_transcript(message)
        if self._budget_ledger is not None:
            # Root runs commit durable usage the same way the child boundary
            # does: per model terminal, keyed by the run's own deterministic
            # model-transaction id, before the model record itself.
            tokens, cost, usage_complete, cost_complete = _usage_accounting(
                message, model_pricing=self._model_pricing
            )
            await self._budget_ledger.commit(
                origin_run_id=self._journal.run_id,
                transaction_id=f"{self._journal.run_id}:turn:{turn}:model",
                tokens=tokens,
                cost_usd=cost,
                usage_complete=usage_complete,
                cost_complete=cost_complete,
            )
        return await self._journal.append(
            JournalRecordType.MODEL_COMPLETED,
            encode_model_completed(turn, request, message_record_id),
            record_id=f"{self._journal.run_id}:turn:{turn}:model",
        )

    async def tool_started(self, turn: int, call: ToolCall) -> JournalPosition:
        self._require_turn(turn)
        return await self._journal.append(
            JournalRecordType.TOOL_STARTED,
            encode_tool_started(turn, call),
            record_id=f"{self._journal.run_id}:turn:{turn}:tool:{call.id}:started",
        )

    async def tool_terminal(
        self, turn: int, call: ToolCall, result: ToolResult
    ) -> JournalPosition:
        self._require_turn(turn)
        message = ToolResultMessage(
            tool_call_id=call.id,
            tool_name=call.name,
            result=result,
            usage=result.usage,
            added_tool_names=result.added_tool_names,
        )
        message_record_id = await self._append_transcript(message)
        terminal_record_id = (
            f"{self._journal.run_id}:turn:{turn}:tool:{call.id}:terminal"
        )
        position = await self._journal.append(
            JournalRecordType.TOOL_TERMINAL,
            encode_tool_terminal(turn, call, message_record_id),
            record_id=terminal_record_id,
        )
        self._tool_terminal_record_ids[(turn, call.id)] = terminal_record_id
        return position

    async def turn_committed(
        self, turn: int, new_messages: tuple[Message, ...]
    ) -> JournalPosition:
        self._require_turn(turn)
        transcript_record_ids: list[str] = []
        tool_terminal_record_ids: list[str] = []
        for message in new_messages:
            record_id: str | None
            if isinstance(message, ToolResultMessage):
                record_id = self._tool_message_record_ids.get(
                    (turn, message.tool_call_id)
                )
                if record_id is None:
                    raise ValueError(
                        "turn_committed received a ToolResultMessage whose "
                        "transcript entry was never recorded"
                    )
                terminal_record_id = self._tool_terminal_record_ids.get(
                    (turn, message.tool_call_id)
                )
                if terminal_record_id is not None:
                    tool_terminal_record_ids.append(terminal_record_id)
            else:
                record_id = self._message_record_ids.get(id(message))
            if record_id is None:
                record_id = await self._append_transcript(message)
            transcript_record_ids.append(record_id)
        return await self._journal.append(
            JournalRecordType.STEP_COMMITTED,
            encode_step_committed(
                turn, transcript_record_ids, tool_terminal_record_ids
            ),
            record_id=f"{self._journal.run_id}:turn:{turn}:committed",
        )

    async def run_terminal(self, result: AgentLoopResult) -> JournalPosition:
        record_type = (
            JournalRecordType.RUN_INTERRUPTED
            if result.status is AgentRunStatus.ABORTED
            else JournalRecordType.RUN_COMPLETED
        )
        return await self._journal.append(
            record_type,
            encode_run_terminal(result.status, result.error),
            record_id=f"{self._journal.run_id}:run:terminal",
        )

    async def runtime_input_consumed(self, event_id: str) -> JournalPosition:
        """Mark one posted runtime input as consumed (idempotent)."""

        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event_id must be non-empty text")
        return await self._journal.append(
            JournalRecordType.RUNTIME_INPUT_CONSUMED,
            encode_runtime_input_consumed(event_id),
            record_id=f"{self._journal.run_id}:runtime:{event_id}:consumed",
        )

    async def compaction(
        self,
        *,
        summary: str,
        first_kept_transcript_id: str,
        tokens_before: int,
        usage: ModelUsage | None = None,
    ) -> JournalPosition:
        """Append one durable compaction entry (the flow itself is S2b)."""

        payload = encode_compaction(
            summary, first_kept_transcript_id, tokens_before, usage
        )
        # Validate before writing: the codec round trip is the fail-closed gate.
        decode_compaction(payload)
        return await self._journal.append(
            JournalRecordType.COMPACTION,
            payload,
            record_id=(
                f"{self._journal.run_id}:compaction:{first_kept_transcript_id}"
            ),
        )


def _usage_accounting(
    message: AssistantMessage,
    *,
    model_pricing: ModelPricing | None,
) -> tuple[int, float, bool, bool]:
    """Token/cost accounting of one model terminal (child-boundary parity)."""

    usage = message.usage
    if usage is None:
        return (0, 0.0, False, False)
    if usage.total_tokens is not None:
        tokens = usage.total_tokens
        tokens_known = True
    elif usage.input_tokens is not None and usage.output_tokens is not None:
        tokens = usage.input_tokens + usage.output_tokens
        tokens_known = True
    else:
        tokens = 0
        tokens_known = False
    cost_known = (
        model_pricing is not None
        and usage.input_tokens is not None
        and usage.output_tokens is not None
    )
    cost = model_pricing.cost_usd(usage) if cost_known and model_pricing else 0.0
    return (tokens, cost, tokens_known, cost_known)


__all__ = [
    "JournalTurnTransaction",
    "RecoveredRecorderState",
    "decode_compaction",
    "decode_input_accepted",
    "decode_model_change",
    "decode_model_completed",
    "decode_run_terminal",
    "decode_step_committed",
    "decode_thinking_change",
    "decode_tool_started",
    "decode_tool_terminal",
    "decode_tools_change",
    "decode_transcript_message",
    "decode_runtime_input_consumed",
    "encode_compaction",
    "encode_model_change",
    "encode_model_completed",
    "encode_run_terminal",
    "encode_step_committed",
    "encode_thinking_change",
    "encode_tool_started",
    "encode_tool_terminal",
    "encode_tools_change",
    "encode_transcript_message",
    "encode_runtime_input_consumed",
]
