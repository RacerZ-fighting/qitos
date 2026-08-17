"""SessionJournal-backed transaction boundary for the minimal agent loop.

The loop only knows the ``TurnTransactionBoundary`` protocol; this recorder
maps its barriers onto canonical journal records with deterministic record
ids (one model transaction and one turn commit per turn, one terminal record
per Tool call, one run terminal per run). Payload schemas for the loop path
are owned here; journals written by the retired Engine path are not mixed
into the same run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ...core.agent_loop import AgentLoopResult, AgentRunStatus, TurnTransactionBoundary
from ...core.journal import (
    JournalPosition,
    JournalRecord,
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
from ...core.tool_result import ToolResult


class JournalTurnTransaction(TurnTransactionBoundary):
    """Record loop transaction barriers into one run's SessionJournal."""

    def __init__(self, journal: SessionJournal) -> None:
        self._journal = journal

    @property
    def journal(self) -> SessionJournal:
        return self._journal

    @classmethod
    async def create(
        cls,
        journal: SessionJournal,
        run_id: str,
        metadata: Mapping[str, Any],
    ) -> "JournalTurnTransaction":
        """Create the run journal and return its transaction recorder."""

        await journal.create(run_id, metadata)
        return cls(journal)

    async def model_terminal(
        self, turn: int, request: ModelRequest, message: AssistantMessage
    ) -> JournalPosition:
        return await self._journal.append(
            JournalRecordType.MODEL_COMPLETED,
            {
                "turn": turn,
                "request": request.to_dict(),
                "message": message_to_dict(message),
            },
            record_id=f"{self._journal.run_id}:turn:{turn}:model",
        )

    async def tool_started(self, turn: int, call: ToolCall) -> JournalPosition:
        return await self._journal.append(
            JournalRecordType.TOOL_STARTED,
            {"turn": turn, "call": call.to_dict()},
            record_id=f"{self._journal.run_id}:turn:{turn}:tool:{call.id}:started",
        )

    async def tool_terminal(
        self, turn: int, call: ToolCall, result: ToolResult
    ) -> JournalPosition:
        return await self._journal.append(
            JournalRecordType.TOOL_TERMINAL,
            {
                "turn": turn,
                "call_id": call.id,
                "call": call.to_dict(),
                "result": result.to_dict(),
            },
            record_id=f"{self._journal.run_id}:turn:{turn}:tool:{call.id}:terminal",
        )

    async def turn_committed(
        self, turn: int, new_messages: tuple[Message, ...]
    ) -> JournalPosition:
        # The JSONL index links committed Tool terminals to their commit
        # record through terminal_record_ids; without them a committed Tool
        # transaction of this run could not be queried back or recovered.
        terminal_record_ids = [
            f"{self._journal.run_id}:turn:{turn}:tool:{message.tool_call_id}:terminal"
            for message in new_messages
            if isinstance(message, ToolResultMessage)
        ]
        return await self._journal.append(
            JournalRecordType.STEP_COMMITTED,
            {
                "turn": turn,
                "messages": [message_to_dict(message) for message in new_messages],
                "terminal_record_ids": terminal_record_ids,
            },
            record_id=f"{self._journal.run_id}:turn:{turn}:committed",
        )

    async def run_terminal(self, result: AgentLoopResult) -> JournalPosition:
        record_type = (
            JournalRecordType.RUN_INTERRUPTED
            if result.status.value == "aborted"
            else JournalRecordType.RUN_COMPLETED
        )
        return await self._journal.append(
            record_type,
            {
                "status": result.status.value,
                "error": result.error,
                "messages": [message_to_dict(message) for message in result.messages],
            },
            record_id=f"{self._journal.run_id}:run:terminal",
        )


@dataclass(frozen=True, slots=True)
class RecoveredRunOutcome:
    """Terminal outcome decoded from one run journal's run terminal record."""

    status: AgentRunStatus
    error: str | None
    messages: tuple[Message, ...]


def recover_run_outcome(
    records: Sequence[JournalRecord],
) -> RecoveredRunOutcome | None:
    """Decode the run terminal record written by ``JournalTurnTransaction``.

    Returns ``None`` when the run never reached a terminal record (the process
    exited mid-run). Raises ``ValueError`` when a terminal record exists but
    does not match this recorder's payload schema — journals written by the
    retired Engine path are not recoverable through this read path.
    """

    terminal = [
        record
        for record in records
        if record.type
        in (JournalRecordType.RUN_COMPLETED, JournalRecordType.RUN_INTERRUPTED)
    ]
    if not terminal:
        return None
    record = terminal[-1]
    if any(item != record for item in terminal[:-1]):
        raise ValueError("run journal contains conflicting terminal records")
    payload = record.payload
    if not isinstance(payload, Mapping) or set(payload) != {
        "status",
        "error",
        "messages",
    }:
        raise ValueError("run terminal payload is not a turn-recorder record")
    try:
        status = AgentRunStatus(str(payload["status"]))
    except ValueError as exc:
        raise ValueError("run terminal status is not a loop run status") from exc
    if record.type is JournalRecordType.RUN_INTERRUPTED:
        if status is not AgentRunStatus.ABORTED:
            raise ValueError("run.interrupted must carry an aborted status")
    elif status is AgentRunStatus.ABORTED:
        raise ValueError("run.completed cannot carry an aborted status")
    error = payload["error"]
    if error is not None and not isinstance(error, str):
        raise ValueError("run terminal error must be text or None")
    raw_messages = payload["messages"]
    if not isinstance(raw_messages, Sequence) or isinstance(
        raw_messages, (str, bytes)
    ):
        raise ValueError("run terminal messages must be a sequence")
    try:
        messages = tuple(message_from_dict(item) for item in raw_messages)
    except (TypeError, ValueError) as exc:
        raise ValueError("run terminal messages are not decodable") from exc
    return RecoveredRunOutcome(status=status, error=error, messages=messages)


__all__ = ["JournalTurnTransaction", "RecoveredRunOutcome", "recover_run_outcome"]
