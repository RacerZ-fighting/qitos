"""SessionJournal-backed transaction boundary for the minimal agent loop.

The loop only knows the ``TurnTransactionBoundary`` protocol; this recorder
maps its barriers onto canonical journal records with deterministic record
ids (one model transaction and one turn commit per turn, one terminal record
per Tool call, one run terminal per run). Payload schemas for the loop path
are owned here; journals written by the retired Engine path are not mixed
into the same run.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...core.agent_loop import AgentLoopResult, TurnTransactionBoundary
from ...core.journal import JournalPosition, JournalRecordType, SessionJournal
from ...core.message import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolResultMessage,
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


__all__ = ["JournalTurnTransaction"]
