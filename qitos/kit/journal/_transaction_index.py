"""Committed Tool terminal indexing shared by SessionJournal implementations.

The index tracks transcript entries, Tool terminals and turn commit markers
through INHERITED wrappers and answers ``find_tool_transaction`` by joining a
committed terminal with the ``transcript.message`` record it references.
Payloads decode fail closed at index time: records written by the retired
Engine taxonomy (embedded ``result``, ``step_id``/``action_index``, legacy
``terminal_record_ids``) raise ``JournalCorruptionError`` instead of being
silently skipped.
"""

from __future__ import annotations

from ...core.journal import (
    JournalCorruptionError,
    JournalRecord,
    JournalRecordRef,
    JournalRecordType,
    ToolTransaction,
    resolve_inherited_record,
)
from ...core.message import ToolCall, ToolResultMessage
from .turn_recorder import (
    decode_step_committed,
    decode_tool_terminal,
    decode_transcript_message,
)


class ToolTransactionIndex:
    """In-memory committed-Tool query view over one journal's replay."""

    def __init__(self) -> None:
        self._transcript: dict[JournalRecordRef, JournalRecord] = {}
        self._terminals: dict[JournalRecordRef, tuple[int, ToolCall, str]] = {}
        self._commits: dict[JournalRecordRef, JournalRecord] = {}

    def reset(self, records: tuple[JournalRecord, ...] | list[JournalRecord]) -> None:
        self._transcript = {}
        self._terminals = {}
        self._commits = {}
        for record in records:
            self.add(record)

    def add(self, record: JournalRecord) -> None:
        """Index one durable record, decoding its payload fail closed."""

        effective = resolve_inherited_record(record)
        try:
            if effective.type is JournalRecordType.TRANSCRIPT_MESSAGE:
                decode_transcript_message(effective.payload)
                reference = JournalRecordRef(effective.run_id, effective.record_id)
                existing = self._transcript.get(reference)
                if existing is not None and existing.to_dict() != effective.to_dict():
                    raise JournalCorruptionError(
                        "conflicting inherited transcript reference"
                    )
                self._transcript[reference] = effective
                return
            if effective.type is JournalRecordType.TOOL_TERMINAL:
                decoded = decode_tool_terminal(effective.payload)
                reference = JournalRecordRef(effective.run_id, effective.record_id)
                existing = self._terminals.get(reference)
                if existing is not None and existing != decoded:
                    raise JournalCorruptionError(
                        "conflicting inherited Tool terminal reference"
                    )
                self._terminals[reference] = decoded
                return
            if effective.type is not JournalRecordType.STEP_COMMITTED:
                return
            _turn, _transcript_ids, terminal_ids = decode_step_committed(
                effective.payload
            )
        except ValueError as exc:
            raise JournalCorruptionError(str(exc)) from exc
        for record_id in terminal_ids:
            reference = JournalRecordRef(effective.run_id, record_id)
            if reference in self._terminals:
                self._commits[reference] = effective

    def find(self, reference: JournalRecordRef) -> ToolTransaction | None:
        """Return one committed Tool terminal, joining its transcript entry."""

        if not isinstance(reference, JournalRecordRef):
            raise TypeError("reference must be a JournalRecordRef")
        decoded = self._terminals.get(reference)
        committed = self._commits.get(reference)
        if decoded is None or committed is None:
            return None
        _turn, call, message_record_id = decoded
        transcript = self._transcript.get(
            JournalRecordRef(reference.run_id, message_record_id)
        )
        if transcript is None:
            raise JournalCorruptionError(
                "tool.terminal references an unknown transcript entry"
            )
        message = decode_transcript_message(transcript.payload)
        if not isinstance(message, ToolResultMessage):
            raise JournalCorruptionError(
                "tool.terminal must reference a tool message"
            )
        if message.tool_call_id != call.id or message.tool_name != call.name:
            raise JournalCorruptionError(
                "tool.terminal does not match its transcript entry"
            )
        return ToolTransaction(
            terminal=reference,
            committed_at=committed.position,
            action=call,
            result=message.result,
        )


__all__ = ["ToolTransactionIndex"]
