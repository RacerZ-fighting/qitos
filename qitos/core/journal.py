"""Canonical append-only Run journal contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Protocol

from .action import Action
from .tool_result import ToolResult


class JournalError(RuntimeError):
    """Base error raised by Run journal implementations."""


class JournalCorruptionError(JournalError):
    """Raised when a journal cannot be replayed without guessing."""


class JournalRecordType(str, Enum):
    RUN_STARTED = "run.started"
    INPUT_ACCEPTED = "input.accepted"
    MODEL_COMPLETED = "model.completed"
    TOOL_STARTED = "tool.started"
    TOOL_TERMINAL = "tool.terminal"
    STEP_COMMITTED = "step.committed"
    STATE_SNAPSHOT = "state.snapshot"
    RUN_INTERRUPTED = "run.interrupted"
    RUN_COMPLETED = "run.completed"
    RUN_FORKED = "run.forked"
    INHERITED = "journal.inherited"


@dataclass(frozen=True)
class JournalPosition:
    run_id: str
    seq: int
    record_id: str


@dataclass(frozen=True, slots=True)
class JournalRecordRef:
    """Stable origin identity for one canonical Journal record."""

    run_id: str
    record_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("Journal record run_id must be non-empty")
        if not isinstance(self.record_id, str) or not self.record_id:
            raise ValueError("Journal record_id must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {"run_id": self.run_id, "record_id": self.record_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "JournalRecordRef":
        if set(value) != {"run_id", "record_id"}:
            raise ValueError("Journal record reference fields are invalid")
        return cls(run_id=value["run_id"], record_id=value["record_id"])


@dataclass(frozen=True, slots=True)
class ToolTransaction:
    """One committed canonical Tool terminal reconstructed from a Journal."""

    terminal: JournalRecordRef
    committed_at: JournalPosition
    step_id: int
    action_index: int
    action: Action
    result: ToolResult


@dataclass(frozen=True)
class JournalRecord:
    schema_version: int
    seq: int
    record_id: str
    type: JournalRecordType
    run_id: str
    timestamp: str
    payload: dict[str, Any]

    @classmethod
    def create(
        cls,
        *,
        seq: int,
        record_id: str,
        type: JournalRecordType,
        run_id: str,
        payload: Mapping[str, Any],
    ) -> "JournalRecord":
        return cls(
            schema_version=1,
            seq=seq,
            record_id=record_id,
            type=type,
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload=dict(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seq": self.seq,
            "record_id": self.record_id,
            "type": self.type.value,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "JournalRecord":
        required = {
            "schema_version",
            "seq",
            "record_id",
            "type",
            "run_id",
            "timestamp",
            "payload",
        }
        if set(value) != required:
            raise JournalCorruptionError("journal record fields are invalid")
        if value["schema_version"] != 1:
            raise JournalCorruptionError("journal schema version is unsupported")
        seq = value["seq"]
        if isinstance(seq, bool) or not isinstance(seq, int) or seq <= 0:
            raise JournalCorruptionError("journal seq is invalid")
        record_id = value["record_id"]
        run_id = value["run_id"]
        timestamp = value["timestamp"]
        payload = value["payload"]
        if not isinstance(record_id, str) or not record_id:
            raise JournalCorruptionError("journal record_id is invalid")
        if not isinstance(run_id, str) or not run_id:
            raise JournalCorruptionError("journal run_id is invalid")
        if not isinstance(timestamp, str) or not timestamp:
            raise JournalCorruptionError("journal timestamp is invalid")
        if not isinstance(payload, dict):
            raise JournalCorruptionError("journal payload is invalid")
        try:
            record_type = JournalRecordType(str(value["type"]))
        except ValueError as exc:
            raise JournalCorruptionError("journal record type is unsupported") from exc
        return cls(
            schema_version=1,
            seq=seq,
            record_id=record_id,
            type=record_type,
            run_id=run_id,
            timestamp=timestamp,
            payload=dict(payload),
        )

    @property
    def position(self) -> JournalPosition:
        return JournalPosition(self.run_id, self.seq, self.record_id)


class SessionJournal(Protocol):
    """Durable single-Run journal used by the Engine."""

    @property
    def run_id(self) -> str:
        ...

    async def create(
        self, run_id: str, metadata: Mapping[str, Any]
    ) -> JournalPosition:
        ...

    async def open(self, run_id: str) -> None:
        ...

    async def append(
        self,
        record_type: JournalRecordType,
        payload: Mapping[str, Any],
        *,
        record_id: str,
    ) -> JournalPosition:
        ...

    async def replay(self) -> tuple[JournalRecord, ...]:
        ...

    def find_tool_transaction(
        self, reference: JournalRecordRef
    ) -> ToolTransaction | None:
        """Return one committed Tool terminal from the currently open Run view."""

        ...

    async def flush(self) -> None:
        ...

    async def fork(
        self, parent_position: JournalPosition, new_run_id: str
    ) -> "SessionJournal":
        ...


__all__ = [
    "JournalCorruptionError",
    "JournalError",
    "JournalPosition",
    "JournalRecord",
    "JournalRecordRef",
    "JournalRecordType",
    "SessionJournal",
    "ToolTransaction",
]
