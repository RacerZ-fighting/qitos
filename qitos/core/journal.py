"""Canonical append-only Run journal contracts."""

from __future__ import annotations

import copy
import json
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


class JournalUnsupportedVersionError(JournalError):
    """Raised when a valid journal uses an unsupported schema version."""

    def __init__(self, found_version: int, supported_version: int) -> None:
        self.found_version = found_version
        self.supported_version = supported_version
        super().__init__(
            "journal schema version is unsupported: "
            f"found {found_version}; this QitOS version supports {supported_version}. "
            "Upgrade QitOS or migrate the journal before replay."
        )


class JournalOwnershipError(JournalError):
    """Raised when another writer already owns the requested Run journal."""


class JournalClosedError(JournalError):
    """Raised when an operation requires a journal whose lifecycle has ended."""


class JournalRecordType(str, Enum):
    RUN_STARTED = "run.started"
    INPUT_ACCEPTED = "input.accepted"
    MODEL_COMPLETED = "model.completed"
    TOOL_STARTED = "tool.started"
    TOOL_TERMINAL = "tool.terminal"
    PROCESS_STARTED = "process.started"
    PROCESS_TERMINAL = "process.terminal"
    RUNTIME_INPUT_POSTED = "runtime_input.posted"
    STEP_COMMITTED = "step.committed"
    STATE_SNAPSHOT = "state.snapshot"
    RUN_INTERRUPTED = "run.interrupted"
    RUN_COMPLETED = "run.completed"
    RUN_FORKED = "run.forked"
    INHERITED = "journal.inherited"


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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
            payload=_normalize_json_payload(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seq": self.seq,
            "record_id": self.record_id,
            "type": self.type.value,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "payload": copy.deepcopy(self.payload),
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
        schema_version = value["schema_version"]
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise JournalCorruptionError("journal schema version is invalid")
        if schema_version != 1:
            raise JournalUnsupportedVersionError(schema_version, 1)
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
        try:
            canonical_payload = _normalize_json_payload(payload)
        except JournalError as exc:
            raise JournalCorruptionError("journal payload is invalid") from exc
        if not _has_canonical_json_shape(payload):
            raise JournalCorruptionError("journal payload is not canonical JSON")
        return cls(
            schema_version=1,
            seq=seq,
            record_id=record_id,
            type=record_type,
            run_id=run_id,
            timestamp=timestamp,
            payload=canonical_payload,
        )

    @property
    def position(self) -> JournalPosition:
        return JournalPosition(self.run_id, self.seq, self.record_id)


def _normalize_json_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return one isolated JSON representation used by every Journal boundary."""

    if not isinstance(payload, Mapping):
        raise JournalError("journal payload must be a strict JSON value")
    try:
        encoded = json.dumps(
            dict(payload),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        normalized = json.loads(encoded)
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise JournalError("journal payload must be a strict JSON value") from exc
    if not isinstance(normalized, dict):
        raise JournalError("journal payload must be a strict JSON value")
    if not _has_only_string_mapping_keys(payload):
        raise JournalError("journal payload must be a strict JSON value")
    return normalized


def _has_only_string_mapping_keys(value: Any) -> bool:
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _has_only_string_mapping_keys(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(_has_only_string_mapping_keys(item) for item in value)
    return True


def _has_canonical_json_shape(value: Any) -> bool:
    if value is None or type(value) in {bool, int, float, str}:
        return True
    if type(value) is list:
        return all(_has_canonical_json_shape(item) for item in value)
    if type(value) is dict:
        return all(
            type(key) is str and _has_canonical_json_shape(item)
            for key, item in value.items()
        )
    return False


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

    async def close(self) -> None:
        """Flush pending writes and permanently release this journal instance."""

        ...

    async def fork(
        self, parent_position: JournalPosition, new_run_id: str
    ) -> "SessionJournal":
        ...


__all__ = [
    "JournalCorruptionError",
    "JournalClosedError",
    "JournalError",
    "JournalOwnershipError",
    "JournalPosition",
    "JournalRecord",
    "JournalRecordRef",
    "JournalRecordType",
    "JournalUnsupportedVersionError",
    "SessionJournal",
    "ToolTransaction",
]
