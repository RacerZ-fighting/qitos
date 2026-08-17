"""Canonical append-only Run journal contracts."""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable

from .message import ToolCall
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


class JournalCommitState(str, Enum):
    """Durable outcome of one canonical append attempt."""

    NOT_COMMITTED = "not_committed"
    COMMITTED = "committed"
    UNKNOWN = "unknown"


class JournalRecordType(str, Enum):
    RUN_STARTED = "run.started"
    INPUT_ACCEPTED = "input.accepted"
    MODEL_COMPLETED = "model.completed"
    BUDGET_COMMITTED = "budget.committed"
    TOOL_STARTED = "tool.started"
    TOOL_TERMINAL = "tool.terminal"
    PROCESS_STARTED = "process.started"
    PROCESS_TERMINAL = "process.terminal"
    CHILD_STARTED = "child.started"
    CHILD_TERMINAL = "child.terminal"
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


class JournalCommitError(JournalError):
    """Canonical append failure with an explicit durable outcome."""

    def __init__(
        self,
        position: JournalPosition,
        commit_state: JournalCommitState,
        *,
        cause: BaseException,
        rollback_error: BaseException | None = None,
    ) -> None:
        if not isinstance(position, JournalPosition):
            raise TypeError("position must be a JournalPosition")
        if not isinstance(commit_state, JournalCommitState):
            raise TypeError("commit_state must be a JournalCommitState")
        self.position = position
        self.commit_state = commit_state
        self.cause = cause
        self.rollback_error = rollback_error
        if commit_state is JournalCommitState.UNKNOWN:
            message = (
                "journal append outcome is unknown; close and reopen before "
                "continuing"
            )
        elif commit_state is JournalCommitState.COMMITTED:
            message = "journal append committed but local finalization failed"
        else:
            message = "journal append failed and was rolled back"
        super().__init__(message)


class JournalAppendCancelled(asyncio.CancelledError):
    """Cancellation observed after a Journal append had started settling.

    ``committed_position`` is present only when the canonical record was durably
    appended before cancellation propagated. Callers that reserve external state
    for the record can then commit instead of guessing from the filesystem.
    """

    def __init__(
        self,
        committed_position: JournalPosition | None,
        *,
        commit_state: JournalCommitState | None = None,
        pending_position: JournalPosition | None = None,
        commit_error: BaseException | None = None,
    ) -> None:
        resolved_state = commit_state or (
            JournalCommitState.COMMITTED
            if committed_position is not None
            else JournalCommitState.NOT_COMMITTED
        )
        if (
            committed_position is not None
            and resolved_state is not JournalCommitState.COMMITTED
        ):
            raise ValueError("committed_position requires committed commit_state")
        if (
            committed_position is None
            and resolved_state is JournalCommitState.COMMITTED
        ):
            raise ValueError("committed commit_state requires committed_position")
        self.committed_position = committed_position
        self.commit_state = resolved_state
        self.pending_position = pending_position or committed_position
        self.commit_error = commit_error
        if resolved_state is JournalCommitState.COMMITTED:
            message = (
                "journal append was cancelled after its canonical record committed"
            )
        elif resolved_state is JournalCommitState.UNKNOWN:
            message = "journal append was cancelled with an unknown durable outcome"
        else:
            message = (
                "journal append was cancelled before its canonical record committed"
            )
        super().__init__(message)


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
    """One committed canonical Tool terminal reconstructed from a Journal.

    Only the minimal agent loop's schema is decodable: records carry ``turn``
    and ``call`` (a ``ToolCall``). Records written by the retired Engine path
    (``step_id`` / ``action_index`` / ``action``) fail closed at decode time.
    """

    terminal: JournalRecordRef
    committed_at: JournalPosition
    action: ToolCall
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


def resolve_inherited_record(record: JournalRecord) -> JournalRecord:
    """Resolve an inherited record to its canonical origin.

    Forks embed their complete parent prefix, so a fork of a fork may contain
    multiple wrappers. Each wrapper must describe the record it embeds exactly;
    recovery fails closed instead of accepting an ambiguous lineage.
    """

    current = record
    for _ in range(64):
        if current.type is not JournalRecordType.INHERITED:
            return current
        raw_record = current.payload.get("record")
        if not isinstance(raw_record, Mapping):
            raise JournalCorruptionError(
                "journal.inherited is missing its origin record"
            )
        embedded = JournalRecord.from_dict(raw_record)
        if (
            current.payload.get("origin_run_id") != embedded.run_id
            or current.payload.get("origin_seq") != embedded.seq
            or current.payload.get("origin_record_id") != embedded.record_id
        ):
            raise JournalCorruptionError(
                "journal.inherited origin identity does not match its record"
            )
        current = embedded
    raise JournalCorruptionError("journal.inherited nesting is too deep")


@runtime_checkable
class SessionJournal(Protocol):
    """Durable single-Run journal used by the Agent Session boundary.

    An append cancelled before admission raises ordinary ``CancelledError`` and
    commits nothing. Once canonical I/O starts, implementations settle it before
    propagating :class:`JournalAppendCancelled`; its ``commit_state`` distinguishes
    a committed, rolled-back, or unknown durable outcome. An unknown outcome makes
    that journal instance unusable until it is closed and replayed.
    """

    @property
    def run_id(self) -> str:
        raise NotImplementedError

    async def create(self, run_id: str, metadata: Mapping[str, Any]) -> JournalPosition:
        raise NotImplementedError

    async def open(self, run_id: str) -> None:
        raise NotImplementedError

    async def append(
        self,
        record_type: JournalRecordType,
        payload: Mapping[str, Any],
        *,
        record_id: str,
    ) -> JournalPosition:
        raise NotImplementedError

    async def replay(self) -> tuple[JournalRecord, ...]:
        raise NotImplementedError

    def find_tool_transaction(
        self, reference: JournalRecordRef
    ) -> ToolTransaction | None:
        """Return one committed Tool terminal from the currently open Run view."""

        ...

    async def flush(self) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        """Flush pending writes and permanently release this journal instance."""

        ...

    async def fork(
        self, parent_position: JournalPosition, new_run_id: str
    ) -> "SessionJournal":
        raise NotImplementedError


__all__ = [
    "JournalCorruptionError",
    "JournalClosedError",
    "JournalAppendCancelled",
    "JournalCommitError",
    "JournalCommitState",
    "JournalError",
    "JournalOwnershipError",
    "JournalPosition",
    "JournalRecord",
    "JournalRecordRef",
    "JournalRecordType",
    "JournalUnsupportedVersionError",
    "SessionJournal",
    "ToolTransaction",
    "resolve_inherited_record",
]
