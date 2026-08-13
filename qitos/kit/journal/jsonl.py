"""Strict durable JSONL implementation of the canonical Run journal."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import sys
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ...core.journal import (
    JournalClosedError,
    JournalCorruptionError,
    JournalError,
    JournalPosition,
    JournalRecord,
    JournalRecordRef,
    JournalRecordType,
    ToolTransaction,
)
from ...core.action import Action
from ...core.tool_result import ToolResult
from ._sqlite_index import (
    JournalFingerprint,
    JournalIndexError,
    SqliteJournalIndex,
)
from ._writer_lease import JournalWriterLease

FileSync = Callable[[int, str], None]
DirectorySync = Callable[[Path], None]

_logger = logging.getLogger(__name__)
_INDEX_FILENAME = "journal.index.sqlite3"


class JsonlSessionJournal:
    """One durable, single-writer JSONL journal per Run."""

    def __init__(
        self,
        root: str | Path,
        *,
        sync_file: FileSync | None = None,
        sync_directory: DirectorySync | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self._sync_file = sync_file or _sync_file
        self._sync_directory = sync_directory or _sync_directory
        self._run_id = ""
        self._path: Path | None = None
        self._records: list[JournalRecord] = []
        self._closed = False
        self._writer_lease: JournalWriterLease | None = None
        self._sqlite_index: SqliteJournalIndex | None = None
        self._lock = asyncio.Lock()
        self._query_lock = threading.RLock()
        self._terminal_records: dict[JournalRecordRef, JournalRecord] = {}
        self._terminal_commits: dict[JournalRecordRef, JournalRecord] = {}

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def path(self) -> Path:
        if self._path is None:
            raise JournalError("journal is not open")
        return self._path

    @property
    def index_path(self) -> Path:
        """Return the disposable projection path for the current Run."""

        return self.path.parent / _INDEX_FILENAME

    @property
    def closed(self) -> bool:
        return self._closed

    async def __aenter__(self) -> "JsonlSessionJournal":
        if self._closed:
            raise JournalClosedError("journal is closed")
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def create(self, run_id: str, metadata: Mapping[str, Any]) -> JournalPosition:
        _validate_run_id(run_id)
        async with self._lock:
            self._ensure_can_open()
            await asyncio.to_thread(self._create_sync, run_id)
            self._run_id = run_id
            self._path = self.root / run_id / "journal.jsonl"
            self._records = []
            self._rebuild_tool_transaction_index(())
            try:
                self._writer_lease = await asyncio.to_thread(
                    JournalWriterLease.acquire,
                    self._path.parent,
                    run_id,
                )
                position = await self._append_locked(
                    JournalRecordType.RUN_STARTED,
                    metadata,
                    record_id=f"{run_id}:start",
                )
                await self._rebuild_sqlite_index_locked()
                return position
            except BaseException:
                await self._close_locked(flush=False)
                raise

    def _create_sync(self, run_id: str) -> None:
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        path = run_dir / "journal.jsonl"
        with path.open("xb"):
            pass
        self._sync_directory(run_dir)
        self._sync_directory(run_dir.parent)

    async def open(self, run_id: str) -> None:
        _validate_run_id(run_id)
        async with self._lock:
            self._ensure_can_open()
            path = self.root / run_id / "journal.jsonl"
            self._run_id = run_id
            self._path = path
            try:
                self._writer_lease = await asyncio.to_thread(
                    JournalWriterLease.acquire,
                    path.parent,
                    run_id,
                )
                records, sqlite_index = await asyncio.to_thread(
                    self._load_records,
                    path,
                    run_id,
                )
                self._records = records
                self._sqlite_index = sqlite_index
                self._rebuild_tool_transaction_index(records)
            except BaseException:
                await self._close_locked(flush=False)
                raise

    async def append(
        self,
        record_type: JournalRecordType,
        payload: Mapping[str, Any],
        *,
        record_id: str,
    ) -> JournalPosition:
        async with self._lock:
            return await self._append_locked(record_type, payload, record_id=record_id)

    async def _append_locked(
        self,
        record_type: JournalRecordType,
        payload: Mapping[str, Any],
        *,
        record_id: str,
    ) -> JournalPosition:
        path = self._require_open_path()
        if not isinstance(record_type, JournalRecordType):
            raise TypeError("record_type must be a JournalRecordType")
        for existing in self._records:
            if existing.record_id != record_id:
                continue
            candidate = JournalRecord.create(
                seq=existing.seq,
                record_id=record_id,
                type=record_type,
                run_id=self._run_id,
                payload=payload,
            )
            if existing.type != candidate.type or existing.payload != candidate.payload:
                raise JournalError("record_id was reused with different content")
            return existing.position
        record = JournalRecord.create(
            seq=len(self._records) + 1,
            record_id=record_id,
            type=record_type,
            run_id=self._run_id,
            payload=payload,
        )
        encoded = _encode_record(record)
        await asyncio.to_thread(self._write_sync, path, encoded)
        self._records.append(record)
        self._index_tool_transaction_record(record)
        await self._project_append_locked(record, path)
        return record.position

    def _write_sync(self, path: Path, encoded: bytes) -> None:
        with path.open("ab", buffering=0) as stream:
            stream.write(encoded)
            stream.flush()
            self._sync_file(stream.fileno(), _file_sync_mode())

    async def replay(self) -> tuple[JournalRecord, ...]:
        async with self._lock:
            self._require_records_available()
            return tuple(_clone_record(record) for record in self._records)

    def find_tool_transaction(
        self,
        reference: JournalRecordRef,
    ) -> ToolTransaction | None:
        """Return a fresh typed view of one committed Tool terminal."""

        if not isinstance(reference, JournalRecordRef):
            raise TypeError("reference must be a JournalRecordRef")
        self._require_records_available()
        with self._query_lock:
            terminal = self._terminal_records.get(reference)
            committed = self._terminal_commits.get(reference)
            if terminal is None or committed is None:
                return None
            return _tool_transaction(terminal, committed)

    def _rebuild_tool_transaction_index(
        self,
        records: tuple[JournalRecord, ...] | list[JournalRecord],
    ) -> None:
        with self._query_lock:
            self._terminal_records = {}
            self._terminal_commits = {}
            for record in records:
                self._index_tool_transaction_record_locked(record)

    def _index_tool_transaction_record(self, record: JournalRecord) -> None:
        with self._query_lock:
            self._index_tool_transaction_record_locked(record)

    def _index_tool_transaction_record_locked(self, record: JournalRecord) -> None:
        effective = _origin_record(record)
        if effective.type is JournalRecordType.TOOL_TERMINAL:
            reference = JournalRecordRef(effective.run_id, effective.record_id)
            existing = self._terminal_records.get(reference)
            if existing is not None and existing.to_dict() != effective.to_dict():
                raise JournalCorruptionError(
                    "conflicting inherited Tool terminal reference"
                )
            self._terminal_records[reference] = effective
            return
        if effective.type is not JournalRecordType.STEP_COMMITTED:
            return
        raw_terminal_ids = effective.payload.get("terminal_record_ids", [])
        if not isinstance(raw_terminal_ids, list) or any(
            not isinstance(record_id, str) or not record_id
            for record_id in raw_terminal_ids
        ):
            raise JournalCorruptionError(
                "step.committed terminal_record_ids are invalid"
            )
        for record_id in raw_terminal_ids:
            reference = JournalRecordRef(effective.run_id, record_id)
            if reference in self._terminal_records:
                self._terminal_commits[reference] = effective

    async def flush(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._flush_sync, self._require_open_path())

    async def close(self) -> None:
        """Flush the canonical file and permanently release writer ownership."""

        async with self._lock:
            await self._close_locked(flush=True)

    def _flush_sync(self, path: Path) -> None:
        with path.open("rb") as stream:
            self._sync_file(stream.fileno(), _file_sync_mode())

    async def fork(
        self,
        parent_position: JournalPosition,
        new_run_id: str,
    ) -> "JsonlSessionJournal":
        async with self._lock:
            if parent_position.run_id != self._run_id:
                raise ValueError("fork position belongs to another Run")
            if not 1 <= parent_position.seq <= len(self._records):
                raise ValueError("fork position is unavailable")
            source = self._records[parent_position.seq - 1]
            if source.record_id != parent_position.record_id:
                raise ValueError("fork position does not match the journal")
            if source.type not in {
                JournalRecordType.STEP_COMMITTED,
                JournalRecordType.STATE_SNAPSHOT,
            }:
                raise ValueError("fork position is not a committed boundary")
            inherited = tuple(
                _clone_record(record)
                for record in self._records[: parent_position.seq]
            )
            await asyncio.to_thread(self._flush_sync, self.path)
        child = type(self)(
            self.root,
            sync_file=self._sync_file,
            sync_directory=self._sync_directory,
        )
        try:
            await child.create(new_run_id, {"forked_from": self._run_id})
            await child.append(
                JournalRecordType.RUN_FORKED,
                {
                    "parent_run_id": self._run_id,
                    "parent_seq": parent_position.seq,
                    "parent_record_id": parent_position.record_id,
                },
                record_id=f"{new_run_id}:fork",
            )
            for record in inherited:
                await child.append(
                    JournalRecordType.INHERITED,
                    {
                        "origin_run_id": record.run_id,
                        "origin_seq": record.seq,
                        "origin_record_id": record.record_id,
                        "record": record.to_dict(),
                    },
                    record_id=f"{new_run_id}:inherited:{record.record_id}",
                )
        except BaseException:
            await child.close()
            raise
        return child

    def _ensure_can_open(self) -> None:
        if self._closed:
            raise JournalClosedError("journal is closed")
        if self._path is not None:
            raise JournalError("journal is already open")

    def _require_open_path(self) -> Path:
        if self._closed:
            raise JournalClosedError("journal is closed")
        if self._path is None or self._writer_lease is None:
            raise JournalError("journal is not open")
        return self._path

    def _require_records_available(self) -> None:
        if self._path is None:
            raise JournalError("journal is not open")

    async def _close_locked(self, *, flush: bool) -> None:
        if self._closed:
            return
        path = self._path
        failure: BaseException | None = None
        try:
            if flush and path is not None and self._writer_lease is not None:
                await asyncio.to_thread(self._flush_sync, path)
        except BaseException as exc:
            failure = exc
        sqlite_index = self._sqlite_index
        writer_lease = self._writer_lease
        self._sqlite_index = None
        self._writer_lease = None
        self._closed = True
        if sqlite_index is not None:
            try:
                await asyncio.to_thread(sqlite_index.close)
            except BaseException as exc:
                failure = self._retain_close_failure(failure, exc)
        if writer_lease is not None:
            try:
                await asyncio.to_thread(writer_lease.release)
            except BaseException as exc:
                failure = self._retain_close_failure(failure, exc)
        if failure is not None:
            raise failure

    def _retain_close_failure(
        self,
        existing: BaseException | None,
        later: BaseException,
    ) -> BaseException:
        if existing is None:
            return later
        _logger.warning(
            "Additional Journal close failure for Run %s: %s",
            self._run_id,
            later,
        )
        return existing

    def _load_records(
        self,
        path: Path,
        run_id: str,
    ) -> tuple[list[JournalRecord], SqliteJournalIndex | None]:
        index_path = path.parent / _INDEX_FILENAME
        records = self._load_and_repair(path, run_id)
        try:
            sqlite_index = SqliteJournalIndex.load_if_current(
                index_path,
                path,
                run_id,
                records,
            )
        except JournalIndexError as exc:
            _logger.warning(
                "Discarding unreadable Journal projection for Run %s: %s",
                run_id,
                exc,
            )
            sqlite_index = None
        if sqlite_index is not None:
            return records, sqlite_index
        try:
            sqlite_index = SqliteJournalIndex.rebuild(
                index_path,
                path,
                run_id,
                records,
            )
        except JournalIndexError as exc:
            _logger.warning(
                "Journal projection unavailable for Run %s: %s",
                run_id,
                exc,
            )
            sqlite_index = None
        return records, sqlite_index

    async def _rebuild_sqlite_index_locked(self) -> None:
        path = self._require_open_path()
        try:
            self._sqlite_index = await asyncio.to_thread(
                SqliteJournalIndex.rebuild,
                path.parent / _INDEX_FILENAME,
                path,
                self._run_id,
                self._records,
            )
        except JournalIndexError as exc:
            self._sqlite_index = None
            _logger.warning(
                "Journal projection unavailable for Run %s: %s",
                self._run_id,
                exc,
            )

    async def _project_append_locked(
        self,
        record: JournalRecord,
        path: Path,
    ) -> None:
        sqlite_index = self._sqlite_index
        if sqlite_index is None:
            return
        try:
            fingerprint = await asyncio.to_thread(JournalFingerprint.from_path, path)
            await asyncio.to_thread(
                sqlite_index.append,
                record,
                fingerprint,
                len(self._records),
            )
        except JournalIndexError as exc:
            _logger.warning(
                "Journal projection update failed for Run %s: %s",
                self._run_id,
                exc,
            )
            await asyncio.to_thread(sqlite_index.close)
            self._sqlite_index = None

    def _load_and_repair(self, path: Path, run_id: str) -> list[JournalRecord]:
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise JournalError(f"failed to read journal for {run_id}") from exc
        if not raw:
            raise JournalCorruptionError("journal is empty")
        ends_with_newline = raw.endswith(b"\n")
        chunks = raw.splitlines()
        records: list[JournalRecord] = []
        record_ids: set[str] = set()
        repair_offset: int | None = None
        consumed = 0
        for index, chunk in enumerate(chunks, start=1):
            line_end = consumed + len(chunk) + 1
            is_last = index == len(chunks)
            try:
                value = json.loads(chunk)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if is_last and not ends_with_newline:
                    repair_offset = consumed
                    break
                raise JournalCorruptionError(f"journal corruption at line {index}") from exc
            if not isinstance(value, dict):
                raise JournalCorruptionError(
                    f"journal corruption at line {index}: record must be an object"
                )
            try:
                record = JournalRecord.from_dict(value)
            except JournalCorruptionError as exc:
                raise JournalCorruptionError(
                    f"journal corruption at line {index}: {exc}"
                ) from exc
            if record.run_id != run_id:
                raise JournalCorruptionError(f"journal run_id mismatch at line {index}")
            if record.seq != index:
                raise JournalCorruptionError(f"journal seq mismatch at line {index}")
            if record.record_id in record_ids:
                raise JournalCorruptionError(f"duplicate record_id at line {index}")
            records.append(record)
            record_ids.add(record.record_id)
            consumed = line_end
        if repair_offset is not None:
            with path.open("r+b") as stream:
                stream.truncate(repair_offset)
                stream.flush()
                self._sync_file(stream.fileno(), _file_sync_mode())
        elif not ends_with_newline:
            with path.open("ab", buffering=0) as stream:
                stream.write(b"\n")
                stream.flush()
                self._sync_file(stream.fileno(), _file_sync_mode())
        if not records or records[0].type is not JournalRecordType.RUN_STARTED:
            raise JournalCorruptionError("journal does not begin with run.started")
        return records


def _encode_record(record: JournalRecord) -> bytes:
    try:
        encoded = json.dumps(
            record.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise JournalError("journal payload is not JSON serializable") from exc
    return encoded.encode("utf-8") + b"\n"


def _clone_record(record: JournalRecord) -> JournalRecord:
    return JournalRecord.from_dict(record.to_dict())


def _origin_record(record: JournalRecord) -> JournalRecord:
    current = record
    for _ in range(64):
        if current.type is not JournalRecordType.INHERITED:
            return current
        raw_record = current.payload.get("record")
        if not isinstance(raw_record, Mapping):
            raise JournalCorruptionError(
                "journal.inherited is missing its origin record"
            )
        current = JournalRecord.from_dict(raw_record)
    raise JournalCorruptionError("journal.inherited nesting is too deep")


def _tool_transaction(
    terminal: JournalRecord,
    committed: JournalRecord,
) -> ToolTransaction:
    payload = terminal.payload
    step_id = payload.get("step_id")
    action_index = payload.get("action_index")
    raw_action = payload.get("action")
    raw_result = payload.get("result")
    if (
        isinstance(step_id, bool)
        or not isinstance(step_id, int)
        or step_id < 0
        or isinstance(action_index, bool)
        or not isinstance(action_index, int)
        or action_index < 0
        or not isinstance(raw_action, Mapping)
        or not isinstance(raw_result, Mapping)
    ):
        raise JournalCorruptionError("tool.terminal payload is invalid")
    try:
        action = Action.from_dict(copy.deepcopy(dict(raw_action)))
        result = ToolResult.from_value(copy.deepcopy(dict(raw_result)))
    except (TypeError, ValueError) as exc:
        raise JournalCorruptionError("tool.terminal payload is invalid") from exc
    return ToolTransaction(
        terminal=JournalRecordRef(terminal.run_id, terminal.record_id),
        committed_at=committed.position,
        step_id=step_id,
        action_index=action_index,
        action=action,
        result=result,
    )


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be non-empty")
    if "/" in run_id or "\\" in run_id or "\x00" in run_id:
        raise ValueError("run_id contains a path separator")


def _file_sync_mode() -> str:
    if sys.platform == "darwin":
        return "full"
    return "data" if hasattr(os, "fdatasync") else "file"


def _sync_file(descriptor: int, mode: str) -> None:
    if mode == "full":
        try:
            import fcntl

            command = getattr(fcntl, "F_FULLFSYNC")
        except (AttributeError, ImportError):
            os.fsync(descriptor)
        else:
            fcntl.fcntl(descriptor, command)
        return
    if mode == "data" and hasattr(os, "fdatasync"):
        os.fdatasync(descriptor)
        return
    os.fsync(descriptor)


def _sync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["JsonlSessionJournal"]
