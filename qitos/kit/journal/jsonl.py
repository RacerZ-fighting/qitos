"""Strict durable JSONL implementation of the canonical Run journal."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ...core.journal import (
    JournalCorruptionError,
    JournalError,
    JournalPosition,
    JournalRecord,
    JournalRecordType,
)

FileSync = Callable[[int, str], None]
DirectorySync = Callable[[Path], None]


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
        self._lock = asyncio.Lock()

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def path(self) -> Path:
        if self._path is None:
            raise JournalError("journal is not open")
        return self._path

    async def create(self, run_id: str, metadata: Mapping[str, Any]) -> JournalPosition:
        _validate_run_id(run_id)
        async with self._lock:
            if self._path is not None:
                raise JournalError("journal is already open")
            await asyncio.to_thread(self._create_sync, run_id)
            self._run_id = run_id
            self._path = self.root / run_id / "journal.jsonl"
            self._records = []
            return await self._append_locked(
                JournalRecordType.RUN_STARTED,
                dict(metadata),
                record_id=f"{run_id}:start",
            )

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
            if self._path is not None:
                raise JournalError("journal is already open")
            path = self.root / run_id / "journal.jsonl"
            records = await asyncio.to_thread(self._load_and_repair, path, run_id)
            self._run_id = run_id
            self._path = path
            self._records = records

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
        if self._path is None:
            raise JournalError("journal is not open")
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
        await asyncio.to_thread(self._write_sync, self._path, encoded)
        self._records.append(record)
        return record.position

    def _write_sync(self, path: Path, encoded: bytes) -> None:
        with path.open("ab", buffering=0) as stream:
            stream.write(encoded)
            stream.flush()
            self._sync_file(stream.fileno(), _file_sync_mode())

    async def replay(self) -> tuple[JournalRecord, ...]:
        async with self._lock:
            return tuple(self._records)

    async def flush(self) -> None:
        async with self._lock:
            if self._path is None:
                raise JournalError("journal is not open")
            await asyncio.to_thread(self._flush_sync, self._path)

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
            inherited = tuple(self._records[: parent_position.seq])
            await asyncio.to_thread(self._flush_sync, self.path)
        child = type(self)(
            self.root,
            sync_file=self._sync_file,
            sync_directory=self._sync_directory,
        )
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
        return child

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
                if not isinstance(value, dict):
                    raise JournalCorruptionError("journal record must be an object")
                record = JournalRecord.from_dict(value)
            except (UnicodeDecodeError, json.JSONDecodeError, JournalCorruptionError) as exc:
                if is_last and not ends_with_newline:
                    repair_offset = consumed
                    break
                raise JournalCorruptionError(f"journal corruption at line {index}") from exc
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
