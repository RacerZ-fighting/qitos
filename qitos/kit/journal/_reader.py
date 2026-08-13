"""Strict canonical Journal decoding shared by writer recovery and readers."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from ...core.journal import (
    JournalCorruptionError,
    JournalError,
    JournalRecord,
    JournalRecordType,
)

FileSync = Callable[[int, str], None]


def read_journal_records(
    path: Path,
    run_id: str,
    *,
    repair_tail: bool,
    sync_file: FileSync | None = None,
    sync_mode: str = "file",
) -> list[JournalRecord]:
    """Decode one consistent file snapshot and optionally repair its final tail.

    A malformed non-final record always fails closed. A partial final line is omitted
    from the returned snapshot; only the writer-owned recovery path may truncate it or
    append its missing newline.
    """

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
            raise JournalCorruptionError(
                f"journal corruption at line {index}"
            ) from exc
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
            raise JournalCorruptionError(
                f"journal run_id mismatch at line {index}"
            )
        if record.seq != index:
            raise JournalCorruptionError(f"journal seq mismatch at line {index}")
        if record.record_id in record_ids:
            raise JournalCorruptionError(f"duplicate record_id at line {index}")
        records.append(record)
        record_ids.add(record.record_id)
        consumed = line_end
    if repair_tail:
        if sync_file is None:
            raise TypeError("sync_file is required when repair_tail is enabled")
        if repair_offset is not None:
            with path.open("r+b") as stream:
                stream.truncate(repair_offset)
                stream.flush()
                sync_file(stream.fileno(), sync_mode)
        elif not ends_with_newline:
            with path.open("ab", buffering=0) as stream:
                stream.write(b"\n")
                stream.flush()
                sync_file(stream.fileno(), sync_mode)
    if not records or records[0].type is not JournalRecordType.RUN_STARTED:
        raise JournalCorruptionError("journal does not begin with run.started")
    return records


__all__ = ["read_journal_records"]
