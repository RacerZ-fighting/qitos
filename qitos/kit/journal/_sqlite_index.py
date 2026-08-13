"""Disposable SQLite projection for one canonical JSONL Run journal."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ...core.journal import JournalRecord

_INDEX_SCHEMA_VERSION = 2


class JournalIndexError(RuntimeError):
    """Raised when a disposable Journal projection cannot be used."""


@dataclass(frozen=True, slots=True)
class JournalFingerprint:
    """Cheap identity of the canonical file represented by one projection."""

    device: int
    inode: int
    size: int
    mtime_ns: int

    @classmethod
    def from_path(cls, path: Path) -> "JournalFingerprint":
        stat = path.stat()
        return cls(
            device=int(stat.st_dev),
            inode=int(stat.st_ino),
            size=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
        )


class SqliteJournalIndex:
    """Open projection handle updated only after canonical JSONL durability."""

    def __init__(
        self,
        path: Path,
        run_id: str,
        connection: sqlite3.Connection,
    ) -> None:
        self.path = path
        self.run_id = run_id
        self._connection = connection
        self._closed = False

    @classmethod
    def load_if_current(
        cls,
        path: Path,
        journal_path: Path,
        run_id: str,
        records: Iterable[JournalRecord],
    ) -> "SqliteJournalIndex" | None:
        """Open a projection only when it matches validated canonical records."""

        if not path.is_file():
            return None
        canonical_records = list(records)
        connection: sqlite3.Connection | None = None
        try:
            connection = _connect(path)
            if _user_version(connection) != _INDEX_SCHEMA_VERSION:
                connection.close()
                return None
            quick_check = connection.execute("PRAGMA quick_check").fetchone()
            if quick_check != ("ok",):
                raise JournalIndexError("Journal projection integrity check failed")
            row = connection.execute(
                """
                SELECT run_id, device, inode, size, mtime_ns,
                       record_count, last_record_id
                FROM journal_source
                WHERE singleton = 1
                """
            ).fetchone()
            if row is None:
                connection.close()
                return None
            fingerprint = JournalFingerprint.from_path(journal_path)
            expected_source = (
                run_id,
                fingerprint.device,
                fingerprint.inode,
                fingerprint.size,
                fingerprint.mtime_ns,
            )
            if tuple(row[:5]) != expected_source:
                connection.close()
                return None
            record_count = int(row[5])
            last_record_id = str(row[6])
            if len(canonical_records) != record_count:
                raise JournalIndexError("Journal projection record count is invalid")
            if (
                not canonical_records
                or canonical_records[-1].record_id != last_record_id
            ):
                raise JournalIndexError("Journal projection tail identity is invalid")
            if not _matches_canonical_records(connection, canonical_records):
                raise JournalIndexError(
                    "Journal projection content differs from canonical JSONL"
                )
            return cls(path, run_id, connection)
        except (
            OSError,
            sqlite3.Error,
            JournalIndexError,
            ValueError,
        ) as exc:
            if connection is not None:
                connection.close()
            raise JournalIndexError("Journal projection is unreadable") from exc

    @classmethod
    def rebuild(
        cls,
        path: Path,
        journal_path: Path,
        run_id: str,
        records: Iterable[JournalRecord],
    ) -> "SqliteJournalIndex":
        """Atomically replace a projection from validated canonical records."""

        materialized = list(records)
        temporary = path.with_name(f".{path.name}.rebuild")
        connection: sqlite3.Connection | None = None
        try:
            temporary.unlink(missing_ok=True)
            connection = _connect(temporary)
            _initialize_schema(connection)
            with connection:
                connection.executemany(
                    """
                    INSERT INTO journal_record(
                        seq, record_id, record_type, record_sha256
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        (
                            record.seq,
                            record.record_id,
                            record.type.value,
                            _record_sha256(record),
                        )
                        for record in materialized
                    ),
                )
                _replace_source(
                    connection,
                    run_id,
                    JournalFingerprint.from_path(journal_path),
                    materialized,
                )
            connection.close()
            connection = None
            os.replace(temporary, path)
            return cls(path, run_id, _connect(path))
        except (OSError, sqlite3.Error, ValueError) as exc:
            if connection is not None:
                connection.close()
            temporary.unlink(missing_ok=True)
            raise JournalIndexError("Journal projection rebuild failed") from exc

    def append(
        self,
        record: JournalRecord,
        fingerprint: JournalFingerprint,
        record_count: int,
    ) -> None:
        """Project one already-durable canonical record in one SQLite commit."""

        if self._closed:
            raise JournalIndexError("Journal projection is closed")
        try:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO journal_record(
                        seq, record_id, record_type, record_sha256
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        record.seq,
                        record.record_id,
                        record.type.value,
                        _record_sha256(record),
                    ),
                )
                _replace_source(
                    self._connection,
                    self.run_id,
                    fingerprint,
                    [record],
                    record_count=record_count,
                )
        except (sqlite3.Error, ValueError) as exc:
            raise JournalIndexError("Journal projection append failed") from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._connection.close()


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=5.0, check_same_thread=False)
    try:
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = FULL")
    except sqlite3.Error:
        connection.close()
        raise
    return connection


def _initialize_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE journal_source (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            run_id TEXT NOT NULL,
            device INTEGER NOT NULL,
            inode INTEGER NOT NULL,
            size INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            record_count INTEGER NOT NULL,
            last_record_id TEXT NOT NULL
        );
        CREATE TABLE journal_record (
            seq INTEGER PRIMARY KEY,
            record_id TEXT NOT NULL UNIQUE,
            record_type TEXT NOT NULL,
            record_sha256 TEXT NOT NULL
        );
        CREATE INDEX journal_record_type_seq
            ON journal_record(record_type, seq);
        """
    )
    connection.execute(f"PRAGMA user_version = {_INDEX_SCHEMA_VERSION}")


def _replace_source(
    connection: sqlite3.Connection,
    run_id: str,
    fingerprint: JournalFingerprint,
    records: list[JournalRecord],
    *,
    record_count: int | None = None,
) -> None:
    if not records:
        raise ValueError("Journal projection requires at least one record")
    connection.execute("DELETE FROM journal_source")
    connection.execute(
        """
        INSERT INTO journal_source(
            singleton, run_id, device, inode, size, mtime_ns,
            record_count, last_record_id
        ) VALUES (1, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            fingerprint.device,
            fingerprint.inode,
            fingerprint.size,
            fingerprint.mtime_ns,
            len(records) if record_count is None else record_count,
            records[-1].record_id,
        ),
    )


def _matches_canonical_records(
    connection: sqlite3.Connection,
    canonical_records: list[JournalRecord],
) -> bool:
    rows = connection.execute(
        """
        SELECT seq, record_id, record_type, record_sha256
        FROM journal_record
        ORDER BY seq
        """
    ).fetchall()
    if len(rows) != len(canonical_records):
        return False
    for canonical, row in zip(canonical_records, rows):
        stored_seq, record_id, record_type, stored_digest = row
        if (
            stored_seq != canonical.seq
            or record_id != canonical.record_id
            or record_type != canonical.type.value
            or not isinstance(stored_digest, str)
            or stored_digest != _record_sha256(canonical)
        ):
            return False
    return True


def _record_sha256(record: JournalRecord) -> str:
    encoded = json.dumps(
        record.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _user_version(connection: sqlite3.Connection) -> int:
    row = connection.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row else 0


__all__ = [
    "JournalFingerprint",
    "JournalIndexError",
    "SqliteJournalIndex",
]
