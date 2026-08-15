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
from typing import Any, TypeVar

from ...core.journal import (
    JournalAppendCancelled,
    JournalClosedError,
    JournalCommitError,
    JournalCommitState,
    JournalCorruptionError,
    JournalError,
    JournalOwnershipError,
    JournalPosition,
    JournalRecord,
    JournalRecordRef,
    JournalRecordType,
    ToolTransaction,
    resolve_inherited_record,
)
from ...core.action import Action
from ...core.tool_result import ToolResult
from ._sqlite_index import (
    JournalFingerprint,
    JournalIndexError,
    SqliteJournalIndex,
)
from ._paths import INDEX_FILENAME, JOURNAL_FILENAME, journal_path, validate_run_id
from ._reader import read_journal_records
from ._writer_lease import JournalWriterLease

FileSync = Callable[[int, str], None]
DirectorySync = Callable[[Path], None]

_logger = logging.getLogger(__name__)
_T = TypeVar("_T")


class _RunCreationError(JournalError):
    def __init__(self, cause: BaseException, *, created: bool) -> None:
        self.cause = cause
        self.created = created
        super().__init__("failed to create Journal Run files")


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
        self._poisoned_error: JournalCommitError | None = None
        self._owns_unstarted_run = False

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

        return self.path.parent / INDEX_FILENAME

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
        validate_run_id(run_id)
        async with self._lock:
            self._ensure_can_open()
            self._run_id = run_id
            self._path = journal_path(self.root, run_id)
            self._records = []
            self._rebuild_tool_transaction_index(())
            try:
                await self._create_run_files_settled(run_id)
                await self._acquire_writer_lease_settled(
                    self._path.parent,
                    run_id,
                )
                position = await self._append_settled(
                    JournalRecordType.RUN_STARTED,
                    metadata,
                    record_id=f"{run_id}:start",
                )
                self._owns_unstarted_run = False
                await self._rebuild_sqlite_index_locked()
                return position
            except BaseException as error:
                remove_unstarted = (
                    self._owns_unstarted_run
                    and not self._records
                    and self._poisoned_error is None
                    and not isinstance(error, JournalOwnershipError)
                )
                await self._abort_open_locked(remove_unstarted=remove_unstarted)
                raise

    async def _create_run_files_settled(self, run_id: str) -> None:
        create = asyncio.create_task(
            asyncio.to_thread(self._create_sync, run_id),
            name=f"qitos-journal-create-{run_id}",
        )
        try:
            await asyncio.shield(create)
        except asyncio.CancelledError as cancellation:
            try:
                await _settle_task(create)
            except _RunCreationError as create_error:
                self._owns_unstarted_run = create_error.created
                raise cancellation from create_error.cause
            except BaseException as create_error:
                raise cancellation from create_error
            self._owns_unstarted_run = True
            raise
        except _RunCreationError as create_error:
            self._owns_unstarted_run = create_error.created
            raise create_error.cause from create_error
        self._owns_unstarted_run = True

    def _create_sync(self, run_id: str) -> None:
        run_dir = self.root / run_id
        created = False
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            created = True
            path = run_dir / JOURNAL_FILENAME
            with path.open("xb"):
                pass
            self._sync_directory(run_dir)
            self._sync_directory(run_dir.parent)
        except BaseException as exc:
            raise _RunCreationError(exc, created=created) from exc

    async def open(self, run_id: str) -> None:
        validate_run_id(run_id)
        async with self._lock:
            self._ensure_can_open()
            path = journal_path(self.root, run_id)
            self._run_id = run_id
            self._path = path
            try:
                await self._acquire_writer_lease_settled(
                    path.parent,
                    run_id,
                )
                records, sqlite_index = await self._load_records_settled(
                    path,
                    run_id,
                )
                self._records = records
                self._sqlite_index = sqlite_index
                self._rebuild_tool_transaction_index(records)
            except BaseException:
                await self._abort_open_locked(remove_unstarted=False)
                raise

    async def _acquire_writer_lease_settled(
        self,
        run_directory: Path,
        run_id: str,
    ) -> None:
        acquire = asyncio.create_task(
            asyncio.to_thread(
                JournalWriterLease.acquire,
                run_directory,
                run_id,
            ),
            name=f"qitos-journal-lease-{run_id}",
        )
        try:
            lease = await asyncio.shield(acquire)
        except asyncio.CancelledError as cancellation:
            try:
                lease = await _settle_task(acquire)
            except BaseException as acquire_error:
                raise cancellation from acquire_error
            self._writer_lease = lease
            raise
        self._writer_lease = lease

    async def _load_records_settled(
        self,
        path: Path,
        run_id: str,
    ) -> tuple[list[JournalRecord], SqliteJournalIndex | None]:
        load = asyncio.create_task(
            asyncio.to_thread(self._load_records, path, run_id),
            name=f"qitos-journal-load-{run_id}",
        )
        try:
            return await asyncio.shield(load)
        except asyncio.CancelledError as cancellation:
            try:
                records, sqlite_index = await _settle_task(load)
            except BaseException as load_error:
                raise cancellation from load_error
            self._records = records
            self._sqlite_index = sqlite_index
            raise

    async def append(
        self,
        record_type: JournalRecordType,
        payload: Mapping[str, Any],
        *,
        record_id: str,
    ) -> JournalPosition:
        async with self._lock:
            return await self._append_settled(
                record_type,
                payload,
                record_id=record_id,
            )

    async def _append_settled(
        self,
        record_type: JournalRecordType,
        payload: Mapping[str, Any],
        *,
        record_id: str,
    ) -> JournalPosition:
        commit = asyncio.create_task(
            self._append_locked(record_type, payload, record_id=record_id),
            name=f"qitos-journal-append-{record_id}",
        )
        try:
            return await asyncio.shield(commit)
        except asyncio.CancelledError as cancellation:
            try:
                position = await _settle_task(commit)
            except JournalCommitError as commit_error:
                committed_position = (
                    commit_error.position
                    if commit_error.commit_state is JournalCommitState.COMMITTED
                    else None
                )
                raise JournalAppendCancelled(
                    committed_position,
                    commit_state=commit_error.commit_state,
                    pending_position=commit_error.position,
                    commit_error=commit_error,
                ) from cancellation
            except BaseException as commit_error:
                raise JournalAppendCancelled(
                    None,
                    commit_error=commit_error,
                ) from cancellation
            raise JournalAppendCancelled(position) from cancellation

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
        try:
            await asyncio.to_thread(self._write_sync, path, encoded, record.position)
        except JournalCommitError as exc:
            if exc.commit_state is JournalCommitState.UNKNOWN:
                self._poisoned_error = exc
            raise
        self._records.append(record)
        self._index_tool_transaction_record(record)
        await self._project_append_locked(record, path)
        return record.position

    def _write_sync(
        self,
        path: Path,
        encoded: bytes,
        position: JournalPosition,
    ) -> None:
        try:
            stream = path.open("ab", buffering=0)
        except BaseException as exc:
            raise JournalCommitError(
                position,
                JournalCommitState.NOT_COMMITTED,
                cause=exc,
            ) from exc

        with stream:
            try:
                original_size = os.fstat(stream.fileno()).st_size
            except BaseException as exc:
                raise JournalCommitError(
                    position,
                    JournalCommitState.NOT_COMMITTED,
                    cause=exc,
                ) from exc

            write_started = False
            try:
                write_started = True
                written = stream.write(encoded)
                if written != len(encoded):
                    raise OSError(
                        "journal append wrote " f"{written!r} of {len(encoded)} bytes"
                    )
                stream.flush()
                self._sync_file(stream.fileno(), _file_sync_mode())
            except BaseException as exc:
                rollback_error: BaseException | None = None
                if write_started:
                    try:
                        os.ftruncate(stream.fileno(), original_size)
                        stream.flush()
                        self._sync_file(stream.fileno(), _file_sync_mode())
                    except BaseException as rollback_exc:
                        rollback_error = rollback_exc
                commit_state = (
                    JournalCommitState.NOT_COMMITTED
                    if rollback_error is None
                    else JournalCommitState.UNKNOWN
                )
                raise JournalCommitError(
                    position,
                    commit_state,
                    cause=exc,
                    rollback_error=rollback_error,
                ) from exc

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
        effective = resolve_inherited_record(record)
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
            await _to_thread_settled(self._flush_sync, self._require_open_path())

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
            if resolve_inherited_record(source).type not in {
                JournalRecordType.STEP_COMMITTED,
                JournalRecordType.STATE_SNAPSHOT,
            }:
                raise ValueError("fork position is not a committed boundary")
            inherited = tuple(
                _clone_record(record) for record in self._records[: parent_position.seq]
            )
            await _to_thread_settled(self._flush_sync, self.path)
        child = type(self)(
            self.root,
            sync_file=self._sync_file,
            sync_directory=self._sync_directory,
        )
        try:
            child_metadata = copy.deepcopy(self._records[0].payload)
            child_metadata["forked_from"] = self._run_id
            await child.create(new_run_id, child_metadata)
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
        self._raise_if_poisoned()
        if self._path is None or self._writer_lease is None:
            raise JournalError("journal is not open")
        return self._path

    def _require_records_available(self) -> None:
        self._raise_if_poisoned()
        if self._path is None:
            raise JournalError("journal is not open")

    def _raise_if_poisoned(self) -> None:
        if self._poisoned_error is not None:
            raise JournalError(
                "journal append outcome is unknown; close and reopen before continuing"
            ) from self._poisoned_error

    async def _abort_open_locked(self, *, remove_unstarted: bool) -> None:
        try:
            await self._close_locked(flush=False)
        except BaseException as cleanup_error:
            _logger.warning(
                "Journal cleanup after open failure also failed for Run %s",
                self._run_id,
                exc_info=cleanup_error,
            )
        if not remove_unstarted:
            return
        try:
            await _to_thread_settled(self._remove_unstarted_run_sync)
        except BaseException as cleanup_error:
            _logger.warning(
                "Failed to remove unstarted Journal Run %s",
                self._run_id,
                exc_info=cleanup_error,
            )
        else:
            self._owns_unstarted_run = False

    def _remove_unstarted_run_sync(self) -> None:
        path = self._path
        if path is None:
            return
        run_directory = path.parent
        if not run_directory.exists():
            return
        allowed_names = {
            JOURNAL_FILENAME,
            INDEX_FILENAME,
            "journal.writer.lock",
        }
        entries = tuple(run_directory.iterdir())
        unexpected = [
            entry.name for entry in entries if entry.name not in allowed_names
        ]
        if unexpected:
            raise JournalError(
                "refusing to remove an unstarted Run with unexpected files: "
                + ", ".join(sorted(unexpected))
            )
        for entry in entries:
            entry.unlink(missing_ok=True)
        run_directory.rmdir()
        self._sync_directory(run_directory.parent)

    async def _close_locked(self, *, flush: bool) -> None:
        if self._closed:
            return
        path = self._path
        failure: BaseException | None = None
        try:
            if flush and path is not None and self._writer_lease is not None:
                await _to_thread_settled(self._flush_sync, path)
        except BaseException as exc:
            failure = exc
        sqlite_index = self._sqlite_index
        writer_lease = self._writer_lease
        self._sqlite_index = None
        self._writer_lease = None
        self._closed = True
        if sqlite_index is not None:
            try:
                await _to_thread_settled(sqlite_index.close)
            except BaseException as exc:
                failure = self._retain_close_failure(failure, exc)
        if writer_lease is not None:
            try:
                await _to_thread_settled(writer_lease.release)
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
        index_path = path.parent / INDEX_FILENAME
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
        rebuild = asyncio.create_task(
            asyncio.to_thread(
                SqliteJournalIndex.rebuild,
                path.parent / INDEX_FILENAME,
                path,
                self._run_id,
                self._records,
            ),
            name=f"qitos-journal-index-{self._run_id}",
        )
        try:
            self._sqlite_index = await asyncio.shield(rebuild)
        except asyncio.CancelledError as cancellation:
            try:
                self._sqlite_index = await _settle_task(rebuild)
            except JournalIndexError as rebuild_error:
                self._sqlite_index = None
                raise cancellation from rebuild_error
            raise
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
        return read_journal_records(
            path,
            run_id,
            repair_tail=True,
            sync_file=self._sync_file,
            sync_mode=_file_sync_mode(),
        )


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


async def _settle_task(task: asyncio.Task[_T]) -> _T:
    """Wait through repeated caller cancellation for one owned lifecycle Task."""

    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()


async def _to_thread_settled(
    operation: Callable[..., _T],
    *args: Any,
) -> _T:
    task = asyncio.create_task(asyncio.to_thread(operation, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await _settle_task(task)
        raise


def _clone_record(record: JournalRecord) -> JournalRecord:
    return JournalRecord.from_dict(record.to_dict())


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
        result = ToolResult.from_dict(copy.deepcopy(dict(raw_result)))
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
