"""Lease-free queries over canonical JSONL Run journals."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ...core.journal import (
    JournalCorruptionError,
    JournalError,
    JournalPosition,
    JournalRecord,
    JournalRecordType,
    resolve_inherited_record,
)
from ...core.run import RunHandle, RunNotFoundError, RunStatus
from ._paths import JOURNAL_FILENAME, journal_path, validate_run_id
from ._reader import read_journal_records
from .turn_recorder import (
    decode_input_accepted,
    decode_run_terminal,
    decode_step_committed,
    decode_task_created,
)

_COMMITTED_BOUNDARIES = {
    JournalRecordType.STEP_COMMITTED,
}


@dataclass(frozen=True, slots=True)
class _RunView:
    handle: RunHandle
    records: tuple[JournalRecord, ...]


class JsonlRunCatalog:
    """Inspect JSONL Runs without acquiring a writer lease or repairing files.

    JSONL is the canonical record source. This reader never consults, repairs, or
    rebuilds the disposable SQLite projection.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    async def inspect_run(self, run_id: str) -> RunHandle:
        """Return one immutable summary from the latest readable durable prefix."""

        return (await asyncio.to_thread(self._inspect_sync, run_id)).handle

    async def list_runs(self) -> tuple[RunHandle, ...]:
        """List Runs by newest durable update, then by Run identifier."""

        views = await asyncio.to_thread(self._list_views_sync)
        return tuple(view.handle for view in views)

    async def lineage(self, run_id: str) -> tuple[RunHandle, ...]:
        """Return validated ancestry in root-to-target order."""

        return await asyncio.to_thread(self._lineage_sync, run_id)

    async def children(self, run_id: str) -> tuple[RunHandle, ...]:
        """Return validated direct children in catalog order."""

        return await asyncio.to_thread(self._children_sync, run_id)

    def _inspect_sync(self, run_id: str) -> _RunView:
        records = tuple(self._read_records_sync(run_id))
        return _RunView(_summarize_run(run_id, records), records)

    def _list_views_sync(self) -> tuple[_RunView, ...]:
        if not self.root.exists():
            return ()
        if not self.root.is_dir():
            raise JournalError("Run catalog root is not a directory")
        views: list[_RunView] = []
        try:
            entries = sorted(self.root.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            raise JournalError("failed to list Run catalog") from exc
        for entry in entries:
            if entry.is_symlink() or not entry.is_dir():
                continue
            try:
                validate_run_id(entry.name)
            except ValueError:
                continue
            candidate = entry / JOURNAL_FILENAME
            if not candidate.exists():
                continue
            views.append(self._inspect_sync(entry.name))
        views.sort(key=lambda view: view.handle.run_id)
        views.sort(key=lambda view: view.handle.updated_at, reverse=True)
        return tuple(views)

    def _lineage_sync(self, run_id: str) -> tuple[RunHandle, ...]:
        validate_run_id(run_id)
        chain: list[_RunView] = []
        seen: set[str] = set()
        current = self._inspect_sync(run_id)
        while True:
            current_id = current.handle.run_id
            if current_id in seen:
                raise JournalCorruptionError("Run lineage contains a cycle")
            seen.add(current_id)
            chain.append(current)
            origin = current.handle.forked_from
            if origin is None:
                break
            if origin.run_id in seen:
                raise JournalCorruptionError("Run lineage contains a cycle")
            parent = self._inspect_sync(origin.run_id)
            _validate_fork(parent, current)
            current = parent
        chain.reverse()
        return tuple(view.handle for view in chain)

    def _children_sync(self, run_id: str) -> tuple[RunHandle, ...]:
        parent = self._inspect_sync(run_id)
        children: list[RunHandle] = []
        for candidate in self._list_views_sync():
            origin = candidate.handle.forked_from
            if origin is None or origin.run_id != run_id:
                continue
            _validate_fork(parent, candidate)
            children.append(candidate.handle)
        return tuple(children)

    def _read_records_sync(self, run_id: str) -> list[JournalRecord]:
        path = journal_path(self.root, run_id)
        run_directory = path.parent
        if run_directory.is_symlink() or path.is_symlink():
            raise JournalError("Run journal path must not be a symbolic link")
        if not path.is_file():
            raise RunNotFoundError(f"Run {run_id!r} was not found")
        return read_journal_records(path, run_id, repair_tail=False)


def _summarize_run(
    run_id: str,
    records: tuple[JournalRecord, ...],
) -> RunHandle:
    start = records[0]
    created_at = _parse_timestamp(start)
    updated_at = _parse_timestamp(records[-1])
    forked_from = _fork_origin(records)
    committed_position: JournalPosition | None = None
    committed_records: list[tuple[int, JournalRecord]] = []
    completed: JournalRecord | None = None
    interrupted: JournalRecord | None = None
    agent_name = _optional_text(start.payload, "agent")
    task = ""
    for index, record in enumerate(records):
        effective = resolve_inherited_record(record)
        try:
            if effective.type in _COMMITTED_BOUNDARIES:
                # Old-shape commits (legacy ``terminal_record_ids`` or
                # Engine-era fields) fail closed instead of being skipped.
                decode_step_committed(effective.payload)
                committed_position = record.position
                committed_records.append((index, record))
            elif effective.type is JournalRecordType.INPUT_ACCEPTED:
                # The task text arrives with the goal-bearing Task (S3);
                # input.accepted references prompt transcript entries only.
                decode_input_accepted(effective.payload)
            elif effective.type is JournalRecordType.TASK_CREATED:
                task = decode_task_created(effective.payload).objective
            elif effective.type in (
                JournalRecordType.RUN_COMPLETED,
                JournalRecordType.RUN_INTERRUPTED,
            ):
                decode_run_terminal(effective.type, effective.payload)
        except ValueError as exc:
            raise JournalCorruptionError(str(exc)) from exc
        if agent_name is None and effective.type is JournalRecordType.RUN_STARTED:
            agent_name = _optional_text(effective.payload, "agent")
        if record.type is JournalRecordType.RUN_INTERRUPTED:
            interrupted = record
        if record.type is JournalRecordType.RUN_COMPLETED:
            if completed is not None:
                raise JournalCorruptionError(
                    "Run journal contains multiple completion records"
                )
            completed = record
    return RunHandle(
        run_id=run_id,
        lineage_id=_lineage_id(run_id, records),
        status=(RunStatus.COMPLETED if completed is not None else RunStatus.RESUMABLE),
        created_at=created_at,
        updated_at=updated_at,
        latest_position=records[-1].position,
        committed_position=committed_position,
        continuation_position=_continuation_position(
            records,
            committed_records,
        ),
        forked_from=forked_from,
        record_count=len(records),
        agent_name=agent_name,
        task=task,
        stop_reason=(
            str(completed.payload["status"]) if completed is not None else None
        ),
        interrupted_at=interrupted.position if interrupted is not None else None,
        interruption_reason=(
            interrupted.payload["error"] if interrupted is not None else None
        ),
    )


def _lineage_id(run_id: str, records: tuple[JournalRecord, ...]) -> str:
    declared: set[str] = set()
    inherited_starts: list[JournalRecord] = []
    for record in records:
        effective = resolve_inherited_record(record)
        if effective.type is not JournalRecordType.RUN_STARTED:
            continue
        if effective.run_id != run_id:
            inherited_starts.append(effective)
        value = effective.payload.get("lineage_id")
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise JournalCorruptionError("run.started lineage_id is invalid")
        declared.add(value)
    if len(declared) > 1:
        raise JournalCorruptionError("Run lineage contains conflicting lineage ids")
    if declared:
        return next(iter(declared))
    return inherited_starts[-1].run_id if inherited_starts else run_id


def _continuation_position(
    records: tuple[JournalRecord, ...],
    committed_records: list[tuple[int, JournalRecord]],
) -> JournalPosition | None:
    """Return where this Run may be continued or explicitly forked from.

    That is the last own-run ``step.committed``; a forked journal without an
    own commit continues at its fork boundary (the inherited prefix tip). A
    root journal without any committed turn exposes no continuation point.
    """

    for _index, record in reversed(committed_records):
        if record.type is not JournalRecordType.INHERITED:
            return record.position
    for record in reversed(records):
        if record.type is JournalRecordType.INHERITED:
            return record.position
    return None


def _fork_origin(records: tuple[JournalRecord, ...]) -> JournalPosition | None:
    forks = [
        record for record in records if record.type is JournalRecordType.RUN_FORKED
    ]
    if len(forks) > 1:
        raise JournalCorruptionError("Run journal contains multiple fork records")
    inherited = any(record.type is JournalRecordType.INHERITED for record in records)
    if not forks:
        if inherited:
            raise JournalCorruptionError("Run journal inherits records without a fork")
        return None
    fork = forks[0]
    if fork.seq != 2:
        raise JournalCorruptionError("run.forked is not the first Run event")
    parent_run_id = fork.payload.get("parent_run_id")
    parent_seq = fork.payload.get("parent_seq")
    parent_record_id = fork.payload.get("parent_record_id")
    if (
        not isinstance(parent_run_id, str)
        or not parent_run_id
        or isinstance(parent_seq, bool)
        or not isinstance(parent_seq, int)
        or parent_seq <= 0
        or not isinstance(parent_record_id, str)
        or not parent_record_id
    ):
        raise JournalCorruptionError("run.forked payload is invalid")
    return JournalPosition(parent_run_id, parent_seq, parent_record_id)


def _validate_fork(parent: _RunView, child: _RunView) -> None:
    origin = child.handle.forked_from
    if origin is None or origin.run_id != parent.handle.run_id:
        raise JournalCorruptionError("Run fork parent does not match its lineage")
    if child.handle.run_id == parent.handle.run_id:
        raise JournalCorruptionError("Run lineage contains a self fork")
    if origin.seq > len(parent.records):
        raise JournalCorruptionError("Run fork cutoff is unavailable in its parent")
    boundary = parent.records[origin.seq - 1]
    if boundary.record_id != origin.record_id:
        raise JournalCorruptionError(
            "Run fork cutoff identity does not match its parent"
        )
    if resolve_inherited_record(boundary).type not in _COMMITTED_BOUNDARIES:
        raise JournalCorruptionError("Run fork cutoff is not a committed boundary")
    inherited_end = 2 + origin.seq
    inherited = child.records[2:inherited_end]
    if len(inherited) != origin.seq or any(
        record.type is not JournalRecordType.INHERITED for record in inherited
    ):
        raise JournalCorruptionError("Run fork inherited prefix is incomplete")
    if any(
        record.type is JournalRecordType.INHERITED
        for record in child.records[inherited_end:]
    ):
        raise JournalCorruptionError("Run fork inherited prefix is not contiguous")
    for expected, wrapper in zip(
        parent.records[: origin.seq],
        inherited,
        strict=True,
    ):
        raw_record = wrapper.payload.get("record")
        if not isinstance(raw_record, Mapping):
            raise JournalCorruptionError(
                "journal.inherited is missing its origin record"
            )
        embedded = JournalRecord.from_dict(raw_record)
        resolve_inherited_record(wrapper)
        if embedded.to_dict() != expected.to_dict():
            raise JournalCorruptionError(
                "Run fork inherited prefix does not match its parent"
            )


def _parse_timestamp(record: JournalRecord) -> datetime:
    try:
        value = datetime.fromisoformat(record.timestamp)
    except ValueError as exc:
        raise JournalCorruptionError("Run journal timestamp is invalid") from exc
    if value.tzinfo is None or value.utcoffset() is None:
        raise JournalCorruptionError("Run journal timestamp has no timezone")
    return value


def _optional_text(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise JournalCorruptionError(f"Run journal {key} is not text")
    return value or None


__all__ = ["JsonlRunCatalog"]
