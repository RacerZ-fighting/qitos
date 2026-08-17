"""In-memory implementation of the canonical Run journal.

``InMemorySessionJournal`` shares the canonical record, idempotency, fork
lineage and committed-Tool query contract with ``JsonlSessionJournal``;
durability is scoped to one process-visible ``InMemoryJournalStore``. Every
operation settles synchronously without awaiting I/O, so cancellation before
admission raises ordinary ``CancelledError`` and commits nothing — the
``JournalAppendCancelled`` post-admission path never applies here.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from ...core.journal import (
    JournalClosedError,
    JournalError,
    JournalOwnershipError,
    JournalPosition,
    JournalRecord,
    JournalRecordRef,
    JournalRecordType,
    ToolTransaction,
    resolve_inherited_record,
)
from ._paths import validate_run_id
from ._transaction_index import ToolTransactionIndex


class InMemoryJournalStore:
    """Process-local durable medium shared by in-memory journal instances.

    This is the in-memory analog of the JSONL catalog root: records become
    visible to later instances as soon as the append settles, and at most one
    open instance may own a Run at a time.
    """

    def __init__(self) -> None:
        self.records: dict[str, list[JournalRecord]] = {}
        self.open_run_ids: set[str] = set()


class InMemorySessionJournal:
    """One in-memory, single-writer journal per Run."""

    def __init__(self, store: InMemoryJournalStore | None = None) -> None:
        self._store = store if store is not None else InMemoryJournalStore()
        self._run_id = ""
        self._records: list[JournalRecord] | None = None
        self._closed = False
        self._tool_index = ToolTransactionIndex()

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def closed(self) -> bool:
        return self._closed

    async def __aenter__(self) -> "InMemorySessionJournal":
        if self._closed:
            raise JournalClosedError("journal is closed")
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def create(self, run_id: str, metadata: Mapping[str, Any]) -> JournalPosition:
        validate_run_id(run_id)
        self._ensure_can_open()
        if run_id in self._store.records:
            raise JournalError("journal run already exists")
        self._store.records[run_id] = []
        self._store.open_run_ids.add(run_id)
        self._run_id = run_id
        self._records = self._store.records[run_id]
        try:
            return self._append_settled(
                JournalRecordType.RUN_STARTED,
                metadata,
                record_id=f"{run_id}:start",
            )
        except BaseException:
            self._store.records.pop(run_id, None)
            self._store.open_run_ids.discard(run_id)
            self._run_id = ""
            self._records = None
            raise

    async def open(self, run_id: str) -> None:
        validate_run_id(run_id)
        self._ensure_can_open()
        stored = self._store.records.get(run_id)
        if stored is None:
            raise JournalError("journal run does not exist")
        if run_id in self._store.open_run_ids:
            raise JournalOwnershipError("journal run has an active writer")
        self._store.open_run_ids.add(run_id)
        self._run_id = run_id
        self._records = stored
        try:
            self._tool_index.reset(tuple(_clone_record(r) for r in stored))
        except BaseException:
            self._store.open_run_ids.discard(run_id)
            self._run_id = ""
            self._records = None
            raise

    async def append(
        self,
        record_type: JournalRecordType,
        payload: Mapping[str, Any],
        *,
        record_id: str,
    ) -> JournalPosition:
        return self._append_settled(record_type, payload, record_id=record_id)

    def _append_settled(
        self,
        record_type: JournalRecordType,
        payload: Mapping[str, Any],
        *,
        record_id: str,
    ) -> JournalPosition:
        records = self._require_open_records()
        if not isinstance(record_type, JournalRecordType):
            raise TypeError("record_type must be a JournalRecordType")
        for existing in records:
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
            seq=len(records) + 1,
            record_id=record_id,
            type=record_type,
            run_id=self._run_id,
            payload=payload,
        )
        # The store write is the durable commit; indexing follows it exactly
        # like the JSONL canonical write precedes its query projection.
        records.append(record)
        self._tool_index.add(record)
        return record.position

    async def replay(self) -> tuple[JournalRecord, ...]:
        records = self._require_records_available()
        return tuple(_clone_record(record) for record in records)

    def find_tool_transaction(
        self, reference: JournalRecordRef
    ) -> ToolTransaction | None:
        """Return a fresh typed view of one committed Tool terminal."""

        self._require_records_available()
        return self._tool_index.find(reference)

    async def flush(self) -> None:
        """Settle immediately; in-memory appends are durable when they return."""

        self._require_open_records()

    async def close(self) -> None:
        """Permanently release writer ownership of this journal instance."""

        if self._closed:
            return
        if self._run_id:
            self._store.open_run_ids.discard(self._run_id)
        self._closed = True

    async def fork(
        self,
        parent_position: JournalPosition,
        new_run_id: str,
    ) -> "InMemorySessionJournal":
        records = self._require_open_records()
        if parent_position.run_id != self._run_id:
            raise ValueError("fork position belongs to another Run")
        if not 1 <= parent_position.seq <= len(records):
            raise ValueError("fork position is unavailable")
        source = records[parent_position.seq - 1]
        if source.record_id != parent_position.record_id:
            raise ValueError("fork position does not match the journal")
        if resolve_inherited_record(source).type is not JournalRecordType.STEP_COMMITTED:
            raise ValueError("fork position is not a committed boundary")
        inherited = tuple(_clone_record(r) for r in records[: parent_position.seq])
        child = type(self)(self._store)
        try:
            child_metadata = copy.deepcopy(records[0].payload)
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
        if self._records is not None:
            raise JournalError("journal is already open")

    def _require_open_records(self) -> list[JournalRecord]:
        if self._closed:
            raise JournalClosedError("journal is closed")
        if self._records is None:
            raise JournalError("journal is not open")
        return self._records

    def _require_records_available(self) -> list[JournalRecord]:
        # Replay and the committed-Tool query remain available after close for
        # diagnostics, matching the JSONL lifecycle.
        if self._records is None:
            raise JournalError("journal is not open")
        return self._records


def _clone_record(record: JournalRecord) -> JournalRecord:
    return JournalRecord.from_dict(record.to_dict())


__all__ = ["InMemoryJournalStore", "InMemorySessionJournal"]
