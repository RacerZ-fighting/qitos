"""SQLite-backed asynchronous checkpoint persistence."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable
from copy import deepcopy
from typing import Any, List, Optional, TypeVar

from .store import (
    Checkpoint,
    CheckpointConfig,
    CheckpointId,
    CheckpointMetadata,
    CheckpointStore,
    CheckpointTuple,
)


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id  TEXT PRIMARY KEY,
    thread_id      TEXT NOT NULL,
    step           INTEGER NOT NULL,
    state_data     TEXT NOT NULL,
    task_text      TEXT NOT NULL DEFAULT '',
    task_data      TEXT,
    history_data   TEXT,
    parent_id      TEXT,
    parent_thread_id TEXT,
    created_at     TEXT NOT NULL,
    schema_version TEXT NOT NULL DEFAULT 'v3'
);

CREATE TABLE IF NOT EXISTS checkpoint_metadata (
    checkpoint_id TEXT PRIMARY KEY REFERENCES checkpoints(checkpoint_id),
    source        TEXT,
    step_int      INTEGER,
    parents       TEXT DEFAULT '{}',
    run_id        TEXT
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_thread
    ON checkpoints(thread_id, step);

"""

_CHECKPOINT_COLUMNS = (
    "checkpoint_id, thread_id, step, state_data, task_text, task_data, "
    "history_data, parent_id, parent_thread_id, created_at, schema_version"
)

T = TypeVar("T")


class SqliteCheckpointStore(CheckpointStore):
    """Serialize SQLite work off the caller's event loop.

    One connection and one async lock preserve operation ordering. Cancellation
    waits for the active SQLite call to settle before it propagates, so callers
    never have an unknown in-process write still using the connection.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    async def _run(self, operation: Callable[[sqlite3.Connection], T]) -> T:
        async with self._lock:
            if self._closed:
                raise RuntimeError("checkpoint store is closed")
            task = asyncio.create_task(
                asyncio.to_thread(self._run_sync, operation)
            )
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError as cancelled:
                try:
                    await asyncio.shield(task)
                except Exception as exc:
                    raise cancelled from exc
                raise

    def _run_sync(self, operation: Callable[[sqlite3.Connection], T]) -> T:
        return operation(self._ensure_connection())

    def _ensure_connection(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(_SCHEMA_SQL)
            columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(checkpoints)").fetchall()
            }
            for name, declaration in (
                ("task_text", "TEXT NOT NULL DEFAULT ''"),
                ("task_data", "TEXT"),
                ("history_data", "TEXT"),
                ("parent_thread_id", "TEXT"),
            ):
                if name not in columns:
                    conn.execute(
                        f"ALTER TABLE checkpoints ADD COLUMN {name} {declaration}"
                    )
            conn.commit()
        except BaseException:
            conn.close()
            raise
        self._conn = conn
        return conn

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            conn = self._conn
            self._conn = None
            if conn is None:
                return
            task = asyncio.create_task(asyncio.to_thread(conn.close))
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as cancelled:
                try:
                    await asyncio.shield(task)
                except Exception as exc:
                    raise cancelled from exc
                raise

    @staticmethod
    def _row_to_checkpoint(row: tuple[Any, ...]) -> Checkpoint:
        (
            checkpoint_id,
            thread_id,
            step,
            state_data_json,
            task_text,
            task_data_json,
            history_data_json,
            parent_id,
            parent_thread_id,
            created_at,
            schema_version,
        ) = row
        try:
            payload: dict[str, Any] = {
                "id": checkpoint_id,
                "thread_id": thread_id,
                "step": step,
                "state_data": json.loads(state_data_json),
                "task_text": task_text,
                "task_data": (
                    json.loads(task_data_json) if task_data_json is not None else None
                ),
                "history": (
                    json.loads(history_data_json)
                    if history_data_json is not None
                    else None
                ),
                "parent_id": parent_id,
                "parent_thread_id": parent_thread_id,
                "created_at": created_at,
                "schema_version": schema_version,
            }
            return Checkpoint.from_dict(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("stored checkpoint payload is invalid") from exc

    @staticmethod
    def _load_metadata(
        conn: sqlite3.Connection, checkpoint_id: str
    ) -> CheckpointMetadata:
        row = conn.execute(
            "SELECT source, step_int, parents, run_id FROM checkpoint_metadata "
            "WHERE checkpoint_id = ?",
            (checkpoint_id,),
        ).fetchone()
        if row is None:
            return CheckpointMetadata()
        source, step_int, parents_json, run_id = row
        metadata: CheckpointMetadata = {}
        if source is not None:
            metadata["source"] = source
        if step_int is not None:
            metadata["step"] = step_int
        if parents_json is not None:
            try:
                parents = json.loads(parents_json)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("stored checkpoint metadata is invalid") from exc
            if not isinstance(parents, dict):
                raise ValueError("stored checkpoint parents must be an object")
            metadata["parents"] = parents
        if run_id is not None:
            metadata["run_id"] = run_id
        return metadata

    @staticmethod
    def _resolve_id(
        conn: sqlite3.Connection, config: CheckpointConfig
    ) -> CheckpointId | None:
        if config.checkpoint_id is not None:
            return config.checkpoint_id
        row = conn.execute(
            "SELECT checkpoint_id FROM checkpoints "
            "WHERE thread_id = ? ORDER BY step DESC, rowid DESC LIMIT 1",
            (config.thread_id,),
        ).fetchone()
        return CheckpointId(row[0]) if row else None

    @classmethod
    def _get_tuple_sync(
        cls, conn: sqlite3.Connection, config: CheckpointConfig
    ) -> CheckpointTuple | None:
        checkpoint_id = cls._resolve_id(conn, config)
        if checkpoint_id is None:
            return None
        row = conn.execute(
            f"SELECT {_CHECKPOINT_COLUMNS} FROM checkpoints "
            "WHERE checkpoint_id = ? AND thread_id = ?",
            (checkpoint_id, config.thread_id),
        ).fetchone()
        if row is None:
            return None
        checkpoint = cls._row_to_checkpoint(row)
        parent_config = (
            None
            if checkpoint.parent_id is None
            else CheckpointConfig(
                thread_id=checkpoint.parent_thread_id or checkpoint.thread_id,
                checkpoint_id=checkpoint.parent_id,
            )
        )
        return CheckpointTuple(
            config=CheckpointConfig(
                thread_id=checkpoint.thread_id,
                checkpoint_id=checkpoint.id,
            ),
            checkpoint=checkpoint,
            metadata=cls._load_metadata(conn, checkpoint_id),
            parent_config=parent_config,
        )

    async def put(
        self,
        config: CheckpointConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> CheckpointConfig:
        if config.thread_id != checkpoint.thread_id:
            raise ValueError("checkpoint thread_id does not match config")
        durable = checkpoint.to_dict()
        Checkpoint.from_dict(durable)
        owned_metadata = deepcopy(metadata)

        def operation(conn: sqlite3.Connection) -> CheckpointConfig:
            with conn:
                conn.execute(
                    "INSERT INTO checkpoints "
                    "(checkpoint_id, thread_id, step, state_data, task_text, "
                    "task_data, history_data, parent_id, parent_thread_id, "
                    "created_at, schema_version) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(checkpoint_id) DO UPDATE SET "
                    "thread_id = excluded.thread_id, "
                    "step = excluded.step, "
                    "state_data = excluded.state_data, "
                    "task_text = excluded.task_text, "
                    "task_data = excluded.task_data, "
                    "history_data = excluded.history_data, "
                    "parent_id = excluded.parent_id, "
                    "parent_thread_id = excluded.parent_thread_id, "
                    "created_at = excluded.created_at, "
                    "schema_version = excluded.schema_version "
                    "WHERE checkpoints.thread_id = excluded.thread_id",
                    (
                        durable["id"],
                        durable["thread_id"],
                        durable["step"],
                        json.dumps(
                            durable["state_data"],
                            ensure_ascii=False,
                            allow_nan=False,
                        ),
                        durable["task_text"],
                        (
                            json.dumps(
                                durable["task_data"],
                                ensure_ascii=False,
                                allow_nan=False,
                            )
                            if durable["task_data"] is not None
                            else None
                        ),
                        (
                            json.dumps(
                                durable["history"],
                                ensure_ascii=False,
                                allow_nan=False,
                            )
                            if durable["history"] is not None
                            else None
                        ),
                        durable["parent_id"],
                        durable["parent_thread_id"],
                        durable["created_at"],
                        durable["schema_version"],
                    ),
                )
                row = conn.execute(
                    "SELECT thread_id FROM checkpoints WHERE checkpoint_id = ?",
                    (durable["id"],),
                ).fetchone()
                if row is None or row[0] != durable["thread_id"]:
                    raise ValueError(
                        "checkpoint id already belongs to another thread"
                    )
                conn.execute(
                    "INSERT INTO checkpoint_metadata "
                    "(checkpoint_id, source, step_int, parents, run_id) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(checkpoint_id) DO UPDATE SET "
                    "source = excluded.source, "
                    "step_int = excluded.step_int, "
                    "parents = excluded.parents, "
                    "run_id = excluded.run_id",
                    (
                        durable["id"],
                        owned_metadata.get("source"),
                        owned_metadata.get("step"),
                        json.dumps(
                            owned_metadata.get("parents", {}),
                            ensure_ascii=False,
                            allow_nan=False,
                        ),
                        owned_metadata.get("run_id"),
                    ),
                )
            return CheckpointConfig(
                thread_id=config.thread_id,
                checkpoint_id=CheckpointId(durable["id"]),
            )

        return await self._run(operation)

    async def get_tuple(
        self, config: CheckpointConfig
    ) -> Optional[CheckpointTuple]:
        return await self._run(lambda conn: self._get_tuple_sync(conn, config))

    async def list(
        self,
        config: CheckpointConfig,
        *,
        limit: Optional[int] = None,
        before: Optional[CheckpointConfig] = None,
    ) -> List[CheckpointTuple]:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative or None")

        def operation(conn: sqlite3.Connection) -> List[CheckpointTuple]:
            params: list[Any] = [config.thread_id]
            sql = f"SELECT {_CHECKPOINT_COLUMNS} FROM checkpoints WHERE thread_id = ?"
            if before is not None and before.checkpoint_id is not None:
                before_row = conn.execute(
                    "SELECT step, rowid FROM checkpoints "
                    "WHERE checkpoint_id = ? AND thread_id = ?",
                    (before.checkpoint_id, config.thread_id),
                ).fetchone()
                if before_row is not None:
                    sql += (
                        " AND (step < ? OR (step = ? AND rowid < ?))"
                    )
                    params.extend([before_row[0], before_row[0], before_row[1]])
            sql += " ORDER BY step DESC, rowid DESC"
            if limit is not None:
                sql += " LIMIT ?"
                params.append(limit)

            results: List[CheckpointTuple] = []
            for row in conn.execute(sql, params).fetchall():
                checkpoint = self._row_to_checkpoint(row)
                parent_config = (
                    None
                    if checkpoint.parent_id is None
                    else CheckpointConfig(
                        thread_id=(
                            checkpoint.parent_thread_id or checkpoint.thread_id
                        ),
                        checkpoint_id=checkpoint.parent_id,
                    )
                )
                results.append(
                    CheckpointTuple(
                        config=CheckpointConfig(
                            thread_id=checkpoint.thread_id,
                            checkpoint_id=checkpoint.id,
                        ),
                        checkpoint=checkpoint,
                        metadata=self._load_metadata(conn, checkpoint.id),
                        parent_config=parent_config,
                    )
                )
            return results

        return await self._run(operation)

    async def delete(self, config: CheckpointConfig) -> None:
        def operation(conn: sqlite3.Connection) -> None:
            checkpoint_id = self._resolve_id(conn, config)
            if checkpoint_id is None:
                return
            row = conn.execute(
                "SELECT 1 FROM checkpoints WHERE checkpoint_id = ? "
                "AND thread_id = ?",
                (checkpoint_id, config.thread_id),
            ).fetchone()
            if row is None:
                return
            with conn:
                conn.execute(
                    "DELETE FROM checkpoint_metadata WHERE checkpoint_id = ?",
                    (checkpoint_id,),
                )
                conn.execute(
                    "DELETE FROM checkpoints WHERE checkpoint_id = ?",
                    (checkpoint_id,),
                )

        await self._run(operation)


__all__ = ["SqliteCheckpointStore"]
