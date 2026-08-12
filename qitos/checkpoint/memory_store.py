"""In-memory checkpoint store for development and testing.

Dict-backed implementation of :class:`CheckpointStore`.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from .store import (
    Checkpoint,
    CheckpointConfig,
    CheckpointId,
    CheckpointMetadata,
    CheckpointStore,
    CheckpointTuple,
)


class InMemoryCheckpointStore(CheckpointStore):
    """Task-safe, dict-backed checkpoint store.

    Suitable for development, testing, and single-process scenarios.
    All data is lost when the process exits.
    """

    def __init__(self) -> None:
        self._store: Dict[CheckpointId, CheckpointTuple] = {}
        self._thread_index: Dict[str, List[CheckpointId]] = {}
        self._lock = asyncio.Lock()

    # ---- helpers ----

    def _latest_id(self, thread_id: str) -> Optional[CheckpointId]:
        ids = self._thread_index.get(thread_id)
        if not ids:
            return None
        candidates = [
            (self._store[checkpoint_id].checkpoint.step, position, checkpoint_id)
            for position, checkpoint_id in enumerate(ids)
            if checkpoint_id in self._store
        ]
        if not candidates:
            return None
        return max(candidates)[2]

    def _resolve_id(self, config: CheckpointConfig) -> Optional[CheckpointId]:
        if config.checkpoint_id is not None:
            return config.checkpoint_id
        return self._latest_id(config.thread_id)

    async def put(
        self,
        config: CheckpointConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> CheckpointConfig:
        self._bind_owner_loop()
        if config.thread_id != checkpoint.thread_id:
            raise ValueError("checkpoint thread_id does not match config")
        owned_checkpoint = Checkpoint.from_dict(checkpoint.to_dict())
        owned_metadata = self._validate_metadata(metadata)
        async with self._lock:
            existing = self._store.get(owned_checkpoint.id)
            if existing is not None and (
                existing.checkpoint.thread_id != owned_checkpoint.thread_id
            ):
                raise ValueError("checkpoint id already belongs to another thread")

            parent_config: Optional[CheckpointConfig] = None
            if owned_checkpoint.parent_id is not None:
                parent_config = CheckpointConfig(
                    thread_id=(
                        owned_checkpoint.parent_thread_id
                        or owned_checkpoint.thread_id
                    ),
                    checkpoint_id=owned_checkpoint.parent_id,
                )

            persisted_config = CheckpointConfig(
                thread_id=config.thread_id, checkpoint_id=owned_checkpoint.id
            )
            tuple_ = CheckpointTuple(
                config=persisted_config,
                checkpoint=owned_checkpoint,
                metadata=owned_metadata,
                parent_config=parent_config,
            )
            self._store[owned_checkpoint.id] = tuple_

            tid = config.thread_id
            if tid not in self._thread_index:
                self._thread_index[tid] = []
            if owned_checkpoint.id not in self._thread_index[tid]:
                self._thread_index[tid].append(owned_checkpoint.id)

            return persisted_config

    async def get_tuple(
        self, config: CheckpointConfig
    ) -> Optional[CheckpointTuple]:
        self._bind_owner_loop()
        async with self._lock:
            cp_id = self._resolve_id(config)
            if cp_id is None:
                return None
            entry = self._store.get(cp_id)
            if entry is not None and entry.checkpoint.thread_id != config.thread_id:
                return None
            if entry is None:
                return None
            return CheckpointTuple(
                config=entry.config,
                checkpoint=Checkpoint.from_dict(entry.checkpoint.to_dict()),
                metadata=self._validate_metadata(entry.metadata),
                parent_config=entry.parent_config,
            )

    async def list(
        self,
        config: CheckpointConfig,
        *,
        limit: Optional[int] = None,
        before: Optional[CheckpointConfig] = None,
    ) -> List[CheckpointTuple]:
        self._bind_owner_loop()
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative or None")
        async with self._lock:
            indexed = [
                (position, checkpoint_id)
                for position, checkpoint_id in enumerate(
                    self._thread_index.get(config.thread_id, [])
                )
                if checkpoint_id in self._store
            ]
            indexed.sort(
                key=lambda item: (self._store[item[1]].checkpoint.step, item[0]),
                reverse=True,
            )
            ids = [checkpoint_id for _, checkpoint_id in indexed]
            if before is not None and before.checkpoint_id is not None:
                try:
                    idx = ids.index(before.checkpoint_id)
                    ids = ids[idx + 1 :]
                except ValueError:
                    pass
            if limit is not None:
                ids = ids[:limit]
            results: List[CheckpointTuple] = []
            for checkpoint_id in ids:
                entry = self._store.get(checkpoint_id)
                if entry is None:
                    continue
                results.append(
                    CheckpointTuple(
                        config=entry.config,
                        checkpoint=Checkpoint.from_dict(entry.checkpoint.to_dict()),
                        metadata=self._validate_metadata(entry.metadata),
                        parent_config=entry.parent_config,
                    )
                )
            return results

    async def delete(self, config: CheckpointConfig) -> None:
        self._bind_owner_loop()
        async with self._lock:
            cp_id = self._resolve_id(config)
            if cp_id is None:
                return
            existing = self._store.get(cp_id)
            if existing is None or existing.checkpoint.thread_id != config.thread_id:
                return
            self._store.pop(cp_id, None)
            ids = self._thread_index.get(config.thread_id, [])
            while cp_id in ids:
                ids.remove(cp_id)


__all__ = ["InMemoryCheckpointStore"]
