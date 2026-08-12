"""Asynchronous checkpoint persistence contracts and snapshot data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, NamedTuple, NewType, Optional, TypedDict

from ..core.history import (
    HistoryMessage,
    HistorySnapshot,
    complete_history_prefix,
)

# ---------------------------------------------------------------------------
# Core type aliases
# ---------------------------------------------------------------------------

CheckpointId = NewType("CheckpointId", str)
"""Unique identifier for a checkpoint (UUID-based)."""

# ---------------------------------------------------------------------------
# CheckpointConfig — addresses a checkpoint within a store
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointConfig:
    """Address a specific checkpoint within a store.

    If ``checkpoint_id`` is ``None``, operations target the *latest*
    checkpoint for the given ``thread_id``.
    """

    thread_id: str
    checkpoint_id: Optional[CheckpointId] = None


# ---------------------------------------------------------------------------
# CheckpointMetadata
# ---------------------------------------------------------------------------

class CheckpointMetadata(TypedDict, total=False):
    """Metadata associated with a checkpoint."""

    source: str  # "input" | "loop" | "update" | "fork"
    step: int
    parents: Dict[str, str]
    run_id: str


# ---------------------------------------------------------------------------
# Checkpoint — the core snapshot data model
# ---------------------------------------------------------------------------

@dataclass
class Checkpoint:
    """One safe-boundary state and model-history snapshot."""

    id: CheckpointId
    thread_id: str
    step: int
    state_data: Dict[str, Any]
    task_text: str = ""
    task_data: Optional[Dict[str, Any]] = None
    history: Optional[HistorySnapshot] = None
    parent_id: Optional[CheckpointId] = None
    parent_thread_id: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    schema_version: str = "v3"

    # ---- serialization helpers ----

    def to_dict(self) -> Dict[str, Any]:
        """Return an owned, JSON-compatible checkpoint payload.

        The store receives live engine state objects.  Serialization is kept
        at this persistence boundary and returns no containers owned by the
        caller, so an in-flight write cannot observe later state mutations.
        """
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "step": self.step,
            "state_data": deepcopy(self.state_data),
            "task_text": self.task_text,
            "task_data": deepcopy(self.task_data),
            "history": (
                None
                if self.history is None
                else deepcopy(
                    {
                        "messages": [
                            asdict(message) for message in self.history.messages
                        ],
                        "source_revision": self.history.source_revision,
                    }
                )
            ),
            "parent_id": self.parent_id,
            "parent_thread_id": self.parent_thread_id,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> Checkpoint:
        """Build a checkpoint from a validated durable payload.

        Malformed persisted data raises ``ValueError`` at the storage
        boundary instead of leaking ``KeyError``/``TypeError`` from the
        dataclass constructor into resume callers.
        """
        if not isinstance(payload, dict):
            raise ValueError("checkpoint payload must be an object")
        checkpoint_id = payload.get("id")
        thread_id = payload.get("thread_id")
        step = payload.get("step")
        state_data = payload.get("state_data")
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise ValueError("checkpoint id must be a non-empty string")
        if not isinstance(thread_id, str) or not thread_id:
            raise ValueError("checkpoint thread_id must be a non-empty string")
        if not isinstance(step, int) or isinstance(step, bool):
            raise ValueError("checkpoint step must be an integer")
        if not isinstance(state_data, dict):
            raise ValueError("checkpoint state_data must be an object")

        raw_history = payload.get("history")
        history: Optional[HistorySnapshot] = None
        if raw_history is not None:
            if not isinstance(raw_history, dict):
                raise ValueError("checkpoint history must be an object or null")
            raw_messages = raw_history.get("messages")
            if not isinstance(raw_messages, list):
                raise ValueError("checkpoint history messages must be a list")
            messages: List[HistoryMessage] = []
            for item in raw_messages:
                if not isinstance(item, dict):
                    raise ValueError("checkpoint history message must be an object")
                try:
                    messages.append(HistoryMessage(**item))
                except (TypeError, ValueError) as exc:
                    raise ValueError("checkpoint history message is invalid") from exc
            source_revision = raw_history.get("source_revision")
            if source_revision is not None and (
                not isinstance(source_revision, int)
                or isinstance(source_revision, bool)
            ):
                raise ValueError(
                    "checkpoint history source_revision must be an integer"
                )
            if len(complete_history_prefix(messages)) != len(messages):
                raise ValueError(
                    "checkpoint history contains an incomplete tool transaction"
                )
            history = HistorySnapshot(
                messages=tuple(messages),
                source_revision=source_revision,
            )

        task_data = payload.get("task_data")
        if task_data is not None and not isinstance(task_data, dict):
            raise ValueError("checkpoint task_data must be an object or null")
        task_text = payload.get("task_text", "")
        if not isinstance(task_text, str):
            raise ValueError("checkpoint task_text must be a string")
        parent_id = payload.get("parent_id")
        if parent_id is not None and (
            not isinstance(parent_id, str) or not parent_id
        ):
            raise ValueError("checkpoint parent_id must be a string or null")
        parent_thread_id = payload.get("parent_thread_id")
        if parent_thread_id is not None and (
            not isinstance(parent_thread_id, str) or not parent_thread_id
        ):
            raise ValueError(
                "checkpoint parent_thread_id must be a string or null"
            )
        created_at = payload.get("created_at", "")
        schema_version = payload.get("schema_version", "v3")
        if not isinstance(created_at, str) or not isinstance(schema_version, str):
            raise ValueError("checkpoint timestamps and schema version must be strings")
        return cls(
            id=CheckpointId(checkpoint_id),
            thread_id=thread_id,
            step=step,
            state_data=deepcopy(state_data),
            task_text=task_text,
            task_data=deepcopy(task_data),
            history=history,
            parent_id=CheckpointId(parent_id) if parent_id is not None else None,
            parent_thread_id=parent_thread_id,
            created_at=created_at,
            schema_version=schema_version,
        )


# ---------------------------------------------------------------------------
# CheckpointTuple — bundles checkpoint + metadata + parent for retrieval
# ---------------------------------------------------------------------------

class CheckpointTuple(NamedTuple):
    """A checkpoint together with its metadata and parent reference."""

    config: CheckpointConfig
    checkpoint: Checkpoint
    metadata: CheckpointMetadata
    parent_config: Optional[CheckpointConfig] = None


# ---------------------------------------------------------------------------
# CheckpointStore ABC
# ---------------------------------------------------------------------------

class CheckpointStore(ABC):
    """Single asynchronous owner for checkpoint persistence."""

    @abstractmethod
    async def put(
        self,
        config: CheckpointConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> CheckpointConfig:
        """Store a checkpoint.  Returns the updated config."""

    @abstractmethod
    async def get_tuple(
        self, config: CheckpointConfig
    ) -> Optional[CheckpointTuple]:
        """Fetch a checkpoint tuple.  Returns ``None`` if not found."""

    async def get(self, config: CheckpointConfig) -> Optional[Checkpoint]:
        """Fetch just the checkpoint (convenience wrapper)."""
        result = await self.get_tuple(config)
        return result.checkpoint if result else None

    @abstractmethod
    async def list(
        self,
        config: CheckpointConfig,
        *,
        limit: Optional[int] = None,
        before: Optional[CheckpointConfig] = None,
    ) -> List[CheckpointTuple]:
        """List checkpoints for a thread, newest first."""

    @abstractmethod
    async def delete(self, config: CheckpointConfig) -> None:
        """Delete a single checkpoint."""

    async def close(self) -> None:
        """Release store-owned resources."""

    async def __aenter__(self) -> CheckpointStore:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.close()


__all__ = [
    "CheckpointId",
    "CheckpointConfig",
    "Checkpoint",
    "CheckpointMetadata",
    "CheckpointTuple",
    "CheckpointStore",
]
