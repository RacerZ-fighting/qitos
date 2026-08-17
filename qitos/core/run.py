"""Immutable summaries for persisted QitOS Runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

from .journal import JournalError, JournalPosition


class RunNotFoundError(JournalError):
    """Raised when a persisted Run cannot be found in a catalog."""


class RunStatus(str, Enum):
    """Lifecycle state that can be proven from durable Journal records alone."""

    RESUMABLE = "resumable"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class RunHandle:
    """Lease-free, immutable locator and summary for one persisted Run.

    ``RESUMABLE`` deliberately does not claim that a process is currently running.
    It means only that the canonical Journal has no durable ``run.completed`` record.
    """

    run_id: str
    lineage_id: str
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    latest_position: JournalPosition
    committed_position: JournalPosition | None
    continuation_position: JournalPosition | None
    forked_from: JournalPosition | None
    record_count: int
    agent_name: str | None = None
    task: str = ""
    stop_reason: str | None = None
    interrupted_at: JournalPosition | None = None
    interruption_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("Run handle run_id must be non-empty")
        if not isinstance(self.lineage_id, str) or not self.lineage_id.strip():
            raise ValueError("Run handle lineage_id must be non-empty")
        if not isinstance(self.status, RunStatus):
            raise TypeError("Run handle status must be a RunStatus")
        if self.latest_position.run_id != self.run_id:
            raise ValueError("Run handle latest position belongs to another Run")
        if (
            self.committed_position is not None
            and self.committed_position.run_id != self.run_id
        ):
            raise ValueError("Run handle committed position belongs to another Run")
        if (
            self.continuation_position is not None
            and self.continuation_position.run_id != self.run_id
        ):
            raise ValueError("Run handle continuation position belongs to another Run")
        if (
            self.interrupted_at is not None
            and self.interrupted_at.run_id != self.run_id
        ):
            raise ValueError("Run handle interruption belongs to another Run")
        if isinstance(self.record_count, bool) or self.record_count <= 0:
            raise ValueError("Run handle record_count must be positive")

    @property
    def is_terminal(self) -> bool:
        """Whether the Journal has a durable terminal completion record."""

        return self.status is RunStatus.COMPLETED

    @property
    def has_terminal_state(self) -> bool:
        """Whether the latest committed state is terminal, completed or not."""

        return self.is_terminal or (
            self.committed_position is not None
            and self.continuation_position != self.committed_position
        )

    @property
    def can_resume(self) -> bool:
        """Whether Session recovery may continue this Run in place."""

        return self.status is RunStatus.RESUMABLE

    @property
    def can_fork(self) -> bool:
        """Whether the Run exposes a durable committed fork boundary."""

        return self.committed_position is not None

    @property
    def can_continue(self) -> bool:
        """Whether an explicit fork can create a non-terminal continuation."""

        return self.continuation_position is not None

    @property
    def parent_run_id(self) -> str | None:
        """Return the immediate fork parent without hiding the exact cutoff."""

        return self.forked_from.run_id if self.forked_from is not None else None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe read projection without exposing a storage path."""

        return {
            "run_id": self.run_id,
            "lineage_id": self.lineage_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "latest_position": _position_to_dict(self.latest_position),
            "committed_position": _optional_position_to_dict(self.committed_position),
            "continuation_position": _optional_position_to_dict(
                self.continuation_position
            ),
            "forked_from": _optional_position_to_dict(self.forked_from),
            "parent_run_id": self.parent_run_id,
            "record_count": self.record_count,
            "agent_name": self.agent_name,
            "task": self.task,
            "stop_reason": self.stop_reason,
            "interrupted_at": _optional_position_to_dict(self.interrupted_at),
            "interruption_reason": self.interruption_reason,
            "is_terminal": self.is_terminal,
            "has_terminal_state": self.has_terminal_state,
            "can_resume": self.can_resume,
            "can_fork": self.can_fork,
            "can_continue": self.can_continue,
        }


class RunCatalog(Protocol):
    """Lease-free query contract for canonical persisted Runs."""

    async def inspect_run(self, run_id: str) -> RunHandle:
        """Return the current durable summary for one Run."""

        ...

    async def list_runs(self) -> tuple[RunHandle, ...]:
        """Return every discoverable Run in deterministic order."""

        ...

    async def lineage(self, run_id: str) -> tuple[RunHandle, ...]:
        """Return the validated root-to-Run ancestry."""

        ...

    async def children(self, run_id: str) -> tuple[RunHandle, ...]:
        """Return validated direct children for one Run."""

        ...


def _position_to_dict(position: JournalPosition) -> dict[str, Any]:
    return {
        "run_id": position.run_id,
        "seq": position.seq,
        "record_id": position.record_id,
    }


def _optional_position_to_dict(
    position: JournalPosition | None,
) -> dict[str, Any] | None:
    return _position_to_dict(position) if position is not None else None


__all__ = ["RunCatalog", "RunHandle", "RunNotFoundError", "RunStatus"]
