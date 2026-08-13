"""Stable contracts for Run-owned managed processes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class ProcessStatus(str, Enum):
    """Lifecycle states observable through the managed process API."""

    RUNNING = "running"
    EXITED = "exited"
    FAILED = "failed"
    TERMINATED = "terminated"
    LOST = "lost"


class ProcessError(RuntimeError):
    """Base error raised by managed process capabilities."""


class ProcessNotFoundError(ProcessError):
    """Raised when a process handle is unknown to the current runtime."""


class ProcessPersistenceError(ProcessError):
    """Raised when a process lifecycle transition cannot be journaled."""


@dataclass(frozen=True, slots=True)
class ProcessHandle:
    """Opaque identity for one process owned by a QitOS Run."""

    process_id: str
    owner_run_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.process_id, str) or not self.process_id.strip():
            raise ValueError("process_id must be a non-empty string")
        if not isinstance(self.owner_run_id, str) or not self.owner_run_id.strip():
            raise ValueError("owner_run_id must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        return {
            "process_id": self.process_id,
            "owner_run_id": self.owner_run_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProcessHandle":
        if set(value) != {"process_id", "owner_run_id"}:
            raise ValueError("process handle fields are invalid")
        return cls(
            process_id=str(value["process_id"]),
            owner_run_id=str(value["owner_run_id"]),
        )


@dataclass(frozen=True, slots=True)
class ProcessOutput:
    """Bounded incremental UTF-8 view of one process transcript."""

    content: str
    cursor: int
    next_cursor: int
    total_bytes: int
    omitted_bytes: int
    truncated: bool
    log_path: str

    def __post_init__(self) -> None:
        for name in ("cursor", "next_cursor", "total_bytes", "omitted_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.next_cursor > self.total_bytes:
            raise ValueError("next_cursor cannot exceed total_bytes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "cursor": self.cursor,
            "next_cursor": self.next_cursor,
            "total_bytes": self.total_bytes,
            "omitted_bytes": self.omitted_bytes,
            "truncated": self.truncated,
            "log_path": self.log_path,
        }


@dataclass(frozen=True, slots=True)
class ProcessSnapshot:
    """One immutable observation of a managed process."""

    handle: ProcessHandle
    status: ProcessStatus
    command: str
    cwd: str
    pid: int | None
    tty: bool
    started_at: str
    ended_at: str | None
    exit_code: int | None
    output: ProcessOutput
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ProcessStatus):
            raise TypeError("status must be a ProcessStatus")
        if not isinstance(self.command, str) or not self.command.strip():
            raise ValueError("command must be a non-empty string")
        if not isinstance(self.cwd, str) or not self.cwd:
            raise ValueError("cwd must be a non-empty string")

    @property
    def terminal(self) -> bool:
        return self.status is not ProcessStatus.RUNNING

    def to_dict(self) -> dict[str, Any]:
        return {
            "handle": self.handle.to_dict(),
            "process_id": self.handle.process_id,
            "owner_run_id": self.handle.owner_run_id,
            "status": self.status.value,
            "terminal": self.terminal,
            "command": self.command,
            "cwd": self.cwd,
            "pid": self.pid,
            "tty": self.tty,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "exit_code": self.exit_code,
            "output": self.output.to_dict(),
            "error": self.error,
        }


__all__ = [
    "ProcessError",
    "ProcessHandle",
    "ProcessNotFoundError",
    "ProcessOutput",
    "ProcessPersistenceError",
    "ProcessSnapshot",
    "ProcessStatus",
]
