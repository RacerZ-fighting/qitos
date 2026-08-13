"""Stable contracts for Run-owned managed processes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer or null")
    return value


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

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProcessOutput":
        required = {
            "content",
            "cursor",
            "next_cursor",
            "total_bytes",
            "omitted_bytes",
            "truncated",
            "log_path",
        }
        if set(value) != required:
            raise ValueError("process output fields are invalid")
        if not isinstance(value["content"], str):
            raise ValueError("process output content is invalid")
        if not isinstance(value["truncated"], bool):
            raise ValueError("process output truncated flag is invalid")
        if not isinstance(value["log_path"], str):
            raise ValueError("process output log_path is invalid")
        return cls(
            content=value["content"],
            cursor=_non_negative_int(value["cursor"], "cursor"),
            next_cursor=_non_negative_int(value["next_cursor"], "next_cursor"),
            total_bytes=_non_negative_int(value["total_bytes"], "total_bytes"),
            omitted_bytes=_non_negative_int(
                value["omitted_bytes"], "omitted_bytes"
            ),
            truncated=value["truncated"],
            log_path=value["log_path"],
        )


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

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProcessSnapshot":
        required = {
            "handle",
            "process_id",
            "owner_run_id",
            "status",
            "terminal",
            "command",
            "cwd",
            "pid",
            "tty",
            "started_at",
            "ended_at",
            "exit_code",
            "output",
            "error",
        }
        if set(value) != required:
            raise ValueError("process snapshot fields are invalid")
        handle_value = value["handle"]
        output_value = value["output"]
        if not isinstance(handle_value, Mapping):
            raise ValueError("process snapshot handle is invalid")
        if not isinstance(output_value, Mapping):
            raise ValueError("process snapshot output is invalid")
        handle = ProcessHandle.from_dict(handle_value)
        if value["process_id"] != handle.process_id:
            raise ValueError("process snapshot process_id is inconsistent")
        if value["owner_run_id"] != handle.owner_run_id:
            raise ValueError("process snapshot owner_run_id is inconsistent")
        for name in ("terminal", "tty"):
            if not isinstance(value[name], bool):
                raise ValueError(f"process snapshot {name} flag is invalid")
        for name in ("command", "cwd", "started_at"):
            if not isinstance(value[name], str) or not value[name]:
                raise ValueError(f"process snapshot {name} is invalid")
        if value["ended_at"] is not None and not isinstance(value["ended_at"], str):
            raise ValueError("process snapshot ended_at is invalid")
        if value["error"] is not None and not isinstance(value["error"], str):
            raise ValueError("process snapshot error is invalid")
        status = ProcessStatus(str(value["status"]))
        snapshot = cls(
            handle=handle,
            status=status,
            command=value["command"],
            cwd=value["cwd"],
            pid=_optional_int(value["pid"], "pid"),
            tty=value["tty"],
            started_at=value["started_at"],
            ended_at=value["ended_at"],
            exit_code=_optional_int(value["exit_code"], "exit_code"),
            output=ProcessOutput.from_dict(output_value),
            error=value["error"],
        )
        if value["terminal"] != snapshot.terminal:
            raise ValueError("process snapshot terminal flag is inconsistent")
        return snapshot


__all__ = [
    "ProcessError",
    "ProcessHandle",
    "ProcessNotFoundError",
    "ProcessOutput",
    "ProcessPersistenceError",
    "ProcessSnapshot",
    "ProcessStatus",
]
