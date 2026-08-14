"""Canonical external input delivered to an active Engine run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .child import ChildResult
    from .process import ProcessSnapshot


_PROCESS_TERMINAL_EVENT_MAX_CHARS = 8_000


@dataclass(frozen=True)
class RuntimeInput:
    """A small external event delivered before the next model request."""

    event_id: str
    kind: str
    correlation_id: str
    source: str
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        for name in ("event_id", "kind", "correlation_id", "source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"RuntimeInput.{name} must be a non-empty string")
        if not isinstance(self.payload, dict):
            raise TypeError("RuntimeInput.payload must be a dict")
        try:
            payload = json.loads(
                json.dumps(self.payload, ensure_ascii=False, allow_nan=False)
            )
        except (TypeError, ValueError) as exc:
            raise TypeError("RuntimeInput.payload must be JSON serializable") from exc
        object.__setattr__(self, "payload", payload)

    def to_dict(self) -> dict[str, Any]:
        """Return the provider-neutral event representation."""

        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "correlation_id": self.correlation_id,
            "source": self.source,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeInput":
        """Restore one event from its strict journal representation."""

        expected = {"event_id", "kind", "correlation_id", "source", "payload"}
        if set(value) != expected:
            raise ValueError("RuntimeInput fields are invalid")
        payload = value["payload"]
        if not isinstance(payload, Mapping):
            raise TypeError("RuntimeInput.payload must be a mapping")
        return cls(
            event_id=value["event_id"],
            kind=value["kind"],
            correlation_id=value["correlation_id"],
            source=value["source"],
            payload=dict(payload),
        )


def child_result_payload(result: ChildResult) -> dict[str, Any]:
    """Return the stable model-facing projection of one child result."""

    from .child import ChildResult, ChildStatus

    if not isinstance(result, ChildResult):
        raise TypeError("result must be a ChildResult")
    if result.status is ChildStatus.COMPLETED:
        execution_status = "success"
    elif result.status in {ChildStatus.PENDING, ChildStatus.RUNNING}:
        execution_status = "running"
    elif result.status in {ChildStatus.CANCELLED, ChildStatus.INTERRUPTED}:
        execution_status = "cancelled"
    elif result.conclusion.summary:
        execution_status = "partial"
    else:
        execution_status = "error"
    return {
        "status": execution_status,
        "child_status": result.status.value,
        "ready": result.ready,
        "handle": result.handle.to_dict(),
        "child_id": result.handle.child_id,
        "agent_type": result.request.agent_type,
        "name": result.request.name,
        "description": result.request.description,
        "output": result.conclusion.summary,
        "conclusion": result.conclusion.to_dict(),
        "error": result.error,
        "steps": result.steps,
        "total_tokens": result.total_tokens,
        "total_cost_usd": result.total_cost_usd,
        "usage_complete": result.usage_complete,
        "cost_complete": result.cost_complete,
        "elapsed_seconds": result.elapsed_seconds,
        "stop_reason": result.status.value,
        "run_id": result.child_run_id,
    }


def child_terminal_runtime_input(result: ChildResult) -> RuntimeInput:
    """Derive the idempotent parent input for one canonical child terminal."""

    payload = child_result_payload(result)
    if not result.ready:
        raise ValueError("child result must be terminal")
    child_id = result.handle.child_id
    return RuntimeInput(
        event_id=f"{child_id}:terminal",
        kind="agent.child.completed",
        correlation_id=child_id,
        source="qitos.agent",
        payload=payload,
    )


def process_terminal_payload(snapshot: ProcessSnapshot) -> dict[str, Any]:
    """Return the bounded stable projection of one process terminal."""

    from .process import ProcessSnapshot

    if not isinstance(snapshot, ProcessSnapshot):
        raise TypeError("snapshot must be a ProcessSnapshot")
    if not snapshot.terminal:
        raise ValueError("process snapshot must be terminal")
    content = snapshot.output.content
    notification_truncated = len(content) > _PROCESS_TERMINAL_EVENT_MAX_CHARS
    if notification_truncated:
        content = (
            content[:_PROCESS_TERMINAL_EVENT_MAX_CHARS] + "\n... [truncated]"
        )
    return {
        "handle": snapshot.handle.to_dict(),
        "process_id": snapshot.handle.process_id,
        "owner_run_id": snapshot.handle.owner_run_id,
        "status": snapshot.status.value,
        "terminal": snapshot.terminal,
        "ended_at": snapshot.ended_at,
        "exit_code": snapshot.exit_code,
        "error": snapshot.error,
        "output": {
            "content": content,
            "next_cursor": snapshot.output.next_cursor,
            "total_bytes": snapshot.output.total_bytes,
            "truncated": snapshot.output.truncated or notification_truncated,
            "notification_truncated": notification_truncated,
            "log_path": snapshot.output.log_path,
        },
    }


def process_terminal_runtime_input(snapshot: ProcessSnapshot) -> RuntimeInput:
    """Derive the idempotent Run input for one canonical process terminal."""

    payload = process_terminal_payload(snapshot)
    process_id = snapshot.handle.process_id
    return RuntimeInput(
        event_id=f"{process_id}:terminal",
        kind="process.completed",
        correlation_id=process_id,
        source="qitos.process",
        payload=payload,
    )


__all__ = [
    "RuntimeInput",
    "child_result_payload",
    "child_terminal_runtime_input",
    "process_terminal_payload",
    "process_terminal_runtime_input",
]
