"""Canonical external input delivered to an active Agent run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .subagent import SubagentResult
    from .process import ProcessSnapshot


_PROCESS_TERMINAL_EVENT_MAX_CHARS = 8_000


def _bounded_head_tail(content: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(content) <= max_chars:
        return content
    marker = "\n... [middle omitted from runtime notification] ...\n"
    content_budget = max(0, max_chars - len(marker))
    head_chars = content_budget // 4
    tail_chars = content_budget - head_chars
    return (
        content[:head_chars]
        + marker
        + (content[-tail_chars:] if tail_chars else "")
    )[:max_chars]


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
        kind = value["kind"]
        migrated_payload = dict(payload)
        if isinstance(kind, str) and kind in {
            "agent.child.completed",
            "agent.child.snapshot",
        }:
            kind = kind.replace("agent.child.", "agent.subagent.", 1)
            _migrate_legacy_subagent_payload(migrated_payload)
        return cls(
            event_id=value["event_id"],
            kind=kind,
            correlation_id=value["correlation_id"],
            source=value["source"],
            payload=migrated_payload,
        )


def _migrate_legacy_subagent_payload(payload: dict[str, Any]) -> None:
    for old, new in (
        ("child_status", "subagent_status"),
        ("child_id", "subagent_id"),
        ("run_id", "subagent_run_id"),
    ):
        if old not in payload:
            continue
        if new in payload and payload[new] != payload[old]:
            raise ValueError(f"legacy {old} conflicts with {new}")
        payload[new] = payload.pop(old)
    handle = payload.get("handle")
    if not isinstance(handle, Mapping):
        return
    migrated_handle = dict(handle)
    if "child_id" in migrated_handle:
        if (
            "subagent_id" in migrated_handle
            and migrated_handle["subagent_id"] != migrated_handle["child_id"]
        ):
            raise ValueError("legacy handle child_id conflicts with subagent_id")
        migrated_handle["subagent_id"] = migrated_handle.pop("child_id")
    payload["handle"] = migrated_handle


def subagent_result_payload(result: SubagentResult) -> dict[str, Any]:
    """Return the stable model-facing projection of one Subagent result.

    Token and cost accounting is not part of this untyped projection: at
    the Tool boundary it rides the typed ``ToolResult.usage`` carrier, and
    the durable ``subagent.terminal`` journal record keeps the full
    ``SubagentResult`` facts. Completeness flags stay here so the model can
    tell whether the accounting it cannot see is known to be partial.
    """

    from .subagent import SubagentResult, SubagentStatus

    if not isinstance(result, SubagentResult):
        raise TypeError("result must be a SubagentResult")
    if result.status is SubagentStatus.COMPLETED:
        execution_status = "success"
    elif result.status in {SubagentStatus.PENDING, SubagentStatus.RUNNING}:
        execution_status = "running"
    elif result.status in {SubagentStatus.CANCELLED, SubagentStatus.INTERRUPTED}:
        execution_status = "cancelled"
    elif result.conclusion.summary:
        execution_status = "partial"
    else:
        execution_status = "error"
    return {
        "status": execution_status,
        "subagent_status": result.status.value,
        "ready": result.ready,
        "handle": result.handle.to_dict(),
        "subagent_id": result.handle.subagent_id,
        "agent_type": result.request.agent_type,
        "name": result.request.name,
        "description": result.request.description,
        "output": result.conclusion.summary,
        "conclusion": result.conclusion.to_dict(),
        "error": result.error,
        "steps": result.steps,
        "usage_complete": result.usage_complete,
        "cost_complete": result.cost_complete,
        "elapsed_seconds": result.elapsed_seconds,
        "stop_reason": result.status.value,
        "subagent_run_id": result.subagent_run_id,
    }


def subagent_terminal_runtime_input(result: SubagentResult) -> RuntimeInput:
    """Derive the idempotent parent input for one canonical Subagent terminal."""

    payload = subagent_result_payload(result)
    if not result.ready:
        raise ValueError("Subagent result must be terminal")
    subagent_id = result.handle.subagent_id
    return RuntimeInput(
        event_id=f"{subagent_id}:terminal",
        kind="agent.subagent.completed",
        correlation_id=subagent_id,
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
        content = _bounded_head_tail(
            content,
            _PROCESS_TERMINAL_EVENT_MAX_CHARS,
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
    "subagent_result_payload",
    "subagent_terminal_runtime_input",
    "process_terminal_payload",
    "process_terminal_runtime_input",
]
