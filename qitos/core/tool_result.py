"""Canonical tool-result contract used by Engine observations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, TypeAlias, cast

from .action import ActionStatus


ToolResultStatus: TypeAlias = Literal[
    "success",
    "partial",
    "running",
    "error",
    "skipped",
    "denied",
    "needs_input",
    "needs_approval",
    "timed_out",
    "cancelled",
]

_MESSAGE_ERROR_STATUSES = frozenset(
    {"error", "denied", "timed_out", "cancelled"}
)


@dataclass
class ToolResult:
    """Normalized tool execution result."""

    status: ToolResultStatus = "success"
    output: Any = None
    error: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raw_status = str(self.status).strip()
        try:
            self.status = cast(ToolResultStatus, ActionStatus(raw_status).value)
        except ValueError:
            self.status = "error"
            if self.error in (None, ""):
                self.error = f"unknown tool result status: {raw_status}"
        if self.status == "success" and self.error not in (None, ""):
            self.status = "error"

    @property
    def is_success(self) -> bool:
        return self.status == "success"

    @property
    def text(self) -> str:
        if isinstance(self.output, str):
            return self.output
        try:
            return json.dumps(self.output, ensure_ascii=False, default=str)
        except Exception:
            return str(self.output)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": str(self.status),
            "output": self.output,
            "error": self.error,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_value(cls, payload: Any) -> "ToolResult":
        if isinstance(payload, ToolResult):
            return cls(
                status=payload.status,
                output=payload.output,
                error=payload.error,
                metadata=dict(payload.metadata),
            )
        if isinstance(payload, dict):
            if "status" not in payload:
                return cls(status="success", output=payload)
            raw_status = str(payload.get("status")).strip()
            raw_error = payload.get("error")
            error: str | None = None
            if raw_error not in (None, ""):
                error = str(raw_error)
            elif raw_status in _MESSAGE_ERROR_STATUSES:
                message = payload.get("message")
                if message not in (None, ""):
                    error = str(message)

            domain_payload = {
                str(key): value
                for key, value in payload.items()
                if key not in {"status", "error", "metadata"}
            }
            output = (
                domain_payload["output"]
                if set(domain_payload) == {"output"}
                else domain_payload
            )
            return cls(
                status=cast(ToolResultStatus, raw_status),
                output=output,
                error=error,
                metadata=(
                    dict(payload.get("metadata", {}))
                    if isinstance(payload.get("metadata"), dict)
                    else {}
                ),
            )
        if isinstance(payload, str):
            return cls(status="success", output=payload)
        return cls(status="success", output=payload)


__all__ = ["ToolResult", "ToolResultStatus"]
