"""Canonical external input delivered to an active Engine run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any


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


__all__ = ["RuntimeInput"]
