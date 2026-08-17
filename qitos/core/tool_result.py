"""Canonical tool-result contract for ToolCall/ToolResult transactions."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Literal, Mapping, Sequence, TypeAlias, cast

from ._freeze import thaw_deep
from .artifact import ArtifactRef
from .model_response import ModelUsage


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

_KNOWN_STATUSES = frozenset(
    (
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
    )
)


def _freeze_json(value: Any, *, field_name: str) -> Any:
    """Validate and deeply freeze one strict JSON value."""

    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"tool result {field_name} keys must be strings")
        return MappingProxyType(
            {
                key: _freeze_json(item, field_name=field_name)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, field_name=field_name) for item in value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"tool result {field_name} numbers must be finite")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"tool result {field_name} must contain JSON-compatible values")


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Normalized, deeply immutable tool execution result.

    ``usage`` carries typed token/cost accounting for work the Tool itself
    performed (for example a child Agent run); ``added_tool_names`` lists
    the names of Tools this result activated, available from this transcript
    point onward. Both are durable facts and ride the canonical codec; they
    are not provider wire data and stay out of :meth:`to_model_dict`.
    """

    status: ToolResultStatus = "success"
    output: Any = None
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifacts: tuple[ArtifactRef, ...] = ()
    model_output: str | None = None
    call_id: str | None = None
    usage: ModelUsage | None = None
    added_tool_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, str):
            raise TypeError("tool result status must be text")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("tool result error must be text or null")
        if self.model_output is not None and not isinstance(self.model_output, str):
            raise TypeError("tool result model_output must be text or null")
        raw_status = self.status.strip()
        if raw_status in _KNOWN_STATUSES:
            status = cast(ToolResultStatus, raw_status)
            error = self.error
        else:
            status = cast(ToolResultStatus, "error")
            error = self.error
            if error in (None, ""):
                error = f"unknown tool result status: {raw_status}"
        if status == "success" and error not in (None, ""):
            status = cast(ToolResultStatus, "error")
        if self.call_id is not None and (
            not isinstance(self.call_id, str) or not self.call_id
        ):
            raise ValueError("tool result call_id must be non-empty text or null")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("tool result metadata must be a mapping")
        if not isinstance(self.artifacts, tuple) or not all(
            isinstance(item, ArtifactRef) for item in self.artifacts
        ):
            raise TypeError("tool result artifacts must be a tuple of ArtifactRef")
        if self.usage is not None and not isinstance(self.usage, ModelUsage):
            raise TypeError("tool result usage must be a ModelUsage or None")
        if not isinstance(self.added_tool_names, tuple) or not all(
            isinstance(name, str) and name.strip() for name in self.added_tool_names
        ):
            raise TypeError("tool result added_tool_names must contain non-empty strings")
        if len(self.added_tool_names) != len(set(self.added_tool_names)):
            raise ValueError("tool result added_tool_names must be unique")

        object.__setattr__(self, "status", status)
        object.__setattr__(self, "error", error)
        object.__setattr__(
            self, "output", _freeze_json(self.output, field_name="output")
        )
        object.__setattr__(
            self, "metadata", _freeze_json(self.metadata, field_name="metadata")
        )

    @property
    def is_success(self) -> bool:
        return self.status == "success"

    @property
    def text(self) -> str:
        if isinstance(self.output, str):
            return self.output
        try:
            return json.dumps(thaw_deep(self.output), ensure_ascii=False, default=str)
        except Exception:
            return str(self.output)

    @property
    def model_visible_output(self) -> Any:
        if self.model_output is not None:
            return self.model_output
        if isinstance(self.output, Mapping):
            summary = self.output.get("model_summary")
            if isinstance(summary, str) and summary.strip():
                return summary.strip()
        return self.output

    def frozen(self) -> "ToolResult":
        """Return this already deeply immutable result.

        Construction freezes every result before it can cross journal, event
        or Message boundaries, so a listener cannot make durable records
        contradict each other. This method remains as the explicit boundary
        marker used by the executor; serialization thaws plain copies.
        """

        return self

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "status": str(self.status),
            "output": thaw_deep(self.output),
            "error": self.error,
            "metadata": thaw_deep(self.metadata),
        }
        if self.artifacts:
            payload["artifacts"] = [item.to_dict() for item in self.artifacts]
        if self.model_output is not None:
            payload["model_output"] = self.model_output
        if self.call_id is not None:
            payload["call_id"] = self.call_id
        if self.usage is not None:
            payload["usage"] = self.usage.to_dict()
        if self.added_tool_names:
            payload["added_tool_names"] = list(self.added_tool_names)
        return payload

    def to_model_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "status": str(self.status),
            "output": thaw_deep(self.model_visible_output),
            "error": self.error,
            "metadata": thaw_deep(self.metadata),
        }
        if self.artifacts:
            payload["artifacts"] = [item.to_dict() for item in self.artifacts]
        if self.call_id is not None:
            payload["call_id"] = self.call_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolResult":
        """Decode one canonical result previously produced by :meth:`to_dict`.

        Unlike :meth:`from_value`, this decoder does not reinterpret empty fields or
        domain payloads. Journal replay relies on that exact wire round trip.
        """
        required = {"status", "output", "error", "metadata"}
        optional = {"artifacts", "model_output", "call_id", "usage", "added_tool_names"}
        fields = set(payload)
        if not required.issubset(fields) or not fields.issubset(required | optional):
            raise ValueError("tool result fields are invalid")

        raw_status = payload["status"]
        if not isinstance(raw_status, str):
            raise ValueError("tool result status must be text")
        raw_error = payload["error"]
        if raw_error is not None and not isinstance(raw_error, str):
            raise ValueError("tool result error must be text or null")
        raw_metadata = payload["metadata"]
        if not isinstance(raw_metadata, dict):
            raise ValueError("tool result metadata must be an object")

        raw_artifacts = payload.get("artifacts", ())
        if not isinstance(raw_artifacts, Sequence) or isinstance(
            raw_artifacts, (str, bytes)
        ):
            raise ValueError("tool result artifacts must be a sequence")
        artifacts = tuple(
            item
            if isinstance(item, ArtifactRef)
            else ArtifactRef.from_dict(dict(item))
            for item in raw_artifacts
            if isinstance(item, (ArtifactRef, Mapping))
        )
        if len(artifacts) != len(raw_artifacts):
            raise ValueError("tool result artifacts must contain references")

        raw_model_output = payload.get("model_output")
        if raw_model_output is not None and not isinstance(raw_model_output, str):
            raise ValueError("tool result model_output must be text or null")
        raw_call_id = payload.get("call_id")
        if raw_call_id is not None and (
            not isinstance(raw_call_id, str) or not raw_call_id
        ):
            raise ValueError("tool result call_id must be non-empty text or null")
        raw_usage = payload.get("usage")
        if raw_usage is not None and not isinstance(raw_usage, dict):
            raise ValueError("tool result usage must be an object or null")
        raw_added_tool_names = payload.get("added_tool_names", ())
        if not isinstance(raw_added_tool_names, Sequence) or isinstance(
            raw_added_tool_names, (str, bytes)
        ):
            raise ValueError("tool result added_tool_names must be a sequence")

        try:
            result = cls(
                status=cast(ToolResultStatus, raw_status),
                output=payload["output"],
                error=raw_error,
                metadata=dict(raw_metadata),
                artifacts=artifacts,
                model_output=raw_model_output,
                call_id=raw_call_id,
                usage=(
                    ModelUsage.from_mapping(raw_usage)
                    if raw_usage is not None
                    else None
                ),
                added_tool_names=tuple(raw_added_tool_names),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("tool result payload is invalid") from exc
        if result.to_dict() != dict(payload):
            raise ValueError("tool result payload is not canonical")
        return result

    @classmethod
    def from_value(cls, payload: Any) -> "ToolResult":
        if isinstance(payload, ToolResult):
            return payload
        # Plain handler values are successful domain output. Lifecycle status
        # is explicit and typed: only a ToolResult can set it. In particular,
        # a domain mapping with a key named ``status`` must not be reinterpreted
        # as executor control state.
        return cls(status="success", output=payload)


__all__ = ["ToolResult", "ToolResultStatus"]
