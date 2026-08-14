"""Credential redaction shared by persisted QitOS output surfaces."""

from __future__ import annotations

from collections.abc import Mapping, Set
from typing import Any

REDACTED_FIELDS: frozenset[str] = frozenset(
    {
        "tool_args",
        "input_content",
        "output_content",
        "model_response",
        "api_key",
        "authorization",
        "token",
        "secret",
        "password",
        "access_token",
        "refresh_token",
        "private_key",
        "credentials",
    }
)

REDACTED_MARKER = "__redacted__"


def redact_mapping(
    data: Mapping[str, Any],
    *,
    fields: Set[str] = REDACTED_FIELDS,
) -> dict[str, Any]:
    """Return a recursively copied mapping with sensitive fields redacted."""

    redacted: dict[str, Any] = {}
    for key, value in data.items():
        if key in fields:
            redacted[key] = REDACTED_MARKER
        else:
            redacted[key] = _redact_value(value, fields=fields)
    return redacted


def _redact_value(value: Any, *, fields: Set[str]) -> Any:
    if isinstance(value, Mapping):
        return redact_mapping(value, fields=fields)
    if isinstance(value, list):
        return [_redact_value(item, fields=fields) for item in value]
    return value


__all__ = ["REDACTED_FIELDS", "REDACTED_MARKER", "redact_mapping"]
