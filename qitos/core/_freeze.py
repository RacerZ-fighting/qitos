"""Deep freeze/thaw helpers for values crossing durable or listener boundaries.

``freeze_deep`` converts mappings and lists/tuples into immutable
``MappingProxyType``/``tuple`` structures so a value handed to listeners or
held by a durable Message cannot be mutated after the fact. Other objects
pass through unchanged (callers are expected to keep JSON-compatible data).

``thaw_deep`` reverses the container conversion so frozen values serialize
through plain ``json.dumps`` and compare equal to their pre-freeze shape.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping


def freeze_deep(value: Any) -> Any:
    """Return a deeply immutable view of one value (containers only)."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze_deep(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_deep(item) for item in value)
    return value


def thaw_deep(value: Any) -> Any:
    """Return a plain mutable copy of one frozen value (containers only)."""

    if isinstance(value, Mapping):
        return {key: thaw_deep(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_deep(item) for item in value]
    if isinstance(value, list):
        return [thaw_deep(item) for item in value]
    return value


__all__ = ["freeze_deep", "thaw_deep"]
