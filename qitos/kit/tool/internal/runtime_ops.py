"""Helpers for selecting environment-provided tool capabilities."""

from __future__ import annotations

from typing import Any, Mapping, TypeVar


OpsT = TypeVar("OpsT")


def select_runtime_ops(
    runtime_context: Mapping[str, Any] | None,
    group: str,
    fallback: OpsT | None,
) -> OpsT:
    """Select one capability without crossing an active environment boundary.

    Direct tool calls have no runtime context and retain the toolset's local
    fallback.  Once an engine supplies an environment context, a missing group
    is an integration error; silently using the controller fallback would run
    the tool in the wrong environment.
    """

    if runtime_context is None and fallback is not None:
        return fallback
    if runtime_context is None:
        raise RuntimeError(f"tool requires runtime ops group: {group}")

    raw_ops = runtime_context.get("ops")
    if isinstance(raw_ops, Mapping):
        provider = raw_ops.get(group)
        if provider is not None:
            return provider  # type: ignore[return-value]

    if runtime_context.get("env") is not None:
        raise RuntimeError(f"runtime environment does not provide ops group: {group}")
    if fallback is None:
        raise RuntimeError(f"tool requires runtime ops group: {group}")
    return fallback


__all__ = ["select_runtime_ops"]
