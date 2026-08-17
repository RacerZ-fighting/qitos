"""Explicit Tool lifecycle results for kit-owned handlers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qitos.core.tool_result import ToolResult, ToolResultStatus


def tool_result(
    payload: Mapping[str, Any],
    *,
    status: ToolResultStatus,
    error: str | None = None,
) -> ToolResult:
    """Preserve a domain payload while explicitly assigning Tool lifecycle.

    Callers choose ``status`` at the concrete Tool handler boundary. This
    helper deliberately does not inspect a payload's ``status`` field: plain
    mappings remain domain output unless their handler opts into a typed
    lifecycle result here.
    """

    output = dict(payload)
    resolved_error = error
    if resolved_error is None:
        raw_error = output.get("error")
        if raw_error in (None, ""):
            raw_error = output.get("message")
        if raw_error not in (None, ""):
            resolved_error = str(raw_error)
    raw_model_output = output.get("model_output")
    model_output = raw_model_output if isinstance(raw_model_output, str) else None
    return ToolResult(
        status=status,
        output=output,
        error=resolved_error,
        model_output=model_output,
    )


def error_result(payload: Mapping[str, Any]) -> ToolResult:
    """Return one explicit error result for a concrete Kit Tool boundary."""

    return tool_result(payload, status="error")


__all__ = ["error_result", "tool_result"]
