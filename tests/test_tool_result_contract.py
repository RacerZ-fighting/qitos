from __future__ import annotations

import pytest

from qitos.core.tool_result import ToolResult


@pytest.mark.parametrize(
    "status",
    [
        "error",
        "partial",
        "running",
        "skipped",
        "denied",
        "needs_input",
        "needs_approval",
        "timed_out",
        "cancelled",
    ],
)
def test_tool_result_preserves_each_canonical_non_success_status(
    status: str,
) -> None:
    result = ToolResult.from_value({"status": status, "message": "not complete"})

    assert result.status == status
    assert result.is_success is False
    assert result.output == {"message": "not complete"}
    if status in {"error", "denied", "timed_out", "cancelled"}:
        assert result.error == "not complete"
    else:
        assert result.error is None


@pytest.mark.parametrize(
    "noncanonical",
    [
        "failed",
        "validation_error",
        "needs_user_input",
        "approval_required",
        "timeout",
        "canceled",
        "output_limit_exceeded",
        "cancelling",
        "completed",
        "ok",
        "usable",
        "",
    ],
)
def test_tool_result_rejects_noncanonical_statuses(noncanonical: str) -> None:
    result = ToolResult.from_value(
        {"status": noncanonical, "message": "opaque", "value": 1}
    )

    assert result.status == "error"
    assert result.is_success is False
    assert result.output == {"message": "opaque", "value": 1}
    assert result.error == f"unknown tool result status: {noncanonical}"


def test_tool_result_treats_a_dict_without_lifecycle_status_as_output() -> None:
    payload = {"domain_outcome": "ready", "session_id": "s1"}

    result = ToolResult.from_value(payload)

    assert result.status == "success"
    assert result.output == payload


def test_tool_result_round_trip_preserves_domain_output() -> None:
    result = ToolResult(
        call_id="call-report",
        status="success",
        output={"path": "report.json", "content": "done"},
        metadata={"tool_name": "report"},
    )

    restored = ToolResult.from_value(result.to_dict())

    assert restored.status == result.status
    assert restored.call_id == result.call_id
    assert restored.output == result.output
    assert restored.error == result.error
    assert restored.metadata == result.metadata


def test_canonical_tool_result_round_trip_preserves_empty_error() -> None:
    payload = ToolResult(
        call_id="call-timeout",
        status="timed_out",
        output={"process_status": "running"},
        error="",
        metadata={"tool_name": "run_command"},
        model_output="process remains available",
    ).to_dict()

    restored = ToolResult.from_dict(payload)

    assert restored.to_dict() == payload


def test_tool_result_restores_legacy_payload_without_call_id() -> None:
    restored = ToolResult.from_value(
        {"status": "success", "output": "done", "metadata": {}}
    )

    assert restored.call_id is None


def test_tool_result_rejects_a_contradictory_success_error() -> None:
    result = ToolResult.from_value(
        {"status": "success", "output": "opaque", "error": "backend failed"}
    )

    assert result.status == "error"
    assert result.output == "opaque"
    assert result.error == "backend failed"
