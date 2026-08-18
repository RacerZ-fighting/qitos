from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qitos.core.model_response import ModelUsage
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
    error = (
        "not complete"
        if status in {"error", "denied", "timed_out", "cancelled"}
        else None
    )
    result = ToolResult(
        status=status,  # type: ignore[arg-type]
        output={"message": "not complete"},
        error=error,
    )

    assert result.status == status
    assert result.is_success is False
    assert result.output == {"message": "not complete"}
    assert result.error == error


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
    result = ToolResult(
        status=noncanonical,  # type: ignore[arg-type]
        output={"message": "opaque", "value": 1},
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

    restored = ToolResult.from_dict(result.to_dict())

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


def test_plain_mapping_status_is_domain_output_not_lifecycle_control() -> None:
    payload = {"status": "running", "handle": "process-1"}

    restored = ToolResult.from_value(payload)

    assert restored.status == "success"
    assert restored.output == payload
    assert restored.call_id is None


def test_tool_result_rejects_a_contradictory_success_error() -> None:
    result = ToolResult(
        status="success", output="opaque", error="backend failed"
    )

    assert result.status == "error"
    assert result.output == "opaque"
    assert result.error == "backend failed"


def test_tool_result_is_top_level_and_deeply_immutable() -> None:
    output = {"rows": [{"value": 1}]}
    metadata = {"nested": {"attempt": 1}}
    result = ToolResult(output=output, metadata=metadata)

    output["rows"][0]["value"] = 2
    metadata["nested"]["attempt"] = 2

    assert result.to_dict()["output"] == {"rows": [{"value": 1}]}
    assert result.to_dict()["metadata"] == {"nested": {"attempt": 1}}
    with pytest.raises(FrozenInstanceError):
        result.status = "error"  # type: ignore[misc]
    with pytest.raises(TypeError):
        result.output["rows"][0]["value"] = 3  # type: ignore[index]
    with pytest.raises(TypeError):
        result.metadata["nested"]["attempt"] = 3  # type: ignore[index]


def test_tool_result_from_dict_requires_canonical_fields_and_types() -> None:
    payload = ToolResult(output="ok").to_dict()
    payload.pop("metadata")
    with pytest.raises(ValueError, match="fields"):
        ToolResult.from_dict(payload)

    payload = ToolResult(output="ok").to_dict()
    payload["metadata"] = None
    with pytest.raises(ValueError, match="metadata"):
        ToolResult.from_dict(payload)


def test_tool_result_rejects_non_text_error_and_model_output() -> None:
    with pytest.raises(TypeError, match="error"):
        ToolResult(error=123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="model_output"):
        ToolResult(model_output=456)  # type: ignore[arg-type]


def test_tool_result_rejects_mutable_non_json_output() -> None:
    with pytest.raises(TypeError, match="JSON-compatible"):
        ToolResult(output=bytearray(b"mutable"))


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
def test_tool_result_rejects_non_finite_json_numbers(number: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        ToolResult(output={"value": number})
    with pytest.raises(ValueError, match="finite"):
        ToolResult(metadata={"value": number})


# ── typed usage and added Tool names ─────────────────────────────────────────


def _usage() -> ModelUsage:
    return ModelUsage.from_mapping({"total_tokens": 11, "cost_usd": 0.0025})


def test_tool_result_usage_and_added_names_round_trip() -> None:
    result = ToolResult(
        output={"subagent_status": "completed"},
        usage=_usage(),
        added_tool_names=("skill_tool", "mcp_tool"),
    )

    payload = result.to_dict()
    assert payload["usage"] == {"total_tokens": 11, "cost_usd": 0.0025}
    assert payload["added_tool_names"] == ["skill_tool", "mcp_tool"]

    restored = ToolResult.from_dict(payload)
    assert restored.to_dict() == payload
    assert restored.usage is not None
    assert restored.usage.total_tokens == 11
    assert restored.usage["cost_usd"] == 0.0025
    assert restored.added_tool_names == ("skill_tool", "mcp_tool")


def test_tool_result_omits_unset_usage_and_added_names_in_codec() -> None:
    payload = ToolResult(output="ok").to_dict()

    assert "usage" not in payload
    assert "added_tool_names" not in payload
    restored = ToolResult.from_dict(payload)
    assert restored.usage is None
    assert restored.added_tool_names == ()


def test_tool_result_usage_and_added_names_stay_out_of_the_model_projection() -> None:
    result = ToolResult(
        output="ok",
        usage=_usage(),
        added_tool_names=("skill_tool",),
    )

    projected = result.to_model_dict()
    assert "usage" not in projected
    assert "added_tool_names" not in projected


def test_tool_result_rejects_untyped_usage() -> None:
    with pytest.raises(TypeError, match="ModelUsage"):
        ToolResult(output="ok", usage={"total_tokens": 1})  # type: ignore[arg-type]


def test_tool_result_rejects_invalid_added_tool_names() -> None:
    with pytest.raises(TypeError, match="added_tool_names"):
        ToolResult(output="ok", added_tool_names=["a"])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="added_tool_names"):
        ToolResult(output="ok", added_tool_names=("a", ""))
    with pytest.raises(TypeError, match="added_tool_names"):
        ToolResult(output="ok", added_tool_names=("a", "  "))
    with pytest.raises(ValueError, match="unique"):
        ToolResult(output="ok", added_tool_names=("a", "a"))


def test_tool_result_from_dict_fails_closed_on_tampered_new_fields() -> None:
    payload = ToolResult(
        output="ok", usage=_usage(), added_tool_names=("a",)
    ).to_dict()

    tampered_usage = {**payload, "usage": {"total_tokens": -3}}
    with pytest.raises(ValueError):
        ToolResult.from_dict(tampered_usage)

    with pytest.raises(ValueError, match="usage"):
        ToolResult.from_dict({**payload, "usage": "total_tokens"})

    with pytest.raises(ValueError, match="added_tool_names"):
        ToolResult.from_dict({**payload, "added_tool_names": "a"})

    with pytest.raises(ValueError):
        ToolResult.from_dict({**payload, "added_tool_names": ["a", ""]})

    with pytest.raises(ValueError):
        ToolResult.from_dict({**payload, "added_tool_names": ["a", "a"]})

    with pytest.raises(ValueError, match="fields"):
        ToolResult.from_dict({**payload, "unknown_fact": 1})
