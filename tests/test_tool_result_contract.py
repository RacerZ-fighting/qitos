from __future__ import annotations

from typing import Any

import pytest

from qitos import tool
from qitos.core.action import Action, ActionStatus
from qitos.core.artifact import ArtifactRef
from qitos.core.tool import BaseTool, ToolSpec
from qitos.core.tool_registry import ToolRegistry
from qitos.core.tool_result import ToolResult
from qitos.engine.action_executor import ActionExecutor


class _ResultTool(BaseTool):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__(ToolSpec(name="result", description="return a fixed result"))
        self._payload = payload

    def execute(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = args, runtime_context
        return dict(self._payload)


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
        status="success",
        output={"path": "report.json", "content": "done"},
        metadata={"tool_name": "report"},
    )

    restored = ToolResult.from_value(result.to_dict())

    assert restored.status == result.status
    assert restored.output == result.output
    assert restored.error == result.error
    assert restored.metadata == result.metadata


def test_action_executor_preserves_typed_artifact_projection() -> None:
    content = "evidence"
    artifact = ArtifactRef(
        artifact_id="run:step:call",
        path="artifacts/evidence.md",
        media_type="text/markdown",
        size_bytes=len(content.encode("utf-8")),
        sha256="a" * 64,
    )

    @tool(name="artifact_result")
    def artifact_result() -> ToolResult:
        return ToolResult(
            output=content,
            artifacts=(artifact,),
            model_output=artifact.path,
        )

    result = ActionExecutor(ToolRegistry().register(artifact_result)).execute(
        [Action(name="artifact_result")]
    )[0]

    assert result.output == content
    assert result.artifacts == (artifact,)
    assert result.model_output == artifact.path


def test_tool_result_rejects_a_contradictory_success_error() -> None:
    result = ToolResult.from_value(
        {"status": "success", "output": "opaque", "error": "backend failed"}
    )

    assert result.status == "error"
    assert result.output == "opaque"
    assert result.error == "backend failed"


@pytest.mark.parametrize(
    ("reported", "action_status"),
    [
        ("success", ActionStatus.SUCCESS),
        ("partial", ActionStatus.PARTIAL),
        ("running", ActionStatus.RUNNING),
        ("error", ActionStatus.ERROR),
        ("skipped", ActionStatus.SKIPPED),
        ("denied", ActionStatus.DENIED),
        ("needs_input", ActionStatus.NEEDS_INPUT),
        ("needs_approval", ActionStatus.NEEDS_APPROVAL),
        ("timed_out", ActionStatus.TIMED_OUT),
        ("cancelled", ActionStatus.CANCELLED),
    ],
)
def test_action_executor_preserves_structured_tool_lifecycle_status(
    reported: str,
    action_status: ActionStatus,
) -> None:
    tool = _ResultTool({"status": reported, "message": "result state"})
    executor = ActionExecutor(ToolRegistry().register(tool))

    result = executor.execute([Action(name="result")])[0]

    assert result.status is action_status
    assert result.output == {"message": "result state"}
    if action_status in {
        ActionStatus.SUCCESS,
        ActionStatus.PARTIAL,
        ActionStatus.RUNNING,
        ActionStatus.SKIPPED,
        ActionStatus.NEEDS_INPUT,
        ActionStatus.NEEDS_APPROVAL,
    }:
        assert result.error is None
    else:
        assert result.error == "result state"


def test_action_executor_unwraps_the_exact_success_envelope() -> None:
    tool = _ResultTool({"status": "success", "output": "done"})

    result = ActionExecutor(ToolRegistry().register(tool)).execute(
        [Action(name="result")]
    )[0]

    assert result.status is ActionStatus.SUCCESS
    assert result.output == "done"
