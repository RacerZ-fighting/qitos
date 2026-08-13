from __future__ import annotations

from typing import Any

import pytest

from qitos import Action, ToolRegistry
from qitos.core.action import ActionStatus
from qitos.core.tool import BaseTool, ToolSpec, ToolValidationResult
from qitos.engine.action_executor import ActionExecutor


class _StrictInputTool(BaseTool):
    def __init__(self) -> None:
        self.validation_calls = 0
        self.execution_calls = 0
        super().__init__(
            ToolSpec(
                name="strict_input",
                description="Validate one bounded request.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "count": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 3,
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["safe", "fast"],
                        },
                        "label": {"type": "string", "minLength": 2},
                    },
                    "required": ["count", "mode"],
                },
            )
        )

    def validate_input(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> ToolValidationResult:
        _ = runtime_context
        self.validation_calls += 1
        if args.get("label") == "blocked":
            return ToolValidationResult.fail("blocked by domain rule", code="blocked")
        return ToolValidationResult.ok()

    async def execute(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = runtime_context
        self.execution_calls += 1
        return dict(args)


@pytest.mark.parametrize(
    "arguments",
    [
        {"mode": "safe"},
        {"count": 2, "mode": "safe", "unknown": True},
        {"count": "2", "mode": "safe"},
        {"count": 2, "mode": "unsafe"},
        {"count": 0, "mode": "safe"},
        {"count": 2, "mode": "safe", "label": "x"},
    ],
)
@pytest.mark.asyncio
async def test_schema_violation_is_terminal_before_custom_validation_or_handler(
    arguments: dict[str, Any],
) -> None:
    tool = _StrictInputTool()

    result = (
        await ActionExecutor(ToolRegistry().register(tool)).execute(
            [Action(name=tool.name, args=arguments, action_id="call-1")]
        )
    )[0]

    assert result.status is ActionStatus.ERROR
    assert result.attempts == 0
    assert result.metadata["error_category"] == "invalid_tool_arguments"
    assert result.metadata["executed"] is False
    assert result.metadata["validation"]["valid"] is False
    assert tool.validation_calls == 0
    assert tool.execution_calls == 0


@pytest.mark.asyncio
async def test_custom_validation_runs_after_schema_validation() -> None:
    tool = _StrictInputTool()

    result = (
        await ActionExecutor(ToolRegistry().register(tool)).execute(
            [
                Action(
                    name=tool.name,
                    args={"count": 2, "mode": "safe", "label": "blocked"},
                )
            ]
        )
    )[0]

    assert result.status is ActionStatus.ERROR
    assert result.metadata["error_category"] == "blocked"
    assert tool.validation_calls == 1
    assert tool.execution_calls == 0


def test_registry_projects_the_schema_used_by_execution() -> None:
    tool = _StrictInputTool()
    registry = ToolRegistry().register(tool)

    projected = registry.get_all_specs()[0]["function"]["parameters"]

    assert projected == tool.spec.input_schema
    assert projected["additionalProperties"] is False


def test_invalid_schema_fails_when_tool_is_constructed() -> None:
    with pytest.raises(ValueError, match="input_schema is invalid"):
        _InvalidSchemaTool()


class _InvalidSchemaTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            ToolSpec(
                name="invalid_schema",
                description="Never registers.",
                input_schema={
                    "type": "object",
                    "properties": {"value": {"type": "not-a-json-type"}},
                },
            )
        )

    async def execute(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> None:
        _ = args, runtime_context
