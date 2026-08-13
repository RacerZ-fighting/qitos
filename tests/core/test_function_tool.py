"""Tests for qitos.core.function_tool_decorator — @function_tool decorator."""

from typing import Literal, Optional

import pytest

from qitos.core.function_tool_decorator import function_tool
from qitos.core.tool import FunctionTool, RetryPolicy, tool
from qitos.core.tool_registry import ToolRegistry


class TestFunctionToolDecorator:
    def test_without_parens_creates_function_tool(self) -> None:
        @function_tool
        def greet(name: str) -> str:
            """Say hello."""
            return f"Hello {name}"

        assert isinstance(greet, FunctionTool)
        assert greet.name == "greet"
        assert greet.spec.description == "Say hello."

    def test_with_parens_overrides_name(self) -> None:
        @function_tool(name="custom_name")
        def greet(name: str) -> str:
            """Say hello."""
            return f"Hello {name}"

        assert isinstance(greet, FunctionTool)
        assert greet.name == "custom_name"

    def test_needs_approval_true(self) -> None:
        @function_tool(needs_approval=True)
        def dangerous(x: int) -> int:
            """Do something risky."""
            return x

        assert isinstance(dangerous, FunctionTool)
        assert dangerous.spec.needs_approval is True

    def test_needs_approval_default_false(self) -> None:
        @function_tool
        def safe(x: int) -> int:
            """Safe operation."""
            return x

        assert isinstance(safe, FunctionTool)
        assert safe.spec.needs_approval is False

    def test_with_description_override(self) -> None:
        @function_tool(description="Custom description")
        def my_fn(x: int) -> int:
            """Original docstring."""
            return x

        assert my_fn.spec.description == "Custom description"

    def test_with_timeout_and_retry_policy(self) -> None:
        retry_policy = RetryPolicy(max_attempts=4)

        @function_tool(timeout_s=30.0, retry_policy=retry_policy)
        def slow_fn(x: int) -> int:
            return x

        assert slow_fn.spec.timeout_s == 30.0
        assert slow_fn.spec.retry_policy is retry_policy

    def test_with_read_only_and_concurrency_safe(self) -> None:
        @function_tool(read_only=True, concurrency_safe=True)
        def safe_fn(x: int) -> int:
            return x

        assert safe_fn.spec.read_only is True
        assert safe_fn.spec.concurrency_safe is True

    def test_function_tool_callable(self) -> None:
        @function_tool
        def add(a: int, b: int) -> int:
            return a + b

        assert isinstance(add, FunctionTool)
        result = add(a=2, b=3)
        assert result == 5

    def test_function_tool_works_with_registry(self) -> None:
        @function_tool(name="math.add")
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        registry = ToolRegistry()
        registry.register(add)
        assert "math.add" in registry.list_tools()

    @pytest.mark.asyncio
    async def test_function_tool_context_injection(self) -> None:
        @function_tool
        def use_context(x: int, runtime_context: dict = None) -> dict:
            return {"x": x, "ctx": runtime_context}

        result = await use_context.execute(
            {"x": 42}, runtime_context={"env": "test"}
        )
        assert result["x"] == 42
        assert result["ctx"] == {"env": "test"}

    def test_optional_param_schema(self) -> None:
        @function_tool
        def fn_with_optional(x: int, y: Optional[str] = None) -> str:
            """Do something.

            Args:
                x: The x value
                y: An optional y value
            """
            return y or "default"

        assert "y" in fn_with_optional.spec.parameters
        assert fn_with_optional.spec.parameters["y"]["anyOf"] == [
            {"type": "string"},
            {"type": "null"},
        ]
        assert "x" in fn_with_optional.spec.required
        assert "y" not in fn_with_optional.spec.required

    def test_generated_schema_is_closed_and_preserves_literal(self) -> None:
        @function_tool
        def choose(mode: Literal["safe", "fast"]) -> str:
            return mode

        assert choose.spec.input_schema == {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["safe", "fast"],
                    "description": "",
                }
            },
            "required": ["mode"],
            "additionalProperties": False,
        }


class TestExistingToolDecoratorStillWorks:
    def test_tool_decorator_basic(self) -> None:
        @tool(name="legacy_tool")
        def legacy(x: int) -> int:
            """Legacy tool."""
            return x

        # @tool returns the function, not a FunctionTool
        assert callable(legacy)
        meta = legacy.__qitos_tool_meta__
        assert meta.name == "legacy_tool"

    def test_tool_with_needs_approval(self) -> None:
        @tool(name="needs_ok", needs_approval=True)
        def needs_ok(x: int) -> int:
            return x

        meta = needs_ok.__qitos_tool_meta__
        assert meta.needs_approval is True

    def test_tool_needs_approval_default(self) -> None:
        @tool()
        def no_approval(x: int) -> int:
            return x

        meta = no_approval.__qitos_tool_meta__
        assert meta.needs_approval is False
