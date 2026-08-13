"""Tests for qitos.core.tool_schema — automatic schema generation."""

from typing import Any, Dict, List, Optional

import pytest

try:
    from typing import Literal, Annotated
except ImportError:
    from typing_extensions import Literal, Annotated

from qitos.core.tool_schema import (
    function_schema,
    parse_docstring,
    type_to_json_schema,
)


# ---------------------------------------------------------------------------
# type_to_json_schema
# ---------------------------------------------------------------------------


class TestTypeToJsonSchema:
    def test_str(self) -> None:
        assert type_to_json_schema(str) == {"type": "string"}

    def test_int(self) -> None:
        assert type_to_json_schema(int) == {"type": "integer"}

    def test_float(self) -> None:
        assert type_to_json_schema(float) == {"type": "number"}

    def test_bool(self) -> None:
        assert type_to_json_schema(bool) == {"type": "boolean"}

    def test_optional_str(self) -> None:
        result = type_to_json_schema(Optional[str])
        assert result == {
            "anyOf": [{"type": "string"}, {"type": "null"}],
        }

    def test_list_int(self) -> None:
        result = type_to_json_schema(list[int])
        assert result == {"type": "array", "items": {"type": "integer"}}

    def test_dict_str_any(self) -> None:
        result = type_to_json_schema(dict[str, Any])
        assert result == {"type": "object", "additionalProperties": True}

    def test_literal_strings(self) -> None:
        result = type_to_json_schema(Literal["a", "b"])
        assert result == {"type": "string", "enum": ["a", "b"]}

    def test_literal_ints(self) -> None:
        result = type_to_json_schema(Literal[1, 2, 3])
        assert result == {"type": "integer", "enum": [1, 2, 3]}

    def test_annotated_int(self) -> None:
        result = type_to_json_schema(Annotated[int, "some metadata"])
        assert result == {"type": "integer"}

    def test_annotated_optional_str(self) -> None:
        result = type_to_json_schema(Annotated[Optional[str], "desc"])
        assert result == {
            "anyOf": [{"type": "string"}, {"type": "null"}],
        }

    def test_pep_604_union(self) -> None:
        assert type_to_json_schema(str | int) == {
            "anyOf": [{"type": "string"}, {"type": "integer"}],
        }

    def test_bare_list(self) -> None:
        assert type_to_json_schema(list) == {"type": "array"}

    def test_bare_dict(self) -> None:
        assert type_to_json_schema(dict) == {"type": "object"}

    def test_empty_annotation(self) -> None:
        import inspect
        assert type_to_json_schema(inspect.Parameter.empty) == {}

    def test_none_annotation(self) -> None:
        assert type_to_json_schema(None) == {}

    def test_unknown_type_fallback(self) -> None:
        # A custom class that isn't a known type should return {}
        class Custom:
            pass
        assert type_to_json_schema(Custom) == {}

    def test_nested_list(self) -> None:
        result = type_to_json_schema(list[list[int]])
        assert result == {
            "type": "array",
            "items": {"type": "array", "items": {"type": "integer"}},
        }


# ---------------------------------------------------------------------------
# parse_docstring
# ---------------------------------------------------------------------------


class TestParseDocstring:
    def test_google_style_args(self) -> None:
        doc = """Do something.

        Args:
            x: The x value
            y: The y value
        """
        result = parse_docstring(doc)
        assert result == {"x": "The x value", "y": "The y value"}

    def test_empty_docstring(self) -> None:
        assert parse_docstring("") == {}

    def test_no_args_section(self) -> None:
        doc = "Just a description."
        assert parse_docstring(doc) == {}

    def test_args_with_returns_section(self) -> None:
        doc = """Compute something.

        Args:
            a: First value
            b: Second value

        Returns:
            The result.
        """
        result = parse_docstring(doc)
        assert result == {"a": "First value", "b": "Second value"}

    def test_multiline_description(self) -> None:
        doc = """Do something.

        Args:
            x: This is a long
                description that spans
                multiple lines.
            y: Short
        """
        result = parse_docstring(doc)
        assert "x" in result
        assert "multiple lines" in result["x"]
        assert result["y"] == "Short"


# ---------------------------------------------------------------------------
# function_schema
# ---------------------------------------------------------------------------


class TestFunctionSchema:
    def test_basic_function(self) -> None:
        def greet(name: str, age: int = 0) -> str:
            """Greet someone.

            Args:
                name: The person's name
                age: The person's age
            """
            return f"Hello {name}"

        schema = function_schema(greet)
        assert "name" in schema["parameters"]
        assert "age" in schema["parameters"]
        assert "name" in schema["required"]
        assert "age" not in schema["required"]
        assert schema["parameters"]["name"]["description"] == "The person's name"
        assert schema["parameters"]["age"]["description"] == "The person's age"

    def test_skips_injected_params(self) -> None:
        def my_tool(x: int, runtime_context: dict, env: dict) -> None:
            pass

        schema = function_schema(my_tool)
        assert "x" in schema["parameters"]
        assert "runtime_context" not in schema["parameters"]
        assert "env" not in schema["parameters"]
