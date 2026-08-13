"""Automatic schema generation from function signatures for QitOS tools."""

from __future__ import annotations

from copy import deepcopy
import inspect
import re
import types
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    get_args,
    get_origin,
    get_type_hints,
)

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for


_MAX_VALIDATION_ERRORS = 8
_MAX_VALIDATION_ERROR_CHARS = 500


def normalize_tool_input_schema(
    input_schema: Optional[Dict[str, Any]],
    *,
    parameters: Optional[Dict[str, Dict[str, Any]]] = None,
    required: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return one detached, closed, valid object schema for a tool call.

    Tool protocols always deliver a JSON object. QitOS therefore closes the
    root argument object by default while preserving an explicit
    ``additionalProperties`` declaration for tools that intentionally accept a
    free-form mapping. The returned schema is both the model projection and the
    executor's validation authority.
    """

    if input_schema is not None and not isinstance(input_schema, dict):
        raise TypeError("tool input_schema must be a JSON Schema object")

    schema: Dict[str, Any]
    if input_schema:
        schema = deepcopy(input_schema)
    else:
        schema = {
            "type": "object",
            "properties": deepcopy(parameters or {}),
            "required": list(required or []),
        }

    schema_type = schema.setdefault("type", "object")
    if schema_type != "object":
        raise ValueError("tool input_schema root type must be 'object'")

    properties = schema.setdefault("properties", {})
    if not isinstance(properties, dict):
        raise ValueError("tool input_schema properties must be an object")

    required_fields = schema.setdefault("required", [])
    if not isinstance(required_fields, list) or not all(
        isinstance(item, str) for item in required_fields
    ):
        raise ValueError("tool input_schema required must be a list of strings")

    schema.setdefault("additionalProperties", False)
    try:
        validator_for(schema).check_schema(schema)
    except SchemaError as exc:
        raise ValueError(f"tool input_schema is invalid: {exc.message}") from exc
    return schema


def tool_input_schema_errors(
    schema: Dict[str, Any],
    arguments: Dict[str, Any],
) -> tuple[str, ...]:
    """Return bounded JSON Schema violations for one tool argument object."""

    validator_class = validator_for(schema)
    validator_class.check_schema(schema)
    errors = sorted(
        validator_class(schema).iter_errors(arguments),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    rendered: List[str] = []
    for error in errors[:_MAX_VALIDATION_ERRORS]:
        path = "$" + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}"
            for part in error.absolute_path
        )
        message = " ".join(error.message.split())
        if len(message) > _MAX_VALIDATION_ERROR_CHARS:
            message = f"{message[:_MAX_VALIDATION_ERROR_CHARS]}..."
        rendered.append(f"{path}: {message}")
    if len(errors) > len(rendered):
        rendered.append(f"$:{len(errors) - len(rendered)} more validation error(s)")
    return tuple(rendered)


def function_schema(func: Any) -> Dict[str, Any]:
    """Extract parameter names, type annotations, and defaults from a function signature.

    Returns a dict with keys:
      - ``parameters``: mapping of param name -> {"type": <json schema>, "description": "", "default": ...}
      - ``required``: list of parameter names without defaults
      - ``descriptions``: mapping of param name -> description from docstring
    """
    sig = inspect.signature(func)
    hints = {}
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception:
        pass

    docstring = inspect.getdoc(func) or ""
    descriptions = parse_docstring(docstring)

    skip = {"self", "cls", "runtime_context", "env", "ops", "file_ops", "process_ops"}
    parameters: Dict[str, Dict[str, Any]] = {}
    required: List[str] = []

    for name, p in sig.parameters.items():
        if name in skip:
            continue
        annotation = hints.get(name, p.annotation)
        schema = type_to_json_schema(annotation)
        entry: Dict[str, Any] = dict(schema)
        entry["description"] = descriptions.get(name, "")
        if p.default is not inspect.Parameter.empty:
            entry["default"] = p.default
        else:
            required.append(name)
        parameters[name] = entry

    return {
        "parameters": parameters,
        "required": required,
        "descriptions": descriptions,
    }


def parse_docstring(docstring: str) -> Dict[str, str]:
    """Parse a Google-style docstring and extract Args descriptions.

    Supports the format::

        Args:
            x: The x value
            y: The y value

    Returns a dict mapping parameter name -> description string.
    """
    if not docstring:
        return {}

    # Strip common leading indentation (like inspect.cleandoc)
    docstring = inspect.cleandoc(docstring)

    result: Dict[str, str] = {}

    # Find the Args section
    match = re.search(r"^Args:\s*\n", docstring, re.MULTILINE)
    if not match:
        return result

    args_start = match.end()
    # Find the next section (e.g. Returns:, Raises:, or end of docstring)
    next_section = re.search(r"^\w+:\s*\n", docstring[args_start:], re.MULTILINE)
    args_block = (
        docstring[args_start : args_start + next_section.start()]
        if next_section
        else docstring[args_start:]
    )

    # Parse each parameter line — supports:
    #   name: description
    #   name (type): description
    #   name: multi-line description
    current_name: Optional[str] = None
    current_desc_lines: List[str] = []

    for line in args_block.split("\n"):
        stripped = line.strip()
        if not stripped:
            if current_name is not None:
                current_desc_lines.append("")
            continue

        # Check if this is a new parameter line
        param_match = re.match(r"^(\w+)(?:\s*\([^)]*\))?\s*:\s*(.*)", stripped)
        if param_match:
            # Save previous param
            if current_name is not None:
                desc = " ".join(current_desc_lines).strip()
                # Collapse multiple spaces
                desc = re.sub(r"\s+", " ", desc)
                result[current_name] = desc
            current_name = param_match.group(1)
            current_desc_lines = [param_match.group(2)] if param_match.group(2) else []
        elif current_name is not None:
            # Continuation of previous description
            current_desc_lines.append(stripped)

    # Save last param
    if current_name is not None:
        desc = " ".join(current_desc_lines).strip()
        desc = re.sub(r"\s+", " ", desc)
        result[current_name] = desc

    return result


def type_to_json_schema(annotation: Any) -> Dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema dict.

    Supported types:
      - Basic: str, int, float, bool -> {type: string|integer|number|boolean}
      - Optional[X] -> standard JSON Schema union with ``null``
      - list[X] -> {type: array, items: X}
      - dict[K,V] -> {type: object}
      - Literal[...] -> {type: X, enum: [...]}
      - Annotated[type, ...] -> transparent passthrough to inner type
      - Fallback: {}
    """
    if annotation is inspect.Parameter.empty or annotation is None:
        return {}

    if annotation is type(None):
        return {"type": "null"}

    if isinstance(annotation, str):
        normalized = annotation.strip().removeprefix("typing.")
        schema_type = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "dict": "object",
            "Dict": "object",
            "list": "array",
            "List": "array",
            "None": "null",
            "NoneType": "null",
        }.get(normalized)
        return {"type": schema_type} if schema_type is not None else {}

    # Unwrap Annotated — transparent passthrough to inner type
    origin = get_origin(annotation)
    if origin is Annotated:
        args = get_args(annotation)
        if args:
            return type_to_json_schema(args[0])

    # Handle both typing.Union and PEP 604 ``X | Y``.
    import typing

    if origin in {getattr(typing, "Union", None), types.UnionType}:
        args = get_args(annotation)
        variants = [type_to_json_schema(arg) for arg in args]
        return {"anyOf": variants} if variants else {}

    # Handle Literal[...]
    if origin is Literal:
        args = get_args(annotation)
        if not args:
            return {}
        # Infer type from first value
        first = args[0]
        if isinstance(first, bool):
            base_type = "boolean"
        elif isinstance(first, int):
            base_type = "integer"
        elif isinstance(first, float):
            base_type = "number"
        elif isinstance(first, str):
            base_type = "string"
        else:
            base_type = "string"
        return {"type": base_type, "enum": list(args)}

    # Handle list[X]
    if origin is list:
        args = get_args(annotation)
        if args:
            return {"type": "array", "items": type_to_json_schema(args[0])}
        return {"type": "array"}

    # Handle dict[K, V]
    if origin is dict:
        args = get_args(annotation)
        value_schema = type_to_json_schema(args[1]) if len(args) == 2 else {}
        return {
            "type": "object",
            "additionalProperties": value_schema or True,
        }

    # Basic types
    basic: Dict[type, str] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }
    if annotation in basic:
        return {"type": basic[annotation]}

    # Fallback for bare dict/list without parameters
    if annotation is dict:
        return {"type": "object"}
    if annotation is list:
        return {"type": "array"}

    return {}
