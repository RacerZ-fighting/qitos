"""Tests for MCP Server support.

Covers:
- MCPToolInfo dataclass
- ToolFilter matches logic
- Schema conversion (convert_mcp_schema_to_tool_spec)
- Internal helpers (_map_type, _resolve_refs, _convert_property)
- Bridge (mcp_server_to_function_tools) with mock servers
- MCPServerStdio construction and cleanup without live process
- MCPServerStreamableHttp construction without live server
- MCPServer ABC contract
- ToolRegistry integration
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
from typing import Any, Awaitable, Callable, Dict, List, Optional

import pytest

from qitos.core.action import Action, ActionStatus
from qitos.core.tool import FunctionTool
from qitos.core.tool_registry import ToolRegistry
from qitos.core.tool_schema import tool_input_schema_errors
from qitos.engine.action_executor import ActionExecutor
from qitos.mcp import (
    MCPCallToolResult,
    MCPRequestError,
    MCPServer,
    MCPToolAnnotations,
    MCPToolInfo,
    MCPServerStdio,
    MCPServerStreamableHttp,
    ToolFilter,
    mcp_server_to_function_tools,
)
from qitos.mcp.schema_convert import (
    _convert_property,
    _map_type,
    _resolve_refs,
    convert_mcp_schema_to_tool_spec,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class _MockMCPServer(MCPServer):
    """In-memory mock MCP server for testing the bridge and schema convert."""

    def __init__(
        self,
        tools: Optional[List[MCPToolInfo]] = None,
        name: str = "mock",
        call_results: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._name = name
        self._tools = tools or []
        self._call_results = call_results or {}
        self._connected = False
        self._cleaned_up = False

    @property
    def name(self) -> str:
        return self._name

    async def connect(self) -> None:
        self._connected = True

    async def cleanup(self) -> None:
        self._cleaned_up = True

    async def list_tools(self) -> List[MCPToolInfo]:
        return list(self._tools)

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> MCPCallToolResult:
        value = self._call_results.get(
            tool_name,
            {"status": "success", "tool": tool_name, "arguments": arguments},
        )
        if isinstance(value, MCPCallToolResult):
            return value
        return MCPCallToolResult(content=({"type": "text", "text": json.dumps(value)},))


class _FakeStdioProcess:
    pid = 42

    class _Writer:
        def __init__(self, owner: "_FakeStdioProcess") -> None:
            self._owner = owner

        def write(self, data: bytes) -> None:
            message = json.loads(data.decode("utf-8"))
            self._owner.messages.append(message)
            if self._owner.on_message is not None:
                self._owner.on_message(message)

        async def drain(self) -> None:
            return None

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.stdin = self._Writer(self)
        self.messages: list[dict[str, Any]] = []
        self.on_message: Optional[Callable[[dict[str, Any]], None]] = None

    def send(self, message: dict[str, Any]) -> None:
        self.stdout.feed_data((json.dumps(message) + "\n").encode("utf-8"))

    def terminate(self) -> None:
        self.stdout.feed_eof()
        self.stderr.feed_eof()

    def kill(self) -> None:
        self.terminate()

    async def wait(self) -> int:
        self.returncode = 0
        return 0


class _AwaitedResponseContext:
    def __init__(self, response: Awaitable[Any]) -> None:
        self._response = response

    async def __aenter__(self) -> Any:
        return await self._response

    async def __aexit__(self, *args: Any) -> None:
        _ = args


class _FakeHTTPResponse:
    """Typed defaults shared by the transport's lightweight HTTP fakes."""

    status_code = 200
    headers: dict[str, str] = {}
    _payload: Any = None

    async def aread(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


# --------------------------------------------------------------------------- #
# 1. MCPToolInfo
# --------------------------------------------------------------------------- #


class TestMCPToolInfo:
    """Test MCPToolInfo dataclass."""

    def test_basic_creation(self) -> None:
        info = MCPToolInfo(
            name="read_file",
            description="Read a file from disk",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["path"],
            },
        )
        assert info.name == "read_file"
        assert info.description == "Read a file from disk"
        assert "properties" in info.input_schema

    def test_defaults(self) -> None:
        info = MCPToolInfo(name="tool_a")
        assert info.description == ""
        assert info.input_schema == {}


# --------------------------------------------------------------------------- #
# 2. MCPServer ABC
# --------------------------------------------------------------------------- #


class TestMCPServerABC:
    """Test MCPServer abstract interface."""

    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            MCPServer()

    def test_subclass_must_implement(self) -> None:
        class IncompleteServer(MCPServer):
            pass

        with pytest.raises(TypeError):
            IncompleteServer()

    def test_concrete_subclass(self) -> None:
        class FakeServer(MCPServer):
            @property
            def name(self) -> str:
                return "fake"

            async def connect(self):
                pass

            async def cleanup(self):
                pass

            async def list_tools(self):
                return []

            async def call_tool(self, tool_name, arguments):
                return MCPCallToolResult(content=({"type": "text", "text": "ok"},))

        server = FakeServer()
        assert server.name == "fake"

    def test_mock_server_fulfills_contract(self) -> None:
        server = _MockMCPServer(name="test")
        assert server.name == "test"


# --------------------------------------------------------------------------- #
# 3. ToolFilter
# --------------------------------------------------------------------------- #


class TestToolFilter:
    """Test MCP ToolFilter."""

    def test_no_filter_passes_all(self) -> None:
        f = ToolFilter()
        assert f.matches("anything") is True
        assert f.matches("other") is True

    def test_allowed_list(self) -> None:
        f = ToolFilter(allowed_tool_names={"search", "read"})
        assert f.matches("search") is True
        assert f.matches("read") is True
        assert f.matches("write") is False

    def test_blocked_list(self) -> None:
        f = ToolFilter(blocked_tool_names={"dangerous"})
        assert f.matches("safe_tool") is True
        assert f.matches("dangerous") is False

    def test_allowed_and_blocked_combined(self) -> None:
        # Name must be in allowed AND not in blocked
        f = ToolFilter(
            allowed_tool_names={"search", "dangerous"},
            blocked_tool_names={"dangerous"},
        )
        assert f.matches("search") is True
        assert f.matches("dangerous") is False
        assert f.matches("other") is False

    def test_filter_func_overrides_all(self) -> None:
        f = ToolFilter(
            allowed_tool_names={"search"},
            blocked_tool_names={"search"},
            filter_func=lambda name: name.startswith("fs_"),
        )
        assert f.matches("fs_read") is True
        assert f.matches("search") is False  # filter_func takes priority

    def test_filter_func_false(self) -> None:
        f = ToolFilter(filter_func=lambda name: False)
        assert f.matches("any") is False

    def test_blocklist_overrides_allowlist(self) -> None:
        f = ToolFilter(
            allowed_tool_names={"read", "write"}, blocked_tool_names={"write"}
        )
        assert f.matches("read") is True
        assert f.matches("write") is False  # blocked overrides


# --------------------------------------------------------------------------- #
# 4. Schema conversion
# --------------------------------------------------------------------------- #


class TestSchemaConvert:
    """Test MCP JSON Schema to ToolSpec conversion."""

    def test_simple_schema(self) -> None:
        tool = MCPToolInfo(
            name="read_file",
            description="Read a file",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["path"],
            },
        )
        spec = convert_mcp_schema_to_tool_spec(tool)
        assert spec.name == "read_file"
        assert spec.description == "Read a file"
        assert "path" in spec.parameters
        assert "path" in spec.required
        assert spec.parameters["path"]["type"] == "string"
        assert spec.parameters["path"]["description"] == "File path"

    def test_name_prefix(self) -> None:
        tool = MCPToolInfo(
            name="read",
            description="Read",
            input_schema={"type": "object", "properties": {}},
        )
        spec = convert_mcp_schema_to_tool_spec(tool, name_prefix="fs")
        assert spec.name == "fs__read"

    def test_multiple_types(self) -> None:
        tool = MCPToolInfo(
            name="multi",
            description="Multi type tool",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "count": {"type": "integer", "description": "Max results"},
                    "ratio": {"type": "number"},
                    "flag": {"type": "boolean"},
                    "items": {"type": "array", "items": {"type": "string"}},
                    "config": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                    },
                },
                "required": ["query"],
            },
        )
        spec = convert_mcp_schema_to_tool_spec(tool)
        assert spec.parameters["query"]["type"] == "string"
        assert spec.parameters["count"]["type"] == "integer"
        assert spec.parameters["ratio"]["type"] == "number"
        assert spec.parameters["flag"]["type"] == "boolean"
        assert spec.parameters["items"]["type"] == "array"
        assert spec.parameters["items"]["items"]["type"] == "string"
        assert spec.parameters["config"]["type"] == "object"
        assert spec.parameters["config"]["properties"]["key"]["type"] == "string"

    def test_no_properties(self) -> None:
        tool = MCPToolInfo(
            name="list",
            description="List all",
            input_schema={"type": "object", "properties": {}},
        )
        spec = convert_mcp_schema_to_tool_spec(tool)
        assert spec.parameters == {}
        assert spec.required == []

    def test_empty_schema(self) -> None:
        tool = MCPToolInfo(name="no_args", description="No args tool")
        spec = convert_mcp_schema_to_tool_spec(tool)
        assert spec.name == "no_args"
        assert spec.parameters == {}
        assert spec.required == []

    def test_nullable_anyof(self) -> None:
        tool = MCPToolInfo(
            name="nullable_tool",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
            },
        )
        spec = convert_mcp_schema_to_tool_spec(tool)
        assert spec.parameters["query"]["type"] == "string"
        assert spec.parameters["query"]["nullable"] is True

    def test_enum_values(self) -> None:
        tool = MCPToolInfo(
            name="enum_tool",
            input_schema={
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["fast", "slow"]},
                },
            },
        )
        spec = convert_mcp_schema_to_tool_spec(tool)
        assert spec.parameters["mode"]["enum"] == ["fast", "slow"]

    def test_ref_resolution(self) -> None:
        tool = MCPToolInfo(
            name="ref_tool",
            input_schema={
                "type": "object",
                "properties": {
                    "item": {"$ref": "#/$defs/Item"},
                },
                "$defs": {
                    "Item": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                    },
                },
            },
        )
        spec = convert_mcp_schema_to_tool_spec(tool)
        assert spec.parameters["item"]["type"] == "object"
        assert spec.parameters["item"]["properties"]["id"]["type"] == "string"

    def test_default_values(self) -> None:
        tool = MCPToolInfo(
            name="defaults_tool",
            input_schema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 10},
                },
            },
        )
        spec = convert_mcp_schema_to_tool_spec(tool)
        assert spec.parameters["limit"]["default"] == 10

    def test_string_format(self) -> None:
        tool = MCPToolInfo(
            name="date_tool",
            input_schema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "format": "date"},
                },
            },
        )
        spec = convert_mcp_schema_to_tool_spec(tool)
        assert spec.parameters["date"]["format"] == "date"

    def test_additional_properties_false(self) -> None:
        tool = MCPToolInfo(
            name="strict_obj",
            input_schema={
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                },
                "additionalProperties": False,
            },
        )
        spec = convert_mcp_schema_to_tool_spec(tool)
        assert spec.input_schema.get("additionalProperties") is False

    def test_root_schema_constraints_remain_validation_authority(self) -> None:
        tool = MCPToolInfo(
            name="dependent_fields",
            input_schema={
                "type": "object",
                "properties": {
                    "mode": {"type": "string"},
                    "detail": {"type": "string"},
                },
                "dependentRequired": {"mode": ["detail"]},
                "additionalProperties": False,
            },
        )

        spec = convert_mcp_schema_to_tool_spec(tool)

        assert tool_input_schema_errors(spec.input_schema, {"mode": "advanced"})
        assert not tool_input_schema_errors(
            spec.input_schema,
            {"mode": "advanced", "detail": "bounded"},
        )

    def test_mcp_hints_project_conservative_tool_execution_metadata(self) -> None:
        unspecified = convert_mcp_schema_to_tool_spec(
            MCPToolInfo(name="unspecified", input_schema={"type": "object"})
        )
        declared_read_only = convert_mcp_schema_to_tool_spec(
            MCPToolInfo(
                name="declared_read_only",
                input_schema={"type": "object"},
                annotations=MCPToolAnnotations(read_only_hint=True),
            )
        )

        assert unspecified.read_only is False
        assert declared_read_only.read_only is True
        assert unspecified.permissions.network is True
        assert unspecified.concurrency_safe is None
        assert unspecified.retry_policy is None


# --------------------------------------------------------------------------- #
# 5. Internal helpers
# --------------------------------------------------------------------------- #


class TestInternalHelpers:
    def test_map_type_known(self) -> None:
        assert _map_type({"type": "string"}) == "string"
        assert _map_type({"type": "integer"}) == "integer"
        assert _map_type({"type": "number"}) == "number"
        assert _map_type({"type": "boolean"}) == "boolean"
        assert _map_type({"type": "array"}) == "array"
        assert _map_type({"type": "object"}) == "object"

    def test_map_type_unknown(self) -> None:
        assert _map_type({"type": "custom"}) == "any"
        assert _map_type({}) == "any"

    def test_map_type_null(self) -> None:
        assert _map_type({"type": "null"}) == "any"

    def test_resolve_refs_simple(self) -> None:
        defs = {"Foo": {"type": "string"}}
        result = _resolve_refs({"$ref": "#/$defs/Foo"}, defs)
        assert result == {"type": "string"}

    def test_resolve_refs_nested(self) -> None:
        defs = {"Bar": {"$ref": "#/$defs/Baz"}, "Baz": {"type": "integer"}}
        result = _resolve_refs({"$ref": "#/$defs/Bar"}, defs)
        assert result == {"type": "integer"}

    def test_resolve_refs_depth_limit(self) -> None:
        defs: Dict[str, Any] = {}
        defs["A"] = {"$ref": "#/$defs/A"}
        result = _resolve_refs({"$ref": "#/$defs/A"}, defs)
        # Should not infinite loop; returns something
        assert isinstance(result, dict)

    def test_resolve_refs_unresolvable(self) -> None:
        defs = {}
        result = _resolve_refs({"$ref": "#/$defs/Missing"}, defs)
        # Returns the original ref dict since the target is not found
        assert "$ref" in result

    def test_convert_property_allof_single(self) -> None:
        result = _convert_property(
            {"allOf": [{"type": "string", "description": "desc"}]}
        )
        assert result["type"] == "string"
        assert result["description"] == "desc"

    def test_convert_property_allof_multiple(self) -> None:
        result = _convert_property(
            {
                "allOf": [
                    {"type": "object", "description": "base"},
                    {"description": "extra"},
                ],
            }
        )
        # allOf with mixed schemas falls back to "any" since we can't
        # fully merge arbitrary JSON Schema compositions
        assert result["type"] in ("object", "any")

    def test_convert_property_oneof(self) -> None:
        result = _convert_property({"oneOf": [{"type": "integer"}, {"type": "string"}]})
        assert result["type"] == "integer"

    def test_convert_property_additional_properties(self) -> None:
        result = _convert_property(
            {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "additionalProperties": False,
            }
        )
        assert result["type"] == "object"
        assert result["additionalProperties"] is False

    def test_convert_property_anyof_multiple_non_null(self) -> None:
        result = _convert_property(
            {
                "anyOf": [{"type": "string"}, {"type": "integer"}],
            }
        )
        # Falls through to first non-null variant
        assert result["type"] == "string"


# --------------------------------------------------------------------------- #
# 6. Bridge with mock server
# --------------------------------------------------------------------------- #


class TestBridge:
    @pytest.mark.asyncio
    async def test_bridge_produces_function_tools(self) -> None:
        tools = [
            MCPToolInfo(
                name="search",
                description="Search for items",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),
            MCPToolInfo(
                name="count",
                description="Count items",
                input_schema={
                    "type": "object",
                    "properties": {"filter": {"type": "string"}},
                    "required": [],
                },
            ),
        ]
        server = _MockMCPServer(tools=tools)
        result = await mcp_server_to_function_tools(server)
        assert len(result) == 2
        assert all(isinstance(t, FunctionTool) for t in result)
        names = {t.name for t in result}
        assert "search" in names
        assert "count" in names

    @pytest.mark.asyncio
    async def test_bridge_with_allowed_filter(self) -> None:
        tools = [
            MCPToolInfo(name="search", input_schema={"type": "object"}),
            MCPToolInfo(name="write", input_schema={"type": "object"}),
            MCPToolInfo(name="delete", input_schema={"type": "object"}),
        ]
        server = _MockMCPServer(tools=tools)
        f = ToolFilter(allowed_tool_names={"search"})
        result = await mcp_server_to_function_tools(server, tool_filter=f)
        assert len(result) == 1
        assert result[0].name == "search"

    @pytest.mark.asyncio
    async def test_bridge_with_blocked_filter(self) -> None:
        tools = [
            MCPToolInfo(name="safe", input_schema={"type": "object"}),
            MCPToolInfo(name="dangerous", input_schema={"type": "object"}),
        ]
        server = _MockMCPServer(tools=tools)
        f = ToolFilter(blocked_tool_names={"dangerous"})
        result = await mcp_server_to_function_tools(server, tool_filter=f)
        assert len(result) == 1
        assert result[0].name == "safe"

    @pytest.mark.asyncio
    async def test_bridge_with_name_prefix(self) -> None:
        tools = [
            MCPToolInfo(
                name="search", description="Search", input_schema={"type": "object"}
            ),
        ]
        server = _MockMCPServer(tools=tools)
        result = await mcp_server_to_function_tools(server, name_prefix="db")
        assert len(result) == 1
        assert result[0].name == "db__search"
        assert result[0].spec.name == "db__search"

    @pytest.mark.asyncio
    async def test_bridge_sanitizes_model_name_but_calls_raw_name(self) -> None:
        server = _MockMCPServer(
            name="server.one",
            tools=[
                MCPToolInfo(
                    name="tool.two-three",
                    input_schema={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    },
                )
            ],
        )
        bridged = await mcp_server_to_function_tools(
            server,
            name_prefix=f"mcp__{server.name}",
        )
        result = await bridged[0].execute({"value": "evidence"})

        assert bridged[0].name == "mcp__server_one__tool_two_three"
        assert result.output == {
            "status": "success",
            "tool": "tool.two-three",
            "arguments": {"value": "evidence"},
        }
        assert result.metadata == {
            "mcp_server": "server.one",
            "mcp_tool": "tool.two-three",
        }

    @pytest.mark.asyncio
    async def test_bridge_preserves_mcp_arguments_named_like_runtime_injections(
        self,
    ) -> None:
        argument_names = (
            "runtime_context",
            "env",
            "ops",
            "file_ops",
            "process_ops",
        )
        server = _MockMCPServer(
            tools=[
                MCPToolInfo(
                    name="reserved_arguments",
                    input_schema={
                        "type": "object",
                        "properties": {
                            name: {"type": "string"} for name in argument_names
                        },
                        "required": list(argument_names),
                    },
                )
            ]
        )
        bridged = await mcp_server_to_function_tools(server)
        arguments = {
            name: f"value-{index}" for index, name in enumerate(argument_names)
        }

        result = await bridged[0].execute(
            arguments,
            runtime_context={"env": object(), "ops": {"file": object()}},
        )

        assert result.output["arguments"] == arguments

    @pytest.mark.asyncio
    async def test_bridge_disambiguates_and_bounds_model_names(self) -> None:
        server = _MockMCPServer(
            tools=[
                MCPToolInfo(name="tool-name"),
                MCPToolInfo(name="tool_name"),
                MCPToolInfo(name="x" * 100),
            ]
        )
        names_in_use = {"mcp__mock__tool_name"}
        bridged = await mcp_server_to_function_tools(
            server,
            name_prefix="mcp__mock",
            used_names=names_in_use,
        )

        model_names = [tool.name for tool in bridged]
        assert len(model_names) == len(set(model_names))
        assert all(len(name) <= 64 for name in model_names)
        assert all(name in names_in_use for name in model_names)

    @pytest.mark.asyncio
    async def test_bridge_tool_spec_has_correct_schema(self) -> None:
        tools = [
            MCPToolInfo(
                name="greet",
                description="Greet someone",
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Person name"},
                    },
                    "required": ["name"],
                },
            ),
        ]
        server = _MockMCPServer(tools=tools)
        result = await mcp_server_to_function_tools(server)
        assert len(result) == 1
        spec = result[0].spec
        assert spec.name == "greet"
        assert spec.description == "Greet someone"
        assert "name" in spec.parameters
        assert spec.parameters["name"]["type"] == "string"
        assert spec.required == ["name"]
        assert spec.input_schema["additionalProperties"] is False

    @pytest.mark.asyncio
    async def test_bridge_empty_tools(self) -> None:
        server = _MockMCPServer(tools=[])
        result = await mcp_server_to_function_tools(server)
        assert result == []

    @pytest.mark.asyncio
    async def test_remote_is_error_becomes_typed_terminal_error(self) -> None:
        server = _MockMCPServer(
            tools=[MCPToolInfo(name="fails")],
            call_results={
                "fails": MCPCallToolResult(
                    content=({"type": "text", "text": "remote rejected"},),
                    is_error=True,
                )
            },
        )
        tool = (await mcp_server_to_function_tools(server))[0]

        result = (
            await ActionExecutor(ToolRegistry().register(tool)).execute(
                [Action(name="fails")]
            )
        )[0]

        assert result.status is ActionStatus.ERROR
        assert result.output == "remote rejected"
        assert result.error == "remote rejected"
        assert result.metadata["error_category"] == "mcp_tool_error"
        assert result.metadata["error_code"] == "MCP_TOOL_ERROR"
        assert result.metadata["mcp_server"] == "mock"
        assert result.metadata["mcp_tool"] == "fails"


# --------------------------------------------------------------------------- #
# 7. MCPServerStdio (no live subprocess)
# --------------------------------------------------------------------------- #


class TestMCPServerStdio:
    def test_construction(self) -> None:
        server = MCPServerStdio(
            command="npx",
            args=["-y", "@mcp/server"],
            env={"KEY": "val"},
            cwd="/tmp",
            name="test-stdio",
        )
        assert server.name == "test-stdio"
        assert server._process is None

    def test_default_name(self) -> None:
        server = MCPServerStdio(command="python")
        assert server.name == "stdio:python"

    @pytest.mark.asyncio
    async def test_cleanup_without_connect(self) -> None:
        server = MCPServerStdio(command="echo")
        # Should not raise even if never connected
        await server.cleanup()

    @pytest.mark.asyncio
    async def test_operations_without_connect_raise(self) -> None:
        server = MCPServerStdio(command="echo")
        with pytest.raises(RuntimeError, match="not connected"):
            await server.list_tools()
        with pytest.raises(RuntimeError, match="not connected"):
            await server.call_tool("x", {})

    @pytest.mark.asyncio
    async def test_reader_routes_out_of_order_responses_and_notifications(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        process = _FakeStdioProcess()
        pending_list_id: int | None = None

        def respond(message: dict[str, Any]) -> None:
            nonlocal pending_list_id
            method = message["method"]
            request_id = message.get("id")
            if method == "initialize":
                process.send({"jsonrpc": "2.0", "id": request_id, "result": {}})
            elif method == "tools/list":
                pending_list_id = request_id
            elif method == "tools/call":
                process.send(
                    {
                        "jsonrpc": "2.0",
                        "method": "notifications/tools/list_changed",
                    }
                )
                process.send(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [{"type": "text", "text": "call-result"}]
                        },
                    }
                )
                process.send(
                    {
                        "jsonrpc": "2.0",
                        "id": pending_list_id,
                        "result": {"tools": [{"name": "probe"}]},
                    }
                )

        process.on_message = respond

        async def create_process(*args: Any, **kwargs: Any) -> _FakeStdioProcess:
            _ = args, kwargs
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
        server = MCPServerStdio(command="fake")
        tools_changed = asyncio.Event()
        server.set_tools_changed_handler(tools_changed.set)
        await server.connect()

        listed = asyncio.create_task(server.list_tools())
        await asyncio.sleep(0)
        called = asyncio.create_task(server.call_tool("probe", {}))

        call_result = await asyncio.wait_for(called, timeout=1)
        assert (
            call_result.to_tool_result(
                server_name=server.name,
                tool_name="probe",
            ).output
            == "call-result"
        )
        tools = await asyncio.wait_for(listed, timeout=1)
        assert [tool.name for tool in tools] == ["probe"]
        assert tools_changed.is_set()
        await server.cleanup()
        assert server._pending == {}

    @pytest.mark.asyncio
    async def test_tools_list_collects_cursor_pages(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        process = _FakeStdioProcess()
        cursors: list[str | None] = []

        def respond(message: dict[str, Any]) -> None:
            method = message["method"]
            if method == "initialize":
                process.send({"jsonrpc": "2.0", "id": message["id"], "result": {}})
            elif method == "tools/list":
                cursor = message["params"].get("cursor")
                cursors.append(cursor)
                result = (
                    {"tools": [{"name": "first"}], "nextCursor": "page-2"}
                    if cursor is None
                    else {"tools": [{"name": "second"}]}
                )
                process.send({"jsonrpc": "2.0", "id": message["id"], "result": result})

        process.on_message = respond

        async def create_process(*args: Any, **kwargs: Any) -> _FakeStdioProcess:
            _ = args, kwargs
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
        server = MCPServerStdio(command="fake")
        await server.connect()

        tools = await server.list_tools()

        assert [tool.name for tool in tools] == ["first", "second"]
        assert cursors == [None, "page-2"]
        await server.cleanup()

    @pytest.mark.asyncio
    async def test_cancelled_request_notifies_stdio_server(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        process = _FakeStdioProcess()
        call_started = asyncio.Event()

        def respond(message: dict[str, Any]) -> None:
            if message["method"] == "initialize":
                process.send({"jsonrpc": "2.0", "id": message["id"], "result": {}})
            elif message["method"] == "tools/call":
                call_started.set()

        process.on_message = respond

        async def create_process(*args: Any, **kwargs: Any) -> _FakeStdioProcess:
            _ = args, kwargs
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
        server = MCPServerStdio(command="fake")
        await server.connect()
        request = asyncio.create_task(server.call_tool("probe", {}))
        await asyncio.wait_for(call_started.wait(), timeout=1)

        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request

        call_message = next(
            message for message in process.messages if message["method"] == "tools/call"
        )
        cancellation = next(
            message
            for message in process.messages
            if message["method"] == "notifications/cancelled"
        )
        assert cancellation["params"]["requestId"] == call_message["id"]
        await server.cleanup()

    @pytest.mark.asyncio
    async def test_cancelled_request_does_not_wait_on_blocked_notification(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        process = _FakeStdioProcess()
        call_started = asyncio.Event()

        def respond(message: dict[str, Any]) -> None:
            if message["method"] == "initialize":
                process.send({"jsonrpc": "2.0", "id": message["id"], "result": {}})
            elif message["method"] == "tools/call":
                call_started.set()

        process.on_message = respond

        async def create_process(*args: Any, **kwargs: Any) -> _FakeStdioProcess:
            _ = args, kwargs
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
        server = MCPServerStdio(
            command="fake",
            cancel_notification_timeout_seconds=0.01,
        )
        await server.connect()
        request = asyncio.create_task(server.call_tool("probe", {}))
        await asyncio.wait_for(call_started.wait(), timeout=1)
        await server._write_lock.acquire()

        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(request, timeout=1)

        server._write_lock.release()
        assert all(
            message["method"] != "notifications/cancelled"
            for message in process.messages
        )
        await server.cleanup()

    @pytest.mark.asyncio
    async def test_reader_failure_rejects_later_requests_without_hanging(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        process = _FakeStdioProcess()

        def respond(message: dict[str, Any]) -> None:
            if message["method"] == "initialize":
                process.send({"jsonrpc": "2.0", "id": message["id"], "result": {}})

        process.on_message = respond

        async def create_process(*args: Any, **kwargs: Any) -> _FakeStdioProcess:
            _ = args, kwargs
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
        server = MCPServerStdio(command="fake")
        await server.connect()
        process.stdout.feed_data(b"not-json\n")
        assert server._reader_task is not None
        await asyncio.wait_for(server._reader_task, timeout=1)
        cleanup = server._cleanup_task
        if cleanup is not None:
            await asyncio.wait_for(asyncio.shield(cleanup), timeout=1)

        with pytest.raises(RuntimeError, match="reader is unavailable"):
            await asyncio.wait_for(server.list_tools(), timeout=1)
        assert server._process is None

    @pytest.mark.asyncio
    async def test_oversized_stdio_frame_has_stable_protocol_error_and_closes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        process = _FakeStdioProcess()

        def respond(message: dict[str, Any]) -> None:
            if message["method"] == "initialize":
                process.send({"jsonrpc": "2.0", "id": message["id"], "result": {}})

        process.on_message = respond

        async def create_process(*args: Any, **kwargs: Any) -> _FakeStdioProcess:
            _ = args, kwargs
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
        server = MCPServerStdio(command="fake", max_frame_bytes=64)
        await server.connect()
        process.stdout.feed_data(b"x" * 65 + b"\n")
        assert server._reader_task is not None
        await asyncio.wait_for(server._reader_task, timeout=1)
        cleanup = server._cleanup_task
        if cleanup is not None:
            await asyncio.wait_for(asyncio.shield(cleanup), timeout=1)

        error = server._reader_error
        assert isinstance(error, MCPRequestError)
        assert error.error_code == "MCP_PROTOCOL_ERROR"
        assert "byte limit" in str(error)
        assert server._process is None

    @pytest.mark.asyncio
    async def test_reader_eof_rejects_later_requests_and_cleanup_drains_tasks(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        process = _FakeStdioProcess()

        def respond(message: dict[str, Any]) -> None:
            if message["method"] == "initialize":
                process.send({"jsonrpc": "2.0", "id": message["id"], "result": {}})

        process.on_message = respond

        async def create_process(*args: Any, **kwargs: Any) -> _FakeStdioProcess:
            _ = args, kwargs
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
        server = MCPServerStdio(command="fake")
        await server.connect()
        process.stdout.feed_eof()
        assert server._reader_task is not None
        await asyncio.wait_for(server._reader_task, timeout=1)

        with pytest.raises(RuntimeError, match="reader is unavailable"):
            await asyncio.wait_for(server.list_tools(), timeout=1)

        await server.cleanup()
        assert server._process is None
        assert server._reader_task is None
        assert server._stderr_task is None
        assert server._pending == {}

    @pytest.mark.asyncio
    async def test_cancelled_cleanup_still_reaps_the_stdio_process(self) -> None:
        class BlockingProcess:
            def __init__(self) -> None:
                self.stdin = None
                self.stdout = None
                self.stderr = None
                self.pid = 42
                self.terminated = False
                self.killed = False
                self.wait_started = asyncio.Event()
                self.release_wait = asyncio.Event()
                self.reaped = False

            def terminate(self) -> None:
                self.terminated = True

            def kill(self) -> None:
                self.killed = True
                self.release_wait.set()

            async def wait(self) -> int:
                self.wait_started.set()
                await self.release_wait.wait()
                self.reaped = True
                return 0

        process = BlockingProcess()
        server = MCPServerStdio(command="fake")
        server._process = process  # type: ignore[assignment]
        cleanup = asyncio.create_task(server.cleanup())
        await asyncio.wait_for(process.wait_started.wait(), timeout=1)

        cleanup.cancel()
        await asyncio.sleep(0)
        assert process.reaped is False
        process.release_wait.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(cleanup, timeout=1)

        assert process.terminated is True
        assert process.killed is False
        assert process.reaped is True
        assert server._process is None
        assert server._cleanup_task is None

    @pytest.mark.asyncio
    @pytest.mark.skipif(os.name == "nt", reason="POSIX process-group contract")
    async def test_stdio_cleanup_signals_the_owned_process_group(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        process = _FakeStdioProcess()
        signals: list[tuple[int, signal.Signals]] = []

        def respond(message: dict[str, Any]) -> None:
            if message["method"] == "initialize":
                process.send({"jsonrpc": "2.0", "id": message["id"], "result": {}})

        process.on_message = respond

        async def create_process(*args: Any, **kwargs: Any) -> _FakeStdioProcess:
            _ = args
            assert kwargs["start_new_session"] is True
            return process

        def kill_group(pid: int, requested: signal.Signals) -> None:
            signals.append((pid, requested))

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
        monkeypatch.setattr(os, "killpg", kill_group)
        server = MCPServerStdio(command="fake")
        await server.connect()

        await server.cleanup()

        assert signals == [
            (process.pid, signal.SIGTERM),
            (process.pid, signal.SIGKILL),
        ]


# --------------------------------------------------------------------------- #
# 8. MCPServerStreamableHttp (no live server)
# --------------------------------------------------------------------------- #


class TestMCPServerStreamableHttp:
    def test_construction(self) -> None:
        try:
            server = MCPServerStreamableHttp(
                url="http://localhost:8080/mcp",
                headers={"Authorization": "Bearer tok"},
                name="test-http",
            )
            assert server.name == "test-http"
            assert server._client is None
        except ImportError:
            pytest.skip("httpx not installed")

    def test_default_name(self) -> None:
        try:
            server = MCPServerStreamableHttp(url="http://localhost:8080/mcp")
            assert server.name == "http:http://localhost:8080/mcp"
        except ImportError:
            pytest.skip("httpx not installed")

    @pytest.mark.asyncio
    async def test_cleanup_without_connect(self) -> None:
        try:
            server = MCPServerStreamableHttp(url="http://localhost:8080/mcp")
            await server.cleanup()  # should not raise
        except ImportError:
            pytest.skip("httpx not installed")

    @pytest.mark.asyncio
    async def test_cleanup_retries_failed_http_client_close(self) -> None:
        server = MCPServerStreamableHttp(url="http://localhost:8080/mcp")

        class Client:
            def __init__(self) -> None:
                self.close_calls = 0

            async def aclose(self) -> None:
                self.close_calls += 1
                if self.close_calls == 1:
                    raise RuntimeError("client close failed")

        client = Client()
        server._client = client

        with pytest.raises(RuntimeError, match="client close failed"):
            await server.cleanup()

        assert server._client is client
        assert server._cleanup_task is None

        await server.cleanup()

        assert client.close_calls == 2
        assert server._client is None

    @pytest.mark.asyncio
    async def test_operations_without_connect_raise(self) -> None:
        try:
            server = MCPServerStreamableHttp(url="http://localhost:8080/mcp")
            with pytest.raises(RuntimeError, match="not connected"):
                await server.list_tools()
            with pytest.raises(RuntimeError, match="not connected"):
                await server.call_tool("x", {})
        except ImportError:
            pytest.skip("httpx not installed")

    @pytest.mark.asyncio
    async def test_cancelled_cleanup_waits_for_active_http_request(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qitos.mcp.http as http_module

        if http_module.httpx is None:
            pytest.skip("httpx not installed")

        request_started = asyncio.Event()
        release_request = asyncio.Event()

        class Response(_FakeHTTPResponse):
            headers: dict[str, str] = {}

            def __init__(
                self,
                payload: dict[str, Any] | None = None,
                *,
                block: bool = False,
            ) -> None:
                self._payload = payload
                self._block = block

            def raise_for_status(self) -> None:
                return None

            async def aread(self) -> bytes:
                if self._block:
                    request_started.set()
                    await release_request.wait()
                return await super().aread()

        class StreamContext:
            def __init__(self, response: Response) -> None:
                self._response = response

            async def __aenter__(self) -> Response:
                return self._response

            async def __aexit__(self, *args: Any) -> None:
                _ = args

        class Client:
            def __init__(self, **kwargs: Any) -> None:
                self.headers = dict(kwargs.get("headers", {}))
                self.closed = False

            def stream(self, method: str, path: str, **kwargs: Any) -> StreamContext:
                _ = path
                assert method == "POST"
                payload = kwargs["json"]
                if payload["method"] == "initialize":
                    result: dict[str, Any] = {}
                    block = False
                else:
                    assert payload["method"] == "tools/call"
                    result = {"content": [{"type": "text", "text": "done"}]}
                    block = True
                return StreamContext(
                    Response(
                        {
                            "jsonrpc": "2.0",
                            "id": payload["id"],
                            "result": result,
                        },
                        block=block,
                    )
                )

            async def post(
                self,
                path: str,
                *,
                json: dict[str, Any],
            ) -> Response:
                _ = path
                assert json["method"] == "notifications/initialized"
                return Response()

            async def aclose(self) -> None:
                self.closed = True

        clients: list[Client] = []

        def build_client(**kwargs: Any) -> Client:
            client = Client(**kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(http_module.httpx, "AsyncClient", build_client)
        server = MCPServerStreamableHttp(url="http://mcp.invalid")
        await server.connect()

        request = asyncio.create_task(server.call_tool("probe", {}))
        await asyncio.wait_for(request_started.wait(), timeout=1)
        cleanup = asyncio.create_task(server.cleanup())
        await asyncio.sleep(0)
        cleanup.cancel()
        await asyncio.sleep(0)

        assert clients[0].closed is False
        release_request.set()
        result = await asyncio.wait_for(request, timeout=1)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(cleanup, timeout=1)

        assert result.content[0]["text"] == "done"
        assert clients[0].closed is True
        assert server._client is None
        assert server._cleanup_task is None

    @pytest.mark.asyncio
    async def test_cleanup_waits_for_active_http_notification(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qitos.mcp.http as http_module

        if http_module.httpx is None:
            pytest.skip("httpx not installed")

        notification_started = asyncio.Event()
        release_notification = asyncio.Event()

        class Response(_FakeHTTPResponse):
            headers: dict[str, str] = {}

            def __init__(self, payload: dict[str, Any] | None = None) -> None:
                self._payload = payload

            def raise_for_status(self) -> None:
                return None

        class StreamContext:
            def __init__(self, response: Response) -> None:
                self._response = response

            async def __aenter__(self) -> Response:
                return self._response

            async def __aexit__(self, *args: Any) -> None:
                _ = args

        class Client:
            def __init__(self, **kwargs: Any) -> None:
                self.headers = dict(kwargs.get("headers", {}))
                self.closed = False

            def stream(self, method: str, path: str, **kwargs: Any) -> StreamContext:
                _ = path
                assert method == "POST"
                payload = kwargs["json"]
                return StreamContext(
                    Response(
                        {
                            "jsonrpc": "2.0",
                            "id": payload["id"],
                            "result": {},
                        }
                    )
                )

            async def post(
                self,
                path: str,
                *,
                json: dict[str, Any],
            ) -> Response:
                _ = path
                if json["method"] != "notifications/initialized":
                    notification_started.set()
                    await release_notification.wait()
                return Response()

            async def aclose(self) -> None:
                self.closed = True

        clients: list[Client] = []

        def build_client(**kwargs: Any) -> Client:
            client = Client(**kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(http_module.httpx, "AsyncClient", build_client)
        server = MCPServerStreamableHttp(url="http://mcp.invalid")
        await server.connect()

        notification = asyncio.create_task(
            server._send_notification("notifications/progress")
        )
        await asyncio.wait_for(notification_started.wait(), timeout=1)
        cleanup = asyncio.create_task(server.cleanup())
        await asyncio.sleep(0)

        assert clients[0].closed is False
        release_notification.set()
        await asyncio.wait_for(notification, timeout=1)
        await asyncio.wait_for(cleanup, timeout=1)

        assert clients[0].closed is True
        assert server._client is None

    @pytest.mark.asyncio
    async def test_handshake_http_failure_closes_client(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qitos.mcp.http as http_module

        if http_module.httpx is None:
            pytest.skip("httpx not installed")

        class Response(_FakeHTTPResponse):
            status_code = 500

            def raise_for_status(self) -> None:
                raise RuntimeError("HTTP failure")

            def json(self) -> dict[str, Any]:
                return {}

        class Client:
            def __init__(self, **kwargs: Any) -> None:
                _ = kwargs
                self.closed = False

            async def post(
                self,
                path: str,
                *,
                json: dict[str, Any],
            ) -> Response:
                _ = path, json
                return Response()

            def stream(self, method: str, path: str, **kwargs: Any):
                assert method == "POST"
                return _AwaitedResponseContext(self.post(path, json=kwargs["json"]))

            async def aclose(self) -> None:
                self.closed = True

        clients: list[Client] = []

        def build_client(**kwargs: Any) -> Client:
            client = Client(**kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(http_module.httpx, "AsyncClient", build_client)
        server = MCPServerStreamableHttp(url="http://mcp.invalid")

        with pytest.raises(RuntimeError, match="HTTP failure"):
            await server.connect()

        assert len(clients) == 1
        assert clients[0].closed is True
        assert server._client is None

    @pytest.mark.asyncio
    async def test_notification_http_status_is_wrapped_and_closes_client(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qitos.mcp.http as http_module

        if http_module.httpx is None:
            pytest.skip("httpx not installed")

        class Response(_FakeHTTPResponse):
            status_code = 200
            headers: dict[str, str] = {}

            def __init__(self, payload: Any = None, *, reject: bool = False) -> None:
                self._payload = payload
                self._reject = reject

            def raise_for_status(self) -> None:
                if self._reject:
                    request = http_module.httpx.Request("POST", "http://mcp.invalid")
                    response = http_module.httpx.Response(503, request=request)
                    raise http_module.httpx.HTTPStatusError(
                        "service unavailable",
                        request=request,
                        response=response,
                    )

            def json(self) -> Any:
                return self._payload

        class Client:
            def __init__(self, **kwargs: Any) -> None:
                _ = kwargs
                self.closed = False

            async def post(
                self,
                path: str,
                *,
                json: dict[str, Any],
            ) -> Response:
                _ = path
                assert json["method"] == "notifications/initialized"
                return Response(reject=True)

            def stream(self, method: str, path: str, **kwargs: Any):
                _ = path
                assert method == "POST"
                payload = kwargs["json"]
                return _AwaitedResponseContext(
                    _return_value(
                        Response(
                            {
                                "jsonrpc": "2.0",
                                "id": payload["id"],
                                "result": {},
                            }
                        )
                    )
                )

            async def aclose(self) -> None:
                self.closed = True

        async def _return_value(value: Any) -> Any:
            return value

        clients: list[Client] = []

        def build_client(**kwargs: Any) -> Client:
            client = Client(**kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(http_module.httpx, "AsyncClient", build_client)
        server = MCPServerStreamableHttp(url="http://mcp.invalid")

        with pytest.raises(MCPRequestError, match="notification failed"):
            await server.connect()

        assert clients[0].closed is True
        assert server._client is None

    @pytest.mark.asyncio
    async def test_session_post_404_has_stable_error_without_replaying_tool(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qitos.mcp.http as http_module

        if http_module.httpx is None:
            pytest.skip("httpx not installed")

        class Response(_FakeHTTPResponse):
            def __init__(
                self,
                payload: Any = None,
                *,
                status_code: int = 200,
                headers: dict[str, str] | None = None,
            ) -> None:
                self._payload = payload
                self.status_code = status_code
                self.headers = headers or {}

            def raise_for_status(self) -> None:
                return None

            def json(self) -> Any:
                return self._payload

        class Client:
            def __init__(self, **kwargs: Any) -> None:
                self.headers = dict(kwargs.get("headers", {}))
                self.tool_calls = 0

            async def post(
                self,
                path: str,
                *,
                json: dict[str, Any],
            ) -> Response:
                _ = path
                if json["method"] == "notifications/initialized":
                    return Response(status_code=202)
                raise AssertionError(json["method"])

            def stream(self, method: str, path: str, **kwargs: Any):
                _ = path
                assert method == "POST"
                payload = kwargs["json"]
                if payload["method"] == "initialize":
                    response = Response(
                        {
                            "jsonrpc": "2.0",
                            "id": payload["id"],
                            "result": {},
                        },
                        headers={"Mcp-Session-Id": "expired-session"},
                    )
                else:
                    assert payload["method"] == "tools/call"
                    self.tool_calls += 1
                    response = Response(status_code=404)
                return _AwaitedResponseContext(_return_value(response))

            async def delete(self, path: str) -> Response:
                _ = path
                return Response(status_code=404)

            async def aclose(self) -> None:
                return None

        async def _return_value(value: Any) -> Any:
            return value

        clients: list[Client] = []

        def build_client(**kwargs: Any) -> Client:
            client = Client(**kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(http_module.httpx, "AsyncClient", build_client)
        server = MCPServerStreamableHttp(url="http://mcp.invalid")
        tools_changed = 0

        def mark_changed() -> None:
            nonlocal tools_changed
            tools_changed += 1

        server.set_tools_changed_handler(mark_changed)
        await server.connect()

        with pytest.raises(MCPRequestError) as raised:
            await server.call_tool("probe", {})

        assert raised.value.error_code == "MCP_SESSION_EXPIRED"
        assert raised.value.error_category == "mcp_transport_error"
        assert clients[0].tool_calls == 1
        assert tools_changed == 1
        await server.cleanup()

    @pytest.mark.asyncio
    async def test_batch_response_applies_notification_and_matching_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qitos.mcp.http as http_module

        if http_module.httpx is None:
            pytest.skip("httpx not installed")

        class Response(_FakeHTTPResponse):
            def __init__(self, payload: Any, status_code: int = 200) -> None:
                self._payload = payload
                self.status_code = status_code

            def raise_for_status(self) -> None:
                return None

            def json(self) -> Any:
                return self._payload

        class Client:
            def __init__(self, **kwargs: Any) -> None:
                _ = kwargs
                self.closed = False

            async def post(
                self,
                path: str,
                *,
                json: dict[str, Any],
            ) -> Response:
                _ = path
                method = json["method"]
                if method == "notifications/initialized":
                    return Response(None, status_code=202)
                if method == "initialize":
                    return Response(
                        {
                            "jsonrpc": "2.0",
                            "id": json["id"],
                            "result": {},
                        }
                    )
                assert method == "tools/list"
                return Response(
                    [
                        {
                            "jsonrpc": "2.0",
                            "method": "notifications/tools/list_changed",
                        },
                        {
                            "jsonrpc": "2.0",
                            "id": json["id"],
                            "result": {"tools": [{"name": "probe"}]},
                        },
                    ]
                )

            def stream(self, method: str, path: str, **kwargs: Any):
                assert method == "POST"
                return _AwaitedResponseContext(self.post(path, json=kwargs["json"]))

            async def aclose(self) -> None:
                self.closed = True

        clients: list[Client] = []

        def build_client(**kwargs: Any) -> Client:
            client = Client(**kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(http_module.httpx, "AsyncClient", build_client)
        server = MCPServerStreamableHttp(url="http://mcp.invalid")
        tools_changed = asyncio.Event()
        server.set_tools_changed_handler(tools_changed.set)
        await server.connect()

        tools = await server.list_tools()

        assert [tool.name for tool in tools] == ["probe"]
        assert tools_changed.is_set()
        await server.cleanup()
        assert clients[0].closed is True

    @pytest.mark.asyncio
    async def test_post_event_stream_returns_matching_response_and_notifications(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qitos.mcp.http as http_module

        if http_module.httpx is None:
            pytest.skip("httpx not installed")

        class Response(_FakeHTTPResponse):
            status_code = 200
            headers = {"Content-Type": "text/event-stream"}

            def __init__(self, lines: list[str]) -> None:
                self._lines = lines

            def raise_for_status(self) -> None:
                return None

            async def aiter_lines(self):
                for line in self._lines:
                    yield line

        class BufferedResponse(_FakeHTTPResponse):
            status_code = 202
            headers: dict[str, str] = {}

            def raise_for_status(self) -> None:
                return None

        class StreamContext:
            def __init__(self, response: Response) -> None:
                self._response = response

            async def __aenter__(self) -> Response:
                return self._response

            async def __aexit__(self, *args: Any) -> None:
                _ = args

        class Client:
            def __init__(self, **kwargs: Any) -> None:
                self.headers = dict(kwargs.get("headers", {}))

            async def post(
                self,
                path: str,
                *,
                json: dict[str, Any],
            ) -> BufferedResponse:
                _ = path
                assert json["method"] == "notifications/initialized"
                return BufferedResponse()

            def stream(self, method: str, path: str, **kwargs: Any) -> StreamContext:
                _ = path
                assert method == "POST"
                message = kwargs["json"]
                request_id = message["id"]
                if message["method"] == "initialize":
                    result = {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                    }
                else:
                    assert message["method"] == "tools/list"
                    result = {"tools": [{"name": "probe"}]}
                lines = []
                if message["method"] == "tools/list":
                    lines.extend(
                        [
                            "event: message",
                            'data: {"jsonrpc":"2.0","method":'
                            '"notifications/tools/list_changed"}',
                            "",
                        ]
                    )
                lines.extend(
                    [
                        "event: message",
                        "data: "
                        + json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "result": result,
                            }
                        ),
                        "",
                    ]
                )
                return StreamContext(Response(lines))

            async def aclose(self) -> None:
                return None

        monkeypatch.setattr(http_module.httpx, "AsyncClient", Client)
        server = MCPServerStreamableHttp(url="http://mcp.invalid")
        tools_changed = asyncio.Event()
        server.set_tools_changed_handler(tools_changed.set)

        await server.connect()
        tools = await server.list_tools()

        assert [tool.name for tool in tools] == ["probe"]
        assert tools_changed.is_set()
        await server.cleanup()

    @pytest.mark.asyncio
    async def test_batch_response_rejects_duplicate_matching_ids(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qitos.mcp.http as http_module

        if http_module.httpx is None:
            pytest.skip("httpx not installed")

        class Response(_FakeHTTPResponse):
            def __init__(self, payload: Any) -> None:
                self._payload = payload

            def raise_for_status(self) -> None:
                return None

            def json(self) -> Any:
                return self._payload

        class Client:
            def __init__(self, **kwargs: Any) -> None:
                _ = kwargs

            async def post(
                self,
                path: str,
                *,
                json: dict[str, Any],
            ) -> Response:
                _ = path
                if json["method"] == "notifications/initialized":
                    return Response(None)
                response = {
                    "jsonrpc": "2.0",
                    "id": json["id"],
                    "result": {},
                }
                if json["method"] == "initialize":
                    return Response(response)
                return Response([response, dict(response)])

            def stream(self, method: str, path: str, **kwargs: Any):
                assert method == "POST"
                return _AwaitedResponseContext(self.post(path, json=kwargs["json"]))

            async def aclose(self) -> None:
                return None

        monkeypatch.setattr(http_module.httpx, "AsyncClient", Client)
        server = MCPServerStreamableHttp(url="http://mcp.invalid")
        await server.connect()

        with pytest.raises(RuntimeError, match="duplicate matching ids"):
            await server.list_tools()

        await server.cleanup()

    @pytest.mark.asyncio
    async def test_response_rejects_boolean_request_id(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qitos.mcp.http as http_module

        if http_module.httpx is None:
            pytest.skip("httpx not installed")

        class Response(_FakeHTTPResponse):
            _payload = {"jsonrpc": "2.0", "id": True, "result": {}}

            def raise_for_status(self) -> None:
                return None

        class Client:
            def __init__(self, **kwargs: Any) -> None:
                _ = kwargs

            async def post(
                self,
                path: str,
                *,
                json: dict[str, Any],
            ) -> Response:
                _ = path, json
                return Response()

            def stream(self, method: str, path: str, **kwargs: Any):
                assert method == "POST"
                return _AwaitedResponseContext(self.post(path, json=kwargs["json"]))

            async def aclose(self) -> None:
                return None

        monkeypatch.setattr(http_module.httpx, "AsyncClient", Client)
        server = MCPServerStreamableHttp(url="http://mcp.invalid")

        with pytest.raises(MCPRequestError, match="id must be an integer"):
            await server.connect()

    @pytest.mark.asyncio
    async def test_http_timeout_maps_to_async_timeout(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qitos.mcp.http as http_module

        if http_module.httpx is None:
            pytest.skip("httpx not installed")

        class Response(_FakeHTTPResponse):
            def __init__(self, payload: Any = None) -> None:
                self._payload = payload

            def raise_for_status(self) -> None:
                return None

            def json(self) -> Any:
                return self._payload

        class Client:
            def __init__(self, **kwargs: Any) -> None:
                _ = kwargs

            async def post(
                self,
                path: str,
                *,
                json: dict[str, Any],
            ) -> Response:
                _ = path
                if json["method"] == "initialize":
                    return Response({"jsonrpc": "2.0", "id": json["id"], "result": {}})
                if json["method"] == "tools/call":
                    raise http_module.httpx.ReadTimeout("request timed out")
                return Response()

            def stream(self, method: str, path: str, **kwargs: Any):
                assert method == "POST"
                return _AwaitedResponseContext(self.post(path, json=kwargs["json"]))

            async def aclose(self) -> None:
                return None

        monkeypatch.setattr(http_module.httpx, "AsyncClient", Client)
        server = MCPServerStreamableHttp(url="http://mcp.invalid")
        await server.connect()

        with pytest.raises(asyncio.TimeoutError, match="tools/call"):
            await server.call_tool("probe", {})

        await server.cleanup()

    @pytest.mark.asyncio
    async def test_cancelled_http_request_notifies_server(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qitos.mcp.http as http_module

        if http_module.httpx is None:
            pytest.skip("httpx not installed")

        class Response(_FakeHTTPResponse):
            def __init__(self, payload: Any = None) -> None:
                self._payload = payload

            def raise_for_status(self) -> None:
                return None

            def json(self) -> Any:
                return self._payload

        class Client:
            def __init__(self, **kwargs: Any) -> None:
                _ = kwargs
                self.messages: list[dict[str, Any]] = []
                self.call_started = asyncio.Event()

            async def post(
                self,
                path: str,
                *,
                json: dict[str, Any],
            ) -> Response:
                _ = path
                self.messages.append(dict(json))
                if json["method"] == "initialize":
                    return Response({"jsonrpc": "2.0", "id": json["id"], "result": {}})
                if json["method"] == "tools/call":
                    self.call_started.set()
                    await asyncio.Event().wait()
                return Response()

            def stream(self, method: str, path: str, **kwargs: Any):
                assert method == "POST"
                return _AwaitedResponseContext(self.post(path, json=kwargs["json"]))

            async def aclose(self) -> None:
                return None

        clients: list[Client] = []

        def build_client(**kwargs: Any) -> Client:
            client = Client(**kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(http_module.httpx, "AsyncClient", build_client)
        server = MCPServerStreamableHttp(url="http://mcp.invalid")
        await server.connect()
        request = asyncio.create_task(server.call_tool("probe", {}))
        await asyncio.wait_for(clients[0].call_started.wait(), timeout=1)

        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request

        tool_call = next(
            message
            for message in clients[0].messages
            if message["method"] == "tools/call"
        )
        cancellation = next(
            message
            for message in clients[0].messages
            if message["method"] == "notifications/cancelled"
        )
        assert cancellation["params"]["requestId"] == tool_call["id"]
        await server.cleanup()

    @pytest.mark.asyncio
    async def test_cancelled_http_request_bounds_notification_wait(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qitos.mcp.http as http_module

        if http_module.httpx is None:
            pytest.skip("httpx not installed")

        class Response(_FakeHTTPResponse):
            def __init__(self, payload: Any = None) -> None:
                self._payload = payload

            def raise_for_status(self) -> None:
                return None

        class Client:
            def __init__(self, **kwargs: Any) -> None:
                _ = kwargs
                self.call_started = asyncio.Event()
                self.notification_started = asyncio.Event()

            async def post(
                self,
                path: str,
                *,
                json: dict[str, Any],
            ) -> Response:
                _ = path
                if json["method"] == "initialize":
                    return Response({"jsonrpc": "2.0", "id": json["id"], "result": {}})
                if json["method"] == "tools/call":
                    self.call_started.set()
                    await asyncio.Event().wait()
                if json["method"] == "notifications/cancelled":
                    self.notification_started.set()
                    await asyncio.Event().wait()
                return Response()

            def stream(self, method: str, path: str, **kwargs: Any):
                assert method == "POST"
                return _AwaitedResponseContext(self.post(path, json=kwargs["json"]))

            async def aclose(self) -> None:
                return None

        clients: list[Client] = []

        def build_client(**kwargs: Any) -> Client:
            client = Client(**kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(http_module.httpx, "AsyncClient", build_client)
        server = MCPServerStreamableHttp(
            url="http://mcp.invalid",
            cancel_notification_timeout_seconds=0.01,
        )
        await server.connect()
        request = asyncio.create_task(server.call_tool("probe", {}))
        await asyncio.wait_for(clients[0].call_started.wait(), timeout=1)

        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(request, timeout=1)

        assert clients[0].notification_started.is_set()
        await server.cleanup()

    @pytest.mark.asyncio
    async def test_idle_sse_notification_uses_session_and_cleanup_closes_listener(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qitos.mcp.http as http_module

        if http_module.httpx is None:
            pytest.skip("httpx not installed")

        session_id = "session-123"
        stream_lines: asyncio.Queue[str] = asyncio.Queue()

        class Response(_FakeHTTPResponse):
            def __init__(
                self,
                payload: Any = None,
                *,
                status_code: int = 200,
                headers: dict[str, str] | None = None,
            ) -> None:
                self._payload = payload
                self.status_code = status_code
                self.headers = headers or {}

            def raise_for_status(self) -> None:
                return None

            def json(self) -> Any:
                return self._payload

            async def aiter_lines(self):
                while True:
                    yield await stream_lines.get()

        class StreamContext:
            def __init__(self, owner: "Client") -> None:
                self._owner = owner

            async def __aenter__(self) -> Response:
                self._owner.stream_opened.set()
                return Response(headers={"Content-Type": "text/event-stream"})

            async def __aexit__(self, *args: Any) -> None:
                _ = args
                self._owner.stream_closed.set()

        class Client:
            def __init__(self, **kwargs: Any) -> None:
                self.headers = dict(kwargs.get("headers", {}))
                self.closed = False
                self.stream_opened = asyncio.Event()
                self.stream_closed = asyncio.Event()
                self.deleted = False
                self.initialized_headers: dict[str, str] | None = None

            async def post(
                self,
                path: str,
                *,
                json: dict[str, Any],
            ) -> Response:
                _ = path
                if json["method"] == "initialize":
                    return Response(
                        {
                            "jsonrpc": "2.0",
                            "id": json["id"],
                            "result": {
                                "protocolVersion": "2024-11-05",
                                "capabilities": {"tools": {"listChanged": True}},
                            },
                        },
                        headers={"Mcp-Session-Id": session_id},
                    )
                if json["method"] == "notifications/initialized":
                    self.initialized_headers = dict(self.headers)
                    return Response(status_code=202)
                raise AssertionError(json["method"])

            def stream(self, method: str, path: str, **kwargs: Any) -> StreamContext:
                if method == "POST":
                    return _AwaitedResponseContext(self.post(path, json=kwargs["json"]))
                _ = kwargs
                assert method == "GET"
                assert self.headers["Mcp-Session-Id"] == session_id
                assert "text/event-stream" in self.headers["Accept"]
                return StreamContext(self)

            async def delete(self, path: str) -> Response:
                _ = path
                assert self.headers["Mcp-Session-Id"] == session_id
                self.deleted = True
                return Response(status_code=200)

            async def aclose(self) -> None:
                self.closed = True

        clients: list[Client] = []

        def build_client(**kwargs: Any) -> Client:
            client = Client(**kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(http_module.httpx, "AsyncClient", build_client)
        server = MCPServerStreamableHttp(url="http://mcp.invalid")
        tools_changed = asyncio.Event()
        server.set_tools_changed_handler(tools_changed.set)

        await server.connect()
        assert clients[0].stream_opened.is_set()
        assert clients[0].initialized_headers is not None
        assert clients[0].initialized_headers["Mcp-Session-Id"] == session_id
        assert clients[0].initialized_headers["MCP-Protocol-Version"] == "2024-11-05"

        await stream_lines.put("id: catalog-v2")
        await stream_lines.put("event: message")
        await stream_lines.put(
            'data: {"jsonrpc":"2.0","method":"notifications/tools/list_changed"}'
        )
        await stream_lines.put("")
        await asyncio.wait_for(tools_changed.wait(), timeout=1)

        await server.cleanup()

        assert clients[0].deleted is True
        assert clients[0].stream_closed.is_set()
        assert clients[0].closed is True
        assert server._listener_task is None

    @pytest.mark.asyncio
    async def test_http_listener_reconnects_with_last_event_id(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qitos.mcp.http as http_module

        if http_module.httpx is None:
            pytest.skip("httpx not installed")

        class Response(_FakeHTTPResponse):
            status_code = 200

            def __init__(
                self,
                lines: list[str] | None = None,
                *,
                payload: Any = None,
                headers: dict[str, str] | None = None,
                blocker: asyncio.Event | None = None,
            ) -> None:
                self._lines = lines or []
                self._payload = payload
                self.headers = headers or {}
                self._blocker = blocker

            def raise_for_status(self) -> None:
                return None

            def json(self) -> Any:
                return self._payload

            async def aiter_lines(self):
                for line in self._lines:
                    yield line
                if self._blocker is not None:
                    await self._blocker.wait()

        class StreamContext:
            def __init__(self, response: Response) -> None:
                self._response = response

            async def __aenter__(self) -> Response:
                return self._response

            async def __aexit__(self, *args: Any) -> None:
                _ = args

        class Client:
            def __init__(self, **kwargs: Any) -> None:
                self.headers = dict(kwargs.get("headers", {}))
                self.get_headers: list[dict[str, str] | None] = []
                self.second_stream_opened = asyncio.Event()
                self.block_second_stream = asyncio.Event()

            async def post(
                self,
                path: str,
                *,
                json: dict[str, Any],
            ) -> Response:
                _ = path
                if json["method"] == "notifications/initialized":
                    return Response()
                raise AssertionError(json["method"])

            def stream(self, method: str, path: str, **kwargs: Any):
                _ = path
                if method == "POST":
                    payload = kwargs["json"]
                    return StreamContext(
                        Response(
                            [
                                "id: post-response-cursor",
                                "event: message",
                                "data: "
                                + json.dumps(
                                    {
                                        "jsonrpc": "2.0",
                                        "id": payload["id"],
                                        "result": {
                                            "capabilities": {
                                                "tools": {"listChanged": True}
                                            }
                                        },
                                    }
                                ),
                                "",
                            ],
                            headers={"Content-Type": "text/event-stream"},
                        )
                    )
                assert method == "GET"
                request_headers = kwargs.get("headers")
                self.get_headers.append(request_headers)
                if len(self.get_headers) == 1:
                    return StreamContext(
                        Response(
                            [
                                "id: catalog-v1",
                                "",
                                "id: catalog-v2",
                                "event: heartbeat",
                                "data: ignored",
                                "",
                                'data: {"jsonrpc":"2.0","method":'
                                '"notifications/tools/list_changed"}',
                                "",
                            ],
                            headers={"Content-Type": "text/event-stream"},
                        )
                    )
                self.second_stream_opened.set()
                return StreamContext(
                    Response(
                        headers={"Content-Type": "text/event-stream"},
                        blocker=self.block_second_stream,
                    )
                )

            async def aclose(self) -> None:
                return None

        clients: list[Client] = []

        def build_client(**kwargs: Any) -> Client:
            client = Client(**kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(http_module.httpx, "AsyncClient", build_client)
        server = MCPServerStreamableHttp(url="http://mcp.invalid")
        await server.connect()
        await asyncio.wait_for(clients[0].second_stream_opened.wait(), timeout=1)

        assert clients[0].get_headers == [None, {"Last-Event-ID": "catalog-v2"}]
        await server.cleanup()

    @pytest.mark.asyncio
    async def test_http_listener_treats_method_not_allowed_as_unsupported(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qitos.mcp.http as http_module

        if http_module.httpx is None:
            pytest.skip("httpx not installed")

        class Response(_FakeHTTPResponse):
            def __init__(self, payload: Any = None, *, status_code: int = 200) -> None:
                self._payload = payload
                self.status_code = status_code
                self.headers: dict[str, str] = {}

            def raise_for_status(self) -> None:
                return None

            def json(self) -> Any:
                return self._payload

        class StreamContext:
            async def __aenter__(self) -> Response:
                return Response(status_code=405)

            async def __aexit__(self, *args: Any) -> None:
                _ = args

        class Client:
            def __init__(self, **kwargs: Any) -> None:
                self.headers = dict(kwargs.get("headers", {}))
                self.closed = False

            async def post(
                self,
                path: str,
                *,
                json: dict[str, Any],
            ) -> Response:
                _ = path
                if json["method"] == "initialize":
                    return Response(
                        {
                            "jsonrpc": "2.0",
                            "id": json["id"],
                            "result": {
                                "capabilities": {"tools": {"listChanged": True}}
                            },
                        }
                    )
                return Response(status_code=202)

            def stream(self, method: str, path: str, **kwargs: Any) -> StreamContext:
                if method == "POST":
                    return _AwaitedResponseContext(self.post(path, json=kwargs["json"]))
                _ = path, kwargs
                assert method == "GET"
                return StreamContext()

            async def aclose(self) -> None:
                self.closed = True

        clients: list[Client] = []

        def build_client(**kwargs: Any) -> Client:
            client = Client(**kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(http_module.httpx, "AsyncClient", build_client)
        server = MCPServerStreamableHttp(url="http://mcp.invalid")

        await server.connect()
        assert server._listener_task is not None
        await asyncio.wait_for(server._listener_task, timeout=1)

        await server.cleanup()
        assert clients[0].closed is True
        assert server._listener_task is None


# --------------------------------------------------------------------------- #
# 9. ToolRegistry integration
# --------------------------------------------------------------------------- #


class TestToolRegistryIntegration:
    @pytest.mark.asyncio
    async def test_register_mcp_tools_in_registry(self) -> None:
        from qitos.core.tool_registry import ToolRegistry

        tools = [
            MCPToolInfo(
                name="search",
                description="Search items",
                input_schema={
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            ),
        ]
        server = _MockMCPServer(tools=tools)
        function_tools = await mcp_server_to_function_tools(server)

        registry = ToolRegistry()
        for ft in function_tools:
            registry.register(ft)

        assert "search" in registry
        tool = registry.get("search")
        assert tool is not None
        assert tool.spec.description == "Search items"

    @pytest.mark.asyncio
    async def test_register_prefixed_mcp_tools_in_registry(self) -> None:
        from qitos.core.tool_registry import ToolRegistry

        tools = [
            MCPToolInfo(
                name="search",
                description="Search items",
                input_schema={"type": "object"},
            ),
        ]
        server = _MockMCPServer(tools=tools)
        function_tools = await mcp_server_to_function_tools(server, name_prefix="db")

        registry = ToolRegistry()
        for ft in function_tools:
            registry.register(ft)

        assert "db__search" in registry
