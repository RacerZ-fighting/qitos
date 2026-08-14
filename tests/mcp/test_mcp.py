"""MCP SDK integration and QitOS projection contracts."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from mcp.types import (
    CallToolResult,
    ListToolsResult,
    ServerNotification,
    TextContent,
    Tool,
    ToolAnnotations,
    ToolListChangedNotification,
)

from qitos.mcp import (
    MCPRequestError,
    MCPServer,
    MCPServerStdio,
    MCPServerStreamableHttp,
    ToolFilter,
    mcp_server_to_function_tools,
)
from qitos.mcp.schema_convert import convert_mcp_schema_to_tool_spec


class _FakeServer(MCPServer):
    def __init__(
        self,
        *,
        name: str = "fake",
        tools: list[Tool] | None = None,
        result: CallToolResult | None = None,
    ) -> None:
        self._name = name
        self.tools = tools or []
        self.result = result or CallToolResult(
            content=[TextContent(type="text", text="ok")]
        )
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def name(self) -> str:
        return self._name

    async def connect(self) -> None:
        return None

    async def cleanup(self) -> None:
        return None

    async def list_tools(self) -> list[Tool]:
        return list(self.tools)

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> CallToolResult:
        self.calls.append((tool_name, dict(arguments)))
        return self.result


def test_mcp_server_contract_is_abstract() -> None:
    with pytest.raises(TypeError):
        MCPServer()


def test_filter_combines_allow_and_block_lists() -> None:
    tool_filter = ToolFilter(
        allowed_tool_names={"read", "write"},
        blocked_tool_names={"write"},
    )

    assert tool_filter.matches("read") is True
    assert tool_filter.matches("write") is False
    assert tool_filter.matches("delete") is False


def test_schema_conversion_consumes_official_tool_model() -> None:
    tool = Tool(
        name="read",
        description="Read a file",
        inputSchema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        annotations=ToolAnnotations(readOnlyHint=True),
    )

    spec = convert_mcp_schema_to_tool_spec(tool)

    assert spec.name == "read"
    assert spec.description == "Read a file"
    assert spec.required == ["path"]
    assert spec.input_schema == tool.inputSchema
    assert spec.read_only is True


@pytest.mark.asyncio
async def test_bridge_preserves_raw_name_and_projects_json_text() -> None:
    server = _FakeServer(
        name="files",
        tools=[
            Tool(
                name="read-file",
                inputSchema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            )
        ],
        result=CallToolResult(
            content=[TextContent(type="text", text=json.dumps({"value": 7}))]
        ),
    )

    tools = await mcp_server_to_function_tools(server, name_prefix="files")
    result = await tools[0].execute({"path": "README.md"})

    assert tools[0].name == "files__read_file"
    assert server.calls == [("read-file", {"path": "README.md"})]
    assert result.status == "success"
    assert result.output == {"value": 7}
    assert result.metadata == {"mcp_server": "files", "mcp_tool": "read-file"}


@pytest.mark.asyncio
async def test_bridge_preserves_remote_tool_error() -> None:
    server = _FakeServer(
        tools=[Tool(name="fail", inputSchema={"type": "object"})],
        result=CallToolResult(
            content=[TextContent(type="text", text="remote failure")],
            isError=True,
        ),
    )

    tool = (await mcp_server_to_function_tools(server))[0]
    result = await tool.execute({})

    assert result.status == "error"
    assert result.error == "remote failure"
    assert result.output == "remote failure"
    assert result.metadata["error_code"] == "MCP_TOOL_ERROR"


@pytest.mark.asyncio
async def test_bridge_maps_stable_request_errors() -> None:
    class _FailingServer(_FakeServer):
        async def call_tool(
            self,
            tool_name: str,
            arguments: dict[str, Any],
        ) -> CallToolResult:
            raise MCPRequestError(
                "closed",
                error_code="MCP_TRANSPORT_CLOSED",
                error_category="mcp_transport_error",
            )

    server = _FailingServer(tools=[Tool(name="fail", inputSchema={"type": "object"})])
    tool = (await mcp_server_to_function_tools(server))[0]

    result = await tool.execute({})

    assert result.status == "error"
    assert result.metadata["error_code"] == "MCP_TRANSPORT_CLOSED"
    assert result.metadata["error_category"] == "mcp_transport_error"


class _FakeClientSession:
    pages: dict[str | None, ListToolsResult] = {}
    call_result = CallToolResult(content=[TextContent(type="text", text="done")])
    call_started: asyncio.Event | None = None
    call_release: asyncio.Event | None = None
    instance: _FakeClientSession | None = None

    def __init__(
        self,
        read_stream: object,
        write_stream: object,
        *,
        read_timeout_seconds: object,
        message_handler: object,
        client_info: object,
    ) -> None:
        _ = read_stream, write_stream, read_timeout_seconds, client_info
        self.message_handler = message_handler
        self.enter_task: asyncio.Task[Any] | None = None
        self.exit_task: asyncio.Task[Any] | None = None
        self.initialized = False
        self.calls: list[tuple[str, dict[str, Any]]] = []
        type(self).instance = self

    async def __aenter__(self) -> _FakeClientSession:
        self.enter_task = asyncio.current_task()
        return self

    async def __aexit__(self, *args: object) -> None:
        _ = args
        self.exit_task = asyncio.current_task()

    async def initialize(self) -> None:
        self.initialized = True

    async def list_tools(self, cursor: str | None = None) -> ListToolsResult:
        return self.pages[cursor]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> CallToolResult:
        self.calls.append((name, dict(arguments)))
        if self.call_started is not None:
            self.call_started.set()
        if self.call_release is not None:
            await self.call_release.wait()
        return self.call_result


@asynccontextmanager
async def _fake_stdio_transport(
    parameters: object,
) -> AsyncIterator[tuple[object, object]]:
    _ = parameters
    yield object(), object()


@pytest.fixture(autouse=True)
def _reset_fake_session() -> None:
    _FakeClientSession.pages = {
        None: ListToolsResult(
            tools=[Tool(name="first", inputSchema={"type": "object"})],
            nextCursor=None,
        )
    }
    _FakeClientSession.call_result = CallToolResult(
        content=[TextContent(type="text", text="done")]
    )
    _FakeClientSession.call_started = None
    _FakeClientSession.call_release = None
    _FakeClientSession.instance = None


@pytest.mark.asyncio
async def test_stdio_uses_one_owner_task_for_sdk_contexts(monkeypatch: Any) -> None:
    import qitos.mcp.server as server_module
    import qitos.mcp.stdio as stdio_module

    monkeypatch.setattr(server_module, "ClientSession", _FakeClientSession)
    monkeypatch.setattr(stdio_module, "stdio_client", _fake_stdio_transport)
    server = MCPServerStdio("mcp-server", args=["--stdio"])

    await server.connect()
    tools = await server.list_tools()
    result = await server.call_tool("first", {"value": 1})
    await server.cleanup()

    session = _FakeClientSession.instance
    assert session is not None
    assert session.initialized is True
    assert session.enter_task is session.exit_task
    assert session.enter_task is not asyncio.current_task()
    assert [tool.name for tool in tools] == ["first"]
    assert result.content[0].text == "done"
    assert session.calls == [("first", {"value": 1})]


@pytest.mark.asyncio
async def test_sdk_catalog_pagination_is_bounded_and_ordered(monkeypatch: Any) -> None:
    import qitos.mcp.server as server_module
    import qitos.mcp.stdio as stdio_module

    _FakeClientSession.pages = {
        None: ListToolsResult(
            tools=[Tool(name="first", inputSchema={"type": "object"})],
            nextCursor="page-2",
        ),
        "page-2": ListToolsResult(
            tools=[Tool(name="second", inputSchema={"type": "object"})],
            nextCursor=None,
        ),
    }
    monkeypatch.setattr(server_module, "ClientSession", _FakeClientSession)
    monkeypatch.setattr(stdio_module, "stdio_client", _fake_stdio_transport)
    server = MCPServerStdio("mcp-server")
    await server.connect()

    tools = await server.list_tools()

    assert [tool.name for tool in tools] == ["first", "second"]
    await server.cleanup()


@pytest.mark.asyncio
async def test_cleanup_waits_for_in_flight_sdk_request(monkeypatch: Any) -> None:
    import qitos.mcp.server as server_module
    import qitos.mcp.stdio as stdio_module

    started = asyncio.Event()
    release = asyncio.Event()
    _FakeClientSession.call_started = started
    _FakeClientSession.call_release = release
    monkeypatch.setattr(server_module, "ClientSession", _FakeClientSession)
    monkeypatch.setattr(stdio_module, "stdio_client", _fake_stdio_transport)
    server = MCPServerStdio("mcp-server")
    await server.connect()
    call = asyncio.create_task(server.call_tool("first", {}))
    await asyncio.wait_for(started.wait(), timeout=1)

    cleanup = asyncio.create_task(server.cleanup())
    await asyncio.sleep(0)
    assert cleanup.done() is False
    release.set()
    await call
    await cleanup


@pytest.mark.asyncio
async def test_sdk_tool_change_notification_marks_catalog_dirty(
    monkeypatch: Any,
) -> None:
    import qitos.mcp.server as server_module
    import qitos.mcp.stdio as stdio_module

    monkeypatch.setattr(server_module, "ClientSession", _FakeClientSession)
    monkeypatch.setattr(stdio_module, "stdio_client", _fake_stdio_transport)
    server = MCPServerStdio("mcp-server")
    changed = asyncio.Event()
    server.set_tools_changed_handler(changed.set)
    await server.connect()
    session = _FakeClientSession.instance
    assert session is not None

    await session.message_handler(ServerNotification(ToolListChangedNotification()))

    assert changed.is_set()
    await server.cleanup()


def test_transport_constructors_validate_external_input() -> None:
    with pytest.raises(ValueError, match="command"):
        MCPServerStdio("")
    with pytest.raises(TypeError, match="env"):
        MCPServerStdio("server", env={"KEY": 1})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="url"):
        MCPServerStreamableHttp("")
    with pytest.raises(ValueError, match="request_timeout_seconds"):
        MCPServerStreamableHttp(
            "https://example.invalid/mcp", request_timeout_seconds=0
        )


@pytest.mark.asyncio
async def test_stdio_round_trip_uses_official_sdk_transport(tmp_path: Any) -> None:
    script = tmp_path / "fixture_mcp_server.py"
    script.write_text(
        """
from mcp.server.fastmcp import FastMCP

server = FastMCP("qitos-test")

@server.tool()
def echo(value: str) -> str:
    return value

server.run(transport="stdio")
""".lstrip(),
        encoding="utf-8",
    )
    server = MCPServerStdio(sys.executable, args=[str(script)])

    try:
        await server.connect()
        tools = await server.list_tools()
        result = await server.call_tool("echo", {"value": "hello"})
    finally:
        await server.cleanup()

    assert [tool.name for tool in tools] == ["echo"]
    assert result.isError is False
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "hello"
