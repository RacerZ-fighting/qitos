"""Behavior contracts for the run-owned MCP Tool catalog."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import pytest

from qitos.core.action import Action
from qitos.core.agent_module import AgentModule
from qitos.core.decision import Decision
from qitos.core.state import StateSchema
from qitos.core.tool import BaseTool, ToolSpec
from qitos.core.tool_registry import ToolExposure, ToolRegistry
from qitos.engine.engine import Engine
from qitos.mcp import MCPCallToolResult, MCPRequestError, MCPServer, MCPToolInfo
from qitos.mcp.runtime import MCPRuntime


class _MutableServer(MCPServer):
    def __init__(
        self,
        name: str,
        tools: list[MCPToolInfo],
    ) -> None:
        self._name = name
        self.tools = tools
        self.connect_calls = 0
        self.cleanup_calls = 0
        self.list_calls = 0
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.list_error: Exception | None = None
        self.list_started = asyncio.Event()
        self.list_blocker: asyncio.Event | None = None

    @property
    def name(self) -> str:
        return self._name

    async def connect(self) -> None:
        self.connect_calls += 1

    async def cleanup(self) -> None:
        self.cleanup_calls += 1

    async def list_tools(self) -> list[MCPToolInfo]:
        self.list_calls += 1
        self.list_started.set()
        if self.list_blocker is not None:
            await self.list_blocker.wait()
        if self.list_error is not None:
            raise self.list_error
        return list(self.tools)

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> MCPCallToolResult:
        self.calls.append((tool_name, dict(arguments)))
        return MCPCallToolResult(
            content=(
                {
                    "type": "text",
                    "text": json.dumps({"tool": tool_name, "arguments": arguments}),
                },
            )
        )


@dataclass
class _State(StateSchema):
    task: str = ""


class _ChangeCatalogTool(BaseTool):
    def __init__(self, server: _MutableServer) -> None:
        super().__init__(
            ToolSpec(
                name="change_catalog",
                description="Publish a server Tool-list change.",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            )
        )
        self._server = server

    async def execute(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> str:
        _ = args, runtime_context
        self._server.tools = [MCPToolInfo(name="new")]
        self._server.notify_tools_changed()
        return "changed"


class _SnapshotAgent(AgentModule[_State, Any, Action]):
    def __init__(self, *, tools: ToolRegistry) -> None:
        super().__init__(tool_registry=tools)
        self.exposures: list[tuple[str, ...]] = []

    def init_state(self, task: str, **kwargs: Any) -> _State:
        _ = kwargs
        return _State(task=task, max_steps=2)

    def build_tool_exposure(
        self,
        state: _State,
        tool_registry: ToolRegistry,
    ) -> ToolExposure:
        _ = state
        exposure = tool_registry.freeze()
        self.exposures.append(tuple(exposure.list_tools()))
        return exposure

    def decide(self, state: _State, observation: Any) -> Decision[Action]:
        _ = observation
        if state.current_step == 0:
            return Decision.act(
                [
                    Action(name="change_catalog", args={}),
                    Action(name="mcp__catalog__old", args={"value": "same-turn"}),
                ]
            )
        return Decision.final("done")

    def reduce(
        self,
        state: _State,
        observation: Any,
        decision: Decision[Action],
    ) -> _State:
        _ = observation, decision
        return state


@pytest.mark.asyncio
async def test_notification_refresh_is_visible_only_to_the_next_turn() -> None:
    server = _MutableServer(
        "catalog",
        [
            MCPToolInfo(
                name="old",
                input_schema={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            )
        ],
    )
    registry = ToolRegistry().register(_ChangeCatalogTool(server))
    agent = _SnapshotAgent(tools=registry)

    result = await Engine(
        agent,
        mcp_server_factory=lambda: (server,),
    ).arun("exercise a catalog refresh")

    assert result.state.stop_reason == "completed"
    assert server.calls == [("old", {"value": "same-turn"})]
    assert "mcp__catalog__old" in agent.exposures[0]
    assert "mcp__catalog__new" not in agent.exposures[0]
    assert "mcp__catalog__old" not in agent.exposures[1]
    assert "mcp__catalog__new" in agent.exposures[1]
    assert registry.list_tools() == ["change_catalog"]
    assert server.cleanup_calls == 1


@pytest.mark.asyncio
async def test_failed_refresh_retains_the_last_complete_catalog() -> None:
    server = _MutableServer("catalog", [MCPToolInfo(name="stable")])
    registry = ToolRegistry()
    runtime = MCPRuntime(tool_registry=registry, servers=[server])
    await runtime.start()
    server.list_error = RuntimeError("discovery failed")

    assert runtime.request_refresh("catalog") is True
    published = await runtime.refresh_pending()

    assert published == ()
    assert registry.list_tools() == ["mcp__catalog__stable"]
    assert runtime.has_pending_refresh is False
    await runtime.aclose()


@pytest.mark.asyncio
async def test_session_expiry_reconnects_before_replaying_only_catalog_discovery() -> (
    None
):
    class _ExpiringServer(_MutableServer):
        def __init__(self) -> None:
            super().__init__("catalog", [MCPToolInfo(name="stable")])
            self.expire_next_list = False

        async def list_tools(self) -> list[MCPToolInfo]:
            self.list_calls += 1
            if self.expire_next_list:
                self.expire_next_list = False
                raise MCPRequestError(
                    "expired",
                    error_code="MCP_SESSION_EXPIRED",
                    error_category="mcp_transport_error",
                )
            return list(self.tools)

    server = _ExpiringServer()
    registry = ToolRegistry()
    runtime = MCPRuntime(tool_registry=registry, servers=[server])
    await runtime.start()
    server.tools = [MCPToolInfo(name="replacement")]
    server.expire_next_list = True

    assert runtime.request_refresh("catalog") is True
    assert await runtime.refresh_pending() == ("catalog",)

    assert server.connect_calls == 2
    assert server.cleanup_calls == 1
    assert server.list_calls == 3
    assert server.calls == []
    assert registry.list_tools() == ["mcp__catalog__replacement"]
    await runtime.aclose()


@pytest.mark.asyncio
async def test_cancelled_refresh_stays_pending_without_partial_publication() -> None:
    server = _MutableServer("catalog", [MCPToolInfo(name="stable")])
    registry = ToolRegistry()
    runtime = MCPRuntime(tool_registry=registry, servers=[server])
    await runtime.start()
    server.list_started = asyncio.Event()
    server.list_blocker = asyncio.Event()
    server.tools = [MCPToolInfo(name="replacement")]
    assert runtime.request_refresh("catalog") is True
    refresh = asyncio.create_task(runtime.refresh_pending())
    await asyncio.wait_for(server.list_started.wait(), timeout=1)

    refresh.cancel()
    with pytest.raises(asyncio.CancelledError):
        await refresh

    assert runtime.has_pending_refresh is True
    assert registry.list_tools() == ["mcp__catalog__stable"]
    server.list_blocker = None
    await runtime.aclose()


@pytest.mark.asyncio
async def test_expired_startup_deadline_cleans_partial_server_state() -> None:
    server = _MutableServer("catalog", [MCPToolInfo(name="unused")])
    registry = ToolRegistry()
    runtime = MCPRuntime(tool_registry=registry, servers=[server])

    await runtime.start(deadline_monotonic=time.monotonic() - 1)

    assert server.connect_calls == 0
    assert server.cleanup_calls == 1
    assert registry.list_tools() == []
    await runtime.aclose()
    assert server.cleanup_calls == 1


@pytest.mark.asyncio
async def test_startup_discovers_independent_servers_concurrently() -> None:
    slow = _MutableServer("slow", [MCPToolInfo(name="first")])
    healthy = _MutableServer("healthy", [MCPToolInfo(name="second")])
    slow.list_blocker = asyncio.Event()
    runtime = MCPRuntime(
        tool_registry=ToolRegistry(),
        servers=[slow, healthy],
        max_start_concurrency=2,
    )

    startup = asyncio.create_task(runtime.start())
    await asyncio.wait_for(slow.list_started.wait(), timeout=1)
    await asyncio.wait_for(healthy.list_started.wait(), timeout=1)

    slow.list_blocker.set()
    await startup

    assert runtime.published_tool_names == (
        "mcp__slow__first",
        "mcp__healthy__second",
    )
    await runtime.aclose()


@pytest.mark.asyncio
async def test_startup_concurrency_is_bounded() -> None:
    started: asyncio.Queue[str] = asyncio.Queue()
    release = asyncio.Event()
    active = 0
    peak = 0

    class _BlockingConnectServer(_MutableServer):
        async def connect(self) -> None:
            nonlocal active, peak
            self.connect_calls += 1
            active += 1
            peak = max(peak, active)
            await started.put(self.name)
            try:
                await release.wait()
            finally:
                active -= 1

    servers = [
        _BlockingConnectServer(name, [MCPToolInfo(name="probe")])
        for name in ("one", "two", "three")
    ]
    runtime = MCPRuntime(
        tool_registry=ToolRegistry(),
        servers=servers,
        max_start_concurrency=2,
    )

    startup = asyncio.create_task(runtime.start())
    await asyncio.wait_for(started.get(), timeout=1)
    await asyncio.wait_for(started.get(), timeout=1)

    assert started.empty()
    assert peak == 2

    release.set()
    await startup
    assert peak == 2
    assert all(server.connect_calls == 1 for server in servers)
    await runtime.aclose()


@pytest.mark.asyncio
async def test_invalid_replacement_catalog_is_not_partially_published() -> None:
    server = _MutableServer("catalog", [MCPToolInfo(name="stable")])
    registry = ToolRegistry()
    runtime = MCPRuntime(tool_registry=registry, servers=[server])
    await runtime.start()
    server.tools = [MCPToolInfo(name="duplicate"), MCPToolInfo(name="duplicate")]

    assert runtime.request_refresh("catalog") is True
    assert await runtime.refresh_pending() == ()

    assert registry.list_tools() == ["mcp__catalog__stable"]
    assert runtime.has_pending_refresh is True
    await runtime.aclose()


@pytest.mark.asyncio
async def test_refresh_requested_during_discovery_remains_pending() -> None:
    server = _MutableServer("catalog", [MCPToolInfo(name="stable")])
    registry = ToolRegistry()
    runtime = MCPRuntime(tool_registry=registry, servers=[server])
    await runtime.start()
    server.list_started = asyncio.Event()
    server.list_blocker = asyncio.Event()
    server.tools = [MCPToolInfo(name="first")]
    assert runtime.request_refresh("catalog") is True
    first_refresh = asyncio.create_task(runtime.refresh_pending())
    await asyncio.wait_for(server.list_started.wait(), timeout=1)

    assert runtime.request_refresh("catalog") is True
    server.list_blocker.set()
    assert await first_refresh == ("catalog",)
    assert runtime.has_pending_refresh is True
    assert registry.list_tools() == ["mcp__catalog__first"]

    server.list_blocker = None
    server.tools = [MCPToolInfo(name="second")]
    assert await runtime.refresh_pending() == ("catalog",)
    assert runtime.has_pending_refresh is False
    assert registry.list_tools() == ["mcp__catalog__second"]
    await runtime.aclose()


@pytest.mark.asyncio
async def test_cleanup_failure_can_be_retried_without_republishing_tools() -> None:
    class _RetryCleanupServer(_MutableServer):
        async def cleanup(self) -> None:
            self.cleanup_calls += 1
            if self.cleanup_calls == 1:
                raise RuntimeError("transient cleanup failure")

    server = _RetryCleanupServer("catalog", [MCPToolInfo(name="stable")])
    registry = ToolRegistry()
    runtime = MCPRuntime(tool_registry=registry, servers=[server])
    await runtime.start()

    with pytest.raises(RuntimeError, match="transient cleanup failure"):
        await runtime.aclose()

    assert registry.list_tools() == []
    assert server.cleanup_calls == 1
    await runtime.aclose()
    assert registry.list_tools() == []
    assert server.cleanup_calls == 2


@pytest.mark.asyncio
async def test_partial_start_keeps_healthy_catalog_and_retries_failed_cleanup() -> None:
    class _FailedStartServer(_MutableServer):
        def __init__(self) -> None:
            super().__init__("failed", [MCPToolInfo(name="never_published")])
            self.cleanup_error: Exception | None = RuntimeError(
                "transient failed-start cleanup"
            )

        async def list_tools(self) -> list[MCPToolInfo]:
            self.list_calls += 1
            raise RuntimeError("discovery failed")

        async def cleanup(self) -> None:
            self.cleanup_calls += 1
            if self.cleanup_error is not None:
                error = self.cleanup_error
                self.cleanup_error = None
                raise error

    healthy = _MutableServer("healthy", [MCPToolInfo(name="stable")])
    failed = _FailedStartServer()
    registry = ToolRegistry()
    runtime = MCPRuntime(
        tool_registry=registry,
        servers=[healthy, failed],
    )

    await runtime.start()

    assert registry.list_tools() == ["mcp__healthy__stable"]
    assert healthy.cleanup_calls == 0
    assert failed.cleanup_calls == 1
    healthy.tools = [MCPToolInfo(name="refreshed")]
    assert runtime.request_refresh("healthy") is True
    assert await runtime.refresh_pending() == ("healthy",)
    assert registry.list_tools() == ["mcp__healthy__refreshed"]

    await runtime.aclose()

    assert healthy.cleanup_calls == 1
    assert failed.cleanup_calls == 2
    assert registry.list_tools() == []
