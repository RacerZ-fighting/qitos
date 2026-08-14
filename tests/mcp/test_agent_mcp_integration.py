"""Tests for AgentModule MCP integration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import threading
from typing import Any

import pytest

from qitos.core.action import Action
from qitos.core.agent_module import AgentModule
from qitos.core.decision import Decision
from qitos.core.state import StateSchema
from qitos.core.tool import ToolPermissionContext
from qitos.core.tool_registry import ToolRegistry
from qitos.engine.engine import Engine
from qitos.kit.permission import PermissionPipeline
from qitos.mcp import MCPCallToolResult
from qitos.mcp.runtime import MCPRuntime
from qitos.mcp.server import MCPServer, MCPToolInfo


class FakeMCPServer(MCPServer):
    """Fake MCP server for testing."""

    def __init__(self, name: str = "fake", tools: list | None = None):
        self._name = name
        self._tools = (
            tools
            if tools is not None
            else [
                MCPToolInfo(
                    name="read",
                    description="Read a file",
                    input_schema={"type": "object"},
                ),
            ]
        )
        self.connected = False
        self.cleaned_up = False
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.lifecycle_threads: list[int] = []

    @property
    def name(self) -> str:
        return self._name

    async def connect(self) -> None:
        self.connected = True
        self.cleaned_up = False
        self.lifecycle_threads.append(threading.get_ident())

    async def cleanup(self) -> None:
        self.cleaned_up = True
        self.lifecycle_threads.append(threading.get_ident())

    async def list_tools(self) -> list[MCPToolInfo]:
        return self._tools

    async def call_tool(
        self,
        name: str,
        arguments: dict,
    ) -> MCPCallToolResult:
        self.calls.append((name, dict(arguments)))
        self.lifecycle_threads.append(threading.get_ident())
        return MCPCallToolResult(
            content=(
                {
                    "type": "text",
                    "text": json.dumps({"result": name, "arguments": arguments}),
                },
            )
        )


@dataclass
class DummyState(StateSchema):
    task: str = ""


class DummyAgent(AgentModule[DummyState, Any, Any]):
    name = "dummy"

    def init_state(self, task: str, **kwargs: Any) -> DummyState:
        return DummyState(task=task)

    def reduce(self, state: DummyState, observation: Any, decision: Any) -> DummyState:
        return state


class CallingAgent(DummyAgent):
    def init_state(self, task: str, **kwargs: Any) -> DummyState:
        return DummyState(task=task, max_steps=2)

    def decide(self, state: DummyState, observation: Any) -> Decision[Action]:
        if state.current_step == 0:
            return Decision.act(
                [
                    Action(
                        name="mcp__server_one__tool_two_three",
                        args={"value": "evidence"},
                    )
                ]
            )
        return Decision.final("done")


class FinalAgent(DummyAgent):
    def decide(self, state: DummyState, observation: Any) -> Decision[Action]:
        _ = state, observation
        return Decision.final("done")


class TestEngineMCPServerFactory:
    def test_agent_module_rejects_live_transport_ownership(self) -> None:
        with pytest.raises(TypeError):
            DummyAgent(mcp_servers=[FakeMCPServer()])

    @pytest.mark.asyncio
    async def test_engine_factory_creates_fresh_transport_for_each_run(self):
        servers: list[FakeMCPServer] = []

        def create_servers() -> tuple[MCPServer, ...]:
            server = FakeMCPServer(
                name="server.one",
                tools=[
                    MCPToolInfo(
                        name="tool.two-three",
                        input_schema={
                            "type": "object",
                            "properties": {"value": {"type": "string"}},
                            "required": ["value"],
                        },
                    )
                ],
            )
            servers.append(server)
            return (server,)

        registry = ToolRegistry()
        agent = CallingAgent(tool_registry=registry)
        engine = Engine(agent, mcp_server_factory=create_servers)

        first = await engine.arun("first")
        second = await engine.arun("second")

        assert first.state.stop_reason == "completed"
        assert second.state.stop_reason == "completed"
        assert len(servers) == 2
        assert [server.calls for server in servers] == [
            [("tool.two-three", {"value": "evidence"})],
            [("tool.two-three", {"value": "evidence"})],
        ]
        assert all(len(server.lifecycle_threads) == 3 for server in servers)
        assert all(
            set(server.lifecycle_threads) == {threading.get_ident()}
            for server in servers
        )
        assert registry.list_tools() == []
        assert all(server.cleaned_up for server in servers)

    @pytest.mark.asyncio
    async def test_stepwise_session_starts_refreshes_and_closes_mcp_runtime(self):
        server = FakeMCPServer(
            name="server.one",
            tools=[
                MCPToolInfo(
                    name="tool.two-three",
                    input_schema={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                )
            ],
        )
        registry = ToolRegistry()
        engine = Engine(
            CallingAgent(tool_registry=registry),
            mcp_server_factory=lambda: (server,),
        )
        state, observation = engine.init_session("stepwise")

        first = await engine.astep(state, observation)

        assert first.stop is False
        assert server.calls == [("tool.two-three", {"value": "evidence"})]
        assert registry.list_tools() == ["mcp__server_one__tool_two_three"]

        server._tools = [MCPToolInfo(name="replacement")]
        server.notify_tools_changed()
        engine.advance_step(state)
        await engine.astep(state, first.observation)

        assert registry.list_tools() == ["mcp__server_one__replacement"]
        await engine.aclose()
        assert registry.list_tools() == []
        assert server.cleaned_up is True

    @pytest.mark.asyncio
    async def test_sync_step_rejects_mcp_transport_loop_churn(self):
        server = FakeMCPServer()
        engine = Engine(
            FinalAgent(tool_registry=ToolRegistry()),
            mcp_server_factory=lambda: (server,),
        )
        state, observation = engine.init_session("stepwise")

        with pytest.raises(RuntimeError, match="await Engine.astep"):
            engine.step(state, observation)

        assert server.connected is False
        await engine.aclose()

    @pytest.mark.asyncio
    async def test_engine_permission_pipeline_can_deny_published_mcp_tool(self):
        server = FakeMCPServer(
            name="server.one",
            tools=[
                MCPToolInfo(
                    name="tool.two-three",
                    input_schema={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                )
            ],
        )
        agent = CallingAgent(tool_registry=ToolRegistry())

        result = await Engine(
            agent,
            mcp_server_factory=lambda: (server,),
            permission_pipeline=PermissionPipeline(
                context=ToolPermissionContext(default_decision="deny")
            ),
        ).arun("deny the remote call")

        denied = result.records[0].action_results[0]
        assert denied.status == "denied"
        assert denied.metadata["executed"] is False
        assert server.calls == []
        assert server.cleaned_up is True

    @pytest.mark.asyncio
    async def test_engine_mcp_connect_failure_doesnt_crash(self):
        """If an MCP server fails to connect, the engine should continue."""

        class FailingConnectServer(FakeMCPServer):
            async def connect(self) -> None:
                raise RuntimeError("Connection failed")

        bad_server = FailingConnectServer()
        result = await Engine(
            FinalAgent(tool_registry=ToolRegistry()),
            mcp_server_factory=lambda: (bad_server,),
        ).arun("complete without optional MCP")

        assert result.state.stop_reason == "completed"
        assert bad_server.cleaned_up is True

    @pytest.mark.asyncio
    async def test_mcp_cleanup_failure_is_visible_after_other_servers_close(self):
        """Cleanup attempts every server and preserves the first failure."""

        class FailingCleanupServer(FakeMCPServer):
            async def cleanup(self) -> None:
                self.cleaned_up = True
                raise RuntimeError("Cleanup failed")

        failing = FailingCleanupServer(name="failing")
        healthy = FakeMCPServer(name="healthy")
        runtime = MCPRuntime(
            tool_registry=ToolRegistry(),
            servers=[failing, healthy],
        )
        await runtime.start()

        with pytest.raises(RuntimeError, match="Cleanup failed"):
            await runtime.aclose()

        assert failing.cleaned_up is True
        assert healthy.cleaned_up is True

    @pytest.mark.asyncio
    async def test_engine_aclose_retries_failed_mcp_run_cleanup(self):
        class RetryCleanupServer(FakeMCPServer):
            def __init__(self) -> None:
                super().__init__()
                self.cleanup_calls = 0

            async def cleanup(self) -> None:
                self.cleanup_calls += 1
                if self.cleanup_calls == 1:
                    raise RuntimeError("transient MCP cleanup failure")
                await super().cleanup()

        server = RetryCleanupServer()
        engine = Engine(
            FinalAgent(tool_registry=ToolRegistry()),
            mcp_server_factory=lambda: (server,),
        )

        with pytest.raises(RuntimeError, match="transient MCP cleanup failure"):
            await engine.arun("complete")

        assert server.cleanup_calls == 1
        await engine.aclose()
        await engine.aclose()
        assert server.cleanup_calls == 2
        assert server.cleaned_up is True
