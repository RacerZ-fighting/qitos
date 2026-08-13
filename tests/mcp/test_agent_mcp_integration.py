"""Tests for AgentModule MCP integration."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qitos.core.action import Action
from qitos.core.agent_module import AgentModule
from qitos.core.decision import Decision
from qitos.core.state import StateSchema
from qitos.core.tool_registry import ToolRegistry
from qitos.engine.engine import Engine
from qitos.mcp.server import MCPServer, MCPToolInfo


class FakeMCPServer(MCPServer):
    """Fake MCP server for testing."""

    def __init__(self, name: str = "fake", tools: list | None = None):
        self._name = name
        self._tools = tools or [
            MCPToolInfo(
                name="read", description="Read a file", input_schema={"type": "object"}
            ),
        ]
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

    async def call_tool(self, name: str, arguments: dict) -> Any:
        self.calls.append((name, dict(arguments)))
        self.lifecycle_threads.append(threading.get_ident())
        return {"result": name, "arguments": arguments}


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


class TestAgentModuleMCPServers:
    def test_mcp_servers_default_empty(self):
        agent = DummyAgent()
        assert agent.mcp_servers == []

    def test_mcp_servers_passed_to_init(self):
        server = FakeMCPServer()
        agent = DummyAgent(mcp_servers=[server])
        assert len(agent.mcp_servers) == 1
        assert agent.mcp_servers[0] is server

    @pytest.mark.asyncio
    async def test_empty_mcp_configuration_starts_no_runtime(self):
        engine = Engine(DummyAgent(tool_registry=ToolRegistry()))

        await engine._connect_mcp_servers()

        assert engine._mcp_runtime is None
        assert engine._connected_mcp_servers == []
        assert engine.tool_registry.list_tools() == []

    @pytest.mark.asyncio
    async def test_engine_connects_mcp_servers_on_run(self):
        server = FakeMCPServer()
        agent = DummyAgent(tool_registry=ToolRegistry(), mcp_servers=[server])
        # Patch the engine's run loop to avoid actual execution
        engine = Engine(agent)

        # Mock the main run loop to just test MCP lifecycle
        with patch.object(engine, "_normalize_task", return_value=(None, "test task")):
            with patch.object(
                engine.agent, "init_state", return_value=DummyState(task="test")
            ):
                # Directly test connect/cleanup
                engine._connected_mcp_servers = []
                await engine._connect_mcp_servers()
                assert server.connected
                assert len(engine._connected_mcp_servers) == 1
                assert "mcp__fake__read" in engine.tool_registry

                await engine._cleanup_mcp_servers()
                assert server.cleaned_up
                assert engine._connected_mcp_servers == []
                assert "mcp__fake__read" not in engine.tool_registry

    @pytest.mark.asyncio
    async def test_engine_calls_raw_mcp_tool_and_registry_is_reusable(self):
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
        agent = CallingAgent(tool_registry=registry, mcp_servers=[server])

        first = await Engine(agent).arun("first")
        second = await Engine(agent).arun("second")

        assert first.state.stop_reason == "final"
        assert second.state.stop_reason == "final"
        assert server.calls == [
            ("tool.two-three", {"value": "evidence"}),
            ("tool.two-three", {"value": "evidence"}),
        ]
        assert len(server.lifecycle_threads) == 6
        assert len(set(server.lifecycle_threads[:3])) == 1
        assert len(set(server.lifecycle_threads[3:])) == 1
        assert server.lifecycle_threads[0] != threading.get_ident()
        assert registry.list_tools() == []
        assert server.cleaned_up

    @pytest.mark.asyncio
    async def test_engine_mcp_connect_failure_doesnt_crash(self):
        """If an MCP server fails to connect, the engine should continue."""
        bad_server = MagicMock()
        bad_server.connect = AsyncMock()
        bad_server.connect.side_effect = RuntimeError("Connection failed")
        bad_server.cleanup = AsyncMock()

        agent = DummyAgent(mcp_servers=[bad_server])
        engine = Engine(agent)
        engine._connected_mcp_servers = []
        await engine._connect_mcp_servers()

        # Should not have crashed, and the bad server should not be in connected list
        assert len(engine._connected_mcp_servers) == 0
        bad_server.cleanup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_engine_mcp_cleanup_failure_doesnt_crash(self):
        """If cleanup fails, other servers should still be cleaned up."""
        server1 = MagicMock()
        server1.cleanup = AsyncMock()
        server1.cleanup.side_effect = RuntimeError("Cleanup failed")
        server2 = MagicMock()
        server2.cleanup = AsyncMock()

        agent = DummyAgent(mcp_servers=[server1, server2])
        engine = Engine(agent)
        engine._connected_mcp_servers = [server1, server2]
        await engine._cleanup_mcp_servers()

        # Both should have been attempted
        server1.cleanup.assert_called_once()
        server2.cleanup.assert_called_once()
        assert engine._connected_mcp_servers == []
