"""Run-owned MCP connection, catalog publication, and refresh lifecycle."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from mcp.types import Tool

from ..core.tool import BaseTool
from ..core.tool_registry import ToolRegistry
from .bridge import mcp_server_to_function_tools
from .server import MCPRequestError, MCPServer

_logger = logging.getLogger(__name__)
_T = TypeVar("_T")


@dataclass(slots=True)
class _ServerState:
    name: str
    server: MCPServer
    connected: bool = False
    cleanup_complete: bool = False
    tools: tuple[BaseTool, ...] = ()
    pending_generation: int | None = None

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(tool.name for tool in self.tools)


@dataclass(frozen=True, slots=True)
class _StartupDiscovery:
    state: _ServerState
    tool_infos: tuple[Tool, ...] = ()
    error: Exception | None = None


class MCPRuntime:
    """Own MCP transports and their published Tool catalogs for one Engine run.

    Refresh requests are only applied when :meth:`refresh_pending` is awaited by
    the Engine at a turn safe point. A failed refresh leaves that server's last
    complete catalog installed. Cancellation propagates without consuming the
    pending generation.
    """

    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        servers: Sequence[MCPServer],
        max_start_concurrency: int = 8,
        startup_timeout_seconds: float = 10.0,
    ) -> None:
        if not isinstance(tool_registry, ToolRegistry):
            raise TypeError("tool_registry must be a ToolRegistry")
        if (
            isinstance(max_start_concurrency, bool)
            or not isinstance(max_start_concurrency, int)
            or max_start_concurrency <= 0
        ):
            raise ValueError("max_start_concurrency must be a positive integer")
        if (
            isinstance(startup_timeout_seconds, bool)
            or not isinstance(startup_timeout_seconds, (int, float))
            or startup_timeout_seconds <= 0
        ):
            raise ValueError("startup_timeout_seconds must be positive")
        states: list[_ServerState] = []
        names: set[str] = set()
        identities: set[int] = set()
        for server in servers:
            if not isinstance(server, MCPServer):
                raise TypeError("servers must contain MCPServer values")
            name = server.name.strip()
            if not name:
                raise ValueError("MCP server name must be non-empty")
            if name in names:
                raise ValueError(f"duplicate MCP server name: {name}")
            if id(server) in identities:
                raise ValueError(
                    "the same MCP server instance cannot be registered twice"
                )
            names.add(name)
            identities.add(id(server))
            states.append(_ServerState(name=name, server=server))
        self._tool_registry = tool_registry
        self._states = tuple(states)
        self._states_by_name = {state.name: state for state in states}
        self._refresh_lock = asyncio.Lock()
        self._start_limit = asyncio.Semaphore(max_start_concurrency)
        self._startup_timeout_seconds = float(startup_timeout_seconds)
        self._generation = 0
        self._started = False
        self._closed = False

    @property
    def server_names(self) -> tuple[str, ...]:
        return tuple(state.name for state in self._states)

    @property
    def published_tool_names(self) -> tuple[str, ...]:
        return tuple(name for state in self._states for name in state.tool_names)

    @property
    def has_pending_refresh(self) -> bool:
        return any(state.pending_generation is not None for state in self._states)

    async def start(self, *, deadline_monotonic: float | None = None) -> None:
        """Discover servers concurrently, then publish catalogs deterministically."""

        if self._closed:
            raise RuntimeError("MCP runtime is closed")
        if self._started:
            raise RuntimeError("MCP runtime is already started")
        self._started = True
        try:
            tasks: list[asyncio.Task[_StartupDiscovery]] = []
            async with asyncio.TaskGroup() as group:
                for state in self._states:
                    tasks.append(
                        group.create_task(
                            self._discover_startup_catalog(
                                state,
                                run_deadline_monotonic=deadline_monotonic,
                            ),
                            name=f"qitos-mcp-start-{state.name}",
                        )
                    )
            discoveries = tuple(task.result() for task in tasks)
            for discovery in discoveries:
                state = discovery.state
                error = discovery.error
                if error is None:
                    try:
                        tools = await self._build_tools(
                            state,
                            discovery.tool_infos,
                        )
                        self._publish(state, tools)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        error = exc
                if error is not None:
                    await self._discard_failed_start(state)
                    _logger.warning(
                        "MCP server %s could not be exposed: %s",
                        state.name,
                        error,
                    )
        except BaseException:
            try:
                await self.aclose()
            except BaseException as cleanup_error:
                _logger.warning(
                    "MCP cleanup after startup failure also failed",
                    exc_info=cleanup_error,
                )
            raise

    async def _discover_startup_catalog(
        self,
        state: _ServerState,
        *,
        run_deadline_monotonic: float | None,
    ) -> _StartupDiscovery:
        deadline = time.monotonic() + self._startup_timeout_seconds
        if run_deadline_monotonic is not None:
            deadline = min(deadline, run_deadline_monotonic)

        async def discover() -> tuple[Tool, ...]:
            async with self._start_limit:
                state.server.set_tools_changed_handler(
                    self._refresh_handler(state.name)
                )
                await state.server.connect()
                state.connected = True
                state.cleanup_complete = False
                tool_infos = await state.server.list_tools()
                return tuple(tool_infos)

        try:
            tool_infos = await _await_with_deadline(
                discover,
                deadline_monotonic=deadline,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return _StartupDiscovery(state=state, error=exc)
        return _StartupDiscovery(state=state, tool_infos=tool_infos)

    def request_refresh(self, server_name: str | None = None) -> bool:
        """Mark one or all connected catalogs for the next turn safe point."""

        if self._closed:
            return False
        if server_name is None:
            selected = tuple(state for state in self._states if state.connected)
        else:
            state = self._states_by_name.get(server_name)
            selected = (state,) if state is not None and state.connected else ()
        if not selected:
            return False
        self._generation += 1
        for state in selected:
            state.pending_generation = self._generation
        return True

    def _refresh_handler(self, server_name: str) -> Callable[[], None]:
        def request_refresh() -> None:
            self.request_refresh(server_name)

        return request_refresh

    async def refresh_pending(
        self,
        *,
        deadline_monotonic: float | None = None,
    ) -> tuple[str, ...]:
        """Publish pending catalogs, retaining old catalogs on server failure."""

        if self._closed or not self._started:
            return ()
        published: list[str] = []
        async with self._refresh_lock:
            for state in self._states:
                generation = state.pending_generation
                if generation is None:
                    continue
                retry_pending = False
                try:
                    try:
                        tool_infos = await self._list_tools_for_refresh(
                            state,
                            deadline_monotonic=deadline_monotonic,
                        )
                    except MCPRequestError as exc:
                        if exc.error_code != "MCP_SESSION_EXPIRED":
                            raise
                        retry_pending = True
                        tool_infos = await self._reconnect_expired_session(
                            state,
                            deadline_monotonic=deadline_monotonic,
                        )
                    # Discovery failures consume the notification: repeatedly
                    # replaying a server-declared error at every turn would make
                    # no progress.  Once discovery succeeds, however, conversion
                    # and publication are local transactions and a failed one must
                    # stay pending until the complete catalog can be installed.
                    retry_pending = True
                    tools = await self._build_tools(state, tool_infos)
                    self._publish(state, tools)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    _logger.warning(
                        "MCP Tool refresh failed for %s; retaining its last catalog: %s",
                        state.name,
                        exc,
                    )
                else:
                    published.append(state.name)
                    retry_pending = False
                if state.pending_generation == generation and not retry_pending:
                    state.pending_generation = None
        return tuple(published)

    async def aclose(self) -> None:
        """Unpublish every owned Tool and close all transports exactly once."""

        if self._closed and all(state.cleanup_complete for state in self._states):
            return
        cancellation: asyncio.CancelledError | None = None
        cleanup_error: BaseException | None = None
        async with self._refresh_lock:
            self._closed = True
            for state in reversed(self._states):
                self._unpublish_owned(state)
            for state in reversed(self._states):
                if state.cleanup_complete:
                    continue
                try:
                    state.server.set_tools_changed_handler(None)
                    await state.server.cleanup()
                except asyncio.CancelledError as exc:
                    if cancellation is None:
                        cancellation = exc
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
                    else:
                        _logger.warning(
                            "Additional MCP server cleanup failure for %s",
                            state.name,
                            exc_info=exc,
                        )
                else:
                    state.cleanup_complete = True
                finally:
                    state.connected = False
                    state.pending_generation = None
        if cancellation is not None:
            raise cancellation
        if cleanup_error is not None:
            raise cleanup_error

    async def _build_tools(
        self,
        state: _ServerState,
        tool_infos: Sequence[Tool],
    ) -> tuple[BaseTool, ...]:
        if not isinstance(tool_infos, Sequence) or any(
            not isinstance(tool, Tool) for tool in tool_infos
        ):
            raise TypeError("MCP list_tools() must return mcp.types.Tool values")
        raw_names = [tool.name for tool in tool_infos]
        if len(raw_names) != len(set(raw_names)):
            raise ValueError(f"MCP server {state.name} returned duplicate Tool names")
        used_names = set(self._tool_registry.list_tools())
        used_names.difference_update(state.tool_names)
        tools = await mcp_server_to_function_tools(
            state.server,
            name_prefix=f"mcp__{state.name}",
            used_names=used_names,
            tool_infos=tool_infos,
        )
        return tuple(tools)

    async def _list_tools_for_refresh(
        self,
        state: _ServerState,
        *,
        deadline_monotonic: float | None,
    ) -> Sequence[Tool]:
        if not state.connected:
            await _await_with_deadline(
                state.server.connect,
                deadline_monotonic=deadline_monotonic,
            )
            state.connected = True
            state.cleanup_complete = False
            state.server.set_tools_changed_handler(self._refresh_handler(state.name))
        return await _await_with_deadline(
            state.server.list_tools,
            deadline_monotonic=deadline_monotonic,
        )

    async def _reconnect_expired_session(
        self,
        state: _ServerState,
        *,
        deadline_monotonic: float | None,
    ) -> Sequence[Tool]:
        """Recreate one expired transport before replaying only safe discovery."""

        await _await_with_deadline(
            state.server.cleanup,
            deadline_monotonic=deadline_monotonic,
        )
        state.connected = False
        state.cleanup_complete = True
        return await self._list_tools_for_refresh(
            state,
            deadline_monotonic=deadline_monotonic,
        )

    def _publish(
        self,
        state: _ServerState,
        tools: tuple[BaseTool, ...],
    ) -> None:
        old_tools = state.tools
        old_names = state.tool_names
        for expected in old_tools:
            current = self._tool_registry.get(expected.name)
            if current is not expected:
                raise RuntimeError(
                    f"MCP Tool ownership changed unexpectedly: {expected.name}"
                )
        old_name_set = set(old_names)
        for tool in tools:
            current = self._tool_registry.get(tool.name)
            if current is not None and tool.name not in old_name_set:
                raise ValueError(f"MCP Tool name conflicts with registry: {tool.name}")

        registered: list[str] = []
        try:
            for name in old_names:
                self._tool_registry.unregister(name)
            for tool in tools:
                self._tool_registry.register(tool)
                registered.append(tool.name)
        except BaseException:
            for name in reversed(registered):
                if self._tool_registry.get(name) is not None:
                    self._tool_registry.unregister(name)
            for tool in old_tools:
                if self._tool_registry.get(tool.name) is None:
                    self._tool_registry.register(tool)
            raise
        state.tools = tools

    def _unpublish_owned(self, state: _ServerState) -> None:
        for tool in reversed(state.tools):
            current = self._tool_registry.get(tool.name)
            if current is tool:
                self._tool_registry.unregister(tool.name)
            elif current is not None:
                _logger.warning(
                    "MCP Tool %s was replaced outside its owning runtime",
                    tool.name,
                )
        state.tools = ()

    async def _discard_failed_start(self, state: _ServerState) -> None:
        self._unpublish_owned(state)
        state.server.set_tools_changed_handler(None)
        try:
            await state.server.cleanup()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _logger.warning(
                "MCP server cleanup after setup failure failed for %s: %s",
                state.name,
                exc,
            )
        else:
            state.cleanup_complete = True
        finally:
            state.connected = False
            state.pending_generation = None


async def _await_with_deadline(
    operation: Callable[[], Awaitable[_T]],
    *,
    deadline_monotonic: float | None,
) -> _T:
    if deadline_monotonic is None:
        return await operation()
    remaining = float(deadline_monotonic) - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("MCP operation deadline expired")
    try:
        return await asyncio.wait_for(operation(), timeout=remaining)
    except asyncio.TimeoutError as exc:
        raise TimeoutError("MCP operation deadline expired") from exc


__all__ = ["MCPRuntime"]
