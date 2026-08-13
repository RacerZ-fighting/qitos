"""Bridge MCP server tools into QitOS FunctionTool instances.

The bridge is the key integration point: it discovers tools from an MCP
server, converts their JSON Schema into QitOS ``ToolSpec`` objects, and
wraps each one in a ``FunctionTool`` whose ``execute`` method calls the
MCP server remotely.

Usage::

    from qitos.mcp import MCPServerStdio, mcp_server_to_function_tools, ToolFilter

    server = MCPServerStdio(command="npx", args=["-y", "@mcp/server-fs", "/tmp"])
    await server.connect()

    tools = await mcp_server_to_function_tools(
        server,
        tool_filter=ToolFilter(blocked_tool_names={"dangerous_op"}),
        name_prefix="fs",
    )
    # tools is a list of FunctionTool instances ready to register
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any, Callable, Coroutine, List, Optional, Set

from ..core.tool import FunctionTool, ToolMeta, ToolSpec
from .filter import ToolFilter
from .schema_convert import convert_mcp_schema_to_tool_spec
from .server import MCPServer


async def mcp_server_to_function_tools(
    server: MCPServer,
    tool_filter: Optional[ToolFilter] = None,
    name_prefix: Optional[str] = None,
    call_runner: Optional[Callable[[Coroutine[Any, Any, Any]], Any]] = None,
    used_names: Optional[Set[str]] = None,
) -> List[FunctionTool]:
    """Convert all tools exposed by an MCP server into QitOS FunctionTools.

    :param server: A connected MCP server instance.
    :param tool_filter: Optional filter to include/exclude tools by name.
    :param name_prefix: Optional prefix to disambiguate tool names when
        multiple MCP servers are bridged into the same registry.  When
        provided, tool names become ``{prefix}__{original_name}``.
    :param call_runner: Synchronous owner for transport coroutines. Engine
        integrations should provide a long-lived MCP event-loop runtime.
    :param used_names: Existing model-visible names to avoid when normalizing.
    :returns: A list of ``FunctionTool`` instances, one per MCP tool that
        passes the filter.
    """
    mcp_tools = await server.list_tools()
    tools: List[FunctionTool] = []
    owner_loop = asyncio.get_running_loop()
    runner = call_runner or _owner_loop_runner(owner_loop)
    occupied = used_names if used_names is not None else set()

    for mcp_tool in mcp_tools:
        # Apply filter
        if tool_filter is not None and not tool_filter.matches(mcp_tool.name):
            continue

        # Convert schema
        spec = convert_mcp_schema_to_tool_spec(mcp_tool)
        spec.name = _unique_model_tool_name(
            name_prefix=name_prefix,
            raw_tool_name=mcp_tool.name,
            used_names=occupied,
        )

        # Create a closure that captures the server and original tool name
        tool_name = mcp_tool.name
        tool = _make_function_tool(server, tool_name, spec, runner)
        tools.append(tool)

    return tools


def _make_function_tool(
    server: MCPServer,
    original_name: str,
    spec: ToolSpec,
    call_runner: Callable[[Coroutine[Any, Any, Any]], Any],
) -> FunctionTool:
    """Create a FunctionTool that delegates to ``server.call_tool``.

    The function wrapped by FunctionTool must accept keyword arguments
    matching the spec parameters, plus optional ``runtime_context``.
    Since the actual MCP call is async but FunctionTool.execute is
    synchronous, the supplied runner submits it to the transport's owner loop.
    """
    # Build a callable with the right parameter signature for FunctionTool.
    # FunctionTool inspects the function signature to build its own spec,
    # but we want to use *our* spec (from MCP schema conversion).  We
    # override by providing a ToolMeta that carries our custom spec fields.

    def _sync_wrapper(**kwargs: Any) -> Any:
        """Synchronous wrapper that runs the async MCP call."""
        call_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ("runtime_context", "env", "ops", "file_ops", "process_ops")
        }
        return call_runner(server.call_tool(original_name, call_kwargs))

    # Attach metadata so FunctionTool uses our spec fields.
    meta = ToolMeta(
        name=spec.name,
        description=spec.description,
        input_schema=spec.input_schema,
        read_only=spec.read_only,
        concurrency_safe=spec.concurrency_safe,
    )

    tool = FunctionTool(_sync_wrapper, meta=meta)
    # Override the spec with our MCP-derived spec (preserving all fields)
    tool.spec = spec
    return tool


def _owner_loop_runner(
    owner_loop: asyncio.AbstractEventLoop,
) -> Callable[[Coroutine[Any, Any, Any]], Any]:
    """Build a safe fallback runner for manually bridged tools."""

    def _run(coroutine: Coroutine[Any, Any, Any]) -> Any:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is owner_loop:
            coroutine.close()
            raise RuntimeError(
                "MCP tool execution must not block its transport event loop"
            )
        if owner_loop.is_closed() or not owner_loop.is_running():
            coroutine.close()
            raise RuntimeError("MCP transport event loop is not running")
        return asyncio.run_coroutine_threadsafe(coroutine, owner_loop).result()

    return _run


def _sanitize_model_tool_name(value: str) -> str:
    """Match provider-safe MCP naming by replacing unsupported characters."""
    # Keep raw protocol names separate from the model-visible projection, as in
    # codex:codex-mcp/src/tools.rs and codex:codex-mcp/src/mcp/mod.rs.
    sanitized = "".join(
        char if char.isascii() and (char.isalnum() or char == "_") else "_"
        for char in value
    )
    return sanitized or "_"


def _name_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]


def _fit_hashed_name(base: str, identity: str, attempt: int = 0) -> str:
    hash_input = identity if attempt == 0 else f"{identity}\0{attempt}"
    suffix = f"_{_name_hash(hash_input)}"
    return f"{base[: 64 - len(suffix)]}{suffix}"


def _unique_model_tool_name(
    *,
    name_prefix: Optional[str],
    raw_tool_name: str,
    used_names: Set[str],
) -> str:
    """Return one provider-safe, deterministic, <=64-character tool name."""
    raw_name = f"{name_prefix}__{raw_tool_name}" if name_prefix else raw_tool_name
    base = _sanitize_model_tool_name(raw_name)
    if len(base) <= 64 and base not in used_names:
        used_names.add(base)
        return base

    attempt = 0
    while True:
        candidate = _fit_hashed_name(base, raw_name, attempt)
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        attempt += 1
