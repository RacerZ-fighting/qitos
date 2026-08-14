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
import json
from collections.abc import Sequence
from typing import Any, List, Optional, Set

from mcp.types import CallToolResult, TextContent, Tool

from ..core.tool import FunctionTool, ToolMeta, ToolSpec
from ..core.tool_result import ToolResult
from .filter import ToolFilter
from .schema_convert import convert_mcp_schema_to_tool_spec
from .server import MCPRequestError, MCPServer


async def mcp_server_to_function_tools(
    server: MCPServer,
    tool_filter: Optional[ToolFilter] = None,
    name_prefix: Optional[str] = None,
    used_names: Optional[Set[str]] = None,
    tool_infos: Sequence[Tool] | None = None,
) -> List[FunctionTool]:
    """Convert all tools exposed by an MCP server into QitOS FunctionTools.

    :param server: A connected MCP server instance.
    :param tool_filter: Optional filter to include/exclude tools by name.
    :param name_prefix: Optional prefix to disambiguate tool names when
        multiple MCP servers are bridged into the same registry.  When
        provided, tool names become ``{prefix}__{original_name}``.
    :param used_names: Existing model-visible names to avoid when normalizing.
    :returns: A list of ``FunctionTool`` instances, one per MCP tool that
        passes the filter.
    """
    mcp_tools = (
        list(tool_infos) if tool_infos is not None else await server.list_tools()
    )
    tools: List[FunctionTool] = []
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
        tool = _make_function_tool(server, tool_name, spec)
        tools.append(tool)

    return tools


def _make_function_tool(
    server: MCPServer,
    original_name: str,
    spec: ToolSpec,
) -> FunctionTool:
    """Create a FunctionTool that delegates to ``server.call_tool``.

    The function wrapped by FunctionTool must accept keyword arguments
    matching the spec parameters, plus optional ``runtime_context``.
    The closure remains async so the MCP transport, Engine, and Tool share the
    caller's event loop and cancellation domain.
    """
    # Build a callable with the right parameter signature for FunctionTool.
    # FunctionTool inspects the function signature to build its own spec,
    # but we want to use *our* spec (from MCP schema conversion).  We
    # override by providing a ToolMeta that carries our custom spec fields.

    async def _async_wrapper(**kwargs: Any) -> Any:
        """Call the MCP transport on its owning event loop."""
        try:
            result = await server.call_tool(original_name, dict(kwargs))
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            raise
        except MCPRequestError as exc:
            return ToolResult(
                status="error",
                error=str(exc),
                metadata={
                    "error_category": exc.error_category,
                    "error_code": exc.error_code,
                    "mcp_server": server.name,
                    "mcp_tool": original_name,
                },
            )
        if not isinstance(result, CallToolResult):
            return ToolResult(
                status="error",
                error="MCP transport returned an invalid tools/call result",
                metadata={
                    "error_category": "mcp_protocol_error",
                    "error_code": "MCP_PROTOCOL_ERROR",
                    "mcp_server": server.name,
                    "mcp_tool": original_name,
                },
            )
        return _to_tool_result(
            result,
            server_name=server.name,
            tool_name=original_name,
        )

    # Attach metadata so FunctionTool uses our spec fields.
    meta = ToolMeta(
        name=spec.name,
        description=spec.description,
        input_schema=spec.input_schema,
        permissions=spec.permissions,
        read_only=spec.read_only,
        concurrency_safe=spec.concurrency_safe,
        group=spec.group,
    )

    tool = FunctionTool(_async_wrapper, meta=meta)
    # Override the spec with our MCP-derived spec (preserving all fields)
    tool.spec = spec
    return tool


def _to_tool_result(
    result: CallToolResult,
    *,
    server_name: str,
    tool_name: str,
) -> ToolResult:
    payload = result.model_dump(by_alias=True, mode="json", exclude_none=True)
    output: Any = payload
    if result.structuredContent is None and result.meta is None:
        if len(result.content) == 1 and isinstance(result.content[0], TextContent):
            text = result.content[0].text
            try:
                output = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                output = text

    metadata: dict[str, Any] = {
        "mcp_server": server_name,
        "mcp_tool": tool_name,
    }
    if result.isError:
        metadata.update(
            {
                "error_category": "mcp_tool_error",
                "error_code": "MCP_TOOL_ERROR",
            }
        )
        text_blocks = [
            block.text for block in result.content if isinstance(block, TextContent)
        ]
        return ToolResult(
            status="error",
            output=output,
            error="\n".join(text_blocks) or "MCP tool reported an error",
            metadata=metadata,
        )
    return ToolResult(status="success", output=output, metadata=metadata)


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
