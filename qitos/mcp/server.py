"""Typed MCP transport contracts shared by run-owned clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, List, TypeAlias

from ..core.tool_result import ToolResult

_MCPToolsChangedHandler = Callable[[], None]

_MAX_TOOL_CATALOG_PAGES = 100
_MAX_TOOL_CATALOG_ITEMS = 2_048
_MAX_PAGINATION_CURSOR_BYTES = 64 * 1024


class MCPRequestError(RuntimeError):
    """A failed MCP request with a stable tool-result classification."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "MCP_REQUEST_ERROR",
        error_category: str = "mcp_request_error",
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.error_category = error_category


@dataclass(frozen=True, slots=True)
class MCPToolAnnotations:
    """Standard MCP Tool annotations without treating hints as guarantees."""

    title: str | None = None
    read_only_hint: bool | None = None
    destructive_hint: bool | None = None
    idempotent_hint: bool | None = None
    open_world_hint: bool | None = None

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | None,
    ) -> "MCPToolAnnotations":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("MCP Tool annotations must be an object")
        title = payload.get("title")
        if title is not None and not isinstance(title, str):
            raise TypeError("MCP Tool annotation title must be a string or null")

        def _hint(name: str) -> bool | None:
            value = payload.get(name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"MCP Tool annotation {name} must be boolean or null")
            return value

        return cls(
            title=title,
            read_only_hint=_hint("readOnlyHint"),
            destructive_hint=_hint("destructiveHint"),
            idempotent_hint=_hint("idempotentHint"),
            open_world_hint=_hint("openWorldHint"),
        )

    def to_dict(self) -> dict[str, Any]:
        values = {
            "title": self.title,
            "readOnlyHint": self.read_only_hint,
            "destructiveHint": self.destructive_hint,
            "idempotentHint": self.idempotent_hint,
            "openWorldHint": self.open_world_hint,
        }
        return {name: value for name, value in values.items() if value is not None}


@dataclass(frozen=True, slots=True)
class MCPToolInfo:
    """Descriptor for a single tool exposed by an MCP server.

    Mirrors the ``tools/list`` response item from the MCP specification:
    each tool has a name, a human-readable description, and a JSON Schema
    describing its input parameters.
    """

    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    annotations: MCPToolAnnotations = field(default_factory=MCPToolAnnotations)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("MCP Tool name must be a non-empty string")
        if not isinstance(self.description, str):
            raise TypeError("MCP Tool description must be a string")
        if not isinstance(self.input_schema, dict):
            raise TypeError("MCP Tool inputSchema must be an object")
        if not isinstance(self.annotations, MCPToolAnnotations):
            raise TypeError("annotations must be MCPToolAnnotations")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MCPToolInfo":
        if not isinstance(payload, Mapping):
            raise TypeError("MCP Tool descriptor must be an object")
        name = payload.get("name")
        description = payload.get("description", "")
        input_schema = payload.get("inputSchema", {})
        if not isinstance(name, str):
            raise TypeError("MCP Tool name must be a string")
        if not isinstance(description, str):
            raise TypeError("MCP Tool description must be a string")
        if not isinstance(input_schema, Mapping):
            raise TypeError("MCP Tool inputSchema must be an object")
        return cls(
            name=name,
            description=description,
            input_schema=dict(input_schema),
            annotations=MCPToolAnnotations.from_dict(payload.get("annotations")),
        )


@dataclass(frozen=True, slots=True)
class MCPCallToolResult:
    """Typed MCP ``tools/call`` result before QitOS Tool projection."""

    content: tuple[dict[str, Any], ...]
    structured_content: Any = None
    is_error: bool | None = None
    meta: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MCPCallToolResult":
        if not isinstance(payload, Mapping):
            raise MCPRequestError(
                "MCP tools/call result must be an object",
                error_code="MCP_PROTOCOL_ERROR",
                error_category="mcp_protocol_error",
            )
        content = payload.get("content")
        if not isinstance(content, list):
            raise MCPRequestError(
                "MCP tools/call result must contain a content array",
                error_code="MCP_PROTOCOL_ERROR",
                error_category="mcp_protocol_error",
            )
        normalized_content: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, Mapping):
                raise MCPRequestError(
                    "MCP tools/call content blocks must be objects",
                    error_code="MCP_PROTOCOL_ERROR",
                    error_category="mcp_protocol_error",
                )
            normalized_content.append(dict(block))

        is_error = payload.get("isError")
        if is_error is not None and not isinstance(is_error, bool):
            raise MCPRequestError(
                "MCP tools/call isError must be boolean or null",
                error_code="MCP_PROTOCOL_ERROR",
                error_category="mcp_protocol_error",
            )
        meta = payload.get("_meta")
        if meta is not None and not isinstance(meta, Mapping):
            raise MCPRequestError(
                "MCP tools/call _meta must be an object or null",
                error_code="MCP_PROTOCOL_ERROR",
                error_category="mcp_protocol_error",
            )
        return cls(
            content=tuple(normalized_content),
            structured_content=payload.get("structuredContent"),
            is_error=is_error,
            meta=dict(meta) if meta is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "content": [dict(block) for block in self.content],
        }
        if self.structured_content is not None:
            payload["structuredContent"] = self.structured_content
        if self.is_error is not None:
            payload["isError"] = self.is_error
        if self.meta is not None:
            payload["_meta"] = dict(self.meta)
        return payload

    def to_tool_result(self, *, server_name: str, tool_name: str) -> ToolResult:
        """Project the remote result without confusing ``isError`` with success."""

        output = self._tool_output()
        metadata: dict[str, Any] = {
            "mcp_server": server_name,
            "mcp_tool": tool_name,
        }
        if self.is_error is True:
            metadata.update(
                {
                    "error_category": "mcp_tool_error",
                    "error_code": "MCP_TOOL_ERROR",
                }
            )
            return ToolResult(
                status="error",
                output=output,
                error=self._error_text(),
                metadata=metadata,
            )
        return ToolResult(status="success", output=output, metadata=metadata)

    def _tool_output(self) -> Any:
        if self.structured_content is None and self.meta is None:
            text = self._single_text()
            if text is not None:
                try:
                    return json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    return text
        return self.to_dict()

    def _single_text(self) -> str | None:
        if len(self.content) != 1:
            return None
        block = self.content[0]
        text = block.get("text")
        if block.get("type") == "text" and isinstance(text, str):
            return text
        return None

    def _error_text(self) -> str:
        text_blocks = [
            str(block["text"])
            for block in self.content
            if block.get("type") == "text" and isinstance(block.get("text"), str)
        ]
        return "\n".join(text_blocks) or "MCP tool reported an error"


async def _collect_mcp_tool_catalog(
    fetch_page: Callable[[dict[str, Any]], Awaitable[Mapping[str, Any]]],
) -> list[MCPToolInfo]:
    """Collect a bounded ``tools/list`` catalog across cursor pages."""

    catalog: list[MCPToolInfo] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()

    for _ in range(_MAX_TOOL_CATALOG_PAGES):
        params = {} if cursor is None else {"cursor": cursor}
        result = await fetch_page(params)
        tools_data = result.get("tools")
        if not isinstance(tools_data, list):
            raise MCPRequestError(
                "MCP tools/list result must contain a tools array",
                error_code="MCP_PROTOCOL_ERROR",
                error_category="mcp_protocol_error",
            )
        if len(tools_data) > _MAX_TOOL_CATALOG_ITEMS - len(catalog):
            raise MCPRequestError(
                f"MCP tools/list exceeded {_MAX_TOOL_CATALOG_ITEMS} tools",
                error_code="MCP_CATALOG_LIMIT",
                error_category="mcp_protocol_error",
            )
        catalog.extend(MCPToolInfo.from_dict(tool) for tool in tools_data)

        next_cursor = result.get("nextCursor")
        if next_cursor is None:
            return catalog
        if not isinstance(next_cursor, str):
            raise MCPRequestError(
                "MCP tools/list nextCursor must be a string or null",
                error_code="MCP_PROTOCOL_ERROR",
                error_category="mcp_protocol_error",
            )
        if len(next_cursor.encode("utf-8")) > _MAX_PAGINATION_CURSOR_BYTES:
            raise MCPRequestError(
                "MCP tools/list pagination cursor is too large",
                error_code="MCP_CATALOG_LIMIT",
                error_category="mcp_protocol_error",
            )
        if next_cursor in seen_cursors:
            raise MCPRequestError(
                "MCP tools/list returned a repeated pagination cursor",
                error_code="MCP_PROTOCOL_ERROR",
                error_category="mcp_protocol_error",
            )
        seen_cursors.add(next_cursor)
        cursor = next_cursor

    raise MCPRequestError(
        f"MCP tools/list exceeded {_MAX_TOOL_CATALOG_PAGES} pages",
        error_code="MCP_CATALOG_LIMIT",
        error_category="mcp_protocol_error",
    )


class MCPServer(ABC):
    """Abstract base for MCP server transports.

    Subclasses implement the transport-specific details (stdio, HTTP, etc.)
    while this contract defines the lifecycle that consumers rely on:

    1. ``connect()``  -- establish the transport and perform the MCP handshake.
    2. ``list_tools()`` -- discover available tools.
    3. ``call_tool()``  -- invoke a tool by name with JSON-serialisable arguments.
    4. ``cleanup()``    -- tear down the transport gracefully.
    """

    _tools_changed_handler: _MCPToolsChangedHandler | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this server connection."""

    @abstractmethod
    async def connect(self) -> None:
        """Open the transport and complete the MCP initialization handshake.

        After this method returns, the server is ready to accept
        ``list_tools`` and ``call_tool`` requests.
        """

    @abstractmethod
    async def cleanup(self) -> None:
        """Shut down the transport and release all resources."""

    @abstractmethod
    async def list_tools(self) -> List[MCPToolInfo]:
        """Return the list of tools exposed by the connected MCP server.

        Corresponds to the ``tools/list`` MCP method.
        """

    @abstractmethod
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> MCPCallToolResult:
        """Invoke a tool on the connected MCP server.

        Corresponds to the ``tools/call`` MCP method.

        :param tool_name: The exact tool name as returned by ``list_tools``.
        :param arguments: JSON-serialisable dict of argument values.
        :returns: The typed MCP result, including remote ``isError`` state.
        """

    def set_tools_changed_handler(
        self,
        handler: _MCPToolsChangedHandler | None,
    ) -> None:
        """Install the run owner's lightweight Tool-catalog invalidation hook."""

        self._tools_changed_handler = handler

    def notify_tools_changed(self) -> None:
        """Mark this server's catalog dirty after a protocol notification."""

        handler = self._tools_changed_handler
        if handler is not None:
            handler()


MCPServerFactory: TypeAlias = Callable[[], Sequence[MCPServer]]
"""Create fresh, unconnected MCP transports for one Engine run."""


__all__ = [
    "MCPCallToolResult",
    "MCPRequestError",
    "MCPServer",
    "MCPServerFactory",
    "MCPToolAnnotations",
    "MCPToolInfo",
]
