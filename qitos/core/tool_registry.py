"""Canonical tool registry with function and ToolSet support."""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import inspect
import json
import logging
import re
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    cast,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    runtime_checkable,
)

from .tool import BaseTool, FunctionTool, ToolMeta, get_tool_meta

if TYPE_CHECKING:
    from .tool import ToolPermissionSpec


_logger = logging.getLogger(__name__)


async def _await_lifecycle_result(
    result: object,
    *,
    owner: object,
    method: str,
) -> None:
    if not inspect.isawaitable(result):
        raise TypeError(f"{type(owner).__name__}.{method}() must return an awaitable")
    await cast(Awaitable[object], result)


@runtime_checkable
class _SyncSetup(Protocol):
    def setup(self, context: Dict[str, Any]) -> None:
        raise NotImplementedError


@runtime_checkable
class _AsyncSetup(Protocol):
    async def asetup(self, context: Dict[str, Any]) -> None:
        raise NotImplementedError


@runtime_checkable
class _SyncTeardown(Protocol):
    def teardown(self, context: Dict[str, Any]) -> None:
        raise NotImplementedError


@runtime_checkable
class _AsyncTeardown(Protocol):
    async def ateardown(self, context: Dict[str, Any]) -> None:
        raise NotImplementedError


@runtime_checkable
class _AsyncClose(Protocol):
    async def aclose(self) -> object:
        raise NotImplementedError


@dataclass(frozen=True)
class ToolOrigin:
    source: str  # function | toolset
    toolset_name: Optional[str] = None
    toolset_version: Optional[str] = None


class _FrozenTool(BaseTool):
    """One turn's immutable Tool definition bound to its live handler owner."""

    def __init__(self, handler: BaseTool) -> None:
        self._handler = handler
        self.spec = deepcopy(handler.spec)
        if hasattr(handler, "meta"):
            self.meta = deepcopy(handler.meta)

    def validate_input(
        self,
        args: Dict[str, Any],
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return self._handler.validate_input(args, runtime_context=runtime_context)

    def check_permissions(
        self,
        args: Dict[str, Any],
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return self._handler.check_permissions(args, runtime_context=runtime_context)

    def build_rule_scope(self, args: Dict[str, Any]) -> str:
        return self._handler.build_rule_scope(args)

    async def execute(
        self,
        args: Dict[str, Any],
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return await self._handler.execute(args, runtime_context=runtime_context)


class ToolRegistry:
    """Registry for function tools, bound methods, tool objects, and ToolSets."""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}
        self._origins: Dict[str, ToolOrigin] = {}
        self._toolsets: List[Any] = []
        self._setup_done: bool = False
        self._revision: int = 0

    @property
    def revision(self) -> int:
        """Return the monotonic revision of this registry's membership."""

        return self._revision

    def freeze(
        self,
        names: Iterable[str] | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ToolExposure":
        """Capture an immutable model-and-execution view of selected tools."""

        selected = self.list_tools() if names is None else sorted(set(names))
        missing = [name for name in selected if name not in self._tools]
        if missing:
            raise ValueError(f"Cannot expose unknown tools: {missing}")
        return ToolExposure._capture(self, selected, metadata=metadata)

    def register(
        self, item: Any, name: Optional[str] = None, meta: Optional[ToolMeta] = None
    ) -> "ToolRegistry":
        tool_obj = self._to_tool(item, meta=meta)
        if name is not None:
            tool_obj = copy(tool_obj)
            tool_obj.spec = deepcopy(tool_obj.spec)
            tool_obj.spec.name = str(name)
        self._register_tool_object(tool_obj, origin=ToolOrigin(source="function"))
        return self

    def register_toolset(
        self, toolset: Any, namespace: Optional[str] = None
    ) -> "ToolRegistry":
        if not hasattr(toolset, "tools"):
            raise TypeError("register_toolset() expects an object with tools()")

        toolset_name = str(getattr(toolset, "name", toolset.__class__.__name__.lower()))
        toolset_version = str(getattr(toolset, "version", "0"))
        prefix = namespace if namespace is not None else toolset_name

        if toolset not in self._toolsets:
            self._toolsets.append(toolset)

        for item in toolset.tools():
            tool_obj = self._to_tool(item)
            base_name = tool_obj.spec.name
            full_name = f"{prefix}.{base_name}" if prefix else base_name
            # Create a shallow copy to avoid mutating the original tool's spec
            # when adding the namespace prefix. Without this, registering the
            # same toolset under different namespaces would corrupt the spec.
            named_tool = copy(tool_obj)
            named_tool.spec = deepcopy(tool_obj.spec)
            named_tool.spec.name = full_name
            self._register_tool_object(
                named_tool,
                origin=ToolOrigin(
                    source="toolset",
                    toolset_name=toolset_name,
                    toolset_version=toolset_version,
                ),
            )

        return self

    def include_toolset(self, items: Any) -> "ToolRegistry":
        """Include tools, toolsets, registries, or nested collections as one bundle.

        This is the default composition-oriented API for end users. It accepts:

        - one atomic tool
        - one toolset object with ``tools()``
        - one existing ``ToolRegistry``
        - a nested list/tuple/set containing any mix of the above
        """
        for item in self._iter_toolset_items(items):
            self._include_toolset_item(item)
        return self

    def include(self, obj: Any) -> "ToolRegistry":
        if hasattr(obj, "tools") and callable(getattr(obj, "tools")):
            if obj not in self._toolsets:
                self._toolsets.append(obj)
            for item in obj.tools():
                self.register(item)
            return self
        for attr_name in dir(obj):
            if attr_name.startswith("_"):
                continue
            attr = getattr(obj, attr_name)
            if isinstance(attr, BaseTool):
                self.register(attr)
                continue
            if not callable(attr):
                continue

            meta = get_tool_meta(attr)
            if meta is not None:
                self.register(attr, meta=meta)

        return self

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def unregister(self, name: str) -> BaseTool:
        """Remove one exact registered tool and return it.

        Dynamic tool providers use this to keep their run-scoped registrations
        from leaking into a reused registry. Unknown names fail explicitly.
        """
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found")
        tool = self._tools.pop(name)
        self._origins.pop(name, None)
        self._revision += 1
        return tool

    def suggest(self, name: str, limit: int = 3) -> List[str]:
        needle = str(name or "").strip()
        if not needle:
            return []
        candidates = self.list_tools()
        normalized_candidates = {self._normalize_tool_key(x): x for x in candidates}
        normalized_needle = self._normalize_tool_key(needle)
        ordered: List[str] = []
        if normalized_needle in normalized_candidates:
            ordered.append(normalized_candidates[normalized_needle])
        close = difflib.get_close_matches(
            needle, candidates, n=max(1, int(limit)), cutoff=0.5
        )
        for item in close:
            if item not in ordered:
                ordered.append(item)
        close_norm = difflib.get_close_matches(
            normalized_needle,
            list(normalized_candidates.keys()),
            n=max(1, int(limit)),
            cutoff=0.5,
        )
        for key in close_norm:
            name_item = normalized_candidates.get(key)
            if name_item and name_item not in ordered:
                ordered.append(name_item)
        return ordered[: max(1, int(limit))]

    def list_tools(self) -> List[str]:
        return sorted(self._tools.keys())

    def list_toolsets(self) -> List[str]:
        names: List[str] = []
        for toolset in self._toolsets:
            names.append(
                str(getattr(toolset, "name", toolset.__class__.__name__.lower()))
            )
        return names

    def describe_tool(self, name: str) -> Dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' not found")
        origin = self._origins.get(name, ToolOrigin(source="function"))
        return {
            "name": tool.name,
            "description": tool.spec.description,
            "group": tool.spec.group,
            "prompt": tool.spec.prompt,
            "required_ops": list(tool.spec.required_ops),
            "environment_ops": list(tool.spec.environment_ops),
            "input_schema": deepcopy(tool.spec.input_schema or {}),
            "output_schema": deepcopy(tool.spec.output_schema or {}),
            "read_only": bool(tool.spec.read_only),
            "concurrency_safe": bool(tool.spec.concurrency_safe),
            "requires_user_interaction": bool(tool.spec.requires_user_interaction),
            "supports_background": bool(tool.spec.supports_background),
            "produces_artifact": bool(tool.spec.produces_artifact),
            "origin": {
                "source": origin.source,
                "toolset_name": origin.toolset_name,
                "toolset_version": origin.toolset_version,
            },
        }

    def setup(self, context: Optional[Dict[str, Any]] = None) -> None:
        if self._setup_done:
            return
        payload = context or {}
        seen: set[int] = set()
        for tool in self._tools.values():
            identity = id(tool)
            if identity in seen:
                continue
            seen.add(identity)
            if isinstance(tool, _SyncSetup):
                tool.setup(payload)
        for toolset in self._toolsets:
            if isinstance(toolset, _SyncSetup):
                toolset.setup(payload)
        self._setup_done = True

    async def asetup(self, context: Optional[Dict[str, Any]] = None) -> None:
        """Set up toolsets without blocking the caller's event loop."""

        if self._setup_done:
            return
        payload = context or {}
        seen: set[int] = set()
        try:
            for tool in self._tools.values():
                identity = id(tool)
                if identity in seen:
                    continue
                seen.add(identity)
                if isinstance(tool, _AsyncSetup):
                    await _await_lifecycle_result(
                        tool.asetup(payload),
                        owner=tool,
                        method="asetup",
                    )
            for toolset in self._toolsets:
                if isinstance(toolset, _AsyncSetup):
                    await _await_lifecycle_result(
                        toolset.asetup(payload),
                        owner=toolset,
                        method="asetup",
                    )
                    continue
                if isinstance(toolset, _SyncSetup):
                    await asyncio.to_thread(toolset.setup, payload)
        except BaseException:
            try:
                await self.ateardown(payload)
            except BaseException as cleanup_error:
                _logger.warning(
                    "ToolRegistry cleanup after setup failure also failed",
                    exc_info=cleanup_error,
                )
            raise
        self._setup_done = True

    def teardown(self, context: Optional[Dict[str, Any]] = None) -> None:
        payload = context or {}
        for toolset in reversed(self._toolsets):
            if isinstance(toolset, _SyncTeardown):
                toolset.teardown(payload)
        self._setup_done = False

    async def ateardown(self, context: Optional[Dict[str, Any]] = None) -> None:
        """Await tool-owned tasks before tearing down legacy toolsets."""

        payload = context or {}
        seen: set[int] = set()
        cleanup_error: BaseException | None = None
        cancellation: asyncio.CancelledError | None = None
        for tool in reversed(list(self._tools.values())):
            identity = id(tool)
            if identity in seen:
                continue
            seen.add(identity)
            if isinstance(tool, _AsyncClose):
                try:
                    await _await_lifecycle_result(
                        tool.aclose(),
                        owner=tool,
                        method="aclose",
                    )
                except asyncio.CancelledError as exc:
                    if cancellation is None:
                        cancellation = exc
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
                    else:
                        _logger.warning(
                            "Additional Tool cleanup failure",
                            exc_info=exc,
                        )
        for toolset in reversed(self._toolsets):
            try:
                if isinstance(toolset, _AsyncTeardown):
                    await _await_lifecycle_result(
                        toolset.ateardown(payload),
                        owner=toolset,
                        method="ateardown",
                    )
                    continue
                if isinstance(toolset, _SyncTeardown):
                    await asyncio.to_thread(toolset.teardown, payload)
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
                else:
                    _logger.warning(
                        "Additional ToolSet cleanup failure",
                        exc_info=exc,
                    )
        self._setup_done = False
        if cancellation is not None:
            raise cancellation
        if cleanup_error is not None:
            raise cleanup_error

    def get_tool_descriptions(self, protocol: Any = None, renderer: Any = None) -> str:
        if renderer is not None:
            return str(renderer(self))
        if protocol is not None:
            try:
                from qitos.protocols import render_protocol_tool_schema

                return render_protocol_tool_schema(self, protocol)
            except Exception:
                pass
        lines: List[str] = []
        for name in self.list_tools():
            tool = self._tools[name]
            origin = self._origins.get(name, ToolOrigin(source="function"))
            lines.append(f"## {tool.name}")
            lines.append(f"Description: {tool.spec.description}")
            lines.append(f"Source: {origin.source}")
            if tool.spec.required_ops:
                lines.append(f"Required Ops: {', '.join(tool.spec.required_ops)}")
            if tool.spec.environment_ops:
                lines.append(f"Environment Ops: {', '.join(tool.spec.environment_ops)}")
            if tool.spec.group != "default":
                lines.append(f"Group: {tool.spec.group}")
            if origin.toolset_name:
                lines.append(f"ToolSet: {origin.toolset_name}@{origin.toolset_version}")
            lines.append("Parameters:")
            for param, p_spec in tool.spec.parameters.items():
                t = p_spec.get("type", "any")
                lines.append(f"  - {param} ({t})")
            lines.append("")
        return "\n".join(lines)

    def render_tool_schema(self, protocol: Any = None, renderer: Any = None) -> str:
        return self.get_tool_descriptions(protocol=protocol, renderer=renderer)

    def get_all_specs(self) -> List[Dict[str, Any]]:
        specs = []
        for name in self.list_tools():
            tool = self._tools[name]
            origin = self._origins.get(name, ToolOrigin(source="function"))
            specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.spec.name,
                        "description": tool.spec.description,
                        "parameters": deepcopy(tool.spec.input_schema)
                        or {
                            "type": "object",
                            "properties": deepcopy(tool.spec.parameters),
                            "required": list(tool.spec.required),
                        },
                        "output_schema": deepcopy(tool.spec.output_schema),
                    },
                    "group": tool.spec.group,
                    "origin": {
                        "source": origin.source,
                        "toolset_name": origin.toolset_name,
                        "toolset_version": origin.toolset_version,
                    },
                    "permissions": {
                        "filesystem_read": tool.spec.permissions.filesystem_read,
                        "filesystem_write": tool.spec.permissions.filesystem_write,
                        "network": tool.spec.permissions.network,
                        "command": tool.spec.permissions.command,
                    },
                    "required_ops": list(tool.spec.required_ops),
                    "environment_ops": list(tool.spec.environment_ops),
                    "capabilities": {
                        "read_only": bool(tool.spec.read_only),
                        "concurrency_safe": bool(tool.spec.concurrency_safe),
                        "requires_user_interaction": bool(
                            tool.spec.requires_user_interaction
                        ),
                        "supports_background": bool(tool.spec.supports_background),
                        "produces_artifact": bool(tool.spec.produces_artifact),
                    },
                }
            )
        return specs

    def export_permissions(self) -> List["ToolPermissionSpec"]:
        """Return a ToolPermissionSpec for each registered tool.

        This is a QitOS-native alternative to ``get_all_specs()`` that
        focuses on permission and capability metadata without the
        OpenAI-style schema wrapping.
        """
        from .tool import ToolPermissionSpec

        specs: List[ToolPermissionSpec] = []
        for name in self.list_tools():
            tool = self._tools[name]
            specs.append(
                ToolPermissionSpec(
                    name=tool.spec.name,
                    description=tool.spec.description,
                    group=tool.spec.group,
                    permissions=deepcopy(tool.spec.permissions),
                    needs_approval=tool.spec.needs_approval,
                    read_only=tool.spec.read_only,
                    concurrency_safe=tool.spec.concurrency_safe,
                    required_ops=list(tool.spec.required_ops),
                    environment_ops=list(tool.spec.environment_ops),
                )
            )
        return specs

    def _to_tool(self, item: Any, meta: Optional[ToolMeta] = None) -> BaseTool:
        if isinstance(item, BaseTool):
            return item
        if callable(item):
            return FunctionTool(item, meta=meta or get_tool_meta(item))
        raise TypeError("register() expects BaseTool or callable")

    def _register_tool_object(self, tool_obj: BaseTool, origin: ToolOrigin) -> None:
        if tool_obj.name in self._tools:
            raise ValueError(f"Tool name collision: '{tool_obj.name}'")
        self._tools[tool_obj.name] = tool_obj
        self._origins[tool_obj.name] = origin
        self._revision += 1

    def _include_toolset_item(self, item: Any) -> None:
        if item is None:
            return
        if isinstance(item, ToolRegistry):
            self._merge_registry(item)
            return
        if isinstance(item, BaseTool) or callable(item):
            self.register(item)
            return
        if hasattr(item, "tools") and callable(getattr(item, "tools")):
            if item not in self._toolsets:
                self._toolsets.append(item)
            for nested in item.tools():
                self._include_toolset_item(nested)
            return
        raise TypeError(
            "include_toolset() accepts tools, toolsets, registries, or nested collections"
        )

    def _iter_toolset_items(self, items: Any) -> Iterable[Any]:
        if isinstance(items, (list, tuple, set)):
            for item in items:
                yield from self._iter_toolset_items(item)
            return
        yield items

    def _merge_registry(self, other: "ToolRegistry") -> None:
        for toolset in other._toolsets:
            if toolset not in self._toolsets:
                self._toolsets.append(toolset)
        for name in other.list_tools():
            tool = other.get(name)
            if tool is None:
                continue
            origin = other._origins.get(name, ToolOrigin(source="function"))
            self._register_tool_object(tool, origin=origin)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def _normalize_tool_key(self, value: str) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        return re.sub(r"[.\-_=]+", "", text)


class ToolExposure(ToolRegistry):
    """Read-only registry snapshot shared by one model turn and its dispatch."""

    _selection_metadata: Dict[str, Any]

    def __init__(self) -> None:
        raise TypeError("ToolExposure instances are created by ToolRegistry.freeze()")

    @classmethod
    def _capture(
        cls,
        source: ToolRegistry,
        names: List[str],
        *,
        metadata: Mapping[str, Any] | None,
    ) -> "ToolExposure":
        exposure = object.__new__(cls)
        exposure._tools = {}
        exposure._origins = {}
        exposure._toolsets = []
        exposure._setup_done = True
        exposure._revision = source.revision
        exposure._selection_metadata = json.loads(
            json.dumps(
                dict(metadata or {}),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        for name in names:
            tool = _FrozenTool(source._tools[name])
            exposure._tools[name] = tool
            exposure._origins[name] = deepcopy(
                source._origins.get(name, ToolOrigin(source="function"))
            )
        return exposure

    @property
    def source_registry_revision(self) -> int:
        return self._revision

    @property
    def selection_metadata(self) -> Dict[str, Any]:
        return deepcopy(self._selection_metadata)

    def audit_metadata(self) -> Dict[str, Any]:
        """Return a stable, serializable identity for this frozen exposure."""

        schemas = self.get_all_specs()
        serialized = json.dumps(
            schemas,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return {
            "registry_revision": self.source_registry_revision,
            "tool_names": self.list_tools(),
            "schema_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            "selection": self.selection_metadata,
        }

    def get(self, name: str) -> Optional[BaseTool]:
        tool = self._tools.get(name)
        if tool is None:
            return None
        isolated = copy(tool)
        isolated.spec = deepcopy(tool.spec)
        if hasattr(isolated, "meta"):
            isolated.meta = deepcopy(isolated.meta)
        return isolated

    def register(
        self, item: Any, name: Optional[str] = None, meta: Optional[ToolMeta] = None
    ) -> "ToolRegistry":
        _ = item, name, meta
        raise TypeError("ToolExposure is immutable")

    def register_toolset(
        self, toolset: Any, namespace: Optional[str] = None
    ) -> "ToolRegistry":
        _ = toolset, namespace
        raise TypeError("ToolExposure is immutable")

    def include_toolset(self, items: Any) -> "ToolRegistry":
        _ = items
        raise TypeError("ToolExposure is immutable")

    def include(self, obj: Any) -> "ToolRegistry":
        _ = obj
        raise TypeError("ToolExposure is immutable")

    def unregister(self, name: str) -> BaseTool:
        _ = name
        raise TypeError("ToolExposure is immutable")

    def setup(self, context: Optional[Dict[str, Any]] = None) -> None:
        _ = context

    def teardown(self, context: Optional[Dict[str, Any]] = None) -> None:
        _ = context

    async def asetup(self, context: Optional[Dict[str, Any]] = None) -> None:
        _ = context

    async def ateardown(self, context: Optional[Dict[str, Any]] = None) -> None:
        _ = context


__all__ = ["ToolExposure", "ToolOrigin", "ToolRegistry"]
