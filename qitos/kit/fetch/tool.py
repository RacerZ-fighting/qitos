"""Provider-neutral QitOS public web-fetch tool."""

from __future__ import annotations

from typing import Any

from qitos.core.tool import BaseTool, ToolPermission, ToolSpec

from .capability import WebFetchCapability


class ManagedWebFetchTool(BaseTool):
    """Expose one configured managed fetch capability as ``web_fetch``."""

    def __init__(self, capability: WebFetchCapability) -> None:
        self._capability = capability
        super().__init__(
            ToolSpec(
                name="web_fetch",
                description=(
                    "Fetch and extract bounded text from one public HTTP(S) URL. "
                    "Use shell or browser tools for the target environment."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 2048,
                        }
                    },
                    "required": ["url"],
                    "additionalProperties": False,
                },
                permissions=ToolPermission(network=True),
                read_only=True,
                concurrency_safe=True,
                result_max_chars=110_000,
            )
        )

    async def execute(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> dict[str, object]:
        _ = runtime_context
        url = args.get("url")
        if not isinstance(url, str):
            raise TypeError("web fetch URL must be a string")
        return (await self._capability.fetch(url)).to_dict()

    async def aclose(self) -> None:
        close = getattr(self._capability, "aclose", None)
        if callable(close):
            await close()


__all__ = ["ManagedWebFetchTool"]
