"""Provider-neutral QitOS web-search tool."""

from __future__ import annotations

from typing import Any

from qitos.core.tool import BaseTool, RetryPolicy, ToolPermission, ToolSpec

from .capability import RetryableWebSearchError, WebSearchCapability


_WEB_SEARCH_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    backoff_factor=0.5,
    max_backoff=2.0,
    retryable_exceptions=(RetryableWebSearchError,),
)


class ManagedWebSearchTool(BaseTool):
    """Expose one configured managed search capability as ``web_search``."""

    def __init__(self, capability: WebSearchCapability) -> None:
        self._capability = capability
        super().__init__(
            ToolSpec(
                name="web_search",
                description=(
                    "Search public web sources for documentation, vulnerability "
                    "intelligence, default credentials, and research. Do not use "
                    "this tool to send requests to the target environment."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1, "maxLength": 500},
                        "max_results": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                permissions=ToolPermission(network=True),
                read_only=True,
                concurrency_safe=True,
                retry_policy=_WEB_SEARCH_RETRY_POLICY,
            )
        )

    async def execute(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> dict[str, object]:
        _ = runtime_context
        query = args.get("query")
        max_results = args.get("max_results", 8)
        if not isinstance(query, str):
            raise TypeError("web search query must be a string")
        if isinstance(max_results, bool) or not isinstance(max_results, int):
            raise TypeError("web search max_results must be an integer")
        return (
            await self._capability.search(
                query,
                max_results=max_results,
            )
        ).to_dict()

    async def aclose(self) -> None:
        await self._capability.aclose()


__all__ = ["ManagedWebSearchTool"]
