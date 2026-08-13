"""Kimi managed web-search capability."""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from .capability import (
    WebSearchError,
    WebSearchResponse,
    WebSource,
)

DEFAULT_KIMI_SEARCH_URL = "https://api.kimi.com/coding/v1/search"
DEFAULT_KIMI_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_KIMI_MODEL = "kimi-k3"
_MAX_QUERY_CHARS = 500
_MAX_RESULTS = 10
_MAX_TITLE_CHARS = 500
_MAX_URL_CHARS = 2_048
_MAX_SNIPPET_CHARS = 2_000
_MAX_TOOL_ROUNDS = 4
_URL_RE = re.compile(r"https?://[^\s\])}>\"']+")


class KimiWebSearchCapability:
    """Call the managed ``/search`` endpoint used by Kimi Code."""

    def __init__(
        self,
        *,
        api_key: str,
        search_url: str = DEFAULT_KIMI_SEARCH_URL,
        timeout_seconds: float = 30.0,
        client: Any = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Kimi web search requires a non-empty API key")
        if not search_url.strip():
            raise ValueError("Kimi web search requires a non-empty URL")
        if timeout_seconds <= 0:
            raise ValueError("Kimi web search timeout must be positive")
        self._api_key = api_key
        self._search_url = search_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient()

    async def search(
        self, query: str, *, max_results: int = 8
    ) -> WebSearchResponse:
        normalized_query = _validate_query(query)
        normalized_limit = _validate_limit(max_results)
        try:
            response = await self._client.post(
                self._search_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={"text_query": normalized_query},
                timeout=self._timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise WebSearchError("timeout", "Kimi web search timed out") from exc
        except httpx.RequestError as exc:
            raise WebSearchError("network", "Kimi web search request failed") from exc

        if response.status_code in {401, 403}:
            raise WebSearchError(
                "authentication", "Kimi web search authentication failed"
            )
        if response.status_code == 429:
            raise WebSearchError("rate_limited", "Kimi web search was rate limited")
        if response.status_code != 200:
            raise WebSearchError(
                "provider",
                f"Kimi web search returned HTTP {response.status_code}",
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise WebSearchError(
                "protocol", "Kimi web search returned invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise WebSearchError(
                "protocol", "Kimi web search response must be an object"
            )
        raw_results = payload.get("search_results", [])
        if not isinstance(raw_results, list):
            raise WebSearchError("protocol", "Kimi web search results must be a list")

        sources = tuple(
            source
            for item in raw_results[:normalized_limit]
            if (source := _parse_source(item)) is not None
        )
        return WebSearchResponse(
            text=(
                f"Kimi returned {len(sources)} public web search result(s) "
                f"for: {normalized_query}"
            ),
            sources=sources,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class KimiBuiltinWebSearchCapability:
    """Run Kimi Open Platform's ``$web_search`` sidecar protocol."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_KIMI_BASE_URL,
        model: str = DEFAULT_KIMI_MODEL,
        timeout_seconds: float = 30.0,
        client: Any = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Kimi web search requires a non-empty API key")
        if not base_url.strip() or not model.strip():
            raise ValueError("Kimi web search requires a base URL and model")
        if timeout_seconds <= 0:
            raise ValueError("Kimi web search timeout must be positive")
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds
        self._client = client
        self._owns_client = client is None

    def _client_for_request(self) -> Any:
        if self._client is None:
            try:
                import openai
            except ImportError as exc:
                raise RuntimeError(
                    "Kimi builtin web search requires the qitos models extra"
                ) from exc
            self._client = openai.AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout_seconds,
                max_retries=0,
            )
        return self._client

    async def search(
        self, query: str, *, max_results: int = 8
    ) -> WebSearchResponse:
        normalized_query = _validate_query(query)
        normalized_limit = _validate_limit(max_results)
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    "Search current public web sources for the following query. "
                    "Return a concise factual answer and include source URLs.\n\n"
                    f"{normalized_query}"
                ),
            }
        ]
        sources: list[WebSource] = []
        for _round in range(_MAX_TOOL_ROUNDS):
            choice = await self._completion(messages)
            message = choice.message
            tool_calls = list(getattr(message, "tool_calls", None) or [])
            if tool_calls:
                messages.append(_message_payload(message))
                for tool_call in tool_calls:
                    function = getattr(tool_call, "function", None)
                    if getattr(function, "name", "") != "$web_search":
                        raise WebSearchError(
                            "protocol", "Kimi returned an unexpected builtin tool"
                        )
                    arguments = str(getattr(function, "arguments", "") or "")
                    try:
                        parsed = json.loads(arguments)
                    except ValueError as exc:
                        raise WebSearchError(
                            "protocol", "Kimi web search returned invalid arguments"
                        ) from exc
                    sources.extend(_sources_from_value(parsed))
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(getattr(tool_call, "id", "")),
                            "name": "$web_search",
                            "content": arguments,
                        }
                    )
                continue

            text = str(getattr(message, "content", "") or "").strip()
            if not text:
                bounded_sources = _deduplicate_sources(sources, normalized_limit)
                if bounded_sources:
                    return WebSearchResponse(
                        text=(
                            "Kimi returned public web search sources without a "
                            "synthesized answer."
                        ),
                        sources=bounded_sources,
                    )
                raise WebSearchError(
                    "protocol", "Kimi web search returned no final answer"
                )
            sources.extend(_sources_from_text(text))
            return WebSearchResponse(
                text=text,
                sources=_deduplicate_sources(sources, normalized_limit),
            )
        raise WebSearchError(
            "protocol", "Kimi web search exceeded the builtin tool round limit"
        )

    async def _completion(self, messages: list[dict[str, Any]]) -> Any:
        try:
            completion = await self._client_for_request().chat.completions.create(
                model=self._model,
                messages=messages,
                tools=[
                    {
                        "type": "builtin_function",
                        "function": {"name": "$web_search"},
                    }
                ],
                max_tokens=8_192,
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception as exc:
            raise _classify_client_error(exc) from exc
        choices = list(getattr(completion, "choices", None) or [])
        if not choices:
            raise WebSearchError(
                "protocol", "Kimi web search returned no completion choice"
            )
        return choices[0]

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.close()


def _validate_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("web search query must be a non-empty string")
    normalized = query.strip()
    if len(normalized) > _MAX_QUERY_CHARS:
        raise ValueError(f"web search query exceeds {_MAX_QUERY_CHARS} characters")
    return normalized


def _validate_limit(max_results: int) -> int:
    if (
        isinstance(max_results, bool)
        or not isinstance(max_results, int)
        or not 1 <= max_results <= _MAX_RESULTS
    ):
        raise ValueError(f"max_results must be between 1 and {_MAX_RESULTS}")
    return max_results


def _parse_source(value: Any) -> WebSource | None:
    if not isinstance(value, dict):
        return None
    title = _bounded_text(value.get("title"), _MAX_TITLE_CHARS)
    url = _bounded_text(value.get("url"), _MAX_URL_CHARS)
    snippet = _bounded_text(value.get("snippet"), _MAX_SNIPPET_CHARS)
    if not title and not url and not snippet:
        return None
    return WebSource(
        title=title,
        url=url,
        snippet=snippet,
        date=_bounded_text(value.get("date"), 100),
        site_name=_bounded_text(value.get("site_name"), 200),
    )


def _bounded_text(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:limit]


def _message_payload(message: Any) -> dict[str, Any]:
    tool_calls = []
    for tool_call in list(getattr(message, "tool_calls", None) or []):
        function = getattr(tool_call, "function", None)
        tool_calls.append(
            {
                "id": str(getattr(tool_call, "id", "")),
                # Kimi returns ``builtin_function`` but accepts the historical
                # assistant call only in the standard function-call shape.
                "type": "function",
                "function": {
                    "name": str(getattr(function, "name", "")),
                    "arguments": str(getattr(function, "arguments", "") or ""),
                },
            }
        )
    payload = {
        "role": "assistant",
        "content": getattr(message, "content", None),
        "tool_calls": tool_calls,
    }
    reasoning_content = getattr(message, "reasoning_content", None)
    if isinstance(reasoning_content, str) and reasoning_content:
        payload["reasoning_content"] = reasoning_content
    return payload


def _sources_from_value(value: Any) -> list[WebSource]:
    sources: list[WebSource] = []
    if isinstance(value, dict):
        url = next(
            (
                value.get(key)
                for key in ("url", "link", "href")
                if isinstance(value.get(key), str) and value.get(key)
            ),
            "",
        )
        if url:
            sources.append(
                WebSource(
                    title=_bounded_text(
                        value.get("title") or value.get("name"),
                        _MAX_TITLE_CHARS,
                    ),
                    url=_bounded_text(url, _MAX_URL_CHARS),
                    snippet=_bounded_text(
                        value.get("snippet")
                        or value.get("description")
                        or value.get("content"),
                        _MAX_SNIPPET_CHARS,
                    ),
                    date=_bounded_text(value.get("date"), 100),
                    site_name=_bounded_text(value.get("site_name"), 200),
                )
            )
        for nested in value.values():
            if isinstance(nested, (dict, list)):
                sources.extend(_sources_from_value(nested))
    elif isinstance(value, list):
        for nested in value:
            sources.extend(_sources_from_value(nested))
    return sources


def _sources_from_text(text: str) -> list[WebSource]:
    return [WebSource(title="", url=url) for url in _URL_RE.findall(text)]


def _deduplicate_sources(
    sources: list[WebSource],
    limit: int,
) -> tuple[WebSource, ...]:
    unique: list[WebSource] = []
    seen: set[str] = set()
    for source in sources:
        key = source.url or f"{source.title}\0{source.snippet}"
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(source)
        if len(unique) >= limit:
            break
    return tuple(unique)


def _classify_client_error(exc: Exception) -> WebSearchError:
    name = type(exc).__name__.lower()
    status = getattr(exc, "status_code", None)
    if status in {401, 403} or "authentication" in name:
        return WebSearchError("authentication", "Kimi web search authentication failed")
    if status == 429 or "ratelimit" in name:
        return WebSearchError("rate_limited", "Kimi web search was rate limited")
    if "timeout" in name:
        return WebSearchError("timeout", "Kimi web search timed out")
    if "connection" in name:
        return WebSearchError("network", "Kimi web search request failed")
    return WebSearchError("provider", "Kimi web search model request failed")


__all__ = [
    "DEFAULT_KIMI_BASE_URL",
    "DEFAULT_KIMI_MODEL",
    "DEFAULT_KIMI_SEARCH_URL",
    "KimiBuiltinWebSearchCapability",
    "KimiWebSearchCapability",
]
