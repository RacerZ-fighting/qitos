"""Qwen managed Web search through its OpenAI-compatible Chat endpoint."""

from __future__ import annotations

import re
from collections.abc import Mapping

import httpx

from .capability import WebSearchError, WebSearchResponse, WebSource

DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen-plus"
_MAX_QUERY_CHARS = 500
_MAX_RESULTS = 10
_MAX_TITLE_CHARS = 500
_MAX_URL_CHARS = 2_048
_MAX_SNIPPET_CHARS = 2_000
_URL_RE = re.compile(r"https?://[^\s\])}>\"']+")


class QwenWebSearchCapability:
    """Use Qwen's native ``enable_search`` request option as a managed Tool."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_QWEN_BASE_URL,
        model: str = DEFAULT_QWEN_MODEL,
        timeout_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Qwen web search requires a non-empty API key")
        if not base_url.strip() or not model.strip():
            raise ValueError("Qwen web search requires a base URL and model")
        if timeout_seconds <= 0:
            raise ValueError("Qwen web search timeout must be positive")
        self._api_key = api_key
        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient()

    async def search(
        self,
        query: str,
        *,
        max_results: int = 8,
    ) -> WebSearchResponse:
        normalized_query = _validate_query(query)
        normalized_limit = _validate_limit(max_results)
        try:
            response = await self._client.post(
                self._endpoint,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Search current public web sources and answer the "
                                "following query concisely. Include source URLs when "
                                f"the provider makes them available.\n\n{normalized_query}"
                            ),
                        }
                    ],
                    "enable_search": True,
                    "search_options": {"forced_search": True},
                },
                timeout=self._timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise WebSearchError("timeout", "Qwen web search timed out") from exc
        except httpx.RequestError as exc:
            raise WebSearchError("network", "Qwen web search request failed") from exc

        if response.status_code in {401, 403}:
            raise WebSearchError(
                "authentication",
                "Qwen web search authentication failed",
            )
        if response.status_code == 429:
            raise WebSearchError("rate_limited", "Qwen web search was rate limited")
        if response.status_code != 200:
            raise WebSearchError(
                "provider",
                f"Qwen web search returned HTTP {response.status_code}",
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise WebSearchError(
                "protocol",
                "Qwen web search returned invalid JSON",
            ) from exc
        text = _completion_text(payload)
        sources = _response_sources(payload, text, normalized_limit)
        return WebSearchResponse(text=text, sources=sources)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _completion_text(payload: object) -> str:
    if not isinstance(payload, Mapping):
        raise WebSearchError("protocol", "Qwen web search response must be an object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise WebSearchError("protocol", "Qwen web search returned no completion")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise WebSearchError("protocol", "Qwen web search choice is invalid")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise WebSearchError("protocol", "Qwen web search message is invalid")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise WebSearchError("protocol", "Qwen web search returned no answer")
    return content.strip()


def _response_sources(
    payload: object,
    text: str,
    limit: int,
) -> tuple[WebSource, ...]:
    structured: list[WebSource] = []
    if isinstance(payload, Mapping):
        choices = payload.get("choices")
        first = choices[0] if isinstance(choices, list) and choices else None
        message = first.get("message") if isinstance(first, Mapping) else None
        for owner in (payload, first, message):
            if not isinstance(owner, Mapping):
                continue
            search_info = owner.get("search_info")
            if not isinstance(search_info, Mapping):
                continue
            results = search_info.get("search_results")
            if not isinstance(results, list):
                continue
            structured.extend(
                source
                for item in results
                if (source := _parse_source(item)) is not None
            )

    discovered = [
        *structured,
        *(WebSource(title="", url=url) for url in _URL_RE.findall(text)),
    ]
    unique: list[WebSource] = []
    seen_urls: set[str] = set()
    for source in discovered:
        if source.url in seen_urls:
            continue
        seen_urls.add(source.url)
        unique.append(source)
        if len(unique) >= limit:
            break
    return tuple(unique)


def _parse_source(value: object) -> WebSource | None:
    if not isinstance(value, Mapping):
        return None
    url = _bounded_text(value.get("url"), _MAX_URL_CHARS)
    if not url.startswith(("http://", "https://")):
        return None
    return WebSource(
        title=_bounded_text(value.get("title"), _MAX_TITLE_CHARS),
        url=url,
        snippet=_bounded_text(
            value.get("snippet") or value.get("content"),
            _MAX_SNIPPET_CHARS,
        ),
        date=_bounded_text(
            value.get("date") or value.get("publish_time"),
            100,
        ),
        site_name=_bounded_text(value.get("site_name"), 200),
    )


def _bounded_text(value: object, limit: int) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


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


__all__ = [
    "DEFAULT_QWEN_BASE_URL",
    "DEFAULT_QWEN_MODEL",
    "QwenWebSearchCapability",
]
