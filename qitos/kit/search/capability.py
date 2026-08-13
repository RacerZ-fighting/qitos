"""Provider-neutral web-search capability contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class WebSearchError(RuntimeError):
    """A configured web-search provider could not complete a request."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class WebSource:
    """One bounded public source returned by web search."""

    title: str
    url: str
    snippet: str = ""
    date: str = ""
    site_name: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            key: value
            for key, value in {
                "title": self.title,
                "url": self.url,
                "snippet": self.snippet,
                "date": self.date,
                "site_name": self.site_name,
            }.items()
            if value
        }


@dataclass(frozen=True)
class WebSearchResponse:
    """Stable model-facing web-search result."""

    text: str
    sources: tuple[WebSource, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "sources": [source.to_dict() for source in self.sources],
        }


class WebSearchCapability(Protocol):
    """Search public information through one configured provider."""

    async def search(
        self, query: str, *, max_results: int = 8
    ) -> WebSearchResponse:
        ...


__all__ = [
    "WebSearchCapability",
    "WebSearchError",
    "WebSearchResponse",
    "WebSource",
]
