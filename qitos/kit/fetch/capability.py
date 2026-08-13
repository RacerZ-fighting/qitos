"""Provider-neutral public web-fetch capability contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class WebFetchError(RuntimeError):
    """A configured web-fetch provider could not complete a request."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class WebFetchResponse:
    """Bounded text extracted from one public URL."""

    url: str
    content: str
    content_type: str = "text/markdown"
    truncated: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "url": self.url,
            "content": self.content,
            "content_type": self.content_type,
            "truncated": self.truncated,
        }


class WebFetchCapability(Protocol):
    """Fetch one public URL through a configured managed provider."""

    async def fetch(self, url: str) -> WebFetchResponse:
        ...


__all__ = ["WebFetchCapability", "WebFetchError", "WebFetchResponse"]
