"""Search backends for web and information retrieval."""

from .base import SearchBackend, SearchResult
from .capability import (
    WebSearchCapability,
    WebSearchError,
    WebSearchResponse,
    WebSource,
)
from .duckduckgo import DuckDuckGoSearchBackend
from .factory import build_web_search_capability
from .google_cse import GoogleCSESearchBackend
from .kimi import (
    DEFAULT_KIMI_SEARCH_URL,
    KimiBuiltinWebSearchCapability,
    KimiWebSearchCapability,
)
from .perplexity import PerplexitySearchBackend
from .searxng import SearXNGSearchBackend
from .sploitus import SploitusSearchBackend
from .tavily import TavilySearchBackend
from .tool import ManagedWebSearchTool
from .traversaal import TraversaalSearchBackend

__all__ = [
    "SearchBackend",
    "SearchResult",
    "WebSearchCapability",
    "WebSearchError",
    "WebSearchResponse",
    "WebSource",
    "build_web_search_capability",
    "DEFAULT_KIMI_SEARCH_URL",
    "KimiWebSearchCapability",
    "KimiBuiltinWebSearchCapability",
    "ManagedWebSearchTool",
    "DuckDuckGoSearchBackend",
    "SearXNGSearchBackend",
    "TavilySearchBackend",
    "GoogleCSESearchBackend",
    "TraversaalSearchBackend",
    "PerplexitySearchBackend",
    "SploitusSearchBackend",
]
