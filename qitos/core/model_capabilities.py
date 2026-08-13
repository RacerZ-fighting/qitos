"""Immutable, provider-neutral model transport capability facts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ModelAPI(str, Enum):
    """Provider transport semantics exposed by one model adapter."""

    LEGACY = "legacy"
    RESPONSES = "responses"
    ANTHROPIC_MESSAGES = "anthropic_messages"
    CHAT_COMPLETIONS = "chat_completions"


class ReasoningCapability(str, Enum):
    """Reasoning forms that an adapter preserves across its wire boundary."""

    SUMMARY = "summary"
    THINKING = "thinking"
    OPAQUE_REPLAY = "opaque_replay"


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Immutable facts about one configured model transport.

    Capabilities describe adapter behavior, not model-family marketing claims.
    Features stay false until the adapter has a tested request/result contract.
    """

    api: ModelAPI = ModelAPI.LEGACY
    native_tool_calls: bool = False
    reasoning: tuple[ReasoningCapability, ...] = ()
    opaque_replay: bool = False
    continuation: bool = False
    usage: bool = False
    prompt_cache_usage: bool = False
    multimodal_input: bool = False
    hosted_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.api, ModelAPI):
            raise TypeError("api must be a ModelAPI")
        for name in (
            "native_tool_calls",
            "opaque_replay",
            "continuation",
            "usage",
            "prompt_cache_usage",
            "multimodal_input",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if not isinstance(self.reasoning, tuple) or not all(
            isinstance(item, ReasoningCapability) for item in self.reasoning
        ):
            raise TypeError("reasoning must contain ReasoningCapability values")
        if len(self.reasoning) != len(set(self.reasoning)):
            raise ValueError("reasoning capabilities must be unique")
        if not isinstance(self.hosted_tools, tuple) or not all(
            isinstance(item, str) and item.strip() for item in self.hosted_tools
        ):
            raise TypeError("hosted_tools must contain non-empty strings")
        if len(self.hosted_tools) != len(set(self.hosted_tools)):
            raise ValueError("hosted_tools must be unique")


__all__ = ["ModelAPI", "ModelCapabilities", "ReasoningCapability"]
