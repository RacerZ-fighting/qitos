"""Canonical model interface and provider construction."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.model_capabilities import ModelCapabilities
from ..core.model_request import ModelRequest
from ..core.model_request import ModelContinuation
from ..core.multimodal import normalize_messages
from ..core.model_response import ModelUsage, normalize_model_usage
from .context_registry import infer_context_window


@dataclass(slots=True)
class ModelStreamChunk:
    """One provider-neutral model stream event.

    A normally completed stream ends with exactly one chunk whose ``done``
    field is true. Provider adapters raise transport failures and re-raise
    ``asyncio.CancelledError``; they do not manufacture successful terminal
    chunks after an error or cancellation.
    """

    text: str = ""
    done: bool = False
    usage: ModelUsage | Mapping[str, Any] | None = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    native_items: Optional[List[Dict[str, Any]]] = None
    event_type: Optional[str] = None
    event_metadata: Dict[str, Any] = field(default_factory=dict)
    reasoning_content: Optional[str] = None
    finish_reason: Optional[str] = None
    continuation: ModelContinuation | None = None

    def __post_init__(self) -> None:
        self.usage = normalize_model_usage(self.usage)
        if self.continuation is not None and not isinstance(
            self.continuation, ModelContinuation
        ):
            raise TypeError("continuation must be a ModelContinuation or None")

    @property
    def is_final(self) -> bool:
        """Return whether this is the stream's terminal chunk."""

        return self.done


class Model(ABC):
    """Asynchronous provider interface consumed by the QitOS Engine.

    Provider implementations accept QitOS's existing normalized message
    dictionaries and expose one async stream. The absolute monotonic deadline
    applies to the complete logical request, including all retry attempts.
    """

    provider_name = "model"

    def __init__(
        self,
        model: str = "default",
        system_prompt: Optional[str] = None,
        temperature: float | None = 0.7,
        max_tokens: int = 2048,
        context_window: Optional[int] = None,
        provider_name: str | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_window = self._resolve_context_window(context_window)
        if provider_name is not None:
            if not isinstance(provider_name, str) or not provider_name.strip():
                raise ValueError("provider_name must be a non-empty string")
            self.provider_name = provider_name.strip()

    @abstractmethod
    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamChunk]:
        """Yield one complete logical model transaction.

        Implementations own retry and transport cleanup. A retrying adapter
        must not expose chunks from a failed attempt.
        """

        if False:  # pragma: no cover - keeps this an async-generator contract
            yield ModelStreamChunk()

    async def close(self) -> None:
        """Release model-owned resources.

        Current adapters own clients per request, so the default lifecycle has
        no persistent resource to close.
        """

    def validate_request(self, request: ModelRequest) -> None:
        """Reject a request captured for another configured Provider."""

        if not isinstance(request, ModelRequest):
            raise TypeError("Model.stream() requires a ModelRequest")
        if request.provider != self.provider_name:
            raise ValueError("model request provider does not match the adapter")
        if request.model != self.model:
            raise ValueError("model request model does not match the adapter")

    @property
    def capabilities(self) -> ModelCapabilities:
        """Return conservative transport facts for this adapter."""

        return ModelCapabilities(multimodal_input=self.supports_multimodal_input())

    async def __aenter__(self) -> Model:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.close()

    def supports_tool_schema_delivery(
        self, delivery: str, protocol: Any = None
    ) -> bool:
        """Return whether the adapter supports the requested tool delivery."""

        _ = protocol
        return str(delivery or "prompt_injection") == "prompt_injection"

    def build_tool_schema_request_options(
        self,
        tool_schema_payload: Optional[List[Dict[str, Any]]],
        *,
        protocol: Any = None,
        delivery: str = "prompt_injection",
    ) -> Dict[str, Any]:
        """Project QitOS tool schemas into provider request options."""

        _ = tool_schema_payload, protocol, delivery
        return {}

    def supports_multimodal_input(self) -> bool:
        """Return whether this adapter accepts multimodal message content."""

        return False

    def count_tokens(self, messages_or_text: Any) -> Optional[int]:
        """Estimate tokens for messages or plain text."""

        text = self._stringify_token_payload(messages_or_text)
        if not text:
            return 0
        pieces = re.findall(r"\w+|[^\s\w]", text, flags=re.UNICODE)
        return len(pieces)

    def count_request_tokens(
        self,
        messages: List[Dict[str, Any]],
        request_options: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """Estimate tokens for one complete provider request."""

        payload: Dict[str, Any] = {"messages": list(messages)}
        payload.update(dict(request_options or {}))
        return self.count_tokens(payload)

    def _resolve_context_window(self, explicit: Optional[int]) -> Optional[int]:
        if (
            isinstance(explicit, int)
            and not isinstance(explicit, bool)
            and explicit > 0
        ):
            return explicit
        inferred = infer_context_window(self.model)
        if isinstance(inferred, int) and inferred > 0:
            return inferred
        return 128000

    @staticmethod
    def _stringify_token_payload(payload: Any) -> str:
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload
        if isinstance(payload, (dict, list)):
            return json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        return str(payload)

    def format_messages(
        self, user_content: str, history: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Build normalized messages from the configured system prompt."""

        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if history:
            messages.extend(normalize_messages(history))
        messages.append({"role": "user", "content": user_content})
        return messages

    @staticmethod
    def format_tool_response(tool_name: str, args: Dict[str, Any], result: Any) -> str:
        """Render a generic text-protocol tool observation."""

        _ = args
        return (
            f"Observed result from {tool_name}:\n{result}\n\n"
            "Please decide on the next action, or provide a Final Answer if "
            "the task is complete."
        )

    @staticmethod
    def format_final_answer(answer: str) -> str:
        """Render a generic text-protocol final answer."""

        return f"Final Answer: {answer}"

    @staticmethod
    def format_action(tool_name: str, args: Mapping[str, Any]) -> str:
        """Render a generic text-protocol action."""

        args_str = ", ".join(
            f'{key}="{value}"' if isinstance(value, str) else f"{key}={value}"
            for key, value in args.items()
        )
        return f"Action: {tool_name}({args_str})"

    @property
    def config(self) -> Dict[str, Any]:
        """Return the stable model configuration used for diagnostics."""

        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "context_window": self.context_window,
            "system_prompt": self.system_prompt,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"


class ModelFactory:
    """Instance-scoped registry for model constructors.

    Registration is explicit and duplicate names fail. Importing a provider
    module never mutates process-global state, so applications and tests can
    own isolated factories.
    """

    def __init__(
        self,
        providers: Optional[Mapping[str, Callable[..., Model]]] = None,
    ) -> None:
        self._providers: Dict[str, Callable[..., Model]] = {}
        for name, constructor in dict(providers or {}).items():
            self.register(name, constructor)

    def register(self, name: str, constructor: Callable[..., Model]) -> None:
        """Register one constructor under a normalized provider name."""

        normalized = str(name or "").strip().lower()
        if not normalized:
            raise ValueError("model provider name must be non-empty")
        if normalized in self._providers:
            raise ValueError(f"Model provider already registered: {normalized}")
        self._providers[normalized] = constructor

    def create(self, provider: str, **kwargs: Any) -> Model:
        """Create one model from this factory's provider snapshot."""

        normalized = str(provider or "").strip().lower()
        try:
            constructor = self._providers[normalized]
        except KeyError as exc:
            raise ValueError(f"Unknown model provider: {provider}") from exc
        model = constructor(**kwargs)
        if not isinstance(model, Model):
            raise TypeError(
                f"Model provider {normalized!r} returned {type(model).__name__}, "
                "expected Model"
            )
        return model

    @property
    def provider_names(self) -> tuple[str, ...]:
        """Return registered names in deterministic order."""

        return tuple(sorted(self._providers))


__all__ = ["Model", "ModelFactory", "ModelStreamChunk"]
