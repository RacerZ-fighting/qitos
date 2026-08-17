"""Provider-neutral reasoning effort resolution for model harnesses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..core.thinking import ThinkingLevel, thinking_request_options


class ReasoningEffort(str, Enum):
    """Reasoning effort levels accepted by the QitOS harness."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


_EFFORT_ORDER = tuple(ReasoningEffort)
_EFFORT_RANK = {effort: rank for rank, effort in enumerate(_EFFORT_ORDER)}


@dataclass(frozen=True)
class ReasoningPolicy:
    """Reasoning capabilities and request encoding for one model family."""

    supported_efforts: tuple[ReasoningEffort, ...] = (ReasoningEffort.HIGH,)
    wire_format: str = "provider_default"

    def resolve(self, requested: ReasoningEffort) -> ReasoningEffort:
        """Map an unsupported effort to the closest supported level.

        A tie resolves upward so a request for ``medium`` maps to ``high`` on
        Kimi K3, matching Kimi Code's public behavior.
        """
        if requested in self.supported_efforts:
            return requested
        requested_rank = _EFFORT_RANK[requested]
        return min(
            self.supported_efforts,
            key=lambda effort: (
                abs(_EFFORT_RANK[effort] - requested_rank),
                -_EFFORT_RANK[effort],
            ),
        )


@dataclass(frozen=True)
class ReasoningResolution:
    """Resolved reasoning intent and its provider request options."""

    requested: ReasoningEffort
    resolved: ReasoningEffort
    supported_efforts: tuple[ReasoningEffort, ...]
    request_options: dict[str, Any]
    effective_budget_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return trace-safe reasoning metadata."""
        return {
            "requested": self.requested.value,
            "resolved": self.resolved.value,
            "supported_efforts": [item.value for item in self.supported_efforts],
            "request_configured": bool(self.request_options),
            "effective_budget_tokens": self.effective_budget_tokens,
        }


_DEFAULT_POLICY = ReasoningPolicy()
_KIMI_K3_POLICY = ReasoningPolicy(
    supported_efforts=(
        ReasoningEffort.LOW,
        ReasoningEffort.HIGH,
        ReasoningEffort.MAX,
    ),
    wire_format="openai_effort",
)
_KIMI_K3_ANTHROPIC_POLICY = ReasoningPolicy(
    supported_efforts=_KIMI_K3_POLICY.supported_efforts,
    wire_format="kimi_anthropic_thinking",
)
_OPENAI_REASONING_POLICY = ReasoningPolicy(
    supported_efforts=(
        ReasoningEffort.LOW,
        ReasoningEffort.MEDIUM,
        ReasoningEffort.HIGH,
        ReasoningEffort.XHIGH,
    ),
    wire_format="openai_effort",
)
_OPENAI_GPT_56_REASONING_POLICY = ReasoningPolicy(
    supported_efforts=(
        *_OPENAI_REASONING_POLICY.supported_efforts,
        ReasoningEffort.MAX,
    ),
    wire_format="openai_effort",
)
_GLM_52_POLICY = ReasoningPolicy(
    supported_efforts=(ReasoningEffort.HIGH, ReasoningEffort.MAX),
    wire_format="glm_effort",
)
_ANTHROPIC_45_POLICY = ReasoningPolicy(
    supported_efforts=_EFFORT_ORDER,
    wire_format="anthropic_manual_thinking",
)
_ENABLE_THINKING_POLICY = ReasoningPolicy(wire_format="enable_thinking")
_THINKING_OBJECT_POLICY = ReasoningPolicy(wire_format="thinking_object")


def parse_reasoning_effort(
    value: ReasoningEffort | str | None,
    *,
    default: ReasoningEffort = ReasoningEffort.HIGH,
) -> ReasoningEffort:
    """Parse one public reasoning effort value."""
    if value is None:
        return default
    if isinstance(value, ReasoningEffort):
        return value
    normalized = str(value).strip().lower()
    try:
        return ReasoningEffort(normalized)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in ReasoningEffort)
        raise ValueError(
            f"unsupported reasoning effort {value!r}; expected one of: {allowed}"
        ) from exc


def resolve_reasoning(
    *,
    family_id: str,
    model_name: str,
    api_mode: str,
    requested: ReasoningEffort | str | None,
    max_output_tokens: int | None = None,
) -> ReasoningResolution:
    """Resolve reasoning intent without sending unsupported provider fields."""
    effort = parse_reasoning_effort(requested)
    policy = _policy_for_model(family_id, model_name, api_mode=api_mode)
    resolved = policy.resolve(effort)
    request_options = _request_options(
        policy.wire_format,
        resolved,
        api_mode=api_mode,
        max_output_tokens=max_output_tokens,
    )
    thinking = request_options.get("thinking")
    effective_budget_tokens = (
        thinking.get("budget_tokens") if isinstance(thinking, dict) else None
    )
    return ReasoningResolution(
        requested=effort,
        resolved=resolved,
        supported_efforts=policy.supported_efforts,
        request_options=request_options,
        effective_budget_tokens=(
            effective_budget_tokens
            if isinstance(effective_budget_tokens, int)
            and not isinstance(effective_budget_tokens, bool)
            else None
        ),
    )


def _policy_for_model(
    family_id: str,
    model_name: str,
    *,
    api_mode: str,
) -> ReasoningPolicy:
    family = family_id.strip().lower()
    model = model_name.strip().lower()
    if family == "anthropic" and "-4-5" in model:
        return _ANTHROPIC_45_POLICY
    if family == "kimi" and "k3" in model:
        if api_mode.strip().lower() in {"messages", "anthropic_messages"}:
            return _KIMI_K3_ANTHROPIC_POLICY
        return _KIMI_K3_POLICY
    if family == "kimi" and "k2" in model:
        return _THINKING_OBJECT_POLICY
    if family == "qwen" and ("qwen3" in model or "qwq" in model):
        return _ENABLE_THINKING_POLICY
    if family == "glm" and model.startswith("glm-5.2"):
        return _GLM_52_POLICY
    if family == "glm" and model.startswith("glm-"):
        return _THINKING_OBJECT_POLICY
    if family == "openai" and model.startswith("gpt-5.6"):
        return _OPENAI_GPT_56_REASONING_POLICY
    if family == "openai" and model.startswith(("gpt-5", "o3", "o4", "codex")):
        return _OPENAI_REASONING_POLICY
    return _DEFAULT_POLICY


def _request_options(
    wire_format: str,
    effort: ReasoningEffort,
    *,
    api_mode: str,
    max_output_tokens: int | None,
) -> dict[str, Any]:
    # The canonical thinking wire encoding lives in qitos.core.thinking so
    # the harness policy and the typed ModelRequest.thinking_level adapter
    # path share one mapping; ReasoningEffort values are a subset of
    # ThinkingLevel values.
    return thinking_request_options(
        ThinkingLevel(effort.value),
        wire_format=wire_format,
        api_mode=api_mode,
        max_output_tokens=max_output_tokens,
    )


__all__ = [
    "ReasoningEffort",
    "ReasoningPolicy",
    "ReasoningResolution",
    "parse_reasoning_effort",
    "resolve_reasoning",
]
