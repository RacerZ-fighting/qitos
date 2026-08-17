"""Provider-neutral typed thinking level owned by the Model boundary.

``ThinkingLevel`` is Pi's exact seven-value vocabulary
(``pi:packages/agent/src/types.ts``). The loop clamps a requested level
against the levels a configured model declares on
``ModelCapabilities.thinking_levels`` with Pi's nearest-up-then-down rule
and puts the result on ``ModelRequest.thinking_level``; adapters then
translate that typed field into provider request options through
``thinking_request_options``, the single wire encoding shared with the
construction-time reasoning policy in ``qitos.harness``.

Off mapping: ``off`` is an explicit disable signal, not an omission, so a
typed level can neutralize construction-time reasoning kwargs. OpenAI
transports emit ``effort: "none"`` (Pi's default off effort), Anthropic
variants emit a disabled ``thinking`` object.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict


class ThinkingLevel(str, Enum):
    """Reasoning intensity requested for one model transaction."""

    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


#: Master ordered list used for clamping (Pi's EXTENDED_THINKING_LEVELS).
THINKING_LEVEL_ORDER: tuple[ThinkingLevel, ...] = tuple(ThinkingLevel)


def clamp_thinking_level(
    level: ThinkingLevel | None,
    supported: tuple[ThinkingLevel, ...],
) -> ThinkingLevel | None:
    """Clamp one requested level to a model's declared supported set.

    The rule is Pi's nearest-up-then-down over ``THINKING_LEVEL_ORDER``:
    an exact hit wins; otherwise the nearest higher supported level, then
    the nearest lower one. ``None`` (no request) and an empty supported
    set (no typed thinking support) both yield ``None``.
    """

    if level is None:
        return None
    if not isinstance(level, ThinkingLevel):
        raise TypeError("level must be a ThinkingLevel or None")
    if not isinstance(supported, tuple) or not all(
        isinstance(item, ThinkingLevel) for item in supported
    ):
        raise TypeError("supported must be a tuple of ThinkingLevel")
    if not supported:
        return None
    if level in supported:
        return level
    index = THINKING_LEVEL_ORDER.index(level)
    for candidate in THINKING_LEVEL_ORDER[index:]:
        if candidate in supported:
            return candidate
    for candidate in reversed(THINKING_LEVEL_ORDER[:index]):
        if candidate in supported:
            return candidate
    return supported[0]


_ANTHROPIC_MANUAL_BUDGETS: Dict[ThinkingLevel, int] = {
    ThinkingLevel.MINIMAL: 1_024,
    ThinkingLevel.LOW: 1_024,
    ThinkingLevel.MEDIUM: 2_048,
    ThinkingLevel.HIGH: 4_096,
    ThinkingLevel.XHIGH: 8_192,
    ThinkingLevel.MAX: 16_384,
}


def thinking_request_options(
    level: ThinkingLevel,
    *,
    wire_format: str,
    api_mode: str,
    max_output_tokens: int | None = None,
) -> Dict[str, Any]:
    """Encode one typed level as provider request options.

    This is the canonical thinking wire encoding: the harness reasoning
    policy (``qitos.harness``) resolves construction-time defaults through
    the same function, and adapters apply it to a typed
    ``ModelRequest.thinking_level`` at request time. Unknown or
    ``provider_default`` wire formats emit no fields.
    """

    if not isinstance(level, ThinkingLevel):
        raise TypeError("level must be a ThinkingLevel")
    if wire_format == "openai_effort":
        effort = "none" if level is ThinkingLevel.OFF else level.value
        if api_mode.strip().lower() == "responses":
            return {"reasoning": {"effort": effort}}
        return {"reasoning_effort": effort}
    if wire_format == "glm_effort":
        if level is ThinkingLevel.OFF:
            return {"extra_body": {"thinking": {"type": "disabled"}}}
        return {
            "reasoning_effort": level.value,
            "extra_body": {"thinking": {"type": "enabled"}},
        }
    if wire_format == "enable_thinking":
        return {"extra_body": {"enable_thinking": level is not ThinkingLevel.OFF}}
    if wire_format == "thinking_object":
        thinking_type = "disabled" if level is ThinkingLevel.OFF else "enabled"
        return {"extra_body": {"thinking": {"type": thinking_type}}}
    if wire_format == "anthropic_manual_thinking":
        if level is ThinkingLevel.OFF:
            return {"thinking": {"type": "disabled"}}
        budget = _ANTHROPIC_MANUAL_BUDGETS[level]
        if max_output_tokens is not None:
            if isinstance(max_output_tokens, bool) or max_output_tokens < 2_048:
                raise ValueError(
                    "Anthropic manual thinking requires max_output_tokens >= 2048"
                )
            visible_reserve = max(1_024, max_output_tokens // 4)
            budget = min(budget, max_output_tokens - visible_reserve)
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}
    if wire_format == "kimi_anthropic_thinking":
        if level is ThinkingLevel.OFF:
            return {"thinking": {"type": "disabled"}}
        return {
            "thinking": {"type": "enabled"},
            "output_config": {"effort": level.value},
        }
    return {}


__all__ = [
    "THINKING_LEVEL_ORDER",
    "ThinkingLevel",
    "clamp_thinking_level",
    "thinking_request_options",
]
