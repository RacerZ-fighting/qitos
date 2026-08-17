"""Preset-backed model harness helpers."""

from __future__ import annotations

from typing import Any

from ._adapters import AnthropicAdapter, OpenAICompatibleAdapter, adapter_for_kind
from ._presets import known_family_presets, resolve_builtin_preset
from ._reasoning import (
    ReasoningEffort,
    ReasoningPolicy,
    ReasoningResolution,
    parse_reasoning_effort,
    resolve_reasoning,
)
from ._types import (
    ContextPolicy,
    FamilyPreset,
    HarnessPolicy,
    ModelAdapter,
    ToolPolicy,
)


def resolve_family_preset(
    identifier: str | None = None, *, family_id: str | None = None
) -> FamilyPreset:
    target = family_id if family_id is not None else identifier
    return resolve_builtin_preset(target)


def build_harness_policy(
    *,
    model_name: str | None = None,
    family_id: str | None = None,
    protocol: str | None = None,
    tool_delivery: str | None = None,
    adapter_kind: str | None = None,
    resolution_source: str = "family_preset",
) -> HarnessPolicy:
    preset = resolve_family_preset(model_name, family_id=family_id)
    adapter = adapter_for_kind(adapter_kind or preset.adapter_kind)
    protocol_id = str(protocol or preset.default_protocol)
    fallback_ids = (
        () if protocol is not None else tuple(preset.fallback_protocols)
    )
    return HarnessPolicy(
        family_preset=preset,
        adapter=adapter,
        protocol_id=protocol_id,
        fallback_protocol_ids=fallback_ids,
        tool_policy=preset.tool_policy,
        context_policy=preset.context_policy,
        tool_delivery=str(tool_delivery or preset.tool_policy.primary_delivery),
        resolution_source=resolution_source,
    )


def build_model_for_preset(
    *,
    model_name: str,
    family_id: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    protocol: str | None = None,
    tool_delivery: str | None = None,
    adapter_kind: str | None = None,
    temperature: float | None = 0.2,
    max_tokens: int = 2048,
    timeout: int = 120,
    system_prompt: str | None = None,
    context_window: int | None = None,
    default_request_kwargs: dict[str, Any] | None = None,
    api_mode: str = "chat_completions",
    max_attempts: int = 2,
    stream_idle_timeout: float = 60.0,
    retry_window_seconds: float = 300.0,
    reasoning_effort: ReasoningEffort | str | None = ReasoningEffort.HIGH,
) -> Any:
    harness = build_harness_policy(
        model_name=model_name,
        family_id=family_id,
        protocol=protocol,
        tool_delivery=tool_delivery,
        adapter_kind=adapter_kind,
        resolution_source=(
            "explicit_adapter" if adapter_kind is not None else "family_preset"
        ),
    )
    reasoning = resolve_reasoning(
        family_id=harness.family_preset.id,
        model_name=model_name,
        api_mode=api_mode,
        requested=reasoning_effort,
        max_output_tokens=max_tokens,
    )
    # Merge preset recommendations, caller options, then the resolved reasoning
    # contract. Explicit reasoning intent is authoritative for its wire fields.
    preset_kwargs = (
        harness.family_preset.recommended_request_kwargs
        if harness.adapter.kind == harness.family_preset.adapter_kind
        else None
    )
    effective_kwargs = _merge_request_options(
        preset_kwargs,
        default_request_kwargs,
        reasoning.request_options,
    )

    llm = harness.adapter.build_model(
        preset=harness.family_preset,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        context_policy=harness.context_policy,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        system_prompt=system_prompt,
        context_window=context_window,
        default_request_kwargs=effective_kwargs,
        api_mode=api_mode,
        max_attempts=max_attempts,
        stream_idle_timeout=stream_idle_timeout,
        retry_window_seconds=retry_window_seconds,
    )
    metadata = dict(getattr(llm, "qitos_harness_metadata", {}) or {})
    metadata.update(harness.to_dict())
    metadata.setdefault(
        "decision_lane_preference",
        (
            "native_tool_calls"
            if harness.tool_policy.native_tool_call_preferred
            else "parser"
        ),
    )
    metadata.setdefault(
        "native_tool_call_preferred", harness.tool_policy.native_tool_call_preferred
    )
    metadata.setdefault("effective_tool_delivery", harness.tool_delivery)
    metadata["reasoning"] = reasoning.to_dict()
    setattr(llm, "qitos_harness_metadata", metadata)
    setattr(llm, "qitos_family_preset", harness.family_preset.id)
    return llm


def _merge_request_options(
    *options: dict[str, Any] | None,
) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    for option in options:
        if not option:
            continue
        for key, value in option.items():
            if key in {
                "extra_body",
                "reasoning",
                "thinking",
                "output_config",
            } and isinstance(value, dict):
                current = merged.get(key)
                if (
                    key == "thinking"
                    and isinstance(current, dict)
                    and value.get("type") is not None
                    and value.get("type") != current.get("type")
                ):
                    merged[key] = dict(value)
                    continue
                nested = dict(merged.get(key) or {})
                nested.update(value)
                merged[key] = nested
            else:
                merged[key] = value
    return merged or None


__all__ = [
    "ModelAdapter",
    "AnthropicAdapter",
    "OpenAICompatibleAdapter",
    "ToolPolicy",
    "ContextPolicy",
    "HarnessPolicy",
    "FamilyPreset",
    "ReasoningEffort",
    "ReasoningPolicy",
    "ReasoningResolution",
    "parse_reasoning_effort",
    "resolve_reasoning",
    "resolve_family_preset",
    "build_model_for_preset",
    "build_harness_policy",
    "known_family_presets",
]
