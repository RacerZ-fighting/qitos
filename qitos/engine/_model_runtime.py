"""Private model/runtime helpers for Engine."""

from __future__ import annotations

import asyncio
import html
import hashlib
import inspect
import json
import logging
import os
import re
import time
from collections import Counter
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, TypeVar, cast

from ..core._json_repair import escape_json_string_control_chars
from ..core.action import Action
from ..core.decision import Decision
from ..core.errors import (
    ErrorCategory,
    ModelExecutionError,
    ModelRequestDeadlineExceeded,
    ModelTransportError,
    ParseExecutionError,
    RuntimeErrorInfo,
)
from ..core.history import (
    HistoryMessage,
    group_history_rounds,
    message_tool_call_ids,
    message_tool_result_ids,
    select_recent_history,
)
from ..core.model_response import ModelResponse, ModelTiming
from ..core.multimodal import (
    content_to_text,
    image_base64_block,
    image_file_block,
    image_url_block,
    normalize_content_block,
    normalize_observation_pack,
    observation_modalities,
    observation_visual_assets,
    text_block,
)
from ..core.observation import Observation
from ..harness._types import native_tool_calls_preferred
from ..models.base import Model, ModelStreamChunk
from ..protocols import get_protocol, resolve_protocol_chain
from ..core.state import StateSchema
from ._context_runtime import (
    ContextCompactionRequired,
    ContextOverflowError,
    DecisionContextConfigurationError,
)
from .streaming import to_stream_handler
from .parser import (
    build_parser_diagnostics,
    normalize_parser_diagnostics,
    parser_contract,
    parser_name,
)
from .states import RuntimePhase, StepRecord

if TYPE_CHECKING:
    from .engine import Engine


_logger = logging.getLogger("qitos.engine._model_runtime")


StateT = TypeVar("StateT", bound=StateSchema)
ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


def _chunk_has_model_content(chunk: ModelStreamChunk) -> bool:
    """Return whether a stream chunk contains user-visible or actionable content."""

    if chunk.text or chunk.reasoning_content or chunk.tool_calls or chunk.native_items:
        return True
    event_type = str(chunk.event_type or "")
    if event_type in {"tool_call.start", "tool_call.delta", "tool_call.done"}:
        return True
    if event_type.startswith("response.function_call_arguments."):
        return True
    return (
        event_type == "response.output_item.added"
        and chunk.event_metadata.get("item_type") == "function_call"
    )


def _measure_prompt(
    prompt_meter: Any,
    *,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    request_options: Dict[str, Any],
    llm: Any,
) -> Any:
    """Invoke old and new prompt-meter contracts without masking meter errors."""

    measure = prompt_meter.measure
    kwargs: Dict[str, Any] = {
        "messages": messages,
        "tools": tools,
        "llm": llm,
    }
    try:
        parameters = inspect.signature(measure).parameters.values()
    except (TypeError, ValueError):
        # Opaque callables used to receive the current contract. Any resulting
        # error is reported by the caller as meter unavailability.
        kwargs["request_options"] = request_options
    else:
        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            or (
                parameter.name == "request_options"
                and parameter.kind != inspect.Parameter.POSITIONAL_ONLY
            )
            for parameter in parameters
        ):
            kwargs["request_options"] = request_options
    return measure(**kwargs)


def _escape_runtime_context_content(content: str) -> str:
    """Escape literal closing tags that would break the XML wrapper."""
    return content.replace("</RUNTIME_CONTEXT>", "&lt;/RUNTIME_CONTEXT&gt;")


_DECISION_CONTEXT_PATTERN = re.compile(
    r"<DECISION_CONTEXT\b[^>]*>.*?</DECISION_CONTEXT>", re.DOTALL
)


def _is_decision_context_packet(content: str) -> bool:
    """Return whether content carries one authoritative Decision Context.

    Decision Context already has its own semantic boundary and contract.  A
    second RUNTIME_CONTEXT wrapper adds no authority and makes the provider
    packet harder to read.  Other agents' generic runtime state keeps the
    existing wrapper.
    """
    return len(_DECISION_CONTEXT_PATTERN.findall(str(content or ""))) == 1


def _strip_decision_context_content(content: Any) -> Any:
    """Remove transient Decision Context blocks without changing other content."""
    if isinstance(content, str):
        return _DECISION_CONTEXT_PATTERN.sub("", content).rstrip()
    if isinstance(content, list):
        cleaned: list[Any] = []
        for block in content:
            if isinstance(block, dict) and str(block.get("type") or "") == "text":
                updated = dict(block)
                updated["text"] = _DECISION_CONTEXT_PATTERN.sub(
                    "", str(updated.get("text") or "")
                ).rstrip()
                cleaned.append(updated)
            else:
                cleaned.append(block)
        return cleaned
    return content


def _wrap_runtime_context(content: str) -> str:
    """Wrap runtime-state user message in semantic XML tags."""
    if _is_decision_context_packet(content):
        return str(content or "")
    safe_content = _escape_runtime_context_content(content)
    return (
        "<RUNTIME_CONTEXT\n"
        '  source="agent_runtime_controller"\n'
        '  kind="authoritative_state"\n'
        '  task_continuation="true">\n'
        f"{safe_content}\n"
        "</RUNTIME_CONTEXT>"
    )


_RUNTIME_CONTEXT_IN_TOOL_NOTICE = (
    "NOTE: The RUNTIME_CONTEXT block below was appended by the Agent Runtime "
    "Controller and is NOT part of the tool result above. It is an "
    "authoritative machine-generated working-state update; treat it as "
    "current working state, not as tool output."
)


class _ModelRuntime(Generic[StateT, ObservationT, ActionT]):
    def __init__(self, engine: Engine[StateT, ObservationT, ActionT]):
        self.engine = engine
        self.stream_callback: Optional[Any] = (
            None  # Callable[[str], None] or StreamHandler
        )
        # Incremental sidecar tracking (Fix 1A)
        self._last_message_count: int = 0
        self._last_full_step: int = -1

    async def run_decide(
        self, state: StateT, observation: ObservationT, record: StepRecord
    ) -> Decision[ActionT]:
        engine = self.engine
        engine._dispatch_hook(
            "on_before_decide",
            engine._hook_context(
                step_id=record.step_id,
                phase=RuntimePhase.DECIDE,
                state=state,
                observation=observation,
                record=record,
            ),
        )
        engine._emit(
            record.step_id,
            RuntimePhase.DECIDE,
            payload={
                "stage": "state_ready",
                "observation_type": type(observation).__name__,
                "observation_summary": self._observation_summary(observation),
            },
        )
        engine._memory_append("state", state.to_dict(), record.step_id)
        engine._emit(record.step_id, RuntimePhase.DECIDE, payload={"stage": "start"})
        raw_decision: Decision[ActionT] | ModelResponse | None = engine.agent.decide(
            state, observation
        )
        model_response: ModelResponse | None = None
        if raw_decision is None:
            model_response = await self._run_llm_decide(
                state=state, observation=observation, record=record
            )
            native_tool_calls_are_authoritative = (
                isinstance(model_response.tool_calls, list)
                and bool(model_response.tool_calls)
                and self._native_tool_call_preferred()
            )
            if native_tool_calls_are_authoritative:
                raw_decision = model_response
            else:
                interpreted = self._interpret_model_response(
                    state=state,
                    observation=observation,
                    response=model_response,
                    record=record,
                )
                if interpreted is None:
                    self._raise_for_empty_model_response(
                        response=model_response,
                        step=record.step_id,
                    )
                raw_decision = (
                    interpreted if interpreted is not None else model_response
                )

        decision = self.normalize_decision(
            raw_decision, step=record.step_id, record=record
        )
        if decision.mode == "branch":
            decision = self.select_branch(state, observation, decision)

        if decision.mode not in {"act", "final", "wait", "handoff"}:
            raise ValueError(f"Invalid decision mode: {decision.mode}")

        decision.validate()
        if (
            model_response is not None
            and decision.mode == "act"
            and not record.native_tool_call_used
        ):
            self._append_parser_tool_call_history(
                response=model_response,
                decision=decision,
                record=record,
            )
        elif model_response is not None and decision.mode != "act":
            # A final/wait response has no tool result to pair with, but its
            # real assistant text is still useful audit history (and preserves
            # the generic QitOS history contract).
            content: Any = (
                model_response.text if str(model_response.text or "").strip() else None
            )
            if content is not None:
                engine._history_append(
                    "assistant",
                    content,
                    record.step_id,
                    metadata={"source": "engine"},
                )
        record.decision = decision
        record.actions = list(decision.actions)
        engine._memory_append("decision", decision, record.step_id)
        engine._emit(
            record.step_id,
            RuntimePhase.DECIDE,
            payload={
                "stage": "decision_ready",
                "mode": decision.mode,
                "rationale": decision.rationale,
                "actions": decision.actions,
                "final_answer": decision.final_answer,
                "candidate_count": len(decision.candidates),
            },
        )
        engine._dispatch_hook(
            "on_after_decide",
            engine._hook_context(
                step_id=record.step_id,
                phase=RuntimePhase.DECIDE,
                state=state,
                observation=observation,
                decision=decision,
                model_response=(
                    dict(record.model_response) if record.model_response else None
                ),
                record=record,
                payload=(
                    {"model_response": dict(record.model_response)}
                    if record.model_response
                    else {}
                ),
            ),
        )
        return cast(Decision[ActionT], decision)

    def _raise_for_empty_model_response(
        self, *, response: ModelResponse, step: int
    ) -> None:
        if str(response.text or "").strip() or response.tool_calls:
            return
        finish_reason = response.finish_reason
        raise ModelExecutionError(
            RuntimeErrorInfo(
                category=ErrorCategory.MODEL,
                message=(
                    "Model returned no text or tool calls "
                    f"(finish_reason={finish_reason!r})."
                ),
                phase=RuntimePhase.DECIDE.value,
                step_id=step,
                recoverable=True,
                details={
                    "code": "empty_model_response",
                    "finish_reason": finish_reason,
                    "usage": response.usage,
                    "model_name": response.model_name,
                    "provider": response.provider,
                    "max_recoveries": 1,
                },
            )
        )

    async def _run_llm_decide(
        self, state: StateT, observation: ObservationT, record: StepRecord
    ) -> ModelResponse:
        engine = self.engine
        if engine.agent.llm is None:
            raise ValueError("No llm configured and Agent.decide returned None")
        protocol = engine.resolve_protocol()
        setattr(engine.agent, "_runtime_observation", observation)
        setattr(engine.agent, "_runtime_step_id", record.step_id)
        setattr(engine.agent, "_runtime_protocol", protocol)
        setattr(
            engine.agent, "_runtime_protocol_source", engine._resolved_protocol_source
        )
        try:
            prompt_bundle = engine.agent.build_prompt_bundle(state)
            system_prompt = prompt_bundle.system_prompt
            prepared = engine.agent.prepare(state)
        finally:
            if hasattr(engine.agent, "_runtime_observation"):
                delattr(engine.agent, "_runtime_observation")
            if hasattr(engine.agent, "_runtime_step_id"):
                delattr(engine.agent, "_runtime_step_id")
            if hasattr(engine.agent, "_runtime_protocol"):
                delattr(engine.agent, "_runtime_protocol")
            if hasattr(engine.agent, "_runtime_protocol_source"):
                delattr(engine.agent, "_runtime_protocol_source")
        prompt_metadata = dict(getattr(prompt_bundle, "metadata", {}) or {})
        engine._last_prompt_metadata = dict(prompt_metadata)
        if engine.trace_writer is not None:
            engine.trace_writer.metadata.update(
                {
                    "prompt_hash": prompt_metadata.get("prompt_hash_full", "unknown"),
                    "prompt_hash_static": prompt_metadata.get(
                        "prompt_hash_static", "unknown"
                    ),
                    "prompt_builder": prompt_metadata.get("prompt_builder"),
                    "protocol": prompt_metadata.get("protocol"),
                }
            )
        prompt_messages = list(getattr(prompt_bundle, "message_injections", []) or [])
        prompt_user_content_blocks = list(
            getattr(prompt_bundle, "user_content_blocks", []) or []
        )
        context_runtime = engine._context_runtime
        # Apply critic patches if present
        effective_system_prompt = (
            system_prompt if isinstance(system_prompt, str) else ""
        )
        modified_prompt = engine._critic_modified_prompt
        if modified_prompt is not None:
            effective_system_prompt = modified_prompt
            engine._critic_modified_prompt = None  # Consume once
        instruction_patch = engine._critic_instruction_patch
        if instruction_patch is not None:
            engine._critic_instruction_patch = None  # Consume once
            effective_system_prompt = (
                effective_system_prompt + "\n\n" + instruction_patch
            )
        request_options = self._build_model_request_options(
            prompt_bundle=prompt_bundle,
            protocol=protocol,
        )
        pre_context = context_runtime.build_pre_request(
            llm=engine.agent.llm,
            system_prompt=effective_system_prompt,
            prepared=str(prepared),
            message_injections=prompt_messages,
            user_content_blocks=prompt_user_content_blocks,
            request_options=request_options,
        )
        messages: List[Dict[str, Any]] = []
        pending_system_history: Optional[str] = None
        pending_builder_history: List[Dict[str, Any]] = []
        if effective_system_prompt.strip():
            system = effective_system_prompt.strip()
            messages.append({"role": "system", "content": system})
            if system != engine._last_system_prompt:
                pending_system_history = system
        history: List[Dict[str, Any]] = []
        query = engine.history_policy.build_query(
            step_id=record.step_id,
            phase=RuntimePhase.DECIDE.value,
            query_kind="decide",
        )
        if isinstance(query, dict):
            query.setdefault("pending_content", str(prepared))
            query.setdefault(
                "model_name", getattr(getattr(engine.agent, "llm", None), "model", None)
            )
            query.setdefault("step_id", record.step_id)
            query.setdefault(
                "warning_ratio", float(engine.context_config.warning_ratio)
            )
            history_budget = context_runtime.compact_trigger_budget(pre_context)
            if history_budget is not None:
                current_max = query.get("max_tokens")
                if current_max is None:
                    query["max_tokens"] = history_budget
                else:
                    try:
                        query["max_tokens"] = min(int(current_max), int(history_budget))
                    except Exception:
                        query["max_tokens"] = history_budget
        history_impl: Any = None
        try:
            history_impl = engine._history()
            retrieved = history_impl.retrieve(
                state=state, observation=observation, query=query
            )
            history = engine._normalize_history_messages(retrieved)
            compact_events = []
            consume_runtime_events = getattr(
                history_impl, "consume_runtime_events", None
            )
            if callable(consume_runtime_events):
                compact_events = list(consume_runtime_events() or [])
            history_metadata = []
            get_last_message_metadata = getattr(
                history_impl, "get_last_message_metadata", None
            )
            if callable(get_last_message_metadata):
                history_metadata = list(get_last_message_metadata() or [])
        except Exception as exc:
            raw_history = getattr(history_impl, "messages", None)
            if not isinstance(raw_history, list) or not all(
                isinstance(message, HistoryMessage) for message in raw_history
            ):
                raise
            fallback = list(raw_history)
            roles = query.get("roles") if isinstance(query, dict) else None
            if roles:
                allowed_roles = {str(role) for role in roles}
                fallback = [
                    message for message in fallback if message.role in allowed_roles
                ]
            step_min = query.get("step_min") if isinstance(query, dict) else None
            step_max = query.get("step_max") if isinstance(query, dict) else None
            if step_min is not None:
                fallback = [
                    message for message in fallback if message.step_id >= int(step_min)
                ]
            if step_max is not None:
                fallback = [
                    message for message in fallback if message.step_id <= int(step_max)
                ]
            max_items = int(query.get("max_items") or 0)
            if max_items > 0:
                fallback = select_recent_history(fallback, max_items)
            history = engine._normalize_history_messages(fallback)
            history_metadata = [
                {
                    "role": message.role,
                    "step_id": message.step_id,
                    "content_chars": len(str(message.content or "")),
                    "projection_fallback": True,
                }
                for message in fallback
            ]
            compact_events = [
                {
                    "stage": "context_history",
                    "context": {
                        "stage": "compact_failed_fallback",
                        "strategy": history_impl.__class__.__name__,
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                        "messages_before": len(raw_history),
                        "messages_after": len(fallback),
                    },
                }
            ]
            _logger.warning(
                "history projection failed; using canonical bounded history: %s",
                exc,
            )
        pre_context = context_runtime.finalize_input(
            llm=engine.agent.llm,
            telemetry=pre_context,
            history_messages=history,
            compact_events=compact_events,
        )
        # --- Custom MessageBuilder support ---
        from ..core.message_builder import MessageBuilder as _MessageBuilderProto

        custom_builder = getattr(engine.agent, "message_builder", None)
        runtime_context_delivery: Dict[str, Any] = {
            "requested": "none",
            "effective": "none",
            "target_tool_call_id": None,
            "fallback_reason": None,
        }
        runtime_context_display: Dict[str, Any] | None = None
        # Only agents that explicitly opt in to CyberGym's controller-owned
        # Decision Context receive this strict packet invariant.  Generic
        # QitOS agents may use ``prepare()`` for ordinary text and must retain
        # their existing message-building behavior.
        decision_context_required = bool(
            getattr(custom_builder, "requires_decision_context", False)
        )
        # The default builder has no separate transient context, but the
        # final sidecar uses one common schema for both builder modes.
        runtime_context = ""
        if custom_builder is not None and isinstance(
            custom_builder, _MessageBuilderProto
        ):
            configured_rounds = int(
                getattr(engine.context_config, "conversation_max_rounds", 16)
            )
            if configured_rounds > 0:
                history = self._trim_native_tool_history(
                    history, max_rounds=configured_rounds
                )
            from ..core.message_builder import MessageBuildRequest as _MBReq

            build_req = _MBReq(
                step_id=record.step_id,
                state=state,
                observation=observation,
                prompt_bundle=prompt_bundle,
                prepared=str(prepared),
                history=history,
                record=record,
            )
            build_result = custom_builder.build_messages(build_req)
            messages = list(build_result.messages)
            runtime_context = str(
                getattr(build_result, "runtime_context", "") or ""
            ).strip()
            requested_delivery = (
                str(getattr(build_result, "runtime_context_delivery", "none") or "none")
                .strip()
                .lower()
            )
            if requested_delivery not in {"none", "merge_tool", "user"}:
                _logger.warning(
                    "Unknown MessageBuildResult runtime-context delivery %r; ignoring it",
                    requested_delivery,
                )
                requested_delivery = "none"
            runtime_context_delivery["requested"] = requested_delivery
            if runtime_context and requested_delivery != "none":
                wrapped_runtime_context = _wrap_runtime_context(runtime_context)
                should_merge = (
                    requested_delivery == "merge_tool"
                    and not self._current_user_has_multimodal_content(
                        prompt_user_content_blocks, observation, record
                    )
                )
                if should_merge:
                    merged, tool_call_id = self._merge_runtime_context_into_last_tool(
                        messages, wrapped_runtime_context
                    )
                    if merged:
                        runtime_context_delivery.update(
                            {
                                "effective": "merge_tool",
                                "target_tool_call_id": tool_call_id,
                            }
                        )
                        # Runtime Context is transient provider state, but every
                        # step must remain observable in the TUI/trace for
                        # offline decision analysis.
                        runtime_context_display = {
                            "tool_call_id": tool_call_id,
                            "content": wrapped_runtime_context,
                        }
                    else:
                        runtime_context_delivery["fallback_reason"] = (
                            "no_text_tool_result"
                        )
                elif requested_delivery == "merge_tool":
                    runtime_context_delivery["fallback_reason"] = (
                        "multimodal_user_content"
                    )

                if runtime_context_delivery["effective"] != "merge_tool":
                    current_user = self._build_current_user_message(
                        prepared_text=wrapped_runtime_context,
                        prompt_user_content_blocks=prompt_user_content_blocks,
                        observation=observation,
                        record=record,
                    )
                    messages.append(current_user)
                    runtime_context_delivery["effective"] = "user"
            pending_builder_history = [
                dict(entry)
                for entry in build_result.history_entries
                if isinstance(entry, dict)
            ]
            prepared_full = str(prepared)
        else:
            # --- Default message construction (original logic) ---
            injection_prefixes: List[str] = []
            if self._native_tool_call_preferred():
                if os.environ.get(
                    "CYBERGYM_DISABLE_HISTORY_TRIM", ""
                ).strip().lower() not in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }:
                    configured_rounds = int(
                        getattr(engine.context_config, "conversation_max_rounds", 10)
                    )
                    if configured_rounds > 0:
                        history = self._trim_native_tool_history(
                            history,
                            max_rounds=configured_rounds,
                        )
            messages.extend(history)
            for item in prompt_messages:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role") or "user")
                content = str(item.get("content") or "").strip()
                if not content:
                    continue
                if role == "user":
                    injection_prefixes.append(content)
                    continue
                messages.append({"role": role, "content": content})
            current_user_content = "\n\n".join(injection_prefixes + [str(prepared)])
            # Wrap runtime-state messages (step > 0) in semantic XML tags.
            # Step 0 is the initial task assignment and must remain unwrapped.
            if record.step_id > 0:
                current_user_content = _wrap_runtime_context(current_user_content)
            current_user = self._build_current_user_message(
                prepared_text=current_user_content,
                prompt_user_content_blocks=prompt_user_content_blocks,
                observation=observation,
                record=record,
            )
            messages.append(current_user)
            prepared_full = content_to_text(current_user.get("content"))
        # --- End custom MessageBuilder support ---
        # Normalize dangling calls before auditing or dispatching. The sidecar
        # and digest must describe the exact payload handed to the provider.
        messages = self._ensure_chain_consistency(messages)
        llm_messages = self._strip_internal_message_keys(messages)
        decision_context_source = runtime_context or str(prepared or "")
        source_context_blocks = _DECISION_CONTEXT_PATTERN.findall(
            decision_context_source
        )
        # A malformed transient source is a fixed controller/configuration
        # problem for Decision-Context-aware agents, but first give the state
        # projector one fresh chance to render it.  Normal duplicate/stale
        # packet recovery never reaches this branch.
        if decision_context_required and len(source_context_blocks) != 1:
            try:
                decision_context_source = str(engine.agent.prepare(state) or "")
            except Exception as exc:
                raise DecisionContextConfigurationError(
                    "could not render the authoritative DECISION_CONTEXT"
                ) from exc
            source_context_blocks = _DECISION_CONTEXT_PATTERN.findall(
                decision_context_source
            )
        if decision_context_required and len(source_context_blocks) != 1:
            raise DecisionContextConfigurationError(
                "authoritative runtime state must contain exactly one DECISION_CONTEXT"
            )
        pre_rebuild_messages = list(llm_messages)
        decision_context_recovery: Dict[str, Any] = {
            "rebuild_required": False,
            "reason": "",
            "before_count": 0,
            "after_count": 0,
            "authoritative_context": "",
        }
        if decision_context_required:
            llm_messages, decision_context_recovery = (
                self._normalize_decision_context_packet(
                    messages=llm_messages,
                    authoritative_source=decision_context_source,
                    delivery=runtime_context_delivery,
                )
            )
        pre_context = context_runtime.finalize_assembled_input(
            llm=engine.agent.llm,
            telemetry=pre_context,
            messages=llm_messages,
            request_options=request_options,
            compact_events=compact_events,
        )
        prompt_meter = getattr(engine.agent, "prompt_meter", None)
        if prompt_meter is not None and callable(
            getattr(prompt_meter, "measure", None)
        ):
            try:
                meter_result = _measure_prompt(
                    prompt_meter,
                    messages=llm_messages,
                    tools=list(request_options.get("tools") or []),
                    request_options=dict(request_options),
                    llm=engine.agent.llm,
                )
            except Exception as exc:
                meter_result = {
                    "status": "unavailable",
                    "meter_source": "provider_tokenize",
                    "meter_error": f"{type(exc).__name__}: {exc}",
                }
            pre_context = context_runtime.apply_prompt_meter(pre_context, meter_result)
            if (
                bool(getattr(prompt_meter, "required", False))
                and str(meter_result.get("status") or "") != "ready"
            ):
                raise ContextOverflowError("required prompt meter is unavailable")
        # Historical blocks are an audit signal only.  They are stripped from
        # the projected provider packet by the normalizer and never make a
        # long-running CyberGym task terminal.
        history_context_blocks = self._decision_context_blocks(
            self._strip_internal_message_keys(history)
        )
        provider_context_blocks = self._decision_context_blocks(llm_messages)
        expects_decision_context = decision_context_required
        if expects_decision_context and len(provider_context_blocks) != 1:
            raise DecisionContextConfigurationError(
                "packet normalization did not produce one current DECISION_CONTEXT"
            )
        if expects_decision_context and (
            len(source_context_blocks) != 1
            or provider_context_blocks[0] != source_context_blocks[0]
        ):
            raise DecisionContextConfigurationError(
                "packet normalization did not preserve the authoritative DECISION_CONTEXT"
            )
        if decision_context_recovery.get("rebuild_required"):
            self._write_pre_rebuild_packet_sidecar(
                state,
                record.step_id,
                messages=pre_rebuild_messages,
                recovery=decision_context_recovery,
            )
            engine._emit(
                record.step_id,
                RuntimePhase.DECIDE,
                payload={
                    "stage": "decision_context_packet_rebuilt",
                    "reason": decision_context_recovery.get("reason"),
                    "before_count": decision_context_recovery.get("before_count"),
                    "after_count": decision_context_recovery.get("after_count"),
                    "delivery": runtime_context_delivery.get("effective"),
                    "authoritative_context_hash": hashlib.sha256(
                        str(
                            decision_context_recovery.get("authoritative_context") or ""
                        ).encode("utf-8")
                    ).hexdigest(),
                },
            )
        parity = self._tool_transaction_parity(llm_messages)
        normalized_compact_events = context_runtime.normalize_history_events(
            compact_events, pre_context
        )
        if not normalized_compact_events:
            warning_event = context_runtime.maybe_note_warning(pre_context)
            if warning_event is not None:
                normalized_compact_events = [warning_event]
        for compact_event in normalized_compact_events:
            engine._emit(record.step_id, RuntimePhase.DECIDE, payload=compact_event)
        if context_runtime.should_overflow(pre_context):
            engine._emit(
                record.step_id,
                RuntimePhase.DECIDE,
                payload=context_runtime.overflow_event(pre_context),
            )
            raise ContextOverflowError(
                f"context overflow: input_tokens={pre_context.input_tokens_total} budget={pre_context.available_input_budget}"
            )
        if context_runtime.should_force_compact(pre_context):
            engine._emit(
                record.step_id,
                RuntimePhase.COMPACT,
                payload=context_runtime.force_compaction_event(pre_context),
            )
            raise ContextCompactionRequired(
                "context compaction required: "
                f"input_tokens={pre_context.input_tokens_total} "
                f"budget={pre_context.available_input_budget} "
                f"ratio={engine.context_config.compact_ratio}"
            )
        model_input_digest = self._model_input_digest(
            state, record.step_id, llm_messages
        )
        tool_schema_digest = self._tool_schema_digest(request_options.get("tools"))
        model_input_digest["tool_schema"] = tool_schema_digest
        self._write_assembled_messages_sidecar(state, record.step_id, llm_messages)
        self._write_model_input_bundle_sidecar(
            state,
            record.step_id,
            messages=llm_messages,
            request_options=request_options,
            prompt_bundle=prompt_bundle,
            protocol=protocol,
            runtime_context=runtime_context,
            runtime_context_delivery=runtime_context_delivery,
            context=context_runtime.telemetry_dict(pre_context),
            decision_context_recovery=decision_context_recovery,
        )
        record.prompt_metadata = dict(prompt_metadata)
        record.prompt_metadata.update(
            {
                "model_input_modalities": list(record.model_input_modalities),
                "model_input_visual_count": int(record.model_input_visual_count),
                "observation_modalities": list(record.observation_modalities),
                "runtime_context_delivery": dict(runtime_context_delivery),
                "tool_transaction_parity": parity,
                "decision_context_block_count": len(provider_context_blocks),
                "historical_decision_context_block_count": len(history_context_blocks),
            }
        )
        record.context = context_runtime.telemetry_dict(pre_context)
        engine._last_context_telemetry = dict(record.context)
        engine._emit(
            record.step_id,
            RuntimePhase.DECIDE,
            payload={
                "stage": "model_input",
                "tool_transaction_parity": parity,
                "prepared": str(prepared),
                "prepared_full": prepared_full,
                "history_message_count": len(history),
                "history_messages_meta": history_metadata,
                "message_count": len(llm_messages),
                "messages_summary": [
                    {
                        "role": m.get("role"),
                        "content_len": len(str(m.get("content", ""))),
                    }
                    for m in llm_messages
                ],
                "model_input_digest": model_input_digest,
                "tool_schema_digest": tool_schema_digest,
                "context": dict(record.context),
                "state_stats": self._state_stats(
                    observation, record.context, state=state
                ),
                "prompt": dict(record.prompt_metadata),
                "runtime_context_delivery": dict(runtime_context_delivery),
                "runtime_context_display": runtime_context_display,
            },
        )
        response = self._normalize_model_response(
            await self._call_llm(
                engine.agent.llm,
                llm_messages,
                request_options,
            )
        )
        post_context = context_runtime.finalize_output(
            llm=engine.agent.llm,
            telemetry=pre_context,
            raw_output=response.text,
            usage=response.usage,
        )
        if pending_system_history is not None:
            engine._history_append(
                "system",
                pending_system_history,
                record.step_id,
                metadata={"source": "engine"},
            )
            engine._last_system_prompt = pending_system_history
        for entry in pending_builder_history:
            engine._history_append(
                entry.get("role", "user"),
                entry.get("content", ""),
                entry.get("step_id", record.step_id),
                metadata=entry.get("metadata", {}),
                reasoning_content=entry.get("reasoning_content"),
                tool_calls=entry.get("tool_calls"),
                tool_call_id=entry.get("tool_call_id"),
                name=entry.get("name"),
                native_items=entry.get("native_items"),
            )
        if custom_builder is None:
            history_content = str(prepared)
            if record.step_id > 0:
                history_content = _wrap_runtime_context(history_content)
            engine._history_append(
                "user",
                history_content,
                record.step_id,
                metadata={"source": "engine"},
            )
        record.context = context_runtime.telemetry_dict(post_context)
        self._write_model_input_bundle_sidecar(
            state,
            record.step_id,
            messages=llm_messages,
            request_options=request_options,
            prompt_bundle=prompt_bundle,
            protocol=protocol,
            runtime_context=runtime_context,
            runtime_context_delivery=runtime_context_delivery,
            context=record.context,
            decision_context_recovery=decision_context_recovery,
        )
        record.model_response = response.to_summary_dict()
        engine._last_context_telemetry = dict(record.context)
        engine._emit(
            record.step_id,
            RuntimePhase.DECIDE,
            payload={
                "stage": "model_output",
                "raw_output": response.text,
                "reasoning_content": response.reasoning_content,
                "model_response": dict(record.model_response),
                "context": dict(record.context),
                "prompt": prompt_metadata,
            },
        )
        assistant_tool_calls = []
        if response.tool_calls and self._native_tool_call_preferred():
            assistant_tool_calls = [
                {
                    "id": item.get("id"),
                    "type": item.get("type", "function"),
                    "function": (
                        dict(item.get("function", {}))
                        if isinstance(item.get("function", {}), dict)
                        else {}
                    ),
                }
                for item in list(response.tool_calls or [])
                if isinstance(item, dict)
            ]
        # Native calls can be recorded immediately. Parser-derived actions are
        # normalized after parsing, when we know their action ids and schema.
        # That prevents an orphan plain assistant turn before a tool result.
        if assistant_tool_calls:
            assistant_content: Any = response.text
            if not str(response.text or "").strip():
                assistant_content = None
            engine._history_append(
                "assistant",
                assistant_content,
                record.step_id,
                metadata={"source": "engine"},
                reasoning_content=response.reasoning_content,
                tool_calls=assistant_tool_calls,
                native_items=response.native_items,
            )

        return response

    def _append_parser_tool_call_history(
        self,
        *,
        response: ModelResponse,
        decision: Decision[Any],
        record: StepRecord,
    ) -> None:
        """Store parsed tool actions in the same assistant -> tool shape.

        Text protocols do not provide provider-assigned call ids. Stable ids
        are generated once here and copied onto the Action, so the executor's
        matching tool result has the exact same ``tool_call_id``.
        """
        calls: List[Dict[str, Any]] = []
        for index, item in enumerate(list(decision.actions or [])):
            action = item if isinstance(item, Action) else Action.from_dict(dict(item))
            call_id = action.action_id or f"call_{record.step_id}_{index}"
            action.action_id = call_id
            if action is not item:
                decision.actions[index] = action
            try:
                arguments = json.dumps(dict(action.args or {}), ensure_ascii=False)
            except Exception:
                arguments = "{}"
            calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": action.name, "arguments": arguments},
                }
            )
        if not calls:
            return
        content: Any = response.text if str(response.text or "").strip() else None
        self.engine._history_append(
            "assistant",
            content,
            record.step_id,
            metadata={"source": "engine", "decision_source": "parser"},
            reasoning_content=response.reasoning_content,
            tool_calls=calls,
        )
        record.history_tool_calls_pending = True

    def _model_input_digest(
        self,
        state: StateT,
        step_id: int,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        try:
            serialized = json.dumps(
                messages, ensure_ascii=False, sort_keys=True, default=str
            )
        except Exception:
            serialized = json.dumps([str(m) for m in messages], ensure_ascii=False)

        role_counts: Dict[str, int] = {}
        tool_call_count = 0
        message_summaries: List[Dict[str, Any]] = []
        for idx, message in enumerate(messages):
            role = str(message.get("role") or "")
            role_counts[role] = role_counts.get(role, 0) + 1
            content_text = content_to_text(message.get("content"))
            tool_calls = message.get("tool_calls")
            tool_names: List[str] = []
            if isinstance(tool_calls, list):
                tool_call_count += len(tool_calls)
                for call in tool_calls[:6]:
                    if not isinstance(call, dict):
                        continue
                    fn = call.get("function")
                    if isinstance(fn, dict) and fn.get("name"):
                        tool_names.append(str(fn.get("name")))
                    elif call.get("name"):
                        tool_names.append(str(call.get("name")))
            summary: Dict[str, Any] = {
                "index": idx,
                "role": role,
                "content_len": len(content_text),
            }
            if message.get("name"):
                summary["name"] = message.get("name")
            if message.get("tool_call_id"):
                summary["tool_call_id"] = message.get("tool_call_id")
            if tool_names:
                summary["tool_names"] = tool_names
            message_summaries.append(summary)

        system_text = "\n".join(
            content_to_text(message.get("content"))
            for message in messages
            if message.get("role") == "system"
        )
        sections = {
            "contract": "# CyberGym Agent Contract" in system_text
            or "Core Rules" in system_text,
            "runtime_context": "<runtime_context>" in system_text,
            "exploration_prompt": "# Exploration Phase" in system_text,
            "construction_prompt": "# Construction Phase" in system_text,
            "runtime_reminder": "<runtime_reminder>" in system_text,
        }

        sidecar_path = ""
        try:
            metadata = dict(getattr(state, "metadata", {}) or {})
            trace_root = str(metadata.get("trace_run_dir") or "").strip()
            if trace_root:
                sidecar_path = str(
                    Path(trace_root)
                    / "agent_steps"
                    / f"step-{int(step_id):04d}"
                    / "assembled_messages.json"
                )
        except Exception:
            sidecar_path = ""

        return {
            "message_count": len(messages),
            "role_counts": role_counts,
            "tool_call_count": tool_call_count,
            "messages_hash": hashlib.sha256(serialized.encode("utf-8")).hexdigest()[
                :16
            ],
            "sections": sections,
            "messages": message_summaries,
            "recent_history": message_summaries[-8:],
            "sidecar_path": sidecar_path,
        }

    def _write_pre_rebuild_packet_sidecar(
        self,
        state: StateT,
        step_id: int,
        *,
        messages: List[Dict[str, Any]],
        recovery: Dict[str, Any],
    ) -> None:
        """Persist a rejected packet only when Context normalization repairs it."""
        try:
            metadata = dict(getattr(state, "metadata", {}) or {})
            trace_root = str(
                metadata.get("trace_run_dir")
                or getattr(state, "workspace_root", "")
                or ""
            ).strip()
            if not trace_root:
                return
            step_dir = Path(trace_root) / "agent_steps" / f"step-{int(step_id):04d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            (step_dir / "assembled_messages.pre_rebuild.json").write_text(
                json.dumps(messages, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            (step_dir / "decision_context_recovery.json").write_text(
                json.dumps(
                    {
                        "reason": recovery.get("reason"),
                        "before_count": recovery.get("before_count"),
                        "after_count": recovery.get("after_count"),
                        "authoritative_context_hash": hashlib.sha256(
                            str(recovery.get("authoritative_context") or "").encode(
                                "utf-8"
                            )
                        ).hexdigest(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            _logger.debug("pre-rebuild packet sidecar write failed", exc_info=True)

    def _write_assembled_messages_sidecar(
        self,
        state: StateT,
        step_id: int,
        messages: List[Dict[str, Any]],
    ) -> None:
        try:
            metadata = dict(getattr(state, "metadata", {}) or {})
            trace_root = str(metadata.get("trace_run_dir") or "").strip()
            if not trace_root:
                return
            step_dir = Path(trace_root) / "agent_steps" / f"step-{int(step_id):04d}"
            step_dir.mkdir(parents=True, exist_ok=True)

            # Context can shrink after compaction, so count-based increments do
            # not form a replayable audit trail. Every step gets the exact full
            # message list that was hashed and sent to the model.
            (step_dir / "assembled_messages.json").write_text(
                json.dumps(messages, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # Keep the canonical CyberGym state beside the exact prompt that
            # consumed it. This makes an exported trace independently
            # auditable even when the task workspace is later removed.
            workspace = str(getattr(state, "workspace_root", "") or "").strip()
            if workspace:
                state_path = Path(workspace) / ".cybergym" / "state.json"
                if state_path.is_file():
                    (step_dir / "cybergym_state.json").write_text(
                        state_path.read_text(encoding="utf-8"), encoding="utf-8"
                    )

            self._last_message_count = len(messages)
            self._last_full_step = step_id
        except Exception:
            return

    @staticmethod
    def _tool_schema_digest(tools: Any) -> Dict[str, Any]:
        payload = list(tools or []) if isinstance(tools, list) else []
        serialized = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, default=str
        )
        names = [
            str((item.get("function") or {}).get("name") or "")
            for item in payload
            if isinstance(item, dict)
        ]
        return {
            "tool_count": len(payload),
            "tool_names": [name for name in names if name],
            "schema_hash": hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16],
        }

    def _write_model_input_bundle_sidecar(
        self,
        state: StateT,
        step_id: int,
        *,
        messages: List[Dict[str, Any]],
        request_options: Dict[str, Any],
        prompt_bundle: Any,
        protocol: Any,
        runtime_context: str = "",
        runtime_context_delivery: Dict[str, Any] | None = None,
        context: Dict[str, Any] | None = None,
        decision_context_recovery: Dict[str, Any] | None = None,
    ) -> None:
        try:
            metadata = dict(getattr(state, "metadata", {}) or {})
            trace_root = str(metadata.get("trace_run_dir") or "").strip()
            if not trace_root:
                return
            step_dir = Path(trace_root) / "agent_steps" / f"step-{int(step_id):04d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            tools = list(request_options.get("tools") or [])
            message_json = json.dumps(
                messages, ensure_ascii=False, sort_keys=True, default=str
            )
            tools_json = json.dumps(
                tools, ensure_ascii=False, sort_keys=True, default=str
            )
            combined = message_json + "\n" + tools_json
            decision_context_blocks = self._decision_context_blocks(messages)
            bundle = {
                "messages": messages,
                "tools": tools,
                "prompt_metadata": dict(getattr(prompt_bundle, "metadata", {}) or {}),
                "protocol": str(getattr(protocol, "id", "") or ""),
                "tool_delivery": str(
                    dict(getattr(prompt_bundle, "metadata", {}) or {}).get(
                        "tool_schema_delivery"
                    )
                    or ""
                ),
                "pre_merge_runtime_context": str(runtime_context or ""),
                "runtime_context_delivery": dict(runtime_context_delivery or {}),
                "context": dict(context or {}),
                "messages_hash": hashlib.sha256(
                    message_json.encode("utf-8")
                ).hexdigest()[:16],
                "schema_hash": hashlib.sha256(tools_json.encode("utf-8")).hexdigest()[
                    :16
                ],
                "combined_hash": hashlib.sha256(combined.encode("utf-8")).hexdigest()[
                    :16
                ],
                "decision_context_block_count": len(decision_context_blocks),
                "decision_context_sha256": (
                    hashlib.sha256(
                        decision_context_blocks[0].encode("utf-8")
                    ).hexdigest()
                    if len(decision_context_blocks) == 1
                    else ""
                ),
                "decision_context_recovery": {
                    "rebuilt": bool(
                        (decision_context_recovery or {}).get("rebuild_required")
                    ),
                    "reason": str(
                        (decision_context_recovery or {}).get("reason") or ""
                    ),
                    "before_count": int(
                        (decision_context_recovery or {}).get("before_count") or 0
                    ),
                    "after_count": int(
                        (decision_context_recovery or {}).get("after_count") or 0
                    ),
                },
            }
            (step_dir / "model_input_bundle.json").write_text(
                json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if len(decision_context_blocks) == 1:
                (step_dir / "decision_context.md").write_text(
                    decision_context_blocks[0] + "\n", encoding="utf-8"
                )
        except Exception:
            _logger.debug("model input bundle sidecar write failed", exc_info=True)

    @staticmethod
    def _decision_context_blocks(messages: List[Dict[str, Any]]) -> List[str]:
        """Return actual non-system Decision Context blocks in a packet."""
        blocks: List[str] = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") == "system":
                continue
            blocks.extend(
                _DECISION_CONTEXT_PATTERN.findall(
                    content_to_text(message.get("content"))
                )
            )
        return blocks

    def _build_model_request_options(
        self, *, prompt_bundle: Any, protocol: Any
    ) -> Dict[str, Any]:
        metadata = dict(getattr(prompt_bundle, "metadata", {}) or {})
        delivery = str(metadata.get("tool_schema_delivery") or "prompt_injection")
        payload = getattr(prompt_bundle, "tool_schema_payload", None)
        llm = getattr(self.engine.agent, "llm", None)
        options: Dict[str, Any] = {}

        # Model defaults are the baseline. Turn-owned tool projection is applied
        # afterwards so stale or user-supplied defaults cannot replace the exact
        # tool exposure frozen for this request.
        if llm is not None:
            default_kwargs = getattr(llm, "default_request_kwargs", None)
            if isinstance(default_kwargs, dict) and default_kwargs:
                options.update(default_kwargs)

        if llm is not None and delivery in {"api_parameter", "hybrid"}:
            build_options = getattr(llm, "build_tool_schema_request_options", None)
            if callable(build_options):
                try:
                    options.update(
                        build_options(payload, protocol=protocol, delivery=delivery)
                        or {}
                    )
                except Exception:
                    _logger.debug(
                        "build_tool_schema_request_options failed", exc_info=True
                    )

        return options

    async def _call_llm(
        self,
        llm: Any,
        messages: List[Dict[str, Any]],
        request_options: Dict[str, Any],
    ) -> ModelResponse:
        """Consume the one canonical model stream into a completed response."""

        if not isinstance(llm, Model):
            raise TypeError(
                "Agent.llm must implement the asynchronous qitos.models.Model "
                "stream contract"
            )
        handler = to_stream_handler(self.stream_callback)
        accumulated_text: List[str] = []
        accumulated_reasoning: List[str] = []
        final_usage: Mapping[str, Any] | None = None
        final_tool_calls: Optional[List[Dict[str, Any]]] = None
        final_native_items: Optional[List[Dict[str, Any]]] = None
        final_finish_reason: Optional[str] = None
        final_metadata: Dict[str, Any] = {}
        started = False
        terminal_seen = False
        stream_error: Exception | None = None
        request_started_at = time.monotonic()
        first_event_at: float | None = None
        first_content_at: float | None = None
        stream_iter: AsyncIterator[ModelStreamChunk] = llm.stream(
            messages,
            deadline_monotonic=getattr(self.engine, "runtime_deadline_monotonic", None),
            **request_options,
        )

        iterator = stream_iter.__aiter__()
        try:
            while True:
                remaining = self.engine.remaining_runtime_seconds()
                if remaining is not None and remaining <= 0:
                    raise ModelRequestDeadlineExceeded("model request deadline expired")
                try:
                    next_chunk = iterator.__anext__()
                    chunk = (
                        await asyncio.wait_for(next_chunk, timeout=remaining)
                        if remaining is not None
                        else await next_chunk
                    )
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as exc:
                    raise ModelRequestDeadlineExceeded(
                        "model request deadline expired"
                    ) from exc
                if not isinstance(chunk, ModelStreamChunk):
                    raise TypeError("Model.stream() must yield ModelStreamChunk values")
                chunk_received_at = time.monotonic()
                if first_event_at is None:
                    first_event_at = chunk_received_at
                if first_content_at is None and _chunk_has_model_content(chunk):
                    first_content_at = chunk_received_at
                text = chunk.text
                reasoning = chunk.reasoning_content
                done = chunk.done
                usage = chunk.usage
                tool_calls = chunk.tool_calls
                native_items = chunk.native_items
                event_type = chunk.event_type
                finish_reason = chunk.finish_reason

                observable = bool(
                    text
                    or reasoning
                    or done
                    or tool_calls
                    or native_items
                    or event_type
                )
                if observable and not started:
                    started = True
                    if handler is not None:
                        try:
                            handler.on_start()
                        except Exception:
                            pass

                if observable and handler is not None:
                    on_chunk = getattr(handler, "on_chunk", None)
                    if callable(on_chunk):
                        try:
                            on_chunk(chunk)
                        except Exception:
                            pass

                if text:
                    accumulated_text.append(text)
                    if handler is not None:
                        try:
                            handler.on_delta(text)
                        except Exception:
                            pass

                if reasoning:
                    accumulated_reasoning.append(str(reasoning))

                if done:
                    if terminal_seen:
                        raise ModelTransportError(
                            "model stream emitted more than one terminal chunk",
                            attempts=1,
                            retryable=False,
                        )
                    terminal_seen = True
                    if usage is not None and isinstance(usage, Mapping):
                        final_usage = usage
                    if tool_calls is not None and isinstance(tool_calls, list):
                        final_tool_calls = tool_calls
                    if native_items is not None and isinstance(native_items, list):
                        final_native_items = native_items
                    if finish_reason is not None:
                        final_finish_reason = str(finish_reason)
                    final_metadata = dict(chunk.event_metadata)
            if not terminal_seen:
                raise ModelTransportError(
                    "model stream ended before a terminal chunk",
                    attempts=1,
                    retryable=True,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            stream_error = exc
            if handler is not None:
                on_error = getattr(handler, "on_error", None)
                if callable(on_error):
                    try:
                        on_error(exc)
                    except Exception:
                        pass
            raise
        finally:
            close = getattr(stream_iter, "aclose", None)
            if callable(close):
                try:
                    await close()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    _logger.debug("model stream close failed", exc_info=True)
            if (
                handler is not None
                and started
                and terminal_seen
                and stream_error is None
            ):
                try:
                    handler.on_end()
                except Exception:
                    pass

        completed_at = time.monotonic()
        return ModelResponse(
            text="".join(accumulated_text),
            usage=final_usage,
            finish_reason=final_finish_reason,
            tool_calls=final_tool_calls,
            model_name=llm.model,
            provider=llm.provider_name,
            metadata=final_metadata,
            reasoning_content=(
                "".join(accumulated_reasoning) if accumulated_reasoning else None
            ),
            native_items=final_native_items,
            timing=ModelTiming(
                total_ms=(completed_at - request_started_at) * 1000,
                time_to_first_event_ms=(
                    (first_event_at - request_started_at) * 1000
                    if first_event_at is not None
                    else None
                ),
                time_to_first_content_ms=(
                    (first_content_at - request_started_at) * 1000
                    if first_content_at is not None
                    else None
                ),
            ),
        )

    def _build_current_user_message(
        self,
        *,
        prepared_text: str,
        prompt_user_content_blocks: List[Dict[str, Any]],
        observation: ObservationT,
        record: StepRecord,
    ) -> Dict[str, Any]:
        content_blocks: List[Dict[str, Any]] = []
        if str(prepared_text or "").strip():
            content_blocks.append(text_block(str(prepared_text)))

        task_blocks = self._task_visual_blocks()
        observation_blocks = self._observation_visual_blocks(observation, record)
        content_blocks.extend(
            [normalize_content_block(block) for block in prompt_user_content_blocks]
        )
        content_blocks.extend(task_blocks)
        content_blocks.extend(observation_blocks)

        record.model_input_modalities = self._content_modalities(content_blocks)
        record.model_input_visual_count = sum(
            1 for block in content_blocks if str(block.get("type") or "text") != "text"
        )
        if record.model_input_visual_count > 0 and not self._llm_supports_multimodal(
            getattr(self.engine.agent, "llm", None)
        ):
            raise ValueError(
                "Configured model adapter does not support multimodal input content blocks."
            )
        if record.model_input_visual_count > 0:
            return {"role": "user", "content": content_blocks}
        return {"role": "user", "content": str(prepared_text or "")}

    def _current_user_has_multimodal_content(
        self,
        prompt_user_content_blocks: List[Dict[str, Any]],
        observation: ObservationT,
        record: StepRecord,
    ) -> bool:
        """Return whether a runtime-context turn must stay a user message.

        Tool results are serialized text in the OpenAI-compatible providers
        supported by QitOS.  Do not discard image/file inputs merely to retain
        a tool-terminal conversation shape.
        """
        blocks: List[Dict[str, Any]] = [
            normalize_content_block(block) for block in prompt_user_content_blocks
        ]
        blocks.extend(self._task_visual_blocks())
        blocks.extend(self._observation_visual_blocks(observation, record))
        return any(str(block.get("type") or "text") != "text" for block in blocks)

    def _content_modalities(self, content_blocks: List[Dict[str, Any]]) -> List[str]:
        modalities: List[str] = []
        for block in content_blocks:
            block_type = str(block.get("type") or "text")
            if block_type == "text":
                if "text" not in modalities:
                    modalities.append("text")
                continue
            if block_type in {"image_url", "image_base64", "image_file"}:
                if "image" not in modalities:
                    modalities.append("image")
                continue
            if block_type not in modalities:
                modalities.append(block_type)
        return modalities

    def _llm_supports_multimodal(self, llm: Any) -> bool:
        supports = getattr(llm, "supports_multimodal_input", None)
        if callable(supports):
            try:
                return bool(supports())
            except Exception:
                return False
        return True

    def _task_workspace_root(self) -> Optional[Path]:
        task_obj = getattr(self.engine, "_active_task_obj", None)
        env_spec = getattr(task_obj, "env_spec", None)
        config = getattr(env_spec, "config", None)
        if isinstance(config, dict):
            root = str(config.get("workspace_root") or "").strip()
            if root:
                return Path(root).expanduser().resolve()
        return None

    def _task_visual_blocks(self) -> List[Dict[str, Any]]:
        task_obj = getattr(self.engine, "_active_task_obj", None)
        resources = list(getattr(task_obj, "resources", []) or [])
        workspace_root = self._task_workspace_root()
        blocks: List[Dict[str, Any]] = []
        for item in resources:
            kind = str(getattr(item, "kind", "") or "").strip().lower()
            metadata = dict(getattr(item, "metadata", {}) or {})
            modality = str(metadata.get("modality") or "").strip().lower()
            if kind != "image" and modality != "image":
                continue
            detail = str(metadata.get("detail") or "").strip() or None
            uri = str(getattr(item, "uri", "") or "").strip()
            path = str(getattr(item, "path", "") or "").strip()
            if uri:
                blocks.append(
                    image_url_block(
                        uri,
                        detail=detail,
                        metadata={"source": "task_resource", "kind": kind},
                    )
                )
                continue
            if path:
                resolved = Path(path).expanduser()
                if not resolved.is_absolute() and workspace_root is not None:
                    resolved = (workspace_root / resolved).resolve()
                blocks.append(
                    image_file_block(
                        str(resolved),
                        detail=detail,
                        metadata={"source": "task_resource", "kind": kind},
                    )
                )
        return blocks

    def _observation_visual_blocks(
        self, observation: ObservationT, record: StepRecord
    ) -> List[Dict[str, Any]]:
        env_observation = getattr(self.engine, "_last_env_observation", None)
        payload = self._observation_pack_payload(env_observation, observation)
        if payload is None:
            return []
        record.observation_modalities = observation_modalities(payload)
        record.visual_assets = observation_visual_assets(
            payload, source_step=record.step_id
        )
        record.visual_asset_count = len(record.visual_assets)
        record.has_screenshot = "screenshot" in record.observation_modalities
        record.has_dom = "dom" in record.observation_modalities
        record.has_accessibility_tree = (
            "accessibility_tree" in record.observation_modalities
        )
        pack = normalize_observation_pack(payload)
        if pack is None or not isinstance(pack.screenshot, dict):
            return []
        screenshot = dict(pack.screenshot)
        detail = str(screenshot.get("detail") or "high").strip() or "high"
        metadata: Dict[str, Any] = {"source": "env_observation"}
        if pack.metadata:
            metadata["observation"] = dict(pack.metadata)
        if screenshot.get("url"):
            return [
                image_url_block(
                    str(screenshot.get("url") or ""),
                    detail=detail,
                    mime_type=str(screenshot.get("mime_type") or ""),
                    metadata=metadata,
                )
            ]
        if screenshot.get("path"):
            return [
                image_file_block(
                    str(screenshot.get("path") or ""),
                    mime_type=str(screenshot.get("mime_type") or ""),
                    detail=detail,
                    metadata=metadata,
                )
            ]
        data_value = (
            screenshot.get("data_url")
            or screenshot.get("data")
            or screenshot.get("base64")
        )
        if data_value:
            return [
                image_base64_block(
                    str(data_value),
                    mime_type=str(screenshot.get("mime_type") or "image/png"),
                    detail=detail,
                    metadata=metadata,
                )
            ]
        return []

    def _observation_summary(self, observation: ObservationT) -> Dict[str, Any]:
        """Return a lightweight summary of the observation for trace events.

        Avoids serializing the full observation into events.jsonl — the
        full data is available via steps.jsonl if needed.
        """
        summary: Dict[str, Any] = {"type": type(observation).__name__}
        if isinstance(observation, Observation):
            action_results = getattr(observation, "action_results", None)
            if action_results is not None:
                summary["action_result_count"] = len(action_results)
            if isinstance(observation.state, dict):
                summary["state_keys"] = list(observation.state.keys())[:20]
        elif isinstance(observation, dict):
            summary["keys"] = list(observation.keys())[:20]
            ar = observation.get("action_results")
            if isinstance(ar, list):
                summary["action_result_count"] = len(ar)
        return summary

    def _observation_pack_payload(
        self, env_observation: Any, observation: ObservationT
    ) -> Dict[str, Any] | None:
        if env_observation is not None:
            data = getattr(env_observation, "data", None)
            if isinstance(data, dict):
                multimodal = data.get("multimodal")
                if isinstance(multimodal, dict):
                    return multimodal
                if normalize_observation_pack(data) is not None:
                    return data
        if isinstance(observation, Observation):
            env_payload = observation.env
            if isinstance(env_payload, dict):
                env_obs = env_payload.get("observation")
                if isinstance(env_obs, dict):
                    data = env_obs.get("data")
                    if isinstance(data, dict):
                        multimodal = data.get("multimodal")
                        if isinstance(multimodal, dict):
                            return multimodal
                        if normalize_observation_pack(data) is not None:
                            return data
        if isinstance(observation, dict):
            env_payload_dict = observation.get("env")
            if isinstance(env_payload_dict, dict):
                env_obs = env_payload_dict.get("observation")
                if isinstance(env_obs, dict):
                    data = env_obs.get("data")
                    if isinstance(data, dict):
                        multimodal = data.get("multimodal")
                        if isinstance(multimodal, dict):
                            return multimodal
                        if normalize_observation_pack(data) is not None:
                            return data
        return None

    def _state_stats(
        self,
        observation: ObservationT,
        context: Dict[str, Any],
        state: Any = None,
    ) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        if isinstance(observation, Observation):
            stats["action_results"] = len(observation.action_results or [])
            if isinstance(observation.state, dict):
                scratchpad = observation.state.get("scratchpad")
                if isinstance(scratchpad, list):
                    stats["scratchpad_items"] = len(scratchpad)
        if isinstance(observation, dict):
            scratchpad = observation.get("scratchpad")
            if isinstance(scratchpad, list):
                stats["scratchpad_items"] = len(scratchpad)
            elif isinstance(scratchpad, str) and scratchpad.strip():
                stats["scratchpad_items"] = 1
            memory = observation.get("memory")
            if isinstance(memory, dict) and isinstance(memory.get("records"), list):
                stats["memory_records"] = len(memory.get("records") or [])
            workspace_files = observation.get("workspace_files")
            if isinstance(workspace_files, list):
                stats["workspace_files"] = len(workspace_files)
        for key in (
            "input_tokens_total",
            "available_input_budget",
            "system_prompt_tokens",
            "prepared_tokens",
            "history_tokens",
            "output_tokens",
            "occupancy_ratio",
            "context_window",
            "counting_mode",
            "provider_prompt_tokens",
            "provider_completion_tokens",
            "provider_total_tokens",
            "planned_prompt_tokens",
            "cached_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "usage_source",
            "meter_source",
            "meter_status",
            "token_estimate_error",
        ):
            if key in context:
                stats[key] = context.get(key)
        # Extract chain/gate/memory text from agent state for TUI.
        # Prefer the live state object (passed from run_decide) over the
        # serialised observation.state dict, since the live object has
        # ChainNode/ChainGate dataclass instances and metadata.
        state_obj = state
        if state_obj is None:
            state_obj = getattr(observation, "state", None)
            if state_obj is None and isinstance(observation, dict):
                state_obj = observation.get("state")
        if state_obj is not None:
            # Use the agent's own rendering methods if available
            constraint_lines = self._extract_constraint_board_text(state_obj)
            if constraint_lines:
                stats["constraint_board"] = constraint_lines
            task_memory_text = self._extract_task_memory_text(state_obj)
            if task_memory_text:
                stats["task_memory"] = task_memory_text
            # Agent business phase and control mode for TUI badge
            for attr in ("current_phase", "control_mode"):
                val = getattr(state_obj, attr, None)
                if val:
                    stats[attr] = val
            # Metadata-based overrides (cached by agent.prepare() and reduce())
            metadata = getattr(state_obj, "metadata", None)
            if isinstance(metadata, dict):
                tui_phase = metadata.get("_tui_phase")
                if isinstance(tui_phase, str) and tui_phase.strip():
                    stats["current_phase"] = tui_phase
                sink_text = metadata.get("_tui_sink_candidates")
                if isinstance(sink_text, str) and sink_text.strip():
                    stats["sink_candidates"] = sink_text
                objective_text = metadata.get("_tui_objective")
                if isinstance(objective_text, str) and objective_text.strip():
                    stats["objective"] = objective_text
                task_ctx_text = metadata.get("_tui_task_context")
                if isinstance(task_ctx_text, str) and task_ctx_text.strip():
                    stats["task_context"] = task_ctx_text
                allowed_tools_text = metadata.get("_tui_allowed_tools")
                if isinstance(allowed_tools_text, str) and allowed_tools_text.strip():
                    stats["allowed_tools"] = allowed_tools_text
                suggested_sinks_text = metadata.get("_tui_suggested_sinks")
                if (
                    isinstance(suggested_sinks_text, str)
                    and suggested_sinks_text.strip()
                ):
                    stats["suggested_sinks"] = suggested_sinks_text
        return stats

    @staticmethod
    def _extract_constraint_board_text(state_obj: Any) -> str:
        """Extract the Constraint Board section text from agent state.

        The agent stores the exact same text in state.metadata that the LLM
        sees in the observation packet.  This ensures TUI and LLM always
        see identical content.
        """
        # Primary: use the pre-rendered text from the agent's prepare()
        metadata = getattr(state_obj, "metadata", None)
        if isinstance(metadata, dict):
            cached = metadata.get("_tui_constraint_board")
            if isinstance(cached, str) and cached.strip():
                return cached
        # Fallback: build from state fields directly (for agents that
        # don't store the cached text, or before first prepare())
        nodes = list(getattr(state_obj, "call_chain_nodes", []) or [])
        gates = list(getattr(state_obj, "call_chain_gates", []) or [])
        if not nodes and not gates:
            return ""
        lines: List[str] = []
        confirmed = sum(1 for g in gates if getattr(g, "status", "") == "confirmed")
        open_g = sum(
            1 for g in gates if getattr(g, "status", "") in ("inferred", "unknown")
        )
        refuted = sum(1 for g in gates if getattr(g, "status", "") == "refuted")
        lines.append(
            f"Chain Gates: {confirmed} confirmed / {open_g} open / {refuted} refuted"
        )
        if nodes:
            sorted_nodes = sorted(nodes, key=lambda n: getattr(n, "order", 0))
            for n in sorted_nodes[:10]:
                role = getattr(n, "role", "?")
                func = getattr(n, "function", "?")
                loc = getattr(n, "location", "")
                status = getattr(n, "status", "?")
                lines.append(
                    f"  [{getattr(n, 'order', 0)}] [{status}] {role} {func} ({loc})"
                )
        for g in gates:
            status = getattr(g, "status", "")
            desc = getattr(g, "description", "")
            cond = getattr(g, "required_condition", "")
            hint = getattr(g, "repair_hint", "")
            ev = getattr(g, "evidence", "")
            parts = [f"  [{status}/{getattr(g, 'gate_type', '')}] {desc}"]
            if cond:
                parts.append(f"    required: {cond}")
            if hint:
                parts.append(f"    repair: {hint}")
            if ev:
                parts.append(f"    evidence: {ev}")
            lines.extend(parts)
        return "\n".join(lines)

    @staticmethod
    def _extract_task_memory_text(state_obj: Any) -> str:
        """Extract Task Memory section text from agent state.

        Same text the LLM sees in the observation packet.
        """
        metadata = getattr(state_obj, "metadata", None)
        if isinstance(metadata, dict):
            cached = metadata.get("_tui_task_memory")
            if isinstance(cached, str) and cached.strip():
                return cached
        # Fallback
        parts: List[str] = []
        va = getattr(state_obj, "vulnerability_analysis", "")
        if isinstance(va, str) and va.strip():
            parts.append(f"Analysis: {va.strip()}")
        ch = getattr(state_obj, "current_hypothesis", "")
        if isinstance(ch, str) and ch.strip():
            parts.append(f"Hypothesis: {ch.strip()}")
        pt = getattr(state_obj, "path_trace", None)
        if isinstance(pt, list) and pt:
            parts.append("Path: " + " → ".join(pt[:8]))
        return "\n".join(parts)

    def select_branch(
        self,
        state: StateT,
        observation: ObservationT,
        branch_decision: Decision[ActionT],
    ) -> Decision[ActionT]:
        engine = self.engine
        if engine.search is not None:
            candidates = engine.search.expand(
                state, observation, branch_decision
            ) or list(branch_decision.candidates)
            scores = engine.search.score(state, observation, candidates)
            candidates = engine.search.prune(candidates, scores)
            if not candidates:
                new_state = engine.search.backtrack(state)
                if new_state is not state:
                    state.__dict__.update(new_state.__dict__)
                return Decision.wait(rationale="search backtrack")
            scores = engine.search.score(state, observation, candidates)
            selected = engine.search.select(candidates, scores)
            mark_selected = getattr(engine.search, "mark_selected", None)
            if callable(mark_selected):
                mark_selected(state, selected)
        else:
            selected = engine.branch_selector.select(
                branch_decision.candidates, state, observation
            )
        selected.validate()
        return selected

    def normalize_decision(
        self, raw_decision: Any, step: int, record: StepRecord | None = None
    ) -> Decision[ActionT]:
        if isinstance(raw_decision, Decision):
            if record is not None and not record.decision_source:
                record.decision_source = "agent"
            return raw_decision

        response = raw_decision if isinstance(raw_decision, ModelResponse) else None
        native_decision = self._decision_from_native_tool_calls(
            response=response,
            step=step,
            record=record,
        )
        if native_decision is not None:
            return native_decision
        parser_input = response.text if response is not None else raw_decision

        # When native tool calling is preferred and the model returned plain
        # text without tool_calls, treat it as a final answer — the model is
        # done acting and is giving its summary/conclusion in natural language.
        # Parsers (especially json_decision_v1) will misinterpret natural
        # language as invalid JSON and return wait(), which causes the agent
        # to loop forever without ever producing a final result.
        is_native_text_response = (
            response is not None
            and self._native_tool_call_preferred()
            and not (isinstance(response.tool_calls, list) and response.tool_calls)
            and str(response.text or "").strip()
        )

        if is_native_text_response and response is not None:
            # Still try the parser chain first — if the model happened to
            # produce valid structured output (JSON with a final_answer, or
            # ReAct "Final Answer:" label), let the parser extract it.
            parse_outcome = self._parse_with_protocol_chain(
                parser_input=parser_input,
                step=step,
                record=record,
            )
            if parse_outcome is not None:
                if parse_outcome.mode != "wait":
                    return parse_outcome

                # Some protocol parsers use an unmarked wait as an early-step
                # heuristic for plain text. Preserve the historical final
                # fallback unless parsing actually failed on action-shaped text.
                parser_error = bool(parse_outcome.meta.get("parser_error"))
                if not parser_error and self._looks_like_explicit_wait(parser_input):
                    return parse_outcome
                rejection_reason: str | None = None
                if parser_error:
                    if self._looks_like_structured_action_intent(parser_input):
                        rejection_reason = "structured_action_parse_error"
                    elif self._looks_like_structured_final_intent(parser_input):
                        rejection_reason = "structured_final_parse_error"
                if rejection_reason is not None:
                    self.engine._emit(
                        step,
                        RuntimePhase.DECIDE,
                        payload={
                            "stage": "native_text_final_rejected",
                            "reason": rejection_reason,
                            "parser_diagnostics": parse_outcome.meta.get(
                                "parser_diagnostics"
                            ),
                        },
                    )
                    return parse_outcome

            if record is not None:
                record.decision_source = "native_text_final"
            return Decision.final(
                answer=str(response.text).strip(),
                meta={"decision_source": "native_text_final"},
            )

        parse_outcome = self._parse_with_protocol_chain(
            parser_input=parser_input,
            step=step,
            record=record,
        )
        if parse_outcome is not None:
            return parse_outcome

        raise ValueError(
            "Agent.decide must return Decision when no parser is configured"
        )

    @staticmethod
    def _looks_like_explicit_wait(text: Any) -> bool:
        source = str(text or "").strip()
        if not source:
            return False
        return bool(
            re.search(
                r"(?i)(?:[\"']mode[\"']|(?:^|[\{,])\s*mode)\s*[:=]\s*[\"']?wait[\"']?",
                source,
            )
            or re.search(r"(?i)<[^>]+\bmode\s*=\s*[\"']wait[\"']", source)
        )

    @staticmethod
    def _looks_like_structured_action_intent(text: Any) -> bool:
        source = str(text or "").strip()
        if not source:
            return False

        if re.search(
            r"(?i)(?:"
            r"<\s*(?:minimax:)?tool_call\b|"
            r"<\s*(?:tool_use|tool_name|invoke)\b|"
            r"<\|tool_calls?_section_begin\|>|"
            r"<\|tool_call_(?:begin|argument_begin)\|>"
            r")",
            source,
        ):
            return True

        if re.search(
            r"(?im)^\s*(?:[-*]\s*)?action(?:s)?(?:\s*:|\s+[A-Za-z_][\w.-]*\s*\()",
            source,
        ):
            return True

        field_pattern = (
            r"actions?|tools?|tool[_-]?calls?|calls?|commands?|name|args|arguments"
        )
        json_carrier_pattern = r"actions?|tools?|tool[_-]?calls?|calls?|commands?"
        if re.search(
            rf"(?i)[\"']({json_carrier_pattern})[\"']\s*:", source
        ) or re.search(rf"(?i)[{{,]\s*({json_carrier_pattern})\s*:", source):
            return True

        structured_fields: set[str] = set()
        for pattern in (
            rf"(?im)^\s*(?:[-*]\s*)?[\"']?({field_pattern})[\"']?\s*(?::|=)",
            rf"(?i)[\"']({field_pattern})[\"']\s*:",
            rf"(?i)(?:^|[{{,])\s*({field_pattern})\s*:",
            rf"(?i)<\s*({field_pattern})\b",
        ):
            structured_fields.update(re.findall(pattern, source))

        normalized_fields = {
            re.sub(r"[_-]+", "", field).lower() for field in structured_fields
        }
        unambiguous_carriers = {
            "action",
            "actions",
            "toolcall",
            "toolcalls",
        }
        if normalized_fields & unambiguous_carriers:
            return True

        argument_fields = {"args", "arguments"}
        if "name" in normalized_fields and normalized_fields & argument_fields:
            return True

        ambiguous_carriers = {
            "tool",
            "tools",
            "call",
            "calls",
            "command",
            "commands",
        }
        return bool(
            normalized_fields & ambiguous_carriers
            and normalized_fields & ({"name"} | argument_fields)
        )

    @staticmethod
    def _looks_like_structured_final_intent(text: Any) -> bool:
        source = str(text or "").strip()
        if not source:
            return False
        return bool(
            re.search(
                r"(?i)(?:[\"']mode[\"']|(?:^|[\{,])\s*mode)" r"\s*[:=]\s*[\"']?final\b",
                source,
            )
            or re.search(
                r"(?i)(?:[\"']final[_-]?answer[\"']\s*:|"
                r"(?:^|[\{,])\s*final[_-]?answer\s*:)",
                source,
            )
            or re.search(r"(?i)<\s*(?:final_answer|final)\b", source)
            or re.search(r"(?im)^\s*final\s+answer\s*:", source)
        )

    def _parse_with_protocol_chain(
        self,
        *,
        parser_input: Any,
        step: int,
        record: StepRecord | None,
    ) -> Decision[ActionT] | None:
        parser_attempts: List[Dict[str, Any]] = []
        last_exception: Exception | None = None
        last_diagnostics: Dict[str, Any] | None = None
        candidates = self._candidate_parsers()
        for candidate in candidates:
            parser = candidate["parser"]
            protocol = candidate.get("protocol")
            fallback_used = bool(candidate.get("fallback_used"))
            try:
                decision = parser.parse(
                    parser_input,
                    context={"step": step, "protocol": getattr(protocol, "id", None)},
                )
                normalized = normalize_parser_diagnostics(
                    getattr(decision, "meta", None),
                    parser=parser,
                    raw_output=parser_input,
                    step_id=step,
                )
                if normalized is not None:
                    normalized = dict(normalized)
                    normalized.setdefault("protocol", getattr(protocol, "id", None))
                    normalized.setdefault("selected_parser", parser_name(parser))
                    normalized.setdefault("fallback_used", fallback_used)
                    normalized.setdefault("parser_attempts", list(parser_attempts))
                parser_attempts.append(
                    {
                        "parser": parser_name(parser),
                        "contract": parser_contract(parser),
                        "protocol": getattr(protocol, "id", None),
                        "result": (
                            "success"
                            if normalized is None
                            or normalized.get("severity") != "error"
                            else "error"
                        ),
                        "fallback_used": fallback_used,
                    }
                )
                if (
                    normalized is not None
                    and normalized.get("severity") == "error"
                    and candidate.get("allow_fallback", True)
                ):
                    last_diagnostics = dict(normalized)
                    continue
                self._record_parser_observability(
                    step=step,
                    raw_output=parser_input,
                    decision=decision,
                    record=record,
                    parser=parser,
                    diagnostics=normalized,
                    protocol=protocol,
                    parser_attempts=parser_attempts,
                    fallback_used=fallback_used,
                )
                return decision
            except Exception as exc:
                last_exception = exc
                parser_attempts.append(
                    {
                        "parser": parser_name(parser),
                        "contract": parser_contract(parser),
                        "protocol": getattr(protocol, "id", None),
                        "result": "exception",
                        "fallback_used": fallback_used,
                    }
                )
                last_diagnostics = build_parser_diagnostics(
                    parser=parser,
                    severity="error",
                    code="unexpected_parser_exception",
                    summary="Parser raised an unexpected exception.",
                    raw_output=parser_input,
                    details=str(exc),
                    repair_instruction="The parser failed internally before producing structured repair feedback.",
                    expected_shape="See the configured parser contract for the expected output format.",
                    step_id=step,
                )
                last_diagnostics["protocol"] = getattr(protocol, "id", None)
                last_diagnostics["selected_parser"] = parser_name(parser)
                last_diagnostics["fallback_used"] = fallback_used
                last_diagnostics["parser_attempts"] = list(parser_attempts)
                continue
        if last_diagnostics is not None:
            selected_parser = (
                parser_name(candidates[-1]["parser"])
                if candidates
                else "unknown_parser"
            )
            last_diagnostics.setdefault("selected_parser", selected_parser)
            last_diagnostics.setdefault(
                "fallback_used",
                any(item.get("fallback_used") for item in parser_attempts),
            )
            last_diagnostics.setdefault("parser_attempts", parser_attempts)
            self._record_parser_observability(
                step=step,
                raw_output=parser_input,
                decision=None,
                record=record,
                parser=candidates[-1]["parser"] if candidates else "unknown_parser",
                diagnostics=last_diagnostics,
                protocol=candidates[-1].get("protocol") if candidates else None,
                parser_attempts=parser_attempts,
                fallback_used=any(
                    item.get("fallback_used") for item in parser_attempts
                ),
            )
            if last_exception is not None:
                info = RuntimeErrorInfo(
                    category=ErrorCategory.PARSE,
                    message=str(last_exception),
                    phase="decide",
                    step_id=step,
                    recoverable=True,
                    details={"parser_diagnostics": last_diagnostics},
                )
                raise ParseExecutionError(info) from last_exception
            return Decision.wait(
                rationale=str(last_diagnostics.get("summary") or "Parser error."),
                meta={
                    "parser_error": True,
                    "parser_feedback": str(
                        last_diagnostics.get("repair_instruction")
                        or last_diagnostics.get("summary")
                        or ""
                    ),
                    "parser_diagnostics": last_diagnostics,
                },
            )
        return None

    def _candidate_parsers(self) -> List[Dict[str, Any]]:
        engine = self.engine
        if engine.parser is not None:
            return [
                {
                    "parser": engine.parser,
                    "protocol": get_protocol(engine.protocol),
                    "fallback_used": False,
                    "allow_fallback": False,
                }
            ]
        protocol = engine.resolve_protocol()
        candidates: List[Dict[str, Any]] = []
        agent_parser = getattr(engine.agent, "model_parser", None)
        if agent_parser is not None:
            candidates.append(
                {
                    "parser": agent_parser,
                    "protocol": protocol,
                    "fallback_used": False,
                    "allow_fallback": True,
                }
            )
        for index, item in enumerate(resolve_protocol_chain(protocol)):
            try:
                parser = item.parser_factory()
            except Exception:
                continue
            if agent_parser is not None and parser.__class__ is agent_parser.__class__:
                continue
            candidates.append(
                {
                    "parser": parser,
                    "protocol": item,
                    "fallback_used": bool(agent_parser) or index > 0,
                    "allow_fallback": True,
                }
            )
        return candidates

    def _interpret_model_response(
        self,
        *,
        state: StateT,
        observation: ObservationT,
        response: ModelResponse,
        record: StepRecord,
    ) -> Decision[ActionT] | None:
        interpret = getattr(self.engine.agent, "interpret_model_response", None)
        if not callable(interpret):
            return None
        decision = interpret(state, observation, response)
        if decision is None:
            return None
        if not isinstance(decision, Decision):
            raise ValueError(
                "Agent.interpret_model_response must return Decision or None"
            )
        self.engine._emit(
            record.step_id,
            RuntimePhase.DECIDE,
            payload={
                "stage": "model_response_interpreted",
                "mode": decision.mode,
                "model_response": dict(record.model_response),
            },
        )
        record.decision_source = "agent_interpretation"
        return decision

    def _record_parser_observability(
        self,
        *,
        step: int,
        raw_output: Any,
        decision: Decision[ActionT] | None,
        record: StepRecord | None,
        parser: Any,
        diagnostics: Dict[str, Any] | None = None,
        protocol: Any = None,
        parser_attempts: List[Dict[str, Any]] | None = None,
        fallback_used: bool = False,
    ) -> None:
        engine = self.engine
        contract = parser_contract(parser)
        normalized = diagnostics or normalize_parser_diagnostics(
            getattr(decision, "meta", None),
            parser=parser,
            raw_output=raw_output,
            step_id=step,
        )
        protocol_id = getattr(protocol, "id", None) if protocol is not None else None
        attempts = list(parser_attempts or [])
        if normalized is not None:
            normalized.setdefault("protocol", protocol_id)
            normalized.setdefault("selected_parser", parser_name(parser))
            normalized.setdefault("fallback_used", bool(fallback_used))
            normalized.setdefault("parser_attempts", attempts)
        if (
            decision is not None
            and isinstance(decision.meta, dict)
            and normalized is not None
        ):
            decision.meta["parser_diagnostics"] = normalized
            if normalized.get("severity") == "error":
                decision.meta.setdefault("parser_error", True)
                decision.meta.setdefault(
                    "parser_feedback",
                    normalized.get("repair_instruction")
                    or normalized.get("summary")
                    or "",
                )
            else:
                decision.meta.setdefault(
                    "parser_warning",
                    normalized.get("salvage_summary")
                    or normalized.get("summary")
                    or "",
                )
        parsed_mode = getattr(decision, "mode", None) if decision is not None else None
        result_payload = {
            "stage": "parser_result",
            "parser": parser_name(parser),
            "contract": contract,
            "protocol": protocol_id,
            "selected_parser": parser_name(parser),
            "parsed_mode": parsed_mode,
            "has_diagnostics": normalized is not None,
            "salvage_applied": bool((normalized or {}).get("salvage_applied")),
            "severity": (normalized or {}).get("severity"),
            "fallback_used": bool(fallback_used),
            "parser_attempts": attempts,
        }
        engine._emit(step, RuntimePhase.DECIDE, payload=result_payload)
        if normalized is not None:
            engine._emit(
                step,
                RuntimePhase.DECIDE,
                payload={"stage": "parser_diagnostics", "diagnostics": normalized},
            )
            engine._trace_runtime.record_parser_diagnostics(normalized)
        if record is not None:
            record.protocol_id = protocol_id
            record.parser_selected = parser_name(parser)
            record.parser_fallback_used = bool(fallback_used)
            record.parser_attempts = attempts
            record.parser_contract = contract
            record.parser_diagnostics = dict(normalized or {})
            record.parser_salvage_applied = bool(
                (normalized or {}).get("salvage_applied")
            )
            record.decision_source = "parser"

    def _normalize_model_response(self, response: ModelResponse) -> ModelResponse:
        """Apply Engine-only text tool-call salvage to a completed response."""

        if not isinstance(response, ModelResponse):
            raise TypeError("model response must be a ModelResponse")
        llm = getattr(self.engine.agent, "llm", None)
        model_name = response.model_name or getattr(llm, "model", None)
        provider = response.provider or getattr(llm, "provider_name", None)
        metadata = dict(response.metadata or {})
        text = str(response.text or "")
        tool_calls = (
            [dict(item) for item in (response.tool_calls or [])]
            if isinstance(response.tool_calls, list)
            else None
        )
        if not tool_calls:
            markup_tool_calls = self._extract_text_tool_call_markup(text)
            if markup_tool_calls:
                tool_calls = markup_tool_calls
                metadata["tool_call_markup_salvaged"] = True
                metadata["tool_call_markup_format"] = "glm_text_tool_call"
                if self._contains_only_text_tool_call_markup(text):
                    text = ""
        return ModelResponse(
            text=text,
            usage=response.usage,
            finish_reason=response.finish_reason,
            tool_calls=tool_calls,
            model_name=str(model_name) if model_name is not None else None,
            provider=str(provider) if provider is not None else None,
            metadata=metadata,
            reasoning_content=response.reasoning_content,
            native_items=response.native_items,
            timing=response.timing,
        )

    def _extract_text_tool_call_markup(self, text: str) -> List[Dict[str, Any]] | None:
        """Salvage GLM-style textual tool-call markup into native tool calls."""
        if "<tool_call>" not in text:
            return None
        calls: List[Dict[str, Any]] = []
        for index, match in enumerate(
            re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL),
            start=1,
        ):
            body = match.group(1)
            first_arg = re.search(r"<arg_key>", body)
            name_part = body[: first_arg.start()] if first_arg else body
            name = html.unescape(re.sub(r"<[^>]+>", "", name_part)).strip()
            if not name:
                continue
            args: Dict[str, Any] = {}
            for key, value in re.findall(
                r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>",
                body,
                re.DOTALL,
            ):
                clean_key = html.unescape(re.sub(r"<[^>]+>", "", key)).strip()
                if not clean_key:
                    continue
                args[clean_key] = self._coerce_text_tool_call_arg(value)
            calls.append(
                {
                    "id": f"call_glm_text_{index}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
            )
        return calls or None

    def _coerce_text_tool_call_arg(self, value: str) -> Any:
        text = html.unescape(str(value or "")).strip()
        try:
            return json.loads(text)
        except Exception:
            return text

    def _contains_only_text_tool_call_markup(self, text: str) -> bool:
        stripped = str(text or "").strip()
        if not stripped:
            return False
        remainder = re.sub(
            r"<tool_call>\s*.*?\s*</tool_call>",
            "",
            stripped,
            flags=re.DOTALL,
        ).strip()
        return not remainder

    def _decision_from_native_tool_calls(
        self,
        *,
        response: ModelResponse | None,
        step: int,
        record: StepRecord | None,
    ) -> Decision[ActionT] | None:
        if (
            response is None
            or not isinstance(response.tool_calls, list)
            or not response.tool_calls
        ):
            return None
        if not self._native_tool_call_preferred():
            if record is not None and not record.decision_source:
                record.decision_source = "parser"
            return None
        actions: List[Action] = []
        for item in response.tool_calls:
            actions.append(self._action_from_tool_call(item))
        decision: Decision[ActionT] = cast(
            Decision[ActionT],
            Decision.act(
                actions=actions,
                rationale=(response.text or "").strip() or None,
                meta={
                    "decision_source": "native_tool_calls",
                    "native_tool_call_count": len(actions),
                    "tool_calls": [dict(item) for item in response.tool_calls],
                },
            ),
        )
        self.engine._emit(
            step,
            RuntimePhase.DECIDE,
            payload={
                "stage": "native_tool_calls_decision",
                "tool_call_count": len(actions),
                "tool_calls": [dict(item) for item in response.tool_calls],
            },
        )
        if record is not None:
            record.decision_source = "native_tool_calls"
            record.native_tool_call_used = True
            record.native_tool_call_fallback_reason = None
        return decision

    def _native_tool_call_preferred(self) -> bool:
        llm = getattr(self.engine.agent, "llm", None)
        return native_tool_calls_preferred(
            llm=llm,
            protocol=self.engine.resolve_protocol(),
        )

    def _trim_native_tool_history(
        self, history: List[Dict[str, Any]], *, max_rounds: int
    ) -> List[Dict[str, Any]]:
        if max_rounds <= 0 or not history:
            return history
        wrapped: List[HistoryMessage] = []
        for index, message in enumerate(history):
            step_marker = message.get("_step_id")
            step_id = int(step_marker) if isinstance(step_marker, int) else 0
            wrapped.append(
                HistoryMessage(
                    role=str(message.get("role") or "user"),
                    content=message.get("content"),
                    step_id=step_id,
                    reasoning_content=message.get("reasoning_content"),
                    tool_calls=[
                        dict(call)
                        for call in list(message.get("tool_calls") or [])
                        if isinstance(call, dict)
                    ],
                    tool_call_id=(
                        str(message["tool_call_id"])
                        if message.get("tool_call_id") not in (None, "")
                        else None
                    ),
                    name=(
                        str(message["name"])
                        if message.get("name") not in (None, "")
                        else None
                    ),
                    metadata={"history_index": index},
                    native_items=[
                        dict(item)
                        for item in list(message.get("native_items") or [])
                        if isinstance(item, dict)
                    ],
                )
            )
        groups = group_history_rounds(wrapped)
        selected = groups[-int(max_rounds) :]
        keep_indices = {
            int(message.metadata["history_index"])
            for group in selected
            for message in group
        }
        return [
            message for index, message in enumerate(history) if index in keep_indices
        ]

    def _tool_transaction_parity(
        self,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        offered = [
            call_id
            for message in messages
            if message.get("role") == "assistant"
            for call_id in message_tool_call_ids(message)
        ]
        results = [
            result_id
            for message in messages
            for result_id in message_tool_result_ids(message)
        ]
        offered_counts = Counter(offered)
        result_counts = Counter(results)
        missing = sorted((offered_counts - result_counts).elements())
        orphaned = sorted((result_counts - offered_counts).elements())
        return {
            "offered_call_count": len(offered),
            "result_count": len(results),
            "missing_result_ids": missing,
            "orphan_result_ids": orphaned,
            "valid": not missing and not orphaned,
        }

    def _merge_runtime_context_into_last_tool(
        self, messages: List[Dict[str, Any]], runtime_context: str
    ) -> tuple[bool, str | None]:
        """Attach transient controller state to the final real tool result.

        The provider-visible tool chain remains valid because the message keeps
        the tool call id produced by the model.  A synthetic hidden tool would
        instead require fabricating an assistant call/result pair and degrade
        trace accuracy.
        """
        context = str(runtime_context or "").strip()
        if not context:
            return False, None
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if not isinstance(message, dict) or message.get("role") != "tool":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            updated = dict(message)
            if _is_decision_context_packet(context):
                updated["content"] = f"{content.rstrip()}\n\n{context}"
            else:
                updated["content"] = (
                    f"{content.rstrip()}\n\n[{_RUNTIME_CONTEXT_IN_TOOL_NOTICE}]\n{context}"
                )
            messages[index] = updated
            tool_call_id = updated.get("tool_call_id")
            return True, str(tool_call_id) if tool_call_id is not None else None
        return False, None

    def _normalize_decision_context_packet(
        self,
        *,
        messages: List[Dict[str, Any]],
        authoritative_source: str,
        delivery: Dict[str, Any],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Ensure the actual provider packet has one current Decision Context.

        Decision Context is reconstructed from controller state every turn and
        is never durable conversation history.  Old traces, a partial merge,
        or a retry must therefore not be allowed to turn a duplicate transient
        block into a terminal agent failure.
        """
        source_blocks = _DECISION_CONTEXT_PATTERN.findall(
            str(authoritative_source or "")
        )
        if len(source_blocks) != 1:
            return messages, {
                "rebuild_required": True,
                "reason": "authoritative_invalid",
                "before_count": len(self._decision_context_blocks(messages)),
                "after_count": len(self._decision_context_blocks(messages)),
                "authoritative_context": "",
            }
        authoritative = source_blocks[0]
        before_blocks = self._decision_context_blocks(messages)
        valid = len(before_blocks) == 1 and before_blocks[0] == authoritative
        if valid:
            return messages, {
                "rebuild_required": False,
                "reason": "",
                "before_count": 1,
                "after_count": 1,
                "authoritative_context": authoritative,
            }

        if not before_blocks:
            reason = "missing"
        elif len(before_blocks) > 1:
            reason = "duplicate"
        else:
            reason = "mismatch"
        rebuilt: List[Dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            updated = dict(message)
            # System prompt content is authored stable text.  Only transient
            # non-system delivery is eligible for replacement.
            if updated.get("role") != "system":
                updated["content"] = _strip_decision_context_content(
                    updated.get("content")
                )
            rebuilt.append(updated)

        requested = str(delivery.get("requested") or "").strip().lower()
        merged = False
        if requested == "merge_tool":
            merged, tool_call_id = self._merge_runtime_context_into_last_tool(
                rebuilt, authoritative
            )
            if merged:
                delivery.update(
                    {
                        "effective": "merge_tool",
                        "target_tool_call_id": tool_call_id,
                        "fallback_reason": None,
                    }
                )
        if not merged:
            rebuilt.append({"role": "user", "content": authoritative})
            delivery.update(
                {
                    "effective": "user",
                    "target_tool_call_id": None,
                    "fallback_reason": (
                        "no_text_tool_result" if requested == "merge_tool" else None
                    ),
                }
            )
        after_count = len(self._decision_context_blocks(rebuilt))
        return rebuilt, {
            "rebuild_required": True,
            "reason": reason,
            "before_count": len(before_blocks),
            "after_count": after_count,
            "authoritative_context": authoritative,
        }

    def _ensure_chain_consistency(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Project a strict, immutable assistant/tool transaction chain.

        Every assistant call is followed immediately by exactly one result in
        call order. Existing results are moved next to their declaring call,
        duplicate and orphan results are dropped, and an interrupted call gets
        one deterministic terminal placeholder. The canonical history remains
        unchanged.
        """
        if not messages:
            return messages

        expected_tool_ids: List[str] = []
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            expected_tool_ids.extend(message_tool_call_ids(msg))

        if not expected_tool_ids:
            return [msg for msg in messages if not message_tool_result_ids(msg)]

        expected_set = set(expected_tool_ids)
        results_by_id: Dict[str, List[int]] = {}
        result_ids_by_index: Dict[int, List[str]] = {}
        for index, msg in enumerate(messages):
            result_ids = [
                result_id
                for result_id in message_tool_result_ids(msg)
                if result_id in expected_set
            ]
            if not result_ids:
                continue
            result_ids_by_index[index] = result_ids
            for result_id in result_ids:
                results_by_id.setdefault(result_id, []).append(index)

        projected: List[Dict[str, Any]] = []
        used_result_indices: set[int] = set()
        satisfied_result_counts: Counter[str] = Counter()
        for index, msg in enumerate(messages):
            if index in result_ids_by_index:
                continue
            projected.append(msg)
            if msg.get("role") != "assistant":
                continue
            for normalized_id in message_tool_call_ids(msg):
                if satisfied_result_counts[normalized_id] > 0:
                    satisfied_result_counts[normalized_id] -= 1
                    continue
                available = [
                    result_index
                    for result_index in results_by_id.get(normalized_id, [])
                    if result_index not in used_result_indices
                ]
                if available:
                    result_index = available[0]
                    used_result_indices.add(result_index)
                    satisfied_result_counts.update(result_ids_by_index[result_index])
                    satisfied_result_counts[normalized_id] -= 1
                    projected.append(messages[result_index])
                    continue
                projected.append(
                    {
                        "role": "tool",
                        "tool_call_id": normalized_id,
                        "content": json.dumps(
                            {
                                "status": "error",
                                "code": "tool_call_not_completed",
                                "reason": (
                                    "The tool call did not produce a result in this "
                                    "transaction."
                                ),
                                "next_action": (
                                    "Retry the call if it is still relevant."
                                ),
                            },
                            sort_keys=True,
                        ),
                    }
                )
        return projected

    def _strip_internal_message_keys(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        cleaned: List[Dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            payload = {
                key: value
                for key, value in message.items()
                if not str(key).startswith("_")
            }
            cleaned.append(payload)
        return cleaned

    def _action_from_tool_call(self, tool_call: Dict[str, Any]) -> Action:
        function = tool_call.get("function")
        function_payload = function if isinstance(function, dict) else {}
        name = str(function_payload.get("name") or "").strip()
        arguments = function_payload.get("arguments")
        args: Dict[str, Any] = {}
        repaired_arguments = False
        protocol_error: str | None = None
        if not isinstance(function, dict) or not name:
            protocol_error = "tool_call_invalid"
        elif isinstance(arguments, dict):
            args = dict(arguments)
        elif isinstance(arguments, str):
            text = arguments.strip()
            if text:
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    repaired = escape_json_string_control_chars(text)
                    if repaired is None:
                        protocol_error = "tool_call_arguments_invalid"
                    else:
                        try:
                            parsed = json.loads(repaired)
                        except json.JSONDecodeError:
                            protocol_error = "tool_call_arguments_invalid"
                        else:
                            repaired_arguments = True
                if protocol_error is None:
                    if not isinstance(parsed, dict):
                        protocol_error = "tool_call_arguments_invalid"
                    else:
                        args = dict(parsed)
        elif arguments is not None:
            protocol_error = "tool_call_arguments_invalid"
        metadata = {
            "tool_call_type": tool_call.get("type"),
            "decision_source": "native_tool_calls",
        }
        if repaired_arguments:
            metadata["arguments_repair"] = "escaped_control_chars"
        if protocol_error is not None:
            metadata.update(
                {
                    "protocol_error": protocol_error,
                    "raw_arguments": arguments,
                }
            )
        return Action(
            name=name or "<invalid-native-tool-call>",
            args=args,
            action_id=(
                str(tool_call.get("id")) if tool_call.get("id") is not None else None
            ),
            metadata=metadata,
        )
