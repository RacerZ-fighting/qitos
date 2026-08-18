"""Pi-style context compaction over one journaled Session transcript.

This is the QitOS port of Pi v3's proven compaction algorithm
(``pi:packages/coding-agent/src/core/compaction/compaction.ts`` and
``pi:packages/ai/src/utils/overflow.ts``): chars/4 token estimation, a
keep-recent cut search that never lands on a Tool result and never splits a
Tool call from its result, split-turn prefix merge, and a single structured
summarization request (with previous-summary iteration) issued by the run's
own model. Overflow detection is a conservative port of Pi's provider error
patterns with Pi's non-overflow exclusions.

The module is pure mechanics: it owns no journal, Agent or Session state.
The Session Harness drives it and persists the durable ``compaction``
record itself; historical records are never rewritten or deleted.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Union

from ...core.agent_loop import model_protocol_identity
from ...core.message import (
    AssistantMessage,
    ContextMessage,
    ImageContent,
    Message,
    TextContent,
    ToolResultMessage,
    ToolCall,
    UserMessage,
)
from ...core.model_request import ModelRequest
from ...core.model_response import ModelUsage
from ...core.model_stream import ModelStreamEventType

if TYPE_CHECKING:
    from ...models.base import Model

_ESTIMATED_IMAGE_CHARS = 4800
_TOOL_RESULT_SUMMARY_MAX_CHARS = 2000


# ── settings and result types ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CompactionSettings:
    """Auto-compaction policy; values mirror Pi's proven defaults."""

    enabled: bool = True
    reserve_tokens: int = 16384
    keep_recent_tokens: int = 20000

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean")
        for name in ("reserve_tokens", "keep_recent_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class CompactResult:
    """One applied compaction: durable summary plus its cut reference."""

    summary: str
    first_kept_transcript_id: str
    tokens_before: int
    usage: ModelUsage | None


@dataclass(frozen=True, slots=True)
class CompactRejected:
    """Typed expected rejection for manual compaction."""

    reason: Literal["busy", "nothing_to_compact"]


# ── token estimation ─────────────────────────────────────────────────────────


def estimate_tokens(message: Message) -> int:
    """Estimate one message's tokens with Pi's conservative chars/4 rule."""

    chars = 0
    if isinstance(message, UserMessage):
        chars = _content_chars(message.content)
    elif isinstance(message, ContextMessage):
        chars = len(message.content)
    elif isinstance(message, AssistantMessage):
        chars = len(message.text)
        if message.reasoning_content:
            chars += len(message.reasoning_content)
        for call in message.tool_calls:
            chars += len(call.name) + len(_call_arguments_text(call))
    elif isinstance(message, ToolResultMessage):
        chars = len(_tool_result_text(message))
    else:
        raise TypeError(f"unsupported message type: {type(message).__name__}")
    return -(-chars // 4)


def usage_context_tokens(usage: ModelUsage) -> int:
    """Total context tokens of one assistant usage (Pi's calculation)."""

    if usage.total_tokens:
        return usage.total_tokens
    return (
        (usage.input_tokens or 0)
        + (usage.output_tokens or 0)
        + (usage.cache_read_tokens or 0)
        + (usage.cache_write_tokens or 0)
    )


def estimate_context_tokens(messages: Sequence[Message]) -> int:
    """Estimate current context size from the last valid assistant usage.

    Failed assistants never carry trustworthy usage; the estimate chains
    from the latest non-failed assistant with a positive usage total and
    adds chars/4 estimates for the messages after it. Without any usage the
    whole context is estimated.
    """

    last_usage_index: int | None = None
    last_tokens = 0
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, AssistantMessage) or message.failed:
            continue
        if message.usage is None:
            continue
        tokens = usage_context_tokens(message.usage)
        if tokens <= 0:
            continue
        last_usage_index = index
        last_tokens = tokens
        break
    if last_usage_index is None:
        return sum(estimate_tokens(message) for message in messages)
    trailing = sum(
        estimate_tokens(message) for message in messages[last_usage_index + 1 :]
    )
    return last_tokens + trailing


def should_compact(
    context_tokens: int, context_window: int, settings: CompactionSettings
) -> bool:
    """Pi's threshold rule: compact before the reserve budget is reached."""

    if not settings.enabled:
        return False
    return context_tokens > context_window - settings.reserve_tokens


def _content_chars(content: object) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, tuple):
        chars = 0
        for block in content:
            if isinstance(block, TextContent):
                chars += len(block.text)
            elif isinstance(block, ImageContent):
                chars += _ESTIMATED_IMAGE_CHARS
        return chars
    return 0


def _call_arguments_text(call: ToolCall) -> str:
    return json.dumps(dict(call.arguments), ensure_ascii=False, default=str)


def _tool_result_text(message: ToolResultMessage) -> str:
    visible = message.result.model_visible_output
    if isinstance(visible, str):
        return visible
    if visible is not None:
        return json.dumps(visible, ensure_ascii=False, default=str)
    return message.result.error or ""


# ── cut point detection ─────────────────────────────────────────────────────

#: One context entry: a durable transcript message plus its record id, or a
#: projected summary message (``None`` id) left by an earlier compaction.
ContextEntry = tuple[Union[str, None], Message]


@dataclass(frozen=True, slots=True)
class CompactionCut:
    """Where a compaction splits the context.

    ``first_kept_index`` is the first context entry that survives the
    compaction. ``turn_start_index`` names the user message opening the
    split turn (``-1`` when the cut lands on a turn boundary).
    """

    first_kept_index: int
    turn_start_index: int
    is_split_turn: bool


def _is_cut_point(entry: ContextEntry) -> bool:
    record_id, message = entry
    # Projected summaries are compaction boundaries, not cut points; a Tool
    # result is never a cut point because it must follow its Tool call.
    return record_id is not None and isinstance(
        message, (UserMessage, AssistantMessage)
    )


def find_cut_point(
    entries: Sequence[ContextEntry],
    start_index: int,
    keep_recent_tokens: int,
) -> CompactionCut:
    """Find the cut that keeps approximately ``keep_recent_tokens`` of tail.

    Walks backwards from the newest entry accumulating chars/4 estimates and
    cuts at the closest valid point once the budget is reached. A naive cut
    that would land on a Tool result moves forward to the next user or
    assistant entry, so a Tool call and its result never separate.
    """

    cut_points = [
        index
        for index in range(start_index, len(entries))
        if _is_cut_point(entries[index])
    ]
    if not cut_points:
        return CompactionCut(start_index, -1, False)

    accumulated = 0
    cut_index = cut_points[0]
    for index in range(len(entries) - 1, start_index - 1, -1):
        tokens = estimate_tokens(entries[index][1])
        if tokens == 0:
            continue
        accumulated += tokens
        if accumulated >= keep_recent_tokens:
            cut_index = next(
                candidate for candidate in cut_points if candidate >= index
            )
            break

    cut_entry = entries[cut_index]
    if isinstance(cut_entry[1], UserMessage):
        return CompactionCut(cut_index, -1, False)
    turn_start = -1
    for index in range(cut_index, start_index - 1, -1):
        record_id, message = entries[index]
        if record_id is not None and isinstance(message, UserMessage):
            turn_start = index
            break
    return CompactionCut(cut_index, turn_start, turn_start != -1)


@dataclass(frozen=True, slots=True)
class CompactionPreparation:
    """One planned compaction over the current context entries."""

    first_kept_transcript_id: str
    history: tuple[Message, ...]
    turn_prefix: tuple[Message, ...]
    is_split_turn: bool
    tokens_before: int
    previous_summary: str | None


def prepare_compaction(
    entries: Sequence[ContextEntry],
    settings: CompactionSettings,
) -> CompactionPreparation | None:
    """Plan one compaction; ``None`` when there is nothing to summarize.

    Entries before the first durable transcript entry are projected
    summaries of earlier compactions: the newest one iterates into the
    summarization prompt as ``previous_summary`` and every earlier one is
    already folded into it, so none of them is summarized again.
    """

    boundary = 0
    while boundary < len(entries) and entries[boundary][0] is None:
        boundary += 1
    if boundary >= len(entries):
        return None
    tokens_before = estimate_context_tokens(
        [message for _record_id, message in entries]
    )
    cut = find_cut_point(entries, boundary, settings.keep_recent_tokens)
    first_kept_id = entries[cut.first_kept_index][0]
    if first_kept_id is None:
        return None

    history_end = (
        cut.turn_start_index if cut.is_split_turn else cut.first_kept_index
    )
    history = tuple(
        message
        for record_id, message in entries[boundary:history_end]
        if record_id is not None
    )
    turn_prefix: tuple[Message, ...] = ()
    if cut.is_split_turn:
        turn_prefix = tuple(
            message
            for record_id, message in entries[
                cut.turn_start_index : cut.first_kept_index
            ]
            if record_id is not None
        )
    if not history and not turn_prefix:
        return None
    previous_summary: str | None = None
    if boundary > 0:
        _record_id, latest_summary = entries[boundary - 1]
        if isinstance(latest_summary, UserMessage) and isinstance(
            latest_summary.content, str
        ):
            previous_summary = latest_summary.content
    return CompactionPreparation(
        first_kept_transcript_id=first_kept_id,
        history=history,
        turn_prefix=turn_prefix,
        is_split_turn=cut.is_split_turn,
        tokens_before=tokens_before,
        previous_summary=previous_summary,
    )


# ── summarization ────────────────────────────────────────────────────────────

_SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a "
    "conversation between a user and an AI assistant, then produce a "
    "structured summary following the exact format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in "
    "the conversation. ONLY output the structured summary."
)

_SUMMARIZATION_PROMPT = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

_UPDATE_SUMMARIZATION_PROMPT = """The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

_TURN_PREFIX_SUMMARIZATION_PROMPT = """This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix."""


def serialize_conversation(messages: Sequence[Message]) -> str:
    """Serialize messages to text so the model summarizes, not continues."""

    parts: list[str] = []
    for message in messages:
        if isinstance(message, UserMessage):
            content = _user_text(message)
            if content:
                parts.append(f"[User]: {content}")
        elif isinstance(message, ContextMessage):
            parts.append(f"[Runtime context]: {message.content}")
        elif isinstance(message, AssistantMessage):
            if message.reasoning_content:
                parts.append(f"[Assistant thinking]: {message.reasoning_content}")
            if message.text.strip():
                parts.append(f"[Assistant]: {message.text}")
            if message.tool_calls:
                calls = "; ".join(
                    f"{call.name}({_call_arguments_text(call)})"
                    for call in message.tool_calls
                )
                parts.append(f"[Assistant tool calls]: {calls}")
            if message.error:
                parts.append(f"[Assistant error]: {message.error}")
        elif isinstance(message, ToolResultMessage):
            content = _tool_result_text(message)
            if content:
                parts.append(
                    f"[Tool result]: {_truncate_for_summary(content)}"
                )
        else:
            raise TypeError(
                f"unsupported message type: {type(message).__name__}"
            )
    return "\n\n".join(parts)


def _user_text(message: UserMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "\n".join(
        block.text for block in message.content if isinstance(block, TextContent)
    )


def _truncate_for_summary(text: str) -> str:
    if len(text) <= _TOOL_RESULT_SUMMARY_MAX_CHARS:
        return text
    return text[: _TOOL_RESULT_SUMMARY_MAX_CHARS] + "..."


class SummarizationError(RuntimeError):
    """The summarization request itself failed (compaction did not apply)."""


async def compact_context(
    model: Model,
    preparation: CompactionPreparation,
    settings: CompactionSettings,
) -> CompactResult:
    """Summarize one prepared compaction with the run's own model.

    One structured request per part (history, plus the split-turn prefix
    when the cut divides a turn); no Tool exposure; the output budget is
    bounded by the reserve budget and the model's own ``max_tokens``.
    """

    max_tokens = _bounded_summary_tokens(model, settings.reserve_tokens, 0.8)
    if preparation.is_split_turn and preparation.turn_prefix:
        history_text = "No prior history."
        history_usage: ModelUsage | None = None
        if preparation.history:
            history_text, history_usage = await _summarize(
                model,
                preparation.history,
                prompt=_update_prompt(preparation.previous_summary),
                previous_summary=preparation.previous_summary,
                max_tokens=max_tokens,
            )
        prefix_text, prefix_usage = await _summarize(
            model,
            preparation.turn_prefix,
            prompt=_TURN_PREFIX_SUMMARIZATION_PROMPT,
            previous_summary=None,
            max_tokens=_bounded_summary_tokens(
                model, settings.reserve_tokens, 0.5
            ),
        )
        summary = (
            f"{history_text}\n\n---\n\n**Turn Context (split turn):**\n\n"
            f"{prefix_text}"
        )
        usage = _combine_usage(history_usage, prefix_usage)
    else:
        summary, usage = await _summarize(
            model,
            preparation.history,
            prompt=_update_prompt(preparation.previous_summary),
            previous_summary=preparation.previous_summary,
            max_tokens=max_tokens,
        )
    return CompactResult(
        summary=summary,
        first_kept_transcript_id=preparation.first_kept_transcript_id,
        tokens_before=preparation.tokens_before,
        usage=usage,
    )


def _update_prompt(previous_summary: str | None) -> str:
    return (
        _UPDATE_SUMMARIZATION_PROMPT
        if previous_summary
        else _SUMMARIZATION_PROMPT
    )


def _bounded_summary_tokens(model: Model, reserve_tokens: int, share: float) -> int:
    budget = int(reserve_tokens * share)
    model_cap = getattr(model, "max_tokens", 0)
    if isinstance(model_cap, int) and model_cap > 0:
        return min(budget, model_cap)
    return budget


def _combine_usage(
    first: ModelUsage | None, second: ModelUsage | None
) -> ModelUsage | None:
    if first is None:
        return second
    if second is None:
        return first

    def _sum(left: int | None, right: int | None) -> int | None:
        if left is None and right is None:
            return None
        return (left or 0) + (right or 0)

    return ModelUsage(
        input_tokens=_sum(first.input_tokens, second.input_tokens),
        output_tokens=_sum(first.output_tokens, second.output_tokens),
        total_tokens=_sum(first.total_tokens, second.total_tokens),
        cache_read_tokens=_sum(first.cache_read_tokens, second.cache_read_tokens),
        cache_write_tokens=_sum(
            first.cache_write_tokens, second.cache_write_tokens
        ),
        reasoning_tokens=_sum(first.reasoning_tokens, second.reasoning_tokens),
    )


async def _summarize(
    model: Model,
    messages: Sequence[Message],
    *,
    prompt: str,
    previous_summary: str | None,
    max_tokens: int,
) -> tuple[str, ModelUsage | None]:
    conversation = serialize_conversation(messages)
    prompt_text = f"<conversation>\n{conversation}\n</conversation>\n\n"
    if previous_summary:
        prompt_text += (
            f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
        )
    prompt_text += prompt
    request = ModelRequest(
        run_id=f"compaction-{uuid.uuid4().hex}",
        transaction_id=f"compaction-{uuid.uuid4().hex}",
        provider=model.provider_name,
        model=model.model,
        protocol=model_protocol_identity(model),
        messages=(
            {"role": "system", "content": _SUMMARIZATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ),
        options={"max_tokens": max_tokens},
    )
    parts: list[str] = []
    usage: ModelUsage | None = None
    error: str | None = None
    stream = model.stream(request)
    try:
        async for event in stream:
            if event.type is ModelStreamEventType.TEXT_DELTA:
                parts.append(event.text)
            elif event.type is ModelStreamEventType.COMPLETED:
                usage = event.usage if isinstance(event.usage, ModelUsage) else None
            elif event.type is ModelStreamEventType.FAILED:
                error = event.error or "summarization request failed"
    finally:
        aclose = getattr(stream, "aclose", None)
        if callable(aclose):
            await aclose()
    if error is not None:
        raise SummarizationError(error)
    text = "".join(parts).strip()
    if not text:
        raise SummarizationError("summarization request returned no text")
    return text, usage


# ── context overflow detection ───────────────────────────────────────────────

# Conservative port of Pi's provider overflow patterns; every entry is an
# error message a provider actually returns when the input exceeds the
# model's context window.
_OVERFLOW_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"prompt is too long",
        r"request_too_large",
        r"input is too long for requested model",
        r"exceeds the context window",
        r"exceeds (?:the )?(?:model'?s )?maximum context length",
        r"input token count.*exceeds the maximum",
        r"maximum prompt length is \d+",
        r"reduce the length of the messages",
        r"maximum context length is \d+ tokens",
        r"exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?",
        r"input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)",
        r"exceeds the limit of \d+",
        r"exceeds the available context size",
        r"greater than the context length",
        r"context window exceeds limit",
        r"exceeded model token limit",
        r"too large for model with \d+ maximum context length",
        r"prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?",
        r"model_context_window_exceeded",
        r"prompt too long; exceeded (?:max )?context length",
        r"range of input length should be",
        r"context[_ ]length[_ ]exceeded",
        r"too many tokens",
        r"token limit exceeded",
    )
)

# Non-overflow errors (rate limiting, throttling) win over the overflow
# patterns above even when both match the same message.
_NON_OVERFLOW_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^(Throttling error|Service unavailable):",
        r"rate limit",
        r"too many requests",
    )
)


def is_context_overflow(
    message: AssistantMessage, context_window: int | None
) -> bool:
    """Whether one assistant message represents a context overflow.

    Three conservative cases (Pi parity): an explicit provider overflow
    error; a silent overflow where a successful response reports more input
    than the context window; and a length stop with zero output where the
    input provably filled the window.
    """

    if message.error:
        if any(pattern.search(message.error) for pattern in _NON_OVERFLOW_PATTERNS):
            return False
        if any(pattern.search(message.error) for pattern in _OVERFLOW_PATTERNS):
            return True
    if context_window is None or context_window <= 0 or message.usage is None:
        return False
    usage = message.usage
    input_tokens = (usage.input_tokens or 0) + (usage.cache_read_tokens or 0)
    if not message.failed and message.finish_reason == "stop":
        return input_tokens > context_window
    if (
        not message.failed
        and message.truncated
        and (usage.output_tokens or 0) == 0
    ):
        return input_tokens >= context_window * 0.99
    return False


__all__ = [
    "CompactRejected",
    "CompactResult",
    "CompactionCut",
    "CompactionPreparation",
    "CompactionSettings",
    "ContextEntry",
    "SummarizationError",
    "compact_context",
    "estimate_context_tokens",
    "estimate_tokens",
    "find_cut_point",
    "is_context_overflow",
    "prepare_compaction",
    "serialize_conversation",
    "should_compact",
    "usage_context_tokens",
]
