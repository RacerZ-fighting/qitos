"""Compaction mechanics: estimation, cut points, summarization, overflow."""

from __future__ import annotations

import pytest

from qitos.core.message import (
    AssistantMessage,
    ContextMessage,
    ImageContent,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from qitos.core.model_response import ModelUsage
from qitos.core.tool_result import ToolResult
from qitos.kit.session.compaction import (
    CompactionSettings,
    SummarizationError,
    compact_context,
    estimate_context_tokens,
    estimate_tokens,
    find_cut_point,
    is_context_overflow,
    prepare_compaction,
    should_compact,
    usage_context_tokens,
)

from tests.core.agent_fakes import ScriptedModel, failed_events, text_events


def _user(chars: int) -> UserMessage:
    return UserMessage(content="u" * chars)


def _assistant(chars: int, calls: tuple[ToolCall, ...] = ()) -> AssistantMessage:
    return AssistantMessage(text="a" * chars, tool_calls=calls)


def _tool_result(call_id: str, chars: int) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=call_id,
        tool_name="echo",
        result=ToolResult(status="success", output="r" * chars, call_id=call_id),
    )


# ── estimation ───────────────────────────────────────────────────────────────


def test_estimate_tokens_uses_chars_per_four_with_ceiling() -> None:
    assert estimate_tokens(_user(8)) == 2
    assert estimate_tokens(_user(9)) == 3
    # Assistant text and reasoning both count.
    message = AssistantMessage(text="a" * 8, reasoning_content="r" * 4)
    assert estimate_tokens(message) == 3
    assert estimate_tokens(ContextMessage(content="c" * 9)) == 3
    # Images follow Pi's fixed char estimate.
    image_message = UserMessage(
        content=(TextContent(text="ab"), ImageContent(source={"data": "x"}))
    )
    assert estimate_tokens(image_message) > estimate_tokens(
        UserMessage(content=(TextContent(text="ab"),))
    )


def test_estimate_context_tokens_chains_from_last_assistant_usage() -> None:
    usage = ModelUsage(input_tokens=900, output_tokens=100, total_tokens=1000)
    messages = [
        _user(100),
        AssistantMessage(text="a" * 100, usage=usage),
        _tool_result("c1", 400),
    ]
    # The assistant's own text is not re-estimated; only the trailing
    # messages after it are.
    assert estimate_context_tokens(messages) == 1000 + estimate_tokens(messages[2])


def test_estimate_context_tokens_falls_back_to_full_estimate() -> None:
    messages = [_user(40), _assistant(40)]
    assert estimate_context_tokens(messages) == 20
    failed = AssistantMessage(text="a" * 40, error="boom")
    assert estimate_context_tokens([_user(40), failed]) == 20


def test_usage_context_tokens_prefers_total_then_components() -> None:
    assert usage_context_tokens(ModelUsage(total_tokens=7, input_tokens=1)) == 7
    assert (
        usage_context_tokens(
            ModelUsage(input_tokens=2, output_tokens=3, cache_read_tokens=1)
        )
        == 6
    )


def test_should_compact_threshold_rule() -> None:
    settings = CompactionSettings(reserve_tokens=100)
    assert should_compact(901, 1000, settings) is True
    assert should_compact(900, 1000, settings) is False
    assert should_compact(
        901, 1000, CompactionSettings(enabled=False, reserve_tokens=100)
    ) is False


def test_settings_validation() -> None:
    with pytest.raises(ValueError):
        CompactionSettings(reserve_tokens=0)
    with pytest.raises(ValueError):
        CompactionSettings(keep_recent_tokens=-5)


# ── cut points ───────────────────────────────────────────────────────────────


def test_cut_never_lands_on_a_tool_result() -> None:
    # Every message is ~25 tokens; keep 50 -> the naive stop lands on the
    # tool result and must move forward to the next user message.
    entries = [
        ("id0", _user(100)),
        ("id1", _assistant(100)),
        ("id2", _tool_result("c1", 100)),
        ("id3", _user(100)),
        ("id4", _assistant(100)),
    ]
    cut = find_cut_point(entries, 0, keep_recent_tokens=50)
    first_kept_id = entries[cut.first_kept_index][0]
    assert first_kept_id == "id3"
    assert not isinstance(entries[cut.first_kept_index][1], ToolResultMessage)


def test_cut_at_assistant_keeps_call_and_result_together() -> None:
    calls = (ToolCall(id="c1", name="echo", arguments={"text": "x"}),)
    entries = [
        ("id0", _user(100)),
        ("id1", _assistant(100, calls)),
        ("id2", _tool_result("c1", 100)),
        ("id3", _assistant(100)),
    ]
    cut = find_cut_point(entries, 0, keep_recent_tokens=75)
    # The cut lands on the assistant carrying the call; its result follows
    # inside the kept tail, so the pair survives.
    assert entries[cut.first_kept_index][0] == "id1"
    assert cut.is_split_turn is True
    assert entries[cut.turn_start_index][0] == "id0"


def test_cut_defaults_to_first_cut_point_when_history_is_small() -> None:
    entries = [("id0", _user(10)), ("id1", _assistant(10))]
    cut = find_cut_point(entries, 0, keep_recent_tokens=20_000)
    assert cut.first_kept_index == 0


def test_projected_summaries_are_not_cut_points() -> None:
    entries = [
        (None, UserMessage(content="older summary")),
        ("id0", _user(100)),
        ("id1", _assistant(100)),
    ]
    cut = find_cut_point(entries, 1, keep_recent_tokens=10)
    assert entries[cut.first_kept_index][0] is not None


# ── prepare_compaction ───────────────────────────────────────────────────────


def test_prepare_compaction_returns_none_without_history() -> None:
    entries = [("id0", _user(10)), ("id1", _assistant(10))]
    assert prepare_compaction(entries, CompactionSettings()) is None
    assert prepare_compaction([], CompactionSettings()) is None
    assert (
        prepare_compaction(
            [(None, UserMessage(content="summary only"))],
            CompactionSettings(),
        )
        is None
    )


def test_prepare_compaction_iterates_the_latest_summary() -> None:
    entries = [
        (None, UserMessage(content="latest summary")),
        ("id0", _user(400)),
        ("id1", _assistant(400, (ToolCall(id="c1", name="echo", arguments={}),))),
        ("id2", _tool_result("c1", 400)),
        ("id3", _user(400)),
        ("id4", _assistant(400)),
        ("id5", _tool_result("c2", 400)),
    ]
    preparation = prepare_compaction(
        entries, CompactionSettings(keep_recent_tokens=310)
    )
    assert preparation is not None
    assert preparation.previous_summary == "latest summary"
    # The cut kept roughly the newest 310 tokens, so the durable summary
    # plus the messages before the cut are the summarizable history.
    assert preparation.first_kept_transcript_id == "id3"
    assert all(message is not entries[0][1] for message in preparation.history)


# ── summarization ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compact_context_summarizes_with_usage() -> None:
    model = ScriptedModel(
        [text_events("## Goal\ncontinue the work", usage={"total_tokens": 33})]
    )
    entries = [
        ("id0", _user(400)),
        ("id1", _assistant(400)),
        ("id2", _user(400)),
        ("id3", _assistant(400)),
    ]
    preparation = prepare_compaction(
        entries, CompactionSettings(keep_recent_tokens=150)
    )
    assert preparation is not None
    result = await compact_context(model, preparation, CompactionSettings())
    assert "Goal" in result.summary
    assert result.usage is not None and result.usage.total_tokens == 33
    assert result.first_kept_transcript_id == preparation.first_kept_transcript_id
    # One bounded request, no Tool exposure, conversation wrapped in tags.
    assert len(model.requests) == 1
    request = model.requests[0]
    assert request.options.get("max_tokens", 0) > 0
    assert "tools" not in request.options
    prompt_text = str(request.messages[-1].get("content"))
    assert "<conversation>" in prompt_text
    assert "[User]:" in prompt_text


@pytest.mark.asyncio
async def test_compact_context_includes_previous_summary_on_iteration() -> None:
    model = ScriptedModel([text_events("updated summary")])
    entries = [
        (None, UserMessage(content="previous checkpoint")),
        ("id0", _user(400)),
        ("id1", _assistant(400)),
        ("id2", _user(400)),
        ("id3", _assistant(400)),
    ]
    preparation = prepare_compaction(
        entries, CompactionSettings(keep_recent_tokens=150)
    )
    assert preparation is not None
    await compact_context(model, preparation, CompactionSettings())
    prompt_text = str(model.requests[0].messages[-1].get("content"))
    assert "<previous-summary>" in prompt_text
    assert "previous checkpoint" in prompt_text


@pytest.mark.asyncio
async def test_compact_context_merges_split_turn_prefix() -> None:
    model = ScriptedModel(
        [text_events("history summary"), text_events("prefix summary")]
    )
    calls = (ToolCall(id="c1", name="echo", arguments={"text": "x"}),)
    entries = [
        ("id0", _user(400)),
        ("id1", _assistant(100)),
        ("id2", _user(400)),
        ("id3", _assistant(100, calls)),
        ("id4", _tool_result("c1", 400)),
    ]
    preparation = prepare_compaction(
        entries, CompactionSettings(keep_recent_tokens=130)
    )
    assert preparation is not None and preparation.is_split_turn
    result = await compact_context(model, preparation, CompactionSettings())
    assert "history summary" in result.summary
    assert "prefix summary" in result.summary
    assert len(model.requests) == 2
    # Split-turn usage is the sum of both summarization requests when both
    # report usage; ScriptedModel reports none here.
    assert result.usage is None or result.usage.total_tokens is not None


@pytest.mark.asyncio
async def test_summarization_failure_raises() -> None:
    model = ScriptedModel([failed_events("provider down")])
    entries = [("id0", _user(400)), ("id1", _assistant(400)), ("id2", _user(400))]
    preparation = prepare_compaction(
        entries, CompactionSettings(keep_recent_tokens=150)
    )
    assert preparation is not None
    with pytest.raises(SummarizationError):
        await compact_context(model, preparation, CompactionSettings())


# ── overflow detection ───────────────────────────────────────────────────────


def test_overflow_error_patterns() -> None:
    overflow = AssistantMessage(
        error="prompt is too long: 213462 tokens > 200000 maximum"
    )
    assert is_context_overflow(overflow, 200_000) is True
    other = AssistantMessage(error="provider exploded")
    assert is_context_overflow(other, 200_000) is False


def test_non_overflow_patterns_win() -> None:
    throttled = AssistantMessage(
        error="Throttling error: Too many tokens, please wait"
    )
    assert is_context_overflow(throttled, 200_000) is False
    rate_limited = AssistantMessage(
        error="rate limit: token limit exceeded for this minute"
    )
    assert is_context_overflow(rate_limited, 200_000) is False


def test_silent_overflow_via_usage() -> None:
    silent = AssistantMessage(
        text="ok",
        finish_reason="stop",
        usage=ModelUsage(input_tokens=210_000, output_tokens=50),
    )
    assert is_context_overflow(silent, 200_000) is True
    assert is_context_overflow(silent, 300_000) is False


def test_length_stop_with_zero_output_and_full_window() -> None:
    truncated = AssistantMessage(
        text="",
        finish_reason="length",
        usage=ModelUsage(input_tokens=199_000, output_tokens=0),
    )
    assert is_context_overflow(truncated, 200_000) is True
    partial = AssistantMessage(
        text="half an answer",
        finish_reason="length",
        usage=ModelUsage(input_tokens=50_000, output_tokens=800),
    )
    assert is_context_overflow(partial, 200_000) is False
