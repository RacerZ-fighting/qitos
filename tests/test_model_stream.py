"""Provider-neutral model stream event contract tests."""

from __future__ import annotations

import pytest

from qitos.core import ModelStreamEvent, ModelStreamEventType, ModelUsage


def test_stream_event_requires_one_explicit_semantic_kind() -> None:
    event = ModelStreamEvent(
        type=ModelStreamEventType.TEXT_DELTA,
        text="evidence",
        event_type="provider.text.delta",
    )

    assert event.text == "evidence"
    assert event.done is False
    assert event.is_final is False


def test_stream_event_rejects_ambiguous_payloads() -> None:
    with pytest.raises(ValueError, match="another event type"):
        ModelStreamEvent(
            type=ModelStreamEventType.TEXT_DELTA,
            text="answer",
            reasoning_content="private reasoning",
        )

    with pytest.raises(ValueError, match="finish reason"):
        ModelStreamEvent(
            type=ModelStreamEventType.TEXT_DELTA,
            text="answer",
            finish_reason="stop",
        )


def test_usage_is_normalized_only_on_usage_or_completed_events() -> None:
    usage_event = ModelStreamEvent(
        type=ModelStreamEventType.USAGE,
        usage={"prompt_tokens": 3, "completion_tokens": 2},
    )

    assert isinstance(usage_event.usage, ModelUsage)
    assert usage_event.usage.input_tokens == 3
    assert usage_event.usage.output_tokens == 2
    with pytest.raises(ValueError, match="another event type"):
        ModelStreamEvent(
            type=ModelStreamEventType.REASONING_DELTA,
            reasoning_content="check",
            usage={"prompt_tokens": 1},
        )


def test_failed_terminal_is_not_a_successful_completion() -> None:
    failed = ModelStreamEvent(
        type=ModelStreamEventType.FAILED,
        event_type="provider.failed",
        error="provider rejected the request",
    )

    assert failed.is_final is True
    assert failed.done is False
    with pytest.raises(ValueError, match="must contain an error"):
        ModelStreamEvent(type=ModelStreamEventType.FAILED)
