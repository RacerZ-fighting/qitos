from __future__ import annotations

import pytest

from qitos.core import ModelResponse, ModelTiming, ModelUsage, ModelUsageSource
from qitos.models import ModelStreamChunk


def test_usage_mapping_normalizes_typed_fields_without_losing_provider_details() -> (
    None
):
    raw = {
        "prompt_tokens": 13,
        "completion_tokens": 5,
        "total_tokens": 18,
        "input_tokens_details": {"cached_tokens": 7},
        "output_tokens_details": {"reasoning_tokens": 3},
        "provider_extension": {"tier": "priority"},
    }

    usage = ModelUsage.from_mapping(raw)

    assert usage.input_tokens == 13
    assert usage.output_tokens == 5
    assert usage.total_tokens == 18
    assert usage.cache_read_tokens == 7
    assert usage.reasoning_tokens == 3
    assert usage.cache_write_tokens is None
    assert usage.source is ModelUsageSource.PROVIDER
    assert usage == raw
    assert usage.to_dict() == raw


def test_usage_accepts_anthropic_cache_fields_and_preserves_unknown_counts() -> None:
    usage = ModelUsage.from_mapping(
        {
            "prompt_tokens": 12,
            "completion_tokens": 4,
            "cache_creation_input_tokens": 3,
            "cache_read_input_tokens": 6,
        }
    )

    assert usage.input_tokens == 12
    assert usage.output_tokens == 4
    assert usage.total_tokens is None
    assert usage.cache_read_tokens == 6
    assert usage.cache_write_tokens == 3


def test_usage_details_are_immutable_and_return_defensive_copies() -> None:
    raw = {"prompt_tokens": 2, "details": {"cached": 1}}
    usage = ModelUsage.from_mapping(raw)
    raw["details"]["cached"] = 99

    exposed = usage["details"]
    exposed["cached"] = 42

    assert usage["details"] == {"cached": 1}


@pytest.mark.parametrize("value", [-1, True, "1"])
def test_usage_rejects_invalid_token_counts(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ModelUsage.from_mapping({"prompt_tokens": value})


def test_model_transactions_normalize_compatible_usage_mappings() -> None:
    response = ModelResponse(text="done", usage={"prompt_tokens": 2})
    chunk = ModelStreamChunk(done=True, usage={"completion_tokens": 1})

    assert isinstance(response.usage, ModelUsage)
    assert response.usage.input_tokens == 2
    assert response.to_summary_dict()["usage"] == {"prompt_tokens": 2}
    assert response.to_summary_dict()["usage_source"] == "provider"
    assert isinstance(chunk.usage, ModelUsage)
    assert chunk.usage.output_tokens == 1


def test_model_transaction_preserves_estimated_usage_source() -> None:
    usage = ModelUsage(input_tokens=2, source=ModelUsageSource.ESTIMATE)

    response = ModelResponse(text="done", usage=usage)
    chunk = ModelStreamChunk(done=True, usage=usage)

    assert response.usage is usage
    assert chunk.usage is usage
    assert response.to_summary_dict()["usage_source"] == "estimate"


def test_model_timing_round_trips_through_response_summary() -> None:
    timing = ModelTiming(
        total_ms=25,
        time_to_first_event_ms=4,
        time_to_first_content_ms=9,
    )

    response = ModelResponse(text="done", timing=timing)

    assert response.timing is timing
    assert response.to_summary_dict()["timing"] == timing.to_dict()


def test_model_timing_rejects_impossible_ordering() -> None:
    with pytest.raises(ValueError, match="must not precede"):
        ModelTiming(
            total_ms=25,
            time_to_first_event_ms=9,
            time_to_first_content_ms=4,
        )
