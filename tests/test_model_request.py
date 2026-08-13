"""Contracts for immutable model requests and Provider continuation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qitos.core import ModelContinuation, ModelRequest


def _request(*, run_id: str = "run-1") -> ModelRequest:
    return ModelRequest(
        run_id=run_id,
        transaction_id=f"{run_id}:0",
        provider="openai",
        model="gpt-test",
        protocol="responses",
        messages=({"role": "user", "content": [{"type": "text", "text": "hi"}]},),
        options={
            "tools": [{"type": "function", "name": "lookup"}],
            "headers": {"Authorization": "Bearer secret"},
        },
    )


def test_model_request_freezes_nested_input_and_returns_isolated_copies() -> None:
    request = _request()

    with pytest.raises(TypeError):
        request.messages[0]["role"] = "assistant"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        request.model = "other"  # type: ignore[misc]

    messages = request.message_dicts()
    messages[0]["content"][0]["text"] = "changed"
    options = request.option_dict()
    options["tools"][0]["name"] = "changed"

    assert request.message_dicts()[0]["content"][0]["text"] == "hi"
    assert request.option_dict()["tools"][0]["name"] == "lookup"


def test_model_request_durable_snapshot_redacts_credentials_and_round_trips() -> None:
    request = _request()

    snapshot = request.to_dict()

    assert snapshot["options"]["headers"] == "[REDACTED]"
    restored = ModelRequest.from_dict(snapshot)
    assert restored.request_digest == request.request_digest
    assert restored.messages == request.messages


def test_model_continuation_is_bound_to_run_provider_model_and_protocol() -> None:
    request = _request()
    continuation = ModelContinuation(
        run_id=request.run_id,
        provider=request.provider,
        model=request.model,
        protocol=request.protocol,
        response_id="resp-1",
        prefix_items=2,
        prefix_digest="prefix",
        settings_digest="settings",
    )

    assert continuation.belongs_to(request) is True
    assert continuation.belongs_to(_request(run_id="forked-run")) is False


def test_model_request_rejects_non_json_provider_state() -> None:
    with pytest.raises(TypeError, match="JSON-compatible"):
        ModelRequest(
            run_id="run-1",
            transaction_id="run-1:0",
            provider="provider",
            model="model",
            protocol="protocol",
            messages=({"role": "user", "content": object()},),
        )
