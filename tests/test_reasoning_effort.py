from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from qitos.harness import build_model_for_preset, resolve_reasoning


@pytest.mark.parametrize(
    ("requested", "sent"),
    [
        ("low", "low"),
        ("medium", "high"),
        ("high", "high"),
        ("xhigh", "max"),
        ("max", "max"),
    ],
)
def test_kimi_reasoning_effort_reaches_chat_completion_request(
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    sent: str,
) -> None:
    captured: dict[str, object] = {}

    class Completions:
        def create(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="ok", tool_calls=None)
                    )
                ],
                usage=None,
            )

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = lambda **_: SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    model = build_model_for_preset(
        family_id="kimi",
        model_name="kimi-k3",
        api_key="test-key",
        base_url="https://example.test/v1",
        reasoning_effort=requested,
    )

    response = model.call_raw([{"role": "user", "content": "answer"}])

    assert response.choices[0].message.content == "ok"
    assert captured["reasoning_effort"] == sent


def test_model_reasoning_reaches_responses_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Responses:
        def create(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return {
                "id": "response-1",
                "status": "completed",
                "model": "kimi-k3",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
            }

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = lambda **_: SimpleNamespace(responses=Responses())
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    model = build_model_for_preset(
        family_id="kimi",
        model_name="kimi-k3",
        api_key="test-key",
        base_url="https://example.test/v1",
        api_mode="responses",
        reasoning_effort="xhigh",
    )

    response = model.call_raw([{"role": "user", "content": "answer"}])

    assert response.text == "ok"
    assert captured["reasoning"] == {"effort": "max"}
    assert "include" not in captured


def test_gpt56_max_reasoning_and_opaque_continuation_reach_responses_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Responses:
        def create(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return {
                "id": "response-1",
                "status": "completed",
                "model": "gpt-5.6-luna",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
            }

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = lambda **_: SimpleNamespace(responses=Responses())
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    model = build_model_for_preset(
        model_name="gpt-5.6-luna",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        api_mode="responses",
        reasoning_effort="max",
        default_request_kwargs={"include": ["file_search_call.results"]},
    )

    response = model.call_raw([{"role": "user", "content": "answer"}])

    assert response.text == "ok"
    assert captured["reasoning"] == {"effort": "max"}
    assert captured["include"] == [
        "file_search_call.results",
        "reasoning.encrypted_content",
    ]
    assert model.qitos_harness_metadata["reasoning"]["resolved"] == "max"


def test_older_openai_reasoning_policy_does_not_claim_gpt56_max() -> None:
    resolution = resolve_reasoning(
        family_id="openai",
        model_name="gpt-5.5",
        api_mode="responses",
        requested="max",
    )

    assert resolution.resolved.value == "xhigh"


def test_transactional_stream_preserves_reasoning_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Stream:
        def __iter__(self):
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            reasoning_content="check premise",
                            tool_calls=None,
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="ok",
                            reasoning_content=None,
                            tool_calls=None,
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            )

        def close(self) -> None:
            return None

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = lambda **_: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **__: Stream()))
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    model = build_model_for_preset(
        family_id="kimi",
        model_name="kimi-k3",
        api_key="test-key",
        base_url="https://example.test/v1",
    )

    chunks = list(model.transactional_stream([{"role": "user", "content": "answer"}]))

    assert "".join(chunk.text for chunk in chunks) == "ok"
    assert "".join(chunk.reasoning_content or "" for chunk in chunks) == (
        "check premise"
    )
    assert chunks[-1].done is True
