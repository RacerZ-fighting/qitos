"""Contract tests for the non-OpenAI asynchronous model providers."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from qitos.models import (
    AnthropicModel,
    GeminiModel,
    LiteLLMModel,
    OllamaModel,
    infer_context_window,
)
from qitos.models.base import ModelStreamChunk


class _AsyncListStream(AsyncIterator[Any]):
    def __init__(self, items: list[Any]) -> None:
        self._items = iter(items)
        self.closed = False

    def __aiter__(self) -> _AsyncListStream:
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._items)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def aclose(self) -> None:
        self.closed = True


async def _collect(
    model: Any, messages: list[dict[str, Any]], **kwargs: Any
) -> list[ModelStreamChunk]:
    return [chunk async for chunk in model.stream(messages, **kwargs)]


@pytest.mark.asyncio
async def test_anthropic_preserves_block_order_thinking_tools_usage_and_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    events = _AsyncListStream(
        [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "usage": {
                        "input_tokens": 8,
                        "cache_read_input_tokens": 2,
                    },
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "check evidence"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "sig_1"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "lookup",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"q":"x"}'},
            },
            {"type": "content_block_stop", "index": 1},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 5},
            },
            {"type": "message_stop"},
        ]
    )

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            captured["client"] = kwargs
            self.messages = SimpleNamespace(create=self.create)
            self.closed = False

        async def create(self, **kwargs: Any) -> Any:
            captured["request"] = kwargs
            return events

        async def aclose(self) -> None:
            self.closed = True
            captured["client_closed"] = True

    fake = ModuleType("anthropic")
    fake.AsyncAnthropic = Client
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    model = AnthropicModel(api_key="test-key", model="claude-test", max_attempts=1)
    chunks = await _collect(
        model,
        [
            {"role": "system", "content": "Follow the contract."},
            {"role": "user", "content": "Look up x."},
        ],
    )

    assert captured["client"]["max_retries"] == 0
    assert captured["request"]["system"] == "Follow the contract."
    assert captured["request"]["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Look up x."}]}
    ]
    assert (
        "".join(chunk.reasoning_content or "" for chunk in chunks) == "check evidence"
    )
    terminal = chunks[-1]
    assert terminal.done is True
    assert terminal.finish_reason == "tool_use"
    assert terminal.tool_calls == [
        {
            "id": "toolu_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"q":"x"}'},
        }
    ]
    assert terminal.native_items == [
        {"type": "thinking", "thinking": "check evidence", "signature": "sig_1"},
        {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}},
    ]
    assert terminal.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 2,
    }
    assert events.closed is True
    assert captured["client_closed"] is True

    replay = model._anthropic_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": terminal.tool_calls,
                "native_items": terminal.native_items,
            },
            {
                "role": "tool",
                "tool_call_id": "toolu_1",
                "content": "result",
            },
        ]
    )
    assert replay[0]["content"] == terminal.native_items
    assert replay[1]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "result"}
    ]


@pytest.mark.asyncio
async def test_anthropic_request_defaults_reach_payload_and_allow_call_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []

    class Client:
        def __init__(self, **_: Any) -> None:
            self.messages = SimpleNamespace(create=self.create)

        async def create(self, **kwargs: Any) -> Any:
            requests.append(kwargs)
            return _AsyncListStream(
                [
                    {
                        "type": "message_start",
                        "message": {"id": "msg_1", "usage": {}},
                    },
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {"output_tokens": 1},
                    },
                    {"type": "message_stop"},
                ]
            )

        async def aclose(self) -> None:
            return None

    fake = ModuleType("anthropic")
    fake.AsyncAnthropic = Client
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    defaults = {
        "thinking": {"type": "enabled", "budget_tokens": 2_048},
        "output_config": {"effort": "medium"},
    }
    model = AnthropicModel(
        api_key="test-key",
        model="claude-sonnet-4-5",
        temperature=0.7,
        default_request_kwargs=defaults,
        max_attempts=1,
    )

    await _collect(model, [{"role": "user", "content": "answer"}])
    await _collect(
        model,
        [{"role": "user", "content": "answer"}],
        thinking={"budget_tokens": 4_096},
        output_config={"effort": "low"},
    )
    await _collect(
        model,
        [{"role": "user", "content": "answer"}],
        thinking={"type": "disabled"},
    )
    await _collect(
        model,
        [{"role": "user", "content": "answer"}],
        thinking={"type": "adaptive"},
    )

    assert requests[0]["thinking"] == {
        "type": "enabled",
        "budget_tokens": 2_048,
    }
    assert requests[0]["output_config"] == {"effort": "medium"}
    assert "temperature" not in requests[0]
    assert requests[1]["thinking"] == {
        "type": "enabled",
        "budget_tokens": 4_096,
    }
    assert requests[1]["output_config"] == {"effort": "low"}
    assert requests[2]["thinking"] == {"type": "disabled"}
    assert requests[2]["temperature"] == 0.7
    assert requests[3]["thinking"] == {"type": "adaptive"}
    assert "temperature" not in requests[3]
    assert defaults == {
        "thinking": {"type": "enabled", "budget_tokens": 2_048},
        "output_config": {"effort": "medium"},
    }


@pytest.mark.asyncio
async def test_gemini_uses_native_async_sdk_and_preserves_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    responses = _AsyncListStream(
        [
            {
                "response_id": "resp_1",
                "usage_metadata": {
                    "prompt_token_count": 7,
                    "candidates_token_count": 4,
                    "total_token_count": 11,
                    "thoughts_token_count": 2,
                },
                "candidates": [
                    {
                        "finish_reason": "STOP",
                        "content": {
                            "parts": [
                                {
                                    "text": "reason",
                                    "thought": True,
                                    "thought_signature": "sig",
                                },
                                {"text": "answer"},
                                {
                                    "function_call": {
                                        "id": "call_1",
                                        "name": "lookup",
                                        "args": {"q": "x"},
                                    }
                                },
                            ]
                        },
                    }
                ],
            }
        ]
    )

    class AsyncClient:
        def __init__(self) -> None:
            self.models = SimpleNamespace(generate_content_stream=self.generate)

        async def generate(self, **kwargs: Any) -> Any:
            captured["request"] = kwargs
            return responses

        async def aclose(self) -> None:
            captured["client_closed"] = True

    async_client = AsyncClient()

    class ClientOwner:
        def __init__(self, **kwargs: Any) -> None:
            captured["client"] = kwargs
            self.aio = async_client

    genai = SimpleNamespace(Client=ClientOwner)
    google = ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)

    model = GeminiModel(api_key="test-key", model="gemini-test", max_attempts=1)
    chunks = await _collect(
        model,
        [
            {"role": "system", "content": "Follow the contract."},
            {"role": "user", "content": "Look up x."},
        ],
    )

    request = captured["request"]
    assert request["contents"] == [{"role": "user", "parts": [{"text": "Look up x."}]}]
    assert request["config"]["system_instruction"] == "Follow the contract."
    assert "".join(chunk.reasoning_content or "" for chunk in chunks) == "reason"
    assert "".join(chunk.text for chunk in chunks) == "answer"
    terminal = chunks[-1]
    assert terminal.done is True
    assert terminal.event_metadata == {
        "provider": "gemini",
        "model": "gemini-test",
        "api_mode": "gemini_generate_content",
        "response_id": "resp_1",
    }
    assert terminal.tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"q":"x"}'},
        }
    ]
    assert terminal.usage == {
        "prompt_tokens": 7,
        "completion_tokens": 4,
        "total_tokens": 11,
        "reasoning_tokens": 2,
    }
    assert len(terminal.native_items or []) == 3
    assert responses.closed is True
    assert captured["client_closed"] is True


@pytest.mark.asyncio
async def test_litellm_uses_only_async_completion_and_disables_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    response = _AsyncListStream(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="ok",
                            reasoning_content="check",
                            tool_calls=None,
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=3,
                    completion_tokens=2,
                    total_tokens=5,
                    prompt_tokens_details=None,
                ),
            )
        ]
    )

    async def acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return response

    fake = ModuleType("litellm")
    fake.acompletion = acompletion
    monkeypatch.setitem(sys.modules, "litellm", fake)

    model = LiteLLMModel(model="anthropic/test", api_key="key", max_attempts=1)
    chunks = await _collect(
        model,
        [
            {
                "role": "user",
                "content": "hello",
                "native_items": [{"provider_only": True}],
            }
        ],
    )

    assert captured["num_retries"] == 0
    assert captured["stream"] is True
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
    assert "".join(chunk.text for chunk in chunks) == "ok"
    assert "".join(chunk.reasoning_content or "" for chunk in chunks) == "check"
    assert chunks[-1].usage == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
        "cached_tokens": None,
    }
    assert response.closed is True


@pytest.mark.asyncio
async def test_ollama_uses_official_async_chat_and_projects_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    responses = _AsyncListStream(
        [
            {"message": {}, "done": False},
            {
                "message": {
                    "content": "ok",
                    "thinking": "check",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "lookup", "arguments": {"q": "x"}},
                        }
                    ],
                },
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 4,
                "eval_count": 2,
            },
        ]
    )

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            captured["client"] = kwargs

        async def chat(self, **kwargs: Any) -> Any:
            captured["request"] = kwargs
            return responses

        async def aclose(self) -> None:
            captured["client_closed"] = True

    fake = ModuleType("ollama")
    fake.AsyncClient = Client
    monkeypatch.setitem(sys.modules, "ollama", fake)

    model = OllamaModel(model="llama-test", host="http://ollama.test", max_attempts=1)
    chunks = await _collect(
        model,
        [
            {
                "role": "assistant",
                "content": "calling",
                "native_items": [{"ignored": True}],
                "tool_calls": [
                    {"function": {"name": "lookup", "arguments": {"q": "x"}}}
                ],
            },
            {"role": "tool", "content": "result", "name": "lookup"},
        ],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
    )

    request = captured["request"]
    assert request["stream"] is True
    assert request["messages"] == [
        {
            "role": "assistant",
            "content": "calling",
            "tool_calls": [{"function": {"name": "lookup", "arguments": {"q": "x"}}}],
        },
        {"role": "tool", "content": "result", "tool_name": "lookup"},
    ]
    assert "".join(chunk.text for chunk in chunks) == "ok"
    assert "".join(chunk.reasoning_content or "" for chunk in chunks) == "check"
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].usage == {
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "total_tokens": 6,
    }
    assert chunks[-1].tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"q":"x"}'},
        }
    ]
    assert responses.closed is True
    assert captured["client_closed"] is True


def test_context_registry_infers_anthropic_and_gemini_windows() -> None:
    assert infer_context_window("claude-3-5-sonnet-latest") == 200_000
    assert infer_context_window("gemini-2.5-flash") == 1_048_576
