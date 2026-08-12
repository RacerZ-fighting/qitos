"""OpenAI Responses and Chat Completions adapter contract tests."""

from __future__ import annotations

import asyncio
import base64
import sys
import time
from collections.abc import AsyncIterator
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from qitos.core import ModelTransportError
from qitos.core.model_response import ModelResponse
from qitos.models._openai_responses import (
    _model_response_from_responses,
    _to_responses_input,
    _to_responses_tool_choice,
    _to_responses_tools,
)
from qitos.models.openai import OpenAICompatibleModel, OpenAIModel


class _AsyncListStream(AsyncIterator[Any]):
    def __init__(self, items: list[Any], *, failure: Exception | None = None) -> None:
        self._items = iter(items)
        self._failure = failure
        self.closed = False

    def __aiter__(self) -> _AsyncListStream:
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._items)
        except StopIteration:
            if self._failure is not None:
                failure, self._failure = self._failure, None
                raise failure
            raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


async def _collect(
    model: Any, messages: list[dict[str, Any]], **kwargs: Any
) -> list[Any]:
    return [chunk async for chunk in model.stream(messages, **kwargs)]


def test_model_response_summary_redacts_opaque_reasoning_state() -> None:
    response = ModelResponse(
        text="answer",
        native_items=[
            {
                "type": "reasoning",
                "id": "rs_1",
                "encrypted_content": "opaque-secret",
                "summary": [{"type": "summary_text", "text": "safe"}],
            }
        ],
    )

    summary = response.to_summary_dict()

    assert summary["native_items"] == [
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "safe"}],
        }
    ]


def test_responses_input_replays_native_items_without_mirror_duplicates() -> None:
    messages = [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "answer",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q":"x"}'},
                }
            ],
            "native_items": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "answer"}],
                },
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "encrypted_content": "opaque",
                },
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": '{"q":"x"}',
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "result",
            "native_items": [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "result",
                }
            ],
        },
    ]

    payload = _to_responses_input(messages)

    assert payload[0] == {"role": "user", "content": "question"}
    assert [item["type"] for item in payload[1:]] == [
        "message",
        "reasoning",
        "function_call",
        "function_call_output",
    ]
    assert sum(item.get("call_id") == "call_1" for item in payload) == 2


def test_responses_tools_and_forced_choice_use_native_shape() -> None:
    tools = _to_responses_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a value",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                    "strict": True,
                },
            }
        ]
    )

    assert tools == [
        {
            "type": "function",
            "name": "lookup",
            "description": "Look up a value",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
            },
            "strict": True,
        }
    ]
    assert _to_responses_tool_choice(
        {"type": "function", "function": {"name": "lookup"}}
    ) == {"type": "function", "name": "lookup"}


def test_responses_normalization_preserves_order_ids_and_usage() -> None:
    normalized = _model_response_from_responses(
        {
            "id": "resp_1",
            "status": "completed",
            "model": "gpt-test",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "encrypted_content": "opaque",
                },
                {
                    "type": "message",
                    "id": "msg_1",
                    "content": [{"type": "output_text", "text": "answer"}],
                },
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": '{"q":"x"}',
                    "status": "completed",
                },
            ],
            "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        },
        provider="openai",
    )

    assert normalized.text == "answer"
    assert [item["type"] for item in normalized.native_items or []] == [
        "reasoning",
        "message",
        "function_call",
    ]
    assert normalized.tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"q":"x"}'},
            "metadata": {"response_item_id": "fc_1", "status": "completed"},
        }
    ]
    assert normalized.usage == {
        "prompt_tokens": 5,
        "completion_tokens": 3,
        "total_tokens": 8,
    }


@pytest.mark.asyncio
async def test_openai_defaults_to_responses_and_streams_one_complete_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    events = _AsyncListStream(
        [
            {
                "type": "response.output_text.delta",
                "delta": "answer",
                "sequence_number": 1,
            },
            {
                "type": "response.function_call_arguments.delta",
                "delta": '{"q":',
                "item_id": "fc_1",
                "output_index": 1,
            },
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": '{"q":"x"}',
                    "status": "completed",
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "model": "gpt-test",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_1",
                            "content": [{"type": "output_text", "text": "answer"}],
                        }
                    ],
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 3,
                        "total_tokens": 8,
                    },
                },
            },
        ]
    )

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            captured["client"] = kwargs
            self.responses = SimpleNamespace(create=self.create)

        async def create(self, **kwargs: Any) -> Any:
            captured["request"] = kwargs
            return events

        async def aclose(self) -> None:
            captured["client_closed"] = True

    fake = ModuleType("openai")
    fake.AsyncOpenAI = Client
    monkeypatch.setitem(sys.modules, "openai", fake)

    model = OpenAIModel(api_key="test-key", model="gpt-test", max_attempts=1)
    chunks = await _collect(
        model,
        [{"role": "user", "content": "question"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )

    assert model.api_mode == "responses"
    assert captured["client"]["max_retries"] == 0
    assert captured["request"]["stream"] is True
    assert captured["request"]["input"] == [{"role": "user", "content": "question"}]
    assert "reasoning.encrypted_content" in captured["request"]["include"]
    assert captured["request"]["tools"] == [
        {"type": "function", "name": "lookup", "parameters": {"type": "object"}}
    ]
    assert "".join(chunk.text for chunk in chunks) == "answer"
    assert [
        chunk.event_metadata["arguments_delta"]
        for chunk in chunks
        if chunk.event_type == "response.function_call_arguments.delta"
    ] == ['{"q":']
    terminal = chunks[-1]
    assert terminal.done is True
    assert terminal.tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"q":"x"}'},
            "metadata": {"response_item_id": "fc_1", "status": "completed"},
        }
    ]
    assert terminal.usage == {
        "prompt_tokens": 5,
        "completion_tokens": 3,
        "total_tokens": 8,
    }
    assert events.closed is True
    assert captured["client_closed"] is True


@pytest.mark.asyncio
async def test_responses_incomplete_is_a_terminal_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _AsyncListStream(
        [
            {"type": "response.output_text.delta", "delta": "partial"},
            {
                "type": "response.incomplete",
                "response": {
                    "id": "resp_1",
                    "status": "incomplete",
                    "model": "gpt-test",
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "partial"}],
                        }
                    ],
                    "incomplete_details": {"reason": "max_output_tokens"},
                },
            },
        ]
    )

    class Client:
        def __init__(self, **_: Any) -> None:
            self.responses = SimpleNamespace(create=self.create)

        async def create(self, **_: Any) -> Any:
            return events

        async def aclose(self) -> None:
            return None

    fake = ModuleType("openai")
    fake.AsyncOpenAI = Client
    monkeypatch.setitem(sys.modules, "openai", fake)

    model = OpenAIModel(api_key="key", model="gpt-test", max_attempts=1)
    chunks = await _collect(model, [{"role": "user", "content": "question"}])

    assert "".join(chunk.text for chunk in chunks) == "partial"
    assert chunks[-1].done is True
    assert chunks[-1].finish_reason == "max_output_tokens"
    assert chunks[-1].event_metadata["status"] == "incomplete"


@pytest.mark.asyncio
async def test_responses_mode_never_falls_back_to_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_called = False

    class Client:
        def __init__(self, **_: Any) -> None:
            async def chat_create(**__: Any) -> Any:
                nonlocal chat_called
                chat_called = True
                return _AsyncListStream([])

            self.chat = SimpleNamespace(completions=SimpleNamespace(create=chat_create))

        async def aclose(self) -> None:
            return None

    fake = ModuleType("openai")
    fake.AsyncOpenAI = Client
    monkeypatch.setitem(sys.modules, "openai", fake)

    model = OpenAIModel(api_key="key", model="gpt-test", max_attempts=1)
    with pytest.raises(RuntimeError, match="POST /v1/responses"):
        await _collect(model, [{"role": "user", "content": "question"}])

    assert chat_called is False


@pytest.mark.asyncio
async def test_responses_missing_terminal_fails_and_closes_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _AsyncListStream(
        [{"type": "response.output_text.delta", "delta": "partial"}]
    )
    closed: dict[str, bool] = {}

    class Client:
        def __init__(self, **_: Any) -> None:
            self.responses = SimpleNamespace(create=self.create)

        async def create(self, **_: Any) -> Any:
            return events

        async def aclose(self) -> None:
            closed["client"] = True

    fake = ModuleType("openai")
    fake.AsyncOpenAI = Client
    monkeypatch.setitem(sys.modules, "openai", fake)

    model = OpenAIModel(api_key="key", model="gpt-test", max_attempts=1)
    with pytest.raises(ModelTransportError, match="before response.completed"):
        await _collect(model, [{"role": "user", "content": "question"}])

    assert events.closed is True
    assert closed["client"] is True


@pytest.mark.asyncio
async def test_chat_compatibility_streams_reasoning_parallel_tool_calls_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    events = _AsyncListStream(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="answer",
                            reasoning_content="check",
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="call_1",
                                    type="function",
                                    function=SimpleNamespace(
                                        name="first", arguments='{"x":'
                                    ),
                                ),
                                SimpleNamespace(
                                    index=1,
                                    id="call_2",
                                    type="function",
                                    function=SimpleNamespace(
                                        name="second", arguments='{"y":'
                                    ),
                                ),
                            ],
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="",
                            reasoning_content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id=None,
                                    type=None,
                                    function=SimpleNamespace(name=None, arguments="1}"),
                                ),
                                SimpleNamespace(
                                    index=1,
                                    id=None,
                                    type=None,
                                    function=SimpleNamespace(name=None, arguments="2}"),
                                ),
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=6,
                    completion_tokens=4,
                    total_tokens=10,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=2),
                ),
            ),
        ]
    )

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        async def create(self, **kwargs: Any) -> Any:
            captured["request"] = kwargs
            return events

        async def aclose(self) -> None:
            captured["client_closed"] = True

    fake = ModuleType("openai")
    fake.AsyncOpenAI = Client
    monkeypatch.setitem(sys.modules, "openai", fake)

    model = OpenAICompatibleModel(
        model="compatible-test",
        api_key="key",
        base_url="https://example.test/v1",
        max_attempts=1,
    )
    chunks = await _collect(model, [{"role": "user", "content": "question"}])

    assert model.api_mode == "chat_completions"
    assert captured["client"]["max_retries"] == 0
    assert captured["request"]["stream"] is True
    assert "".join(chunk.text for chunk in chunks) == "answer"
    assert "".join(chunk.reasoning_content or "" for chunk in chunks) == "check"
    assert chunks[-1].tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "first", "arguments": '{"x":1}'},
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "second", "arguments": '{"y":2}'},
        },
    ]
    assert chunks[-1].usage == {
        "prompt_tokens": 6,
        "completion_tokens": 4,
        "total_tokens": 10,
        "cached_tokens": 2,
    }
    assert events.closed is True
    assert captured["client_closed"] is True


@pytest.mark.asyncio
async def test_chat_request_timeout_is_clamped_to_absolute_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    events = _AsyncListStream(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="ok", reasoning_content=None, tool_calls=None
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            )
        ]
    )

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        async def create(self, **kwargs: Any) -> Any:
            captured["request"] = kwargs
            return events

        async def aclose(self) -> None:
            return None

    fake = ModuleType("openai")
    fake.AsyncOpenAI = Client
    monkeypatch.setitem(sys.modules, "openai", fake)

    model = OpenAICompatibleModel(
        model="compatible-test",
        api_key="key",
        base_url="https://example.test/v1",
        timeout=120,
        max_attempts=1,
    )
    deadline = time.monotonic() + 1.0
    await _collect(
        model,
        [{"role": "user", "content": "question"}],
        deadline_monotonic=deadline,
    )

    assert 0 < float(captured["client"]["timeout"]) <= 1.0
    assert 0 < float(captured["request"]["timeout"]) <= 1.0


@pytest.mark.asyncio
async def test_chat_cancellation_closes_stream_and_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    closed: dict[str, bool] = {}

    class BlockingStream(AsyncIterator[Any]):
        def __aiter__(self) -> BlockingStream:
            return self

        async def __anext__(self) -> Any:
            entered.set()
            await release.wait()
            raise StopAsyncIteration

        async def aclose(self) -> None:
            closed["stream"] = True

    class Client:
        def __init__(self, **_: Any) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        async def create(self, **_: Any) -> Any:
            return BlockingStream()

        async def aclose(self) -> None:
            closed["client"] = True

    fake = ModuleType("openai")
    fake.AsyncOpenAI = Client
    monkeypatch.setitem(sys.modules, "openai", fake)

    model = OpenAICompatibleModel(
        model="compatible-test",
        api_key="key",
        base_url="https://example.test/v1",
        max_attempts=2,
    )
    task = asyncio.create_task(
        _collect(model, [{"role": "user", "content": "question"}])
    )
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert closed == {"stream": True, "client": True}


@pytest.mark.asyncio
async def test_chat_formats_multimodal_file_content(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    png_path = tmp_path / "shot.png"
    png_path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wn2gbcAAAAASUVORK5CYII="
        )
    )
    events = _AsyncListStream(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="ok", reasoning_content=None, tool_calls=None
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            )
        ]
    )

    class Client:
        def __init__(self, **_: Any) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        async def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return events

        async def aclose(self) -> None:
            return None

    fake = ModuleType("openai")
    fake.AsyncOpenAI = Client
    monkeypatch.setitem(sys.modules, "openai", fake)

    model = OpenAICompatibleModel(
        model="vision-test",
        api_key="key",
        base_url="https://example.test/v1",
        max_attempts=1,
    )
    await _collect(
        model,
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect."},
                    {"type": "image_file", "path": str(png_path), "detail": "high"},
                ],
            }
        ],
    )

    content = captured["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Inspect."}
    assert content[1]["image_url"]["detail"] == "high"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_api_mode_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="Unsupported api_mode"):
        OpenAICompatibleModel(
            model="test",
            api_key="key",
            base_url="https://example.test/v1",
            api_mode="automatic-fallback",
        )
