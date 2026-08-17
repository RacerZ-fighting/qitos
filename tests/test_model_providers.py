"""Contract tests for the non-OpenAI asynchronous model providers."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections.abc import AsyncIterator
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from qitos.core import ModelTransportError
from qitos.models import (
    AnthropicModel,
    GeminiModel,
    LiteLLMModel,
    ModelRequest,
    ModelStreamEventType,
    OllamaModel,
    infer_context_window,
)
from qitos.models.base import ModelStreamEvent
from qitos.models.anthropic import _AnthropicEventStream


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


class _AsyncCloser:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


async def _collect(
    model: Any, messages: list[dict[str, Any]], **kwargs: Any
) -> list[ModelStreamEvent]:
    deadline = kwargs.pop("deadline_monotonic", None)
    request = ModelRequest(
        run_id="provider-test",
        transaction_id="provider-test:0",
        provider=model.provider_name,
        model=model.model,
        protocol=model.capabilities.api.value,
        messages=tuple(messages),
        options=kwargs,
        deadline_monotonic=deadline,
    )
    return [chunk async for chunk in model.stream(request)]


def _managed_web_search_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search current public sources",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
    ]


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
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "answer"},
            },
            {"type": "content_block_stop", "index": 1},
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "lookup",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "input_json_delta", "partial_json": '{"q":"x"}'},
            },
            {"type": "content_block_stop", "index": 2},
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
    assert captured["request"]["metadata"] == {"user_id": "provider-test"}
    assert captured["request"]["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Look up x."}]}
    ]
    assert (
        "".join(chunk.reasoning_content or "" for chunk in chunks) == "check evidence"
    )
    assert "".join(chunk.text for chunk in chunks) == "answer"
    terminal = chunks[-1]
    assert terminal.done is True
    assert terminal.finish_reason == "tool_use"
    assert terminal.event_metadata["id"] == "msg_1"
    assert terminal.continuation is None
    assert terminal.tool_calls == [
        {
            "id": "toolu_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"q":"x"}'},
        }
    ]
    assert terminal.native_items == [
        {"type": "thinking", "thinking": "check evidence", "signature": "sig_1"},
        {"type": "text", "text": "answer"},
        {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}},
    ]
    assert terminal.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
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
async def test_anthropic_retries_without_affinity_when_endpoint_rejects_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    events = _AsyncListStream(
        [
            {
                "type": "message_start",
                "message": {"id": "msg_1", "usage": {"input_tokens": 1}},
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        ]
    )

    class UnsupportedAffinityError(RuntimeError):
        status_code = 400
        body = {"error": "unknown parameter metadata.user_id"}

    class Client:
        def __init__(self, **_: Any) -> None:
            self.messages = SimpleNamespace(create=self.create)

        async def create(self, **kwargs: Any) -> Any:
            requests.append(dict(kwargs))
            if len(requests) == 1:
                raise UnsupportedAffinityError("metadata user_id is not supported")
            return events

        async def aclose(self) -> None:
            return None

    fake = ModuleType("anthropic")
    fake.AsyncAnthropic = Client
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    model = AnthropicModel(api_key="test-key", model="claude-test", max_attempts=1)
    chunks = await _collect(model, [{"role": "user", "content": "question"}])

    assert chunks[-1].done is True
    assert requests[0]["metadata"] == {"user_id": "provider-test"}
    assert "metadata" not in requests[1]


@pytest.mark.asyncio
async def test_anthropic_prefers_hosted_search_and_preserves_native_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    search_results = [
        {
            "type": "web_search_result",
            "title": "Vendor advisory",
            "url": "https://vendor.example/advisory",
        }
    ]
    citation = {
        "type": "web_search_result_location",
        "url": "https://vendor.example/advisory",
        "title": "Vendor advisory",
        "cited_text": "Affected versions",
    }
    events = _AsyncListStream(
        [
            {
                "type": "message_start",
                "message": {"id": "msg_web", "usage": {"input_tokens": 3}},
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "server_tool_use",
                    "id": "srvtoolu_1",
                    "name": "web_search",
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"query":"vendor advisory"}',
                },
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_1",
                    "content": search_results,
                },
            },
            {"type": "content_block_stop", "index": 1},
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "text_delta", "text": "Affected versions"},
            },
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "citations_delta", "citation": citation},
            },
            {"type": "content_block_stop", "index": 2},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 4},
            },
            {"type": "message_stop"},
        ]
    )

    class Client:
        def __init__(self, **_: Any) -> None:
            self.messages = SimpleNamespace(create=self.create)

        async def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return events

        async def aclose(self) -> None:
            return None

    fake = ModuleType("anthropic")
    fake.AsyncAnthropic = Client
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    model = AnthropicModel(api_key="test-key", model="claude-test", max_attempts=1)
    options = model.build_tool_schema_request_options(
        _managed_web_search_schema(),
        delivery="api_parameter",
    )
    chunks = await _collect(
        model,
        [{"role": "user", "content": "Find the current advisory"}],
        **options,
    )

    assert captured["tools"] == [
        {"type": "web_search_20250305", "name": "web_search"}
    ]
    assert [chunk.type for chunk in chunks] == [
        ModelStreamEventType.LIFECYCLE,
        ModelStreamEventType.LIFECYCLE,
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.COMPLETED,
    ]
    terminal = chunks[-1]
    assert terminal.tool_calls is None
    assert terminal.native_items == [
        {
            "type": "server_tool_use",
            "id": "srvtoolu_1",
            "name": "web_search",
            "input": {"query": "vendor advisory"},
        },
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": search_results,
        },
        {"type": "text", "text": "Affected versions", "citations": [citation]},
    ]
    replay = model._anthropic_messages(
        [
            {
                "role": "assistant",
                "content": "Affected versions",
                "native_items": terminal.native_items,
            }
        ]
    )
    assert replay[0]["content"] == terminal.native_items


@pytest.mark.asyncio
async def test_anthropic_falls_back_to_managed_search_on_request_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    terminal = _AsyncListStream(
        [
            {
                "type": "message_start",
                "message": {"id": "msg_1", "usage": {}},
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "web_search",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"query":"vendor advisory"}',
                },
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        ]
    )

    class UnsupportedWebSearchError(Exception):
        status_code = 400
        body = {"error": "web_search is not enabled for this organization"}

    class Client:
        def __init__(self, **_: Any) -> None:
            self.messages = SimpleNamespace(create=self.create)

        async def create(self, **kwargs: Any) -> Any:
            requests.append(kwargs)
            if kwargs["tools"][0].get("type") == "web_search_20250305":
                raise UnsupportedWebSearchError("web search is not enabled")
            return terminal

        async def aclose(self) -> None:
            return None

    fake = ModuleType("anthropic")
    fake.AsyncAnthropic = Client
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    model = AnthropicModel(api_key="test-key", model="claude-test", max_attempts=1)
    options = model.build_tool_schema_request_options(
        _managed_web_search_schema(),
        delivery="api_parameter",
    )
    chunks = await _collect(
        model,
        [{"role": "user", "content": "Find the current advisory"}],
        **options,
    )

    assert requests[0]["tools"] == [
        {"type": "web_search_20250305", "name": "web_search"}
    ]
    assert requests[1]["tools"][0]["name"] == "web_search"
    assert requests[1]["tools"][0]["input_schema"]["required"] == ["query"]
    assert chunks[-1].tool_calls == [
        {
            "id": "toolu_1",
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": '{"query":"vendor advisory"}',
            },
        }
    ]


@pytest.mark.asyncio
async def test_anthropic_does_not_replay_after_server_tool_use_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []

    class StreamFailure(ConnectionError):
        pass

    class FailingStream(AsyncIterator[Any]):
        def __init__(self) -> None:
            self._events = iter(
                [
                    {
                        "type": "message_start",
                        "message": {"id": "msg_1", "usage": {}},
                    },
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "server_tool_use",
                            "id": "srvtoolu_1",
                            "name": "web_search",
                            "input": {"query": "vendor advisory"},
                        },
                    },
                ]
            )

        def __aiter__(self) -> FailingStream:
            return self

        async def __anext__(self) -> Any:
            try:
                return next(self._events)
            except StopIteration:
                raise StreamFailure("provider disconnected after server tool use")

        async def aclose(self) -> None:
            return None

    class Client:
        def __init__(self, **_: Any) -> None:
            self.messages = SimpleNamespace(create=self.create)

        async def create(self, **kwargs: Any) -> Any:
            requests.append(kwargs)
            return FailingStream()

        async def aclose(self) -> None:
            return None

    fake = ModuleType("anthropic")
    fake.AsyncAnthropic = Client
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    model = AnthropicModel(
        api_key="test-key",
        model="claude-test",
        max_attempts=2,
    )
    options = model.build_tool_schema_request_options(
        _managed_web_search_schema(),
        delivery="api_parameter",
    )

    with pytest.raises(ModelTransportError, match="provider disconnected"):
        await _collect(
            model,
            [{"role": "user", "content": "Find the current advisory"}],
            **options,
        )

    assert len(requests) == 1


def test_kimi_anthropic_transport_keeps_managed_search_schema() -> None:
    model = AnthropicModel(
        api_key="test-key",
        base_url="https://api.moonshot.example/anthropic",
        model="kimi-test",
        provider_name="kimi",
    )

    options = model.build_tool_schema_request_options(
        _managed_web_search_schema(),
        delivery="api_parameter",
    )

    assert options == {
        "tools": [
            {
                "name": "web_search",
                "description": "Search current public sources",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            }
        ]
    }


@pytest.mark.asyncio
async def test_anthropic_interleaved_tool_blocks_keep_arguments_separate() -> None:
    marker = time.monotonic_ns()
    expected = {
        "call-left": {"value": f"left-{marker}"},
        "call-right": {"value": f"right-{marker}"},
    }
    specs = [
        (index, call_id, json.dumps(arguments, separators=(",", ":")))
        for index, (call_id, arguments) in enumerate(expected.items(), start=2)
    ]
    payload: list[dict[str, Any]] = []
    for index, call_id, _ in specs:
        payload.append(
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": call_id,
                    "name": "lookup",
                    "input": {},
                },
            }
        )
    fragments = {
        index: (arguments[: len(arguments) // 2], arguments[len(arguments) // 2 :])
        for index, _, arguments in specs
    }
    for fragment_index in range(2):
        for index, _, _ in reversed(specs):
            payload.append(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": fragments[index][fragment_index],
                    },
                }
            )
    for index, _, _ in specs:
        payload.append({"type": "content_block_stop", "index": index})
    payload.extend(
        [
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
            {"type": "message_stop"},
        ]
    )
    events = _AsyncListStream(payload)
    client = _AsyncCloser()
    stream = _AnthropicEventStream(
        events,
        client,
        provider="anthropic",
        model="claude-test",
    )

    chunks = [chunk async for chunk in stream]
    await stream.aclose()

    terminal = chunks[-1]
    assert terminal.tool_calls is not None
    actual = {
        call["id"]: json.loads(call["function"]["arguments"])
        for call in terminal.tool_calls
    }
    assert actual == expected
    assert terminal.event_metadata["invalid_tool_calls"] == []
    assert events.closed is True
    assert client.closed is True


@pytest.mark.parametrize(
    ("close_block", "arguments", "stop_reason", "expected_code"),
    [
        (False, '{"value":', "max_tokens", "tool_call_not_completed"),
        (True, '{"value":', "tool_use", "tool_call_arguments_invalid"),
        (
            True,
            '{"value":"complete"}',
            "max_tokens",
            "tool_call_unexpected_stop_reason",
        ),
    ],
)
@pytest.mark.asyncio
async def test_anthropic_invalid_tool_blocks_never_become_calls_or_replay(
    close_block: bool,
    arguments: str,
    stop_reason: str,
    expected_code: str,
) -> None:
    block_index = 3
    call_id = f"call-{time.monotonic_ns()}"
    payload: list[dict[str, Any]] = [
        {
            "type": "content_block_start",
            "index": block_index,
            "content_block": {
                "type": "tool_use",
                "id": call_id,
                "name": "lookup",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": block_index,
            "delta": {"type": "input_json_delta", "partial_json": arguments},
        },
    ]
    if close_block:
        payload.append({"type": "content_block_stop", "index": block_index})
    payload.extend(
        [
            {"type": "message_delta", "delta": {"stop_reason": stop_reason}},
            {"type": "message_stop"},
        ]
    )
    stream = _AnthropicEventStream(
        _AsyncListStream(payload),
        _AsyncCloser(),
        provider="anthropic",
        model="claude-test",
    )

    chunks = [chunk async for chunk in stream]

    terminal = chunks[-1]
    assert terminal.tool_calls is None
    invalid = terminal.event_metadata["invalid_tool_calls"]
    assert invalid[0]["call_id"] == call_id
    assert invalid[0]["code"] == expected_code
    assert invalid[0]["arguments_chars"] == len(arguments)
    assert terminal.native_items is None

    model = AnthropicModel(api_key="test-key", model="claude-test")
    replay = model._anthropic_messages(
        [
            {
                "role": "assistant",
                "content": "recover",
                "native_items": terminal.native_items,
            }
        ]
    )
    assert all(
        block.get("type") != "tool_use"
        for message in replay
        for block in message["content"]
    )


@pytest.mark.asyncio
async def test_anthropic_error_is_an_explicit_failed_terminal() -> None:
    events = _AsyncListStream(
        [{"type": "error", "error": {"message": "overloaded"}}]
    )
    client = _AsyncCloser()
    stream = _AnthropicEventStream(
        events,
        client,
        provider="anthropic",
        model="claude-test",
    )

    terminal = await stream.__anext__()

    assert terminal.type is ModelStreamEventType.FAILED
    assert terminal.is_final is True
    assert terminal.done is False
    assert "overloaded" in str(terminal.error)
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()
    await stream.aclose()
    assert events.closed is True
    assert client.closed is True


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
async def test_anthropic_cancellation_closes_stream_and_client_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    attempts = 0
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
            self.messages = SimpleNamespace(create=self.create)

        async def create(self, **_: Any) -> Any:
            nonlocal attempts
            attempts += 1
            return BlockingStream()

        async def aclose(self) -> None:
            closed["client"] = True

    fake = ModuleType("anthropic")
    fake.AsyncAnthropic = Client
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    model = AnthropicModel(
        api_key="test-key",
        model="claude-test",
        max_attempts=3,
    )
    task = asyncio.create_task(_collect(model, [{"role": "user", "content": "answer"}]))
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert attempts == 1
    assert closed == {"stream": True, "client": True}


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


def test_anthropic_tool_result_block_carries_is_error() -> None:
    model = AnthropicModel(api_key="test-key", model="claude-test", max_attempts=1)
    replay = model._anthropic_messages(
        [
            {
                "role": "tool",
                "tool_call_id": "toolu_9",
                "content": "permission denied",
                "is_error": True,
            },
            {
                "role": "tool",
                "tool_call_id": "toolu_8",
                "content": "fine",
                "is_error": False,
            },
        ]
    )
    # Consecutive tool results merge into one user message; the error flag
    # rides on the failing block only (Pi keeps isError visible to the model).
    blocks = replay[0]["content"]
    assert blocks[0]["is_error"] is True
    assert "is_error" not in blocks[1]
