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

from qitos.core import ModelRequest, ModelStreamEventType, ModelTransportError
from qitos.core.model_response import ModelResponse
from qitos.models._openai_responses import (
    _ResponsesEventStream,
    _model_response_from_responses,
    _to_responses_input,
    _to_responses_tool_choice,
    _to_responses_tools,
)
from qitos.models.openai import (
    ChatStreamAccumulator,
    OpenAICompatibleModel,
    OpenAIModel,
    _to_openai_messages,
)


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
    deadline = kwargs.pop("deadline_monotonic", None)
    request = ModelRequest(
        run_id="openai-test",
        transaction_id="openai-test:0",
        provider=model.provider_name,
        model=model.model,
        protocol=model.capabilities.api.value,
        messages=tuple(messages),
        options=kwargs,
        deadline_monotonic=deadline,
    )
    return [chunk async for chunk in model.stream(request)]


def _request_for(
    model: Any,
    messages: list[dict[str, Any]],
    *,
    run_id: str = "continuation-run",
    continuation: Any = None,
    **options: Any,
) -> ModelRequest:
    return ModelRequest(
        run_id=run_id,
        transaction_id=f"{run_id}:request",
        provider=model.provider_name,
        model=model.model,
        protocol=model.capabilities.api.value,
        messages=tuple(messages),
        options=options,
        continuation=continuation,
    )


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


def test_responses_input_preserves_developer_context_role() -> None:
    payload = _to_responses_input(
        [{"role": "developer", "content": "Current state revision: 3"}]
    )

    assert payload == [
        {"role": "developer", "content": "Current state revision: 3"}
    ]


def test_compatibility_chat_tags_developer_context_as_user_content() -> None:
    payload = _to_openai_messages(
        [{"role": "developer", "content": "Current state revision: 3"}]
    )

    assert payload[0]["role"] == "user"
    assert "<runtime-context>" in payload[0]["content"]
    assert "Current state revision: 3" in payload[0]["content"]


def test_official_openai_chat_preserves_developer_context_role() -> None:
    payload = _to_openai_messages(
        [{"role": "developer", "content": "Current state revision: 3"}],
        developer_role=True,
    )

    assert payload == [
        {"role": "developer", "content": "Current state revision: 3"}
    ]


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
    assert _to_responses_tool_choice(
        {"type": "function", "function": {"name": "web_search"}},
        hosted_web_search=True,
    ) == {"type": "web_search"}


def _managed_web_search_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search public sources",
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


def test_responses_normalization_preserves_refusal_content() -> None:
    refusal = f"refusal-{time.monotonic_ns()}"

    normalized = _model_response_from_responses(
        {
            "id": "response-refusal",
            "status": "completed",
            "model": "gpt-test",
            "output": [
                {
                    "type": "message",
                    "id": "message-refusal",
                    "content": [{"type": "refusal", "refusal": refusal}],
                }
            ],
        },
        provider="openai",
    )

    assert normalized.text == refusal
    assert normalized.native_items is not None
    assert normalized.native_items[0]["content"][0]["refusal"] == refusal


@pytest.mark.asyncio
async def test_responses_lifecycle_backfills_terminal_only_content() -> None:
    response_id = f"resp_{time.monotonic_ns()}"
    answer = f"answer-{time.monotonic_ns()}"
    reasoning = f"reasoning-{time.monotonic_ns()}"
    events = _AsyncListStream(
        [
            {
                "type": "response.created",
                "response": {"id": response_id, "status": "in_progress"},
            },
            {
                "type": "response.in_progress",
                "response": {"id": response_id, "status": "in_progress"},
            },
            {
                "type": "response.output_item.added",
                "item": {"type": "message", "id": "message"},
            },
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "reasoning",
                    "id": "reasoning",
                    "summary": [{"type": "summary_text", "text": reasoning}],
                },
            },
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "id": "message",
                    "content": [{"type": "output_text", "text": answer}],
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "id": response_id,
                    "status": "completed",
                    "model": "gpt-test",
                    "output": [],
                },
            },
        ]
    )

    chunks = [chunk async for chunk in _ResponsesEventStream(events, provider="openai")]

    observed_types = {chunk.event_type for chunk in chunks}
    assert {
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.output_item.done",
        "response.completed",
    } <= observed_types
    assert "".join(chunk.text for chunk in chunks) == answer
    assert "".join(chunk.reasoning_content or "" for chunk in chunks) == reasoning
    assert chunks[-1].done is True
    assert chunks[-1].event_metadata["id"] == response_id


@pytest.mark.asyncio
async def test_responses_interleaved_function_deltas_keep_separate_state() -> None:
    call_specs = [
        ("item-a", "call-a", '{"a":', "1}"),
        ("item-b", "call-b", '{"b":', "2}"),
    ]
    events_payload: list[dict[str, Any]] = []
    for item_id, call_id, _, _ in call_specs:
        events_payload.append(
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": item_id,
                    "call_id": call_id,
                    "name": "lookup",
                    "arguments": "",
                },
            }
        )
    for fragment_index in range(2):
        for item_id, _, first, second in call_specs:
            events_payload.append(
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item_id,
                    "delta": (first, second)[fragment_index],
                }
            )
    completed_items = []
    for item_id, call_id, first, second in call_specs:
        completed = {
            "type": "function_call",
            "id": item_id,
            "call_id": call_id,
            "name": "lookup",
            "arguments": first + second,
            "status": "completed",
        }
        completed_items.append(completed)
        events_payload.append({"type": "response.output_item.done", "item": completed})
    events_payload.append(
        {
            "type": "response.completed",
            "response": {
                "id": "response-tools",
                "status": "completed",
                "model": "gpt-test",
                "output": [],
            },
        }
    )

    chunks = [
        chunk
        async for chunk in _ResponsesEventStream(
            _AsyncListStream(events_payload), provider="openai"
        )
    ]

    streamed_arguments: dict[str, str] = {}
    streamed_lengths: dict[str, int] = {}
    for chunk in chunks:
        if chunk.event_type != "response.function_call_arguments.delta":
            continue
        item_id = str(chunk.event_metadata.get("item_id"))
        streamed_arguments[item_id] = streamed_arguments.get(item_id, "") + str(
            chunk.event_metadata.get("arguments_delta")
        )
        streamed_lengths[item_id] = int(chunk.event_metadata["arguments_chars"])
    expected_arguments = {
        item_id: first + second for item_id, _, first, second in call_specs
    }
    assert streamed_arguments == expected_arguments
    assert streamed_lengths == {
        item_id: len(arguments) for item_id, arguments in expected_arguments.items()
    }
    terminal = chunks[-1]
    assert terminal.done is True
    assert terminal.tool_calls is not None
    assert {
        call["id"]: call["function"]["arguments"] for call in terminal.tool_calls
    } == {call_id: expected_arguments[item_id] for item_id, call_id, _, _ in call_specs}


@pytest.mark.asyncio
async def test_responses_incomplete_does_not_publish_tool_calls() -> None:
    partial_item = {
        "type": "function_call",
        "id": "partial-item",
        "call_id": "partial-call",
        "name": "lookup",
        "arguments": '{"q":',
        "status": "in_progress",
    }
    events = _AsyncListStream(
        [
            {"type": "response.output_item.done", "item": partial_item},
            {
                "type": "response.incomplete",
                "response": {
                    "id": "response-partial",
                    "status": "incomplete",
                    "model": "gpt-test",
                    "output": [partial_item],
                    "incomplete_details": {"reason": "max_output_tokens"},
                },
            },
        ]
    )

    chunks = [chunk async for chunk in _ResponsesEventStream(events, provider="openai")]

    terminal = chunks[-1]
    assert terminal.done is True
    assert terminal.finish_reason == "max_output_tokens"
    assert terminal.tool_calls is None
    assert terminal.native_items == [partial_item]


@pytest.mark.asyncio
async def test_responses_rejects_malformed_terminal_lifecycle_event() -> None:
    stream = _ResponsesEventStream(
        _AsyncListStream([{"type": "response.completed"}]),
        provider="openai",
    )

    with pytest.raises(ModelTransportError, match="missing a valid response"):
        await stream.__anext__()

    conflicting = _ResponsesEventStream(
        _AsyncListStream(
            [
                {
                    "type": "response.incomplete",
                    "response": {"id": "response", "status": "completed"},
                }
            ]
        ),
        provider="openai",
    )
    with pytest.raises(ModelTransportError, match="conflicting status"):
        await conflicting.__anext__()


@pytest.mark.asyncio
async def test_responses_failed_event_is_a_terminal_error() -> None:
    marker = f"provider-failure-{time.monotonic_ns()}"
    stream = _ResponsesEventStream(
        _AsyncListStream(
            [
                {
                    "type": "response.failed",
                    "response": {
                        "status": "failed",
                        "error": {"message": marker},
                    },
                }
            ]
        ),
        provider="openai",
    )

    terminal = await stream.__anext__()

    assert terminal.type is ModelStreamEventType.FAILED
    assert marker in str(terminal.error)
    assert terminal.is_final is True
    assert terminal.done is False
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()


@pytest.mark.asyncio
async def test_openai_defaults_to_responses_and_streams_one_complete_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    events = _AsyncListStream(
        [
            {
                "type": "response.reasoning_summary_text.delta",
                "delta": "check evidence",
                "sequence_number": 0,
            },
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
    assert captured["request"]["prompt_cache_key"] == "openai-test"
    assert captured["request"]["input"] == [{"role": "user", "content": "question"}]
    assert "reasoning.encrypted_content" in captured["request"]["include"]
    assert captured["request"]["tools"] == [
        {"type": "function", "name": "lookup", "parameters": {"type": "object"}}
    ]
    assert "".join(chunk.text for chunk in chunks) == "answer"
    assert (
        "".join(chunk.reasoning_content or "" for chunk in chunks) == "check evidence"
    )
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
async def test_responses_retries_without_prompt_cache_key_when_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    events = _AsyncListStream(
        [
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "model": "gpt-test",
                    "output": [],
                },
            }
        ]
    )

    class UnsupportedCacheKeyError(RuntimeError):
        status_code = 400
        body = {"error": "unknown parameter prompt_cache_key"}

    class Client:
        def __init__(self, **_: Any) -> None:
            self.responses = SimpleNamespace(create=self.create)

        async def create(self, **kwargs: Any) -> Any:
            requests.append(dict(kwargs))
            if len(requests) == 1:
                raise UnsupportedCacheKeyError(
                    "prompt_cache_key is not supported"
                )
            return events

        async def aclose(self) -> None:
            return None

    fake = ModuleType("openai")
    fake.AsyncOpenAI = Client
    monkeypatch.setitem(sys.modules, "openai", fake)

    model = OpenAIModel(api_key="test-key", model="gpt-test", max_attempts=1)
    chunks = await _collect(model, [{"role": "user", "content": "question"}])

    assert chunks[-1].done is True
    assert requests[0]["prompt_cache_key"] == "openai-test"
    assert "prompt_cache_key" not in requests[1]


@pytest.mark.asyncio
async def test_openai_responses_prefers_hosted_web_search_and_preserves_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    output = [
        {
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {"type": "search", "query": "vendor advisory"},
        },
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "Vendor advisory",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url": "https://vendor.example/advisory",
                            "title": "Vendor advisory",
                            "start_index": 0,
                            "end_index": 15,
                        }
                    ],
                }
            ],
        },
    ]
    events = _AsyncListStream(
        [
            {"type": "response.output_item.done", "item": item}
            for item in output
        ]
        + [
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "model": "gpt-test",
                    "output": output,
                },
            }
        ]
    )

    class Client:
        def __init__(self, **_: Any) -> None:
            self.responses = SimpleNamespace(create=self.create)

        async def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return events

        async def aclose(self) -> None:
            return None

    fake = ModuleType("openai")
    fake.AsyncOpenAI = Client
    monkeypatch.setitem(sys.modules, "openai", fake)

    model = OpenAIModel(api_key="key", model="gpt-test", max_attempts=1)
    options = model.build_tool_schema_request_options(
        _managed_web_search_schema(),
        delivery="api_parameter",
    )
    chunks = await _collect(
        model,
        [{"role": "user", "content": "Find the vendor advisory"}],
        **options,
    )

    assert captured["tools"] == [{"type": "web_search"}]
    assert chunks[-1].native_items == output
    message = chunks[-1].native_items[1]
    assert message["content"][0]["annotations"][0]["url"] == (
        "https://vendor.example/advisory"
    )


@pytest.mark.asyncio
async def test_openai_responses_falls_back_to_managed_search_on_request_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    terminal = _AsyncListStream(
        [
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "model": "gpt-test",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "web_search",
                            "arguments": '{"query":"vendor advisory"}',
                            "status": "completed",
                        }
                    ],
                },
            }
        ]
    )

    class UnsupportedWebSearchError(Exception):
        status_code = 400
        body = {"error": "web_search is not supported by this model"}

    class Client:
        def __init__(self, **_: Any) -> None:
            self.responses = SimpleNamespace(create=self.create)

        async def create(self, **kwargs: Any) -> Any:
            requests.append(kwargs)
            if any(tool.get("type") == "web_search" for tool in kwargs["tools"]):
                raise UnsupportedWebSearchError("unsupported web_search tool")
            return terminal

        async def aclose(self) -> None:
            return None

    fake = ModuleType("openai")
    fake.AsyncOpenAI = Client
    monkeypatch.setitem(sys.modules, "openai", fake)

    model = OpenAIModel(api_key="key", model="gpt-test", max_attempts=1)
    options = model.build_tool_schema_request_options(
        _managed_web_search_schema(),
        delivery="api_parameter",
    )
    chunks = await _collect(
        model,
        [{"role": "user", "content": "Find the vendor advisory"}],
        **options,
    )

    assert requests[0]["tools"] == [{"type": "web_search"}]
    assert requests[1]["tools"][0]["type"] == "function"
    assert requests[1]["tools"][0]["name"] == "web_search"
    assert chunks[-1].tool_calls is not None
    assert chunks[-1].tool_calls[0]["function"]["name"] == "web_search"


@pytest.mark.asyncio
async def test_openai_responses_does_not_fallback_after_published_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []

    class UnsupportedWebSearchError(Exception):
        status_code = 400
        body = {"error": "web_search is not supported by this model"}

    events = _AsyncListStream(
        [{"type": "response.output_text.delta", "delta": "partial"}],
        failure=UnsupportedWebSearchError("unsupported web_search tool"),
    )

    class Client:
        def __init__(self, **_: Any) -> None:
            self.responses = SimpleNamespace(create=self.create)

        async def create(self, **kwargs: Any) -> Any:
            requests.append(kwargs)
            return events

        async def aclose(self) -> None:
            return None

    fake = ModuleType("openai")
    fake.AsyncOpenAI = Client
    monkeypatch.setitem(sys.modules, "openai", fake)

    model = OpenAIModel(api_key="key", model="gpt-test", max_attempts=1)
    options = model.build_tool_schema_request_options(
        _managed_web_search_schema(),
        delivery="api_parameter",
    )

    with pytest.raises(ModelTransportError, match="unsupported web_search"):
        await _collect(
            model,
            [{"role": "user", "content": "Find the vendor advisory"}],
            **options,
        )

    assert len(requests) == 1


def test_qwen_responses_projects_its_native_web_search_tool() -> None:
    model = OpenAICompatibleModel(
        model="qwen3.7-max",
        api_key="key",
        base_url="https://dashscope.example/compatible-mode/v1",
        api_mode="responses",
        provider_name="qwen",
    )

    options = model.build_tool_schema_request_options(
        _managed_web_search_schema(),
        delivery="api_parameter",
    )

    assert options["tools"] == [{"type": "web_search"}]


@pytest.mark.asyncio
async def test_qwen_chat_uses_native_search_for_admitted_web_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    events = _AsyncListStream(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="Current answer",
                            reasoning_content=None,
                            tool_calls=None,
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
        model="qwen3.7-max",
        api_key="key",
        base_url="https://dashscope.example/compatible-mode/v1",
        provider_name="qwen",
        max_attempts=1,
    )
    options = model.build_tool_schema_request_options(
        _managed_web_search_schema(),
        delivery="api_parameter",
    )
    chunks = await _collect(
        model,
        [{"role": "user", "content": "Find current information"}],
        **options,
    )

    assert "tools" not in captured
    assert captured["extra_body"] == {"enable_search": True}
    assert chunks[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_qwen_chat_falls_back_to_managed_search_on_request_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    events = _AsyncListStream(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            reasoning_content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="call_1",
                                    type="function",
                                    function=SimpleNamespace(
                                        name="web_search",
                                        arguments='{"query":"current information"}',
                                    ),
                                )
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=None,
            )
        ]
    )

    class UnsupportedWebSearchError(Exception):
        status_code = 400
        body = {"error": "enable_search is not supported by this model"}

    class Client:
        def __init__(self, **_: Any) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        async def create(self, **kwargs: Any) -> Any:
            requests.append(kwargs)
            if kwargs.get("extra_body", {}).get("enable_search") is True:
                raise UnsupportedWebSearchError("unsupported enable_search")
            return events

        async def aclose(self) -> None:
            return None

    fake = ModuleType("openai")
    fake.AsyncOpenAI = Client
    monkeypatch.setitem(sys.modules, "openai", fake)

    model = OpenAICompatibleModel(
        model="qwen3.7-max",
        api_key="key",
        base_url="https://dashscope.example/compatible-mode/v1",
        provider_name="qwen",
        max_attempts=1,
    )
    options = model.build_tool_schema_request_options(
        _managed_web_search_schema(),
        delivery="api_parameter",
    )
    chunks = await _collect(
        model,
        [{"role": "user", "content": "Find current information"}],
        **options,
    )

    assert requests[0]["extra_body"] == {"enable_search": True}
    assert "extra_body" not in requests[1]
    assert requests[1]["tools"][0]["function"]["name"] == "web_search"
    assert chunks[-1].tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": '{"query":"current information"}',
            },
        }
    ]


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
async def test_responses_continuation_sends_only_verified_canonical_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    streams = [
        _AsyncListStream(
            [
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
                                "role": "assistant",
                                "content": [
                                    {"type": "output_text", "text": "first"}
                                ],
                            }
                        ],
                    },
                }
            ]
        ),
        _AsyncListStream(
            [
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_2",
                        "status": "completed",
                        "model": "gpt-test",
                        "output": [
                            {
                                "type": "message",
                                "id": "msg_2",
                                "role": "assistant",
                                "content": [
                                    {"type": "output_text", "text": "second"}
                                ],
                            }
                        ],
                    },
                }
            ]
        ),
    ]

    class Client:
        def __init__(self, **_: Any) -> None:
            self.responses = SimpleNamespace(create=self.create)

        async def create(self, **kwargs: Any) -> Any:
            requests.append(kwargs)
            return streams.pop(0)

        async def aclose(self) -> None:
            return None

    fake = ModuleType("openai")
    fake.AsyncOpenAI = Client
    monkeypatch.setitem(sys.modules, "openai", fake)

    model = OpenAIModel(api_key="key", model="gpt-test", max_attempts=1)
    prior_messages: list[dict[str, Any]] = []
    for index in range(32):
        prior_messages.extend(
            [
                {"role": "user", "content": f"question-{index}"},
                {
                    "role": "assistant",
                    "content": f"answer-{index}",
                    "native_items": [
                        {
                            "type": "message",
                            "id": f"history-{index}",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": f"answer-{index}",
                                }
                            ],
                        }
                    ],
                },
            ]
        )
    prior_messages.append({"role": "user", "content": "question"})
    first_request = _request_for(
        model,
        prior_messages,
    )
    first = [chunk async for chunk in model.stream(first_request)]
    continuation = first[-1].continuation
    assert continuation is not None

    second_request = _request_for(
        model,
        [
            *prior_messages,
            {
                "role": "assistant",
                "content": "first",
                "native_items": first[-1].native_items,
            },
            {"role": "user", "content": "follow up"},
        ],
        continuation=continuation,
    )
    second = [chunk async for chunk in model.stream(second_request)]

    assert requests[0]["input"] == _to_responses_input(prior_messages)
    assert requests[1]["previous_response_id"] == "resp_1"
    assert requests[1]["input"] == [{"role": "user", "content": "follow up"}]
    assert second[-1].event_metadata["continuation_applied"] is True
    assert second[-1].continuation is not None
    assert second[-1].continuation.response_id == "resp_2"


@pytest.mark.parametrize(
    ("current_messages", "current_options", "expected_reason"),
    [
        (
            [{"role": "user", "content": "compacted summary"}],
            {},
            "canonical_prefix_changed",
        ),
        (
            [{"role": "user", "content": "original evidence"}],
            {"max_output_tokens": 4096},
            "request_settings_changed",
        ),
    ],
)
def test_responses_continuation_rejects_context_or_settings_drift(
    current_messages: list[dict[str, Any]],
    current_options: dict[str, Any],
    expected_reason: str,
) -> None:
    from qitos.core.model_request import ModelContinuation, model_json_digest
    from qitos.models._openai_responses import (
        _apply_continuation,
        _continuation_settings,
        _request_payload,
    )

    model = OpenAIModel(api_key="key", model="gpt-test", max_attempts=1)
    original_messages = [{"role": "user", "content": "original evidence"}]
    original_request = _request_for(model, original_messages)
    original_payload = _request_payload(
        model,
        original_request,
        {"prompt_cache_key": original_request.cache_affinity},
        provider="openai",
    )
    continuation = ModelContinuation(
        run_id=original_request.run_id,
        provider=original_request.provider,
        model=original_request.model,
        protocol=original_request.protocol,
        response_id="resp-original",
        prefix_items=len(original_payload["input"]),
        prefix_digest=model_json_digest(original_payload["input"]),
        settings_digest=model_json_digest(
            _continuation_settings(original_payload)
        ),
    )
    request = _request_for(
        model,
        current_messages,
        continuation=continuation,
        **current_options,
    )
    current_payload = _request_payload(
        model,
        request,
        {
            **request.option_dict(),
            "prompt_cache_key": request.cache_affinity,
        },
        provider="openai",
    )

    outbound, applied, reason = _apply_continuation(request, current_payload)

    assert applied is False
    assert reason == expected_reason
    assert outbound == current_payload
    assert "previous_response_id" not in outbound


@pytest.mark.asyncio
async def test_responses_invalid_continuation_falls_back_to_full_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    terminal_stream = _AsyncListStream(
        [
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_2",
                    "status": "completed",
                    "model": "gpt-test",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_2",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "second"}],
                        }
                    ],
                },
            }
        ]
    )

    class InvalidContinuationError(Exception):
        status_code = 400
        body = {"error": "previous_response_id is invalid or expired"}

    class Client:
        def __init__(self, **_: Any) -> None:
            self.responses = SimpleNamespace(create=self.create)

        async def create(self, **kwargs: Any) -> Any:
            requests.append(kwargs)
            if kwargs.get("previous_response_id"):
                raise InvalidContinuationError(
                    "previous_response_id is invalid or expired"
                )
            return terminal_stream

        async def aclose(self) -> None:
            return None

    fake = ModuleType("openai")
    fake.AsyncOpenAI = Client
    monkeypatch.setitem(sys.modules, "openai", fake)

    model = OpenAIModel(api_key="key", model="gpt-test", max_attempts=1)
    prior_input = [{"role": "user", "content": "question"}]
    prior_output = {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "first"}],
    }
    from qitos.core.model_request import ModelContinuation, model_json_digest
    from qitos.models._openai_responses import _continuation_settings, _request_payload

    base = _request_for(model, prior_input)
    payload = _request_payload(
        model,
        base,
        {"prompt_cache_key": base.cache_affinity},
        provider="openai",
    )
    prefix = list(payload["input"]) + [prior_output]
    continuation = ModelContinuation(
        run_id=base.run_id,
        provider=base.provider,
        model=base.model,
        protocol=base.protocol,
        response_id="resp_1",
        prefix_items=len(prefix),
        prefix_digest=model_json_digest(prefix),
        settings_digest=model_json_digest(_continuation_settings(payload)),
    )
    request = _request_for(
        model,
        [
            *prior_input,
            {"role": "assistant", "content": "first", "native_items": [prior_output]},
            {"role": "user", "content": "follow up"},
        ],
        continuation=continuation,
    )

    chunks = [chunk async for chunk in model.stream(request)]

    assert requests[0]["previous_response_id"] == "resp_1"
    assert "previous_response_id" not in requests[1]
    assert requests[1]["input"] == [
        {"role": "user", "content": "question"},
        prior_output,
        {"role": "user", "content": "follow up"},
    ]
    assert chunks[-1].event_metadata["continuation_fallback"] is True


@pytest.mark.asyncio
async def test_responses_forked_request_discards_parent_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    stream = _AsyncListStream(
        [
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_child",
                    "status": "completed",
                    "model": "gpt-test",
                    "output": [],
                },
            }
        ]
    )

    class Client:
        def __init__(self, **_: Any) -> None:
            self.responses = SimpleNamespace(create=self.create)

        async def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return stream

        async def aclose(self) -> None:
            return None

    fake = ModuleType("openai")
    fake.AsyncOpenAI = Client
    monkeypatch.setitem(sys.modules, "openai", fake)

    model = OpenAIModel(api_key="key", model="gpt-test", max_attempts=1)
    from qitos.core import ModelContinuation

    continuation = ModelContinuation(
        run_id="parent-run",
        provider="openai",
        model="gpt-test",
        protocol="responses",
        response_id="resp_parent",
        prefix_items=1,
        prefix_digest="unused",
        settings_digest="unused",
    )
    request = _request_for(
        model,
        [{"role": "user", "content": "forked"}],
        run_id="child-run",
        continuation=continuation,
    )

    chunks = [chunk async for chunk in model.stream(request)]

    assert "previous_response_id" not in captured
    assert captured["input"] == [{"role": "user", "content": "forked"}]
    assert chunks[-1].event_metadata["continuation_reason"] == (
        "request_identity_changed"
    )


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
async def test_responses_cancellation_closes_stream_and_client_without_retry(
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
            self.responses = SimpleNamespace(create=self.create)

        async def create(self, **_: Any) -> Any:
            nonlocal attempts
            attempts += 1
            return BlockingStream()

        async def aclose(self) -> None:
            closed["client"] = True

    fake = ModuleType("openai")
    fake.AsyncOpenAI = Client
    monkeypatch.setitem(sys.modules, "openai", fake)

    model = OpenAIModel(
        api_key="key",
        model="gpt-test",
        max_attempts=3,
    )
    task = asyncio.create_task(
        _collect(model, [{"role": "user", "content": "question"}])
    )
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert attempts == 1
    assert closed == {"stream": True, "client": True}


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
    assert captured["request"]["prompt_cache_key"] == "openai-test"
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
async def test_chat_retries_without_prompt_cache_key_when_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    events = _AsyncListStream(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="done",
                            reasoning_content=None,
                            tool_calls=None,
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            )
        ]
    )

    class UnsupportedCacheKeyError(RuntimeError):
        status_code = 400
        body = {"error": "unknown parameter prompt_cache_key"}

    class Client:
        def __init__(self, **_: Any) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        async def create(self, **kwargs: Any) -> Any:
            requests.append(dict(kwargs))
            if len(requests) == 1:
                raise UnsupportedCacheKeyError(
                    "prompt_cache_key is not supported"
                )
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
        max_attempts=1,
    )
    chunks = await _collect(model, [{"role": "user", "content": "question"}])

    assert chunks[-1].done is True
    assert requests[0]["prompt_cache_key"] == "openai-test"
    assert "prompt_cache_key" not in requests[1]


def test_chat_output_limit_does_not_publish_partial_tool_calls() -> None:
    accumulator = ChatStreamAccumulator(provider="compatible", model="model")
    call_id = f"call-{time.monotonic_ns()}"
    accumulator.consume(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="",
                        reasoning_content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id=call_id,
                                type="function",
                                function=SimpleNamespace(
                                    name="lookup",
                                    arguments='{"query":',
                                ),
                            )
                        ],
                    ),
                    finish_reason="length",
                )
            ],
            usage=None,
        )
    )

    terminal = accumulator.complete()

    assert terminal.done is True
    assert terminal.tool_calls is None
    invalid = terminal.event_metadata["invalid_tool_calls"]
    assert invalid[0]["call_id"] == call_id
    assert invalid[0]["code"] == "tool_call_unexpected_finish_reason"


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
