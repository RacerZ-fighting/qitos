from __future__ import annotations

import base64
import sys
import urllib.error
from types import ModuleType
from types import SimpleNamespace

import pytest
import requests

from qitos.models import (
    AnthropicModel,
    GeminiModel,
    LiteLLMModel,
    LMStudioModel,
    ModelFactory,
    OpenAICompatibleModel,
    OllamaGenerateModel,
    OllamaModel,
    VLLMModel,
    infer_context_window,
)


class _FakeHTTPResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            error = requests.HTTPError(f"status={self.status_code}")
            error.response = self
            raise error


class _FakeStreamHTTPResponse(_FakeHTTPResponse):
    def __init__(self, events: list[dict[str, object]]) -> None:
        super().__init__({})
        self._events = events

    def iter_lines(self, decode_unicode: bool = False):
        _ = decode_unicode
        for event in self._events:
            import json

            yield f"data: {json.dumps(event)}"


def test_anthropic_native_messages_adapter(monkeypatch) -> None:
    captured = {}

    # Ensure the default Anthropic endpoint is used, even if ANTHROPIC_BASE_URL
    # is set in the environment (e.g., a corporate proxy).
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeHTTPResponse(
            {
                "content": [
                    {"type": "text", "text": "Final Answer: native anthropic works"}
                ],
                "usage": {
                    "input_tokens": 100,
                    "cache_read_input_tokens": 3,
                    "output_tokens": 22,
                },
            }
        )

    monkeypatch.setattr("qitos.models.anthropic.requests.post", fake_post)
    llm = AnthropicModel(api_key="anthropic-test", model="claude-test")
    out = llm(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Say hello."},
        ]
    )

    assert out == "Final Answer: native anthropic works"
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "anthropic-test"
    assert captured["json"]["system"] == "You are helpful."
    assert captured["json"]["messages"] == [{"role": "user", "content": "Say hello."}]
    assert llm.extract_usage() == {
        "prompt_tokens": 103,
        "completion_tokens": 22,
        "total_tokens": 125,
    }


def test_anthropic_stream_preserves_reasoning_tool_delta_and_finish_reason(
    monkeypatch,
) -> None:
    events = [
        {
            "type": "message_start",
            "message": {"usage": {"input_tokens": 8}},
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "lookup",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"q":"x"}'},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "thinking_delta", "thinking": "check evidence"},
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 5},
        },
        {"type": "message_stop"},
    ]

    def fake_post(*args, **kwargs):
        _ = args, kwargs
        return _FakeStreamHTTPResponse(events)

    monkeypatch.setattr("qitos.models.anthropic.requests.post", fake_post)
    model = AnthropicModel(api_key="anthropic-test", model="claude-test")

    chunks = list(model.stream([{"role": "user", "content": "look up x"}]))

    tool_delta = next(chunk for chunk in chunks if chunk.event_type == "tool_call.delta")
    assert tool_delta.event_metadata == {
        "index": 0,
        "call_id": "toolu_1",
        "name": "lookup",
        "arguments_delta": '{"q":"x"}',
    }
    assert "".join(chunk.reasoning_content or "" for chunk in chunks) == "check evidence"
    terminal = [chunk for chunk in chunks if chunk.done]
    assert len(terminal) == 1
    assert terminal[0].finish_reason == "tool_use"
    assert terminal[0].tool_calls == [
        {
            "id": "toolu_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"q":"x"}'},
        }
    ]
    assert terminal[0].usage == {
        "prompt_tokens": 8,
        "completion_tokens": 5,
        "total_tokens": 13,
    }


def test_gemini_native_adapter(monkeypatch) -> None:
    captured = {}

    def fake_post(url, params=None, json=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["json"] = json
        return _FakeHTTPResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": 'Action: search(query="gemini")'},
                            ]
                        }
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 18,
                    "candidatesTokenCount": 7,
                    "totalTokenCount": 25,
                },
            }
        )

    monkeypatch.setattr("qitos.models.gemini.requests.post", fake_post)
    llm = GeminiModel(api_key="gemini-test", model="gemini-2.5-flash")
    out = llm(
        [
            {"role": "system", "content": "Follow protocol."},
            {"role": "user", "content": "Search for docs."},
            {"role": "assistant", "content": "Thinking."},
        ]
    )

    assert out == 'Action: search(query="gemini")'
    assert captured["params"] == {"key": "gemini-test"}
    assert (
        captured["json"]["systemInstruction"]["parts"][0]["text"] == "Follow protocol."
    )
    assert captured["json"]["contents"][0]["role"] == "user"
    assert captured["json"]["contents"][1]["role"] == "model"
    assert llm.extract_usage() == {
        "prompt_tokens": 18,
        "completion_tokens": 7,
        "total_tokens": 25,
    }


def test_litellm_adapter_and_usage(monkeypatch) -> None:
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return {
            "choices": [{"message": {"content": "Final Answer: litellm works"}}],
            "usage": {"prompt_tokens": 13, "completion_tokens": 5, "total_tokens": 18},
        }

    monkeypatch.setitem(
        sys.modules, "litellm", SimpleNamespace(completion=fake_completion)
    )
    llm = LiteLLMModel(model="anthropic/claude-3-5-sonnet-latest", api_key="lite-key")
    out = llm([{"role": "user", "content": "Say hi"}])

    assert out == "Final Answer: litellm works"
    assert captured["model"] == "anthropic/claude-3-5-sonnet-latest"
    assert captured["api_key"] == "lite-key"
    assert llm.extract_usage() == {
        "prompt_tokens": 13,
        "completion_tokens": 5,
        "total_tokens": 18,
    }


@pytest.mark.parametrize(
    ("post_target", "model"),
    [
        (
            "qitos.models.anthropic.requests.post",
            AnthropicModel(api_key="anthropic-test"),
        ),
        (
            "qitos.models.gemini.requests.post",
            GeminiModel(api_key="gemini-test"),
        ),
    ],
)
def test_native_provider_transport_errors_are_not_model_text(
    monkeypatch, post_target, model
) -> None:
    def fail_request(*args, **kwargs):
        _ = args, kwargs
        raise requests.ConnectionError("provider unavailable")

    monkeypatch.setattr(post_target, fail_request)

    with pytest.raises(requests.ConnectionError, match="provider unavailable"):
        model([{"role": "user", "content": "hello"}])


def test_litellm_transport_errors_are_not_model_text(monkeypatch) -> None:
    def fail_completion(**kwargs):
        _ = kwargs
        raise RuntimeError("provider unavailable")

    monkeypatch.setitem(
        sys.modules, "litellm", SimpleNamespace(completion=fail_completion)
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        LiteLLMModel(model="test")([{"role": "user", "content": "hello"}])


@pytest.mark.parametrize(
    "model",
    [
        OllamaModel(model="test"),
        OllamaGenerateModel(model="test"),
        LMStudioModel(model="test"),
        VLLMModel(model="test"),
    ],
)
def test_local_provider_transport_errors_are_not_model_text(monkeypatch, model) -> None:
    def fail_request(*args, **kwargs):
        _ = args, kwargs
        raise urllib.error.URLError("provider unavailable")

    monkeypatch.setattr("urllib.request.urlopen", fail_request)

    with pytest.raises(urllib.error.URLError, match="provider unavailable"):
        model([{"role": "user", "content": "hello"}])


def test_model_factory_from_env_supports_new_providers(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("LM_STUDIO_BASE_URL", raising=False)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-env")
    llm = ModelFactory.from_env(model="claude-test")
    assert isinstance(llm, AnthropicModel)

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-env")
    llm = ModelFactory.from_env(model="gemini-test")
    assert isinstance(llm, GeminiModel)

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("LITELLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("LITELLM_API_KEY", "lite-env")
    llm = ModelFactory.from_env()
    assert isinstance(llm, LiteLLMModel)
    assert llm.model == "openai/gpt-4o-mini"

    monkeypatch.setenv("QITOS_MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    llm = ModelFactory.from_env(model="llama3.1")
    assert isinstance(llm, OllamaModel)

    monkeypatch.setenv("QITOS_MODEL_PROVIDER", "lmstudio")
    monkeypatch.setenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    llm = ModelFactory.from_env(model="local-model")
    assert isinstance(llm, LMStudioModel)


def test_local_openai_compatible_like_parsing_supports_tool_calls() -> None:
    lmstudio = LMStudioModel(model="local-model")
    out = lmstudio._parse_response(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "search",
                                    "arguments": '{"query": "lmstudio"}',
                                }
                            }
                        ]
                    }
                }
            ]
        }
    )
    assert out == 'Action: search(query="lmstudio")'

    ollama = OllamaModel(model="llama3.1")
    out = ollama._parse_response(
        {
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "grep_files",
                            "arguments": {"pattern": "TODO"},
                        }
                    }
                ]
            }
        }
    )
    assert out == 'Action: grep_files(pattern="TODO")'


def test_context_registry_infers_anthropic_and_gemini_windows() -> None:
    assert infer_context_window("claude-3-5-sonnet-latest") == 200_000
    assert infer_context_window("gemini-2.5-flash") == 1_048_576


def test_openai_compatible_model_formats_multimodal_chat_messages(
    tmp_path, monkeypatch
) -> None:
    captured = {}
    png_path = tmp_path / "shot.png"
    png_path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wn2gbcAAAAASUVORK5CYII="
        )
    )

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="Final Answer: visual ok", tool_calls=None
                        )
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=11, completion_tokens=5, total_tokens=16
                ),
            )

    class _FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.chat = SimpleNamespace(completions=_FakeCompletions())

    fake_openai = SimpleNamespace(
        OpenAI=lambda **kwargs: _FakeClient(**kwargs), APIError=Exception
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    llm = OpenAICompatibleModel(
        model="gpt-4.1-mini",
        api_key="test-key",
        base_url="https://example.test/v1",
    )
    out = llm(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect this screenshot."},
                    {"type": "image_file", "path": str(png_path), "detail": "high"},
                ],
            }
        ]
    )

    assert out == "Final Answer: visual ok"
    message = captured["messages"][0]
    assert message["role"] == "user"
    assert isinstance(message["content"], list)
    assert message["content"][0] == {"type": "text", "text": "Inspect this screenshot."}
    image_block = message["content"][1]
    assert image_block["type"] == "image_url"
    assert image_block["image_url"]["detail"] == "high"
    assert image_block["image_url"]["url"].startswith("data:image/png;base64,")


def test_openai_compatible_model_retries_and_uses_120s_timeout(monkeypatch) -> None:
    captured = {"attempts": 0, "client_kwargs": None}

    class _FakeCompletions:
        def create(self, **kwargs):
            captured["attempts"] += 1
            if captured["attempts"] < 2:
                raise TimeoutError("request timed out")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="Final Answer: retried ok", tool_calls=None
                        )
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=9, completion_tokens=4, total_tokens=13
                ),
            )

    class _FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.chat = SimpleNamespace(completions=_FakeCompletions())

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = lambda **kwargs: _FakeClient(**kwargs)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setattr("qitos.models._request_runtime.time.sleep", lambda _: None)

    llm = OpenAICompatibleModel(
        model="gpt-4.1-mini",
        api_key="test-key",
        base_url="https://example.test/v1",
    )
    out = llm([{"role": "user", "content": "Retry please"}])

    assert out == "Final Answer: retried ok"
    assert captured["attempts"] == 2
    assert captured["client_kwargs"]["timeout"] == 120
    assert captured["client_kwargs"]["max_retries"] == 0
    assert llm.timeout == 120


def test_openai_compatible_model_call_raw_retries_on_transient_errors(
    monkeypatch,
) -> None:
    captured = {"attempts": 0}

    class APIError(Exception):
        pass

    class _FakeCompletions:
        def create(self, **kwargs):
            captured["attempts"] += 1
            if captured["attempts"] < 2:
                raise APIError("stream timeout")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="Final Answer: raw retried ok", tool_calls=None
                        )
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=7, completion_tokens=3, total_tokens=10
                ),
            )

    class _FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=_FakeCompletions())

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = lambda **kwargs: _FakeClient(**kwargs)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setattr("qitos.models._request_runtime.time.sleep", lambda _: None)

    llm = OpenAICompatibleModel(
        model="gpt-4.1-mini",
        api_key="test-key",
        base_url="https://example.test/v1",
    )
    response = llm.call_raw([{"role": "user", "content": "Retry raw please"}])

    assert captured["attempts"] == 2
    assert response.choices[0].message.content == "Final Answer: raw retried ok"


def test_openai_compatible_model_does_not_retry_bad_requests(monkeypatch) -> None:
    from qitos.core import ModelTransportError

    captured = {"attempts": 0}

    class _BadRequest(Exception):
        status_code = 400

    class _FakeCompletions:
        def create(self, **kwargs):
            captured["attempts"] += 1
            raise _BadRequest("invalid request")

    class _FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=_FakeCompletions())

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = lambda **kwargs: _FakeClient(**kwargs)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    llm = OpenAICompatibleModel(
        model="gpt-4.1-mini",
        api_key="test-key",
        base_url="https://example.test/v1",
    )
    with pytest.raises(ModelTransportError) as exc_info:
        llm.call_raw([{"role": "user", "content": "bad request"}])

    assert captured["attempts"] == 1
    assert exc_info.value.attempts == 1
    assert exc_info.value.retryable is False
    assert exc_info.value.status_code == 400


def test_async_openai_compatible_stream_retries_and_preserves_errors(
    monkeypatch,
) -> None:
    import asyncio

    from qitos.core import ModelTransportError
    from qitos.models.openai import AsyncOpenAICompatibleModel

    attempts = 0
    client_kwargs: dict[str, object] = {}

    class _UsageThenTimeoutStream:
        def __init__(self) -> None:
            self._sent_usage = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._sent_usage:
                self._sent_usage = True
                return SimpleNamespace(
                    choices=[],
                    usage=SimpleNamespace(
                        prompt_tokens=999,
                        completion_tokens=0,
                        total_tokens=999,
                    ),
                )
            raise TimeoutError("request timed out")

    class _Stream:
        def __init__(self) -> None:
            self._chunks = iter(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    content='{"ok":true}',
                                    tool_calls=[
                                        SimpleNamespace(
                                            index=0,
                                            id="call_1",
                                            type="function",
                                            function=SimpleNamespace(
                                                name="lookup", arguments='{"q":'
                                            ),
                                        )
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
                                    tool_calls=[
                                        SimpleNamespace(
                                            index=0,
                                            id=None,
                                            type=None,
                                            function=SimpleNamespace(
                                                name=None, arguments='"x"}'
                                            ),
                                        )
                                    ],
                                ),
                                finish_reason="tool_calls",
                            )
                        ],
                        usage=SimpleNamespace(
                            prompt_tokens=3,
                            completion_tokens=2,
                            total_tokens=5,
                        ),
                    ),
                ]
            )

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._chunks)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    class _Completions:
        async def create(self, **kwargs):
            nonlocal attempts
            attempts += 1
            assert kwargs["stream"] is True
            if attempts == 1:
                return _UsageThenTimeoutStream()
            return _Stream()

    class _Client:
        def __init__(self, **kwargs):
            nonlocal client_kwargs
            client_kwargs = kwargs
            self.chat = SimpleNamespace(completions=_Completions())

    fake_openai = ModuleType("openai")
    fake_openai.AsyncOpenAI = lambda **kwargs: _Client(**kwargs)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    async def _no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("qitos.models._request_runtime.asyncio.sleep", _no_sleep)
    model = AsyncOpenAICompatibleModel(
        model="test-model",
        api_key="test-key",
        base_url="https://example.test/v1",
        max_attempts=2,
    )

    async def _collect():
        return [
            chunk async for chunk in model.astream([{"role": "user", "content": "go"}])
        ]

    chunks = asyncio.run(_collect())

    assert attempts == 2
    assert client_kwargs["max_retries"] == 0
    assert "".join(chunk.text for chunk in chunks) == '{"ok":true}'
    assert [
        chunk.event_metadata["arguments_delta"]
        for chunk in chunks
        if chunk.event_type == "tool_call.delta"
    ] == ['{"q":', '"x"}']
    assert chunks[-1].done is True
    assert chunks[-1].finish_reason == "tool_calls"
    assert chunks[-1].tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"q":"x"}'},
        }
    ]
    assert model.extract_usage() == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
        "cached_tokens": None,
    }

    class _BadRequest(Exception):
        status_code = 400

    class _FailingCompletions:
        async def create(self, **kwargs):
            _ = kwargs
            raise _BadRequest("invalid request")

    fake_openai.AsyncOpenAI = lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=_FailingCompletions())
    )
    failing = AsyncOpenAICompatibleModel(
        model="test-model",
        api_key="test-key",
        base_url="https://example.test/v1",
        max_attempts=2,
    )

    async def _drain_failing():
        return [
            chunk
            async for chunk in failing.astream(
                [{"role": "user", "content": "bad request"}]
            )
        ]

    with pytest.raises(ModelTransportError) as exc_info:
        asyncio.run(_drain_failing())

    assert exc_info.value.status_code == 400
    assert exc_info.value.attempts == 1


def test_openai_compatible_model_disables_thinking_for_forced_tool_choice(
    monkeypatch,
) -> None:
    captured: list[dict[str, object]] = []

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.append(dict(kwargs))
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="ok", tool_calls=None)
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=1, completion_tokens=1, total_tokens=2
                ),
            )

    class _FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=_FakeCompletions())

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = lambda **kwargs: _FakeClient(**kwargs)
    fake_openai.APIError = Exception
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    llm = OpenAICompatibleModel(
        model="qwen3.7-max",
        api_key="test-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    llm.call_raw(
        [{"role": "user", "content": "Call a tool"}],
        tools=[{"type": "function", "function": {"name": "greet"}}],
        tool_choice="required",
        reasoning_effort="max",
        extra_body={"enable_thinking": True},
    )
    llm.call_raw(
        [{"role": "user", "content": "Maybe call a tool"}],
        tools=[{"type": "function", "function": {"name": "greet"}}],
        tool_choice="auto",
        reasoning_effort="max",
        extra_body={"enable_thinking": True},
    )
    llm.call_raw(
        [{"role": "user", "content": "Call greet"}],
        tools=[{"type": "function", "function": {"name": "greet"}}],
        tool_choice={"type": "function", "function": {"name": "greet"}},
        reasoning_effort="max",
        extra_body={"thinking": {"type": "enabled"}},
    )

    assert captured[0]["extra_body"] == {"enable_thinking": False}
    assert "reasoning_effort" not in captured[0]
    assert captured[1]["extra_body"] == {"enable_thinking": True}
    assert captured[1]["reasoning_effort"] == "max"
    assert captured[2]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in captured[2]


def test_explicit_provider_override_wins(monkeypatch) -> None:
    monkeypatch.setenv("QITOS_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-env")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-env")
    llm = ModelFactory.from_env(model="claude-test")
    assert isinstance(llm, AnthropicModel)

    monkeypatch.setenv("QITOS_MODEL_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-env")
    llm = ModelFactory.from_env(model="gemini-test")
    assert isinstance(llm, GeminiModel)

    monkeypatch.delenv("QITOS_MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "LITELLM_MODEL",
        "LITELLM_API_KEY",
        "OLLAMA_HOST",
        "OLLAMA_BASE_URL",
        "LM_STUDIO_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    assert ModelFactory.from_env() is None
