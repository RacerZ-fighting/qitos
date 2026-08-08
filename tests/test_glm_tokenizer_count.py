from __future__ import annotations

from qitos.models.openai import OpenAICompatibleModel


class _FakeTokenizer:
    def __init__(self):
        self.last_messages = None
        self.last_encoded_text = None

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        assert tokenize is True
        assert add_generation_prompt is False
        self.last_messages = list(messages)
        return {"input_ids": list(range(37))}

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        self.last_encoded_text = str(text)
        return list(range(len(str(text).split())))


def test_glm_openai_compatible_model_uses_local_glm_tokenizer(monkeypatch):
    tokenizer = _FakeTokenizer()
    monkeypatch.setattr("qitos.models.openai._glm_tokenizer_path", lambda: "/tmp/glm-tokenizer")
    monkeypatch.setattr("qitos.models.openai._load_glm_tokenizer", lambda path: tokenizer)

    model = OpenAICompatibleModel(
        model="GLM-5.1",
        base_url="http://localhost/v1",
    )

    count = model.count_tokens(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello", "tool_calls": [{"name": "x"}]},
        ]
    )

    assert count == 37
    assert tokenizer.last_messages[1]["role"] == "user"
    assert "tool_calls" in tokenizer.last_messages[1]["content"]


def test_non_glm_model_keeps_default_token_estimate(monkeypatch):
    def _boom(_path):
        raise AssertionError("tokenizer should not load for non-GLM models")

    monkeypatch.setattr("qitos.models.openai._glm_tokenizer_path", lambda: "/tmp/glm-tokenizer")
    monkeypatch.setattr("qitos.models.openai._load_glm_tokenizer", _boom)

    model = OpenAICompatibleModel(
        model="qwen-plus",
        base_url="http://localhost/v1",
    )

    assert model.count_tokens("hello world") == 2


def test_glm_responses_count_serializes_native_transaction_fields(monkeypatch):
    tokenizer = _FakeTokenizer()
    monkeypatch.setattr(
        "qitos.models.openai._glm_tokenizer_path", lambda: "/tmp/glm-tokenizer"
    )
    monkeypatch.setattr(
        "qitos.models.openai._load_glm_tokenizer", lambda path: tokenizer
    )
    model = OpenAICompatibleModel(
        model="GLM-5.1",
        base_url="http://localhost/v1",
        api_mode="responses",
    )

    count = model.count_request_tokens(
        [
            {
                "role": "assistant",
                "content": None,
                "native_items": [
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "inspect"}],
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_native",
                        "name": "lookup",
                        "arguments": '{"key":"target"}',
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_native",
                "content": "verified output",
            },
        ]
    )

    assert isinstance(count, int) and count > 0
    encoded = tokenizer.last_encoded_text or ""
    assert '"type":"reasoning"' in encoded
    assert '"type":"function_call"' in encoded
    assert '"call_id":"call_native"' in encoded
    assert '"arguments":"{\\"key\\":\\"target\\"}"' in encoded
    assert '"type":"function_call_output"' in encoded
    assert '"output":"verified output"' in encoded
