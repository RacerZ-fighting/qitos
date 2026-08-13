"""Managed provider web-search capability tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from qitos.kit.search import (
    KimiBuiltinWebSearchCapability,
    KimiWebSearchCapability,
    ManagedWebSearchTool,
    WebSearchError,
    build_web_search_capability,
)


@dataclass
class _Response:
    status_code: int
    payload: Any

    def json(self) -> Any:
        if isinstance(self.payload, ValueError):
            raise self.payload
        return self.payload


@dataclass
class _Client:
    response: _Response
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def post(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append({"url": url, **kwargs})
        return self.response


@pytest.mark.asyncio
async def test_kimi_search_uses_managed_contract_and_bounds_results() -> None:
    client = _Client(
        _Response(
            200,
            {
                "search_results": [
                    {
                        "title": "Vendor advisory",
                        "url": "https://vendor.example/advisory",
                        "snippet": "Affected versions and remediation.",
                        "site_name": "Vendor",
                        "date": "2026-08-04",
                    },
                    {
                        "title": "Research",
                        "url": "https://research.example/post",
                        "snippet": "Technical details.",
                    },
                ]
            },
        )
    )
    capability = KimiWebSearchCapability(
        api_key="secret",
        search_url="https://api.example/search",
        client=client,
    )

    result = await capability.search("product 1.2 CVE", max_results=1)

    assert [source.title for source in result.sources] == ["Vendor advisory"]
    assert client.calls[0]["url"] == "https://api.example/search"
    assert client.calls[0]["json"] == {"text_query": "product 1.2 CVE"}
    assert client.calls[0]["headers"]["Authorization"] == "Bearer secret"


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (401, "authentication"),
        (403, "authentication"),
        (429, "rate_limited"),
        (503, "provider"),
    ],
)
@pytest.mark.asyncio
async def test_kimi_search_preserves_failure_kind(status: int, kind: str) -> None:
    capability = KimiWebSearchCapability(
        api_key="secret",
        client=_Client(_Response(status, {})),
    )

    with pytest.raises(WebSearchError) as error:
        await capability.search("query")

    assert error.value.kind == kind


@pytest.mark.asyncio
async def test_kimi_search_rejects_invalid_protocol_response() -> None:
    capability = KimiWebSearchCapability(
        api_key="secret",
        client=_Client(_Response(200, {"search_results": "wrong"})),
    )

    with pytest.raises(WebSearchError) as error:
        await capability.search("query")

    assert error.value.kind == "protocol"


def test_factory_is_extensible_without_faking_unsupported_search() -> None:
    assert build_web_search_capability(provider="unknown", api_key="secret") is None
    assert isinstance(
        build_web_search_capability(
            provider="kimi",
            api_key="secret",
            base_url="https://api.kimi.com/coding/v1",
        ),
        KimiBuiltinWebSearchCapability,
    )
    assert isinstance(
        build_web_search_capability(
            provider="kimi",
            api_key="secret",
            base_url="https://api.moonshot.cn/v1",
            model="kimi-k3",
        ),
        KimiBuiltinWebSearchCapability,
    )
    assert isinstance(
        build_web_search_capability(
            provider="kimi",
            api_key="search-secret",
            search_url="https://search.example/v1/search",
            base_url="https://api.kimi.com/coding/v1",
        ),
        KimiWebSearchCapability,
    )


@dataclass
class _Completions:
    responses: list[Any]
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_kimi_builtin_search_round_trips_server_arguments() -> None:
    arguments = (
        '{"results":[{"title":"Advisory","url":"https://vendor.example/a",'
        '"snippet":"Affected versions"}],"usage":{"total_tokens":42}}'
    )
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="$web_search", arguments=arguments),
    )
    first_message = SimpleNamespace(
        content=None,
        reasoning_content="The query needs current sources.",
        tool_calls=[tool_call],
    )
    second_message = SimpleNamespace(
        content="Current details: https://vendor.example/a",
        tool_calls=[],
    )
    completions = _Completions(
        [
            SimpleNamespace(choices=[SimpleNamespace(message=first_message)]),
            SimpleNamespace(choices=[SimpleNamespace(message=second_message)]),
        ]
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    capability = KimiBuiltinWebSearchCapability(
        api_key="secret",
        client=client,
    )

    result = await capability.search("product advisory")

    assert result.text == "Current details: https://vendor.example/a"
    assert [source.url for source in result.sources] == ["https://vendor.example/a"]
    assert completions.calls[0]["tools"] == [
        {
            "type": "builtin_function",
            "function": {"name": "$web_search"},
        }
    ]
    assert completions.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert completions.calls[1]["messages"][-2]["tool_calls"][0]["type"] == "function"
    assert completions.calls[1]["messages"][-2]["reasoning_content"] == (
        "The query needs current sources."
    )
    assert completions.calls[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "name": "$web_search",
        "content": arguments,
    }


@pytest.mark.asyncio
async def test_kimi_builtin_search_keeps_sources_when_final_text_is_empty() -> None:
    arguments = (
        '{"results":[{"title":"Advisory","url":"https://vendor.example/a",'
        '"snippet":"Affected versions"}]}'
    )
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="$web_search", arguments=arguments),
    )
    completions = _Completions(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            reasoning_content=None,
                            tool_calls=[tool_call],
                        )
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=None, tool_calls=[])
                    )
                ]
            ),
        ]
    )
    capability = KimiBuiltinWebSearchCapability(
        api_key="secret",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    result = await capability.search("product advisory")

    assert result.text == (
        "Kimi returned public web search sources without a synthesized answer."
    )
    assert [source.url for source in result.sources] == ["https://vendor.example/a"]


@pytest.mark.asyncio
async def test_managed_tool_is_read_only_and_returns_structured_sources() -> None:
    capability = KimiWebSearchCapability(
        api_key="secret",
        client=_Client(
            _Response(
                200,
                {
                    "search_results": [
                        {
                            "title": "Documentation",
                            "url": "https://docs.example/",
                            "snippet": "Reference.",
                        }
                    ]
                },
            )
        ),
    )
    tool = ManagedWebSearchTool(capability)

    result = await tool.execute({"query": "docs"})

    assert tool.spec.read_only is True
    assert tool.spec.concurrency_safe is True
    assert result["sources"] == [
        {
            "title": "Documentation",
            "url": "https://docs.example/",
            "snippet": "Reference.",
        }
    ]
