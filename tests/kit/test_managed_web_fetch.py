"""Managed provider web-fetch capability tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from qitos.kit.fetch import (
    KimiWebFetchCapability,
    ManagedWebFetchTool,
    WebFetchError,
    build_web_fetch_capability,
)

_FETCH_URL = "https://fetch.example.test/v1/fetch"


@dataclass
class _Response:
    status_code: int
    body: bytes = b""
    headers: dict[str, str] = field(
        default_factory=lambda: {"Content-Type": "text/markdown"}
    )
    encoding: str | None = "utf-8"
    closed: bool = False

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return [
            self.body[offset : offset + chunk_size]
            for offset in range(0, len(self.body), chunk_size)
        ]

    def close(self) -> None:
        self.closed = True


@dataclass
class _Session:
    response: _Response
    calls: list[dict[str, Any]] = field(default_factory=list)

    def post(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append({"url": url, **kwargs})
        return self.response


def test_kimi_fetch_uses_managed_contract_and_returns_bounded_markdown() -> None:
    response = _Response(200, b"# Vendor advisory\n\nAffected versions.")
    session = _Session(response)
    capability = KimiWebFetchCapability(
        api_key="secret",
        fetch_url=_FETCH_URL,
        session=session,  # type: ignore[arg-type]
    )

    result = capability.fetch("https://vendor.example/advisory")

    assert result.content == "# Vendor advisory\n\nAffected versions."
    assert result.url == "https://vendor.example/advisory"
    assert result.truncated is False
    assert response.closed is True
    assert session.calls == [
        {
            "url": _FETCH_URL,
            "headers": {
                "Authorization": "Bearer secret",
                "Accept": "text/markdown",
                "Content-Type": "application/json",
            },
            "json": {"url": "https://vendor.example/advisory"},
            "timeout": 30.0,
            "stream": True,
        }
    ]


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (401, "authentication"),
        (402, "billing"),
        (403, "authentication"),
        (429, "rate_limited"),
        (503, "provider"),
    ],
)
def test_kimi_fetch_preserves_failure_kind(status: int, kind: str) -> None:
    capability = KimiWebFetchCapability(
        api_key="secret",
        fetch_url=_FETCH_URL,
        session=_Session(_Response(status)),  # type: ignore[arg-type]
    )

    with pytest.raises(WebFetchError) as error:
        capability.fetch("https://vendor.example/advisory")

    assert error.value.kind == kind


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://localhost/admin",
        "http://127.0.0.1/admin",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/admin",
        "https://user:password@example.com/",
    ],
)
def test_kimi_fetch_rejects_non_public_urls_before_provider_call(url: str) -> None:
    session = _Session(_Response(200, b"unused"))
    capability = KimiWebFetchCapability(
        api_key="secret",
        fetch_url=_FETCH_URL,
        session=session,  # type: ignore[arg-type]
    )

    with pytest.raises((PermissionError, ValueError)):
        capability.fetch(url)

    assert session.calls == []


def test_factory_and_managed_tool_keep_provider_contract_separate() -> None:
    assert build_web_fetch_capability(provider="unknown", api_key="secret") is None
    assert build_web_fetch_capability(provider="kimi", api_key="secret") is None
    capability = KimiWebFetchCapability(
        api_key="secret",
        fetch_url=_FETCH_URL,
        session=_Session(_Response(200, b"Extracted documentation.")),  # type: ignore[arg-type]
    )
    tool = ManagedWebFetchTool(capability)

    result = tool.execute({"url": "https://docs.example/reference"})

    assert tool.spec.read_only is True
    assert tool.spec.concurrency_safe is True
    assert result == {
        "url": "https://docs.example/reference",
        "content": "Extracted documentation.",
        "content_type": "text/markdown",
        "truncated": False,
    }
