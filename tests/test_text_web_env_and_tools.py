from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from qitos.core.tool_result import ToolResult
from qitos.kit.env import TextWebBrowserOps, TextWebEnv
from qitos.kit.tool import FindInPage, FindNext, PageDown, PageUp


def test_text_web_env_follows_redirects_and_keeps_final_url() -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/page")
                self.end_headers()
                return
            if self.path == "/page":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    b"<html><title>Final page</title><body>redirected content</body></html>"
                )
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            _ = (format, args)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        env = TextWebEnv(workspace_root=".")
        ops = env.get_ops("web_browser")
        result = ops.visit(f"{endpoint}/redirect")

        assert result["status"] == "success"
        assert ops.state.url == f"{endpoint}/page"
        assert ops.state.title == "Final page"
        assert any("redirected content" in line for line in ops.state.lines)
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_text_web_env_exposes_web_browser_ops():
    env = TextWebEnv(workspace_root=".")
    env.reset()
    ops = env.get_ops("web_browser")
    assert ops is not None
    summary = ops.summary()
    assert "active_url" in summary


def test_text_web_search_parses_nested_result_links(monkeypatch) -> None:
    response_html = """
    <html><body>
      <a class="result__a featured" href="https://example.com/result">
        Maintained <strong>parser &amp; result</strong>
      </a>
      <a class="other" href="https://example.com/ignored">Ignored</a>
    </body></html>
    """

    class Response:
        text = response_html

    monkeypatch.setattr(
        "qitos.kit.env.text_web_env.httpx.get",
        lambda *args, **kwargs: Response(),
    )

    result = TextWebBrowserOps().search("parser")

    assert result["status"] == "success"
    assert result["results"] == [
        {
            "title": "Maintained parser & result",
            "url": "https://example.com/result",
        }
    ]


@pytest.mark.asyncio
async def test_text_web_atomic_tools_use_ops_context():
    env = TextWebEnv(workspace_root=".")
    env.reset()
    ops = env.get_ops("web_browser")
    ops.state.lines = [f"line {i}" for i in range(120)]  # type: ignore[attr-defined]
    ops.state.url = "https://example.com"  # type: ignore[attr-defined]
    ops.state.title = "Example"  # type: ignore[attr-defined]

    ctx = {"ops": {"web_browser": ops}, "env": env}
    down = await PageDown().execute({"lines": 20}, runtime_context=ctx)
    assert down["status"] == "success"
    assert down["line_start"] == 20

    up = await PageUp().execute({"lines": 10}, runtime_context=ctx)
    assert up["status"] == "success"
    assert up["line_start"] == 10

    find = await FindInPage().execute({"keyword": "line 42"}, runtime_context=ctx)
    assert find["status"] == "success"
    assert find["matched_line"] == 42

    next_match = await FindNext().execute({}, runtime_context=ctx)
    assert isinstance(next_match, ToolResult)
    assert next_match.status == "error"
