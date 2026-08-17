from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from qitos.core.agent_events import ToolExecutionEnd
from qitos.core.message import ToolCall
from qitos.core.tool_executor import ToolBatchExecutor, ToolExecutionConfig
from qitos.core.tool_registry import ToolRegistry
from qitos.core.tool_result import ToolResult
from qitos.kit.tool.epub import EpubToolSet


def test_epub_reader_uses_html_parser_for_chapter_text(tmp_path: Path) -> None:
    book = tmp_path / "book.epub"
    with zipfile.ZipFile(book, "w") as archive:
        archive.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0"?>
            <container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
              <rootfiles>
                <rootfile full-path="content.opf" />
              </rootfiles>
            </container>
            """,
        )
        archive.writestr(
            "content.opf",
            """<?xml version="1.0"?>
            <package xmlns="http://www.idpf.org/2007/opf">
              <manifest><item id="chapter" href="chapter.xhtml" /></manifest>
              <spine><itemref idref="chapter" /></spine>
            </package>
            """,
        )
        archive.writestr(
            "chapter.xhtml",
            """<html><head><title>One &amp; Chapter</title></head>
            <body><script>ignore()</script><p>A &amp; B</p></body></html>""",
        )

    result = EpubToolSet(workspace_root=str(tmp_path)).read_chapter(
        path=book.name,
        chapter_index=0,
    )

    assert result["status"] == "success"
    assert result["title"] == "One & Chapter"
    assert "A & B" in result["content"]
    assert "ignore()" not in result["content"]


def test_epub_search_returns_typed_error_for_empty_query(tmp_path: Path) -> None:
    result = EpubToolSet(workspace_root=str(tmp_path)).search(
        path="missing.epub", query=""
    )

    assert isinstance(result, ToolResult)
    assert result.status == "error"
    assert result.error == "query cannot be empty"
    assert result.output == {
        "status": "error",
        "message": "query cannot be empty",
    }


@pytest.mark.asyncio
async def test_epub_error_is_error_through_tool_executor(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register_toolset(
        EpubToolSet(workspace_root=str(tmp_path)), namespace=""
    )
    events = []
    executor = ToolBatchExecutor(
        registry.freeze(), ToolExecutionConfig(), emit=events.append
    )

    result = (
        await executor.execute_batch(
            [
                ToolCall(
                    id="search-1",
                    name="search",
                    arguments={"path": "missing.epub", "query": ""},
                )
            ]
        )
    )[0]

    assert result.status == "error"
    terminal = next(event for event in events if isinstance(event, ToolExecutionEnd))
    assert terminal.result.status == "error"
    assert terminal.is_error is True
