from __future__ import annotations

import zipfile
from pathlib import Path

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
