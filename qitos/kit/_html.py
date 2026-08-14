"""Shared HTML-to-text policy backed by Beautiful Soup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from bs4 import BeautifulSoup


@dataclass(frozen=True, slots=True)
class ExtractedHTML:
    """Readable document text and its optional title."""

    text: str
    title: str | None


def extract_html_text(
    html: str,
    *,
    keep_links: bool = False,
    layout: Literal["inline", "lines"] = "inline",
) -> ExtractedHTML:
    """Extract readable text using the parser required by QitOS packaging."""

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "canvas"]):
        tag.decompose()

    if keep_links:
        for anchor in soup.find_all("a"):
            href = anchor.get("href")
            if href:
                anchor.append(f" ({href})")

    title = soup.title.get_text(" ", strip=True) if soup.title is not None else None
    separator = "\n" if layout == "lines" else " "
    raw_text = soup.get_text(separator=separator, strip=True)
    if layout == "lines":
        text = "\n".join(line.strip() for line in raw_text.splitlines() if line.strip())
    else:
        text = " ".join(raw_text.split())
    return ExtractedHTML(text=text, title=title or None)


__all__ = ["ExtractedHTML", "extract_html_text"]
