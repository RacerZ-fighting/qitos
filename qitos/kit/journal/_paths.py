"""Shared on-disk layout rules for JSONL Run journals."""

from __future__ import annotations

from pathlib import Path

INDEX_FILENAME = "journal.index.sqlite3"
JOURNAL_FILENAME = "journal.jsonl"


def validate_run_id(run_id: str) -> None:
    """Reject Run identifiers that could escape a catalog root."""

    if not isinstance(run_id, str) or not run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be non-empty")
    if "/" in run_id or "\\" in run_id or "\x00" in run_id:
        raise ValueError("run_id contains a path separator")


def journal_path(root: Path, run_id: str) -> Path:
    """Return the canonical path after validating the Run identifier."""

    validate_run_id(run_id)
    return root / run_id / JOURNAL_FILENAME


__all__ = ["INDEX_FILENAME", "JOURNAL_FILENAME", "journal_path", "validate_run_id"]
