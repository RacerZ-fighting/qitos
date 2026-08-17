"""Tests for engine-free kit memory adapters."""

from __future__ import annotations

from qitos.core.memory import MemoryRecord
from qitos.kit.memory import SummaryMemory, VectorMemory, WindowMemory


def test_memory_adapters_basic():
    win = WindowMemory(window_size=2)
    win.append(MemoryRecord(role="user", content="a", step_id=0))
    win.append(MemoryRecord(role="assistant", content="b", step_id=1))
    win.append(MemoryRecord(role="user", content="c", step_id=2))
    assert [r.content for r in win.retrieve()] == ["b", "c"]

    summary = SummaryMemory(keep_last=2)
    summary.append(MemoryRecord(role="user", content="alpha", step_id=0))
    summary.append(MemoryRecord(role="assistant", content="beta", step_id=1))
    assert "alpha" in summary.summarize(max_items=2)

    vec = VectorMemory(top_k=1)
    vec.append(MemoryRecord(role="user", content="python docs", step_id=0))
    vec.append(MemoryRecord(role="user", content="flight booking", step_id=1))
    top = vec.retrieve({"text": "python", "top_k": 1})
    assert len(top) == 1
