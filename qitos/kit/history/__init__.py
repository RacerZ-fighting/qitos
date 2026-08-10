"""Concrete history implementations."""

from qitos.core.history import History, HistoryMessage, HistorySnapshot

from .compact_history import (
    CompactConfig,
    CompactHistory,
    MessageGrouper,
    compact_history,
)
from .token_budget_history import TokenBudgetSummaryHistory
from .window_history import WindowHistory


def window_history(window_size: int = 24) -> WindowHistory:
    return WindowHistory(window_size=window_size)


__all__ = [
    "History",
    "HistoryMessage",
    "HistorySnapshot",
    "WindowHistory",
    "TokenBudgetSummaryHistory",
    "CompactConfig",
    "CompactHistory",
    "MessageGrouper",
    "compact_history",
    "window_history",
]
