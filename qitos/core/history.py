"""Transaction-complete history snapshots for turn contracts.

Only the snapshot closure remains: the Engine-era ``History`` store,
``HistoryPolicy`` and grouping/selection helpers left with the old lifecycle.
The canonical transcript for the minimal agent loop lives in
``core.message`` (``UserMessage``/``AssistantMessage``/``ToolResultMessage``).
"""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class HistoryMessage:
    role: str
    step_id: int
    content: Any = ""
    reasoning_content: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    native_items: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class HistorySnapshot:
    """Transaction-complete model history snapshot."""

    messages: tuple[HistoryMessage, ...]
    source_revision: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "messages",
            tuple(deepcopy(message) for message in self.messages),
        )

    @classmethod
    def from_messages(
        cls,
        messages: Iterable[HistoryMessage],
        *,
        source_revision: Optional[int] = None,
    ) -> "HistorySnapshot":
        """Snapshot the largest complete transaction prefix."""
        return cls(
            messages=tuple(complete_history_prefix(messages)),
            source_revision=source_revision,
        )


def _message_value(message: Any, name: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(name, default)
    return getattr(message, name, default)


def message_tool_call_ids(message: Any) -> List[str]:
    """Return unique generic and native function-call ids in wire order."""

    call_ids: List[str] = []
    seen: set[str] = set()
    for call in list(_message_value(message, "tool_calls", []) or []):
        call_id = call.get("id") if isinstance(call, dict) else None
        normalized = str(call_id) if call_id not in (None, "") else ""
        if normalized and normalized not in seen:
            seen.add(normalized)
            call_ids.append(normalized)
    for item in list(_message_value(message, "native_items", []) or []):
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        call_id = item.get("call_id") or item.get("id")
        normalized = str(call_id) if call_id not in (None, "") else ""
        if normalized and normalized not in seen:
            seen.add(normalized)
            call_ids.append(normalized)
    return call_ids


def message_tool_result_ids(message: Any) -> List[str]:
    """Return unique generic and native function-call result ids."""

    result_ids: List[str] = []
    seen: set[str] = set()
    generic_id = _message_value(message, "tool_call_id")
    if generic_id not in (None, ""):
        normalized = str(generic_id)
        seen.add(normalized)
        result_ids.append(normalized)
    for item in list(_message_value(message, "native_items", []) or []):
        if not isinstance(item, dict) or item.get("type") != "function_call_output":
            continue
        result_id = item.get("call_id") or item.get("id")
        normalized = str(result_id) if result_id not in (None, "") else ""
        if normalized and normalized not in seen:
            seen.add(normalized)
            result_ids.append(normalized)
    return result_ids


def complete_history_prefix(
    messages: Iterable[HistoryMessage],
) -> List[HistoryMessage]:
    """Return the largest prefix containing complete ordered tool rounds.

    Tool ids are scoped to their ordered model call, not to the whole history.
    A result before its declaration, a duplicate result, or a missing result
    therefore cuts the prefix at the first invalid transaction.  This keeps a
    checkpoint or compacted provider input from retaining an orphan result or
    a dangling assistant call.
    """
    items = list(messages)
    pending: Dict[str, deque[int]] = defaultdict(deque)
    for index, message in enumerate(items):
        for call_id in message_tool_call_ids(message):
            pending[call_id].append(index)
        for result_id in message_tool_result_ids(message):
            declarations = pending.get(result_id)
            if not declarations:
                first_pending = min(
                    (
                        declaration_index
                        for queued in pending.values()
                        for declaration_index in queued
                    ),
                    default=index,
                )
                return items[: min(index, first_pending)]
            declarations.popleft()
            if not declarations:
                pending.pop(result_id, None)

    if pending:
        first_unmatched = min(
            declaration_index
            for declarations in pending.values()
            for declaration_index in declarations
        )
        return items[:first_unmatched]
    return items
