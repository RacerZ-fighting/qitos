"""Canonical history contracts for model message context."""

from __future__ import annotations

from abc import ABC, abstractmethod
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


def message_token_payloads(message: Any) -> List[Any]:
    """Return one provider-shaped token payload without mirror double counting.

    Responses history deliberately retains both canonical generic fields and
    native response items. Native items are authoritative for their matching
    reasoning, function call, message, or function output; unmatched generic
    fields remain represented for Chat-compatible histories.
    """

    native_items: List[Dict[str, Any]] = []
    native_call_ids: set[str] = set()
    native_types: set[str] = set()
    seen_transactions: set[tuple[str, str]] = set()
    for raw_item in list(_message_value(message, "native_items", []) or []):
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        item_type = str(item.get("type") or "")
        native_types.add(item_type)
        call_id = str(item.get("call_id") or item.get("id") or "")
        if item_type in {"function_call", "function_call_output"} and call_id:
            transaction = (item_type, call_id)
            if transaction in seen_transactions:
                continue
            seen_transactions.add(transaction)
            if item_type == "function_call":
                native_call_ids.add(call_id)
        native_items.append(item)

    payloads: List[Any] = []
    role = str(_message_value(message, "role", "") or "")
    content = _message_value(message, "content")
    native_content_is_authoritative = "message" in native_types or (
        role == "tool" and "function_call_output" in native_types
    )
    if content not in (None, "") and not native_content_is_authoritative:
        payloads.append(content)

    reasoning = _message_value(message, "reasoning_content")
    if reasoning not in (None, "") and "reasoning" not in native_types:
        payloads.append(reasoning)

    generic_calls: List[Dict[str, Any]] = []
    for call in list(_message_value(message, "tool_calls", []) or []):
        if not isinstance(call, dict):
            continue
        call_id = str(call.get("id") or "")
        if call_id and call_id in native_call_ids:
            continue
        generic_calls.append(call)
    if generic_calls:
        payloads.append(generic_calls)
    if native_items:
        payloads.append(native_items)
    return payloads


def group_history_rounds(
    messages: Iterable[HistoryMessage],
) -> List[List[HistoryMessage]]:
    """Group messages without splitting a model/tool transaction.

    Engine-produced assistant calls and their tool results normally share a
    ``step_id``. Matching call ids additionally merge groups when an adapter
    records the result on a later step, so every returned boundary is safe for
    windowing, eviction, and compaction.
    """

    items = list(messages)
    if not items:
        return []

    groups: List[List[HistoryMessage]] = []
    current: List[HistoryMessage] = []
    current_step: Optional[int] = None
    for message in items:
        step = int(getattr(message, "step_id", 0))
        if current and current_step is not None and step != current_step:
            groups.append(current)
            current = []
        current.append(message)
        current_step = step
    if current:
        groups.append(current)

    if len(groups) == 1:
        groups = []
        current = []
        seen_assistant = False
        for message in items:
            if current and message.role == "assistant" and seen_assistant:
                groups.append(current)
                current = [message]
                seen_assistant = True
                continue
            current.append(message)
            if message.role == "assistant":
                seen_assistant = True
        if current:
            groups.append(current)

    call_groups: Dict[str, int] = {}
    for group_index, group in enumerate(groups):
        for message in group:
            if message.role != "assistant":
                continue
            for call_id in message_tool_call_ids(message):
                call_groups.setdefault(call_id, group_index)

    unsafe_boundaries: set[int] = set()
    for result_group, group in enumerate(groups):
        for message in group:
            for result_id in message_tool_result_ids(message):
                call_group = call_groups.get(result_id)
                if call_group is None or call_group >= result_group:
                    continue
                unsafe_boundaries.update(range(call_group + 1, result_group + 1))

    merged: List[List[HistoryMessage]] = []
    for group_index, group in enumerate(groups):
        if merged and group_index in unsafe_boundaries:
            merged[-1].extend(group)
        else:
            merged.append(list(group))
    return merged


def select_recent_history(
    messages: Iterable[HistoryMessage], max_items: int
) -> List[HistoryMessage]:
    """Select a bounded recent suffix at complete round boundaries.

    The result does not exceed ``max_items`` unless the newest indivisible
    model/tool round itself is larger than the configured window.
    """

    items = list(messages)
    limit = int(max_items)
    if limit <= 0 or len(items) <= limit:
        return items

    selected_reversed: List[List[HistoryMessage]] = []
    selected_count = 0
    for group in reversed(group_history_rounds(items)):
        if selected_reversed and selected_count + len(group) > limit:
            break
        selected_reversed.append(group)
        selected_count += len(group)
        if selected_count >= limit:
            break
    return [
        message
        for group in reversed(selected_reversed)
        for message in group
    ]


@dataclass
class HistoryPolicy:
    """Engine-side policy for selecting/assembling history messages."""

    roles: List[str] = field(default_factory=lambda: ["user", "assistant", "tool"])
    max_messages: int = 24
    step_window: Optional[int] = None
    max_tokens: Optional[int] = None

    def build_query(self, step_id: int, **kwargs: Any) -> Dict[str, Any]:
        query: Dict[str, Any] = {
            "roles": list(self.roles),
            "max_items": int(self.max_messages),
        }
        if self.step_window is not None and self.step_window > 0:
            query["step_min"] = max(0, int(step_id) - int(self.step_window) + 1)
        if self.max_tokens is not None and int(self.max_tokens) > 0:
            query["max_tokens"] = int(self.max_tokens)
        query.update({str(key): value for key, value in kwargs.items()})
        return query


class History(ABC):
    @abstractmethod
    def append(self, message: HistoryMessage) -> None:
        """Append one chat message into history store."""

    @abstractmethod
    def retrieve(
        self,
        query: Optional[Dict[str, Any]] = None,
        state: Any = None,
        observation: Any = None,
    ) -> Any:
        """Retrieve history payload used for model message assembly."""

    @abstractmethod
    def summarize(self, max_items: int = 5) -> str:
        """Return strategy-specific summary for old messages."""

    @abstractmethod
    def evict(self) -> int:
        """Apply retention strategy and return number of evicted messages."""

    @abstractmethod
    def reset(self, run_id: Optional[str] = None) -> None:
        """Reset history runtime state for a new run."""


__all__ = [
    "History",
    "HistoryMessage",
    "HistoryPolicy",
    "group_history_rounds",
    "message_token_payloads",
    "message_tool_call_ids",
    "message_tool_result_ids",
    "select_recent_history",
]
