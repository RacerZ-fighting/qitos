"""Compact-aware history implementation for long-running agent loops."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import OrderedDict
from dataclasses import dataclass, replace
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional

from qitos.core.history import (
    History,
    HistoryMessage,
    HistorySnapshot,
    group_history_rounds,
    message_token_payloads,
    select_recent_history,
)


@dataclass
class CompactConfig:
    """Configuration for `CompactHistory`."""

    max_tokens: int = 16000
    keep_last_rounds: int = 2
    # Canonical history is token-controlled by default. Count-based eviction
    # remains an explicit opt-in for applications that accept permanent loss.
    hard_window: int = 0
    # Ratios are relative to the force-compaction budget supplied by Engine.
    # With Engine's 0.80 total-input trigger these defaults warn near 56% and
    # microcompact near 60%, leaving full summarization for the 80% boundary.
    warning_ratio: float = 0.70
    microcompact_ratio: float = 0.75
    auto_compact: bool = True
    compact_long_messages_over_chars: int = 4_000
    microcompact_preview_chars: int = 800
    summary_max_chars: int = 24_000
    # Number of most recent messages kept verbatim in the summary request; older
    # messages are elided to `microcompact_preview_chars` but never dropped.
    summary_input_message_limit: int = 64
    summary_metadata_source: str = "compact_history"
    emit_skipped_events: bool = True


@dataclass(frozen=True)
class _SummaryCheckpoint:
    """Immutable summary boundary for one canonical history prefix."""

    text: str
    source_digest: str
    message_count: int


class MessageGrouper:
    """Group messages into compactable rounds."""

    def group(self, messages: Iterable[HistoryMessage]) -> List[List[HistoryMessage]]:
        return group_history_rounds(messages)


class MicroCompactor:
    """Apply low-cost compaction to older, high-token messages."""

    def __init__(self, config: CompactConfig):
        self.config = config

    def compact(self, messages: Iterable[HistoryMessage]) -> List[HistoryMessage]:
        return [self._compact_message(msg) for msg in messages]

    def _compact_message(self, message: HistoryMessage) -> HistoryMessage:
        text = str(message.content or "")
        if message.metadata.get("summary"):
            return message
        if len(text) <= int(self.config.compact_long_messages_over_chars):
            return message

        preview = max(60, int(self.config.microcompact_preview_chars))
        head = text[:preview].rstrip()
        tail = text[-min(preview // 2, len(text)) :].lstrip()
        newline_count = text.count("\n")
        blob_kind = self._infer_blob_kind(message, text)
        compacted = (
            f"[compact:start step={message.step_id} kind={blob_kind} "
            f"original_chars={len(text)} original_lines={newline_count + 1}]\n"
            f"{head}"
        )
        if tail and tail != head:
            compacted += f"\n...\n{tail}"
        compacted += "\n[compact:end]"

        metadata = dict(message.metadata)
        metadata.update(
            {
                "compacted": True,
                "compaction_mode": "micro",
                "original_chars": len(text),
                "original_lines": newline_count + 1,
            }
        )
        return HistoryMessage(
            role=message.role,
            content=compacted,
            step_id=message.step_id,
            reasoning_content=message.reasoning_content,
            tool_calls=[dict(x) for x in list(message.tool_calls or [])],
            tool_call_id=message.tool_call_id,
            name=message.name,
            metadata=metadata,
            native_items=[dict(x) for x in list(message.native_items or [])],
        )

    def _infer_blob_kind(self, message: HistoryMessage, text: str) -> str:
        source = str(message.metadata.get("source", "")).strip().lower()
        role = str(message.role).strip().lower() or "message"
        lowered = text.lower()
        if any(
            token in lowered
            for token in ("traceback", "stderr", "stdout", "returncode")
        ):
            return "tool output"
        if any(
            token in lowered
            for token in ("http", "<html", "```html", "response headers")
        ):
            return "web/file result"
        if source:
            return f"{source} {role} message"
        return f"{role} message"


class SummaryCompactor:
    """Summarize older rounds into one continuation message."""

    def __init__(self, config: CompactConfig, llm: Any | None = None):
        self.config = config
        self.llm = llm

    def summarize(
        self,
        messages: Iterable[HistoryMessage],
        *,
        prior_summary: str | None = None,
    ) -> str:
        items = list(messages)
        if not items:
            return ""

        prompt = self._summary_prompt(items, prior_summary=prior_summary)
        if self.llm is not None:
            try:
                response = self.llm(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Create durable continuation state for an agent loop. "
                                "Do not call tools or reveal private reasoning. Return only "
                                "one <summary> block that preserves decisions, evidence, "
                                "constraints, failures, current state, and the next action."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ]
                )
                summary = self._normalize_summary(str(response or ""))
                if summary:
                    return summary[: int(self.config.summary_max_chars)]
            except Exception as exc:
                raise RuntimeError("history summarization failed") from exc
            raise RuntimeError("history summarizer returned an empty summary")

        return self._heuristic_summary(items, prior_summary=prior_summary)

    def _normalize_summary(self, value: str) -> str:
        """Strip the model's drafting scratchpad from the durable summary."""

        text = str(value or "").strip()
        if not text:
            return ""
        summary_match = re.search(
            r"<summary>\s*(.*?)\s*</summary>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if summary_match is not None:
            return summary_match.group(1).strip()
        text = re.sub(
            r"<analysis>.*?</analysis>",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return text.strip()

    def render_body(self, messages: List[HistoryMessage]) -> str:
        """Render every message for the summary request, eliding instead of dropping.

        The most recent `summary_input_message_limit` messages keep their full
        text; older ones are shortened to a preview so the earliest turns are
        still represented. No message is ever removed from the request.
        """

        verbatim = max(1, int(self.config.summary_input_message_limit))
        elide_before = max(0, len(messages) - verbatim)
        preview = max(60, int(self.config.microcompact_preview_chars))
        lines: List[str] = []
        for index, message in enumerate(messages):
            text = str(message.content or "")
            if message.tool_calls:
                calls = json.dumps(
                    message.tool_calls,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                text = f"{text}\ntool_calls={calls}".strip()
            native_calls = [
                item
                for item in list(message.native_items or [])
                if isinstance(item, dict) and item.get("type") == "function_call"
            ]
            if native_calls and not message.tool_calls:
                calls = json.dumps(
                    native_calls,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                text = f"{text}\nnative_function_calls={calls}".strip()
            if message.reasoning_content or any(
                isinstance(item, dict) and item.get("type") == "reasoning"
                for item in list(message.native_items or [])
            ):
                text = f"{text}\n[opaque reasoning continuation omitted]".strip()
            if message.tool_call_id:
                text = (
                    f"tool_result_for={message.tool_call_id}\n{text}"
                ).strip()
            if index < elide_before and len(text) > preview:
                dropped = len(text) - preview
                text = f"{text[:preview].rstrip()} ...[elided {dropped} chars]"
            lines.append(f"[{message.step_id}] {message.role}: {text}")
        return "\n".join(lines)

    def _summary_prompt(
        self, messages: List[HistoryMessage], *, prior_summary: str | None = None
    ) -> str:
        body = self.render_body(messages)
        prior_block = ""
        if prior_summary and str(prior_summary).strip():
            prior_block = (
                "An earlier continuation summary already covers context before the "
                "conversation below. Consolidate it with the new material instead of "
                "discarding it.\n\n"
                "<previous_summary>\n"
                f"{str(prior_summary).strip()}\n"
                "</previous_summary>\n\n"
            )
        return (
            "Summarize the earlier conversation as continuation state. Preserve:\n"
            "- the user's goal, explicit constraints, and unresolved choices;\n"
            "- decisions, evidence, relevant files or identifiers, and completed work;\n"
            "- failed attempts and why they failed;\n"
            "- pending work and the single next action.\n"
            "Keep facts precise, omit private reasoning, and do not invent completion.\n\n"
            f"{prior_block}"
            f"{body}"
        )

    def _heuristic_summary(
        self, messages: List[HistoryMessage], *, prior_summary: str | None = None
    ) -> str:
        user_goal = ""
        assistant_notes: List[str] = []
        for msg in messages:
            content = str(msg.content or "").strip()
            if not content:
                continue
            snippet = content[:160].replace("\n", " ")
            if msg.role == "user" and not user_goal:
                user_goal = snippet
            elif msg.role == "assistant":
                assistant_notes.append(snippet)

        lines = ["Continuation summary of earlier context:"]
        if prior_summary and str(prior_summary).strip():
            carried = str(prior_summary).strip().replace("\n", " ")[:400]
            lines.append(f"- Earlier summary: {carried}")
        if user_goal:
            lines.append(f"- Goal: {user_goal}")
        if assistant_notes:
            lines.append(f"- Findings: {assistant_notes[-1]}")
            if len(assistant_notes) > 1:
                lines.append(f"- Prior attempt: {assistant_notes[-2]}")
        last = messages[-1]
        lines.append(
            f"- Pending: Continue from step {last.step_id} with the latest trajectory in mind."
        )
        return "\n".join(lines)[: int(self.config.summary_max_chars)]


class CompactionController:
    """Coordinate threshold checks, microcompact, and summary compact."""

    def __init__(
        self,
        config: CompactConfig,
        *,
        llm: Any | None = None,
        grouper: MessageGrouper | None = None,
        micro: MicroCompactor | None = None,
        summary: SummaryCompactor | None = None,
    ):
        self.config = config
        self.grouper = grouper or MessageGrouper()
        self.micro = micro or MicroCompactor(config)
        self.summary = summary or SummaryCompactor(config, llm=llm)
        self._summary_cache: OrderedDict[str, str] = OrderedDict()
        self._summary_cache_limit = 8
        self._summary_failures: OrderedDict[str, int] = OrderedDict()
        self._summary_failure_limit = 3

    def clear_cache(self) -> None:
        self._summary_cache.clear()
        self._summary_failures.clear()

    def retrieve(
        self,
        items: List[HistoryMessage],
        *,
        budget: int,
        pending_content: str,
        auto_compact: bool,
        prior_summary: _SummaryCheckpoint | None = None,
    ) -> tuple[List[HistoryMessage], List[Dict[str, Any]], List[Dict[str, Any]]]:
        events: List[Dict[str, Any]] = []
        pending_tokens = self._estimate_text_tokens(pending_content)
        before_tokens = self._estimate_tokens(items) + pending_tokens
        metadata = [self._metadata_for_message(m) for m in items]
        if budget <= 0:
            events.append(
                self._compaction_event(
                    "within_budget",
                    before_tokens=before_tokens,
                    after_tokens=before_tokens,
                    budget=budget,
                    pending_tokens=pending_tokens,
                    messages_before=len(items),
                    messages_after=len(items),
                    reason="budget_disabled",
                )
            )
            return items, events, metadata

        warning_threshold = max(
            1,
            math.ceil(budget * float(self.config.warning_ratio)),
        )
        micro_threshold = max(
            1,
            math.ceil(budget * float(self.config.microcompact_ratio)),
        )

        if before_tokens >= warning_threshold:
            events.append(
                self._compaction_event(
                    "warning",
                    before_tokens=before_tokens,
                    after_tokens=before_tokens,
                    budget=budget,
                    pending_tokens=pending_tokens,
                    messages_before=len(items),
                    messages_after=len(items),
                    warning_threshold=warning_threshold,
                    microcompact_threshold=micro_threshold,
                )
            )

        if before_tokens < micro_threshold:
            if self.config.emit_skipped_events:
                events.append(
                    self._compaction_event(
                        "within_budget",
                        before_tokens=before_tokens,
                        after_tokens=before_tokens,
                        budget=budget,
                        pending_tokens=pending_tokens,
                        messages_before=len(items),
                        messages_after=len(items),
                        reason="below_microcompact_threshold",
                        microcompact_threshold=micro_threshold,
                    )
                )
            return items, events, metadata

        if not auto_compact:
            if self.config.emit_skipped_events:
                events.append(
                    self._compaction_event(
                        "compact_skipped",
                        before_tokens=before_tokens,
                        after_tokens=before_tokens,
                        budget=budget,
                        pending_tokens=pending_tokens,
                        messages_before=len(items),
                        messages_after=len(items),
                        reason="auto_compact_disabled",
                    )
                )
            return items, events, metadata

        groups = self.grouper.group(items)
        if not groups:
            return items, events, metadata
        keep_rounds = max(1, int(self.config.keep_last_rounds))
        preserved_groups = groups[-keep_rounds:]
        older_groups = groups[:-keep_rounds]
        preserved = [msg for group in preserved_groups for msg in group]
        older = [msg for group in older_groups for msg in group]
        compacted_older = self.micro.compact(older)
        micro_candidate = [*compacted_older, *preserved]
        after_micro_tokens = self._estimate_tokens(micro_candidate) + pending_tokens
        micro_applied = after_micro_tokens < before_tokens
        if micro_applied:
            events.append(
                self._compaction_event(
                    "microcompact_applied",
                    before_tokens=before_tokens,
                    after_tokens=after_micro_tokens,
                    budget=budget,
                    pending_tokens=pending_tokens,
                    messages_before=len(items),
                    messages_after=len(micro_candidate),
                    compaction_level=1,
                    microcompact_threshold=micro_threshold,
                    messages_compacted=sum(
                        1
                        for msg in compacted_older
                        if msg.metadata.get("compaction_mode") == "micro"
                    ),
                )
            )
        if before_tokens < budget or after_micro_tokens < budget:
            result = micro_candidate if micro_applied else items
            return (
                result,
                events,
                [self._metadata_for_message(m) for m in result],
            )

        summary_candidate: List[HistoryMessage] | None = None
        if older:
            summary_candidate, summary_trace = self._summary_projection(
                covered=older,
                preserved=preserved,
                budget=max(1, budget - pending_tokens),
                prior_summary=prior_summary,
                compaction_level=2,
            )
            after_summary_tokens = (
                self._estimate_tokens(summary_candidate) + pending_tokens
            )
            events.append(
                self._compaction_event(
                    "summary_compact_applied",
                    before_tokens=before_tokens,
                    after_tokens=after_summary_tokens,
                    budget=budget,
                    pending_tokens=pending_tokens,
                    messages_before=len(items),
                    messages_after=len(summary_candidate),
                    compaction_level=2,
                    preserved_round_count=len(preserved_groups),
                    preserved_message_count=len(preserved),
                    **summary_trace,
                )
            )
            if after_summary_tokens <= budget:
                return (
                    summary_candidate,
                    events,
                    [self._metadata_for_message(m) for m in summary_candidate],
                )

        hard_preserved_groups = groups[-1:]
        hard_older_groups = groups[:-1]
        hard_preserved = [
            message for group in hard_preserved_groups for message in group
        ]
        hard_older = [message for group in hard_older_groups for message in group]
        if not hard_older:
            result = summary_candidate or micro_candidate
            after_hard_tokens = self._estimate_tokens(result) + pending_tokens
            events.append(
                self._compaction_event(
                    "hard_compact_blocked",
                    before_tokens=before_tokens,
                    after_tokens=after_hard_tokens,
                    budget=budget,
                    pending_tokens=pending_tokens,
                    messages_before=len(items),
                    messages_after=len(result),
                    compaction_level=3,
                    reason="latest_transaction_is_indivisible",
                )
            )
            return result, events, [self._metadata_for_message(m) for m in result]

        if len(preserved_groups) == 1 and summary_candidate is not None:
            hard_candidate = summary_candidate
            hard_trace = {
                key: value
                for key, value in summary_candidate[0].metadata.items()
                if key
                not in {
                    "summary",
                    "source",
                    "compaction_level",
                    "summarized_through_step",
                }
            }
        else:
            hard_candidate, hard_trace = self._summary_projection(
                covered=hard_older,
                preserved=hard_preserved,
                budget=max(1, budget - pending_tokens),
                prior_summary=prior_summary,
                compaction_level=3,
            )
        after_hard_tokens = self._estimate_tokens(hard_candidate) + pending_tokens
        hard_stage = (
            "hard_compact_applied"
            if after_hard_tokens <= budget
            else "hard_compact_blocked"
        )
        events.append(
            self._compaction_event(
                hard_stage,
                before_tokens=before_tokens,
                after_tokens=after_hard_tokens,
                budget=budget,
                pending_tokens=pending_tokens,
                messages_before=len(items),
                messages_after=len(hard_candidate),
                compaction_level=3,
                preserved_round_count=1,
                preserved_message_count=len(hard_preserved),
                reason=(
                    None
                    if hard_stage == "hard_compact_applied"
                    else "summary_and_latest_transaction_exceed_budget"
                ),
                **hard_trace,
            )
        )
        return (
            hard_candidate,
            events,
            [self._metadata_for_message(m) for m in hard_candidate],
        )

    def _summary_projection(
        self,
        *,
        covered: List[HistoryMessage],
        preserved: List[HistoryMessage],
        budget: int,
        prior_summary: _SummaryCheckpoint | None,
        compaction_level: int,
    ) -> tuple[List[HistoryMessage], Dict[str, Any]]:
        carried_summary, carried_message_count = self._carried_summary(
            covered,
            prior_summary,
        )
        newly_covered = covered[carried_message_count:]
        compacted = self.micro.compact(newly_covered)
        summary_input = (
            newly_covered
            if self._estimate_tokens(newly_covered) <= budget
            else compacted
        )
        summary_input_mode = (
            "checkpoint"
            if not summary_input and carried_summary
            else "full"
            if summary_input is newly_covered
            else "microcompacted"
        )
        source_digest = self._source_digest(covered)
        summary_token_allowance = max(
            1,
            budget - self._estimate_tokens(preserved),
        )
        summary_char_limit = max(
            1,
            min(
                int(self.config.summary_max_chars),
                summary_token_allowance * 4,
            ),
        )
        cache_key_payload = {
            "source_digest": source_digest,
            "budget": budget,
            "compaction_level": compaction_level,
            "summary_char_limit": summary_char_limit,
            "summary_max_chars": int(self.config.summary_max_chars),
            "summary_input_message_limit": int(
                self.config.summary_input_message_limit
            ),
            "microcompact_preview_chars": int(
                self.config.microcompact_preview_chars
            ),
        }
        cache_key = hashlib.sha256(
            json.dumps(
                cache_key_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        cached_summary = self._summary_cache.get(cache_key)
        if cached_summary is not None:
            self._summary_cache.move_to_end(cache_key)
        summary_text = str(cached_summary or "").strip()
        if not summary_text and carried_summary and not summary_input:
            summary_text = carried_summary
        if not summary_text:
            if (
                self._summary_failures.get(source_digest, 0)
                >= self._summary_failure_limit
            ):
                raise RuntimeError("history summarization circuit is open")
            try:
                summary_text = str(
                    self.summary.summarize(
                        summary_input,
                        prior_summary=carried_summary,
                    )
                    or ""
                ).strip()
            except Exception:
                self._summary_failures[source_digest] = (
                    self._summary_failures.get(source_digest, 0) + 1
                )
                self._summary_failures.move_to_end(source_digest)
                while len(self._summary_failures) > self._summary_cache_limit:
                    self._summary_failures.popitem(last=False)
                raise
        if not summary_text:
            raise RuntimeError("history compactor returned an empty summary")
        summary_text = summary_text[:summary_char_limit].rstrip()
        if not summary_text:
            raise RuntimeError("history compactor returned an empty bounded summary")
        self._summary_failures.pop(source_digest, None)
        self._summary_cache[cache_key] = summary_text
        self._summary_cache.move_to_end(cache_key)
        while len(self._summary_cache) > self._summary_cache_limit:
            self._summary_cache.popitem(last=False)
        summary_input_step_range = (
            [summary_input[0].step_id, summary_input[-1].step_id]
            if summary_input
            else None
        )
        built_on_prior_summary = bool(
            carried_summary and cached_summary is None and summary_input
        )
        trace = {
            "summarized_message_count": len(covered),
            "source_digest": source_digest,
            "summary_input_message_count": len(summary_input),
            "summary_carried_message_count": carried_message_count,
            "summary_input_mode": summary_input_mode,
            "summary_dropped_message_count": 0,
            "summarized_step_range": [covered[0].step_id, covered[-1].step_id],
            "summary_input_step_range": summary_input_step_range,
            "summary_input_chars": (
                len(self.summary.render_body(summary_input)) if summary_input else 0
            ),
            "built_on_prior_summary": built_on_prior_summary,
            "summary_cache_hit": cached_summary is not None,
            "summary_budget": budget,
            "summary_char_limit": summary_char_limit,
            "summary_chars": len(summary_text),
        }
        summary_message = HistoryMessage(
            role="system",
            content=summary_text,
            step_id=covered[-1].step_id,
            metadata={
                "summary": True,
                "source": self.config.summary_metadata_source,
                "compaction_level": compaction_level,
                "summarized_through_step": covered[-1].step_id,
                **trace,
            },
        )
        return [summary_message, *preserved], trace

    def _compaction_event(
        self,
        stage: str,
        *,
        before_tokens: int,
        after_tokens: int,
        budget: int,
        pending_tokens: int,
        messages_before: int,
        messages_after: int,
        **detail: Any,
    ) -> Dict[str, Any]:
        context = {
            "stage": stage,
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "saved_tokens": max(0, before_tokens - after_tokens),
            "budget": budget,
            "pending_tokens": pending_tokens,
            "messages_before": messages_before,
            "messages_after": messages_after,
            "strategy": "compact_history",
            "warning_ratio": float(self.config.warning_ratio),
            "microcompact_ratio": float(self.config.microcompact_ratio),
        }
        context.update(detail)
        return {"stage": "context_history", "context": context}

    def _carried_summary(
        self,
        covered: List[HistoryMessage],
        prior_summary: _SummaryCheckpoint | None,
    ) -> tuple[Optional[str], int]:
        """Reuse a prior summary only when its exact source is still a prefix."""

        if prior_summary is None:
            return None, 0
        count = int(prior_summary.message_count)
        text = str(prior_summary.text or "").strip()
        if not text or count <= 0 or count > len(covered):
            return None, 0
        if self._source_digest(covered[:count]) != prior_summary.source_digest:
            return None, 0
        return text, count

    def _metadata_for_message(self, message: HistoryMessage) -> Dict[str, Any]:
        meta = dict(message.metadata or {})
        meta.setdefault("role", message.role)
        meta.setdefault("step_id", message.step_id)
        meta.setdefault("content_chars", len(str(message.content or "")))
        if message.tool_call_id:
            meta.setdefault("tool_call_id", message.tool_call_id)
        if message.tool_calls:
            meta.setdefault("tool_calls_count", len(message.tool_calls))
        if message.name:
            meta.setdefault("name", message.name)
        return meta

    def _estimate_tokens(self, messages: Iterable[HistoryMessage]) -> int:
        return sum(
            sum(
                self._estimate_text_tokens(payload)
                for payload in message_token_payloads(message)
            )
            for message in messages
        )

    def _source_digest(self, messages: Iterable[HistoryMessage]) -> str:
        payload = [
            {
                "role": message.role,
                "step_id": message.step_id,
                "content": message.content,
                "reasoning_content": message.reasoning_content,
                "tool_calls": message.tool_calls,
                "tool_call_id": message.tool_call_id,
                "name": message.name,
                "native_items": message.native_items,
            }
            for message in messages
        ]
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _estimate_text_tokens(self, text: Any) -> int:
        s = str(text or "")
        if not s:
            return 0
        return max(1, len(s) // 4)


class CompactHistory(History):
    """History implementation with threshold, microcompact, and summary compact."""

    def __init__(
        self,
        *,
        llm: Any | None = None,
        config: CompactConfig | None = None,
        max_tokens: Optional[int] = None,
        keep_last_rounds: Optional[int] = None,
        hard_window: Optional[int] = None,
        auto_compact: Optional[bool] = None,
    ):
        cfg = replace(config) if config is not None else CompactConfig()
        if max_tokens is not None:
            cfg.max_tokens = int(max_tokens)
        if keep_last_rounds is not None:
            cfg.keep_last_rounds = int(keep_last_rounds)
        if hard_window is not None:
            cfg.hard_window = int(hard_window)
        if auto_compact is not None:
            cfg.auto_compact = bool(auto_compact)

        self.llm = llm
        self.config = cfg
        self._base_messages: tuple[HistoryMessage, ...] = ()
        self._messages: List[HistoryMessage] = []
        self._controller = CompactionController(cfg, llm=llm)
        self._lock = RLock()
        self._pending_runtime_events: List[Dict[str, Any]] = []
        self._last_message_metadata: List[Dict[str, Any]] = []
        self._summary_checkpoint: _SummaryCheckpoint | None = None
        self._revision = 0

    def append(self, message: HistoryMessage) -> None:
        with self._lock:
            self._messages.append(message)
            self._revision += 1
            self.evict()

    def retrieve(
        self,
        query: Optional[Dict[str, Any]] = None,
        state: Any = None,
        observation: Any = None,
    ) -> List[HistoryMessage]:
        query = query or {}
        with self._lock:
            source_revision = self._revision
            items = self._filter_messages_unlocked(query)
            prior_summary = self._summary_checkpoint
        budget = int(query.get("max_tokens") or self.config.max_tokens)
        pending = str(query.get("pending_content") or "")
        auto_compact = bool(query.get("auto_compact", self.config.auto_compact))
        max_items = int(
            query.get("max_items", len(items) if items else self.config.hard_window)
        )
        # A token-aware strategy must see the complete canonical prefix so it
        # can summarize it. Count windows remain available when compaction is
        # explicitly disabled.
        if max_items > 0 and not auto_compact:
            items = select_recent_history(items, max_items)

        result, events, metadata = self._controller.retrieve(
            items,
            budget=budget,
            pending_content=pending,
            auto_compact=auto_compact,
            prior_summary=prior_summary,
        )
        with self._lock:
            if self._revision != source_revision:
                raise RuntimeError("history changed while compaction was in progress")
        result = [
            replace(
                message,
                metadata={
                    **message.metadata,
                    "source_history_version": source_revision,
                },
            )
            if message.metadata.get("summary")
            else message
            for message in result
        ]
        for item in metadata:
            if item.get("summary"):
                item.setdefault("source_history_version", source_revision)
        for event in events:
            context = event.get("context")
            if isinstance(context, dict):
                context.setdefault("source_history_version", source_revision)
        with self._lock:
            self._pending_runtime_events = list(events)
            self._last_message_metadata = list(metadata)
            self._remember_summary(result)
        return result

    def summarize(self, max_items: int = 5) -> str:
        with self._lock:
            items = select_recent_history(self._all_messages_unlocked(), max_items)
        return self._controller.summary.summarize(items)

    def evict(self) -> int:
        with self._lock:
            items = self._all_messages_unlocked()
            hard_window = int(self.config.hard_window)
            if hard_window <= 0 or len(items) <= hard_window:
                return 0
            retained = select_recent_history(items, hard_window)
            removed = len(items) - len(retained)
            self._base_messages = tuple(retained)
            self._messages = []
            if removed:
                self._revision += 1
            return removed

    def reset(self, run_id: Optional[str] = None) -> None:
        with self._lock:
            self._base_messages = ()
            self._messages = []
            self._pending_runtime_events = []
            self._last_message_metadata = []
            self._summary_checkpoint = None
            self._controller.clear_cache()
            self._revision += 1

    def snapshot(self) -> HistorySnapshot:
        """Capture a transaction-complete history prefix."""
        with self._lock:
            return HistorySnapshot.from_messages(
                self._all_messages_unlocked(),
                source_revision=self._revision,
            )

    def restore(self, snapshot: HistorySnapshot) -> None:
        """Restore a snapshot as the shared history base."""
        if not isinstance(snapshot, HistorySnapshot):
            raise TypeError("snapshot must be a HistorySnapshot")
        with self._lock:
            self._base_messages = snapshot.messages
            self._messages = []
            self._pending_runtime_events = []
            self._last_message_metadata = []
            self._summary_checkpoint = None
            self._controller.clear_cache()
            self._revision += 1

    def fork(self, snapshot: HistorySnapshot | None = None) -> "CompactHistory":
        """Create a history with a copy-on-write snapshot base."""
        inherited = snapshot if snapshot is not None else self.snapshot()
        child = CompactHistory(llm=self.llm, config=replace(self.config))
        child.restore(inherited)
        return child

    def _remember_summary(self, result: List[HistoryMessage]) -> None:
        """Keep the newest continuation summary so later passes can build on it.

        This does not mutate the stored history: `retrieve()` stays side-effect
        free for callers that read `messages`.
        """

        if not result:
            return
        head = result[0]
        if not head.metadata.get("summary"):
            return
        text = str(head.content or "").strip()
        source_digest = head.metadata.get("source_digest")
        message_count = head.metadata.get("summarized_message_count")
        if (
            text
            and isinstance(source_digest, str)
            and source_digest
            and isinstance(message_count, int)
            and message_count > 0
        ):
            self._summary_checkpoint = _SummaryCheckpoint(
                text=text,
                source_digest=source_digest,
                message_count=message_count,
            )

    @property
    def last_summary(self) -> Optional[str]:
        with self._lock:
            if self._summary_checkpoint is None:
                return None
            return self._summary_checkpoint.text

    @property
    def history_version(self) -> int:
        with self._lock:
            return self._revision

    def consume_runtime_events(self) -> List[Dict[str, Any]]:
        with self._lock:
            events = list(self._pending_runtime_events)
            self._pending_runtime_events = []
            return events

    def get_last_message_metadata(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._last_message_metadata)

    @property
    def messages(self) -> List[HistoryMessage]:
        with self._lock:
            return self._all_messages_unlocked()

    def _filter_messages(self, query: Dict[str, Any]) -> List[HistoryMessage]:
        with self._lock:
            return self._filter_messages_unlocked(query)

    def _filter_messages_unlocked(
        self, query: Dict[str, Any]
    ) -> List[HistoryMessage]:
        items = self._all_messages_unlocked()
        roles = query.get("roles")
        step_min = query.get("step_min")
        step_max = query.get("step_max")
        if roles:
            role_set = {str(x) for x in roles}
            items = [m for m in items if m.role in role_set]
        if step_min is not None:
            items = [m for m in items if m.step_id >= int(step_min)]
        if step_max is not None:
            items = [m for m in items if m.step_id <= int(step_max)]
        return items

    def _all_messages_unlocked(self) -> List[HistoryMessage]:
        return [*self._base_messages, *self._messages]


def compact_history(**kwargs: Any) -> CompactHistory:
    """Convenience builder for the compact history preset."""

    return CompactHistory(**kwargs)


__all__ = [
    "CompactConfig",
    "CompactHistory",
    "CompactionController",
    "MessageGrouper",
    "MicroCompactor",
    "SummaryCompactor",
    "compact_history",
]
