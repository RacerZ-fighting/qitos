from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from examples._support import SequenceModel
from qitos import (
    Action,
    AgentModule,
    Decision,
    Engine,
    History,
    HistoryPolicy,
    StateSchema,
    ToolRegistry,
    tool,
)
from qitos.engine import RuntimeBudget
from qitos.core.history import (
    HistoryMessage,
    HistorySnapshot,
    select_recent_history,
)
from qitos.kit.history import CompactHistory, MessageGrouper
from qitos.kit.parser import ReActTextParser
from qitos.models import Model, ModelStreamChunk


def test_message_grouper_prefers_step_rounds() -> None:
    from qitos.core.history import HistoryMessage

    grouper = MessageGrouper()
    groups = grouper.group(
        [
            HistoryMessage(role="system", content="s0", step_id=0),
            HistoryMessage(role="user", content="u0", step_id=0),
            HistoryMessage(role="assistant", content="a0", step_id=0),
            HistoryMessage(role="user", content="u1", step_id=1),
            HistoryMessage(role="assistant", content="a1", step_id=1),
        ]
    )

    assert len(groups) == 2
    assert [len(group) for group in groups] == [3, 2]
    assert [msg.step_id for msg in groups[-1]] == [1, 1]


def test_compact_history_keeps_active_native_tool_round_atomic() -> None:
    history = CompactHistory(
        max_tokens=1,
        auto_compact=True,
    )
    history.append(HistoryMessage(role="user", content="old", step_id=0))
    history.append(
        HistoryMessage(
            role="assistant",
            content=None,
            step_id=1,
            native_items=[
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": "{}",
                }
            ],
        )
    )
    history.append(
        HistoryMessage(
            role="tool",
            content="result",
            step_id=1,
            tool_call_id="call_1",
        )
    )

    retrieved = history.retrieve({"max_tokens": 1, "auto_compact": True})

    assert [message.role for message in retrieved[-2:]] == ["assistant", "tool"]
    assert retrieved[-2].native_items[0]["call_id"] == "call_1"


def test_compact_history_does_not_count_evict_canonical_messages_by_default() -> None:
    history = CompactHistory(auto_compact=False)
    for step_id in range(120):
        history.append(
            HistoryMessage(role="user", content=f"message {step_id}", step_id=step_id)
        )

    assert len(history.messages) == 120
    assert history.evict() == 0


def test_history_snapshot_stops_before_an_inflight_tool_transaction() -> None:
    history = CompactHistory(auto_compact=False)
    completed = HistoryMessage(role="user", content="known", step_id=0)
    history.append(completed)
    history.append(
        HistoryMessage(
            role="assistant",
            content="",
            step_id=1,
            tool_calls=[
                {"id": "call_a", "function": {"name": "a", "arguments": "{}"}},
                {"id": "call_b", "function": {"name": "b", "arguments": "{}"}},
            ],
        )
    )
    history.append(
        HistoryMessage(
            role="tool",
            content="a done",
            step_id=1,
            tool_call_id="call_a",
        )
    )

    snapshot = history.snapshot()

    assert snapshot.messages == (completed,)


def test_forked_histories_append_independently() -> None:
    parent = CompactHistory(auto_compact=False)
    shared = HistoryMessage(role="user", content="shared", step_id=0)
    parent_only = HistoryMessage(role="assistant", content="parent", step_id=1)
    first_only = HistoryMessage(role="assistant", content="first", step_id=1)
    second_only = HistoryMessage(role="assistant", content="second", step_id=1)
    parent.append(shared)
    snapshot = parent.snapshot()
    first = parent.fork(snapshot)
    second = parent.fork(snapshot)

    parent.append(parent_only)
    first.append(first_only)
    second.append(second_only)

    assert parent.messages == [shared, parent_only]
    assert first.messages == [shared, first_only]
    assert second.messages == [shared, second_only]


def test_recent_history_never_splits_generic_tool_transaction() -> None:
    messages = [
        HistoryMessage(
            role="assistant",
            content=None,
            step_id=1,
            tool_calls=[
                {"id": "call_1", "function": {"name": "lookup"}},
                {"id": "call_2", "function": {"name": "read"}},
            ],
        ),
        HistoryMessage(role="tool", content="one", step_id=1, tool_call_id="call_1"),
        HistoryMessage(role="tool", content="two", step_id=1, tool_call_id="call_2"),
    ]

    selected = select_recent_history(messages, max_items=1)

    assert selected == messages


def test_recent_history_drops_whole_older_transaction_instead_of_orphaning_result() -> (
    None
):
    messages = [
        HistoryMessage(
            role="assistant",
            content=None,
            step_id=1,
            tool_calls=[{"id": "call_1", "function": {"name": "lookup"}}],
        ),
        HistoryMessage(role="tool", content="result", step_id=1, tool_call_id="call_1"),
        HistoryMessage(role="user", content="continue", step_id=2),
    ]

    selected = select_recent_history(messages, max_items=2)

    assert selected == [messages[-1]]


def test_recent_history_merges_cross_step_result_with_declaring_call() -> None:
    messages = [
        HistoryMessage(
            role="assistant",
            content=None,
            step_id=1,
            tool_calls=[{"id": "call_1", "function": {"name": "lookup"}}],
        ),
        HistoryMessage(role="tool", content="result", step_id=2, tool_call_id="call_1"),
    ]

    assert select_recent_history(messages, max_items=1) == messages


def test_recent_history_merges_native_output_with_declaring_call() -> None:
    messages = [
        HistoryMessage(
            role="assistant",
            content=None,
            step_id=1,
            native_items=[
                {
                    "type": "function_call",
                    "call_id": "call_native",
                    "name": "lookup",
                    "arguments": "{}",
                }
            ],
        ),
        HistoryMessage(
            role="tool",
            content="result",
            step_id=2,
            native_items=[
                {
                    "type": "function_call_output",
                    "call_id": "call_native",
                    "output": "result",
                }
            ],
        ),
    ]

    assert select_recent_history(messages, max_items=1) == messages


def test_reused_tool_call_id_keeps_parent_and_child_rounds_independent() -> None:
    parent_call = HistoryMessage(
        role="assistant",
        content=None,
        step_id=0,
        tool_calls=[{"id": "shell_1", "function": {"name": "shell"}}],
    )
    parent_result = HistoryMessage(
        role="tool", content="parent", step_id=0, tool_call_id="shell_1"
    )
    child_call = HistoryMessage(
        role="assistant",
        content=None,
        step_id=0,
        tool_calls=[{"id": "shell_1", "function": {"name": "shell"}}],
    )
    child_result = HistoryMessage(
        role="tool", content="child", step_id=0, tool_call_id="shell_1"
    )

    selected = select_recent_history(
        [parent_call, parent_result, child_call, child_result], max_items=1
    )

    assert selected == [child_call, child_result]


def test_snapshot_stops_before_second_unanswered_use_of_same_call_id() -> None:
    parent_call = HistoryMessage(
        role="assistant",
        content=None,
        step_id=0,
        tool_calls=[{"id": "shell_1", "function": {"name": "shell"}}],
    )
    parent_result = HistoryMessage(
        role="tool", content="parent", step_id=0, tool_call_id="shell_1"
    )
    child_call = HistoryMessage(
        role="assistant",
        content=None,
        step_id=0,
        tool_calls=[{"id": "shell_1", "function": {"name": "shell"}}],
    )

    snapshot = HistorySnapshot.from_messages([parent_call, parent_result, child_call])

    assert snapshot.messages == (parent_call, parent_result)


def test_compact_estimate_deduplicates_generic_native_mirrors() -> None:
    native_items = [
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "inspect"}],
        },
        {
            "type": "function_call",
            "call_id": "call_native",
            "name": "lookup",
            "arguments": '{"key":"target"}',
        },
    ]
    native_only = HistoryMessage(
        role="assistant",
        content=None,
        step_id=1,
        native_items=native_items,
    )
    mirrored = HistoryMessage(
        role="assistant",
        content=None,
        step_id=1,
        reasoning_content="inspect",
        tool_calls=[
            {
                "id": "call_native",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": '{"key":"target"}',
                },
            }
        ],
        native_items=native_items,
    )
    history = CompactHistory(auto_compact=False)

    assert history._controller._estimate_tokens([mirrored]) == (
        history._controller._estimate_tokens([native_only])
    )


def test_compact_history_emits_microcompact_and_summary_events() -> None:
    from qitos.core.history import HistoryMessage

    history = CompactHistory(max_tokens=90, keep_last_rounds=1, hard_window=20)
    for idx in range(6):
        role = "user" if idx % 2 == 0 else "assistant"
        history.append(
            HistoryMessage(
                role=role,
                content=(f"message {idx} " + "with verbose context " * 300).strip(),
                step_id=idx,
                metadata={"source": "engine"},
            )
        )

    retrieved = history.retrieve(
        query={
            "roles": ["user", "assistant"],
            "max_items": 12,
            "max_tokens": 90,
            "pending_content": "next prompt with another long continuation",
        }
    )
    events = history.consume_runtime_events()
    metadata = history.get_last_message_metadata()

    assert retrieved
    assert retrieved[0].metadata.get("summary") is True
    assert any(
        event.get("stage") == "context_history"
        and (event.get("context") or {}).get("stage") == "warning"
        for event in events
    )
    assert any(
        event.get("stage") == "context_history"
        and (event.get("context") or {}).get("stage") == "microcompact_applied"
        for event in events
    )
    assert any(
        event.get("stage") == "context_history"
        and (event.get("context") or {}).get("stage") == "summary_compact_applied"
        for event in events
    )
    assert metadata[0].get("summary") is True
    assert metadata[0].get("source") == "compact_history"


def test_microcompaction_runs_before_force_summary_threshold() -> None:
    from qitos.kit.history.compact_history import CompactConfig

    history = CompactHistory(
        config=CompactConfig(
            max_tokens=100,
            keep_last_rounds=1,
            compact_long_messages_over_chars=100,
            microcompact_preview_chars=60,
        )
    )
    history.append(
        HistoryMessage(
            role="assistant",
            content=None,
            step_id=0,
            tool_calls=[{"id": "old", "function": {"name": "lookup"}}],
        )
    )
    history.append(
        HistoryMessage(role="tool", content="x" * 320, step_id=0, tool_call_id="old")
    )
    history.append(HistoryMessage(role="user", content="continue", step_id=1))

    retrieved = history.retrieve(query={"max_tokens": 100})
    stages = [event["context"]["stage"] for event in history.consume_runtime_events()]

    assert "microcompact_applied" in stages
    assert "summary_compact_applied" not in stages
    assert not any(message.metadata.get("summary") for message in retrieved)
    assert any(
        message.metadata.get("compaction_mode") == "micro" for message in retrieved
    )


def test_hard_compaction_resummarizes_all_but_latest_complete_round() -> None:
    from qitos.kit.history.compact_history import CompactConfig

    llm = _RecordingSummaryLLM()
    history = CompactHistory(
        llm=llm,
        config=CompactConfig(
            max_tokens=80,
            keep_last_rounds=2,
            compact_long_messages_over_chars=10_000,
        ),
    )
    history.append(HistoryMessage(role="user", content="old " * 120, step_id=0))
    history.append(HistoryMessage(role="assistant", content="middle " * 120, step_id=1))
    latest = HistoryMessage(role="user", content="latest", step_id=2)
    history.append(latest)

    retrieved = history.retrieve(query={"max_tokens": 80})
    events = history.consume_runtime_events()
    stages = [event["context"]["stage"] for event in events]

    assert "summary_compact_applied" in stages
    assert "hard_compact_applied" in stages
    assert retrieved[0].metadata.get("summary") is True
    assert retrieved[0].metadata.get("compaction_level") == 3
    assert retrieved[-1] is latest
    assert all(message.step_id != 1 for message in retrieved[1:])
    assert len(llm.prompts) == 2


class _RecordingSummaryLLM:
    """Capture every summary request and hand back a numbered summary."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, messages: list[dict[str, Any]]) -> str:
        self.prompts.append(str(messages[-1]["content"]))
        return f"SUMMARY_{len(self.prompts)}"


def test_summary_compactor_persists_summary_without_drafting_scratchpad() -> None:
    from qitos.kit.history.compact_history import CompactConfig, SummaryCompactor

    class _TaggedSummaryLLM:
        def __call__(self, messages: list[dict[str, Any]]) -> str:
            assert "<analysis>" not in str(messages[0]["content"])
            return (
                "<analysis>private drafting scratchpad</analysis>"
                "<summary>durable continuation state</summary>"
            )

    summary = SummaryCompactor(
        CompactConfig(),
        llm=_TaggedSummaryLLM(),
    ).summarize([HistoryMessage(role="user", content="continue the task", step_id=1)])

    assert summary == "durable continuation state"
    assert "scratchpad" not in summary


def _numbered_history(
    count: int,
    *,
    llm: Any = None,
    chars: int = 200,
    summary_input_message_limit: int = 64,
) -> tuple[Any, Any]:
    from qitos.core.history import HistoryMessage
    from qitos.kit.history.compact_history import CompactConfig

    config = CompactConfig(
        max_tokens=200,
        keep_last_rounds=2,
        compact_long_messages_over_chars=10,
        summary_input_message_limit=summary_input_message_limit,
        auto_compact=True,
    )
    history = CompactHistory(llm=llm, config=config)
    for idx in range(count):
        history.append(
            HistoryMessage(
                role="user" if idx % 2 == 0 else "assistant",
                content=f"ITEM_{idx:02d} " + "x" * chars,
                step_id=idx,
            )
        )
    return history, config


def test_summary_input_covers_full_prefix_including_earliest() -> None:
    """Regression for #36: the replaced prefix head must reach the summary model."""

    llm = _RecordingSummaryLLM()
    history, config = _numbered_history(
        35,
        llm=llm,
        summary_input_message_limit=28,
    )

    retrieved = history.retrieve(query={"max_tokens": 200})

    assert retrieved[0].metadata.get("summary") is True
    prompt = llm.prompts[-1]
    covered = {int(token.split("_")[1]) for token in re.findall(r"ITEM_\d\d", prompt)}
    # 33 messages are replaced by the summary; every one of them must be covered,
    # including the ones beyond `summary_input_message_limit`.
    assert covered == set(range(33))
    assert len(covered) > config.summary_input_message_limit


def test_auto_compaction_summarizes_before_applying_message_count_windows() -> None:
    llm = _RecordingSummaryLLM()
    history, _ = _numbered_history(35, llm=llm)

    retrieved = history.retrieve(
        query={"max_items": 10, "max_tokens": 200, "auto_compact": True}
    )

    assert retrieved[0].metadata.get("summary") is True
    covered = {
        int(token.split("_")[1]) for token in re.findall(r"ITEM_\d\d", llm.prompts[-1])
    }
    assert covered == set(range(33))


def test_summary_input_elides_long_older_messages_without_dropping_them() -> None:
    llm = _RecordingSummaryLLM()
    history, _ = _numbered_history(
        35,
        llm=llm,
        chars=4000,
        summary_input_message_limit=28,
    )

    history.retrieve(query={"max_tokens": 200})

    first_covered = {
        int(token.split("_")[1]) for token in re.findall(r"ITEM_\d\d", llm.prompts[0])
    }
    hard_covered = {
        int(token.split("_")[1]) for token in re.findall(r"ITEM_\d\d", llm.prompts[-1])
    }
    assert first_covered == set(range(33))
    assert hard_covered == set(range(34))
    # Older turns are shortened rather than removed, so the request stays bounded.
    assert "elided" in llm.prompts[-1]
    assert len(llm.prompts[-1]) < 35 * 4000


def test_summary_trace_reports_real_input_counts_and_ranges() -> None:
    llm = _RecordingSummaryLLM()
    history, _ = _numbered_history(35, llm=llm)

    retrieved = history.retrieve(query={"max_tokens": 200})
    events = history.consume_runtime_events()

    summary_events = [
        event["context"]
        for event in events
        if (event.get("context") or {}).get("stage") == "summary_compact_applied"
    ]
    assert summary_events
    context = summary_events[-1]
    meta = retrieved[0].metadata
    for payload in (context, meta):
        assert payload["summarized_message_count"] == 33
        assert payload["summary_input_message_count"] == 33
        assert payload["summary_dropped_message_count"] == 0
        assert payload["summarized_step_range"] == [0, 32]
        assert payload["summary_input_step_range"] == [0, 32]
        assert payload["summary_input_mode"] in {"full", "microcompacted"}
        assert re.fullmatch(r"[0-9a-f]{64}", payload["source_digest"])
        assert payload["source_history_version"] == history.history_version


def test_repeated_compaction_reuses_summary_for_unchanged_source() -> None:
    llm = _RecordingSummaryLLM()
    history, _ = _numbered_history(35, llm=llm)

    first = history.retrieve(query={"max_tokens": 200})
    assert first[0].metadata.get("built_on_prior_summary") is False
    assert history.last_summary == "SUMMARY_1"

    second = history.retrieve(query={"max_tokens": 200})
    assert second[0].metadata.get("built_on_prior_summary") is False
    assert second[0].metadata.get("summary_cache_hit") is True
    assert len(llm.prompts) == 1
    assert history.last_summary == "SUMMARY_1"


def test_smaller_budget_does_not_reuse_an_oversized_cached_summary() -> None:
    class _LongSummaryLLM:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, messages: list[dict[str, Any]]) -> str:
            _ = messages
            self.calls += 1
            return "S" * 2_000

    llm = _LongSummaryLLM()
    history, _ = _numbered_history(35, llm=llm)

    first = history.retrieve(query={"max_tokens": 200})
    second = history.retrieve(query={"max_tokens": 100})

    assert first[0].metadata["summary_cache_hit"] is False
    assert second[0].metadata["summary_cache_hit"] is False
    assert second[0].metadata["summary_budget"] == 100
    assert len(str(second[0].content)) <= second[0].metadata["summary_char_limit"]
    assert len(str(second[0].content)) < len(str(first[0].content))


def test_compaction_carries_prior_summary_after_source_window_advances() -> None:
    llm = _RecordingSummaryLLM()
    history, _ = _numbered_history(35, llm=llm)
    history.retrieve(query={"max_tokens": 200})
    prior_summary = history.last_summary
    assert prior_summary is not None
    prompt_count = len(llm.prompts)

    for step_id in range(35, 105):
        history.append(
            HistoryMessage(
                role="user" if step_id % 2 == 0 else "assistant",
                content=f"ITEM_{step_id:03d} " + "x" * 200,
                step_id=step_id,
            )
        )
    retrieved = history.retrieve(query={"max_tokens": 200})

    incremental_prompts = llm.prompts[prompt_count:]
    assert any(prior_summary in prompt for prompt in incremental_prompts)
    assert all("ITEM_00 " not in prompt for prompt in incremental_prompts)
    assert retrieved[0].metadata["built_on_prior_summary"] is True
    assert retrieved[0].metadata["summary_carried_message_count"] > 0


def test_compaction_discards_stale_projection_when_history_changes() -> None:
    class _AppendingSummaryLLM:
        def __init__(self) -> None:
            self.history: CompactHistory | None = None
            self.appended = False

        def __call__(self, messages: list[dict[str, Any]]) -> str:
            _ = messages
            assert self.history is not None
            if not self.appended:
                self.appended = True
                self.history.append(
                    HistoryMessage(
                        role="user",
                        content="concurrent update",
                        step_id=99,
                    )
                )
            return "STALE_SUMMARY"

    llm = _AppendingSummaryLLM()
    history, _ = _numbered_history(35, llm=llm)
    llm.history = history

    with pytest.raises(RuntimeError, match="history changed"):
        history.retrieve(query={"max_tokens": 200})

    assert history.last_summary is None
    assert history.messages[-1].content == "concurrent update"


def test_configured_summary_failure_does_not_fall_back_to_lossy_heuristics() -> None:
    class _FailingSummaryLLM:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, messages: list[dict[str, Any]]) -> str:
            _ = messages
            self.calls += 1
            raise RuntimeError("summary provider unavailable")

    llm = _FailingSummaryLLM()
    history, _ = _numbered_history(35, llm=llm)
    before = history.messages

    for _ in range(4):
        with pytest.raises(RuntimeError):
            history.retrieve(query={"max_tokens": 200})

    assert llm.calls == 3
    assert history.messages == before
    assert history.last_summary is None


def test_retrieve_does_not_mutate_stored_history() -> None:
    llm = _RecordingSummaryLLM()
    history, _ = _numbered_history(35, llm=llm)

    before = [msg.content for msg in history.messages]
    history.retrieve(query={"max_tokens": 200})
    history.retrieve(query={"roles": ["user"], "max_items": 10, "max_tokens": 200})

    assert [msg.content for msg in history.messages] == before


def test_compact_ratio_moves_the_compaction_trigger() -> None:
    from qitos.engine._context_runtime import _ContextRuntime
    from qitos.engine.states import ContextConfig, ContextTelemetry

    class _LLM:
        context_window = 100_000
        max_tokens = 1_000

    class _Engine:
        class agent:
            llm = _LLM()

        def _estimate_tokens(self, payload: Any) -> int:
            return 0

    runtime = _ContextRuntime(_Engine())

    def probe(compact_ratio: float) -> tuple[int, int, int]:
        runtime.config = ContextConfig(
            safety_reserve_tokens=0,
            compact_ratio=compact_ratio,
        )
        budget = runtime.resolve_request_budget(_LLM())
        ceiling = budget["available_input_budget"]
        soft_target = budget["soft_input_target"]
        telemetry = ContextTelemetry(
            available_input_budget=ceiling,
            system_prompt_tokens=0,
            prepared_tokens=0,
        )
        return ceiling, soft_target, runtime.compact_trigger_budget(telemetry)

    eager_ceiling, eager_soft, eager_trigger = probe(0.20)
    lazy_ceiling, lazy_soft, lazy_trigger = probe(0.95)

    assert eager_trigger < lazy_trigger
    assert eager_ceiling == lazy_ceiling == 99_000
    assert eager_soft == 19_800
    assert lazy_soft == 94_050
    # warning_ratio < compact_ratio < overflow(1.0) must hold at the defaults.
    default_ceiling, _, default_trigger = probe(ContextConfig().compact_ratio)
    assert (
        default_ceiling * ContextConfig().warning_ratio
        < default_trigger
        < default_ceiling
    )
    assert ContextConfig().compact_ratio == 0.80

    runtime.config = ContextConfig(
        safety_reserve_tokens=0,
        compact_ratio=0.80,
    )
    anchored = ContextTelemetry(
        available_input_budget=1_000,
        hard_input_budget=1_000,
        system_prompt_tokens=100,
        prepared_tokens=100,
    )
    # History strategies count pending_content themselves. Their query budget
    # is therefore 700: 600 history tokens plus the 100-token pending prompt.
    assert runtime.compact_trigger_budget(anchored) == 700
    assert runtime.compact_trigger_budget(anchored) - anchored.prepared_tokens == 600
    assert runtime.begin_reactive_compaction() == {
        "attempt": 1,
        "factor": 0.70,
        "max_attempts": 3,
    }
    assert runtime.compact_trigger_budget(anchored) == 520
    assert runtime.begin_reactive_compaction()["factor"] == 0.50
    assert runtime.compact_trigger_budget(anchored) == 400
    assert runtime.begin_reactive_compaction()["factor"] == 0.35
    assert runtime.compact_trigger_budget(anchored) == 310
    assert runtime.begin_reactive_compaction() is None


def test_context_budget_counts_tool_schemas_and_forces_at_exactly_eighty_percent() -> (
    None
):
    from qitos.engine._context_runtime import _ContextRuntime
    from qitos.engine.states import ContextConfig, ContextTelemetry

    class _CountingLLM:
        context_window = 1_000
        max_tokens = 0

        def count_tokens(self, payload: Any) -> int:
            return len(str(payload))

    class _Engine:
        class agent:
            llm = _CountingLLM()

        def _estimate_tokens(self, payload: Any) -> int:
            return len(str(payload))

    runtime = _ContextRuntime(_Engine())
    runtime.config = ContextConfig(safety_reserve_tokens=0, compact_ratio=0.80)
    llm = _CountingLLM()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object"},
            },
        }
    ]
    without_tools = runtime.build_pre_request(
        llm=llm,
        system_prompt="system",
        prepared="task",
    )
    with_tools = runtime.build_pre_request(
        llm=llm,
        system_prompt="system",
        prepared="task",
        request_options={"tools": tools, "temperature": 0.1},
    )

    assert with_tools.request_overhead_tokens > 0
    assert runtime.compact_trigger_budget(with_tools) < runtime.compact_trigger_budget(
        without_tools
    )

    messages = [{"role": "user", "content": "task"}]
    final = runtime.finalize_assembled_input(
        llm=llm,
        telemetry=with_tools,
        messages=messages,
        request_options={"tools": tools, "temperature": 0.1},
        compact_events=[],
    )
    assert final.input_tokens_total == llm.count_tokens(messages) + llm.count_tokens(
        {"tools": tools}
    )

    below = ContextTelemetry(hard_input_budget=100, input_tokens_total=79)
    boundary = ContextTelemetry(hard_input_budget=100, input_tokens_total=80)
    fractional_below = ContextTelemetry(
        hard_input_budget=101,
        input_tokens_total=80,
    )
    fractional_boundary = ContextTelemetry(
        hard_input_budget=101,
        input_tokens_total=81,
    )
    assert not runtime.should_force_compact(below)
    assert runtime.should_force_compact(boundary)
    assert not runtime.should_force_compact(fractional_below)
    assert runtime.should_force_compact(fractional_boundary)


def test_explicit_model_input_limit_takes_precedence_when_safer() -> None:
    from qitos.engine._context_runtime import _ContextRuntime
    from qitos.engine.states import ContextConfig

    class _LLM:
        context_window = 1_000
        max_tokens = 100
        max_input_tokens = 700

    class _Engine:
        class agent:
            llm = _LLM()

        def _estimate_tokens(self, payload: Any) -> int:
            return len(str(payload))

    runtime = _ContextRuntime(_Engine())
    runtime.config = ContextConfig(safety_reserve_tokens=0)

    budget = runtime.resolve_request_budget(_LLM())

    assert budget["hard_input_budget"] == 700
    assert budget["input_budget_source"] == "model_max_input_tokens"


def test_derived_input_limit_takes_precedence_over_unsafe_model_hint() -> None:
    from qitos.engine._context_runtime import _ContextRuntime
    from qitos.engine.states import ContextConfig

    class _LLM:
        context_window = 1_000
        max_tokens = 100
        max_input_tokens = 950

    class _Engine:
        class agent:
            llm = _LLM()

        def _estimate_tokens(self, payload: Any) -> int:
            return len(str(payload))

    runtime = _ContextRuntime(_Engine())
    runtime.config = ContextConfig(safety_reserve_tokens=0)

    budget = runtime.resolve_request_budget(_LLM())

    assert budget["hard_input_budget"] == 900
    assert budget["input_budget_source"] == "context_window_minus_output_reserve"


def test_provider_overflow_retries_do_not_mutate_canonical_history() -> None:
    from qitos.core.errors import StopReason
    from qitos.engine._context_runtime import ContextOverflowError
    from qitos.engine.states import RuntimePhase

    agent = CompactDemoAgent()
    history = CompactHistory(max_tokens=100)
    history.append(HistoryMessage(role="user", content="keep me", step_id=0))
    agent.history = history
    engine = Engine(agent=agent, budget=RuntimeBudget(max_steps=1))
    state = agent.init_state("task")
    before = history.messages

    for _ in range(3):
        assert engine._control_runtime.recover(
            state,
            RuntimePhase.DECIDE,
            ContextOverflowError("provider rejected prompt"),
        )
        assert history.messages == before

    assert not engine._control_runtime.recover(
        state,
        RuntimePhase.DECIDE,
        ContextOverflowError("provider rejected prompt"),
    )
    assert state.stop_reason == StopReason.CONTEXT_OVERFLOW.value
    assert history.messages == before


@dataclass
class CompactDemoState(StateSchema):
    logs: list[str] = field(default_factory=list)


class CompactDemoAgent(AgentModule[CompactDemoState, dict[str, Any], Action]):
    def __init__(self):
        registry = ToolRegistry()

        @tool(name="add")
        def add(a: int, b: int) -> int:
            return a + b

        registry.register(add)
        super().__init__(tool_registry=registry)

    def init_state(self, task: str, **kwargs: Any) -> CompactDemoState:
        return CompactDemoState(task=task, max_steps=3)

    def reduce(
        self,
        state: CompactDemoState,
        observation: dict[str, Any],
        decision: Decision[Action],
    ) -> CompactDemoState:
        action_results = (
            observation.get("action_results", [])
            if isinstance(observation, dict)
            else []
        )
        if action_results:
            state.logs.append(str(action_results[0]))
        return state


def test_force_compaction_does_not_commit_staged_prompt_history() -> None:
    from qitos.engine.states import ContextConfig

    class _ThresholdModel(Model):
        max_input_tokens = 100

        def __init__(self) -> None:
            super().__init__(
                model="threshold-model",
                context_window=100,
                max_tokens=0,
                temperature=None,
            )
            self.calls = 0

        def count_tokens(self, payload: Any) -> int:
            return max(1, len(str(payload)) // 10)

        def count_request_tokens(
            self,
            messages: list[dict[str, Any]],
            request_options: dict[str, Any] | None = None,
        ) -> int:
            _ = messages, request_options
            return 80

        async def stream(
            self,
            messages: list[dict[str, Any]],
            *,
            deadline_monotonic: float | None = None,
            **kwargs: Any,
        ) -> AsyncIterator[ModelStreamChunk]:
            _ = messages, deadline_monotonic, kwargs
            self.calls += 1
            raise AssertionError(
                "the provider must not be called at the force threshold"
            )
            yield ModelStreamChunk()  # pragma: no cover

    model = _ThresholdModel()
    history = CompactHistory()
    agent = CompactDemoAgent()
    agent.llm = model
    agent.model_parser = ReActTextParser()
    agent.history = history

    result = Engine(
        agent=agent,
        budget=RuntimeBudget(max_steps=10),
        context_config=ContextConfig(
            compact_ratio=0.80,
            safety_reserve_tokens=0,
        ),
    ).run("keep canonical history transactional")

    assert model.calls == 0
    assert history.messages == []
    assert result.state.stop_reason == "context_overflow"
    assert (
        sum(
            event.payload.get("stage") == "context_force_compaction"
            for event in result.events
        )
        == 4
    )


def test_engine_surfaces_compact_events_and_history_metadata() -> None:
    model = SequenceModel(
        [
            "Action: add(a=20, b=22)",
            "Action: add(a=1, b=1)",
            "Final Answer: 42",
        ],
        model="dummy-compact",
    )

    class LLMCompactAgent(CompactDemoAgent):
        def __init__(self):
            super().__init__()
            self.llm = model
            self.model_parser = ReActTextParser()
            self.history = CompactHistory(
                max_tokens=110, keep_last_rounds=1, hard_window=24
            )

        def build_system_prompt(self, state: CompactDemoState) -> str | None:
            return "Compact system prompt"

        def prepare(self, state: CompactDemoState) -> str:
            return (
                f"Task={state.task}\n"
                f"Step={state.current_step}\n"
                + ("Observation context and scratchpad detail. " * 50)
            ).strip()

        def decide(self, state: CompactDemoState, observation: dict[str, Any]):
            return None

    result = Engine(
        agent=LLMCompactAgent(),
        budget=RuntimeBudget(max_steps=3),
        history_policy=HistoryPolicy(max_messages=10, max_tokens=110),
    ).run("compute")

    assert result.state.final_result == "42"
    assert len(model.calls) == 3
    compact_stages = [
        (event.payload.get("context") or {}).get("stage")
        for event in result.events
        if getattr(event.phase, "value", event.phase) == "DECIDE"
        and event.payload.get("stage") == "context_history"
    ]
    assert "warning" in compact_stages
    assert any(
        stage in {"microcompact_applied", "summary_compact_applied"}
        for stage in compact_stages
    )

    model_input_events = [
        event for event in result.events if event.payload.get("stage") == "model_input"
    ]
    assert model_input_events
    history_meta = model_input_events[-1].payload.get("history_messages_meta", [])
    assert isinstance(history_meta, list)
    assert history_meta
    assert any(item.get("summary") or item.get("compacted") for item in history_meta)
    context = model_input_events[-1].payload.get("context", {})
    assert context.get("input_tokens_total", 0) > 0


def test_engine_falls_back_to_bounded_canonical_history_when_compaction_fails() -> None:
    class _FailingHistory(History):
        def __init__(self) -> None:
            self._messages = [
                HistoryMessage(
                    role="user",
                    content="canonical context must survive",
                    step_id=0,
                )
            ]

        def append(self, message: HistoryMessage) -> None:
            self._messages.append(message)

        def retrieve(
            self,
            query: dict[str, Any] | None = None,
            state: Any = None,
            observation: Any = None,
        ) -> Any:
            raise RuntimeError("summary backend failed")

        def summarize(self, max_items: int = 5) -> str:
            return ""

        def evict(self) -> int:
            return 0

        def reset(self, run_id: str | None = None) -> None:
            _ = run_id

        @property
        def messages(self) -> list[HistoryMessage]:
            return list(self._messages)

    model = SequenceModel(
        ["Final Answer: recovered"],
        model="history-fallback",
    )

    agent = CompactDemoAgent()
    agent.llm = model
    agent.model_parser = ReActTextParser()
    agent.history = _FailingHistory()

    result = Engine(agent=agent, budget=RuntimeBudget(max_steps=1)).run("compute")

    assert "canonical context must survive" in str(model.calls[0])
    assert any(
        event.payload.get("stage") == "context_history"
        and (event.payload.get("context") or {}).get("stage")
        == "compact_failed_fallback"
        for event in result.events
    )


def test_native_tool_history_preserves_incomplete_tail_as_an_atomic_round() -> None:
    from qitos.engine._model_runtime import _ModelRuntime

    runtime = object.__new__(_ModelRuntime)
    history = [
        {"role": "assistant", "_step_id": 1, "tool_calls": [{"id": "a"}, {"id": "b"}]},
        {"role": "tool", "_step_id": 1, "tool_call_id": "a", "content": "one"},
        {"role": "tool", "_step_id": 1, "tool_call_id": "b", "content": "two"},
        {"role": "assistant", "_step_id": 2, "tool_calls": [{"id": "dangling"}]},
        {"role": "assistant", "_step_id": 3, "tool_calls": [{"id": "c"}]},
        {"role": "tool", "_step_id": 3, "tool_call_id": "c", "content": "three"},
    ]
    result = runtime._trim_native_tool_history(history, max_rounds=2)
    assert {m.get("_step_id") for m in result} == {2, 3}
    consistent = runtime._ensure_chain_consistency(result)
    dangling_index = next(
        index
        for index, message in enumerate(consistent)
        if message.get("_step_id") == 2
    )
    placeholder = consistent[dangling_index + 1]
    assert placeholder["tool_call_id"] == "dangling"
    assert "tool_call_not_completed" in str(placeholder["content"])
