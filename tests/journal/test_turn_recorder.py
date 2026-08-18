"""JournalTurnTransaction integration over the real JSONL journal."""

from __future__ import annotations

import pytest

from qitos.core.agent_loop import (
    AgentContext,
    AgentLoopConfig,
    AgentRunStatus,
    NextTurnUpdate,
    TurnConfigSnapshot,
    run_agent_loop,
)
from qitos.core.cancellation import CancelToken
from qitos.core.journal import JournalError, JournalRecordRef, JournalRecordType
from qitos.core.message import (
    AssistantMessage,
    UserMessage,
    message_from_dict,
)
from qitos.core.model_response import ModelUsage
from qitos.core.thinking import ThinkingLevel
from qitos.core.tool import tool
from qitos.core.tool_registry import ToolRegistry
from qitos.core.tool_result import ToolResult
from qitos.kit.journal import (
    InMemoryJournalStore,
    InMemorySessionJournal,
    JsonlSessionJournal,
    JournalTurnTransaction,
    RecoveredRecorderState,
    close_crashed_tool_calls,
    recover_run_outcome,
    recover_session,
)

from tests.core.agent_fakes import ScriptedModel, text_events, tool_call_wire, tool_events


@tool(name="echo")
def _echo(text: str) -> str:
    return f"echo:{text}"


async def _replay(tmp_path, run_id: str):
    journal = JsonlSessionJournal(tmp_path)
    await journal.open(run_id)
    records = await journal.replay()
    await journal.close()
    return records


@pytest.mark.asyncio
async def test_full_run_records_transcript_entries_and_operation_references(
    tmp_path,
) -> None:
    journal = JsonlSessionJournal(tmp_path)
    transaction = await JournalTurnTransaction.create(
        journal, "run-journal-1", {"purpose": "test"}
    )
    # The loop drains steering before the first turn and after each turn;
    # the second drain injects the message between turn 0 and turn 1.
    steering_drains = [[], [UserMessage(content="steer")], []]

    def _steering():
        return steering_drains.pop(0) if steering_drains else []

    first_model = ScriptedModel(
        [tool_events([tool_call_wire("c1", "echo", {"text": "x"})])]
    )
    second_model = ScriptedModel([text_events("done")], model="other-model")

    def _prepare(_context):
        return NextTurnUpdate(model=second_model)

    context = AgentContext(messages=[], tools=ToolRegistry().register(_echo).freeze())
    config = AgentLoopConfig(
        model=first_model,
        run_id="run-journal-1",
        transaction=transaction,
        get_steering_messages=_steering,
        prepare_next_turn=lambda _context: _prepare(_context),
    )
    result = await run_agent_loop([UserMessage(content="go")], context, config, None)
    assert result.status is AgentRunStatus.COMPLETED
    await journal.close()

    records = await _replay(tmp_path, "run-journal-1")
    types = [record.type for record in records]
    assert types == [
        JournalRecordType.RUN_STARTED,
        JournalRecordType.TRANSCRIPT_MESSAGE,
        JournalRecordType.INPUT_ACCEPTED,
        JournalRecordType.MODEL_CHANGE,
        JournalRecordType.THINKING_CHANGE,
        JournalRecordType.TOOLS_CHANGE,
        JournalRecordType.TRANSCRIPT_MESSAGE,
        JournalRecordType.MODEL_COMPLETED,
        JournalRecordType.TOOL_STARTED,
        JournalRecordType.TRANSCRIPT_MESSAGE,
        JournalRecordType.TOOL_TERMINAL,
        JournalRecordType.STEP_COMMITTED,
        JournalRecordType.MODEL_CHANGE,
        JournalRecordType.TRANSCRIPT_MESSAGE,
        JournalRecordType.MODEL_COMPLETED,
        JournalRecordType.TRANSCRIPT_MESSAGE,
        JournalRecordType.STEP_COMMITTED,
        JournalRecordType.RUN_COMPLETED,
    ]

    # Message content lives in exactly one record; operation records carry
    # references, never embedded copies.
    for record in records:
        if record.type is JournalRecordType.MODEL_COMPLETED:
            assert set(record.payload) == {"turn", "request", "message_record_id"}
        if record.type is JournalRecordType.TOOL_TERMINAL:
            assert set(record.payload) == {
                "turn",
                "call_id",
                "call",
                "message_record_id",
            }
        if record.type is JournalRecordType.STEP_COMMITTED:
            assert set(record.payload) == {
                "turn",
                "transcript_record_ids",
                "tool_terminal_record_ids",
            }
        if record.type is JournalRecordType.RUN_COMPLETED:
            assert record.payload == {
                "status": "completed",
                "error": None,
                "finalization_diagnostic": None,
            }

    # Deterministic per-turn transcript sequence ids.
    transcript_ids = [
        record.record_id
        for record in records
        if record.type is JournalRecordType.TRANSCRIPT_MESSAGE
    ]
    assert transcript_ids == [
        "run-journal-1:turn:0:transcript:0",
        "run-journal-1:turn:0:transcript:1",
        "run-journal-1:turn:0:transcript:2",
        "run-journal-1:turn:1:transcript:0",
        "run-journal-1:turn:1:transcript:1",
    ]

    # INPUT_ACCEPTED commits the prompts before the first model transaction.
    assert types.index(JournalRecordType.INPUT_ACCEPTED) < types.index(
        JournalRecordType.MODEL_COMPLETED
    )
    input_accepted = records[2]
    assert input_accepted.payload == {
        "transcript_record_ids": ["run-journal-1:turn:0:transcript:0"]
    }

    # The configuration trio is written once on turn 0; turn 1 journals only
    # the model diff (thinking and tools did not change).
    config_records = [
        record
        for record in records
        if record.type
        in {
            JournalRecordType.MODEL_CHANGE,
            JournalRecordType.THINKING_CHANGE,
            JournalRecordType.TOOLS_CHANGE,
        }
    ]
    assert [record.record_id for record in config_records] == [
        "run-journal-1:turn:0:config:model",
        "run-journal-1:turn:0:config:thinking",
        "run-journal-1:turn:0:config:tools",
        "run-journal-1:turn:1:config:model",
    ]
    assert config_records[0].payload == {
        "provider": "scripted",
        "model": "scripted-model",
        "api": "legacy",
    }
    assert config_records[1].payload == {"level": "off"}
    assert config_records[2].payload == {"active_tool_names": ["echo"]}
    assert config_records[3].payload == {
        "provider": "scripted",
        "model": "other-model",
        "api": "legacy",
    }
    # Config entries for a turn precede that turn's model record.
    first_model_index = types.index(JournalRecordType.MODEL_COMPLETED)
    assert all(
        types.index(record.type) < first_model_index
        for record in config_records[:3]
    )

    # The model record references the assistant transcript entry by id and
    # keeps the exact request audit.
    model_record = records[7]
    assert model_record.payload["message_record_id"] == transcript_ids[1]
    assert model_record.payload["request"]["run_id"] == "run-journal-1"
    assistant = message_from_dict(records[6].payload["message"])
    assert isinstance(assistant, AssistantMessage)
    assert assistant.tool_calls[0].id == "c1"

    # The terminal references the tool transcript entry; the steering user
    # message is recorded by the turn commit that accepted it. The commit
    # lists the turn's entries in conversation order (steering before the
    # assistant reply), which need not match journal append order.
    terminal = records[10]
    assert terminal.payload["message_record_id"] == transcript_ids[2]
    assert terminal.payload["call_id"] == "c1"
    turn_one_commit = records[16]
    assert turn_one_commit.payload["transcript_record_ids"] == [
        "run-journal-1:turn:1:transcript:1",
        "run-journal-1:turn:1:transcript:0",
    ]
    assert turn_one_commit.payload["tool_terminal_record_ids"] == []
    steering_message = message_from_dict(records[15].payload["message"])
    assert isinstance(steering_message, UserMessage)
    assert steering_message.content == "steer"

    turn_zero_commit = records[11]
    assert turn_zero_commit.payload["transcript_record_ids"] == [
        "run-journal-1:turn:0:transcript:0",
        "run-journal-1:turn:0:transcript:1",
        "run-journal-1:turn:0:transcript:2",
    ]
    assert turn_zero_commit.payload["tool_terminal_record_ids"] == [
        "run-journal-1:turn:0:tool:c1:terminal"
    ]


@pytest.mark.asyncio
async def test_aborted_run_records_run_interrupted_without_messages(tmp_path) -> None:
    token = CancelToken()
    token.request_cancel("immediate")
    journal = JsonlSessionJournal(tmp_path)
    transaction = await JournalTurnTransaction.create(
        journal, "run-journal-2", {"purpose": "test"}
    )
    model = ScriptedModel([text_events("never")])
    context = AgentContext(messages=[])
    config = AgentLoopConfig(
        model=model, run_id="run-journal-2", transaction=transaction
    )
    result = await run_agent_loop(
        [UserMessage(content="go")], context, config, None, token
    )
    assert result.status is AgentRunStatus.ABORTED
    await journal.close()

    records = await _replay(tmp_path, "run-journal-2")
    assert records[-1].type is JournalRecordType.RUN_INTERRUPTED
    assert records[-1].payload == {
        "status": "aborted",
        "error": "run aborted before model admission",
        "finalization_diagnostic": None,
    }

    outcome = recover_run_outcome(records)
    assert outcome is not None
    assert outcome.status is AgentRunStatus.ABORTED
    assert outcome.error == "run aborted before model admission"
    assert outcome.messages == tuple(
        message for message in context.messages
    )


@pytest.mark.asyncio
async def test_committed_tool_transaction_is_queryable_after_reopen(tmp_path) -> None:
    journal = JsonlSessionJournal(tmp_path)
    transaction = await JournalTurnTransaction.create(
        journal, "run-journal-3", {"purpose": "test"}
    )
    model = ScriptedModel(
        [tool_events([tool_call_wire("c1", "echo", {"text": "x"})]), text_events("done")]
    )
    context = AgentContext(messages=[], tools=ToolRegistry().register(_echo).freeze())
    config = AgentLoopConfig(
        model=model, run_id="run-journal-3", transaction=transaction
    )
    result = await run_agent_loop([UserMessage(content="go")], context, config, None)
    assert result.status is AgentRunStatus.COMPLETED
    await journal.close()

    reopened = JsonlSessionJournal(tmp_path)
    await reopened.open("run-journal-3")
    terminal_id = "run-journal-3:turn:0:tool:c1:terminal"
    view = reopened.find_tool_transaction(JournalRecordRef("run-journal-3", terminal_id))
    committed = next(
        record
        for record in await reopened.replay()
        if record.type is JournalRecordType.STEP_COMMITTED
    )
    await reopened.close()

    assert view is not None
    assert view.action.id == "c1"
    assert view.action.name == "echo"
    assert view.result.status == "success"
    assert view.result.model_visible_output == "echo:x"
    assert view.committed_at.record_id == committed.record_id


@tool(name="accounted_echo")
def _accounted_echo(text: str) -> ToolResult:
    return ToolResult(
        output={"echo": text},
        usage=ModelUsage.from_mapping({"total_tokens": 13, "cost_usd": 0.0004}),
        added_tool_names=("late_skill_tool",),
    )


@pytest.mark.asyncio
async def test_tool_result_facts_round_trip_through_the_transcript_entry(
    tmp_path,
) -> None:
    journal = JsonlSessionJournal(tmp_path)
    transaction = await JournalTurnTransaction.create(
        journal, "run-journal-facts", {"purpose": "test"}
    )
    model = ScriptedModel(
        [
            tool_events([tool_call_wire("c1", "accounted_echo", {"text": "x"})]),
            text_events("done"),
        ]
    )
    context = AgentContext(
        messages=[], tools=ToolRegistry().register(_accounted_echo).freeze()
    )
    config = AgentLoopConfig(
        model=model, run_id="run-journal-facts", transaction=transaction
    )
    result = await run_agent_loop([UserMessage(content="go")], context, config, None)
    assert result.status is AgentRunStatus.COMPLETED
    await journal.close()

    reopened = JsonlSessionJournal(tmp_path)
    await reopened.open("run-journal-facts")
    records = await reopened.replay()
    view = reopened.find_tool_transaction(
        JournalRecordRef(
            "run-journal-facts", "run-journal-facts:turn:0:tool:c1:terminal"
        )
    )
    await reopened.close()

    tool_entry = next(
        record
        for record in records
        if record.type is JournalRecordType.TRANSCRIPT_MESSAGE
        and record.payload["message"]["role"] == "tool"
    )
    result_payload = tool_entry.payload["message"]["result"]
    assert result_payload["usage"]["total_tokens"] == 13
    assert result_payload["added_tool_names"] == ["late_skill_tool"]
    assert tool_entry.payload["message"]["added_tool_names"] == ["late_skill_tool"]
    restored_result = ToolResult.from_dict(result_payload)
    assert restored_result.usage is not None
    assert restored_result.usage.total_tokens == 13
    assert restored_result.added_tool_names == ("late_skill_tool",)

    assert view is not None
    assert view.result.usage is not None
    assert view.result.usage.total_tokens == 13
    assert view.result.added_tool_names == ("late_skill_tool",)


@pytest.mark.asyncio
async def test_seeded_recorder_continues_turns_and_journals_diffs_only(
    tmp_path,
) -> None:
    journal = JsonlSessionJournal(tmp_path)
    await journal.create("run-seed", {"purpose": "test"})
    seed = RecoveredRecorderState(
        next_turn=4,
        model_identity=("scripted", "scripted-model", "legacy"),
        thinking_level=ThinkingLevel.LOW,
        active_tool_names=("echo",),
        recorded_message_count=9,
    )
    recorder = JournalTurnTransaction(journal, recovered=seed)

    # An identical freeze writes nothing; a changed dimension journals only
    # its own diff record at the continued turn number.
    unchanged = await recorder.turn_frozen(
        4,
        TurnConfigSnapshot(
            provider="scripted",
            model="scripted-model",
            api="legacy",
            thinking_level=ThinkingLevel.LOW,
            tool_names=("echo",),
        ),
    )
    assert unchanged == ()
    changed = await recorder.turn_frozen(
        4,
        TurnConfigSnapshot(
            provider="scripted",
            model="scripted-model",
            api="legacy",
            thinking_level=ThinkingLevel.HIGH,
            tool_names=("echo",),
        ),
    )
    assert [position.record_id for position in changed] == [
        "run-seed:turn:4:config:thinking"
    ]
    assert recorder.recorded_message_count == 9

    # Repeating the same freeze after recovery writes nothing either.
    repeated = await recorder.turn_frozen(
        4,
        TurnConfigSnapshot(
            provider="scripted",
            model="scripted-model",
            api="legacy",
            thinking_level=ThinkingLevel.HIGH,
            tool_names=("echo",),
        ),
    )
    assert repeated == ()
    await journal.close()

    records = await _replay(tmp_path, "run-seed")
    assert [
        record.record_id
        for record in records
        if record.type is JournalRecordType.THINKING_CHANGE
    ] == ["run-seed:turn:4:config:thinking"]
    assert records[-1].payload == {"level": "high"}
    assert not any(
        record.type is JournalRecordType.MODEL_CHANGE for record in records
    )


@pytest.mark.asyncio
async def test_runtime_input_consumed_and_compaction_records(tmp_path) -> None:
    journal = JsonlSessionJournal(tmp_path)
    transaction = await JournalTurnTransaction.create(
        journal, "run-markers", {"purpose": "test"}
    )
    first = await transaction.runtime_input_consumed("event-1")
    again = await transaction.runtime_input_consumed("event-1")
    assert again == first
    compaction = await transaction.compaction(
        summary="summary so far",
        first_kept_transcript_id="run-markers:turn:2:transcript:0",
        tokens_before=1234,
        usage=ModelUsage.from_mapping({"total_tokens": 40}),
    )
    await journal.close()

    records = await _replay(tmp_path, "run-markers")
    consumed = [
        record
        for record in records
        if record.type is JournalRecordType.RUNTIME_INPUT_CONSUMED
    ]
    assert len(consumed) == 1
    assert consumed[0].payload == {"event_id": "event-1"}
    assert compaction.record_id == (
        "run-markers:compaction:run-markers:turn:2:transcript:0"
    )
    stored = records[-1]
    assert stored.type is JournalRecordType.COMPACTION
    assert stored.payload["summary"] == "summary so far"
    assert stored.payload["tokens_before"] == 1234
    assert stored.payload["usage"]["total_tokens"] == 40


@pytest.mark.asyncio
async def test_recover_run_outcome_rejects_old_payload_shapes(tmp_path) -> None:
    journal = JsonlSessionJournal(tmp_path)
    await journal.create("run-old-shape", {"purpose": "test"})
    await journal.append(
        JournalRecordType.RUN_COMPLETED,
        {"status": "completed", "error": None, "messages": []},
        record_id="run-old-shape:run:terminal",
    )
    records = await journal.replay()
    await journal.close()

    with pytest.raises(ValueError, match="not a recoverable loop journal"):
        recover_run_outcome(records)


@pytest.mark.asyncio
async def test_recorder_commits_budget_per_model_terminal(tmp_path) -> None:
    from qitos.core.budget import BudgetLedger

    journal = JsonlSessionJournal(tmp_path)
    await journal.create("run-budget", {"purpose": "test"})
    ledger = BudgetLedger(max_tokens=1_000)
    ledger.attach(journal, root_run_id="run-budget", records=await journal.replay())
    recorder = JournalTurnTransaction(journal, budget_ledger=ledger)

    model = ScriptedModel(
        [text_events("done", usage={"total_tokens": 7})], model="scripted-model"
    )
    context = AgentContext(messages=[])
    config = AgentLoopConfig(model=model, run_id="run-budget", transaction=recorder)
    result = await run_agent_loop([UserMessage(content="go")], context, config, None)
    assert result.status is AgentRunStatus.COMPLETED

    snapshot = ledger.snapshot()
    assert snapshot.total_tokens == 7
    budget_records = [
        record
        for record in await journal.replay()
        if record.type is JournalRecordType.BUDGET_COMMITTED
    ]
    assert len(budget_records) == 1
    payload = budget_records[0].payload
    # Same idempotency-key scheme the Child boundary uses, keyed by this run.
    assert payload["transaction_id"] == "run-budget:turn:0:model"
    assert payload["origin_run_id"] == "run-budget"
    assert payload["tokens"] == 7
    assert payload["usage_complete"] is True
    await journal.close()

    # A ledger re-attached from the replay restores the committed usage.
    reopened = JsonlSessionJournal(tmp_path)
    await reopened.open("run-budget")
    restored = BudgetLedger(max_tokens=1_000)
    restored.attach(
        reopened, root_run_id="run-budget", records=await reopened.replay()
    )
    assert restored.snapshot().total_tokens == 7
    assert restored.snapshot_after_origin("run-budget") is not None
    await reopened.close()


@pytest.mark.asyncio
async def test_parallel_terminal_append_fault_keeps_a_recoverable_crash_run() -> None:
    class FailFirstToolTerminal(InMemorySessionJournal):
        def __init__(self, store: InMemoryJournalStore) -> None:
            super().__init__(store)
            self.failed = False

        async def append(self, record_type, payload, *, record_id):
            if record_type is JournalRecordType.TOOL_TERMINAL and not self.failed:
                self.failed = True
                raise JournalError("injected first terminal append failure")
            return await super().append(record_type, payload, record_id=record_id)

    executed: list[str] = []

    @tool(name="first", concurrency_safe=True)
    async def _first() -> str:
        executed.append("first")
        return "first"

    @tool(name="second", concurrency_safe=True)
    async def _second() -> str:
        executed.append("second")
        return "second"

    store = InMemoryJournalStore()
    journal = FailFirstToolTerminal(store)
    transaction = await JournalTurnTransaction.create(
        journal, "run-terminal-fault", {"purpose": "test"}
    )
    model = ScriptedModel(
        [
            tool_events(
                [
                    tool_call_wire("c1", "first", {}),
                    tool_call_wire("c2", "second", {}),
                ]
            )
        ]
    )
    context = AgentContext(
        messages=[],
        tools=ToolRegistry().include_toolset([_first, _second]).freeze(),
    )
    config = AgentLoopConfig(
        model=model,
        run_id="run-terminal-fault",
        transaction=transaction,
        tool_execution="parallel",
    )

    with pytest.raises(JournalError, match="unterminated"):
        await run_agent_loop([UserMessage(content="go")], context, config, None)

    assert set(executed) == {"first", "second"}
    records = await journal.replay()
    assert not any(
        record.type
        in (JournalRecordType.RUN_COMPLETED, JournalRecordType.RUN_INTERRUPTED)
        for record in records
    )
    terminals = [
        record
        for record in records
        if record.type is JournalRecordType.TOOL_TERMINAL
    ]
    assert terminals == []

    recovered = recover_session(records)
    assert recovered.outcome is None
    assert [item.call.id for item in recovered.unterminated_calls] == ["c1", "c2"]
    before_close = tuple(executed)
    closed = await close_crashed_tool_calls(journal, recovered)
    assert closed
    assert tuple(executed) == before_close

    settled_records = await journal.replay()
    settled_terminals = [
        record
        for record in settled_records
        if record.type is JournalRecordType.TOOL_TERMINAL
    ]
    assert [record.payload["call_id"] for record in settled_terminals] == [
        "c1",
        "c2",
    ]
    settled = recover_session(settled_records)
    assert settled.unterminated_calls == ()
    assert settled.unstarted_calls == ()
    assert settled.outcome is None
    await journal.close()
