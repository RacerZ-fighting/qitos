"""Pure recovery over canonical journals: replay, crash windows, closure."""

from __future__ import annotations

import pytest

from qitos.core.agent_loop import (
    AgentLoopResult,
    AgentRunStatus,
    RunFinalizationDiagnostic,
    RunFinalizationDiagnosticCode,
    TurnConfigSnapshot,
)
from qitos.core.journal import (
    JournalCorruptionError,
    JournalRecord,
    JournalRecordType,
)
from qitos.core.message import (
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from qitos.core.model_request import ModelRequest
from qitos.core.model_response import ModelUsage
from qitos.core.runtime_input import RuntimeInput
from qitos.core.thinking import ThinkingLevel
from qitos.core.tool_result import ToolResult
from qitos.kit.journal import (
    InMemoryJournalStore,
    InMemorySessionJournal,
    JournalTurnTransaction,
    close_crashed_tool_calls,
    recover_run_outcome,
    recover_session,
)
from qitos.kit.journal.recovery import RecoveredSession
from qitos.kit.journal.turn_recorder import (
    decode_compaction,
    decode_input_accepted,
    decode_model_change,
    decode_model_completed,
    decode_run_terminal,
    decode_runtime_input_consumed,
    decode_step_committed,
    decode_thinking_change,
    decode_tool_started,
    decode_tool_terminal,
    decode_tools_change,
    decode_transcript_message,
    encode_model_completed,
    encode_step_committed,
    encode_transcript_message,
)

_SNAPSHOT = TurnConfigSnapshot(
    provider="prov",
    model="model-x",
    api="legacy",
    thinking_level=None,
    tool_names=("echo",),
)


def _request(run_id: str, turn: int) -> ModelRequest:
    return ModelRequest(
        run_id=run_id,
        transaction_id=f"{run_id}:turn:{turn}:tx",
        provider="prov",
        model="model-x",
        protocol="legacy",
        messages=({"role": "user", "content": "go"},),
    )


def _assistant(*calls: ToolCall, text: str = "reply") -> AssistantMessage:
    return AssistantMessage(
        text=text,
        tool_calls=tuple(calls),
        model_name="model-x",
        provider="prov",
    )


def _call(call_id: str, name: str = "echo") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments={"text": call_id})


class _Run:
    """Script one run through the real recorder up to a chosen crash point."""

    def __init__(
        self,
        journal: InMemorySessionJournal,
        recorder: JournalTurnTransaction,
        run_id: str,
    ) -> None:
        self.journal = journal
        self.recorder = recorder
        self.run_id = run_id
        self.turn_messages: list = []

    async def prompt(self, text: str = "go") -> UserMessage:
        message = UserMessage(content=text)
        await self.recorder.input_accepted((message,))
        self.turn_messages.append(message)
        return message

    async def model(self, turn: int, message: AssistantMessage) -> AssistantMessage:
        await self.recorder.turn_frozen(turn, _SNAPSHOT)
        await self.recorder.model_terminal(turn, _request(self.run_id, turn), message)
        self.turn_messages.append(message)
        return message

    async def tool(
        self, turn: int, call: ToolCall, result: ToolResult | None = None
    ) -> ToolResultMessage:
        await self.recorder.tool_started(turn, call)
        result = result or ToolResult(output=f"out-{call.id}", call_id=call.id)
        await self.recorder.tool_terminal(turn, call, result)
        # Return the message exactly as the journal durably recorded it.
        records = await self.journal.replay()
        entry = records[-2]
        assert entry.type is JournalRecordType.TRANSCRIPT_MESSAGE
        message = decode_transcript_message(entry.payload)
        assert isinstance(message, ToolResultMessage)
        self.turn_messages.append(message)
        return message

    async def commit(self, turn: int) -> None:
        await self.recorder.turn_committed(turn, tuple(self.turn_messages))
        self.turn_messages = []

    async def finish(self, status: AgentRunStatus = AgentRunStatus.COMPLETED) -> None:
        await self.recorder.run_terminal(
            AgentLoopResult(status=status, messages=(), error=None)
        )

    async def recover(self) -> RecoveredSession:
        return recover_session(await self.journal.replay())


async def _start_run(
    store: InMemoryJournalStore | None = None,
    run_id: str = "run",
    *,
    seed=None,
) -> _Run:
    journal = InMemorySessionJournal(store)
    await journal.create(run_id, {"purpose": "test"})
    recorder = JournalTurnTransaction(journal, recovered=seed)
    return _Run(journal, recorder, run_id)


@pytest.mark.asyncio
async def test_clean_run_recovers_full_state() -> None:
    run = await _start_run()
    prompt = await run.prompt()
    first = await run.model(0, _assistant(_call("c1"), text="first"))
    tool_message = await run.tool(0, _call("c1"))
    await run.commit(0)
    second = await run.model(1, _assistant(text="final"))
    await run.commit(1)
    await run.finish()

    recovered = await run.recover()

    assert recovered.run_id == "run"
    assert recovered.transcript == (prompt, first, tool_message, second)
    assert recovered.context_messages == recovered.transcript
    assert recovered.next_turn == 2
    assert recovered.model_identity == ("prov", "model-x", "legacy")
    assert recovered.thinking_level is ThinkingLevel.OFF
    assert recovered.active_tool_names == ("echo",)
    assert recovered.unterminated_calls == ()
    assert recovered.unstarted_calls == ()
    assert recovered.unconsumed_inputs == ()
    assert recovered.crash_turn is None
    assert recovered.outcome is not None
    assert recovered.outcome.status is AgentRunStatus.COMPLETED
    assert recovered.outcome.messages == recovered.transcript
    assert recovered.outcome.finalization_diagnostic is None
    assert recovered.recorder_state.next_turn == 2
    assert recovered.recorder_state.recorded_message_count == len(
        recovered.transcript
    )
    assert recovered.uncommitted_transcript_record_ids == ()


@pytest.mark.asyncio
async def test_run_finalization_diagnostic_round_trips_through_recovery() -> None:
    run = await _start_run()
    await run.prompt()
    await run.model(0, _assistant(text="done"))
    await run.commit(0)
    diagnostic = RunFinalizationDiagnostic(
        code=RunFinalizationDiagnosticCode.RESOURCE_QUIESCE_FAILED,
        message="one process did not settle",
    )
    await run.recorder.run_terminal(
        AgentLoopResult(
            status=AgentRunStatus.COMPLETED,
            messages=(),
            finalization_diagnostic=diagnostic,
        )
    )

    recovered = await run.recover()

    assert recovered.outcome is not None
    assert recovered.outcome.status is AgentRunStatus.COMPLETED
    assert recovered.outcome.finalization_diagnostic == diagnostic


@pytest.mark.asyncio
async def test_crash_after_model_completed_leaves_unstarted_calls() -> None:
    run = await _start_run()
    await run.prompt()
    await run.model(0, _assistant(_call("c1"), _call("c2"), text="first"))

    recovered = await run.recover()

    assert recovered.outcome is None
    assert recovered.next_turn == 1
    assert recovered.unterminated_calls == ()
    assert [crashed.call.id for crashed in recovered.unstarted_calls] == ["c1", "c2"]
    assert all(
        crashed.started_record_id is None for crashed in recovered.unstarted_calls
    )
    assert recovered.crash_turn == 0
    assert recovered.crash_turn_transcript_entries == 2  # prompt + assistant

    closed = await close_crashed_tool_calls(run.journal, recovered)
    assert len(closed) == 2
    # Closing again with the same recovered state is an idempotent no-op.
    again = await close_crashed_tool_calls(run.journal, recovered)
    assert len(again) == 2

    records = await run.journal.replay()
    terminals = [
        record
        for record in records
        if record.type is JournalRecordType.TOOL_TERMINAL
    ]
    assert len(terminals) == 2
    starts = [
        record
        for record in records
        if record.type is JournalRecordType.TOOL_STARTED
    ]
    assert [decode_tool_started(record.payload)[1].id for record in starts] == [
        "c1",
        "c2",
    ]
    tool_entries = [
        record.payload["message"]
        for record in records
        if record.type is JournalRecordType.TRANSCRIPT_MESSAGE
        and record.payload["message"]["role"] == "tool"
    ]
    assert len(tool_entries) == 2
    assert all(entry["result"]["status"] == "cancelled" for entry in tool_entries)
    assert all("never executed" in entry["result"]["error"] for entry in tool_entries)
    assert all(
        entry["result"]["metadata"]["started"] is False for entry in tool_entries
    )

    after = recover_session(records)
    assert after.unterminated_calls == ()
    assert after.unstarted_calls == ()
    assert after.crash_turn is None
    assert after.next_turn == 1
    # The cancelled results follow their calls in conversation order and are
    # committed Tool transactions now.
    assert [message.role for message in after.transcript] == [
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert isinstance(after.transcript[-1], ToolResultMessage)
    assert after.transcript[-1].result.metadata["cancel_source"] == "crash_recovery"
    for reference in closed:
        assert run.journal.find_tool_transaction(reference) is not None
    # A fresh recovery finds nothing left to close.
    assert await close_crashed_tool_calls(run.journal, after) == ()


@pytest.mark.asyncio
async def test_crash_after_tool_started_leaves_unterminated_call() -> None:
    run = await _start_run()
    await run.prompt()
    call = _call("c1")
    await run.model(0, _assistant(call, text="first"))
    await run.recorder.tool_started(0, call)

    recovered = await run.recover()

    assert recovered.unstarted_calls == ()
    assert len(recovered.unterminated_calls) == 1
    crashed = recovered.unterminated_calls[0]
    assert crashed.call.id == "c1"
    assert crashed.started_record_id == "run:turn:0:tool:c1:started"
    assert crashed.transcript_record_id is None

    await close_crashed_tool_calls(run.journal, recovered)
    after = await run.recover()
    assert after.unterminated_calls == ()
    tool_message = after.transcript[-1]
    assert isinstance(tool_message, ToolResultMessage)
    assert tool_message.result.status == "cancelled"
    assert "outcome is unknown" in (tool_message.result.error or "")
    assert tool_message.result.metadata["started"] is True


@pytest.mark.asyncio
async def test_crash_with_torn_tool_entry_links_the_durable_result() -> None:
    run = await _start_run()
    await run.prompt()
    call = _call("c1")
    await run.model(0, _assistant(call, text="first"))
    await run.recorder.tool_started(0, call)
    # The recorder's transcript append landed but the terminal did not: the
    # real result is already durable and must be linked, not replaced.
    torn_message = ToolResultMessage(
        tool_call_id="c1",
        tool_name="echo",
        result=ToolResult(output="real-result", call_id="c1"),
    )
    await run.journal.append(
        JournalRecordType.TRANSCRIPT_MESSAGE,
        encode_transcript_message(torn_message),
        record_id="run:turn:0:transcript:2",
    )

    recovered = await run.recover()

    assert len(recovered.unterminated_calls) == 1
    crashed = recovered.unterminated_calls[0]
    assert crashed.transcript_record_id == "run:turn:0:transcript:2"

    before = len(await run.journal.replay())
    await close_crashed_tool_calls(run.journal, recovered)
    records = await run.journal.replay()
    # Only the terminal and the closing commit are appended.
    assert len(records) == before + 2
    terminal = records[-2]
    assert terminal.type is JournalRecordType.TOOL_TERMINAL
    assert terminal.payload["message_record_id"] == "run:turn:0:transcript:2"

    after = recover_session(records)
    assert after.unterminated_calls == ()
    tool_message = after.transcript[-1]
    assert isinstance(tool_message, ToolResultMessage)
    assert tool_message.result.model_visible_output == "real-result"


@pytest.mark.asyncio
async def test_crash_after_commit_is_a_clean_turn_boundary() -> None:
    run = await _start_run()
    await run.prompt()
    await run.model(0, _assistant(_call("c1"), text="first"))
    await run.tool(0, _call("c1"))
    await run.commit(0)

    recovered = await run.recover()

    assert recovered.unterminated_calls == ()
    assert recovered.unstarted_calls == ()
    assert recovered.crash_turn is None
    assert recovered.next_turn == 1
    assert recovered.uncommitted_transcript_record_ids == ()
    assert await close_crashed_tool_calls(run.journal, recovered) == ()


@pytest.mark.asyncio
async def test_recovery_after_run_terminal_has_no_resume_state() -> None:
    run = await _start_run()
    await run.prompt()
    await run.model(0, _assistant(text="done"))
    await run.commit(0)
    await run.finish()

    recovered = await run.recover()

    assert recovered.outcome is not None
    assert recovered.outcome.status is AgentRunStatus.COMPLETED
    assert recovered.unterminated_calls == ()
    assert recovered.unstarted_calls == ()
    assert await close_crashed_tool_calls(run.journal, recovered) == ()


@pytest.mark.asyncio
async def test_compaction_projects_summary_plus_kept_entries() -> None:
    run = await _start_run()
    await run.prompt()
    await run.model(0, _assistant(_call("c1"), text="first"))
    await run.tool(0, _call("c1"))
    await run.commit(0)
    await run.model(1, _assistant(text="final"))
    await run.commit(1)
    # Cut before the turn-1 assistant: the summary replaces the earlier turns.
    await run.recorder.compaction(
        summary="what happened so far",
        first_kept_transcript_id="run:turn:1:transcript:0",
        tokens_before=500,
    )

    recovered = await run.recover()

    assert [message.role for message in recovered.context_messages] == [
        "user",
        "assistant",
    ]
    summary_message = recovered.context_messages[0]
    assert isinstance(summary_message, UserMessage)
    assert summary_message.content == "what happened so far"
    assert recovered.context_messages[1] == recovered.transcript[-1]
    # The full transcript remains available; compaction rewrites nothing.
    assert [message.role for message in recovered.transcript] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


@pytest.mark.asyncio
async def test_compaction_rejects_bad_cuts() -> None:
    run = await _start_run()
    await run.prompt()
    await run.model(0, _assistant(_call("c1"), text="first"))
    await run.tool(0, _call("c1"))
    await run.commit(0)

    await run.recorder.compaction(
        summary="cut at a tool message",
        first_kept_transcript_id="run:turn:0:transcript:2",
        tokens_before=10,
    )
    with pytest.raises(JournalCorruptionError, match="must not land on a tool"):
        await run.recover()

    storeless = await _start_run(run_id="dangling")
    await storeless.prompt()
    await storeless.model(0, _assistant(text="one"))
    await storeless.commit(0)
    await storeless.recorder.compaction(
        summary="dangling reference",
        first_kept_transcript_id="dangling:turn:9:transcript:9",
        tokens_before=10,
    )
    with pytest.raises(JournalCorruptionError, match="unresolvable"):
        await storeless.recover()


@pytest.mark.asyncio
async def test_unconsumed_inputs_fold_own_records_only() -> None:
    run = await _start_run()
    await run.prompt()
    await run.model(0, _assistant(text="one"))
    await run.commit(0)
    first = RuntimeInput(
        event_id="evt-1",
        kind="agent.child.completed",
        correlation_id="child-1",
        source="qitos.agent",
        payload={"content": "done"},
    )
    second = RuntimeInput(
        event_id="evt-2",
        kind="process.completed",
        correlation_id="proc-1",
        source="qitos.process",
        payload={"content": "exited"},
    )
    await run.journal.append(
        JournalRecordType.RUNTIME_INPUT_POSTED, first.to_dict(), record_id="run:runtime:evt-1"
    )
    await run.journal.append(
        JournalRecordType.RUNTIME_INPUT_POSTED, second.to_dict(), record_id="run:runtime:evt-2"
    )
    await run.recorder.runtime_input_consumed("evt-1")

    recovered = await run.recover()

    assert recovered.unconsumed_inputs == (second,)


@pytest.mark.asyncio
async def test_inherited_runtime_inputs_are_never_redelivered() -> None:
    store = InMemoryJournalStore()
    parent = await _start_run(store, "parent")
    await parent.prompt()
    await parent.model(0, _assistant(text="one"))
    await parent.commit(0)
    inherited_input = RuntimeInput(
        event_id="evt-parent",
        kind="agent.child.completed",
        correlation_id="child-1",
        source="qitos.agent",
        payload={"content": "done"},
    )
    await parent.journal.append(
        JournalRecordType.RUNTIME_INPUT_POSTED,
        inherited_input.to_dict(),
        record_id="parent:runtime:evt-parent",
    )
    boundary_position = None
    for record in await parent.journal.replay():
        if record.type is JournalRecordType.STEP_COMMITTED:
            boundary_position = record.position
    assert boundary_position is not None
    child_journal = await parent.journal.fork(boundary_position, "child")
    child = _Run(child_journal, JournalTurnTransaction(child_journal), "child")
    await child.prompt("continue")
    await child.model(0, _assistant(text="child reply"))
    await child.commit(0)

    recovered = recover_session(await child_journal.replay())

    # The inherited posted input is lineage history, not an unconsumed fact
    # of the child run (D7).
    assert recovered.unconsumed_inputs == ()
    assert recovered.unterminated_calls == ()


@pytest.mark.asyncio
async def test_contradiction_matrix_fails_closed() -> None:
    run = await _start_run()
    await run.prompt()
    call = _call("c1")
    await run.model(0, _assistant(call, text="first"))
    await run.tool(0, call)
    await run.commit(0)
    good_records = await run.journal.replay()

    def _drop(predicate) -> tuple:
        return tuple(
            record for record in good_records if not predicate(record)
        )

    def _mutated(record: JournalRecord, **payload_changes) -> JournalRecord:
        payload = dict(record.payload)
        payload.update(payload_changes)
        return JournalRecord.create(
            seq=record.seq,
            record_id=record.record_id,
            type=record.type,
            run_id=record.run_id,
            payload=payload,
        )

    # Dangling model.completed transcript reference.
    model_record = next(
        record
        for record in good_records
        if record.type is JournalRecordType.MODEL_COMPLETED
    )
    dangling_model = _mutated(model_record, message_record_id="run:turn:0:transcript:99")
    with pytest.raises(JournalCorruptionError, match="unknown transcript entry"):
        recover_session(
            tuple(dangling_model if r == model_record else r for r in good_records)
        )

    # A tool transcript entry with no started/terminal evidence, covered by
    # the commit.
    without_evidence = _drop(
        lambda record: record.type
        in {JournalRecordType.TOOL_STARTED, JournalRecordType.TOOL_TERMINAL}
    )
    with pytest.raises(JournalCorruptionError, match="torn tool transcript entry"):
        recover_session(without_evidence)

    # step.committed referencing unknown records.
    commit_record = next(
        record
        for record in good_records
        if record.type is JournalRecordType.STEP_COMMITTED
    )
    bad_commit = _mutated(
        commit_record, transcript_record_ids=["run:turn:0:transcript:42"]
    )
    with pytest.raises(JournalCorruptionError, match="unknown transcript entry"):
        recover_session(
            tuple(bad_commit if r == commit_record else r for r in good_records)
        )

    # Two run terminals, then a record after the terminal.
    await run.finish()
    terminal_records = await run.journal.replay()
    second_terminal = JournalRecord.create(
        seq=len(terminal_records) + 1,
        record_id="run:run:terminal:2",
        type=JournalRecordType.RUN_COMPLETED,
        run_id="run",
        payload={"status": "completed", "error": None},
    )
    with pytest.raises(JournalCorruptionError, match="record after its run terminal"):
        recover_session((*terminal_records, second_terminal))

    # An unterminated call covered by a later commit.
    open_run = await _start_run(run_id="open-call")
    await open_run.prompt()
    open_call = _call("c9")
    await open_run.model(0, _assistant(open_call, text="first"))
    await open_run.recorder.tool_started(0, open_call)
    await open_run.journal.append(
        JournalRecordType.STEP_COMMITTED,
        encode_step_committed(0, ["open-call:turn:0:transcript:0"], []),
        record_id="open-call:turn:0:committed",
    )
    with pytest.raises(JournalCorruptionError, match="unterminated tool call"):
        recover_session(await open_run.journal.replay())

    # Turn regression (appended directly: the recorder itself refuses it).
    regressed = await _start_run(run_id="regressed")
    await regressed.prompt()
    await regressed.model(1, _assistant(text="one"))
    await regressed.journal.append(
        JournalRecordType.MODEL_COMPLETED,
        encode_model_completed(
            0, _request("regressed", 0), "regressed:turn:1:transcript:1"
        ),
        record_id="regressed:turn:0:model",
    )
    with pytest.raises(JournalCorruptionError, match="must not regress"):
        recover_session(await regressed.journal.replay())

    # Duplicate tool.started for one call id.
    duplicate = await _start_run(run_id="duplicate-started")
    await duplicate.prompt()
    dup_call = _call("c1")
    await duplicate.model(0, _assistant(dup_call, text="first"))
    await duplicate.recorder.tool_started(0, dup_call)
    await duplicate.journal.append(
        JournalRecordType.TOOL_STARTED,
        {"turn": 0, "call": dup_call.to_dict()},
        record_id="duplicate-started:turn:0:tool:c1:started:again",
    )
    with pytest.raises(JournalCorruptionError, match="duplicate tool.started"):
        recover_session(await duplicate.journal.replay())

    # runtime_input.consumed referencing an unknown event.
    consumed = await _start_run(run_id="consumed-unknown")
    await consumed.prompt()
    await consumed.recorder.runtime_input_consumed("evt-missing")
    with pytest.raises(JournalCorruptionError, match="unknown input"):
        recover_session(await consumed.journal.replay())


@pytest.mark.asyncio
async def test_old_payload_shapes_fail_closed() -> None:
    async def _journal_with(record_type, payload, record_id="run:legacy:1"):
        journal = InMemorySessionJournal(InMemoryJournalStore())
        await journal.create("run", {})
        await journal.append(record_type, payload, record_id=record_id)
        return await journal.replay()

    engine_model = await _journal_with(
        JournalRecordType.MODEL_COMPLETED,
        {"step_id": 0, "action_index": 0, "model_response": {"usage": {}}},
    )
    with pytest.raises(JournalCorruptionError):
        recover_session(engine_model)
    with pytest.raises(ValueError, match="not a recoverable loop journal"):
        recover_run_outcome(engine_model)

    # Terminals and commits with retired shapes are rejected by the shared
    # transaction index as soon as they are appended.
    with pytest.raises(JournalCorruptionError):
        await _journal_with(
            JournalRecordType.TOOL_TERMINAL,
            {
                "turn": 0,
                "call_id": "c1",
                "call": {"id": "c1", "name": "echo", "arguments": {}},
                "result": {
                    "status": "success",
                    "output": None,
                    "error": None,
                    "metadata": {},
                },
            },
        )
    with pytest.raises(JournalCorruptionError):
        await _journal_with(
            JournalRecordType.STEP_COMMITTED,
            {"turn": 0, "messages": [], "terminal_record_ids": []},
        )

    legacy_input = await _journal_with(
        JournalRecordType.INPUT_ACCEPTED,
        {"task": "inspect target"},
    )
    with pytest.raises(JournalCorruptionError):
        recover_session(legacy_input)

    legacy_terminal = await _journal_with(
        JournalRecordType.RUN_COMPLETED,
        {"stop_reason": "completed"},
    )
    with pytest.raises(JournalCorruptionError):
        recover_session(legacy_terminal)

    # The deleted state.snapshot type fails at envelope decode.
    snapshot_dict = JournalRecord.create(
        seq=2,
        record_id="run:snapshot",
        type=JournalRecordType.STEP_COMMITTED,
        run_id="run",
        payload={"turn": 0, "transcript_record_ids": [], "tool_terminal_record_ids": []},
    ).to_dict()
    snapshot_dict["type"] = "state.snapshot"
    with pytest.raises(JournalCorruptionError, match="record type is unsupported"):
        JournalRecord.from_dict(snapshot_dict)


@pytest.mark.asyncio
async def test_fork_lineage_recovers_through_inherited_prefix() -> None:
    store = InMemoryJournalStore()
    parent = await _start_run(store, "parent")
    await parent.prompt()
    await parent.model(0, _assistant(_call("c1"), text="first"))
    await parent.tool(0, _call("c1"))
    await parent.commit(0)
    boundary = (
        await parent.journal.replay()
    )[-1].position
    child_journal = await parent.journal.fork(boundary, "child")
    child = _Run(
        child_journal,
        JournalTurnTransaction(child_journal),
        "child",
    )
    await child.prompt("continue")
    await child.model(0, _assistant(text="child reply"))
    await child.commit(0)

    recovered = recover_session(await child_journal.replay())

    assert [message.role for message in recovered.transcript] == [
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
    ]
    # Own-run facts only: the inherited prompt is not new run output.
    assert recovered.outcome is None
    assert recovered.next_turn == 1
    assert recovered.unterminated_calls == ()
    assert recovered.unconsumed_inputs == ()
    # Configuration lineage survives the fork: the child journaled its own
    # freeze on its first turn.
    assert recovered.model_identity == ("prov", "model-x", "legacy")
    assert recovered.active_tool_names == ("echo",)


@pytest.mark.asyncio
async def test_config_lineage_tracks_diffs_and_tool_activation() -> None:
    run = await _start_run()
    await run.prompt()
    await run.model(0, _assistant(_call("c1"), text="first"))
    activated = ToolResult(
        output="ok",
        call_id="c1",
        added_tool_names=("late_tool",),
    )
    await run.tool(0, _call("c1"), activated)
    await run.commit(0)
    await run.recorder.turn_frozen(
        1,
        TurnConfigSnapshot(
            provider="prov",
            model="model-y",
            api="legacy",
            thinking_level=ThinkingLevel.HIGH,
            tool_names=("echo", "late_tool"),
        ),
    )
    await run.recorder.model_terminal(
        1, _request("run", 1), _assistant(text="second")
    )
    await run.commit(1)

    recovered = await run.recover()

    assert recovered.model_identity == ("prov", "model-y", "legacy")
    assert recovered.thinking_level is ThinkingLevel.HIGH
    assert recovered.active_tool_names == ("echo", "late_tool")

    # Without any tools.change the lineage has no tool expectation; without
    # any thinking.change the level is unknown, not "off".
    bare = await _start_run(run_id="bare")
    await bare.prompt()
    await bare.recorder.model_terminal(0, _request("bare", 0), _assistant(text="x"))
    bare_recovered = await bare.recover()
    assert bare_recovered.thinking_level is None
    assert bare_recovered.model_identity is None
    assert bare_recovered.active_tool_names is None


@pytest.mark.asyncio
async def test_added_tool_names_union_only_after_the_last_tools_change() -> None:
    run = await _start_run()
    await run.prompt()
    await run.model(0, _assistant(_call("c1"), text="first"))
    await run.tool(
        0,
        _call("c1"),
        ToolResult(output="ok", call_id="c1", added_tool_names=("late_tool",)),
    )
    await run.commit(0)
    # The turn-1 freeze already includes the activated name; a later result
    # activating yet another name extends the expectation again.
    await run.recorder.turn_frozen(
        1,
        TurnConfigSnapshot(
            provider="prov",
            model="model-x",
            api="legacy",
            thinking_level=None,
            tool_names=("echo", "late_tool"),
        ),
    )
    await run.recorder.model_terminal(
        1, _request("run", 1), _assistant(_call("c2"), text="second")
    )
    await run.tool(
        1,
        _call("c2"),
        ToolResult(output="ok", call_id="c2", added_tool_names=("third_tool",)),
    )
    await run.commit(1)

    recovered = await run.recover()

    assert recovered.active_tool_names == ("echo", "late_tool", "third_tool")


@pytest.mark.asyncio
async def test_seeded_recorder_continues_a_recovered_run() -> None:
    run = await _start_run()
    await run.prompt()
    await run.model(0, _assistant(_call("c1"), text="first"))
    await run.tool(0, _call("c1"))
    await run.commit(0)

    recovered = await run.recover()
    resumed = JournalTurnTransaction(run.journal, recovered=recovered.recorder_state)
    follow_up = UserMessage(content="continue")
    second = _assistant(text="second")
    await resumed.turn_frozen(1, _SNAPSHOT)
    await resumed.model_terminal(1, _request("run", 1), second)
    await resumed.turn_committed(1, (follow_up, second))

    records = await run.journal.replay()
    # No configuration records are rewritten for the resumed turn.
    assert [
        record.type
        for record in records
        if record.type
        in {
            JournalRecordType.MODEL_CHANGE,
            JournalRecordType.THINKING_CHANGE,
            JournalRecordType.TOOLS_CHANGE,
        }
    ] == [
        JournalRecordType.MODEL_CHANGE,
        JournalRecordType.THINKING_CHANGE,
        JournalRecordType.TOOLS_CHANGE,
    ]
    new_transcript = [
        record.record_id
        for record in records
        if record.type is JournalRecordType.TRANSCRIPT_MESSAGE
    ]
    assert "run:turn:1:transcript:0" in new_transcript

    again = recover_session(records)
    assert again.next_turn == 2
    assert [message.role for message in again.transcript] == [
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
    ]


@pytest.mark.asyncio
async def test_recover_run_outcome_returns_own_messages_only() -> None:
    store = InMemoryJournalStore()
    parent = await _start_run(store, "parent")
    await parent.prompt()
    await parent.model(0, _assistant(text="parent reply"))
    await parent.commit(0)
    boundary = (await parent.journal.replay())[-1].position
    child_journal = await parent.journal.fork(boundary, "child")
    child = _Run(child_journal, JournalTurnTransaction(child_journal), "child")
    await child.prompt("child task")
    await child.model(0, _assistant(text="child reply"))
    await child.commit(0)
    await child.finish()

    outcome = recover_run_outcome(await child_journal.replay())

    assert outcome is not None
    assert outcome.status is AgentRunStatus.COMPLETED
    assert [message.role for message in outcome.messages] == ["user", "assistant"]
    assert all(
        isinstance(message, (UserMessage, AssistantMessage))
        for message in outcome.messages
    )
    assert outcome.messages[0].content == "child task"  # type: ignore[union-attr]

    pending = await _start_run(store, "pending")
    await pending.prompt()
    assert recover_run_outcome(await pending.journal.replay()) is None


@pytest.mark.asyncio
async def test_codec_fail_closed_matrix() -> None:
    call_payload = {"id": "c1", "name": "echo", "arguments": {}}
    cases = [
        (decode_transcript_message, {}),
        (decode_transcript_message, {"message": {"role": "alien"}}),
        (decode_model_completed, {"turn": 0, "request": {}}),
        (decode_model_completed, {"turn": -1, "request": {}, "message_record_id": "x"}),
        (decode_tool_started, {"turn": 0}),
        (decode_tool_started, {"turn": "0", "call": call_payload}),
        (
            decode_tool_terminal,
            {"turn": 0, "call_id": "c2", "call": call_payload, "message_record_id": "m"},
        ),
        (
            decode_tool_terminal,
            {"turn": 0, "call_id": "c1", "call": call_payload},
        ),
        (decode_step_committed, {"turn": 0, "transcript_record_ids": []}),
        (
            decode_step_committed,
            {
                "turn": 0,
                "transcript_record_ids": [""],
                "tool_terminal_record_ids": [],
            },
        ),
        (decode_input_accepted, {"transcript_record_ids": "not-a-list"}),
        (decode_model_change, {"provider": "p", "model": "m"}),
        (decode_thinking_change, {"level": "loud"}),
        (decode_tools_change, {"active_tool_names": ["a", "a"]}),
        (decode_compaction, {"summary": "s", "first_kept_transcript_id": "x"}),
        (
            decode_compaction,
            {
                "summary": "s",
                "first_kept_transcript_id": "x",
                "tokens_before": -1,
            },
        ),
        (decode_runtime_input_consumed, {"event_id": ""}),
        (decode_run_terminal, {"status": "completed"}),
    ]
    for decoder, payload in cases:
        with pytest.raises((TypeError, ValueError)):
            decoder(payload)
    with pytest.raises(ValueError):
        decode_run_terminal(
            JournalRecordType.RUN_COMPLETED, {"status": "aborted", "error": None}
        )
    with pytest.raises(ValueError):
        decode_run_terminal(
            JournalRecordType.RUN_INTERRUPTED, {"status": "completed", "error": None}
        )
    with pytest.raises(ValueError, match="diagnostic code"):
        decode_run_terminal(
            JournalRecordType.RUN_COMPLETED,
            {
                "status": "completed",
                "error": None,
                "finalization_diagnostic": {
                    "code": "UNKNOWN_FINALIZER_FAILURE",
                    "message": "failed",
                },
            },
        )
    with pytest.raises(ValueError, match="maximum length"):
        decode_run_terminal(
            JournalRecordType.RUN_COMPLETED,
            {
                "status": "completed",
                "error": None,
                "finalization_diagnostic": {
                    "code": "RESOURCE_QUIESCE_FAILED",
                    "message": "x" * 513,
                },
            },
        )


@pytest.mark.asyncio
async def test_unstarted_call_in_committed_turn_is_corruption() -> None:
    run = await _start_run()
    await run.prompt()
    await run.model(0, _assistant(_call("c1"), text="first"))
    # The commit pretends the turn finished without admitting the call.
    await run.recorder.turn_committed(0, tuple(run.turn_messages))

    with pytest.raises(JournalCorruptionError, match="neither admitted nor closed"):
        await run.recover()
