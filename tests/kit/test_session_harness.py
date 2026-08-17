"""SessionHarness: start, resume, fork, runtime inputs, compaction, close."""

from __future__ import annotations

import asyncio

import pytest

from qitos.core.agent_loop import AgentRunStatus, TurnConfigSnapshot
from qitos.core.budget import BudgetLedger
from qitos.core.journal import (
    JournalCorruptionError,
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
from qitos.core.run import RunStatus
from qitos.core.runtime_input import RuntimeInput
from qitos.core.tool import tool
from qitos.core.tool_registry import ToolRegistry
from qitos.core.tool_result import ToolResult
from qitos.kit.journal import (
    InMemoryJournalStore,
    InMemorySessionJournal,
    JournalTurnTransaction,
    JsonlRunCatalog,
    JsonlSessionJournal,
    recover_session,
)
from qitos.kit.journal.turn_recorder import (
    decode_compaction,
    encode_model_completed,
    encode_runtime_input_consumed,
)
from qitos.kit.session import (
    CompactRejected,
    CompactionSettings,
    ResumeRejected,
    SessionHarness,
    SessionRun,
)

from tests.core.agent_fakes import (
    ScriptedModel,
    failed_events,
    make_hanging_model,
    text_events,
    tool_call_wire,
    tool_events,
)


@tool(name="echo")
def _echo(text: str) -> str:
    return f"echo:{text}"


def _registry() -> ToolRegistry:
    return ToolRegistry().register(_echo)


def _snapshot(tool_names: tuple[str, ...] = ("echo",)) -> TurnConfigSnapshot:
    return TurnConfigSnapshot(
        provider="scripted",
        model="scripted-model",
        api="legacy",
        thinking_level=None,
        tool_names=tool_names,
    )


def _request(run_id: str, turn: int) -> ModelRequest:
    return ModelRequest(
        run_id=run_id,
        transaction_id=f"{run_id}:turn:{turn}:tx",
        provider="scripted",
        model="scripted-model",
        protocol="legacy",
        messages=({"role": "user", "content": "go"},),
    )


def _usage(total: int) -> ModelUsage:
    return ModelUsage(
        input_tokens=total - 100,
        output_tokens=100,
        total_tokens=total,
    )


class _Script:
    """Script one run through the real recorder up to a chosen crash point."""

    def __init__(self, journal: InMemorySessionJournal, run_id: str) -> None:
        self.journal = journal
        self.recorder = JournalTurnTransaction(journal)
        self.run_id = run_id

    async def prompt(self, text: str = "go") -> UserMessage:
        message = UserMessage(content=text)
        await self.recorder.input_accepted((message,))
        return message

    async def model_turn(
        self, turn: int, assistant: AssistantMessage
    ) -> AssistantMessage:
        await self.recorder.turn_frozen(turn, _snapshot())
        await self.recorder.model_terminal(
            turn, _request(self.run_id, turn), assistant
        )
        return assistant

    async def tool_results(
        self, turn: int, assistant: AssistantMessage, *outputs: str
    ) -> list[ToolResultMessage]:
        messages: list[ToolResultMessage] = []
        for call, output in zip(assistant.tool_calls, outputs):
            await self.recorder.tool_started(turn, call)
            result = ToolResult(status="success", output=output, call_id=call.id)
            await self.recorder.tool_terminal(turn, call, result)
            messages.append(
                ToolResultMessage(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    result=result,
                )
            )
        return messages

    async def commit(self, turn: int, messages: list) -> None:
        await self.recorder.turn_committed(turn, tuple(messages))


async def _scripted_journal(
    store: InMemoryJournalStore, run_id: str
) -> tuple[InMemorySessionJournal, _Script]:
    journal = InMemorySessionJournal(store)
    await journal.create(run_id, {"purpose": "script"})
    return journal, _Script(journal, run_id)


def _runtime_input(event_id: str, content: str) -> RuntimeInput:
    return RuntimeInput(
        event_id=event_id,
        kind="agent.child.completed",
        correlation_id="child-1",
        source="qitos.agent",
        payload={"content": content},
    )


async def _replay(store: InMemoryJournalStore, run_id: str):
    journal = InMemorySessionJournal(store)
    await journal.open(run_id)
    records = await journal.replay()
    await journal.close()
    return records


# ── start and full-run integrity ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_full_run_journal_roundtrip_and_catalog(tmp_path) -> None:
    harness = SessionHarness(tmp_path / "journals")
    model = ScriptedModel(
        [
            tool_events([tool_call_wire("c1", "echo", {"text": "hi"})]),
            text_events("final answer", usage={"total_tokens": 120}),
        ]
    )
    session_run = await harness.start(
        model=model, tool_registry=_registry(), system_prompt="sys"
    )
    result = await session_run.prompt("hello")
    assert result.status is AgentRunStatus.COMPLETED
    run_id = session_run.run_id
    await session_run.close()

    reader = JsonlSessionJournal(tmp_path / "journals")
    await reader.open(run_id)
    records = await reader.replay()
    await reader.close()
    recovered = recover_session(records)
    assert recovered.model_identity == ("scripted", "scripted-model", "legacy")
    assert recovered.outcome is not None
    assert recovered.outcome.status is AgentRunStatus.COMPLETED
    roles = [message.role for message in recovered.transcript]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert isinstance(recovered.transcript[0], UserMessage)
    assert recovered.transcript[0].content == "hello"
    assert recovered.next_turn == 2

    catalog = JsonlRunCatalog(tmp_path / "journals")
    handles = await catalog.list_runs()
    assert [handle.run_id for handle in handles] == [run_id]
    handle = handles[0]
    assert handle.status is RunStatus.COMPLETED
    assert handle.continuation_position is not None


@pytest.mark.asyncio
async def test_prompt_after_settle_advances_along_an_explicit_fork() -> None:
    store = InMemoryJournalStore()
    harness = SessionHarness(store)
    model = ScriptedModel([text_events("one"), text_events("two")])
    session_run = await harness.start(model=model)
    first_run_id = session_run.run_id
    first = await session_run.prompt("first")
    assert first.status is AgentRunStatus.COMPLETED

    second = await session_run.prompt("second")
    assert second.status is AgentRunStatus.COMPLETED
    assert session_run.run_id != first_run_id
    await session_run.close()

    records = await _replay(store, session_run.run_id)
    types = [record.type for record in records]
    assert JournalRecordType.RUN_FORKED in types
    assert JournalRecordType.INHERITED in types
    recovered = recover_session(records)
    # The new journal is self-contained: the inherited prefix plus this
    # leg's own prompt and answer, with turn numbering restarted.
    assert [message.role for message in recovered.transcript] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert recovered.next_turn == 1
    own_records = [
        record
        for record in records
        if record.type is not JournalRecordType.INHERITED
        and record.run_id == session_run.run_id
    ]
    assert any(":turn:0:model" in record.record_id for record in own_records)


# ── resume ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_resume_clean_turn_boundary_continues_turn_numbering() -> None:
    store = InMemoryJournalStore()
    journal, script = await _scripted_journal(store, "run-clean")
    user = await script.prompt("start")
    assistant = AssistantMessage(
        text="working",
        tool_calls=(ToolCall(id="c1", name="echo", arguments={"text": "x"}),),
    )
    await script.model_turn(0, assistant)
    results = await script.tool_results(0, assistant, "echo:x")
    await script.commit(0, [user, assistant, *results])
    await journal.close()

    model = ScriptedModel([text_events("done")])
    harness = SessionHarness(store)
    resumed = await harness.resume("run-clean", model=model, tool_registry=_registry())
    assert isinstance(resumed, SessionRun)
    result = await resumed.continue_run()
    assert result.status is AgentRunStatus.COMPLETED
    await resumed.close()

    records = await _replay(store, "run-clean")
    recovered = recover_session(records)
    # No transcript entry was duplicated by the resume; the new turn
    # continues numbering from the recovered boundary.
    assert len(recovered.transcript) == 4
    assert len({id(message) for message in recovered.transcript}) == 4
    assert any(":turn:1:model" in record.record_id for record in records)
    assert recovered.next_turn == 2
    # The model saw exactly the recovered context, nothing re-delivered.
    assert len(model.requests[0].messages) == 3


@pytest.mark.asyncio
async def test_resume_crash_after_model_terminal_closes_unstarted_calls() -> None:
    store = InMemoryJournalStore()
    journal, script = await _scripted_journal(store, "run-crash-unstarted")
    user = await script.prompt("start")
    first = AssistantMessage(
        text="working",
        tool_calls=(ToolCall(id="c1", name="echo", arguments={"text": "x"}),),
    )
    await script.model_turn(0, first)
    first_results = await script.tool_results(0, first, "echo:x")
    await script.commit(0, [user, first, *first_results])
    # Crash: the next model terminal is durable, but Tool admission never ran.
    second = AssistantMessage(
        text="next",
        tool_calls=(
            ToolCall(id="c2", name="echo", arguments={"text": "a"}),
            ToolCall(id="c3", name="echo", arguments={"text": "b"}),
        ),
    )
    await script.model_turn(1, second)
    await journal.close()

    harness = SessionHarness(store)
    model = ScriptedModel([text_events("recovered")])
    resumed = await harness.resume(
        "run-crash-unstarted", model=model, tool_registry=_registry()
    )
    assert isinstance(resumed, SessionRun)
    result = await resumed.continue_run()
    assert result.status is AgentRunStatus.COMPLETED
    await resumed.close()

    records = await _replay(store, "run-crash-unstarted")
    recovered = recover_session(records)
    assert not recovered.unterminated_calls
    assert not recovered.unstarted_calls
    cancelled = [
        message
        for message in recovered.transcript
        if isinstance(message, ToolResultMessage)
        and message.result.metadata.get("cancel_source") == "crash_recovery"
    ]
    assert {message.tool_call_id for message in cancelled} == {"c2", "c3"}
    assert all(message.result.status == "cancelled" for message in cancelled)
    # The closing commit folded the crash turn, so numbering continues at 2.
    assert any(":turn:2:model" in record.record_id for record in records)
    assert recovered.next_turn == 3


@pytest.mark.asyncio
async def test_resume_crash_after_tool_started_closes_unterminated_calls() -> None:
    store = InMemoryJournalStore()
    journal, script = await _scripted_journal(store, "run-crash-started")
    user = await script.prompt("start")
    assistant = AssistantMessage(
        text="working",
        tool_calls=(
            ToolCall(id="c1", name="echo", arguments={"text": "a"}),
            ToolCall(id="c2", name="echo", arguments={"text": "b"}),
        ),
    )
    await script.model_turn(0, assistant)
    # Crash: c1 was admitted (durable permit) but never terminated; c2 never ran.
    await script.recorder.tool_started(0, assistant.tool_calls[0])
    await journal.close()

    harness = SessionHarness(store)
    model = ScriptedModel([text_events("recovered")])
    resumed = await harness.resume(
        "run-crash-started", model=model, tool_registry=_registry()
    )
    assert isinstance(resumed, SessionRun)
    result = await resumed.continue_run()
    assert result.status is AgentRunStatus.COMPLETED
    await resumed.close()

    records = await _replay(store, "run-crash-started")
    recovered = recover_session(records)
    assert not recovered.unterminated_calls
    assert not recovered.unstarted_calls
    cancelled = {
        message.tool_call_id: message
        for message in recovered.transcript
        if isinstance(message, ToolResultMessage)
        and message.result.metadata.get("cancel_source") == "crash_recovery"
    }
    assert set(cancelled) == {"c1", "c2"}
    assert cancelled["c1"].result.metadata["started"] is True
    assert cancelled["c2"].result.metadata["started"] is False


@pytest.mark.asyncio
async def test_resume_rejections_are_typed() -> None:
    store = InMemoryJournalStore()
    harness = SessionHarness(store)
    missing = await harness.resume("run-absent", model=ScriptedModel([]))
    assert isinstance(missing, ResumeRejected)
    assert missing.reason == "not_found"

    # Terminal journals reject and point at fork for explicit continuation.
    journal, script = await _scripted_journal(store, "run-terminal")
    user = await script.prompt("start")
    assistant = AssistantMessage(text="done")
    await script.model_turn(0, assistant)
    await script.commit(0, [user, assistant])
    from qitos.core.agent_loop import AgentLoopResult

    await script.recorder.run_terminal(
        AgentLoopResult(status=AgentRunStatus.COMPLETED, messages=(assistant,))
    )
    await journal.close()
    terminal = await harness.resume("run-terminal", model=ScriptedModel([]))
    assert isinstance(terminal, ResumeRejected)
    assert terminal.reason == "terminal"

    # A model identity different from the journaled lineage rejects.
    journal, script = await _scripted_journal(store, "run-mismatch")
    user = await script.prompt("start")
    assistant = AssistantMessage(text="done")
    await script.model_turn(0, assistant)
    await script.commit(0, [user, assistant])
    await journal.close()
    mismatch = await harness.resume(
        "run-mismatch",
        model=ScriptedModel([], model="other-model"),
    )
    assert isinstance(mismatch, ResumeRejected)
    assert mismatch.reason == "model_mismatch"

    # An unfinished journal resumes fine; an active writer reports busy.
    journal, script = await _scripted_journal(store, "run-unfinished")
    user = await script.prompt("start")
    assistant = AssistantMessage(text="done")
    await script.model_turn(0, assistant)
    await script.commit(0, [user, assistant])
    # Keep the writer lease held.
    busy = await harness.resume("run-unfinished", model=ScriptedModel([]))
    assert isinstance(busy, ResumeRejected)
    assert busy.reason == "busy"
    await journal.close()

    ok = await harness.resume(
        "run-unfinished", model=ScriptedModel([]), tool_registry=_registry()
    )
    assert isinstance(ok, SessionRun)
    await ok.close()


@pytest.mark.asyncio
async def test_resume_rejects_missing_lineage_tools_with_names() -> None:
    store = InMemoryJournalStore()
    journal, script = await _scripted_journal(store, "run-tools")
    user = await script.prompt("start")
    assistant = AssistantMessage(text="done")
    await script.model_turn(0, assistant)
    await script.commit(0, [user, assistant])
    await journal.close()

    harness = SessionHarness(store)
    rejected = await harness.resume("run-tools", model=ScriptedModel([]))
    assert isinstance(rejected, ResumeRejected)
    assert rejected.reason == "tools_missing"
    assert rejected.missing_tools == ("echo",)

    accepted = await harness.resume(
        "run-tools", model=ScriptedModel([]), tool_registry=_registry()
    )
    assert isinstance(accepted, SessionRun)
    await accepted.close()


@pytest.mark.asyncio
async def test_resume_corruption_raises_not_rejects() -> None:
    store = InMemoryJournalStore()
    journal, script = await _scripted_journal(store, "run-corrupt")
    user = await script.prompt("start")
    assistant = AssistantMessage(text="done")
    await script.model_turn(0, assistant)
    await script.commit(0, [user, assistant])
    # A dangling model.completed reference is a contradiction, not a rejection.
    await journal.append(
        JournalRecordType.MODEL_COMPLETED,
        encode_model_completed(
            1, _request("run-corrupt", 1), "run-corrupt:turn:9:transcript:9"
        ),
        record_id="run-corrupt:turn:1:model",
    )
    await journal.close()

    harness = SessionHarness(store)
    with pytest.raises(JournalCorruptionError):
        await harness.resume(
            "run-corrupt", model=ScriptedModel([]), tool_registry=_registry()
        )


# ── fork ──────────────────────────────────────────────────────────────────────


async def _two_turn_terminal_journal(
    store: InMemoryJournalStore, run_id: str
) -> None:
    journal, script = await _scripted_journal(store, run_id)
    user = await script.prompt("start")
    first = AssistantMessage(
        text="working",
        tool_calls=(ToolCall(id="c1", name="echo", arguments={"text": "x"}),),
    )
    await script.model_turn(0, first)
    first_results = await script.tool_results(0, first, "echo:x")
    await script.commit(0, [user, first, *first_results])
    second = AssistantMessage(text="done")
    await script.model_turn(1, second)
    await script.commit(1, [second])
    from qitos.core.agent_loop import AgentLoopResult

    await script.recorder.run_terminal(
        AgentLoopResult(
            status=AgentRunStatus.COMPLETED,
            messages=(user, first, *first_results, second),
        )
    )
    await journal.close()


@pytest.mark.asyncio
async def test_fork_default_and_explicit_position() -> None:
    store = InMemoryJournalStore()
    await _two_turn_terminal_journal(store, "run-parent")
    harness = SessionHarness(store)
    model = ScriptedModel([text_events("child answer"), text_events("grandchild")])

    forked = await harness.fork("run-parent", model=model, tool_registry=_registry())
    assert isinstance(forked, SessionRun)
    assert forked.run_id != "run-parent"
    # Default position is the latest committed boundary: the full transcript.
    assert [message.role for message in forked.agent.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]

    parent_records = await _replay(store, "run-parent")
    first_commit = next(
        record
        for record in parent_records
        if record.type is JournalRecordType.STEP_COMMITTED
    )
    explicit = await harness.fork(
        "run-parent",
        first_commit.position,
        model=model,
        tool_registry=_registry(),
    )
    assert isinstance(explicit, SessionRun)
    assert [message.role for message in explicit.agent.messages] == [
        "user",
        "assistant",
        "tool",
    ]
    await explicit.close()

    result = await forked.prompt("continue from parent")
    assert result.status is AgentRunStatus.COMPLETED
    # Fork of a fork: nested inherited wrappers resolve fail closed.
    forked_run_id = forked.run_id
    await forked.close()
    grandchild = await harness.fork(
        forked_run_id, model=model, tool_registry=_registry()
    )
    assert isinstance(grandchild, SessionRun)
    third = await grandchild.prompt("continue from child")
    assert third.status is AgentRunStatus.COMPLETED
    await grandchild.close()

    records = await _replay(store, grandchild.run_id)
    recovered = recover_session(records)
    roles = [message.role for message in recovered.transcript]
    assert roles == ["user", "assistant", "tool", "assistant", "user", "assistant", "user", "assistant"]


@pytest.mark.asyncio
async def test_fork_never_reprojects_inherited_runtime_inputs() -> None:
    store = InMemoryJournalStore()
    journal, script = await _scripted_journal(store, "run-fork-inputs")
    user = await script.prompt("start")
    first = AssistantMessage(
        text="working",
        tool_calls=(ToolCall(id="c1", name="echo", arguments={"text": "x"}),),
    )
    await script.model_turn(0, first)
    first_results = await script.tool_results(0, first, "echo:x")
    await script.commit(0, [user, first, *first_results])
    posted = _runtime_input("child-1:terminal", "background child finished")
    await journal.append(
        JournalRecordType.RUNTIME_INPUT_POSTED,
        posted.to_dict(),
        record_id="run-fork-inputs:runtime:child-1:terminal",
    )
    consumed = _runtime_input("child-2:terminal", "already delivered child")
    await journal.append(
        JournalRecordType.RUNTIME_INPUT_POSTED,
        consumed.to_dict(),
        record_id="run-fork-inputs:runtime:child-2:terminal",
    )
    await journal.append(
        JournalRecordType.RUNTIME_INPUT_CONSUMED,
        encode_runtime_input_consumed(consumed.event_id),
        record_id="run-fork-inputs:runtime:child-2:terminal:consumed",
    )
    await journal.close()

    harness = SessionHarness(store)
    forked = await harness.fork(
        "run-fork-inputs", model=ScriptedModel([]), tool_registry=_registry()
    )
    assert isinstance(forked, SessionRun)
    # Neither the consumed input nor the parent's own unconsumed input is
    # re-projected across a fork: inherited facts are never redelivered.
    assert not forked.agent.has_queued_messages()
    await forked.close()


# ── runtime input consumption ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_posted_input_is_steered_and_consumed_durably() -> None:
    store = InMemoryJournalStore()
    harness = SessionHarness(store)
    model = ScriptedModel([text_events("seen")])
    session_run = await harness.start(model=model)
    accepted = await session_run.post_runtime_event(
        _runtime_input("child-9:terminal", "child finished")
    )
    assert accepted is True
    result = await session_run.prompt("please continue")
    assert result.status is AgentRunStatus.COMPLETED
    run_id = session_run.run_id
    await session_run.close()

    # The steered message reached the model inside the same turn.
    contents = [
        str(message.get("content"))
        for message in model.requests[0].messages
        if message.get("role") == "user"
    ]
    assert any("child finished" in content for content in contents)
    records = await _replay(store, run_id)
    posted = [
        record
        for record in records
        if record.type is JournalRecordType.RUNTIME_INPUT_POSTED
    ]
    consumed = [
        record
        for record in records
        if record.type is JournalRecordType.RUNTIME_INPUT_CONSUMED
    ]
    assert len(posted) == 1
    assert len(consumed) == 1


@pytest.mark.asyncio
async def test_unconsumed_input_is_resteered_exactly_once_on_resume() -> None:
    store = InMemoryJournalStore()
    journal, script = await _scripted_journal(store, "run-input-crash")
    user = await script.prompt("start")
    assistant = AssistantMessage(
        text="working",
        tool_calls=(ToolCall(id="c1", name="echo", arguments={"text": "x"}),),
    )
    await script.model_turn(0, assistant)
    results = await script.tool_results(0, assistant, "echo:x")
    await script.commit(0, [user, assistant, *results])
    posted = _runtime_input("child-7:terminal", "late child result")
    await journal.append(
        JournalRecordType.RUNTIME_INPUT_POSTED,
        posted.to_dict(),
        record_id="run-input-crash:runtime:child-7:terminal",
    )
    # The process died before the steered message was injected.
    await journal.close()

    harness = SessionHarness(store)
    model = ScriptedModel([text_events("resumed answer")])
    resumed = await harness.resume(
        "run-input-crash", model=model, tool_registry=_registry()
    )
    assert isinstance(resumed, SessionRun)
    assert resumed.agent.has_queued_messages()
    result = await resumed.continue_run()
    assert result.status is AgentRunStatus.COMPLETED
    await resumed.close()

    contents = [
        str(message.get("content"))
        for message in model.requests[0].messages
        if message.get("role") == "user"
    ]
    assert sum("late child result" in content for content in contents) == 1
    records = await _replay(store, "run-input-crash")
    posted_records = [
        record
        for record in records
        if record.type is JournalRecordType.RUNTIME_INPUT_POSTED
    ]
    consumed_records = [
        record
        for record in records
        if record.type is JournalRecordType.RUNTIME_INPUT_CONSUMED
    ]
    # No duplicate POSTED record; exactly one durable consumption.
    assert len(posted_records) == 1
    assert len(consumed_records) == 1

    # Recovery finds nothing left to re-project: the consumed input is done.
    recovered = recover_session(records)
    assert recovered.unconsumed_inputs == ()


@pytest.mark.asyncio
async def test_consumed_input_is_never_resteered_on_resume() -> None:
    store = InMemoryJournalStore()
    journal, script = await _scripted_journal(store, "run-input-done")
    user = await script.prompt("start")
    assistant = AssistantMessage(
        text="working",
        tool_calls=(ToolCall(id="c1", name="echo", arguments={"text": "x"}),),
    )
    await script.model_turn(0, assistant)
    results = await script.tool_results(0, assistant, "echo:x")
    await script.commit(0, [user, assistant, *results])
    posted = _runtime_input("child-8:terminal", "already delivered")
    await journal.append(
        JournalRecordType.RUNTIME_INPUT_POSTED,
        posted.to_dict(),
        record_id="run-input-done:runtime:child-8:terminal",
    )
    await journal.append(
        JournalRecordType.RUNTIME_INPUT_CONSUMED,
        encode_runtime_input_consumed(posted.event_id),
        record_id="run-input-done:runtime:child-8:terminal:consumed",
    )
    await journal.close()

    harness = SessionHarness(store)
    resumed = await harness.resume(
        "run-input-done",
        model=ScriptedModel([text_events("clean resume")]),
        tool_registry=_registry(),
    )
    assert isinstance(resumed, SessionRun)
    assert not resumed.agent.has_queued_messages()
    result = await resumed.continue_run()
    assert result.status is AgentRunStatus.COMPLETED
    await resumed.close()
    records = await _replay(store, "run-input-done")
    assert (
        len(
            [
                record
                for record in records
                if record.type is JournalRecordType.RUNTIME_INPUT_CONSUMED
            ]
        )
        == 1
    )


# ── manual and automatic compaction ──────────────────────────────────────────


async def _three_turn_journal(store: InMemoryJournalStore, run_id: str) -> None:
    """Three committed turns of ~100 estimated tokens per message."""

    journal, script = await _scripted_journal(store, run_id)
    user0 = await script.prompt("u" * 400)
    first = AssistantMessage(
        text="a" * 400,
        tool_calls=(ToolCall(id="c1", name="echo", arguments={"text": "x"}),),
    )
    await script.model_turn(0, first)
    first_results = await script.tool_results(0, first, "r" * 400)
    await script.commit(0, [user0, first, *first_results])
    user1 = UserMessage(content="v" * 400)
    second = AssistantMessage(text="b" * 400)
    await script.recorder.turn_frozen(1, _snapshot())
    await script.recorder.model_terminal(1, _request(run_id, 1), second)
    await script.commit(1, [user1, second])
    user2 = UserMessage(content="w" * 400)
    third = AssistantMessage(
        text="c" * 400,
        tool_calls=(ToolCall(id="c2", name="echo", arguments={"text": "y"}),),
        usage=_usage(50_000),
    )
    await script.recorder.turn_frozen(2, _snapshot())
    await script.recorder.model_terminal(2, _request(run_id, 2), third)
    third_results = await script.tool_results(2, third, "s" * 400)
    await script.commit(2, [user2, third, *third_results])
    await journal.close()


@pytest.mark.asyncio
async def test_manual_compact_rejections() -> None:
    store = InMemoryJournalStore()
    harness = SessionHarness(store)
    session_run = await harness.start(model=ScriptedModel([]))
    empty = await session_run.compact()
    assert isinstance(empty, CompactRejected)
    assert empty.reason == "nothing_to_compact"

    gate = asyncio.Event()
    busy_model = ScriptedModel([make_hanging_model(gate, first_text="working")])
    hanging = await harness.start(model=busy_model)
    running = asyncio.create_task(hanging.prompt("work"))
    await asyncio.wait_for(_wait_until_active(hanging), timeout=1)
    rejected = await hanging.compact()
    assert isinstance(rejected, CompactRejected)
    assert rejected.reason == "busy"
    gate.set()
    await running
    await hanging.close()
    await session_run.close()


async def _wait_until_active(session_run: SessionRun) -> None:
    while not session_run.agent.is_streaming:
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_manual_compact_swaps_context_and_survives_resume() -> None:
    store = InMemoryJournalStore()
    await _three_turn_journal(store, "run-compact")
    settings = CompactionSettings(keep_recent_tokens=410)
    summary_usage = {"total_tokens": 40, "prompt_tokens": 30, "completion_tokens": 10}
    model = ScriptedModel(
        [text_events("summary of early work", usage=summary_usage), text_events("after")]
    )
    harness = SessionHarness(store, compaction=settings)
    resumed = await harness.resume(
        "run-compact", model=model, tool_registry=_registry()
    )
    assert isinstance(resumed, SessionRun)
    result = await resumed.compact()
    assert not isinstance(result, CompactRejected)
    assert result.summary == "summary of early work"
    assert result.tokens_before > 0

    # The context is the durable summary plus the kept tail.
    messages = resumed.agent.messages
    assert isinstance(messages[0], UserMessage)
    assert messages[0].content == "summary of early work"
    assert [message.role for message in messages] == [
        "user",
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
    ]
    await resumed.close()

    records = await _replay(store, "run-compact")
    compaction_records = [
        record for record in records if record.type is JournalRecordType.COMPACTION
    ]
    assert len(compaction_records) == 1
    summary, first_kept, tokens_before, usage = decode_compaction(
        compaction_records[0].payload
    )
    assert summary == "summary of early work"
    assert tokens_before == result.tokens_before
    assert usage is not None and usage.total_tokens == 40

    # Resume across the compaction projects the same context and continues.
    again = await harness.resume(
        "run-compact", model=model, tool_registry=_registry()
    )
    assert isinstance(again, SessionRun)
    assert [message.role for message in again.agent.messages] == [
        "user",
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
    ]
    continued = await again.continue_run()
    assert continued.status is AgentRunStatus.COMPLETED
    await again.close()


@pytest.mark.asyncio
async def test_auto_threshold_compaction_fires_before_the_next_prompt() -> None:
    store = InMemoryJournalStore()
    await _three_turn_journal(store, "run-auto")
    settings = CompactionSettings(keep_recent_tokens=410)
    model = ScriptedModel(
        [text_events("auto summary"), text_events("post compaction answer")]
    )
    model.context_window = 60_000
    harness = SessionHarness(store, compaction=settings)
    resumed = await harness.resume(
        "run-auto", model=model, tool_registry=_registry()
    )
    assert isinstance(resumed, SessionRun)
    result = await resumed.prompt("next work")
    assert result.status is AgentRunStatus.COMPLETED
    await resumed.close()

    # The threshold fired before the leg's first model request: the summary
    # request went first, then the run started from the compacted context.
    assert len(model.requests) == 2
    run_request = model.requests[1]
    first_message = run_request.messages[0]
    assert first_message.get("role") == "user"
    assert first_message.get("content") == "auto summary"
    records = await _replay(store, "run-auto")
    assert any(
        record.type is JournalRecordType.COMPACTION for record in records
    )


@pytest.mark.asyncio
async def test_no_auto_compaction_without_settings_or_below_threshold() -> None:
    store = InMemoryJournalStore()
    await _three_turn_journal(store, "run-no-auto")
    model = ScriptedModel([text_events("answer")])
    model.context_window = 60_000
    harness = SessionHarness(store)
    resumed = await harness.resume(
        "run-no-auto", model=model, tool_registry=_registry()
    )
    assert isinstance(resumed, SessionRun)
    result = await resumed.prompt("next work")
    assert result.status is AgentRunStatus.COMPLETED
    await resumed.close()
    # Without configured settings the model is the only requester.
    assert len(model.requests) == 1

    await _three_turn_journal(store, "run-below")
    below = SessionHarness(store, compaction=CompactionSettings())
    model_below = ScriptedModel([text_events("answer")])
    model_below.context_window = 200_000
    resumed_below = await below.resume(
        "run-below", model=model_below, tool_registry=_registry()
    )
    assert isinstance(resumed_below, SessionRun)
    result_below = await resumed_below.prompt("more work")
    assert result_below.status is AgentRunStatus.COMPLETED
    await resumed_below.close()
    assert len(model_below.requests) == 1


# ── overflow recovery ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_overflow_failure_compacts_and_continues_once() -> None:
    store = InMemoryJournalStore()
    settings = CompactionSettings()
    model = ScriptedModel(
        [
            tool_events([tool_call_wire("c1", "big_echo", {"text": "big"})]),
            failed_events("prompt is too long: 213462 tokens > 200000 maximum"),
            text_events("compact prefix summary"),
            text_events("recovered answer"),
        ]
    )

    @tool(name="big_echo")
    def _big_echo(text: str) -> str:
        return "x" * 100_000

    harness = SessionHarness(store, compaction=settings)
    registry = ToolRegistry().register(_big_echo)
    session_run = await harness.start(model=model, tool_registry=registry)
    result = await session_run.prompt("handle the big result")
    assert result.status is AgentRunStatus.COMPLETED
    first_run_id = session_run.run_id
    await session_run.close()

    # The retry ran on a forked journal carrying the compaction record; the
    # original failure stays durable in the first journal.
    assert first_run_id in store.records
    child_records = await _replay(store, session_run.run_id)
    assert any(
        record.type is JournalRecordType.COMPACTION for record in child_records
    )
    recovered = recover_session(child_records)
    assert recovered.outcome is not None
    assert recovered.outcome.status is AgentRunStatus.COMPLETED
    # turn0 tool turn, turn1 overflow failure, summary request, retry.
    assert len(model.requests) == 4


@pytest.mark.asyncio
async def test_overflow_recovery_is_one_shot() -> None:
    store = InMemoryJournalStore()
    model = ScriptedModel(
        [
            tool_events([tool_call_wire("c1", "big_echo", {"text": "big"})]),
            failed_events("prompt is too long: 213462 tokens > 200000 maximum"),
            text_events("compact prefix summary"),
            failed_events("prompt is too long: 213462 tokens > 200000 maximum"),
        ]
    )

    @tool(name="big_echo")
    def _big_echo(text: str) -> str:
        return "x" * 100_000

    harness = SessionHarness(store, compaction=CompactionSettings())
    session_run = await harness.start(
        model=model, tool_registry=ToolRegistry().register(_big_echo)
    )
    result = await session_run.prompt("handle the big result")
    assert result.status is AgentRunStatus.FAILED
    await session_run.close()
    # No second compact-and-retry attempt happened after the retried failure.
    assert len(model.requests) == 4


# ── lifecycle ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_close_aborts_active_run_and_releases_the_journal() -> None:
    store = InMemoryJournalStore()
    harness = SessionHarness(store)
    gate = asyncio.Event()
    model = ScriptedModel([make_hanging_model(gate)])
    session_run = await harness.start(model=model)
    running = asyncio.create_task(session_run.prompt("hang"))
    await asyncio.wait_for(_wait_until_active(session_run), timeout=1)
    await session_run.close()
    result = await running
    assert result.status is AgentRunStatus.ABORTED
    gate.set()

    records = await _replay(store, session_run.run_id)
    assert records[-1].type is JournalRecordType.RUN_INTERRUPTED

    # The released lease makes the journal resumable instead of busy.
    resumed = await harness.resume(
        session_run.run_id, model=ScriptedModel([text_events("back")])
    )
    assert not (
        isinstance(resumed, ResumeRejected) and resumed.reason == "busy"
    )
    if isinstance(resumed, SessionRun):
        await resumed.close()


@pytest.mark.asyncio
async def test_budget_ledger_follows_the_lineage_across_legs() -> None:
    store = InMemoryJournalStore()
    ledger = BudgetLedger(max_tokens=10_000)
    harness = SessionHarness(store)
    usage = {"total_tokens": 100, "prompt_tokens": 60, "completion_tokens": 40}
    model = ScriptedModel([text_events("one", usage=usage), text_events("two", usage=usage)])
    session_run = await harness.start(model=model, budget_ledger=ledger)
    await session_run.prompt("first")
    first_totals = session_run.budget_ledger.snapshot().total_tokens
    assert first_totals == 100
    await session_run.prompt("second")
    # The advanced journal replays the inherited budget commits, so the
    # lineage totals accumulate under the same limits.
    assert session_run.budget_ledger.snapshot().total_tokens == 200
    await session_run.close()

    records = await _replay(store, session_run.run_id)
    assert any(
        record.type is JournalRecordType.BUDGET_COMMITTED for record in records
    )
