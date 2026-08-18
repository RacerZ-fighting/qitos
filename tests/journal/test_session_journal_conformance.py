"""Shared SessionJournal conformance suite.

One behavior contract, two implementations: the durable JSONL journal and
the in-memory journal. JSONL-only concerns (writer lease, tail repair, the
SQLite projection) stay in ``test_session_journal.py``.
"""

from __future__ import annotations

import pytest

from qitos.core.journal import (
    JournalClosedError,
    JournalCorruptionError,
    JournalError,
    JournalOwnershipError,
    JournalRecordRef,
    JournalRecordType,
    resolve_inherited_record,
)
from qitos.core.message import ToolCall, ToolResultMessage, UserMessage
from qitos.core.tool_result import ToolResult
from qitos.kit.journal import (
    InMemoryJournalStore,
    InMemorySessionJournal,
    JsonlSessionJournal,
    committed_tool_transactions,
    recover_session,
)
from qitos.kit.journal.turn_recorder import (
    encode_step_committed,
    encode_tool_terminal,
    encode_transcript_message,
)


@pytest.fixture(params=["jsonl", "memory"])
def backend(request, tmp_path):
    if request.param == "jsonl":
        root = tmp_path / "journals"

        def factory():
            return JsonlSessionJournal(root)

    else:
        store = InMemoryJournalStore()

        def factory():
            return InMemorySessionJournal(store)

    return factory


def _user_entry(
    run_id: str, turn: int = 0, seq: int = 0, text: str = "go"
) -> dict:
    return encode_transcript_message(UserMessage(content=text, timestamp=1.0))


async def _tool_pair(
    journal, run_id: str, call_id: str = "c1", turn: int = 0, seq: int = 0
):
    call = ToolCall(id=call_id, name="echo", arguments={"text": "x"})
    message = ToolResultMessage(
        tool_call_id=call.id,
        tool_name=call.name,
        result=ToolResult(
            output=f"out-{call_id}",
            error=None,
            metadata={"evidence": call_id},
            call_id=call.id,
        ),
    )
    transcript_id = f"{run_id}:turn:{turn}:transcript:{seq}"
    terminal_id = f"{run_id}:turn:{turn}:tool:{call_id}:terminal"
    await journal.append(
        JournalRecordType.TRANSCRIPT_MESSAGE,
        encode_transcript_message(message),
        record_id=transcript_id,
    )
    await journal.append(
        JournalRecordType.TOOL_TERMINAL,
        encode_tool_terminal(turn, call, transcript_id),
        record_id=terminal_id,
    )
    return call, transcript_id, terminal_id


@pytest.mark.asyncio
async def test_append_replay_round_trip(backend) -> None:
    journal = backend()
    first = await journal.create("run-1", {"task": "inspect"})
    second = await journal.append(
        JournalRecordType.TRANSCRIPT_MESSAGE,
        _user_entry("run-1"),
        record_id="run-1:turn:0:transcript:0",
    )
    third = await journal.append(
        JournalRecordType.INPUT_ACCEPTED,
        {"transcript_record_ids": ["run-1:turn:0:transcript:0"]},
        record_id="run-1:input",
    )

    assert (first.seq, second.seq, third.seq) == (1, 2, 3)
    records = await journal.replay()
    assert [record.record_id for record in records] == [
        "run-1:start",
        "run-1:turn:0:transcript:0",
        "run-1:input",
    ]
    assert all(record.run_id == "run-1" for record in records)
    # Replay returns isolated copies.
    records[1].payload["message"]["content"] = "mutated"
    assert (await journal.replay())[1].payload["message"]["content"] == "go"
    await journal.close()


@pytest.mark.asyncio
async def test_idempotent_and_conflicting_record_id(backend) -> None:
    journal = backend()
    await journal.create("run-1", {})

    first = await journal.append(
        JournalRecordType.TRANSCRIPT_MESSAGE,
        _user_entry("run-1"),
        record_id="run-1:turn:0:transcript:0",
    )
    settled = await journal.append(
        JournalRecordType.TRANSCRIPT_MESSAGE,
        _user_entry("run-1"),
        record_id="run-1:turn:0:transcript:0",
    )
    assert settled == first
    assert [
        record.record_id for record in await journal.replay()
    ].count("run-1:turn:0:transcript:0") == 1

    with pytest.raises(JournalError, match="different content"):
        await journal.append(
            JournalRecordType.TRANSCRIPT_MESSAGE,
            _user_entry("run-1", text="changed"),
            record_id="run-1:turn:0:transcript:0",
        )
    await journal.close()


@pytest.mark.asyncio
async def test_closed_journal_rejects_mutation_but_keeps_read_views(backend) -> None:
    journal = backend()
    await journal.create("run-1", {})
    call, _transcript_id, terminal_id = await _tool_pair(journal, "run-1")
    await journal.append(
        JournalRecordType.STEP_COMMITTED,
        encode_step_committed(0, ["run-1:turn:0:transcript:0"], [terminal_id]),
        record_id="run-1:turn:0:committed",
    )
    await journal.close()

    with pytest.raises(JournalClosedError):
        await journal.append(
            JournalRecordType.INPUT_ACCEPTED,
            {"transcript_record_ids": []},
            record_id="late",
        )
    with pytest.raises(JournalClosedError):
        await journal.flush()
    with pytest.raises(JournalClosedError):
        await journal.create("run-2", {})

    records = await journal.replay()
    assert len(records) == 4
    view = journal.find_tool_transaction(JournalRecordRef("run-1", terminal_id))
    assert view is not None
    assert view.action == call


@pytest.mark.asyncio
async def test_reopen_restores_records_and_query_views(backend) -> None:
    journal = backend()
    await journal.create("run-1", {"agent": "worker"})
    call, _t, terminal_id = await _tool_pair(journal, "run-1")
    await journal.append(
        JournalRecordType.STEP_COMMITTED,
        encode_step_committed(0, ["run-1:turn:0:transcript:0"], [terminal_id]),
        record_id="run-1:turn:0:committed",
    )
    before = await journal.replay()
    await journal.close()

    reopened = backend()
    await reopened.open("run-1")
    assert [record.to_dict() for record in await reopened.replay()] == [
        record.to_dict() for record in before
    ]
    view = reopened.find_tool_transaction(JournalRecordRef("run-1", terminal_id))
    assert view is not None
    assert view.action == call
    assert view.result.model_visible_output == "out-c1"
    await reopened.close()


@pytest.mark.asyncio
async def test_single_writer_ownership(backend) -> None:
    journal = backend()
    await journal.create("run-1", {})

    with pytest.raises((JournalError, OSError)):
        await backend().create("run-1", {})
    with pytest.raises(JournalOwnershipError):
        await backend().open("run-1")

    await journal.close()
    successor = backend()
    await successor.open("run-1")
    await successor.close()


@pytest.mark.asyncio
async def test_tool_transaction_join_requires_commit_and_transcript(backend) -> None:
    journal = backend()
    await journal.create("run-1", {})
    call, transcript_id, terminal_id = await _tool_pair(journal, "run-1")
    reference = JournalRecordRef("run-1", terminal_id)

    assert journal.find_tool_transaction(reference) is None

    commit = await journal.append(
        JournalRecordType.STEP_COMMITTED,
        encode_step_committed(0, [transcript_id], [terminal_id]),
        record_id="run-1:turn:0:committed",
    )
    view = journal.find_tool_transaction(reference)
    assert view is not None
    assert view.terminal == reference
    assert view.committed_at == commit
    assert view.action == call
    assert view.result.metadata == {"evidence": "c1"}

    assert journal.find_tool_transaction(
        JournalRecordRef("run-1", "run-1:turn:0:tool:unknown:terminal")
    ) is None
    assert journal.find_tool_transaction(
        JournalRecordRef("other-run", terminal_id)
    ) is None

    # Retired shapes (embedded result) fail closed at the index boundary.
    with pytest.raises(JournalCorruptionError):
        await journal.append(
            JournalRecordType.TOOL_TERMINAL,
            {
                "turn": 0,
                "call_id": "c9",
                "call": {"id": "c9", "name": "echo", "arguments": {}},
                "result": {
                    "status": "success",
                    "output": None,
                    "error": None,
                    "metadata": {},
                },
            },
            record_id="run-1:turn:0:tool:c9:terminal",
        )
    await journal.close()


@pytest.mark.asyncio
async def test_committed_tool_transactions_preserve_commit_and_fork_order(
    backend,
) -> None:
    parent = backend()
    await parent.create("parent", {})
    first, first_entry, first_terminal = await _tool_pair(
        parent, "parent", call_id="first", turn=0
    )
    boundary = await parent.append(
        JournalRecordType.STEP_COMMITTED,
        encode_step_committed(0, [first_entry], [first_terminal]),
        record_id="parent:turn:0:committed",
    )
    child = await parent.fork(boundary, "child")
    second, second_entry, second_terminal = await _tool_pair(
        child, "child", call_id="second", turn=1
    )
    await child.append(
        JournalRecordType.STEP_COMMITTED,
        encode_step_committed(1, [second_entry], [second_terminal]),
        record_id="child:turn:1:committed",
    )

    projected = await committed_tool_transactions(child)

    assert [item.action for item in projected] == [first, second]
    assert [item.terminal.run_id for item in projected] == ["parent", "child"]
    await child.close()
    await parent.close()


@pytest.mark.asyncio
async def test_fork_lineage_and_nested_inherited_resolution(backend) -> None:
    parent = backend()
    await parent.create("parent", {"task": "inspect"})
    call, transcript_id, terminal_id = await _tool_pair(parent, "parent")
    boundary = await parent.append(
        JournalRecordType.STEP_COMMITTED,
        encode_step_committed(0, [transcript_id], [terminal_id]),
        record_id="parent:turn:0:committed",
    )

    child = await parent.fork(boundary, "child")
    child_records = await child.replay()
    assert [record.type for record in child_records[:2]] == [
        JournalRecordType.RUN_STARTED,
        JournalRecordType.RUN_FORKED,
    ]
    assert all(
        record.type is JournalRecordType.INHERITED for record in child_records[2:]
    )
    wrapper = child_records[-1]
    origin = resolve_inherited_record(wrapper)
    assert origin.record_id == "parent:turn:0:committed"
    assert origin.run_id == "parent"

    inherited_view = child.find_tool_transaction(
        JournalRecordRef("parent", terminal_id)
    )
    assert inherited_view is not None
    assert inherited_view.action == call
    assert inherited_view.result.model_visible_output == "out-c1"
    assert child.find_tool_transaction(
        JournalRecordRef("child", terminal_id)
    ) is None

    # A fork of a fork nests wrappers and still resolves to the origin.
    child_commit = await child.append(
        JournalRecordType.STEP_COMMITTED,
        encode_step_committed(0, [], []),
        record_id="child:turn:0:committed",
    )
    grandchild = await child.fork(child_commit, "grandchild")
    grandchild_records = await grandchild.replay()
    nested = next(
        record
        for record in grandchild_records
        if record.record_id.endswith("parent:turn:0:committed")
    )
    resolved = resolve_inherited_record(nested)
    assert resolved.run_id == "parent"
    assert resolved.record_id == "parent:turn:0:committed"
    assert grandchild.find_tool_transaction(
        JournalRecordRef("parent", terminal_id)
    ) is not None

    # Recovery replays one forked journal as self-contained truth.
    recovered = recover_session(grandchild_records)
    assert recovered.run_id == "grandchild"
    assert recovered.unterminated_calls == ()

    with pytest.raises(ValueError, match="committed boundary"):
        await parent.fork(
            (await parent.replay())[0].position,
            "invalid-fork",
        )
    await grandchild.close()
    await child.close()
    await parent.close()


@pytest.mark.asyncio
async def test_recovery_round_trip_through_both_backends(backend) -> None:
    journal = backend()
    await journal.create("run-1", {})
    await journal.append(
        JournalRecordType.TRANSCRIPT_MESSAGE,
        _user_entry("run-1"),
        record_id="run-1:turn:0:transcript:0",
    )
    await journal.append(
        JournalRecordType.INPUT_ACCEPTED,
        {"transcript_record_ids": ["run-1:turn:0:transcript:0"]},
        record_id="run-1:input",
    )
    _call, transcript_id, terminal_id = await _tool_pair(
        journal, "run-1", turn=0, seq=1
    )
    await journal.append(
        JournalRecordType.STEP_COMMITTED,
        encode_step_committed(0, ["run-1:turn:0:transcript:0", transcript_id], [terminal_id]),
        record_id="run-1:turn:0:committed",
    )
    records = await journal.replay()
    await journal.close()

    recovered = recover_session(records)
    assert recovered.run_id == "run-1"
    assert [message.role for message in recovered.transcript] == ["user", "tool"]
    assert recovered.next_turn == 1
    assert recovered.outcome is None
