"""JournalTurnTransaction integration over the real JSONL journal."""

from __future__ import annotations

import pytest

from qitos.core.agent_loop import AgentContext, AgentLoopConfig, AgentRunStatus
from qitos.core.agent_loop import run_agent_loop
from qitos.core.journal import JournalRecordType
from qitos.core.message import (
    AssistantMessage,
    ToolResultMessage,
    UserMessage,
    message_from_dict,
)
from qitos.core.model_response import ModelUsage
from qitos.core.tool import tool
from qitos.core.tool_registry import ToolRegistry
from qitos.core.tool_result import ToolResult
from qitos.kit.journal import JsonlSessionJournal, JournalTurnTransaction

from tests.core.agent_fakes import ScriptedModel, text_events, tool_call_wire, tool_events


@tool(name="echo")
def _echo(text: str) -> str:
    return f"echo:{text}"


@pytest.mark.asyncio
async def test_full_run_records_the_complete_transaction(tmp_path) -> None:
    journal = JsonlSessionJournal(tmp_path)
    transaction = await JournalTurnTransaction.create(
        journal, "run-journal-1", {"purpose": "test"}
    )
    model = ScriptedModel(
        [tool_events([tool_call_wire("c1", "echo", {"text": "x"})]), text_events("done")]
    )
    context = AgentContext(
        messages=[], tools=ToolRegistry().register(_echo).freeze()
    )
    config = AgentLoopConfig(model=model, run_id="run-journal-1", transaction=transaction)
    result = await run_agent_loop([UserMessage(content="go")], context, config, None)
    assert result.status is AgentRunStatus.COMPLETED
    await journal.close()

    reopened = JsonlSessionJournal(tmp_path)
    await reopened.open("run-journal-1")
    records = await reopened.replay()
    await reopened.close()

    types = [record.type for record in records]
    assert types == [
        JournalRecordType.RUN_STARTED,
        JournalRecordType.MODEL_COMPLETED,
        JournalRecordType.TOOL_STARTED,
        JournalRecordType.TOOL_TERMINAL,
        JournalRecordType.STEP_COMMITTED,
        JournalRecordType.MODEL_COMPLETED,
        JournalRecordType.STEP_COMMITTED,
        JournalRecordType.RUN_COMPLETED,
    ]
    # Deterministic record ids make a replayed run idempotent to audit.
    ids = [record.record_id for record in records]
    assert "run-journal-1:turn:0:tool:c1:started" in ids
    assert "run-journal-1:turn:0:tool:c1:terminal" in ids
    assert "run-journal-1:run:terminal" in ids

    model_record = records[1]
    request_payload = model_record.payload["request"]
    assert request_payload["run_id"] == "run-journal-1"
    assistant = message_from_dict(model_record.payload["message"])
    assert isinstance(assistant, AssistantMessage)
    assert assistant.tool_calls[0].id == "c1"

    terminal = records[3]
    assert terminal.payload["call_id"] == "c1"
    assert terminal.payload["result"]["status"] == "success"

    committed = records[4]
    messages = [message_from_dict(item) for item in committed.payload["messages"]]
    assert any(isinstance(m, ToolResultMessage) for m in messages)

    terminal_record = records[-1]
    assert terminal_record.payload["status"] == "completed"
    assert terminal_record.payload["error"] is None


@pytest.mark.asyncio
async def test_aborted_run_records_run_interrupted(tmp_path) -> None:
    from qitos.core.cancellation import CancelToken

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

    reopened = JsonlSessionJournal(tmp_path)
    await reopened.open("run-journal-2")
    records = await reopened.replay()
    await reopened.close()

    assert records[-1].type is JournalRecordType.RUN_INTERRUPTED
    assert records[-1].payload["status"] == "aborted"


@pytest.mark.asyncio
async def test_committed_tool_transaction_is_queryable_after_reopen(tmp_path) -> None:
    from qitos.core.journal import JournalRecordRef

    journal = JsonlSessionJournal(tmp_path)
    transaction = await JournalTurnTransaction.create(
        journal, "run-journal-3", {"purpose": "test"}
    )
    model = ScriptedModel(
        [tool_events([tool_call_wire("c1", "echo", {"text": "x"})]), text_events("done")]
    )
    context = AgentContext(
        messages=[], tools=ToolRegistry().register(_echo).freeze()
    )
    config = AgentLoopConfig(
        model=model, run_id="run-journal-3", transaction=transaction
    )
    result = await run_agent_loop([UserMessage(content="go")], context, config, None)
    assert result.status is AgentRunStatus.COMPLETED
    await journal.close()

    reopened = JsonlSessionJournal(tmp_path)
    await reopened.open("run-journal-3")
    records = await reopened.replay()
    committed = next(
        record
        for record in records
        if record.type is JournalRecordType.STEP_COMMITTED
    )
    # The commit record links its Tool terminal records, so the index rebuilt
    # from disk can serve committed Tool transaction queries again.
    terminal_id = "run-journal-3:turn:0:tool:c1:terminal"
    assert terminal_id in committed.payload["terminal_record_ids"]

    view = reopened.find_tool_transaction(
        JournalRecordRef("run-journal-3", terminal_id)
    )
    await reopened.close()

    assert view is not None
    assert view.action.id == "c1"
    assert view.action.name == "echo"
    assert view.result.status == "success"
    assert view.committed_at.record_id == committed.record_id


@tool(name="accounted_echo")
def _accounted_echo(text: str) -> ToolResult:
    return ToolResult(
        output={"echo": text},
        usage=ModelUsage.from_mapping({"total_tokens": 13, "cost_usd": 0.0004}),
        added_tool_names=("late_skill_tool",),
    )


@pytest.mark.asyncio
async def test_tool_terminal_and_commit_round_trip_usage_and_added_names(
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
    await reopened.close()

    terminal = next(
        record
        for record in records
        if record.type is JournalRecordType.TOOL_TERMINAL
    )
    result_payload = terminal.payload["result"]
    assert result_payload["usage"]["total_tokens"] == 13
    assert result_payload["added_tool_names"] == ["late_skill_tool"]
    restored_result = ToolResult.from_dict(result_payload)
    assert restored_result.usage is not None
    assert restored_result.usage.total_tokens == 13
    assert restored_result.usage["cost_usd"] == 0.0004
    assert restored_result.added_tool_names == ("late_skill_tool",)

    committed = next(
        record
        for record in records
        if record.type is JournalRecordType.STEP_COMMITTED
    )
    messages = [message_from_dict(item) for item in committed.payload["messages"]]
    tool_message = next(
        item for item in messages if isinstance(item, ToolResultMessage)
    )
    assert tool_message.usage is not None
    assert tool_message.usage.total_tokens == 13
    assert tool_message.added_tool_names == ("late_skill_tool",)
