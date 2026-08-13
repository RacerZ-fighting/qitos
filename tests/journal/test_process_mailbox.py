"""Managed process completion delivery through the durable runtime mailbox."""

from __future__ import annotations

import asyncio
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from qitos import Action, AgentModule, Decision, Engine, StateSchema, ToolRegistry
from qitos.core.journal import JournalRecordType
from qitos.engine import RuntimeBudget
from qitos.kit.history import WindowHistory
from qitos.kit.journal import JsonlSessionJournal
from qitos.kit.tool.internal.coding_impl import CodingToolSet


def _python_command(source: str) -> str:
    return shlex.join([sys.executable, "-u", "-c", source])


@dataclass
class _State(StateSchema):
    pass


class _ProcessWaitAgent(AgentModule[_State, dict[str, Any], Action]):
    name = "process-mailbox"

    def __init__(self, tools: CodingToolSet) -> None:
        self.calls = 0
        self.process_id = ""
        self.waiting = asyncio.Event()
        self.seen_runtime_events: list[dict[str, Any]] = []
        self.history = WindowHistory(window_size=20)
        super().__init__(
            tool_registry=ToolRegistry().include_toolset(tools),
            history=self.history,
        )

    def init_state(self, task: str, **kwargs: Any) -> _State:
        _ = kwargs
        return _State(task=task, max_steps=5)

    def decide(
        self,
        state: _State,
        observation: dict[str, Any],
    ) -> Decision[Action]:
        _ = state
        self.calls += 1
        for message in self.history.messages:
            if message.role != "user" or message.metadata.get("source") != "runtime":
                continue
            for event in json.loads(str(message.content))["runtime_events"]:
                if event["event_id"] not in {
                    seen["event_id"] for seen in self.seen_runtime_events
                }:
                    self.seen_runtime_events.append(event)
        if self.calls == 1:
            return Decision.act(
                [
                    Action(
                        name="run_command",
                        args={
                            "command": _python_command(
                                "print('ready', flush=True); input(); "
                                "print('x' * 9000, flush=True)"
                            ),
                            "run_in_background": True,
                        },
                    )
                ]
            )
        if self.seen_runtime_events:
            return Decision.final("process observed")
        results = observation.get("action_results", [])
        if results:
            output = results[0].get("output", {})
            self.process_id = str(output.get("process_id") or "")
        self.waiting.set()
        return Decision.wait(meta={"runtime_wait": True})

    def reduce(
        self,
        state: _State,
        observation: dict[str, Any],
        decision: Decision[Action],
    ) -> _State:
        _ = observation, decision
        return state


@pytest.mark.asyncio
async def test_process_terminal_wakes_agent_after_durable_mailbox_post(
    tmp_path: Path,
) -> None:
    tools = CodingToolSet(
        workspace_root=str(tmp_path),
        profile="shell",
        auto_approve=True,
    )
    agent = _ProcessWaitAgent(tools)
    journal = JsonlSessionJournal(tmp_path / "journal")
    engine = Engine(
        agent,
        journal=journal,
        budget=RuntimeBudget(max_steps=5),
    )
    running = asyncio.create_task(engine.arun("wait for the background process"))
    await asyncio.wait_for(agent.waiting.wait(), timeout=2)

    assert agent.process_id
    await tools.process_write.execute(
        {"process_id": agent.process_id, "data": "continue\n"},
        runtime_context={"run_id": engine.active_run_id},
    )
    result = await asyncio.wait_for(running, timeout=3)

    assert result.state.final_result == "process observed"
    assert len(agent.seen_runtime_events) == 1
    event = agent.seen_runtime_events[0]
    assert event["event_id"] == f"{agent.process_id}:terminal"
    assert event["kind"] == "process.completed"
    assert event["correlation_id"] == agent.process_id
    assert event["payload"]["terminal"] is True
    assert event["payload"]["status"] == "exited"
    assert event["payload"]["output"]["notification_truncated"] is True

    records = await journal.replay()
    record_types = [record.type for record in records]
    terminal_index = record_types.index(JournalRecordType.PROCESS_TERMINAL)
    mailbox_index = record_types.index(JournalRecordType.RUNTIME_INPUT_POSTED)
    assert terminal_index < mailbox_index
    terminal_record = records[terminal_index]
    assert len(terminal_record.payload["output"]["content"]) > len(
        event["payload"]["output"]["content"]
    )
    assert any(
        record.type is JournalRecordType.MODEL_COMPLETED
        and record.payload["runtime_input_ids"]
        == [f"{agent.process_id}:terminal"]
        for record in records
    )
