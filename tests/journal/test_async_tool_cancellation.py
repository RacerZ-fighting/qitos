from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from qitos import Action, AgentModule, Decision, Engine, StateSchema, ToolRegistry
from qitos.core.journal import JournalRecordType
from qitos.core.tool import BaseTool, ToolSpec
from qitos.kit.journal import JsonlSessionJournal


class _WaitingTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(ToolSpec(name="wait", description="wait until cancelled"))
        self.started = asyncio.Event()
        self.cleaned = asyncio.Event()

    async def execute(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> str:
        _ = args, runtime_context
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.cleaned.set()
        return "unreachable"


@dataclass
class _State(StateSchema):
    pass


class _Agent(AgentModule[_State, dict[str, Any], Action]):
    def __init__(self, waiting: _WaitingTool) -> None:
        super().__init__(tool_registry=ToolRegistry().register(waiting))

    def init_state(self, task: str, **kwargs: Any) -> _State:
        _ = kwargs
        return _State(task=task, max_steps=2)

    def decide(
        self,
        state: _State,
        observation: dict[str, Any],
    ) -> Decision[Action]:
        _ = state, observation
        return Decision.act([Action(name="wait", action_id="call-wait")])

    def reduce(
        self,
        state: _State,
        observation: dict[str, Any],
        decision: Decision[Action],
    ) -> _State:
        _ = observation, decision
        return state


@pytest.mark.asyncio
async def test_caller_cancellation_drains_tool_and_journals_terminal(
    tmp_path,
) -> None:
    waiting = _WaitingTool()
    journal = JsonlSessionJournal(tmp_path)
    run_task = asyncio.create_task(Engine(_Agent(waiting), journal=journal).arun("wait"))

    await asyncio.wait_for(waiting.started.wait(), timeout=1)
    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task

    assert waiting.cleaned.is_set()
    records = await journal.replay()
    types = [record.type for record in records]
    assert JournalRecordType.TOOL_STARTED in types
    assert JournalRecordType.TOOL_TERMINAL in types
    assert JournalRecordType.STEP_COMMITTED not in types
    assert types[-1] is JournalRecordType.RUN_INTERRUPTED
    terminal = next(
        record for record in records if record.type is JournalRecordType.TOOL_TERMINAL
    )
    assert terminal.payload["result"]["status"] == "cancelled"
    pending = [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and task.get_name().startswith(("qitos-tool", "qitos-sync-tool"))
    ]
    assert pending == []
