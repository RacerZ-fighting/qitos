"""Durability tests for the run-scoped asynchronous runtime mailbox."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from qitos import AgentModule, Decision, Engine, RuntimeInput, StateSchema, ToolRegistry
from qitos.core.journal import JournalError, JournalRecordType
from qitos.engine import RuntimeBudget
from qitos.kit.history import WindowHistory
from qitos.kit.journal import JsonlSessionJournal


@dataclass
class _State(StateSchema):
    pass


class _WaitThenFinish(AgentModule[_State, dict[str, Any], Any]):
    name = "mailbox-wait"

    def __init__(self, *, finish_immediately: bool = False) -> None:
        self.calls = 0
        self.first_decision = asyncio.Event()
        self.seen_runtime_events: list[dict[str, Any]] = []
        self.history = WindowHistory(window_size=20)
        self._finish_immediately = finish_immediately
        super().__init__(tool_registry=ToolRegistry(), history=self.history)

    def init_state(self, task: str, **kwargs: Any) -> _State:
        _ = kwargs
        return _State(task=task, max_steps=4)

    def decide(
        self,
        state: _State,
        observation: dict[str, Any],
    ) -> Decision[Any]:
        _ = state, observation
        self.calls += 1
        for message in self.history.messages:
            if message.role != "user" or message.metadata.get("source") != "runtime":
                continue
            for event in json.loads(str(message.content))["runtime_events"]:
                if event["event_id"] not in {
                    seen["event_id"] for seen in self.seen_runtime_events
                }:
                    self.seen_runtime_events.append(event)
        if self._finish_immediately or self.calls > 1:
            return Decision.final("done")
        self.first_decision.set()
        return Decision.wait(meta={"runtime_wait": True})

    def reduce(
        self,
        state: _State,
        observation: dict[str, Any],
        decision: Decision[Any],
    ) -> _State:
        _ = observation, decision
        return state


class _FailRuntimePostJournal(JsonlSessionJournal):
    async def append(
        self,
        record_type: JournalRecordType,
        payload: dict[str, Any],
        *,
        record_id: str,
    ):
        if record_type is JournalRecordType.RUNTIME_INPUT_POSTED:
            raise JournalError("injected runtime input append failure")
        return await super().append(record_type, payload, record_id=record_id)


class _FailSecondModelJournal(JsonlSessionJournal):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self._model_completions = 0

    async def append(
        self,
        record_type: JournalRecordType,
        payload: dict[str, Any],
        *,
        record_id: str,
    ):
        if record_type is JournalRecordType.MODEL_COMPLETED:
            self._model_completions += 1
            if self._model_completions == 2:
                raise JournalError("injected second model completion failure")
        return await super().append(record_type, payload, record_id=record_id)


class _FinalRaceAgent(_WaitThenFinish):
    name = "mailbox-final-race"

    def __init__(self) -> None:
        super().__init__()
        self.engine: Engine[Any, Any, Any] | None = None
        self.late_post: asyncio.Task[bool] | None = None

    def decide(
        self,
        state: _State,
        observation: dict[str, Any],
    ) -> Decision[Any]:
        decision = super().decide(state, observation)
        if self.calls == 2:
            assert self.engine is not None
            self.late_post = asyncio.create_task(
                self.engine.apost_runtime_event(
                    _event("job-2:terminal"),
                    run_id=self.engine.active_run_id,
                )
            )
        return decision


def _event(event_id: str = "job-1:terminal") -> RuntimeInput:
    return RuntimeInput(
        event_id=event_id,
        kind="job.terminal",
        correlation_id="job-1",
        source="test",
        payload={"status": "completed"},
    )


@pytest.mark.asyncio
async def test_runtime_input_is_durable_before_safe_point_delivery(
    tmp_path: Path,
) -> None:
    journal = JsonlSessionJournal(tmp_path)
    agent = _WaitThenFinish()
    engine = Engine(
        agent,
        journal=journal,
        budget=RuntimeBudget(max_steps=4),
    )
    running = asyncio.create_task(engine.arun("wait"))
    await asyncio.wait_for(agent.first_decision.wait(), timeout=1)
    run_id = engine.active_run_id

    assert await engine.apost_runtime_event(_event(), run_id=run_id) is True
    assert await engine.apost_runtime_event(_event(), run_id=run_id) is False
    result = await asyncio.wait_for(running, timeout=1)

    assert result.state.final_result == "done"
    assert [event["event_id"] for event in agent.seen_runtime_events] == [
        "job-1:terminal"
    ]
    records = await journal.replay()
    post_index = next(
        index
        for index, record in enumerate(records)
        if record.type is JournalRecordType.RUNTIME_INPUT_POSTED
    )
    consuming_model_index = next(
        index
        for index, record in enumerate(records)
        if record.type is JournalRecordType.MODEL_COMPLETED
        and record.payload["runtime_input_ids"] == ["job-1:terminal"]
    )
    assert post_index < consuming_model_index


@pytest.mark.asyncio
async def test_failed_runtime_input_append_never_wakes_the_run(tmp_path: Path) -> None:
    journal = _FailRuntimePostJournal(tmp_path)
    agent = _WaitThenFinish()
    engine = Engine(
        agent,
        journal=journal,
        budget=RuntimeBudget(max_steps=4),
    )
    running = asyncio.create_task(engine.arun("wait"))
    await asyncio.wait_for(agent.first_decision.wait(), timeout=1)

    with pytest.raises(JournalError, match="runtime input append failure"):
        await engine.apost_runtime_event(_event(), run_id=engine.active_run_id)

    engine.cancel("immediate")
    result = await asyncio.wait_for(running, timeout=1)
    assert result.state.stop_reason == "cancelled_immediate"
    assert agent.calls == 1
    assert not any(
        record.type is JournalRecordType.RUNTIME_INPUT_POSTED
        for record in await journal.replay()
    )


@pytest.mark.asyncio
async def test_input_accepted_during_final_turn_defers_completion(
    tmp_path: Path,
) -> None:
    journal = JsonlSessionJournal(tmp_path)
    agent = _FinalRaceAgent()
    engine = Engine(
        agent,
        journal=journal,
        budget=RuntimeBudget(max_steps=4),
    )
    agent.engine = engine
    running = asyncio.create_task(engine.arun("wait"))
    await asyncio.wait_for(agent.first_decision.wait(), timeout=1)

    assert await engine.apost_runtime_event(
        _event("job-1:terminal"),
        run_id=engine.active_run_id,
    )
    result = await asyncio.wait_for(running, timeout=1)

    assert agent.late_post is not None
    assert await agent.late_post is True
    assert result.state.final_result == "done"
    assert agent.calls == 3
    assert [event["event_id"] for event in agent.seen_runtime_events] == [
        "job-1:terminal",
        "job-2:terminal",
    ]


@pytest.mark.asyncio
async def test_resume_redelivers_only_an_unconsumed_runtime_input(
    tmp_path: Path,
) -> None:
    failing_journal = _FailSecondModelJournal(tmp_path)
    first_agent = _WaitThenFinish()
    first_engine = Engine(
        first_agent,
        journal=failing_journal,
        budget=RuntimeBudget(max_steps=4),
    )
    running = asyncio.create_task(first_engine.arun("wait"))
    await asyncio.wait_for(first_agent.first_decision.wait(), timeout=1)
    run_id = first_engine.active_run_id
    assert await first_engine.apost_runtime_event(_event(), run_id=run_id) is True

    with pytest.raises(JournalError, match="second model completion failure"):
        await running

    resumed_journal = JsonlSessionJournal(tmp_path)
    resumed_agent = _WaitThenFinish(finish_immediately=True)
    resumed = Engine(
        resumed_agent,
        journal=resumed_journal,
        budget=RuntimeBudget(max_steps=4),
    )
    result = await resumed.aresume_from_journal(run_id)

    assert result.state.final_result == "done"
    assert [event["event_id"] for event in resumed_agent.seen_runtime_events] == [
        "job-1:terminal"
    ]
    records = await resumed_journal.replay()
    assert sum(
        record.type is JournalRecordType.RUNTIME_INPUT_POSTED for record in records
    ) == 1
    assert sum(
        record.type is JournalRecordType.MODEL_COMPLETED
        and record.payload["runtime_input_ids"] == ["job-1:terminal"]
        for record in records
    ) == 1
