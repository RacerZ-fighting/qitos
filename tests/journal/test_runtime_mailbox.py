"""Durability tests for the run-scoped asynchronous runtime mailbox."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from qitos import AgentModule, Decision, Engine, RuntimeInput, StateSchema, ToolRegistry
from qitos.core.child import ChildHandle, ChildLaunchRequest, ChildResult, ChildStatus
from qitos.core.journal import JournalError, JournalRecord, JournalRecordType
from qitos.core.process import (
    ProcessHandle,
    ProcessOutput,
    ProcessSnapshot,
    ProcessStatus,
)
from qitos.core.tool import BaseTool, ToolSpec
from qitos.engine._journal_runtime import recover_pending_runtime_inputs
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


class _ChildTerminalOnSetup(BaseTool):
    def __init__(self, result: ChildResult) -> None:
        super().__init__(
            ToolSpec(
                name="terminal_on_setup",
                description="Append one recovered child terminal during setup.",
            )
        )
        self._result = result

    async def asetup(self, context: dict[str, Any]) -> None:
        journal = context["journal"]
        await journal.append(
            JournalRecordType.CHILD_STARTED,
            {
                "handle": self._result.handle.to_dict(),
                "request": self._result.request.to_dict(),
                "background": True,
            },
            record_id=(
                f"{self._result.handle.parent_run_id}:child:"
                f"{self._result.handle.child_id}:started"
            ),
        )
        await journal.append(
            JournalRecordType.CHILD_TERMINAL,
            self._result.to_dict(),
            record_id=(
                f"{self._result.handle.parent_run_id}:child:"
                f"{self._result.handle.child_id}:terminal"
            ),
        )

    async def execute(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = args, runtime_context
        return {}


def _event(event_id: str = "job-1:terminal") -> RuntimeInput:
    return RuntimeInput(
        event_id=event_id,
        kind="job.terminal",
        correlation_id="job-1",
        source="test",
        payload={"status": "completed"},
    )


def _terminal_record(
    seq: int,
    record_type: JournalRecordType,
    payload: dict[str, Any],
) -> JournalRecord:
    return JournalRecord.create(
        seq=seq,
        record_id=f"terminal-{seq}",
        type=record_type,
        run_id="run-1",
        payload=payload,
    )


def _child_started_record(
    seq: int,
    result: ChildResult,
    *,
    background: bool,
) -> JournalRecord:
    return JournalRecord.create(
        seq=seq,
        record_id=f"child-started-{seq}",
        type=JournalRecordType.CHILD_STARTED,
        run_id="run-1",
        payload={
            "handle": result.handle.to_dict(),
            "request": result.request.to_dict(),
            "background": background,
        },
    )


def test_terminal_facts_rebuild_pending_child_and_process_inputs() -> None:
    child = ChildResult(
        handle=ChildHandle("child-1", "run-1"),
        request=ChildLaunchRequest(task="inspect", description="inspect target"),
        status=ChildStatus.COMPLETED,
        child_run_id="child-run-1",
    )
    process = ProcessSnapshot(
        handle=ProcessHandle("process-1", "run-1"),
        status=ProcessStatus.EXITED,
        command="probe",
        cwd="/workspace",
        pid=1,
        tty=False,
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T00:00:01+00:00",
        exit_code=0,
        output=ProcessOutput(
            content="x" * 9_000,
            cursor=0,
            next_cursor=9_000,
            total_bytes=9_000,
            omitted_bytes=0,
            truncated=False,
            log_path="processes/process-1.log",
        ),
    )

    pending = recover_pending_runtime_inputs(
        (
            _child_started_record(1, child, background=True),
            _terminal_record(2, JournalRecordType.CHILD_TERMINAL, child.to_dict()),
            _terminal_record(3, JournalRecordType.PROCESS_TERMINAL, process.to_dict()),
        )
    )

    assert [(event.event_id, event.kind) for event in pending] == [
        ("child-1:terminal", "agent.child.completed"),
        ("process-1:terminal", "process.completed"),
    ]
    assert pending[1].payload["output"]["notification_truncated"] is True


def test_foreground_child_terminal_does_not_become_runtime_input() -> None:
    child = ChildResult(
        handle=ChildHandle("child-foreground", "run-1"),
        request=ChildLaunchRequest(task="inspect", description="inspect target"),
        status=ChildStatus.COMPLETED,
    )

    pending = recover_pending_runtime_inputs(
        (
            _child_started_record(1, child, background=False),
            _terminal_record(2, JournalRecordType.CHILD_TERMINAL, child.to_dict()),
        )
    )

    assert pending == ()
    assert recover_pending_runtime_inputs(
        (_terminal_record(1, JournalRecordType.CHILD_TERMINAL, child.to_dict()),)
    ) == ()


def test_terminal_derived_inputs_are_idempotent_consumable_and_not_forked() -> None:
    child = ChildResult(
        handle=ChildHandle("child-1", "run-1"),
        request=ChildLaunchRequest(task="inspect", description="inspect target"),
        status=ChildStatus.COMPLETED,
    )
    terminal = _terminal_record(
        2,
        JournalRecordType.CHILD_TERMINAL,
        child.to_dict(),
    )
    started = _child_started_record(1, child, background=True)
    event = recover_pending_runtime_inputs((started, terminal))[0]
    posted = JournalRecord.create(
        seq=3,
        record_id="posted",
        type=JournalRecordType.RUNTIME_INPUT_POSTED,
        run_id="run-1",
        payload={"event": event.to_dict()},
    )
    consumed = JournalRecord.create(
        seq=4,
        record_id="model",
        type=JournalRecordType.MODEL_COMPLETED,
        run_id="run-1",
        payload={"runtime_input_ids": [event.event_id]},
    )
    inherited = JournalRecord.create(
        seq=1,
        record_id="inherited",
        type=JournalRecordType.INHERITED,
        run_id="fork-1",
        payload={
            "origin_run_id": terminal.run_id,
            "origin_seq": terminal.seq,
            "origin_record_id": terminal.record_id,
            "record": terminal.to_dict(),
        },
    )

    assert recover_pending_runtime_inputs((started, terminal, posted)) == (event,)
    assert recover_pending_runtime_inputs((terminal, posted)) == (event,)
    assert recover_pending_runtime_inputs(
        (started, terminal, posted, consumed)
    ) == ()
    assert recover_pending_runtime_inputs((inherited,)) == ()


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


@pytest.mark.asyncio
async def test_resume_replays_terminal_created_during_tool_setup(
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

    child = ChildResult(
        handle=ChildHandle("child-recovered", run_id),
        request=ChildLaunchRequest(task="inspect", description="inspect target"),
        status=ChildStatus.INTERRUPTED,
        error="parent process exited",
    )
    resumed_agent = _WaitThenFinish(finish_immediately=True)
    resumed_agent.tool_registry.register(_ChildTerminalOnSetup(child))
    journal = JsonlSessionJournal(tmp_path)

    result = await Engine(
        resumed_agent,
        journal=journal,
        budget=RuntimeBudget(max_steps=4),
    ).aresume_from_journal(run_id)

    assert result.state.final_result == "done"
    child_events = [
        event
        for event in resumed_agent.seen_runtime_events
        if event["event_id"] == "child-recovered:terminal"
    ]
    assert len(child_events) == 1
    records = await journal.replay()
    assert not any(
        record.type is JournalRecordType.RUNTIME_INPUT_POSTED
        and record.payload["event"]["event_id"] == "child-recovered:terminal"
        for record in records
    )
    assert any(
        record.type is JournalRecordType.MODEL_COMPLETED
        and "child-recovered:terminal" in record.payload["runtime_input_ids"]
        for record in records
    )
