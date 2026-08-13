from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
import threading
import time
from typing import Any, AsyncIterator

import pytest

from qitos import (
    Action,
    AgentModule,
    AgentRegistry,
    AgentSpec,
    Decision,
    Engine,
    StateSchema,
    ToolRegistry,
    tool,
)
from qitos.core.action import ActionExecutionPolicy
from qitos.core.agent_module import ActionResultContext
from qitos.core import JournalRecordType, ToolResult
from qitos.core.journal import JournalError
from qitos.checkpoint import InMemoryCheckpointStore
from qitos.kit.journal import JsonlSessionJournal
from qitos.kit.parser import ReActTextParser
from qitos.models import Model, ModelStreamChunk
from qitos.engine.critic import Critic


@dataclass
class JournalState(StateSchema):
    seen: list[str] = field(default_factory=list)


class JournalAgent(AgentModule[JournalState, dict[str, Any], Action]):
    def __init__(self) -> None:
        registry = ToolRegistry()
        self.executions = 0
        self.finalizer_context_sizes: list[int] = []

        @tool(name="inspect")
        def inspect() -> str:
            self.executions += 1
            return "raw"

        registry.register(inspect)
        super().__init__(tool_registry=registry)

    def init_state(self, task: str, **kwargs: Any) -> JournalState:
        _ = kwargs
        return JournalState(task=task, max_steps=4)

    def decide(
        self,
        state: JournalState,
        observation: dict[str, Any],
    ) -> Decision[Action]:
        _ = observation
        if state.current_step == 0:
            return Decision.act([Action("inspect", action_id="call-1")])
        return Decision.final("done")

    def finalize_action_result(
        self,
        state: JournalState,
        action: Action,
        result: ToolResult,
        *,
        step_id: int,
        context: ActionResultContext,
    ) -> ToolResult:
        _ = state, action, step_id
        self.finalizer_context_sizes.append(len(context.prior_results))
        finalized = ToolResult.from_value(result)
        finalized.metadata["evidence_id"] = "evidence-1"
        finalized.model_output = "canonical"
        return finalized

    def reduce_action_result(
        self,
        state: JournalState,
        action: Action,
        result: ToolResult,
        *,
        step_id: int,
    ) -> JournalState:
        _ = action, step_id
        state.seen.append(str(result.model_visible_output))
        return state

    def reduce(
        self,
        state: JournalState,
        observation: dict[str, Any],
        decision: Decision[Action],
    ) -> JournalState:
        _ = observation
        if decision.mode == "final":
            state.final_result = decision.final_answer
        return state


@pytest.mark.asyncio
async def test_engine_persists_one_finalized_terminal_result(tmp_path: Path) -> None:
    agent = JournalAgent()
    journal = JsonlSessionJournal(tmp_path)
    result = await Engine(agent=agent, journal=journal).arun("inspect")
    records = await journal.replay()
    terminal = next(
        record for record in records if record.type is JournalRecordType.TOOL_TERMINAL
    )
    history_result = result.records[0].action_results[0]

    assert agent.executions == 1
    assert terminal.payload["result"] == history_result.to_dict()
    assert terminal.payload["result"]["model_output"] == "canonical"
    assert terminal.payload["result"]["metadata"]["evidence_id"] == "evidence-1"
    assert result.state.seen == ["canonical"]
    assert any(record.type is JournalRecordType.STEP_COMMITTED for record in records)
    assert any(record.type is JournalRecordType.RUN_COMPLETED for record in records)


@pytest.mark.asyncio
async def test_terminal_run_resumes_without_model_or_tool_replay(tmp_path: Path) -> None:
    original_agent = JournalAgent()
    original_journal = JsonlSessionJournal(tmp_path)
    original = await Engine(agent=original_agent, journal=original_journal).arun("inspect")

    resumed_agent = JournalAgent()
    resumed = await Engine(
        agent=resumed_agent,
        journal=JsonlSessionJournal(tmp_path),
    ).aresume_from_journal(original.run_id)

    assert resumed.state.final_result == "done"
    assert resumed.state.seen == ["canonical"]
    assert resumed_agent.executions == 0


class FailingJournal(JsonlSessionJournal):
    def __init__(self, root: Path, *, fail_type: JournalRecordType) -> None:
        super().__init__(root)
        self._fail_type = fail_type

    async def append(
        self,
        record_type: JournalRecordType,
        payload: dict[str, Any],
        *,
        record_id: str,
    ):
        if record_type is self._fail_type:
            raise JournalError(f"failed {record_type.value}")
        return await super().append(record_type, payload, record_id=record_id)


class OneShotFailingJournal(JsonlSessionJournal):
    def __init__(
        self,
        root: Path,
        *,
        fail_type: JournalRecordType,
        record_id_suffix: str = "",
    ) -> None:
        super().__init__(root)
        self._fail_type = fail_type
        self._record_id_suffix = record_id_suffix
        self._failed = False

    async def append(
        self,
        record_type: JournalRecordType,
        payload: dict[str, Any],
        *,
        record_id: str,
    ):
        if (
            not self._failed
            and record_type is self._fail_type
            and record_id.endswith(self._record_id_suffix)
        ):
            self._failed = True
            raise JournalError(f"failed {record_type.value}")
        return await super().append(record_type, payload, record_id=record_id)


@pytest.mark.asyncio
async def test_tool_does_not_execute_when_started_record_fails(tmp_path: Path) -> None:
    agent = JournalAgent()
    journal = FailingJournal(tmp_path, fail_type=JournalRecordType.TOOL_STARTED)

    with pytest.raises(JournalError, match="tool.started"):
        await Engine(agent=agent, journal=journal).arun("inspect")

    assert agent.executions == 0


@pytest.mark.asyncio
async def test_resume_closes_tool_that_never_received_execution_permission(
    tmp_path: Path,
) -> None:
    original_agent = JournalAgent()
    failed = FailingJournal(tmp_path, fail_type=JournalRecordType.TOOL_STARTED)
    with pytest.raises(JournalError, match="tool.started"):
        await Engine(agent=original_agent, journal=failed).arun("inspect")

    resumed_agent = JournalAgent()
    resumed_journal = JsonlSessionJournal(tmp_path)
    resumed = await Engine(
        agent=resumed_agent,
        journal=resumed_journal,
    ).aresume_from_journal(failed.run_id)

    assert resumed_agent.executions == 0
    assert resumed.state.final_result == "done"
    terminal = next(
        record
        for record in await resumed_journal.replay()
        if record.type is JournalRecordType.TOOL_TERMINAL
    )
    assert terminal.payload["result"]["status"] == "cancelled"
    assert terminal.payload["result"]["metadata"]["side_effect"] == "none"


@pytest.mark.asyncio
async def test_terminal_append_failure_does_not_commit_reduced_state(
    tmp_path: Path,
) -> None:
    agent = JournalAgent()
    journal = FailingJournal(tmp_path, fail_type=JournalRecordType.TOOL_TERMINAL)
    engine = Engine(agent=agent, journal=journal)

    with pytest.raises(JournalError, match="tool.terminal"):
        await engine.arun("inspect")

    assert agent.executions == 1
    assert engine.current_state is not None
    assert engine.current_state.seen == []


@pytest.mark.asyncio
async def test_resume_closes_started_tool_without_replaying_unknown_side_effect(
    tmp_path: Path,
) -> None:
    original_agent = JournalAgent()
    failed = FailingJournal(tmp_path, fail_type=JournalRecordType.TOOL_TERMINAL)
    with pytest.raises(JournalError, match="tool.terminal"):
        await Engine(agent=original_agent, journal=failed).arun("inspect")
    run_id = failed.run_id

    resumed_agent = JournalAgent()
    resumed_journal = JsonlSessionJournal(tmp_path)
    resumed = await Engine(
        agent=resumed_agent,
        journal=resumed_journal,
    ).aresume_from_journal(run_id)

    assert resumed_agent.executions == 0
    assert resumed.state.final_result == "done"
    terminals = [
        record
        for record in await resumed_journal.replay()
        if record.type is JournalRecordType.TOOL_TERMINAL
    ]
    assert len(terminals) == 1
    assert terminals[0].payload["result"]["status"] == "error"
    assert terminals[0].payload["result"]["metadata"]["side_effect"] == "unknown"


def test_engine_rejects_journal_and_checkpoint_dual_writes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="journal and checkpoint_store"):
        Engine(
            agent=JournalAgent(),
            journal=JsonlSessionJournal(tmp_path),
            checkpoint_store=InMemoryCheckpointStore(),
        )


@pytest.mark.asyncio
async def test_resume_reduces_terminal_batch_that_was_not_committed(
    tmp_path: Path,
) -> None:
    original_agent = JournalAgent()
    failed = FailingJournal(tmp_path, fail_type=JournalRecordType.STEP_COMMITTED)
    engine = Engine(agent=original_agent, journal=failed)
    with pytest.raises(JournalError, match="step.committed"):
        await engine.arun("inspect")
    run_id = failed.run_id

    assert original_agent.executions == 1
    assert engine.current_state is not None
    assert engine.current_state.seen == []

    resumed_agent = JournalAgent()
    resumed_journal = JsonlSessionJournal(tmp_path)
    resumed = await Engine(
        agent=resumed_agent,
        journal=resumed_journal,
    ).aresume_from_journal(run_id)

    assert resumed_agent.executions == 0
    assert resumed.state.seen == ["canonical"]
    committed = [
        record
        for record in await resumed_journal.replay()
        if record.type is JournalRecordType.STEP_COMMITTED
    ]
    assert committed[0].payload["recovered"] is True


@pytest.mark.asyncio
async def test_periodic_snapshot_failure_keeps_the_committed_step(
    tmp_path: Path,
) -> None:
    original_agent = JournalAgent()
    journal = OneShotFailingJournal(
        tmp_path,
        fail_type=JournalRecordType.STATE_SNAPSHOT,
        record_id_suffix=":snapshot",
    )
    engine = Engine(
        agent=original_agent,
        journal=journal,
        state_snapshot_interval=1,
    )

    with pytest.raises(JournalError, match="state.snapshot"):
        await engine.arun("inspect")

    assert original_agent.executions == 1
    assert engine.current_state is not None
    assert engine.current_state.seen == ["canonical"]

    resumed_agent = JournalAgent()
    resumed = await Engine(
        agent=resumed_agent,
        journal=JsonlSessionJournal(tmp_path),
    ).aresume_from_journal(journal.run_id)

    assert resumed_agent.executions == 0
    assert resumed.state.seen == ["canonical"]
    assert resumed.state.final_result == "done"


@pytest.mark.asyncio
async def test_resume_settles_a_terminal_step_when_snapshot_failed(
    tmp_path: Path,
) -> None:
    journal = OneShotFailingJournal(
        tmp_path,
        fail_type=JournalRecordType.STATE_SNAPSHOT,
        record_id_suffix=":snapshot",
    )
    agent = JournalAgent()

    with pytest.raises(JournalError, match="state.snapshot"):
        await Engine(
            agent=agent,
            journal=journal,
            state_snapshot_interval=2,
        ).arun("inspect")

    assert agent.executions == 1
    resumed_agent = JournalAgent()
    resumed_journal = JsonlSessionJournal(tmp_path)
    resumed = await Engine(
        agent=resumed_agent,
        journal=resumed_journal,
    ).aresume_from_journal(journal.run_id)

    assert resumed_agent.executions == 0
    assert resumed.state.final_result == "done"
    types = [record.type for record in await resumed_journal.replay()]
    assert JournalRecordType.STATE_SNAPSHOT in types
    assert types[-1] is JournalRecordType.RUN_COMPLETED


@pytest.mark.asyncio
async def test_resume_settles_a_run_completed_append_failure(tmp_path: Path) -> None:
    journal = OneShotFailingJournal(
        tmp_path,
        fail_type=JournalRecordType.RUN_COMPLETED,
    )
    agent = JournalAgent()

    with pytest.raises(JournalError, match="run.completed"):
        await Engine(agent=agent, journal=journal).arun("inspect")

    resumed_agent = JournalAgent()
    resumed_journal = JsonlSessionJournal(tmp_path)
    resumed = await Engine(
        agent=resumed_agent,
        journal=resumed_journal,
    ).aresume_from_journal(journal.run_id)

    assert resumed_agent.executions == 0
    assert resumed.state.final_result == "done"
    assert (await resumed_journal.replay())[-1].type is JournalRecordType.RUN_COMPLETED


@pytest.mark.asyncio
async def test_cancel_appends_interruption_and_resume_uses_committed_state(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()

    class WaitingModel(Model):
        def __init__(self, *, wait: bool) -> None:
            super().__init__(model="journal-test")
            self._wait = wait

        async def stream(
            self,
            messages: list[dict[str, Any]],
            *,
            deadline_monotonic: float | None = None,
            **kwargs: Any,
        ) -> AsyncIterator[ModelStreamChunk]:
            _ = messages, deadline_monotonic, kwargs
            if self._wait:
                started.set()
                await asyncio.Event().wait()
            yield ModelStreamChunk(
                text="Final Answer: resumed",
                done=True,
                finish_reason="stop",
            )

    class WaitingAgent(JournalAgent):
        def __init__(self, *, wait: bool) -> None:
            super().__init__()
            self.llm = WaitingModel(wait=wait)
            self.model_parser = ReActTextParser()

        def decide(
            self,
            state: JournalState,
            observation: dict[str, Any],
        ) -> Decision[Action] | None:
            _ = state, observation
            return None

    journal = JsonlSessionJournal(tmp_path)
    engine = Engine(agent=WaitingAgent(wait=True), journal=journal)
    run_task = asyncio.create_task(engine.arun("wait"))
    await started.wait()

    engine.cancel()

    with pytest.raises(asyncio.CancelledError):
        await run_task
    records = await journal.replay()
    assert records[-1].type is JournalRecordType.RUN_INTERRUPTED
    assert not any(record.type is JournalRecordType.RUN_COMPLETED for record in records)

    resumed = await Engine(
        agent=WaitingAgent(wait=False),
        journal=JsonlSessionJournal(tmp_path),
    ).aresume_from_journal(journal.run_id)
    assert resumed.state.final_result == "resumed"


@pytest.mark.asyncio
async def test_fork_resumes_from_committed_boundary_independently(
    tmp_path: Path,
) -> None:
    original_journal = JsonlSessionJournal(tmp_path)
    original = await Engine(
        agent=JournalAgent(),
        journal=original_journal,
    ).arun("inspect")
    committed = next(
        record
        for record in await original_journal.replay()
        if record.type is JournalRecordType.STEP_COMMITTED
    )
    forked = await Engine(
        agent=JournalAgent(),
        journal=JsonlSessionJournal(tmp_path),
    ).afork_journal(
        original.run_id,
        committed.position,
        new_run_id="forked-run",
    )

    fork_agent = JournalAgent()
    resumed = await Engine(
        agent=fork_agent,
        journal=forked,
    ).aresume_from_journal("forked-run")

    assert fork_agent.executions == 0
    assert resumed.run_id == "forked-run"
    assert resumed.state.seen == ["canonical"]
    assert resumed.state.final_result == "done"


@pytest.mark.asyncio
async def test_journal_keeps_declared_safe_tools_parallel_and_results_ordered(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry()
    barrier = threading.Barrier(2)

    @tool(name="safe_a", concurrency_safe=True)
    def safe_a() -> str:
        barrier.wait(timeout=1)
        time.sleep(0.02)
        return "a"

    @tool(name="safe_b", concurrency_safe=True)
    def safe_b() -> str:
        barrier.wait(timeout=1)
        return "b"

    registry.register(safe_a)
    registry.register(safe_b)

    class ParallelAgent(JournalAgent):
        def __init__(self) -> None:
            super().__init__()
            self.tool_registry = registry

        def decide(
            self,
            state: JournalState,
            observation: dict[str, Any],
        ) -> Decision[Action]:
            _ = observation
            if state.current_step == 0:
                return Decision.act(
                    [
                        Action("safe_a", action_id="a"),
                        Action("safe_b", action_id="b"),
                    ]
                )
            return Decision.final("done")

    journal = JsonlSessionJournal(tmp_path)
    result = await Engine(
        agent=ParallelAgent(),
        journal=journal,
        action_execution_policy=ActionExecutionPolicy(mode="parallel"),
    ).arun("parallel")

    assert result.records[0].action_execution["concurrency_peak"] == 2
    assert [item.output for item in result.records[0].action_results] == ["a", "b"]
    terminal_actions = [
        record.payload["action"]["name"]
        for record in await journal.replay()
        if record.type is JournalRecordType.TOOL_TERMINAL
    ]
    assert terminal_actions == ["safe_a", "safe_b"]
    assert result.state.seen == ["canonical", "canonical"]


@pytest.mark.asyncio
async def test_action_finalizers_see_only_prior_durable_results(tmp_path: Path) -> None:
    agent = JournalAgent()

    def decide(
        state: JournalState,
        observation: dict[str, Any],
    ) -> Decision[Action]:
        _ = observation
        if state.current_step == 0:
            return Decision.act(
                [
                    Action("inspect", action_id="first"),
                    Action("inspect", action_id="second"),
                ]
            )
        return Decision.final("done")

    agent.decide = decide  # type: ignore[method-assign]
    await Engine(
        agent=agent,
        journal=JsonlSessionJournal(tmp_path),
    ).arun("inspect")

    assert agent.finalizer_context_sizes == [0, 1]


@pytest.mark.asyncio
async def test_journal_commits_critic_retry_before_the_next_step(tmp_path: Path) -> None:
    class RetryOnce(Critic):
        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, state, decision, results):
            _ = state, decision, results
            self.calls += 1
            return {"action": "retry" if self.calls == 1 else "continue"}

    journal = JsonlSessionJournal(tmp_path)
    result = await Engine(
        agent=JournalAgent(),
        journal=journal,
        critics=[RetryOnce()],
    ).arun("inspect")

    committed = [
        record
        for record in await journal.replay()
        if record.type is JournalRecordType.STEP_COMMITTED
    ]
    assert len(committed) == result.step_count
    assert result.state.seen == ["canonical"]


@pytest.mark.asyncio
async def test_journal_commits_handoff_state_before_receiver_runs(
    tmp_path: Path,
) -> None:
    class HandoffAgent(JournalAgent):
        name = "handoffer"

        def decide(self, state, observation):
            _ = state, observation
            return Decision.handoff("receiver")

    class ReceiverAgent(JournalAgent):
        name = "receiver"

        def decide(self, state, observation):
            _ = state, observation
            return Decision.final("received")

    registry = AgentRegistry()
    receiver = ReceiverAgent()
    registry.register(AgentSpec(name="receiver", description="", agent=receiver))
    journal = JsonlSessionJournal(tmp_path)
    result = await Engine(
        agent=HandoffAgent(),
        agent_registry=registry,
        journal=journal,
    ).arun("handoff")

    assert result.state.final_result == "received"
    assert result.state.metadata["last_handoff"] == {
        "from": "handoffer",
        "to": "receiver",
    }
    committed = [
        record
        for record in await journal.replay()
        if record.type is JournalRecordType.STEP_COMMITTED
    ]
    assert len(committed) == result.step_count
