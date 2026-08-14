from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
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
    Env,
    StateSchema,
    ToolRegistry,
    tool,
)
from qitos.core.action import ActionExecutionPolicy
from qitos.core.agent_module import ActionResultContext
from qitos.core import (
    JournalRecordType,
    ModelAPI,
    ModelCapabilities,
    ModelContinuation,
    ModelPricing,
    ModelUsage,
    ToolResult,
)
from qitos.core.journal import JournalError, JournalRecordRef
from qitos.core.env import EnvObservation, EnvStepResult
from qitos.core.tool import BaseTool, ToolSpec
from qitos.checkpoint import InMemoryCheckpointStore
from qitos.kit.journal import JsonlRunCatalog, JsonlSessionJournal
from qitos.kit.parser import ReActTextParser
from qitos.models import Model, ModelRequest, ModelStreamEvent, ModelStreamEventType
from qitos.engine.critic import Critic
from qitos.engine.states import RuntimeBudget


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
        finalized.call_id = "finalizer-cannot-rewrite-identity"
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


def _tamper_journal_payload(
    path: Path,
    record_type: JournalRecordType,
    keys: tuple[str | int, ...],
    value: Any,
) -> None:
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    for record in records:
        if record["type"] != record_type.value:
            continue
        target = record["payload"]
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
        break
    else:
        raise AssertionError(f"journal has no {record_type.value} record")
    path.write_text(
        "".join(
            f"{json.dumps(record, ensure_ascii=False, sort_keys=True)}\n"
            for record in records
        ),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_each_terminal_is_reduced_before_the_next_finalizer(
    tmp_path: Path,
) -> None:
    class OrderedAgent(JournalAgent):
        def __init__(self) -> None:
            super().__init__()
            self.finalizer_seen: list[tuple[str, ...]] = []
            self.finalizer_terminals: list[tuple[JournalRecordRef | None, ...]] = []

        def decide(
            self,
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

        def finalize_action_result(
            self,
            state: JournalState,
            action: Action,
            result: ToolResult,
            *,
            step_id: int,
            context: ActionResultContext,
        ) -> ToolResult:
            _ = action, result, step_id, context
            self.finalizer_seen.append(tuple(state.seen))
            self.finalizer_terminals.append(
                tuple(item.terminal for item in context.prior_results)
            )
            return ToolResult(output=f"canonical-{len(self.finalizer_seen)}")

    agent = OrderedAgent()

    result = await Engine(
        agent=agent,
        journal=JsonlSessionJournal(tmp_path),
    ).arun("inspect")

    assert agent.finalizer_seen == [(), ("canonical-1",)]
    assert agent.finalizer_terminals[0] == ()
    assert len(agent.finalizer_terminals[1]) == 1
    first_terminal = agent.finalizer_terminals[1][0]
    assert first_terminal is not None
    assert first_terminal.run_id == result.run_id
    assert first_terminal.record_id.endswith(":tool:0:terminal")
    assert result.state.seen == ["canonical-1", "canonical-2"]


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

    assert journal.closed is True
    assert agent.executions == 1
    assert terminal.payload["result"] == history_result.to_dict()
    assert terminal.payload["result"]["call_id"] == "call-1"
    assert history_result.call_id == "call-1"
    assert terminal.payload["result"]["model_output"] == "canonical"
    assert terminal.payload["result"]["metadata"]["evidence_id"] == "evidence-1"
    assert result.state.seen == ["canonical"]
    assert any(record.type is JournalRecordType.STEP_COMMITTED for record in records)
    assert any(record.type is JournalRecordType.RUN_COMPLETED for record in records)


@pytest.mark.asyncio
async def test_invalid_tool_arguments_commit_unexecuted_terminal(
    tmp_path: Path,
) -> None:
    class InvalidArgumentsAgent(JournalAgent):
        def decide(
            self,
            state: JournalState,
            observation: dict[str, Any],
        ) -> Decision[Action]:
            _ = observation
            if state.current_step == 0:
                return Decision.act(
                    [
                        Action(
                            "inspect",
                            args={"unknown": True},
                            action_id="invalid-call",
                        )
                    ]
                )
            return Decision.final("done")

    agent = InvalidArgumentsAgent()
    journal = JsonlSessionJournal(tmp_path)

    result = await Engine(agent=agent, journal=journal).arun("inspect")
    records = await journal.replay()
    terminal = next(
        record for record in records if record.type is JournalRecordType.TOOL_TERMINAL
    )

    assert agent.executions == 0
    assert terminal.payload["result"]["status"] == "error"
    assert terminal.payload["result"]["metadata"]["executed"] is False
    assert (
        terminal.payload["result"]["metadata"]["error_category"]
        == "invalid_tool_arguments"
    )
    assert result.records[0].action_results[0].status == "error"
    assert any(record.type is JournalRecordType.STEP_COMMITTED for record in records)


@pytest.mark.asyncio
async def test_terminal_run_resumes_without_model_or_tool_replay(
    tmp_path: Path,
) -> None:
    original_agent = JournalAgent()
    original_journal = JsonlSessionJournal(tmp_path)
    original = await Engine(agent=original_agent, journal=original_journal).arun(
        "inspect"
    )

    resumed_agent = JournalAgent()
    resumed = await Engine(
        agent=resumed_agent,
        journal=JsonlSessionJournal(tmp_path),
    ).aresume_from_journal(original.run_id)

    assert resumed.state.final_result == "done"
    assert resumed.state.seen == ["canonical"]
    assert resumed_agent.executions == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("record_type", "keys", "value", "error"),
    [
        (
            JournalRecordType.TOOL_TERMINAL,
            ("action", "name"),
            "tampered",
            "tool.terminal action does not match model decision",
        ),
        (
            JournalRecordType.TOOL_TERMINAL,
            ("action_index",),
            99,
            "tool.terminal action_index is out of range",
        ),
        (
            JournalRecordType.STEP_COMMITTED,
            ("terminal_record_ids",),
            ["wrong-terminal"],
            "step.committed terminal_record_ids do not match its actions",
        ),
        (
            JournalRecordType.MODEL_COMPLETED,
            ("decision", "actions", 0),
            "not-an-action",
            "decision actions are invalid",
        ),
        (
            JournalRecordType.STEP_COMMITTED,
            ("history_append",),
            ["not-a-message"],
            "history_append entries must be objects",
        ),
    ],
)
async def test_resume_rejects_cross_record_transaction_corruption(
    tmp_path: Path,
    record_type: JournalRecordType,
    keys: tuple[str | int, ...],
    value: Any,
    error: str,
) -> None:
    journal = JsonlSessionJournal(tmp_path)
    original = await Engine(agent=JournalAgent(), journal=journal).arun("inspect")
    _tamper_journal_payload(journal.path, record_type, keys, value)

    with pytest.raises(JournalError, match=error):
        await Engine(
            agent=JournalAgent(),
            journal=JsonlSessionJournal(tmp_path),
        ).aresume_from_journal(original.run_id)


@pytest.mark.asyncio
async def test_resume_rejects_tool_terminals_persisted_out_of_order(
    tmp_path: Path,
) -> None:
    class TwoActionAgent(JournalAgent):
        def decide(
            self,
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

    journal = JsonlSessionJournal(tmp_path)
    original = await Engine(agent=TwoActionAgent(), journal=journal).arun("inspect")
    records = [
        json.loads(line)
        for line in journal.path.read_text(encoding="utf-8").splitlines()
    ]
    terminal_indexes = [
        index
        for index, record in enumerate(records)
        if record["type"] == JournalRecordType.TOOL_TERMINAL.value
    ]
    assert len(terminal_indexes) == 2
    first = records[terminal_indexes[0]]
    second = records[terminal_indexes[1]]
    first["record_id"], second["record_id"] = second["record_id"], first["record_id"]
    first["payload"], second["payload"] = second["payload"], first["payload"]
    journal.path.write_text(
        "".join(
            f"{json.dumps(record, ensure_ascii=False, sort_keys=True)}\n"
            for record in records
        ),
        encoding="utf-8",
    )

    with pytest.raises(JournalError, match="tool.terminal actions are out of order"):
        await Engine(
            agent=TwoActionAgent(),
            journal=JsonlSessionJournal(tmp_path),
        ).aresume_from_journal(original.run_id)


@pytest.mark.asyncio
async def test_terminal_resume_restores_model_usage_and_cost(tmp_path: Path) -> None:
    class UsageModel(Model):
        def __init__(self) -> None:
            super().__init__(model="usage-model", temperature=None)
            self.calls = 0

        async def stream(
            self,
            request: ModelRequest,
        ) -> AsyncIterator[ModelStreamEvent]:
            _ = request
            self.calls += 1
            yield ModelStreamEvent(
                text="finished",
                type=ModelStreamEventType.COMPLETED,
                usage=ModelUsage(
                    input_tokens=7,
                    output_tokens=3,
                    total_tokens=10,
                ),
            )

    class UsageAgent(AgentModule[JournalState, dict[str, Any], Action]):
        def __init__(self, model: UsageModel) -> None:
            super().__init__(llm=model)

        def init_state(self, task: str, **kwargs: Any) -> JournalState:
            _ = kwargs
            return JournalState(task=task, max_steps=2)

        def interpret_model_response(
            self,
            state: JournalState,
            observation: dict[str, Any],
            response: Any,
        ) -> Decision[Action]:
            _ = state, observation, response
            return Decision.final("done")

        def reduce(
            self,
            state: JournalState,
            observation: dict[str, Any],
            decision: Decision[Action],
        ) -> JournalState:
            _ = observation, decision
            return state

    pricing = ModelPricing(1_000_000, 1_000_000)
    original_model = UsageModel()
    original = await Engine(
        UsageAgent(original_model),
        budget=RuntimeBudget(max_steps=2, max_cost_usd=100.0),
        model_pricing=pricing,
        journal=JsonlSessionJournal(tmp_path),
    ).arun("usage")
    reader = JsonlSessionJournal(tmp_path)
    await reader.open(original.run_id)
    original_records = await reader.replay()
    await reader.close()

    resumed_model = UsageModel()
    resumed_engine = Engine(
        UsageAgent(resumed_model),
        budget=RuntimeBudget(max_steps=2, max_cost_usd=100.0),
        model_pricing=pricing,
        journal=JsonlSessionJournal(tmp_path),
    )
    resumed = await resumed_engine.aresume_from_journal(original.run_id)

    assert original.total_tokens == 10
    assert original.total_cost_usd == pytest.approx(10.0)
    budget_commit = next(
        record
        for record in original_records
        if record.type is JournalRecordType.BUDGET_COMMITTED
    )
    assert budget_commit.payload["tokens"] == original.total_tokens
    assert budget_commit.payload["cost_usd"] == pytest.approx(original.total_cost_usd)
    assert resumed.total_tokens == original.total_tokens
    assert resumed.total_cost_usd == pytest.approx(original.total_cost_usd)
    assert resumed_engine._token_usage == original.total_tokens
    assert resumed_engine._cost_usage_usd == pytest.approx(original.total_cost_usd)
    assert resumed_engine.budget_ledger.snapshot().total_tokens == original.total_tokens
    assert resumed_model.calls == 0


@pytest.mark.asyncio
async def test_resume_reuses_only_the_last_committed_model_continuation(
    tmp_path: Path,
) -> None:
    second_request_started = asyncio.Event()

    class ContinuationModel(Model):
        def __init__(self, *, block_continuation: bool) -> None:
            super().__init__(model="continuation-model", temperature=None)
            self.block_continuation = block_continuation
            self.requests: list[ModelRequest] = []
            self.qitos_harness_metadata = {
                "tool_policy": {"native_tool_call_preferred": True},
                "parser": "ReActTextParser",
                "protocol": "react_text_v1",
            }

        @property
        def capabilities(self) -> ModelCapabilities:
            return ModelCapabilities(
                api=ModelAPI.RESPONSES,
                native_tool_calls=True,
                continuation=True,
            )

        async def stream(
            self,
            request: ModelRequest,
        ) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if request.continuation is not None:
                if self.block_continuation:
                    second_request_started.set()
                    await asyncio.Event().wait()
                yield ModelStreamEvent(
                    text="Final Answer: resumed",
                    type=ModelStreamEventType.COMPLETED,
                    finish_reason="stop",
                )
                return
            yield ModelStreamEvent(
                type=ModelStreamEventType.COMPLETED,
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "inspect", "arguments": "{}"},
                    }
                ],
                finish_reason="tool_calls",
                continuation=ModelContinuation(
                    run_id=request.run_id,
                    provider=request.provider,
                    model=request.model,
                    protocol=request.protocol,
                    response_id="resp-1",
                    prefix_items=1,
                    prefix_digest="prefix",
                    settings_digest="settings",
                ),
            )

    class ContinuationAgent(JournalAgent):
        def __init__(self, model: ContinuationModel) -> None:
            super().__init__()
            self.llm = model
            self.model_parser = ReActTextParser()

        def decide(
            self,
            state: JournalState,
            observation: dict[str, Any],
        ) -> Decision[Action] | None:
            _ = state, observation
            return None

    original_model = ContinuationModel(block_continuation=True)
    journal = JsonlSessionJournal(tmp_path)
    engine = Engine(ContinuationAgent(original_model), journal=journal)
    running = asyncio.create_task(engine.arun("inspect"))
    await second_request_started.wait()
    engine.cancel()
    cancelled = await running

    assert cancelled.state.stop_reason == "cancelled_immediate"
    model_records = [
        record
        for record in await journal.replay()
        if record.type is JournalRecordType.MODEL_COMPLETED
    ]
    assert model_records[0].payload["model_request"]["provider"] == "model"
    assert (
        model_records[0].payload["model_response"]["continuation"]["response_id"]
        == "resp-1"
    )

    resumed_model = ContinuationModel(block_continuation=False)
    resumed = await Engine(
        ContinuationAgent(resumed_model),
        journal=JsonlSessionJournal(tmp_path),
    ).aresume_from_journal(journal.run_id)

    assert resumed.state.final_result == "resumed"
    assert len(resumed_model.requests) == 1
    assert resumed_model.requests[0].continuation is not None
    assert resumed_model.requests[0].continuation.response_id == "resp-1"


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


class CloseFailingJournal(FailingJournal):
    async def close(self) -> None:
        await super().close()
        raise RuntimeError("injected close failure")


class ForkSourceCloseFailingJournal(JsonlSessionJournal):
    async def close(self) -> None:
        await super().close()
        if self.run_id != "orphan-safe-child":
            raise RuntimeError("source close failed")


@pytest.mark.asyncio
async def test_fresh_run_accepts_an_explicit_stable_run_id(tmp_path: Path) -> None:
    journal = JsonlSessionJournal(tmp_path)
    result = await Engine(
        agent=JournalAgent(),
        journal=journal,
    ).arun("inspect", run_id="stable-child-run")

    assert result.run_id == "stable-child-run"
    assert journal.run_id == "stable-child-run"


@pytest.mark.asyncio
async def test_resume_preserves_continuation_when_complete_tool_batch_was_uncommitted(
    tmp_path: Path,
) -> None:
    class ContinuationModel(Model):
        def __init__(self) -> None:
            super().__init__(model="recovered-continuation", temperature=None)
            self.requests: list[ModelRequest] = []
            self.qitos_harness_metadata = {
                "tool_policy": {"native_tool_call_preferred": True},
                "parser": "ReActTextParser",
                "protocol": "react_text_v1",
            }

        @property
        def capabilities(self) -> ModelCapabilities:
            return ModelCapabilities(
                api=ModelAPI.RESPONSES,
                native_tool_calls=True,
                continuation=True,
            )

        async def stream(
            self,
            request: ModelRequest,
        ) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if request.continuation is not None:
                yield ModelStreamEvent(
                    text="Final Answer: resumed",
                    type=ModelStreamEventType.COMPLETED,
                    finish_reason="stop",
                )
                return
            yield ModelStreamEvent(
                type=ModelStreamEventType.COMPLETED,
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "inspect", "arguments": "{}"},
                    }
                ],
                finish_reason="tool_calls",
                continuation=ModelContinuation(
                    run_id=request.run_id,
                    provider=request.provider,
                    model=request.model,
                    protocol=request.protocol,
                    response_id="resp-uncommitted",
                    prefix_items=1,
                    prefix_digest="prefix",
                    settings_digest="settings",
                ),
            )

    class ContinuationAgent(JournalAgent):
        def __init__(self, model: ContinuationModel) -> None:
            super().__init__()
            self.llm = model
            self.model_parser = ReActTextParser()

        def decide(
            self,
            state: JournalState,
            observation: dict[str, Any],
        ) -> Decision[Action] | None:
            _ = state, observation
            return None

    failed = FailingJournal(tmp_path, fail_type=JournalRecordType.STEP_COMMITTED)
    with pytest.raises(JournalError, match="step.committed"):
        await Engine(
            ContinuationAgent(ContinuationModel()),
            journal=failed,
        ).arun("inspect")

    resumed_model = ContinuationModel()
    resumed = await Engine(
        ContinuationAgent(resumed_model),
        journal=JsonlSessionJournal(tmp_path),
    ).aresume_from_journal(failed.run_id)

    assert resumed.state.final_result == "resumed"
    assert len(resumed_model.requests) == 1
    continuation = resumed_model.requests[0].continuation
    assert continuation is not None
    assert continuation.response_id == "resp-uncommitted"


@pytest.mark.asyncio
async def test_journal_initialization_failure_closes_tools_and_environment(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class TrackingTool(BaseTool):
        def __init__(self) -> None:
            super().__init__(ToolSpec(name="tracking", description="lifecycle fixture"))

        async def asetup(self, context: dict[str, Any]) -> None:
            _ = context
            events.append("tool.setup")

        async def execute(
            self,
            args: dict[str, Any],
            runtime_context: dict[str, Any] | None = None,
        ) -> str:
            _ = args, runtime_context
            return "ok"

        async def aclose(self) -> None:
            events.append("tool.close")

    class TrackingEnv(Env):
        def reset(
            self,
            task: Any = None,
            workspace: str | None = None,
            **kwargs: Any,
        ) -> EnvObservation:
            _ = task, workspace, kwargs
            events.append("env.reset")
            return EnvObservation(data={})

        def observe(self, state: Any = None) -> EnvObservation:
            _ = state
            return EnvObservation(data={})

        def step(self, action: Any, state: Any = None) -> EnvStepResult:
            _ = action, state
            return EnvStepResult(observation=EnvObservation(data={}))

        async def ateardown(self) -> None:
            events.append("env.teardown")

    agent = JournalAgent()
    agent.tool_registry.register(TrackingTool())
    journal = FailingJournal(tmp_path, fail_type=JournalRecordType.INPUT_ACCEPTED)

    with pytest.raises(JournalError, match="input.accepted"):
        await Engine(agent=agent, env=TrackingEnv(), journal=journal).arun("inspect")

    assert journal.closed is True
    assert events == ["tool.setup", "env.reset", "env.teardown", "tool.close"]


@pytest.mark.asyncio
async def test_tool_does_not_execute_when_started_record_fails(tmp_path: Path) -> None:
    agent = JournalAgent()
    journal = FailingJournal(tmp_path, fail_type=JournalRecordType.TOOL_STARTED)

    with pytest.raises(JournalError, match="tool.started"):
        await Engine(agent=agent, journal=journal).arun("inspect")

    assert journal.closed is True
    assert agent.executions == 0


@pytest.mark.asyncio
async def test_close_failure_does_not_mask_run_failure(tmp_path: Path) -> None:
    journal = CloseFailingJournal(
        tmp_path,
        fail_type=JournalRecordType.TOOL_STARTED,
    )

    with pytest.raises(JournalError, match="tool.started"):
        await Engine(agent=JournalAgent(), journal=journal).arun("inspect")

    assert journal.closed is True


@pytest.mark.asyncio
async def test_engine_aclose_retries_only_incomplete_run_cleanup(
    tmp_path: Path,
) -> None:
    class RetryCloseTool(BaseTool):
        def __init__(self) -> None:
            self.close_calls = 0
            super().__init__(ToolSpec(name="retry_close", description="cleanup probe"))

        async def execute(
            self,
            args: dict[str, Any],
            runtime_context: dict[str, Any] | None = None,
        ) -> str:
            _ = args, runtime_context
            return "unused"

        async def aclose(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("tool close failed once")

    class RetryCloseEnv(Env):
        def __init__(self) -> None:
            self.close_calls = 0

        def reset(
            self,
            task: Any = None,
            workspace: str | None = None,
            **kwargs: Any,
        ) -> EnvObservation:
            _ = task, workspace, kwargs
            return EnvObservation(data={})

        def observe(self, state: Any = None) -> EnvObservation:
            _ = state
            return EnvObservation(data={})

        def step(self, action: Any, state: Any = None) -> EnvStepResult:
            _ = action, state
            return EnvStepResult(observation=EnvObservation(data={}))

        async def ateardown(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("env close failed once")

    class RetryCloseJournal(JsonlSessionJournal):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("journal close failed once")
            await super().close()

    agent = JournalAgent()
    tool = RetryCloseTool()
    agent.tool_registry.register(tool)
    env = RetryCloseEnv()
    journal = RetryCloseJournal(tmp_path)
    engine = Engine(agent=agent, env=env, journal=journal)

    with pytest.raises(RuntimeError, match="env close failed once"):
        await engine.arun("inspect")

    assert env.close_calls == 1
    assert tool.close_calls == 1
    assert journal.close_calls == 1

    await engine.aclose()
    await engine.aclose()

    assert env.close_calls == 2
    assert tool.close_calls == 2
    assert journal.close_calls == 2
    assert journal.closed is True


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
async def test_finalizer_failure_aborts_the_uncommitted_transaction(
    tmp_path: Path,
) -> None:
    class BrokenFinalizerAgent(JournalAgent):
        def finalize_action_result(
            self,
            state: JournalState,
            action: Action,
            result: ToolResult,
            *,
            step_id: int,
            context: ActionResultContext,
        ) -> ToolResult:
            _ = state, action, result, step_id, context
            raise RuntimeError("finalizer failed")

    agent = BrokenFinalizerAgent()
    journal = JsonlSessionJournal(tmp_path)

    with pytest.raises(RuntimeError, match="finalizer failed"):
        await Engine(agent=agent, journal=journal).arun("inspect")

    record_types = [record.type for record in await journal.replay()]
    assert agent.executions == 1
    assert JournalRecordType.TOOL_STARTED in record_types
    assert JournalRecordType.TOOL_TERMINAL not in record_types
    assert JournalRecordType.STEP_COMMITTED not in record_types

    resumed_agent = JournalAgent()
    resumed_journal = JsonlSessionJournal(tmp_path)
    resumed = await Engine(
        agent=resumed_agent,
        journal=resumed_journal,
    ).aresume_from_journal(journal.run_id)

    assert resumed_agent.executions == 0
    assert resumed.state.final_result == "done"
    terminal = next(
        record
        for record in await resumed_journal.replay()
        if record.type is JournalRecordType.TOOL_TERMINAL
    )
    assert terminal.payload["result"]["metadata"]["side_effect"] == "unknown"


@pytest.mark.asyncio
async def test_reducer_failure_preserves_terminal_for_resume(tmp_path: Path) -> None:
    class BrokenReducerAgent(JournalAgent):
        def reduce_action_result(
            self,
            state: JournalState,
            action: Action,
            result: ToolResult,
            *,
            step_id: int,
        ) -> JournalState:
            _ = state, action, result, step_id
            raise RuntimeError("reducer failed")

    agent = BrokenReducerAgent()
    journal = JsonlSessionJournal(tmp_path)

    with pytest.raises(RuntimeError, match="reducer failed"):
        await Engine(agent=agent, journal=journal).arun("inspect")

    record_types = [record.type for record in await journal.replay()]
    assert agent.executions == 1
    assert JournalRecordType.TOOL_TERMINAL in record_types
    assert JournalRecordType.STEP_COMMITTED not in record_types

    resumed_agent = JournalAgent()
    resumed = await Engine(
        agent=resumed_agent,
        journal=JsonlSessionJournal(tmp_path),
    ).aresume_from_journal(journal.run_id)

    assert resumed_agent.executions == 0
    assert resumed.state.seen == ["canonical"]
    assert resumed.state.final_result == "done"


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
            request: ModelRequest,
        ) -> AsyncIterator[ModelStreamEvent]:
            _ = request
            if self._wait:
                started.set()
                await asyncio.Event().wait()
            yield ModelStreamEvent(
                text="Final Answer: resumed",
                type=ModelStreamEventType.COMPLETED,
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

    cancelled = await run_task
    assert cancelled.state.stop_reason == "cancelled_immediate"
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
        journal=(source_journal := JsonlSessionJournal(tmp_path)),
    ).afork_journal(
        original.run_id,
        committed.position,
        new_run_id="forked-run",
    )

    assert source_journal.closed is True
    assert forked.closed is False
    fork_agent = JournalAgent()
    resumed = await Engine(
        agent=fork_agent,
        journal=forked,
    ).aresume_from_journal("forked-run")

    assert fork_agent.executions == 0
    assert forked.closed is True
    assert resumed.run_id == "forked-run"
    assert resumed.state.seen == ["canonical"]
    assert resumed.state.final_result == "done"


@pytest.mark.asyncio
async def test_fork_closes_child_when_source_close_fails(tmp_path: Path) -> None:
    source = JsonlSessionJournal(tmp_path)
    result = await Engine(agent=JournalAgent(), journal=source).arun("inspect")
    committed = next(
        record
        for record in await source.replay()
        if record.type is JournalRecordType.STEP_COMMITTED
    )
    source_for_fork = ForkSourceCloseFailingJournal(tmp_path)

    with pytest.raises(RuntimeError, match="source close failed"):
        await Engine(
            agent=JournalAgent(),
            journal=source_for_fork,
        ).afork_journal(
            result.run_id,
            committed.position,
            new_run_id="orphan-safe-child",
        )

    reopened_child = JsonlSessionJournal(tmp_path)
    await reopened_child.open("orphan-safe-child")
    await reopened_child.close()


@pytest.mark.asyncio
async def test_nested_terminal_fork_recovers_without_replaying_tools(
    tmp_path: Path,
) -> None:
    original_journal = JsonlSessionJournal(tmp_path)
    original = await Engine(
        agent=JournalAgent(),
        journal=original_journal,
    ).arun("inspect")
    terminal_boundary = next(
        record
        for record in reversed(await original_journal.replay())
        if record.type is JournalRecordType.STATE_SNAPSHOT
    )

    source = JsonlSessionJournal(tmp_path)
    await source.open(original.run_id)
    child = await source.fork(terminal_boundary.position, "child-run")
    child_boundary = await JsonlRunCatalog(tmp_path).inspect_run("child-run")
    assert child_boundary.committed_position is not None
    grandchild = await child.fork(
        child_boundary.committed_position,
        "grandchild-run",
    )
    await grandchild.close()
    await child.close()
    await source.close()
    (tmp_path / original.run_id / "journal.jsonl").unlink()
    (tmp_path / "child-run" / "journal.jsonl").unlink()

    resumed_agent = JournalAgent()
    resumed = await Engine(
        agent=resumed_agent,
        journal=JsonlSessionJournal(tmp_path),
    ).aresume_from_journal("grandchild-run")

    assert resumed_agent.executions == 0
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
async def test_journal_commits_critic_retry_before_the_next_step(
    tmp_path: Path,
) -> None:
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
