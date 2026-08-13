from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from qitos import AgentModule, Decision, Engine, StateSchema, ToolRegistry
from qitos.checkpoint import CheckpointConfig, InMemoryCheckpointStore
from qitos.core.history import HistoryMessage, HistorySnapshot
from qitos.engine import RuntimeBudget
from qitos.kit.history import WindowHistory
from qitos.kit.parser import ReActTextParser
from qitos.models import Model, ModelRequest, ModelStreamEvent, ModelStreamEventType
from qitos.trace import TraceWriter


@dataclass
class _State(StateSchema):
    marker: str = ""


class _Agent(AgentModule[_State, dict[str, Any], Any]):
    def __init__(self, model: Model) -> None:
        super().__init__(
            tool_registry=ToolRegistry(),
            llm=model,
            history=WindowHistory(window_size=24),
            model_parser=ReActTextParser(),
        )

    def init_state(self, task: str, **kwargs: Any) -> _State:
        return _State(
            task=task,
            marker=str(kwargs.get("marker", "")),
            max_steps=3,
        )

    def build_system_prompt(self, state: _State) -> str | None:
        _ = state
        return "You are a test agent."

    def prepare(self, state: _State) -> str:
        return f"Solve: {state.task}"

    def reduce(
        self,
        state: _State,
        observation: dict[str, Any],
        decision: Decision[Any],
    ) -> _State:
        _ = observation, decision
        return state


def _history_snapshot() -> HistorySnapshot:
    return HistorySnapshot.from_messages(
        [
            HistoryMessage(role="user", step_id=0, content="prior request"),
            HistoryMessage(
                role="assistant",
                step_id=0,
                content=None,
                tool_calls=[
                    {
                        "id": "prior-call",
                        "type": "function",
                        "function": {"name": "inspect", "arguments": "{}"},
                    }
                ],
            ),
            HistoryMessage(
                role="tool",
                step_id=0,
                content="prior result",
                tool_call_id="prior-call",
                name="inspect",
            ),
        ]
    )


class _BlockingModel(Model):
    def __init__(self) -> None:
        super().__init__(model="checkpoint-test", temperature=None)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        _ = request
        self.calls += 1
        self.entered.set()
        await self.release.wait()
        yield ModelStreamEvent(text="Final Answer: resumed", type=ModelStreamEventType.COMPLETED)


class _FailingModel(Model):
    def __init__(self, observe: Callable[[], Awaitable[None]] | None = None) -> None:
        super().__init__(model="checkpoint-failure-test", temperature=None)
        self.observe = observe
        self.calls = 0

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        _ = request
        self.calls += 1
        if self.observe is not None:
            await self.observe()
        raise RuntimeError("provider failed before producing a response")
        if False:  # pragma: no cover
            yield ModelStreamEvent(
                type=ModelStreamEventType.LIFECYCLE,
                event_type="unreachable",
            )


class _FinalModel(Model):
    def __init__(self, answer: str) -> None:
        super().__init__(model="checkpoint-final-test", temperature=None)
        self.answer = answer
        self.calls = 0

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        _ = request
        self.calls += 1
        yield ModelStreamEvent(text=f"Final Answer: {self.answer}", type=ModelStreamEventType.COMPLETED)


@pytest.mark.asyncio
async def test_input_checkpoint_is_visible_while_first_provider_request_blocks() -> None:
    store = InMemoryCheckpointStore()
    model = _BlockingModel()
    engine = Engine(
        _Agent(model),
        checkpoint_store=store,
        budget=RuntimeBudget(max_steps=3),
    )
    assert engine.last_checkpoint_id is None
    run_task = asyncio.create_task(
        engine.arun(
            "new task",
            history_snapshot=_history_snapshot(),
            marker="state-marker",
        )
    )

    try:
        await model.entered.wait()
        thread_id = engine._active_run_id
        input_tuple = await store.get_tuple(CheckpointConfig(thread_id=thread_id))
        assert input_tuple is not None
        assert engine.last_checkpoint_id == input_tuple.checkpoint.id
        assert input_tuple.metadata["source"] == "input"
        assert input_tuple.checkpoint.step == 0
        assert input_tuple.checkpoint.task_text == "new task"
        assert input_tuple.checkpoint.state_data["marker"] == "state-marker"
        assert input_tuple.checkpoint.history is not None
        assert [message.role for message in input_tuple.checkpoint.history.messages] == [
            "user",
            "assistant",
            "tool",
        ]

        model.release.set()
        result = await run_task
        checkpoints = await store.list(CheckpointConfig(thread_id=result.run_id))
        assert [item.metadata.get("source") for item in checkpoints] == [
            "loop",
            "input",
        ]
        assert checkpoints[0].checkpoint.parent_id == checkpoints[1].checkpoint.id
    finally:
        if not run_task.done():
            model.release.set()
            await run_task
        await store.close()


@pytest.mark.asyncio
async def test_input_checkpoint_survives_provider_failure_and_resume_retries_step_zero() -> None:
    store = InMemoryCheckpointStore()
    first_run_id: str = ""
    observed: dict[str, Any] = {}

    async def observe_input() -> None:
        config = CheckpointConfig(thread_id=engine._active_run_id)
        observed["tuple"] = await store.get_tuple(config)
        observed["thread_id"] = engine._active_run_id

    failing_model = _FailingModel(observe_input)
    engine = Engine(
        _Agent(failing_model),
        checkpoint_store=store,
        budget=RuntimeBudget(max_steps=3),
    )
    try:
        failed = await engine.arun("retry me", history_snapshot=_history_snapshot())
        first_run_id = failed.run_id
        input_tuple = observed["tuple"]
        assert input_tuple is not None
        assert input_tuple.metadata["source"] == "input"
        assert input_tuple.checkpoint.history is not None
        assert len(input_tuple.checkpoint.history.messages) == 3
        assert failed.state.stop_reason == "unrecoverable_error"

        resumed_model = _BlockingModel()
        resumed_engine = Engine(
            _Agent(resumed_model),
            checkpoint_store=store,
            budget=RuntimeBudget(max_steps=3),
        )
        resumed_model.release.set()
        resumed = await resumed_engine.aresume_from_checkpoint(
            CheckpointConfig(
                thread_id=first_run_id,
                checkpoint_id=input_tuple.checkpoint.id,
            )
        )
        assert resumed.state.final_result == "resumed"
        assert resumed_model.calls == 1

        checkpoints = await store.list(CheckpointConfig(thread_id=first_run_id))
        assert [item.metadata.get("source") for item in checkpoints] == [
            "loop",
            "input",
        ]
        assert checkpoints[0].checkpoint.parent_id == input_tuple.checkpoint.id
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_terminal_checkpoint_resume_does_not_execute_another_step(
    tmp_path: Path,
) -> None:
    store = InMemoryCheckpointStore()
    answer = uuid4().hex
    model = _FinalModel(answer)
    try:
        completed = await Engine(
            _Agent(model),
            checkpoint_store=store,
            budget=RuntimeBudget(max_steps=3),
        ).arun("complete once")
        before = await store.list(CheckpointConfig(thread_id=completed.run_id))
        resumed_model = _FinalModel(uuid4().hex)
        trace_id = f"resume-{uuid4().hex}"
        resumed_engine = Engine(
            _Agent(resumed_model),
            checkpoint_store=store,
            budget=RuntimeBudget(max_steps=3),
            trace_writer=TraceWriter(str(tmp_path), trace_id),
        )

        resumed = await resumed_engine.aresume_from_checkpoint(before[0].config)

        after = await store.list(CheckpointConfig(thread_id=completed.run_id))
        assert resumed.state.to_dict() == completed.state.to_dict()
        assert resumed.run_id == completed.run_id
        assert resumed.records == []
        assert resumed.events == []
        assert resumed_model.calls == 0
        assert resumed_engine.last_checkpoint_id == before[0].checkpoint.id
        assert [item.config for item in after] == [item.config for item in before]
        manifest = json.loads(
            (tmp_path / trace_id / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["status"] == "completed"
        assert manifest["event_count"] == 0
        assert manifest["step_count"] == 0
        assert manifest["summary"]["stop_reason"] == completed.state.stop_reason
        assert manifest["summary"]["final_result"] == answer
    finally:
        await store.close()
