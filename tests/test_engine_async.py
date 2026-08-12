"""Behavior tests for Engine's canonical async API and event stream."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from qitos import (
    Action,
    AgentModule,
    Decision,
    Engine,
    EngineEvent,
    EngineEventType,
    EventStream,
    StateSchema,
    ToolRegistry,
    tool,
)
from qitos.engine import RuntimeBudget
from qitos.engine.states import RuntimePhase
from qitos.kit.parser import ReActTextParser
from qitos.models import Model, ModelStreamChunk


@dataclass
class DemoState(StateSchema):
    logs: list[str] = field(default_factory=list)


class DemoAgent(AgentModule[DemoState, dict[str, Any], Action]):
    def __init__(self, answer: str = "42") -> None:
        registry = ToolRegistry()

        @tool(name="add")
        def add(a: int, b: int) -> int:
            return a + b

        registry.register(add)
        self._answer = answer
        super().__init__(tool_registry=registry)

    def init_state(self, task: str, **kwargs: Any) -> DemoState:
        _ = kwargs
        return DemoState(task=task, max_steps=3)

    def decide(
        self,
        state: DemoState,
        observation: dict[str, Any],
    ) -> Decision[Action]:
        _ = observation
        if state.current_step == 0:
            return Decision.act(
                actions=[Action(name="add", args={"a": 1, "b": 2})],
                rationale="use tool",
            )
        return Decision.final(self._answer)

    def reduce(
        self,
        state: DemoState,
        observation: dict[str, Any],
        decision: Decision[Action],
    ) -> DemoState:
        _ = observation, decision
        return state


class WaitingModel(Model):
    """Model transaction that remains pending until its caller cancels it."""

    def __init__(self) -> None:
        super().__init__(model="waiting-model")
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        deadline_monotonic: float | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelStreamChunk]:
        _ = messages, deadline_monotonic, kwargs
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.cancelled.set()
        if False:  # pragma: no cover - preserve the async-generator contract
            yield ModelStreamChunk()


class ModelAgent(AgentModule[DemoState, dict[str, Any], Action]):
    def __init__(self, model: Model) -> None:
        super().__init__(llm=model, model_parser=ReActTextParser())

    def init_state(self, task: str, **kwargs: Any) -> DemoState:
        _ = kwargs
        return DemoState(task=task, max_steps=3)

    def reduce(
        self,
        state: DemoState,
        observation: dict[str, Any],
        decision: Decision[Action],
    ) -> DemoState:
        _ = observation, decision
        return state


class TestEventStream:
    @pytest.mark.asyncio
    async def test_emit_and_iterate(self) -> None:
        stream = EventStream()
        events: list[EngineEvent] = []

        async def consume() -> None:
            async for event in stream:
                events.append(event)

        consume_task = asyncio.create_task(consume())
        stream.emit(
            EngineEvent(
                event_type=EngineEventType.RUN_START,
                payload={"task": "test"},
            )
        )
        stream.emit(EngineEvent(event_type=EngineEventType.STEP_START, step_id=0))
        stream.close()
        await consume_task

        assert [event.event_type for event in events] == [
            EngineEventType.RUN_START,
            EngineEventType.STEP_START,
        ]

    def test_to_dict(self) -> None:
        event = EngineEvent(
            event_type=EngineEventType.DECIDE,
            step_id=1,
            agent_id="coder",
            phase=RuntimePhase.DECIDE,
            payload={"mode": "act"},
        )

        assert event.to_dict() == {
            "event_type": "decide",
            "step_id": 1,
            "ok": True,
            "ts": event.ts,
            "agent_id": "coder",
            "phase": "DECIDE",
            "payload": {"mode": "act"},
        }

    @pytest.mark.asyncio
    async def test_subscribe_fanout(self) -> None:
        stream = EventStream()
        first = stream.subscribe()
        second = stream.subscribe()

        stream.emit(EngineEvent(event_type=EngineEventType.RUN_START))
        stream.close()

        first_event = await first.get()
        second_event = await second.get()
        assert first_event is not None
        assert second_event is not None
        assert first_event.event_type is EngineEventType.RUN_START
        assert second_event.event_type is EngineEventType.RUN_START
        assert await first.get() is None
        assert await second.get() is None

    @pytest.mark.asyncio
    async def test_close_ends_iteration_without_events(self) -> None:
        stream = EventStream()
        stream.close()

        assert [event async for event in stream] == []


def test_engine_event_types_are_stable() -> None:
    assert {event_type.value for event_type in EngineEventType} == {
        "step_start",
        "step_end",
        "phase_start",
        "phase_end",
        "decide",
        "act",
        "reduce",
        "critic",
        "check_stop",
        "handoff",
        "delegate",
        "fanout",
        "error",
        "run_start",
        "run_end",
        "step_stream",
        "interrupt",
    }


class TestEngineAsync:
    @pytest.mark.asyncio
    async def test_arun_returns_result(self) -> None:
        engine = Engine(
            agent=DemoAgent(answer="hello world"),
            budget=RuntimeBudget(max_steps=5),
        )

        result = await engine.arun("test task")

        assert result.step_count >= 1
        assert result.state.final_result == "hello world"

    @pytest.mark.asyncio
    async def test_arun_stream_yields_lifecycle_events(self) -> None:
        engine = Engine(
            agent=DemoAgent(answer="stream test"),
            budget=RuntimeBudget(max_steps=5),
        )

        events = [event async for event in engine.arun_stream("test task")]

        event_types = [event.event_type for event in events]
        assert event_types.count(EngineEventType.RUN_START) == 1
        assert event_types[-1] is EngineEventType.RUN_END
        assert EngineEventType.STEP_START in event_types
        assert EngineEventType.STEP_END in event_types

    @pytest.mark.asyncio
    async def test_arun_stream_emits_one_terminal_event_before_error(self) -> None:
        class BrokenAgent(DemoAgent):
            def init_state(self, task: str, **kwargs: Any) -> DemoState:
                _ = task, kwargs
                raise RuntimeError("broken initialization")

        engine = Engine(
            agent=BrokenAgent(),
            budget=RuntimeBudget(max_steps=5),
        )
        events: list[EngineEvent] = []

        with pytest.raises(RuntimeError, match="broken initialization"):
            async for event in engine.arun_stream("test task"):
                events.append(event)

        terminal = [
            event for event in events if event.event_type is EngineEventType.RUN_END
        ]
        assert len(terminal) == 1
        assert terminal[0].ok is False
        assert terminal[0].payload["error_type"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_closing_stream_cancels_the_engine_run(self) -> None:
        model = WaitingModel()
        engine = Engine(
            agent=ModelAgent(model),
            budget=RuntimeBudget(max_steps=5),
        )
        iterator = engine.arun_stream("wait").__aiter__()

        while (await iterator.__anext__()).event_type is not EngineEventType.RUN_START:
            pass
        await model.started.wait()
        await iterator.aclose()

        await asyncio.wait_for(model.cancelled.wait(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_cancel_stops_an_active_async_run(self) -> None:
        model = WaitingModel()
        engine = Engine(
            agent=ModelAgent(model),
            budget=RuntimeBudget(max_steps=5),
        )
        run_task = asyncio.create_task(engine.arun("wait"))
        await model.started.wait()

        engine.cancel()

        with pytest.raises(asyncio.CancelledError):
            await run_task
        assert model.cancelled.is_set()

    @pytest.mark.asyncio
    async def test_run_rejects_an_active_event_loop(self) -> None:
        engine = Engine(
            agent=DemoAgent(answer="sync test"),
            budget=RuntimeBudget(max_steps=5),
        )

        with pytest.raises(
            RuntimeError,
            match=r"Engine\.run\(\) cannot run inside an active event loop",
        ):
            engine.run("test task")

    @pytest.mark.asyncio
    async def test_second_arun_is_rejected_while_run_is_active(self) -> None:
        model = WaitingModel()
        engine = Engine(
            agent=ModelAgent(model),
            budget=RuntimeBudget(max_steps=5),
        )
        first_run = asyncio.create_task(engine.arun("first"))
        await model.started.wait()

        with pytest.raises(RuntimeError, match="Engine already has an active run"):
            await engine.arun("second")

        engine.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_run
