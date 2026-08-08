"""Tests for AsyncEngine, EventStream, and async model adapters."""

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from qitos import (
    AsyncEngine,
    AgentModule,
    Decision,
    Action,
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


# --- Minimal test fixtures ---


@dataclass
class DemoState(StateSchema):
    logs: list[str] = field(default_factory=list)


class DemoAgent(AgentModule[DemoState, dict[str, Any], Action]):
    def __init__(self, answer: str = "42"):
        registry = ToolRegistry()

        @tool(name="add")
        def add(a: int, b: int) -> int:
            return a + b

        registry.register(add)
        self._answer = answer
        super().__init__(tool_registry=registry)

    def init_state(self, task: str, **kwargs: Any) -> DemoState:
        return DemoState(task=task, max_steps=3)

    def decide(self, state: DemoState, observation: dict[str, Any]) -> Decision[Action]:
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
        return state


# --- EventStream tests ---


class TestEventStream:
    def test_emit_and_iterate(self):
        stream = EventStream()
        events = []

        async def _consume():
            async for event in stream:
                events.append(event)

        loop = asyncio.new_event_loop()
        consume_task = loop.create_task(_consume())

        stream.emit(EngineEvent(event_type=EngineEventType.RUN_START, payload={"task": "test"}))
        stream.emit(EngineEvent(event_type=EngineEventType.STEP_START, step_id=0))
        stream.close()

        loop.run_until_complete(consume_task)
        loop.close()

        assert len(events) == 2
        assert events[0].event_type == EngineEventType.RUN_START
        assert events[1].step_id == 0

    def test_to_dict(self):
        event = EngineEvent(
            event_type=EngineEventType.DECIDE,
            step_id=1,
            agent_id="coder",
            phase=RuntimePhase.DECIDE,
            payload={"mode": "act"},
        )
        d = event.to_dict()
        assert d["event_type"] == "decide"
        assert d["step_id"] == 1
        assert d["agent_id"] == "coder"
        assert d["phase"] == "DECIDE"
        assert d["payload"]["mode"] == "act"

    def test_subscribe_fanout(self):
        stream = EventStream()
        q1 = stream.subscribe()
        q2 = stream.subscribe()

        stream.emit(EngineEvent(event_type=EngineEventType.RUN_START))
        stream.close()

        # Both queues should receive the event + close sentinel
        loop = asyncio.new_event_loop()
        e1 = loop.run_until_complete(q1.get())
        e2 = loop.run_until_complete(q2.get())
        loop.close()

        assert e1.event_type == EngineEventType.RUN_START
        assert e2.event_type == EngineEventType.RUN_START

    def test_close_signal(self):
        stream = EventStream()
        events = []

        async def _consume():
            async for event in stream:
                events.append(event)

        loop = asyncio.new_event_loop()
        task = loop.create_task(_consume())
        stream.close()
        loop.run_until_complete(task)
        loop.close()
        assert events == []


# --- EngineEventType tests ---


class TestEngineEventType:
    def test_all_types(self):
        expected = {
            "step_start", "step_end", "phase_start", "phase_end",
            "decide", "act", "reduce", "critic", "check_stop",
            "handoff", "delegate", "fanout", "error",
            "run_start", "run_end", "step_stream",
            "interrupt",
        }
        actual = {t.value for t in EngineEventType}
        assert actual == expected


# --- AsyncEngine tests ---


class TestAsyncEngine:
    def test_arun_returns_result(self):
        agent = DemoAgent(answer="hello world")
        engine = AsyncEngine(agent=agent, budget=RuntimeBudget(max_steps=5))
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(engine.arun("test task"))
        loop.close()

        assert result.step_count >= 1
        assert result.state.final_result == "hello world"

    def test_arun_stream_yields_events(self):
        agent = DemoAgent(answer="stream test")
        engine = AsyncEngine(agent=agent, budget=RuntimeBudget(max_steps=5))
        events = []

        async def _run():
            async for event in engine.arun_stream("test task"):
                events.append(event)

        loop = asyncio.new_event_loop()
        loop.run_until_complete(_run())
        loop.close()

        types = [e.event_type for e in events]
        assert EngineEventType.RUN_START in types
        assert EngineEventType.RUN_END in types
        assert EngineEventType.STEP_START in types or EngineEventType.STEP_END in types

    def test_arun_stream_emits_one_terminal_event_before_run_error(self):
        class _BrokenAgent(DemoAgent):
            def init_state(self, task, **kwargs):
                _ = task, kwargs
                raise RuntimeError("broken initialization")

        engine = AsyncEngine(
            agent=_BrokenAgent(),
            budget=RuntimeBudget(max_steps=5),
        )

        async def _run():
            events = []
            with pytest.raises(RuntimeError, match="broken initialization"):
                async for event in engine.arun_stream("test task"):
                    events.append(event)
            return events

        loop = asyncio.new_event_loop()
        try:
            events = loop.run_until_complete(_run())
        finally:
            loop.close()

        terminal = [
            event for event in events if event.event_type is EngineEventType.RUN_END
        ]
        assert len(terminal) == 1
        assert terminal[0].ok is False
        assert terminal[0].payload["error_type"] == "RuntimeError"

    def test_sync_run_delegates(self):
        agent = DemoAgent(answer="sync test")
        engine = AsyncEngine(agent=agent, budget=RuntimeBudget(max_steps=5))
        result = engine.run("test task")
        assert result.state.final_result == "sync test"

    def test_engine_property(self):
        agent = DemoAgent()
        engine = AsyncEngine(agent=agent, budget=RuntimeBudget(max_steps=5))
        assert isinstance(engine.engine, Engine)
        assert engine.agent is engine.engine.agent

    def test_closing_stream_requests_underlying_engine_cancellation(self):
        class _WaitingAgent(DemoAgent):
            def decide(self, state, observation):
                _ = state, observation
                return Decision.wait(
                    rationale="wait for runtime input",
                    meta={"runtime_wait": True},
                )

        async def _close_after_first_event(engine):
            iterator = engine.arun_stream("wait").__aiter__()
            await iterator.__anext__()
            await iterator.aclose()
            await asyncio.sleep(0.01)

        engine = AsyncEngine(
            agent=_WaitingAgent(),
            budget=RuntimeBudget(max_steps=5),
        )
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_close_after_first_event(engine))
            assert engine.engine._cancel_token.is_cancel_requested
        finally:
            engine.cancel("immediate")
            loop.close()

    def test_cancelled_arun_does_not_wait_for_blocked_engine_thread(self):
        class _BlockingAgent(DemoAgent):
            def __init__(self):
                super().__init__()
                self.entered = threading.Event()
                self.release = threading.Event()
                self.worker_daemon = None

            def decide(self, state, observation):
                _ = state, observation
                self.worker_daemon = threading.current_thread().daemon
                self.entered.set()
                self.release.wait(timeout=1.0)
                return Decision.final("released")

        async def _cancel_blocked_run(engine, agent):
            task = asyncio.create_task(engine.arun("blocked"))
            for _ in range(100):
                if agent.entered.is_set():
                    break
                await asyncio.sleep(0.001)
            assert agent.entered.is_set()
            started = time.monotonic()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return time.monotonic() - started

        agent = _BlockingAgent()
        engine = AsyncEngine(agent=agent, budget=RuntimeBudget(max_steps=5))
        loop = asyncio.new_event_loop()
        try:
            elapsed = loop.run_until_complete(_cancel_blocked_run(engine, agent))
            assert elapsed < 0.1
            assert agent.worker_daemon is True
        finally:
            agent.release.set()
            loop.run_until_complete(asyncio.sleep(0.01))
            loop.close()


# --- Async model tests ---


class TestAsyncOpenAICompatibleModel:
    def test_import(self):
        from qitos.models import AsyncOpenAICompatibleModel, AsyncOpenAIModel
        assert AsyncOpenAICompatibleModel is not None
        assert AsyncOpenAIModel is not None

    def test_factory_registration(self):
        from qitos.models import ModelFactory
        assert "async-openai-compatible" in ModelFactory._providers
        assert "async-openai" in ModelFactory._providers

    def test_async_model_base(self):
        from qitos.models import AsyncModel

        class _TestAsyncModel(AsyncModel):
            async def _acall_api(self, messages):
                return "async response"

        model = _TestAsyncModel(model="test")
        # Sync call should work via asyncio.run fallback
        result = model([{"role": "user", "content": "test"}])
        assert result == "async response"

        # Async call
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(model.acall([{"role": "user", "content": "test"}]))
        loop.close()
        assert result == "async response"
