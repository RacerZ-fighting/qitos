"""Behavior tests for run-scoped runtime input and idle wait."""

from __future__ import annotations

import json
import asyncio
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from qitos import (
    Action,
    AgentModule,
    Decision,
    Engine,
    RuntimeInput,
    StateSchema,
    ToolRegistry,
)
from qitos.engine import RuntimeBudget
from qitos.kit.history import WindowHistory
from qitos.kit.parser import JsonDecisionParser
from qitos.kit.tool.agent import AgentInvocation, AgentTool


@dataclass
class _State(StateSchema):
    pass


class _WaitAgent(AgentModule[_State, dict[str, Any], Any]):
    name = "runtime-wait"

    def __init__(self, *, runtime_wait: bool = True) -> None:
        self.calls = 0
        self.first_decision = threading.Event()
        self.history = WindowHistory(window_size=20)
        self._runtime_wait = runtime_wait
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
        if self.calls == 1:
            self.first_decision.set()
            return Decision.wait(
                rationale="waiting",
                meta={"runtime_wait": True} if self._runtime_wait else {},
            )
        return Decision.final("done")

    def reduce(
        self,
        state: _State,
        observation: dict[str, Any],
        decision: Decision[Any],
    ) -> _State:
        _ = observation, decision
        return state


def _event(event_id: str = "event-1") -> RuntimeInput:
    return RuntimeInput(
        event_id=event_id,
        kind="task.completed",
        correlation_id="task-1",
        source="test",
        payload={"status": "completed"},
    )


@pytest.mark.parametrize("payload", [{"bad": object()}, {"bad": float("nan")}])
def test_runtime_input_rejects_non_json_payload(payload: dict[str, Any]) -> None:
    with pytest.raises(TypeError, match="JSON serializable"):
        RuntimeInput(
            event_id="event-1",
            kind="task.completed",
            correlation_id="task-1",
            source="test",
            payload=payload,
        )


def test_runtime_input_round_trips_its_strict_journal_shape() -> None:
    event = _event()

    assert RuntimeInput.from_dict(event.to_dict()) == event
    with pytest.raises(ValueError, match="fields are invalid"):
        RuntimeInput.from_dict({**event.to_dict(), "unexpected": True})


def test_runtime_wait_sleeps_until_event_and_delivers_it_at_safe_point() -> None:
    agent = _WaitAgent()
    engine = Engine(agent, budget=RuntimeBudget(max_steps=4))
    result_holder: list[Any] = []
    thread = threading.Thread(target=lambda: result_holder.append(engine.run("wait")))

    assert engine.post_runtime_event(_event("too-early"), run_id="not-running") is False
    thread.start()
    assert agent.first_decision.wait(timeout=1)
    run_id = engine.active_run_id

    time.sleep(0.05)
    assert thread.is_alive()
    assert agent.calls == 1
    assert engine.current_state is not None
    assert engine.current_state.current_step == 0

    assert engine.post_runtime_event(_event(), run_id=run_id) is True
    assert engine.post_runtime_event(_event(), run_id=run_id) is False
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert agent.calls == 2
    assert result_holder[0].state.final_result == "done"
    assert engine.post_runtime_event(_event("too-late"), run_id=run_id) is False

    runtime_messages = [
        message
        for message in agent.history.messages
        if message.role == "user" and message.metadata.get("source") == "runtime"
    ]
    assert len(runtime_messages) == 1
    assert (
        json.loads(runtime_messages[0].content)["runtime_events"][0]["event_id"]
        == "event-1"
    )
    assert not any(message.role == "tool" for message in agent.history.messages)

    stages = [event.payload.get("stage") for event in result_holder[0].events]
    assert stages.count("runtime_wait_start") == 1
    assert stages.count("runtime_wait_end") == 1
    assert stages.count("runtime_input") == 1


def test_background_agent_completion_wakes_parent_runtime_wait() -> None:
    child_started = threading.Event()
    release_child = threading.Event()

    class ChildEngine:
        active_run_id = "child-run"

        async def arun(self, task: str, **kwargs: Any) -> Any:
            _ = kwargs
            child_started.set()
            assert await asyncio.to_thread(release_child.wait, 1)
            return SimpleNamespace(
                state=SimpleNamespace(
                    final_result=f"validated:{task}",
                    stop_reason="final",
                ),
                records=[],
                step_count=1,
                total_tokens=2,
                run_id="child-run",
            )

        def cancel(self, mode: str) -> None:
            _ = mode
            release_child.set()

    agent_tool = AgentTool(
        invocation_factory=lambda request, _context: AgentInvocation(
            engine=ChildEngine(),
            task=request.prompt,
        ),
        execution_mode="background",
    )

    class ParentAgent(AgentModule[_State, dict[str, Any], Action]):
        name = "background-parent"

        def __init__(self) -> None:
            self.calls = 0
            self.waiting = threading.Event()
            self.history = WindowHistory(window_size=20)
            super().__init__(
                tool_registry=ToolRegistry().register(agent_tool),
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
            _ = state, observation
            self.calls += 1
            if self.calls == 1:
                return Decision.act(
                    [
                        Action(
                            name="Agent",
                            args={"description": "route", "prompt": "one"},
                        )
                    ]
                )
            if self.calls == 2:
                self.waiting.set()
                return Decision.wait(meta={"runtime_wait": True})
            return Decision.final("parent-done")

        def reduce(
            self,
            state: _State,
            observation: dict[str, Any],
            decision: Decision[Action],
        ) -> _State:
            _ = observation, decision
            return state

    parent = ParentAgent()
    engine = Engine(parent, budget=RuntimeBudget(max_steps=5))
    result_holder: list[Any] = []
    thread = threading.Thread(target=lambda: result_holder.append(engine.run("test")))

    thread.start()
    assert child_started.wait(timeout=1)
    assert parent.waiting.wait(timeout=1)
    assert thread.is_alive()

    release_child.set()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert parent.calls == 3
    assert result_holder[0].state.final_result == "parent-done"
    runtime_messages = [
        json.loads(message.content)
        for message in parent.history.messages
        if message.role == "user" and message.metadata.get("source") == "runtime"
    ]
    assert len(runtime_messages) == 1
    event = runtime_messages[0]["runtime_events"][0]
    assert event["kind"] == "agent.child.completed"
    assert event["payload"]["output"] == "validated:one"


def test_cancel_wakes_runtime_wait_without_another_decide() -> None:
    agent = _WaitAgent()
    engine = Engine(agent, budget=RuntimeBudget(max_steps=4))
    result_holder: list[Any] = []
    thread = threading.Thread(target=lambda: result_holder.append(engine.run("wait")))

    thread.start()
    assert agent.first_decision.wait(timeout=1)
    engine.cancel("immediate")
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert agent.calls == 1
    assert result_holder[0].state.stop_reason == "cancelled_immediate"


def test_runtime_deadline_wakes_wait_without_model_polling() -> None:
    agent = _WaitAgent()
    engine = Engine(
        agent,
        budget=RuntimeBudget(max_steps=4, max_runtime_seconds=0.05),
    )

    result = engine.run("wait")

    assert agent.calls == 1
    assert result.state.stop_reason == "budget_time"
    assert result.runtime_seconds < 1


def test_unmarked_wait_keeps_existing_immediate_loop_behavior() -> None:
    agent = _WaitAgent(runtime_wait=False)

    result = Engine(agent, budget=RuntimeBudget(max_steps=4)).run("repair")

    assert agent.calls == 2
    assert result.state.final_result == "done"


def test_json_parser_requires_explicit_true_for_runtime_wait() -> None:
    parser = JsonDecisionParser()

    idle = parser.parse('{"mode":"wait","runtime_wait":true}')
    ordinary = parser.parse('{"mode":"wait","runtime_wait":"true"}')

    assert idle.meta.get("runtime_wait") is True
    assert "runtime_wait" not in ordinary.meta
