"""Contract tests for run deadlines and cooperative tool cancellation."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from qitos import AgentModule, Decision, Engine, StateSchema, ToolRegistry
from qitos.core.action import Action, ActionResult, ActionStatus
from qitos.core.tool import BaseTool, RetryPolicy, ToolSpec
from qitos.engine.action_executor import ActionExecutor
from qitos.engine.cancellation import CancelToken
from qitos.engine.states import RuntimeBudget


class _Registry:
    def __init__(self, tool: BaseTool) -> None:
        self._tool = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tool if name == self._tool.spec.name else None

    def list_tools(self) -> list[str]:
        return [self._tool.spec.name]


class _RuntimeEngine:
    def __init__(self, deadline_monotonic: float | None) -> None:
        self._deadline_monotonic = deadline_monotonic
        self._cancel_token = CancelToken()
        self.active_run_id = "run-deadline-test"
        self.agent = SimpleNamespace()

    @property
    def runtime_deadline_monotonic(self) -> float | None:
        return self._deadline_monotonic

    def remaining_runtime_seconds(self) -> float | None:
        if self._deadline_monotonic is None:
            return None
        return max(0.0, self._deadline_monotonic - time.monotonic())

    def post_runtime_event(self, event: Any, *, run_id: str) -> bool:
        _ = event, run_id
        return True


def _execute(tool: BaseTool, engine: _RuntimeEngine) -> ActionResult:
    return ActionExecutor(
        tool_registry=_Registry(tool),
        engine=engine,
    ).execute([Action(name=tool.spec.name)])[0]


class _RecordingTool(BaseTool):
    def __init__(self, *, delay: float = 0.0) -> None:
        super().__init__(ToolSpec(name="record", description="record runtime context"))
        self.delay = delay
        self.calls = 0
        self.runtime_context: dict[str, Any] | None = None
        self.worker_daemon: bool | None = None

    def execute(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> str:
        _ = args
        self.calls += 1
        self.runtime_context = runtime_context
        self.worker_daemon = threading.current_thread().daemon
        if self.delay:
            time.sleep(self.delay)
        return "ok"


def test_runtime_budget_accepts_an_absolute_monotonic_deadline() -> None:
    deadline = time.monotonic() + 30.0

    budget = RuntimeBudget(max_steps=3, deadline_monotonic=deadline)

    assert budget.deadline_monotonic == deadline


def test_expired_deadline_prevents_tool_admission() -> None:
    tool = _RecordingTool()
    engine = _RuntimeEngine(time.monotonic() - 1.0)

    result = _execute(tool, engine)

    assert tool.calls == 0
    assert result.status is ActionStatus.TIMED_OUT
    assert result.attempts == 0
    assert result.metadata["timeout_source"] == "runtime_deadline"
    assert result.metadata["started"] is False


def test_runtime_deadline_clamps_a_longer_tool_timeout() -> None:
    tool = _RecordingTool(delay=0.2)
    engine = _RuntimeEngine(time.monotonic() + 0.03)
    started = time.monotonic()

    result = _execute(tool, engine)

    assert time.monotonic() - started < 0.15
    assert result.status is ActionStatus.TIMED_OUT
    assert result.metadata["timeout_source"] == "runtime_deadline"
    assert result.metadata["worker_still_running"] is True
    assert tool.worker_daemon is True


class _RetryingTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            ToolSpec(
                name="retry",
                description="always fail",
                retry_policy=RetryPolicy(
                    max_attempts=3,
                    backoff_factor=0.05,
                    max_backoff=0.05,
                    jitter=False,
                    retryable_exceptions=(RuntimeError,),
                ),
            )
        )
        self.calls = 0

    def execute(self, args: dict[str, Any], runtime_context: Any = None) -> None:
        _ = args, runtime_context
        self.calls += 1
        raise RuntimeError("retry me")


def test_retry_backoff_does_not_cross_the_runtime_deadline() -> None:
    tool = _RetryingTool()
    engine = _RuntimeEngine(time.monotonic() + 0.01)
    started = time.monotonic()

    result = _execute(tool, engine)

    assert time.monotonic() - started < 0.05
    assert tool.calls == 1
    assert result.status is ActionStatus.TIMED_OUT
    assert result.metadata["timeout_source"] == "runtime_deadline"


class _CancellationAwareTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(ToolSpec(name="cancel_aware", description="cooperative cancel"))
        self.entered = threading.Event()
        self.observed_cancel = False
        self.deadline_monotonic: float | None = None
        self.remaining_seconds: Any = None

    def execute(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> str:
        _ = args
        assert runtime_context is not None
        cancelled = runtime_context["agent_cancelled"]
        self.deadline_monotonic = runtime_context["deadline_monotonic"]
        self.remaining_seconds = runtime_context["remaining_seconds"]
        self.entered.set()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if cancelled():
                self.observed_cancel = True
                return "cancelled"
            time.sleep(0.001)
        return "missed cancellation"


def test_tool_runtime_context_exposes_live_deadline_and_cancellation() -> None:
    tool = _CancellationAwareTool()
    engine = _RuntimeEngine(time.monotonic() + 1.0)
    result_holder: list[ActionResult] = []
    thread = threading.Thread(
        target=lambda: result_holder.append(_execute(tool, engine))
    )

    thread.start()
    assert tool.entered.wait(timeout=0.2)
    engine._cancel_token.request_cancel("immediate")
    thread.join(timeout=0.5)

    assert not thread.is_alive()
    assert tool.observed_cancel is True
    assert tool.deadline_monotonic == engine.runtime_deadline_monotonic
    assert callable(tool.remaining_seconds)
    assert result_holder[0].output == "cancelled"


@dataclass
class _WaitState(StateSchema):
    pass


class _WaitAgent(AgentModule[_WaitState, dict[str, Any], Any]):
    def __init__(self) -> None:
        self.calls = 0
        super().__init__(tool_registry=ToolRegistry())

    def init_state(self, task: str, **kwargs: Any) -> _WaitState:
        _ = kwargs
        return _WaitState(task=task, max_steps=3)

    def decide(
        self,
        state: _WaitState,
        observation: dict[str, Any],
    ) -> Decision[Any]:
        _ = state, observation
        self.calls += 1
        return Decision.wait(meta={"runtime_wait": True})

    def reduce(
        self,
        state: _WaitState,
        observation: dict[str, Any],
        decision: Decision[Any],
    ) -> _WaitState:
        _ = observation, decision
        return state


def test_expired_absolute_deadline_stops_before_model_admission() -> None:
    agent = _WaitAgent()
    engine = Engine(
        agent,
        budget=RuntimeBudget(
            max_steps=3,
            max_runtime_seconds=30.0,
            deadline_monotonic=time.monotonic() - 1.0,
        ),
    )

    result = engine.run("expired")

    assert agent.calls == 0
    assert result.state.stop_reason == "budget_time"


def test_absolute_deadline_clamps_relative_runtime_budget() -> None:
    agent = _WaitAgent()
    engine = Engine(
        agent,
        budget=RuntimeBudget(
            max_steps=3,
            max_runtime_seconds=30.0,
            deadline_monotonic=time.monotonic() + 0.03,
        ),
    )
    started = time.monotonic()

    result = engine.run("wait")

    assert time.monotonic() - started < 0.2
    assert agent.calls == 1
    assert result.state.stop_reason == "budget_time"
