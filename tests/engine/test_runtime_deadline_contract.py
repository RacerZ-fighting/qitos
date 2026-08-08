"""Contract tests for run deadlines and cooperative tool cancellation."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from qitos import AgentModule, Decision, Engine, StateSchema, ToolRegistry
from qitos.core.action import Action, ActionResult, ActionStatus
from qitos.core.errors import ModelTransportError
from qitos.core.tool import (
    BaseTool,
    RetryPolicy,
    ToolPermissionDecision,
    ToolSpec,
    ToolValidationResult,
)
from qitos.engine.action_executor import ActionExecutor
from qitos.engine.cancellation import CancelToken
from qitos.engine.states import RuntimeBudget
from qitos.models.base import ModelStreamChunk


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
        tool_registry=ToolRegistry().register(tool),
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


class _AdmissionCountingTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            ToolSpec(
                name="admission_once",
                description="fail twice",
                retry_policy=RetryPolicy(
                    max_attempts=3,
                    backoff_factor=0,
                    jitter=False,
                    retryable_exceptions=(RuntimeError,),
                ),
            )
        )
        self.validation_calls = 0
        self.permission_calls = 0
        self.execution_calls = 0

    def validate_input(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> ToolValidationResult:
        _ = args, runtime_context
        self.validation_calls += 1
        return ToolValidationResult.ok()

    def check_permissions(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> ToolPermissionDecision:
        _ = args, runtime_context
        self.permission_calls += 1
        return ToolPermissionDecision.allow()

    def execute(self, args: dict[str, Any], runtime_context: Any = None) -> str:
        _ = args, runtime_context
        self.execution_calls += 1
        if self.execution_calls < 3:
            raise RuntimeError("retry")
        return "ok"


def test_tool_retry_repeats_only_the_admitted_invocation() -> None:
    tool = _AdmissionCountingTool()
    result = _execute(tool, _RuntimeEngine(time.monotonic() + 1.0))

    assert result.status is ActionStatus.SUCCESS
    assert result.attempts == 3
    assert tool.validation_calls == 1
    assert tool.permission_calls == 1
    assert tool.execution_calls == 3


class _BlockingPermissionTool(BaseTool):
    def __init__(self, release: threading.Event) -> None:
        super().__init__(
            ToolSpec(
                name="blocking_permission",
                description="block during admission",
                timeout_s=0.02,
                concurrency_safe=True,
            )
        )
        self.release = release
        self.executed = False

    def check_permissions(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> ToolPermissionDecision:
        _ = args, runtime_context
        self.release.wait(timeout=1.0)
        return ToolPermissionDecision.allow()

    def execute(self, args: dict[str, Any], runtime_context: Any = None) -> str:
        _ = args, runtime_context
        self.executed = True
        return "unexpected"


def test_tool_timeout_bounds_permission_admission() -> None:
    release = threading.Event()
    tool = _BlockingPermissionTool(release)
    started = time.monotonic()

    result = ActionExecutor(ToolRegistry().register(tool)).execute(
        [Action(name=tool.name)]
    )[0]
    release.set()

    assert time.monotonic() - started < 0.15
    assert result.status is ActionStatus.TIMED_OUT
    assert result.attempts == 0
    assert result.metadata["timeout_source"] == "tool_spec"
    assert result.metadata["worker_still_running"] is True
    assert tool.executed is False


class _SharedRetryBudgetTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            ToolSpec(
                name="shared_retry_budget",
                description="share one deadline across retries",
                timeout_s=0.025,
                retry_policy=RetryPolicy(
                    max_attempts=3,
                    backoff_factor=0,
                    jitter=False,
                    retryable_exceptions=(RuntimeError,),
                ),
            )
        )
        self.calls = 0

    def execute(self, args: dict[str, Any], runtime_context: Any = None) -> str:
        _ = args, runtime_context
        self.calls += 1
        time.sleep(0.015)
        raise RuntimeError("retry")


def test_tool_retries_share_one_tool_deadline() -> None:
    tool = _SharedRetryBudgetTool()
    started = time.monotonic()

    result = ActionExecutor(ToolRegistry().register(tool)).execute(
        [Action(name=tool.name)]
    )[0]

    assert time.monotonic() - started < 0.1
    assert result.status is ActionStatus.TIMED_OUT
    assert result.metadata["timeout_source"] == "tool_spec"
    assert 1 <= result.attempts < 3
    assert tool.calls == result.attempts


class _CancellationAwareTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            ToolSpec(name="cancel_aware", description="cooperative cancel")
        )
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
    assert result_holder[0].status is ActionStatus.CANCELLED
    assert "action cancelled" in result_holder[0].output


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


class _BlockingModel:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.worker_daemon: bool | None = None

    def __call__(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        _ = messages, kwargs
        self.worker_daemon = threading.current_thread().daemon
        self.entered.set()
        self.release.wait(timeout=1.0)
        return "Final Answer: late result"


@dataclass
class _ModelDeadlineState(StateSchema):
    pass


class _ModelDeadlineAgent(AgentModule[_ModelDeadlineState, dict[str, Any], Any]):
    def __init__(self, model: _BlockingModel) -> None:
        super().__init__(tool_registry=ToolRegistry(), llm=model)

    def init_state(self, task: str, **kwargs: Any) -> _ModelDeadlineState:
        _ = kwargs
        return _ModelDeadlineState(task=task, max_steps=3)

    def reduce(
        self,
        state: _ModelDeadlineState,
        observation: dict[str, Any],
        decision: Decision[Any],
    ) -> _ModelDeadlineState:
        _ = observation, decision
        return state


def test_model_request_deadline_detaches_blocking_provider_and_discards_late_result() -> (
    None
):
    model = _BlockingModel()
    engine = Engine(
        _ModelDeadlineAgent(model),
        budget=RuntimeBudget(
            max_steps=3,
            deadline_monotonic=time.monotonic() + 0.03,
        ),
    )
    started = time.monotonic()

    result = engine.run("block")
    model.release.set()

    assert time.monotonic() - started < 0.2
    assert model.entered.is_set()
    assert model.worker_daemon is True
    assert result.state.stop_reason == "budget_time"
    assert result.state.final_result is None


def test_immediate_cancel_stops_waiting_for_blocking_model() -> None:
    model = _BlockingModel()
    engine = Engine(_ModelDeadlineAgent(model), budget=RuntimeBudget(max_steps=3))
    results: list[Any] = []
    thread = threading.Thread(target=lambda: results.append(engine.run("block")))

    thread.start()
    assert model.entered.wait(timeout=0.2)
    engine.cancel("immediate")
    thread.join(timeout=0.2)
    model.release.set()

    assert not thread.is_alive()
    assert results[0].state.stop_reason == "cancelled_immediate"
    assert results[0].state.final_result is None


class _BlockingStreamModel:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def stream(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        _ = messages, kwargs
        try:
            yield ModelStreamChunk(text="before deadline")
            self.entered.set()
            self.release.wait(timeout=1.0)
            yield ModelStreamChunk(text="late", done=True)
        finally:
            self.finished.set()

    def __call__(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        _ = messages, kwargs
        return "Final Answer: fallback"


class _RecordingStreamHandler:
    def __init__(self) -> None:
        self.events: list[tuple[str, str | None]] = []

    def on_start(self) -> None:
        self.events.append(("start", None))

    def on_delta(self, text: str) -> None:
        self.events.append(("delta", text))

    def on_end(self) -> None:
        self.events.append(("end", None))


def test_model_stream_discards_callbacks_after_deadline() -> None:
    model = _BlockingStreamModel()
    handler = _RecordingStreamHandler()
    engine = Engine(
        _ModelDeadlineAgent(model),
        budget=RuntimeBudget(
            max_steps=3,
            deadline_monotonic=time.monotonic() + 0.05,
        ),
    )
    engine.stream_callback = handler

    result = engine.run("stream")
    model.release.set()
    assert model.finished.wait(timeout=0.2)

    assert result.state.stop_reason == "budget_time"
    assert handler.events == [
        ("start", None),
        ("delta", "before deadline"),
    ]


def test_model_stream_reports_error_without_normal_end() -> None:
    class _BrokenStreamModel:
        def transactional_stream(self, messages: list[dict[str, Any]], **kwargs: Any):
            _ = messages, kwargs
            yield ModelStreamChunk(text="partial")
            raise ModelTransportError(
                "stream broke",
                attempts=1,
                retryable=False,
            )

    class _DetailedStreamHandler(_RecordingStreamHandler):
        def on_chunk(self, chunk: ModelStreamChunk) -> None:
            self.events.append(("chunk", chunk.event_type))

        def on_error(self, exc: Exception) -> None:
            self.events.append(("error", type(exc).__name__))

    handler = _DetailedStreamHandler()
    engine = Engine(
        _ModelDeadlineAgent(_BrokenStreamModel()),
        budget=RuntimeBudget(max_steps=1),
    )
    engine.stream_callback = handler

    result = engine.run("stream")

    assert result.state.stop_reason == "unrecoverable_error"
    assert handler.events == [
        ("start", None),
        ("chunk", None),
        ("delta", "partial"),
        ("error", "ModelTransportError"),
    ]
