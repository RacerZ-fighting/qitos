"""Contract tests for run deadlines and cooperative tool cancellation."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from qitos import AgentModule, Decision, Engine, StateSchema, ToolRegistry
from qitos.core.action import Action, ActionResult, ActionStatus
from qitos.core.errors import ModelRequestDeadlineExceeded, ModelTransportError
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
from qitos.models import ModelRequest
from qitos.models import ModelStreamEventType
from qitos.models.base import Model, ModelStreamEvent


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

    async def apost_runtime_event(self, event: Any, *, run_id: str) -> bool:
        _ = event, run_id
        return True


async def _execute(tool: BaseTool, engine: _RuntimeEngine) -> ActionResult:
    return (await ActionExecutor(
        tool_registry=ToolRegistry().register(tool),
        engine=engine,
    ).execute([Action(name=tool.spec.name)]))[0]


class _RecordingTool(BaseTool):
    def __init__(self, *, delay: float = 0.0) -> None:
        super().__init__(ToolSpec(name="record", description="record runtime context"))
        self.delay = delay
        self.calls = 0
        self.runtime_context: dict[str, Any] | None = None
        self.cleaned = False

    async def execute(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> str:
        _ = args
        self.calls += 1
        self.runtime_context = runtime_context
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            return "ok"
        finally:
            self.cleaned = True


def test_runtime_budget_accepts_an_absolute_monotonic_deadline() -> None:
    deadline = time.monotonic() + 30.0

    budget = RuntimeBudget(max_steps=3, deadline_monotonic=deadline)

    assert budget.deadline_monotonic == deadline


@pytest.mark.asyncio
async def test_expired_deadline_prevents_tool_admission() -> None:
    tool = _RecordingTool()
    engine = _RuntimeEngine(time.monotonic() - 1.0)

    result = await _execute(tool, engine)

    assert tool.calls == 0
    assert result.status is ActionStatus.TIMED_OUT
    assert result.attempts == 0
    assert result.metadata["timeout_source"] == "runtime_deadline"
    assert result.metadata["started"] is False


@pytest.mark.asyncio
async def test_runtime_deadline_clamps_a_longer_tool_timeout() -> None:
    tool = _RecordingTool(delay=0.2)
    engine = _RuntimeEngine(time.monotonic() + 0.03)
    started = time.monotonic()

    result = await _execute(tool, engine)

    assert time.monotonic() - started < 0.15
    assert result.status is ActionStatus.TIMED_OUT
    assert result.metadata["timeout_source"] == "runtime_deadline"
    assert result.metadata["worker_still_running"] is False
    assert tool.cleaned is True


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

    async def execute(
        self, args: dict[str, Any], runtime_context: Any = None
    ) -> None:
        _ = args, runtime_context
        self.calls += 1
        raise RuntimeError("retry me")


@pytest.mark.asyncio
async def test_retry_backoff_does_not_cross_the_runtime_deadline() -> None:
    tool = _RetryingTool()
    engine = _RuntimeEngine(time.monotonic() + 0.01)
    started = time.monotonic()

    result = await _execute(tool, engine)

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

    async def execute(
        self, args: dict[str, Any], runtime_context: Any = None
    ) -> str:
        _ = args, runtime_context
        self.execution_calls += 1
        if self.execution_calls < 3:
            raise RuntimeError("retry")
        return "ok"


@pytest.mark.asyncio
async def test_tool_retry_repeats_only_the_admitted_invocation() -> None:
    tool = _AdmissionCountingTool()
    result = await _execute(tool, _RuntimeEngine(time.monotonic() + 1.0))

    assert result.status is ActionStatus.SUCCESS
    assert result.attempts == 3
    assert tool.validation_calls == 1
    assert tool.permission_calls == 1
    assert tool.execution_calls == 3


class _SlowAdmissionTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            ToolSpec(
                name="slow_admission",
                description="consume the admission budget",
                timeout_s=0.02,
                concurrency_safe=True,
            )
        )
        self.executed = False

    def check_permissions(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> ToolPermissionDecision:
        _ = args, runtime_context
        time.sleep(0.03)
        return ToolPermissionDecision.allow()

    async def execute(
        self, args: dict[str, Any], runtime_context: Any = None
    ) -> str:
        _ = args, runtime_context
        self.executed = True
        return "unexpected"


@pytest.mark.asyncio
async def test_tool_timeout_includes_permission_admission() -> None:
    tool = _SlowAdmissionTool()
    started = time.monotonic()

    result = (
        await ActionExecutor(ToolRegistry().register(tool)).execute(
            [Action(name=tool.name)]
        )
    )[0]

    assert time.monotonic() - started < 0.15
    assert result.status is ActionStatus.TIMED_OUT
    assert result.attempts == 0
    assert result.metadata["timeout_source"] == "tool_spec"
    assert result.metadata["worker_still_running"] is False
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

    async def execute(
        self, args: dict[str, Any], runtime_context: Any = None
    ) -> str:
        _ = args, runtime_context
        self.calls += 1
        await asyncio.sleep(0.015)
        raise RuntimeError("retry")


@pytest.mark.asyncio
async def test_tool_retries_share_one_tool_deadline() -> None:
    tool = _SharedRetryBudgetTool()
    started = time.monotonic()

    result = (
        await ActionExecutor(ToolRegistry().register(tool)).execute(
            [Action(name=tool.name)]
        )
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
        self.entered = asyncio.Event()
        self.observed_cancel = False
        self.deadline_monotonic: float | None = None
        self.remaining_seconds: Any = None

    async def execute(
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
            await asyncio.sleep(0)
        return "missed cancellation"


@pytest.mark.asyncio
async def test_tool_runtime_context_exposes_live_deadline_and_cancellation() -> None:
    tool = _CancellationAwareTool()
    engine = _RuntimeEngine(time.monotonic() + 1.0)
    execution = asyncio.create_task(_execute(tool, engine))
    await asyncio.wait_for(tool.entered.wait(), timeout=0.2)
    engine._cancel_token.request_cancel("immediate")
    result = await asyncio.wait_for(execution, timeout=0.5)

    assert tool.observed_cancel is True
    assert tool.deadline_monotonic == engine.runtime_deadline_monotonic
    assert callable(tool.remaining_seconds)
    assert result.status is ActionStatus.CANCELLED
    assert "action cancelled" in result.output


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


class _BlockingModel(Model):
    def __init__(self) -> None:
        super().__init__(model="blocking-test-model", temperature=None)
        self.entered = asyncio.Event()
        self.closed = asyncio.Event()

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        _ = request
        self.entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.closed.set()
        if False:  # pragma: no cover - preserve async-generator typing
            yield ModelStreamEvent(
                type=ModelStreamEventType.LIFECYCLE,
                event_type="unreachable",
            )


@dataclass
class _ModelDeadlineState(StateSchema):
    pass


class _ModelDeadlineAgent(AgentModule[_ModelDeadlineState, dict[str, Any], Any]):
    def __init__(self, model: Model) -> None:
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


@pytest.mark.asyncio
async def test_model_request_deadline_cancels_provider_and_closes_stream() -> None:
    model = _BlockingModel()
    engine = Engine(_ModelDeadlineAgent(model), budget=RuntimeBudget(max_steps=3))
    engine._runtime_deadline_monotonic = time.monotonic() + 0.03

    with pytest.raises(ModelRequestDeadlineExceeded):
        await asyncio.wait_for(
            engine._model_runtime._call_llm(
                model,
                ModelRequest(
                    run_id="deadline-run",
                    transaction_id="deadline-run:0",
                    provider=model.provider_name,
                    model=model.model,
                    protocol=model.capabilities.api.value,
                    messages=(),
                    deadline_monotonic=engine.runtime_deadline_monotonic,
                ),
            ),
            timeout=1.0,
        )

    assert model.entered.is_set()
    assert model.closed.is_set()


@pytest.mark.asyncio
async def test_immediate_cancel_stops_waiting_for_async_model() -> None:
    model = _BlockingModel()
    engine = Engine(_ModelDeadlineAgent(model), budget=RuntimeBudget(max_steps=3))
    run_task = asyncio.create_task(engine.arun("block"))

    await asyncio.wait_for(model.entered.wait(), timeout=0.2)
    engine.cancel("immediate")

    result = await run_task
    assert result.state.stop_reason == "cancelled_immediate"
    assert model.closed.is_set()


class _BlockingStreamModel(Model):
    def __init__(self) -> None:
        super().__init__(model="blocking-stream-model", temperature=None)
        self.entered = asyncio.Event()
        self.finished = asyncio.Event()

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        _ = request
        try:
            yield ModelStreamEvent(
                type=ModelStreamEventType.TEXT_DELTA,
                text="before deadline",
            )
            self.entered.set()
            await asyncio.Event().wait()
        finally:
            self.finished.set()


class _RecordingStreamHandler:
    def __init__(self) -> None:
        self.events: list[tuple[str, str | None]] = []

    def on_start(self) -> None:
        self.events.append(("start", None))

    def on_delta(self, text: str) -> None:
        self.events.append(("delta", text))

    def on_end(self) -> None:
        self.events.append(("end", None))


@pytest.mark.asyncio
async def test_model_stream_stops_callbacks_after_deadline() -> None:
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

    result = await engine.arun("stream")
    await asyncio.wait_for(model.finished.wait(), timeout=0.2)

    assert result.state.stop_reason == "budget_time"
    assert handler.events == [
        ("start", None),
        ("delta", "before deadline"),
    ]


def test_model_stream_reports_error_without_normal_end() -> None:
    class _BrokenStreamModel(Model):
        def __init__(self) -> None:
            super().__init__(model="broken-stream-model", temperature=None)

        async def stream(
            self,
            request: ModelRequest,
        ) -> AsyncIterator[ModelStreamEvent]:
            _ = request
            yield ModelStreamEvent(
                type=ModelStreamEventType.TEXT_DELTA,
                text="partial",
            )
            raise ModelTransportError(
                "stream broke",
                attempts=1,
                retryable=False,
            )

    class _DetailedStreamHandler(_RecordingStreamHandler):
        def on_chunk(self, chunk: ModelStreamEvent) -> None:
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


def test_model_failed_terminal_reports_error_without_normal_end() -> None:
    class _FailedStreamModel(Model):
        def __init__(self) -> None:
            super().__init__(model="failed-stream-model", temperature=None)

        async def stream(
            self,
            request: ModelRequest,
        ) -> AsyncIterator[ModelStreamEvent]:
            _ = request
            yield ModelStreamEvent(
                type=ModelStreamEventType.FAILED,
                event_type="provider.failed",
                error="provider rejected the transaction",
            )

    class _DetailedStreamHandler(_RecordingStreamHandler):
        def on_chunk(self, chunk: ModelStreamEvent) -> None:
            self.events.append(("chunk", chunk.event_type))

        def on_error(self, exc: Exception) -> None:
            self.events.append(("error", type(exc).__name__))

    handler = _DetailedStreamHandler()
    engine = Engine(
        _ModelDeadlineAgent(_FailedStreamModel()),
        budget=RuntimeBudget(max_steps=1),
    )
    engine.stream_callback = handler

    result = engine.run("stream")

    assert result.state.stop_reason == "unrecoverable_error"
    assert handler.events == [
        ("start", None),
        ("chunk", "provider.failed"),
        ("error", "ModelTransportError"),
    ]
