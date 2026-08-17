"""Child engine driven by the minimal-loop Agent façade.

A child is the same ``Agent`` implementation as its parent, constructed with
narrowed tools and budgets. This module provides the ``ChildEngine``
implementation a ``ChildSupervisor`` drives plus a convenience invocation
factory that wires one child run to a ``JournalTurnTransaction`` journal.

Failure semantics: the façade returns typed ``AgentLoopResult`` values for
expected terminal outcomes (completion, abort, model failure, budget) and
raises only for implementation faults; ``arun`` maps the typed outcome onto
the ``ChildRunResult`` read view and lets faults propagate to the supervisor's
failure path. ``cancel`` is cooperative: the run always ends in a durable
terminal journal record before the engine reports a result.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple, Union

from ...core.agent import Agent, AgentRunRejected
from ...core.agent_loop import AgentLoopResult, AgentRunStatus
from ...core.child import (
    ChildInvocation,
    ChildLaunchRequest,
    ChildRuntimeContext,
)
from ...core.env import Env
from ...core.journal import JournalRecordType, SessionJournal
from ...core.message import AssistantMessage, Message, UserMessage
from ...core.runtime_input import RuntimeInput
from ...core.tool_registry import ToolRegistry
from ..journal import JournalTurnTransaction, JsonlSessionJournal

if TYPE_CHECKING:
    from ...models.base import Model


#: Stop reasons the child supervisor classifies into its stable ChildStatus
#: vocabulary (``ChildSupervisor._child_status``).
_STOP_REASON_BY_STATUS = {
    AgentRunStatus.COMPLETED: "completed",
    AgentRunStatus.ABORTED: "cancelled",
    AgentRunStatus.MAX_TURNS: "max_steps",
    AgentRunStatus.DEADLINE_EXCEEDED: "budget_time",
}


def child_stop_reason(status: AgentRunStatus, error: Optional[str]) -> str:
    """Map one loop run status onto the supervisor's stop-reason vocabulary."""

    reason = _STOP_REASON_BY_STATUS.get(status)
    if reason is not None:
        return reason
    return f"failed:{error}" if error else "failed"


def child_final_text(messages: Sequence[Message]) -> str:
    """Return the last non-failed assistant text in one run's messages."""

    for message in reversed(messages):
        if isinstance(message, AssistantMessage) and not message.error:
            text = message.text.strip()
            if text:
                return text
    return ""


@dataclass(frozen=True, slots=True)
class ChildRunStats:
    """Step and token aggregation over one run's committed messages."""

    steps: int
    total_tokens: int
    usage_complete: bool


def child_run_stats(messages: Sequence[Message]) -> ChildRunStats:
    """Aggregate turn count and token usage from assistant messages.

    Cost is not part of the canonical message usage, so cost stays unknown
    to the caller; messages without usage make the aggregate incomplete
    instead of inventing numbers.
    """

    steps = 0
    total_tokens = 0
    usage_complete = True
    saw_assistant = False
    for message in messages:
        if not isinstance(message, AssistantMessage):
            continue
        saw_assistant = True
        steps += 1
        usage = message.usage
        tokens: Optional[int] = None
        if usage is not None:
            if usage.total_tokens is not None:
                tokens = usage.total_tokens
            elif (
                usage.input_tokens is not None
                or usage.output_tokens is not None
            ):
                tokens = (usage.input_tokens or 0) + (usage.output_tokens or 0)
        if tokens is None:
            usage_complete = False
        else:
            total_tokens += tokens
    return ChildRunStats(
        steps=steps,
        total_tokens=total_tokens,
        usage_complete=usage_complete and saw_assistant,
    )


class _AgentChildStateView:
    """Terminal state read view over one façade run outcome."""

    def __init__(self, *, final_result: str, stop_reason: str) -> None:
        self._final_result = final_result
        self._stop_reason = stop_reason

    @property
    def final_result(self) -> Any:
        return self._final_result

    @property
    def stop_reason(self) -> Any:
        return self._stop_reason


class AgentChildRunResult:
    """``ChildRunResult`` projection of one terminal façade run."""

    def __init__(self, *, run_id: str, result: AgentLoopResult) -> None:
        self.run_id = run_id
        self._result = result
        stats = child_run_stats(result.messages)
        self.step_count = stats.steps
        self.total_tokens = stats.total_tokens
        self.total_cost_usd = 0.0
        self.local_total_tokens = stats.total_tokens
        self.local_total_cost_usd = 0.0
        self.usage_complete = stats.usage_complete
        self.cost_complete = False
        self.local_usage_complete = stats.usage_complete
        self.local_cost_complete = False
        self._state = _AgentChildStateView(
            final_result=child_final_text(result.messages),
            stop_reason=child_stop_reason(result.status, result.error),
        )

    @property
    def state(self) -> _AgentChildStateView:
        return self._state

    @property
    def messages(self) -> Tuple[Message, ...]:
        return tuple(self._result.messages)

    @property
    def records(self) -> Sequence[Any]:
        # Legacy step-record consumers were removed with the Engine path;
        # committed Tool evidence is projected from ``messages`` instead.
        return ()


class AgentChildEngine:
    """One-shot ``ChildEngine`` running a child through the Agent façade.

    The engine owns its run journal: ``arun`` creates the journal under the
    supervisor-assigned Run id before the first model side effect and every
    turn commits through ``JournalTurnTransaction``. ``cancel`` aborts the
    active run cooperatively; ``aclose`` aborts, waits for settlement and
    closes the journal, and is idempotent.
    """

    def __init__(
        self,
        *,
        model: "Model",
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: str = "",
        env: Optional[Env] = None,
        tool_execution: Literal["sequential", "parallel"] = "sequential",
        max_tool_concurrency: int = 8,
        max_turns: Optional[int] = None,
        run_timeout_s: Optional[float] = None,
        extra_request_options: Optional[Mapping[str, Any]] = None,
        runtime_context: Optional[Mapping[str, Any]] = None,
        journal_factory: Optional[Callable[[], SessionJournal]] = None,
        journal_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._model = model
        self._tool_registry = tool_registry
        self._system_prompt = system_prompt
        self._env = env
        self._tool_execution = tool_execution
        self._max_tool_concurrency = max_tool_concurrency
        self._max_turns = max_turns
        self._run_timeout_s = run_timeout_s
        self._extra_request_options = dict(extra_request_options or {})
        self._runtime_context = dict(runtime_context or {})
        self._journal_factory = journal_factory
        self._journal_metadata = dict(journal_metadata or {})

        self._agent: Optional[Agent] = None
        self._journal: Optional[SessionJournal] = None
        self._run_id = ""
        self._cancel_requested = False
        self._pending_events: list[tuple[RuntimeInput, str]] = []
        self._closed = False
        self._result: Optional[AgentLoopResult] = None

    @property
    def active_run_id(self) -> str:
        return self._run_id

    @property
    def messages(self) -> Tuple[Message, ...]:
        agent = self._agent
        if agent is not None:
            return agent.messages
        if self._result is not None:
            return tuple(self._result.messages)
        return ()

    @property
    def step_count(self) -> int:
        return child_run_stats(self.messages).steps

    @property
    def token_usage(self) -> int:
        return child_run_stats(self.messages).total_tokens

    @property
    def cost_usage_usd(self) -> float:
        return 0.0

    @property
    def usage_complete(self) -> bool:
        return child_run_stats(self.messages).usage_complete

    @property
    def cost_complete(self) -> bool:
        return False

    async def arun(self, task: str, **kwargs: Any) -> AgentChildRunResult:
        """Run the child task once under the supervisor-assigned Run id."""

        run_id = kwargs.pop("run_id", None)
        if kwargs:
            raise ValueError(f"unsupported child run arguments: {sorted(kwargs)}")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("arun requires the supervisor-assigned run_id")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("arun requires a non-empty task")
        if self._closed:
            raise RuntimeError("child engine is closed")
        if self._agent is not None:
            raise RuntimeError("a child engine runs at most once")
        run_id = run_id.strip()
        self._run_id = run_id

        transaction: Optional[JournalTurnTransaction] = None
        if self._journal_factory is not None:
            journal = self._journal_factory()
            if not isinstance(journal, SessionJournal):
                raise TypeError("journal_factory must return a SessionJournal")
            await journal.create(run_id, dict(self._journal_metadata))
            self._journal = journal
            transaction = JournalTurnTransaction(journal)

        runtime_context = dict(self._runtime_context)
        if self._journal is not None:
            runtime_context.setdefault("journal", self._journal)
        agent = Agent(
            model=self._model,
            tool_registry=self._tool_registry,
            system_prompt=self._system_prompt,
            env=self._env,
            tool_execution=self._tool_execution,
            max_tool_concurrency=self._max_tool_concurrency,
            max_turns=self._max_turns,
            run_timeout_s=self._run_timeout_s,
            extra_request_options=self._extra_request_options,
            runtime_context=runtime_context,
            transaction_factory=lambda _run_id: transaction,
            run_id_factory=lambda: run_id,
        )
        self._agent = agent

        def _abort_if_cancelled(_event: Any) -> None:
            if self._cancel_requested:
                agent.abort()

        unsubscribe = agent.subscribe(_abort_if_cancelled)
        pending, self._pending_events = self._pending_events, []
        for event, text in pending:
            if transaction is not None:
                await transaction.journal.append(
                    JournalRecordType.RUNTIME_INPUT_POSTED,
                    event.to_dict(),
                    record_id=f"{run_id}:runtime:{event.event_id}",
                )
            agent.steer(UserMessage(content=text))
        try:
            outcome = await agent.prompt(task)
        finally:
            unsubscribe()
        if isinstance(outcome, AgentRunRejected):
            raise RuntimeError(
                f"a fresh child Agent rejected its only run: {outcome.reason}"
            )
        self._result = outcome
        return AgentChildRunResult(run_id=run_id, result=outcome)

    def cancel(self, mode: str) -> None:
        """Cooperatively abort the active run; every mode maps to abort."""

        _ = mode
        self._cancel_requested = True
        if self._agent is not None:
            self._agent.abort()

    async def apost_runtime_event(
        self,
        event: RuntimeInput,
        *,
        run_id: Optional[str] = None,
    ) -> bool:
        """Accept one parent message into the child run's steering queue.

        Acceptance is durable when the engine keeps a journal; the message
        itself becomes visible to the child at the next turn safe point. A
        message posted while the child is still starting is buffered and
        drained into the run's initial steering poll.
        """

        if not isinstance(event, RuntimeInput):
            raise TypeError("event must be a RuntimeInput")
        if run_id is not None and run_id != self._run_id:
            return False
        if self._closed or self._result is not None:
            return False
        text = str(event.payload.get("content") or "").strip()
        if not text:
            return False
        agent = self._agent
        if agent is None:
            # The run journal opens inside arun; buffer so acceptance stays
            # durable and the message reaches the initial steering poll.
            self._pending_events.append((event, text))
            return True
        if self._journal is not None:
            await self._journal.append(
                JournalRecordType.RUNTIME_INPUT_POSTED,
                event.to_dict(),
                record_id=f"{self._run_id}:runtime:{event.event_id}",
            )
        agent.steer(UserMessage(content=text))
        return True

    async def aclose(self) -> None:
        """Abort any active run, wait for settlement and close the journal."""

        if self._closed:
            return
        self._closed = True
        agent = self._agent
        if agent is not None:
            agent.abort()
            await agent.wait_for_idle()
        if self._journal is not None:
            await self._journal.close()


def _narrow_tool_registry(
    base: ToolRegistry,
    allowed_groups: Tuple[str, ...],
) -> ToolRegistry:
    """Copy ``base`` into a fresh registry narrowed to allowed Tool groups."""

    registry = ToolRegistry()
    for name in base.list_tools():
        tool = base.get(name)
        if tool is None:
            continue
        if allowed_groups and tool.spec.group not in allowed_groups:
            continue
        registry.register(tool)
    return registry


def build_agent_child_invocation_factory(
    *,
    model: Union["Model", Callable[[], "Model"]],
    tool_registry: Optional[ToolRegistry] = None,
    system_prompt: str = "",
    env: Optional[Env] = None,
    tool_execution: Literal["sequential", "parallel"] = "sequential",
    max_tool_concurrency: int = 8,
    max_turns: Optional[int] = None,
    run_timeout_s: Optional[float] = None,
    extra_request_options: Optional[Mapping[str, Any]] = None,
    journal_directory: Union[str, Path, None] = None,
    journal_factory: Optional[Callable[[], SessionJournal]] = None,
) -> Callable[[ChildLaunchRequest, ChildRuntimeContext], Awaitable[ChildInvocation]]:
    """Build a ``ChildInvocationFactory`` driving children through ``Agent``.

    Every invocation constructs a fresh ``AgentChildEngine`` whose authority
    only narrows the parent's: the Tool registry is copied and filtered to the
    request's allowed groups, and turn/runtime budgets are the tightest of the
    factory defaults, the request budget and the parent's remaining deadline.
    ``journal_directory`` (or an explicit ``journal_factory``) gives each
    child run its own ``JournalTurnTransaction`` journal.
    """

    if journal_directory is not None and journal_factory is not None:
        raise ValueError("journal_directory and journal_factory are exclusive")
    if journal_directory is not None:
        journal_root = Path(journal_directory)

        def journal_factory() -> SessionJournal:  # noqa: F811
            return JsonlSessionJournal(journal_root)

    async def _factory(
        request: ChildLaunchRequest,
        runtime_context: ChildRuntimeContext,
    ) -> ChildInvocation:
        resolved_model = model() if callable(model) else model
        budget = request.budget
        resolved_max_turns = _tightest_int(max_turns, budget.max_steps)
        remaining: Optional[float] = None
        deadline = runtime_context.deadline_monotonic
        if deadline is not None:
            remaining = max(0.0, deadline - time.monotonic())
        resolved_timeout = _tightest_float(
            _tightest_float(run_timeout_s, budget.max_runtime_seconds),
            remaining,
        )
        resolved_concurrency = _tightest_int(
            max_tool_concurrency, budget.max_tool_concurrency
        )
        engine = AgentChildEngine(
            model=resolved_model,
            tool_registry=(
                _narrow_tool_registry(tool_registry, request.allowed_tool_groups)
                if tool_registry is not None
                else None
            ),
            system_prompt=system_prompt,
            env=env,
            tool_execution=tool_execution,
            max_tool_concurrency=resolved_concurrency or 1,
            max_turns=resolved_max_turns,
            run_timeout_s=resolved_timeout,
            extra_request_options=extra_request_options,
            runtime_context={
                "parent_run_id": runtime_context.child_run_id,
                "delegate_depth": runtime_context.delegate_depth + 1,
                "deadline_monotonic": runtime_context.deadline_monotonic,
                "budget_ledger": runtime_context.budget_ledger,
            },
            journal_factory=journal_factory,
            journal_metadata={
                "parent_run_id": runtime_context.parent_run_id,
                "child_id": runtime_context.handle.child_id,
                "agent_type": request.agent_type,
                "description": request.description,
            },
        )
        task = request.task
        if request.context.strip():
            task = f"{request.task}\n\nContext:\n{request.context.strip()}"
        return ChildInvocation(engine=engine, task=task)

    return _factory


def _tightest_int(*values: Optional[int]) -> Optional[int]:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def _tightest_float(*values: Optional[float]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return min(present) if present else None


__all__ = [
    "AgentChildEngine",
    "AgentChildRunResult",
    "build_agent_child_invocation_factory",
    "child_final_text",
    "child_run_stats",
    "child_stop_reason",
    "ChildRunStats",
]
