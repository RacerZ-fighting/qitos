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

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple, Union

from ...core.agent import Agent, AgentRunRejected
from ...core.agent_events import AgentEnd, MessageEnd, TurnEnd, TurnStart
from ...core.agent_loop import (
    AgentLoopResult,
    AgentRunStatus,
    TurnConfigSnapshot,
    TurnHookContext,
    TurnTransactionBoundary,
)
from ...core.budget import BudgetLedger, BudgetSnapshot
from ...core.child import (
    ChildInvocation,
    ChildLaunchRequest,
    ChildRuntimeContext,
)
from ...core.env import Env
from ...core.journal import (
    JournalAppendCancelled,
    JournalCommitError,
    JournalCommitState,
    JournalError,
    JournalRecordType,
    SessionJournal,
)
from ...core.message import AssistantMessage, Message, ToolCall, UserMessage
from ...core.model_request import ModelRequest
from ...core.model_response import ModelPricing, ModelUsage
from ...core.runtime_input import RuntimeInput
from ...core.task import TaskBudget
from ...core.tool import ToolPermissionContext
from ...core.tool_executor import BeforeToolCallContext, BeforeToolCallDecision
from ...core.tool_registry import ToolExposure, ToolRegistry
from ...core.tool_result import ToolResult
from ..journal import JournalTurnTransaction, JsonlSessionJournal
from ..journal.turn_recorder import encode_runtime_input_consumed

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
    """Usage aggregation over one run's committed messages."""

    steps: int
    total_tokens: int
    total_cost_usd: float
    usage_complete: bool
    cost_complete: bool


def child_run_stats(
    messages: Sequence[Message],
    *,
    model_pricing: ModelPricing | None = None,
) -> ChildRunStats:
    """Aggregate complete token and explicitly-priced cost usage."""

    steps = 0
    total_tokens = 0
    total_cost_usd = 0.0
    usage_complete = True
    cost_complete = True
    saw_assistant = False
    for message in messages:
        if not isinstance(message, AssistantMessage):
            continue
        saw_assistant = True
        steps += 1
        tokens, cost, tokens_known, cost_known = _usage_values(
            message.usage,
            model_pricing=model_pricing,
        )
        total_tokens += tokens
        total_cost_usd += cost
        usage_complete = usage_complete and tokens_known
        cost_complete = cost_complete and cost_known
    return ChildRunStats(
        steps=steps,
        total_tokens=total_tokens,
        total_cost_usd=total_cost_usd,
        usage_complete=usage_complete and saw_assistant,
        cost_complete=cost_complete and saw_assistant,
    )


def _usage_values(
    usage: ModelUsage | None,
    *,
    model_pricing: ModelPricing | None,
) -> tuple[int, float, bool, bool]:
    if usage is None:
        return (0, 0.0, False, False)
    if usage.total_tokens is not None:
        tokens = usage.total_tokens
        tokens_known = True
    elif usage.input_tokens is not None and usage.output_tokens is not None:
        tokens = usage.input_tokens + usage.output_tokens
        tokens_known = True
    else:
        tokens = 0
        tokens_known = False
    cost_known = (
        model_pricing is not None
        and usage.input_tokens is not None
        and usage.output_tokens is not None
    )
    cost = model_pricing.cost_usd(usage) if cost_known and model_pricing else 0.0
    return (tokens, cost, tokens_known, cost_known)


def child_budget_stop_reason(
    stats: ChildRunStats,
    budget: TaskBudget,
    root_budget: BudgetSnapshot | None = None,
) -> str | None:
    if budget.max_tokens is not None:
        if not stats.usage_complete:
            return "budget_tokens_unknown"
        if stats.total_tokens >= budget.max_tokens:
            return "budget_tokens"
    if budget.max_cost_usd is not None:
        if not stats.cost_complete:
            return "budget_cost_unknown"
        if stats.total_cost_usd >= budget.max_cost_usd:
            return "budget_cost"
    if root_budget is not None:
        if root_budget.max_tokens is not None:
            if not root_budget.usage_complete:
                return "budget_tokens_unknown"
            if root_budget.tokens_exhausted:
                return "budget_tokens"
        if root_budget.max_cost_usd is not None:
            if not root_budget.cost_complete:
                return "budget_cost_unknown"
            if root_budget.cost_exhausted:
                return "budget_cost"
    return None


class _ChildTurnTransaction(TurnTransactionBoundary):
    """Add Child budget accounting to the loop's durable barriers."""

    def __init__(
        self,
        *,
        run_id: str,
        delegate: TurnTransactionBoundary | None,
        budget: TaskBudget,
        budget_ledger: BudgetLedger | None,
        model_pricing: ModelPricing | None,
        settle_runtime_events: Callable[[], Awaitable[None]],
    ) -> None:
        self._run_id = run_id
        self._delegate = delegate
        self._budget = budget
        self._budget_ledger = budget_ledger
        self._model_pricing = model_pricing
        self._settle_runtime_events = settle_runtime_events
        self._model_messages: dict[int, AssistantMessage] = {}
        self._budget_stop_reason: str | None = None

    @property
    def budget_stop_reason(self) -> str | None:
        return self._budget_stop_reason

    async def model_terminal(
        self,
        turn: int,
        request: ModelRequest,
        message: AssistantMessage,
    ) -> None:
        previous = self._model_messages.get(turn)
        if previous is not None and previous != message:
            raise RuntimeError("Child model transaction conflicts with its turn")
        if previous is not None:
            return
        root_budget: BudgetSnapshot | None = None
        usage = child_run_stats((message,), model_pricing=self._model_pricing)
        if self._budget_ledger is not None:
            root_budget = await self._budget_ledger.commit(
                origin_run_id=self._run_id,
                transaction_id=f"{self._run_id}:turn:{turn}:model",
                tokens=usage.total_tokens,
                cost_usd=usage.total_cost_usd,
                usage_complete=usage.usage_complete,
                cost_complete=usage.cost_complete,
            )
        self._model_messages[turn] = message
        reason = child_budget_stop_reason(
            child_run_stats(
                tuple(self._model_messages[index] for index in sorted(self._model_messages)),
                model_pricing=self._model_pricing,
            ),
            self._budget,
            root_budget,
        )
        if not message.failed and self._budget_stop_reason is None:
            self._budget_stop_reason = reason
        if self._delegate is not None:
            await self._delegate.model_terminal(turn, request, message)

    async def input_accepted(self, prompts: tuple[Message, ...]) -> None:
        if self._delegate is not None:
            await self._delegate.input_accepted(prompts)

    async def turn_frozen(self, turn: int, config: TurnConfigSnapshot) -> None:
        if self._delegate is not None:
            await self._delegate.turn_frozen(turn, config)

    async def tool_started(self, turn: int, call: ToolCall) -> None:
        if self._delegate is not None:
            await self._delegate.tool_started(turn, call)

    async def tool_terminal(
        self,
        turn: int,
        call: ToolCall,
        result: ToolResult,
    ) -> None:
        if self._delegate is not None:
            await self._delegate.tool_terminal(turn, call, result)

    async def turn_committed(
        self,
        turn: int,
        new_messages: tuple[Message, ...],
    ) -> None:
        if self._delegate is not None:
            await self._delegate.turn_committed(turn, new_messages)

    async def run_terminal(self, result: AgentLoopResult) -> None:
        # External cancellation can skip TurnEnd, while the journal terminal
        # barrier still has to follow every parent message reserved during the
        # open turn.
        await self._settle_runtime_events()
        if self._delegate is not None:
            await self._delegate.run_terminal(result)


@dataclass(slots=True)
class _RuntimeEventReservation:
    """One ordered parent message awaiting its turn-boundary decision."""

    event: RuntimeInput
    message: UserMessage
    result: asyncio.Future[bool]
    phase: Literal[
        "pending",
        "appending",
        "committed",
        "rejected",
        "unknown",
        "withdrawn",
    ] = "pending"
    withdrawal_requested: asyncio.Event = dataclass_field(
        default_factory=asyncio.Event
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

    def __init__(
        self,
        *,
        run_id: str,
        result: AgentLoopResult,
        model_pricing: ModelPricing | None = None,
        stop_reason: str | None = None,
    ) -> None:
        self.run_id = run_id
        self._result = result
        stats = child_run_stats(result.messages, model_pricing=model_pricing)
        self.step_count = stats.steps
        self.total_tokens = stats.total_tokens
        self.total_cost_usd = stats.total_cost_usd
        self.local_total_tokens = stats.total_tokens
        self.local_total_cost_usd = stats.total_cost_usd
        self.usage_complete = stats.usage_complete
        self.cost_complete = stats.cost_complete
        self.local_usage_complete = stats.usage_complete
        self.local_cost_complete = stats.cost_complete
        self._state = _AgentChildStateView(
            final_result=child_final_text(result.messages),
            stop_reason=stop_reason or child_stop_reason(result.status, result.error),
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
        budget: Optional[TaskBudget] = None,
        budget_ledger: Optional[BudgetLedger] = None,
        model_pricing: Optional[ModelPricing] = None,
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
        self._budget = budget or TaskBudget()
        if not isinstance(self._budget, TaskBudget):
            raise TypeError("budget must be a TaskBudget or None")
        if budget_ledger is not None and not isinstance(budget_ledger, BudgetLedger):
            raise TypeError("budget_ledger must be a BudgetLedger or None")
        if model_pricing is not None and not isinstance(model_pricing, ModelPricing):
            raise TypeError("model_pricing must be a ModelPricing or None")
        self._budget_ledger = budget_ledger
        self._model_pricing = model_pricing
        self._extra_request_options = dict(extra_request_options or {})
        self._runtime_context = dict(runtime_context or {})
        self._journal_factory = journal_factory
        self._journal_metadata = dict(journal_metadata or {})

        self._agent: Optional[Agent] = None
        self._journal: Optional[SessionJournal] = None
        self._run_id = ""
        self._cancel_requested = False
        self._runtime_started = asyncio.Event()
        self._accepting_runtime_events = False
        self._runtime_event_reservations: list[_RuntimeEventReservation] = []
        self._runtime_input_messages: dict[int, str] = {}
        self._runtime_input_injected: list[str] = []
        self._runtime_input_consumed: set[str] = set()
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
        return self._stats().steps

    @property
    def token_usage(self) -> int:
        return self._stats().total_tokens

    @property
    def cost_usage_usd(self) -> float:
        return self._stats().total_cost_usd

    @property
    def usage_complete(self) -> bool:
        return self._stats().usage_complete

    @property
    def cost_complete(self) -> bool:
        return self._stats().cost_complete

    def _stats(self) -> ChildRunStats:
        return child_run_stats(self.messages, model_pricing=self._model_pricing)

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

        journal_transaction: Optional[JournalTurnTransaction] = None
        if self._journal_factory is not None:
            journal = self._journal_factory()
            if not isinstance(journal, SessionJournal):
                raise TypeError("journal_factory must return a SessionJournal")
            await journal.create(run_id, dict(self._journal_metadata))
            self._journal = journal
            journal_transaction = JournalTurnTransaction(journal)
        transaction = _ChildTurnTransaction(
            run_id=run_id,
            delegate=journal_transaction,
            budget=self._budget,
            budget_ledger=self._budget_ledger,
            model_pricing=self._model_pricing,
            settle_runtime_events=self._reject_runtime_event_admissions,
        )

        runtime_context = dict(self._runtime_context)
        if self._journal is not None:
            runtime_context.setdefault("journal", self._journal)

        async def _post_descendant_event(event: RuntimeInput) -> bool:
            return await self.apost_runtime_event(event, run_id=run_id)

        # A recursively launched background Child reports to this Child's
        # active mailbox, not directly to the Root. Products may provide a
        # stricter durable callback explicitly; otherwise the façade engine is
        # the natural parent-run endpoint.
        runtime_context.setdefault("post_runtime_event", _post_descendant_event)

        def _block_tools_after_budget(
            _context: BeforeToolCallContext,
        ) -> BeforeToolCallDecision | None:
            if transaction.budget_stop_reason is None:
                return None
            return BeforeToolCallDecision(
                block=True,
                reason="Child budget was exhausted by the model transaction.",
                terminate=True,
            )

        def _stop_after_budget(_context: TurnHookContext) -> bool:
            return transaction.budget_stop_reason is not None

        async def _prepare_child_next_turn(context: TurnHookContext) -> None:
            can_continue = (
                not self._cancel_requested
                and transaction.budget_stop_reason is None
                and (
                    self._max_turns is None
                    or context.turn + 1 < self._max_turns
                )
            )
            await self._settle_runtime_event_admissions(
                accept=can_continue,
                agent=agent,
            )

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
            before_tool_call=_block_tools_after_budget,
            should_stop_after_turn=_stop_after_budget,
            prepare_next_turn=_prepare_child_next_turn,
        )
        self._agent = agent

        async def _observe_run(event: Any) -> None:
            if isinstance(event, TurnStart):
                self._accepting_runtime_events = True
                self._runtime_started.set()
            elif isinstance(event, MessageEnd):
                event_id = self._runtime_input_messages.pop(
                    id(event.message), None
                )
                if (
                    event_id is not None
                    and event_id not in self._runtime_input_consumed
                ):
                    self._runtime_input_injected.append(event_id)
            elif isinstance(event, TurnEnd):
                # Close admission before observing the reservations. No await
                # separates a post's open-state check from its reservation, so
                # later posts fail. The existing prepare-next-turn hook then
                # decides and settles reservations immediately before the
                # loop's stop checks and steering drain.
                self._close_runtime_event_admission()
                # TurnEnd follows the turn's durable commit: every steered
                # input the turn injected is now covered by a step.committed,
                # so its consumption becomes durable exactly once.
                await self._consume_injected_runtime_inputs()
            elif isinstance(event, AgentEnd):
                await self._reject_runtime_event_admissions()
            if self._cancel_requested:
                agent.abort()

        unsubscribe_observer = agent.subscribe(_observe_run)
        try:
            outcome = await agent.prompt(task)
        finally:
            await self._reject_runtime_event_admissions()
            self._runtime_started.set()
            unsubscribe_observer()
        if isinstance(outcome, AgentRunRejected):
            raise RuntimeError(
                f"a fresh child Agent rejected its only run: {outcome.reason}"
            )
        self._result = outcome
        return AgentChildRunResult(
            run_id=run_id,
            result=outcome,
            model_pricing=self._model_pricing,
            stop_reason=transaction.budget_stop_reason,
        )

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

        Acceptance is durable when the engine keeps a journal and means the
        message was queued for the next turn safe point. Starting, between-turn
        and terminal-settlement windows reject. A post made during an open turn
        first reserves ordered admission; the existing prepare-next-turn hook
        accepts it only when the one-shot Child can continue, and settles the
        journal plus steering queue before the loop drains that queue. Terminal
        turns reject without writing a runtime-input record. Once the steered
        message enters the committed transcript (its ``MessageEnd`` followed by
        the turn's commit), the engine appends the idempotent
        ``runtime_input.consumed`` record; a run that ends without injecting
        the message consumes nothing.
        """

        if not isinstance(event, RuntimeInput):
            raise TypeError("event must be a RuntimeInput")
        if run_id is not None and run_id != self._run_id:
            return False
        if not self._run_id or self._closed or self._result is not None:
            return False
        text = str(event.payload.get("content") or "").strip()
        if not text:
            return False
        await self._runtime_started.wait()
        agent = self._agent
        if agent is None or not self._accepting_runtime_events:
            return False
        outcome: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        reservation = _RuntimeEventReservation(
            event=event,
            message=UserMessage(content=text),
            result=outcome,
        )
        self._runtime_event_reservations.append(reservation)
        try:
            return await asyncio.shield(outcome)
        except asyncio.CancelledError as cancellation:
            # Pending withdrawal wins without I/O. Once TurnEnd atomically
            # moves the reservation to appending, durable settlement wins the
            # race: wait_for observes True if the record committed, while a
            # rolled-back append preserves the caller's cancellation.
            if reservation.phase == "pending":
                reservation.phase = "withdrawn"
                try:
                    self._runtime_event_reservations.remove(reservation)
                except ValueError:
                    pass
                if not outcome.done():
                    outcome.set_result(False)
                raise
            reservation.withdrawal_requested.set()
            while not outcome.done():
                try:
                    await asyncio.shield(outcome)
                except asyncio.CancelledError:
                    continue
            accepted = outcome.result()
            if accepted:
                return True
            raise cancellation

    def _close_runtime_event_admission(self) -> None:
        """Prevent reservations after the current TurnEnd boundary."""

        self._accepting_runtime_events = False

    async def _reject_runtime_event_admissions(self) -> None:
        """Reject every reservation when no next-turn safe point exists."""

        await self._settle_runtime_event_admissions(accept=False)

    async def _settle_runtime_event_admissions(
        self,
        *,
        accept: bool,
        agent: Agent | None = None,
    ) -> None:
        """Settle this turn's ordered reservations at its next-turn hook."""

        self._accepting_runtime_events = False
        while self._runtime_event_reservations:
            reservation = self._runtime_event_reservations.pop(0)
            if reservation.phase == "withdrawn":
                continue
            if not accept:
                self._reject_runtime_event(reservation)
                continue
            if agent is None:
                raise RuntimeError("runtime event admission requires an Agent")
            # This phase transition is the cancellation linearization point;
            # it occurs before journal append can suspend.
            reservation.phase = "appending"
            await self._accept_runtime_event(reservation, agent)

    @staticmethod
    def _reject_runtime_event(reservation: _RuntimeEventReservation) -> None:
        reservation.phase = "rejected"
        if not reservation.result.done():
            reservation.result.set_result(False)

    def _commit_runtime_event(
        self,
        reservation: _RuntimeEventReservation,
        agent: Agent,
    ) -> None:
        reservation.phase = "committed"
        self._runtime_input_messages[id(reservation.message)] = (
            reservation.event.event_id
        )
        agent.steer(reservation.message)
        if not reservation.result.done():
            reservation.result.set_result(True)

    async def _consume_injected_runtime_inputs(self) -> None:
        """Mark each committed steered input consumed once (idempotent)."""

        if self._journal is None:
            self._runtime_input_injected.clear()
            return
        while self._runtime_input_injected:
            event_id = self._runtime_input_injected.pop(0)
            if event_id in self._runtime_input_consumed:
                continue
            await self._journal.append(
                JournalRecordType.RUNTIME_INPUT_CONSUMED,
                encode_runtime_input_consumed(event_id),
                record_id=f"{self._run_id}:runtime:{event_id}:consumed",
            )
            self._runtime_input_consumed.add(event_id)

    async def _accept_runtime_event(
        self,
        reservation: _RuntimeEventReservation,
        agent: Agent,
    ) -> None:
        """Persist then queue one reservation at the next-turn hook."""

        if self._journal is None:
            self._commit_runtime_event(reservation, agent)
            return

        async def _append_settled() -> JournalAppendCancelled | JournalCommitError | None:
            try:
                await self._journal.append(
                    JournalRecordType.RUNTIME_INPUT_POSTED,
                    reservation.event.to_dict(),
                    record_id=(
                        f"{self._run_id}:runtime:{reservation.event.event_id}"
                    ),
                )
            except (JournalAppendCancelled, JournalCommitError) as exc:
                # CancelledError subclasses lose their concrete type when they
                # cross an asyncio.Task boundary. Preserve the journal's
                # durable outcome as an ordinary Task result instead.
                return exc
            return None

        append = asyncio.create_task(
            _append_settled(),
            name=f"qitos-{self._run_id}-runtime-input-append",
        )
        withdrawal = asyncio.create_task(
            reservation.withdrawal_requested.wait(),
            name=f"qitos-{self._run_id}-runtime-input-withdrawal",
        )
        cancelled_for_withdrawal = False
        try:
            done, _pending = await asyncio.wait(
                (append, withdrawal),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if withdrawal in done and not append.done():
                cancelled_for_withdrawal = True
                append.cancel()
            try:
                append_error = await asyncio.shield(append)
            except asyncio.CancelledError:
                if cancelled_for_withdrawal:
                    self._reject_runtime_event(reservation)
                    return
                raise
            except JournalCommitError as exc:
                self._settle_runtime_event_commit_error(
                    reservation,
                    agent,
                    exc,
                )
                return
            except Exception as exc:
                reservation.phase = "rejected"
                if not reservation.result.done():
                    reservation.result.set_exception(exc)
                return
            if append_error is not None:
                self._settle_runtime_event_append_error(
                    reservation,
                    agent,
                    append_error,
                )
                return
        except asyncio.CancelledError:
            append.cancel()
            try:
                append_error = await append
            except asyncio.CancelledError:
                self._reject_runtime_event(reservation)
            except Exception as exc:
                reservation.phase = "rejected"
                if not reservation.result.done():
                    reservation.result.set_exception(exc)
            else:
                if append_error is None:
                    self._commit_runtime_event(reservation, agent)
                else:
                    self._settle_runtime_event_append_error(
                        reservation,
                        agent,
                        append_error,
                    )
            raise
        finally:
            withdrawal.cancel()
            await asyncio.gather(withdrawal, return_exceptions=True)
        self._commit_runtime_event(reservation, agent)

    def _settle_runtime_event_append_error(
        self,
        reservation: _RuntimeEventReservation,
        agent: Agent,
        error: JournalAppendCancelled | JournalCommitError,
    ) -> None:
        if isinstance(error, JournalAppendCancelled):
            if error.commit_state is JournalCommitState.COMMITTED:
                self._commit_runtime_event(reservation, agent)
            elif error.commit_state is JournalCommitState.NOT_COMMITTED:
                AgentChildEngine._reject_runtime_event(reservation)
            else:
                AgentChildEngine._fail_runtime_event_unknown(reservation, error)
            return
        self._settle_runtime_event_commit_error(
            reservation,
            agent,
            error,
        )

    @staticmethod
    def _fail_runtime_event_unknown(
        reservation: _RuntimeEventReservation,
        error: JournalAppendCancelled,
    ) -> None:
        reservation.phase = "unknown"
        if not reservation.result.done():
            reservation.result.set_exception(
                error.commit_error
                or JournalError("runtime input append has unknown durable outcome")
            )

    def _settle_runtime_event_commit_error(
        self,
        reservation: _RuntimeEventReservation,
        agent: Agent,
        error: JournalCommitError,
    ) -> None:
        if error.commit_state is JournalCommitState.COMMITTED:
            self._commit_runtime_event(reservation, agent)
        elif error.commit_state is JournalCommitState.NOT_COMMITTED:
            AgentChildEngine._reject_runtime_event(reservation)
        else:
            reservation.phase = "unknown"
            if not reservation.result.done():
                reservation.result.set_exception(error)

    async def aclose(self) -> None:
        """Abort any active run, wait for settlement and close the journal."""

        if self._closed:
            return
        self._closed = True
        self._accepting_runtime_events = False
        self._runtime_started.set()
        agent = self._agent
        if agent is not None:
            agent.abort()
            await self._reject_runtime_event_admissions()
            await agent.wait_for_idle()
        if self._journal is not None:
            await self._journal.close()


def _narrow_tool_registry(
    base: ToolRegistry,
    allowed_groups: Tuple[str, ...],
    parent_authority: ToolExposure | None,
) -> ToolRegistry:
    """Intersect the configured Child pool with one frozen parent exposure."""

    registry = ToolRegistry()
    if parent_authority is None:
        return registry
    parent_names = set(parent_authority.list_tools())
    for name in sorted(parent_names.intersection(base.list_tools())):
        base_tool = base.get(name)
        parent_tool = parent_authority.get(name)
        if base_tool is None or parent_tool is None:
            continue
        # ToolExposure freezes each selected BaseTool behind an internal
        # handler wrapper.  Name equality alone is not authority: require the
        # frozen definition to point at this exact configured Tool and retain
        # the same immutable spec, otherwise fail closed on the collision.
        if getattr(parent_tool, "_handler", None) is not base_tool:
            continue
        if parent_tool.spec != base_tool.spec:
            continue
        if allowed_groups and base_tool.spec.group not in allowed_groups:
            continue
        registry.register(base_tool)
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
    model_pricing: Optional[ModelPricing] = None,
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
    if model_pricing is not None and not isinstance(model_pricing, ModelPricing):
        raise TypeError("model_pricing must be a ModelPricing or None")
    if journal_directory is not None:
        journal_root = Path(journal_directory)

        def journal_factory() -> SessionJournal:  # noqa: F811
            return JsonlSessionJournal(journal_root)

    async def _factory(
        request: ChildLaunchRequest,
        runtime_context: ChildRuntimeContext,
    ) -> ChildInvocation:
        if request.profile != "default":
            raise ValueError(
                "build_agent_child_invocation_factory has no profile resolver; "
                f"unsupported Child profile: {request.profile}"
            )
        if request.working_directory is not None:
            raise ValueError(
                "build_agent_child_invocation_factory has no working-directory "
                "resolver"
            )
        parent_permission = runtime_context.parent_permission_context
        env_permission = getattr(env, "tool_permission_context", None)
        if isinstance(env_permission, Mapping):
            env_permission = ToolPermissionContext.from_dict(dict(env_permission))
        if env_permission is not None and not isinstance(
            env_permission, ToolPermissionContext
        ):
            raise TypeError(
                "Child Env tool_permission_context must be a "
                "ToolPermissionContext, mapping, or None"
            )
        if (
            parent_permission is not None
            and env_permission is not None
            and env_permission != parent_permission
        ):
            raise ValueError(
                "Child Env permission policy differs from the frozen parent policy; "
                "the built-in factory cannot prove that it only narrows authority"
            )
        root_budget = (
            runtime_context.budget_ledger.snapshot()
            if runtime_context.budget_ledger is not None
            else None
        )
        cost_limit = request.budget.max_cost_usd is not None or (
            root_budget is not None and root_budget.max_cost_usd is not None
        )
        if cost_limit and model_pricing is None:
            raise ValueError("a Child cost budget requires explicit model_pricing")
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
                _narrow_tool_registry(
                    tool_registry,
                    request.allowed_tool_groups,
                    runtime_context.parent_tool_authority,
                )
                if tool_registry is not None
                else None
            ),
            system_prompt=system_prompt,
            env=env,
            tool_execution=tool_execution,
            max_tool_concurrency=resolved_concurrency or 1,
            max_turns=resolved_max_turns,
            run_timeout_s=resolved_timeout,
            budget=budget,
            budget_ledger=runtime_context.budget_ledger,
            model_pricing=model_pricing,
            extra_request_options=extra_request_options,
            runtime_context={
                "parent_run_id": runtime_context.child_run_id,
                "delegate_depth": runtime_context.delegate_depth + 1,
                "deadline_monotonic": runtime_context.deadline_monotonic,
                "budget_ledger": runtime_context.budget_ledger,
                "permission_context": runtime_context.parent_permission_context,
                "max_children": _tightest_int(
                    runtime_context.launch.max_children or None,
                    budget.max_children,
                )
                or 0,
            },
            journal_factory=journal_factory,
            journal_metadata={
                "parent_run_id": runtime_context.parent_run_id,
                "child_id": runtime_context.handle.child_id,
                "agent_type": request.agent_type,
                "description": request.description,
                "model_pricing": (
                    None
                    if model_pricing is None
                    else {
                        "input_usd_per_million": model_pricing.input_usd_per_million,
                        "output_usd_per_million": model_pricing.output_usd_per_million,
                        "cache_read_usd_per_million": (
                            model_pricing.cache_read_usd_per_million
                        ),
                        "cache_write_usd_per_million": (
                            model_pricing.cache_write_usd_per_million
                        ),
                    }
                ),
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
    "child_budget_stop_reason",
    "child_run_stats",
    "child_stop_reason",
    "ChildRunStats",
]
