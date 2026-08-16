"""Stateful Agent façade over the minimal agent loop.

The façade owns the transcript, lifecycle event subscription and the
steering/follow-up queues; one Agent has at most one active run. Expected
rejections (busy, empty history, assistant tail) return typed results; loop
faults and persistence failures raise.

This is the QitOS port of Pi's ``Agent`` (``pi:packages/agent/src/agent.ts``)
with QitOS's typed-rejection rule (expected rejections are values, faults are
exceptions).
"""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from .agent_events import (
    AgentEnd,
    AgentEvent,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    ToolExecutionEnd,
    ToolExecutionStart,
    TurnEnd,
)
from .agent_loop import (
    AgentContext,
    AgentLoopConfig,
    AgentLoopResult,
    AgentRunStatus,
    PrepareNextTurnHook,
    ShouldStopAfterTurnHook,
    TransformContextHook,
    TurnTransactionBoundary,
    run_agent_loop,
    run_agent_loop_continue,
)
from .cancellation import CancelToken
from .env import Env
from .journal import JournalError
from .message import AssistantMessage, Message, ToolResultMessage, UserMessage
from .tool_executor import AfterToolCallHook, BeforeToolCallHook
from .tool_registry import ToolRegistry

if TYPE_CHECKING:
    from ..models.base import Model


class QueueMode(str, Enum):
    """How a pending-message queue drains at its safe point."""

    ALL = "all"
    ONE_AT_A_TIME = "one_at_a_time"


class AgentBusyError(RuntimeError):
    """A state-mutating call raced an active run."""


@dataclass(frozen=True, slots=True)
class AgentRunRejected:
    """Typed expected rejection for run-entry operations."""

    reason: Literal["busy", "empty_history", "assistant_tail"]


AgentRunResult = Union[AgentLoopResult, AgentRunRejected]


class _PendingMessageQueue:
    def __init__(self, mode: QueueMode) -> None:
        self.mode = mode
        self._messages: List[Message] = []

    def enqueue(self, message: Message) -> None:
        self._messages.append(message)

    def has_items(self) -> bool:
        return bool(self._messages)

    def drain(self) -> List[Message]:
        if self.mode is QueueMode.ALL:
            drained = list(self._messages)
            self._messages = []
            return drained
        if not self._messages:
            return []
        first, self._messages = self._messages[0], self._messages[1:]
        return [first]

    def clear(self) -> None:
        self._messages = []


@dataclass(slots=True)
class _ActiveRun:
    task: asyncio.Task[AgentLoopResult]
    token: CancelToken
    idle: asyncio.Event = field(default_factory=asyncio.Event)


AgentEventListener = Callable[[AgentEvent], Union[Awaitable[None], None]]


class Agent:
    """One model, one Tool registry, one transcript, one active run.

    The façade freezes a Tool exposure, system prompt and model identity at
    each run start; mutating them between runs affects the next run only.
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
        transaction_factory: Optional[
            Callable[[str], Optional[TurnTransactionBoundary]]
        ] = None,
        run_id_factory: Optional[Callable[[], str]] = None,
        steering_mode: QueueMode = QueueMode.ONE_AT_A_TIME,
        follow_up_mode: QueueMode = QueueMode.ONE_AT_A_TIME,
        transform_context: Optional[TransformContextHook] = None,
        before_tool_call: Optional[BeforeToolCallHook] = None,
        after_tool_call: Optional[AfterToolCallHook] = None,
        should_stop_after_turn: Optional[ShouldStopAfterTurnHook] = None,
        prepare_next_turn: Optional[PrepareNextTurnHook] = None,
    ) -> None:
        self._model = model
        self._tool_registry = tool_registry or ToolRegistry()
        self._system_prompt = system_prompt
        self._env = env
        self._tool_execution = tool_execution
        self._max_tool_concurrency = max_tool_concurrency
        self._max_turns = max_turns
        self._run_timeout_s = run_timeout_s
        self._extra_request_options = dict(extra_request_options or {})
        self._runtime_context = dict(runtime_context or {})
        self._transaction_factory = transaction_factory
        self._run_id_factory = run_id_factory or (lambda: uuid.uuid4().hex)
        self._transform_context = transform_context
        self._before_tool_call = before_tool_call
        self._after_tool_call = after_tool_call
        self._should_stop_after_turn = should_stop_after_turn
        self._prepare_next_turn = prepare_next_turn

        self._messages: List[Message] = []
        self._listeners: List[AgentEventListener] = []
        self._steering = _PendingMessageQueue(steering_mode)
        self._follow_up = _PendingMessageQueue(follow_up_mode)
        self._active: Optional[_ActiveRun] = None
        self._is_streaming = False
        self._streaming_message: Optional[Message] = None
        self._pending_tool_calls: frozenset[str] = frozenset()
        self._error_message: Optional[str] = None

    # ── state views ─────────────────────────────────────────────────────

    @property
    def messages(self) -> Tuple[Message, ...]:
        return tuple(self._messages)

    @property
    def model(self) -> "Model":
        return self._model

    @model.setter
    def model(self, value: "Model") -> None:
        self._model = value

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        self._system_prompt = value

    @property
    def tool_registry(self) -> ToolRegistry:
        return self._tool_registry

    @property
    def is_streaming(self) -> bool:
        return self._is_streaming

    @property
    def streaming_message(self) -> Optional[Message]:
        return self._streaming_message

    @property
    def pending_tool_call_ids(self) -> frozenset[str]:
        return self._pending_tool_calls

    @property
    def error_message(self) -> Optional[str]:
        return self._error_message

    @property
    def steering_mode(self) -> QueueMode:
        return self._steering.mode

    @steering_mode.setter
    def steering_mode(self, mode: QueueMode) -> None:
        self._steering.mode = mode

    @property
    def follow_up_mode(self) -> QueueMode:
        return self._follow_up.mode

    @follow_up_mode.setter
    def follow_up_mode(self, mode: QueueMode) -> None:
        self._follow_up.mode = mode

    # ── event subscription ──────────────────────────────────────────────

    def subscribe(self, listener: AgentEventListener) -> Callable[[], None]:
        """Subscribe to lifecycle events.

        Listeners are awaited in subscription order and are part of the run's
        settlement: the run is not idle until every ``agent_end`` listener has
        finished. A raising listener fails the run — persistence listeners
        must not silently lose records.
        """

        self._listeners.append(listener)

        def _unsubscribe() -> None:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

        return _unsubscribe

    # ── queues ──────────────────────────────────────────────────────────

    def steer(self, message: Message) -> None:
        """Queue a message injected after the current turn's Tool batch."""

        self._steering.enqueue(message)

    def follow_up(self, message: Message) -> None:
        """Queue a message that runs when the agent would otherwise stop."""

        self._follow_up.enqueue(message)

    def clear_steering_queue(self) -> None:
        self._steering.clear()

    def clear_follow_up_queue(self) -> None:
        self._follow_up.clear()

    def clear_all_queues(self) -> None:
        self._steering.clear()
        self._follow_up.clear()

    def has_queued_messages(self) -> bool:
        return self._steering.has_items() or self._follow_up.has_items()

    # ── run control ─────────────────────────────────────────────────────

    def abort(self) -> None:
        """Cooperatively abort the active run, if any."""

        if self._active is not None:
            self._active.token.request_cancel("immediate")

    async def wait_for_idle(self) -> None:
        """Resolve when the active run and its listeners have settled."""

        active = self._active
        if active is None:
            return
        await active.idle.wait()

    def reset(self) -> None:
        """Clear transcript, runtime state and queues; busy runs reject."""

        if self._active is not None:
            raise AgentBusyError(
                "Agent is already processing. Wait for completion before resetting."
            )
        self._messages = []
        self._is_streaming = False
        self._streaming_message = None
        self._pending_tool_calls = frozenset()
        self._error_message = None
        self.clear_all_queues()

    async def prompt(
        self, message: Union[str, Message, Sequence[Message]]
    ) -> AgentRunResult:
        """Start a new run from text, one message or a message batch."""

        if self._active is not None:
            return AgentRunRejected(reason="busy")
        messages = self._normalize_prompt(message)
        return await self._run(messages)

    async def continue_run(self) -> AgentRunResult:
        """Continue from the transcript tail (user or Tool-result message)."""

        if self._active is not None:
            return AgentRunRejected(reason="busy")
        if not self._messages:
            return AgentRunRejected(reason="empty_history")
        tail = self._messages[-1]
        if isinstance(tail, AssistantMessage):
            queued = self._steering.drain()
            if queued:
                return await self._run(queued, skip_initial_steering_poll=True)
            queued = self._follow_up.drain()
            if queued:
                return await self._run(queued)
            return AgentRunRejected(reason="assistant_tail")
        return await self._run(None)

    # ── internals ───────────────────────────────────────────────────────

    @staticmethod
    def _normalize_prompt(
        message: Union[str, Message, Sequence[Message]]
    ) -> List[Message]:
        if isinstance(message, str):
            return [UserMessage(content=message)]
        if isinstance(message, (UserMessage, AssistantMessage, ToolResultMessage)):
            return [message]
        if isinstance(message, Sequence) and not isinstance(message, (str, bytes)):
            messages: List[Message] = list(message)
            if not messages:
                raise ValueError("prompt requires at least one message")
            return messages
        raise TypeError("prompt expects text, a Message, or a sequence of Messages")

    async def _run(
        self,
        prompts: Optional[List[Message]],
        *,
        skip_initial_steering_poll: bool = False,
    ) -> AgentLoopResult:
        if self._active is not None:
            raise AgentBusyError("Agent is already processing.")

        run_id = str(self._run_id_factory())
        token = CancelToken()
        transaction = (
            self._transaction_factory(run_id)
            if self._transaction_factory is not None
            else None
        )
        deadline = (
            None
            if self._run_timeout_s is None
            else time.monotonic() + self._run_timeout_s
        )
        steering_queue = self._steering
        follow_up_queue = self._follow_up
        skip_poll = skip_initial_steering_poll

        def _drain_steering() -> List[Message]:
            nonlocal skip_poll
            if skip_poll:
                skip_poll = False
                return []
            return steering_queue.drain()

        config = AgentLoopConfig(
            model=self._model,
            run_id=run_id,
            tool_execution=self._tool_execution,
            max_tool_concurrency=self._max_tool_concurrency,
            max_turns=self._max_turns,
            deadline_monotonic=deadline,
            extra_request_options=self._extra_request_options,
            runtime_context=self._runtime_context,
            transaction=transaction,
            transform_context=self._transform_context,
            before_tool_call=self._before_tool_call,
            after_tool_call=self._after_tool_call,
            should_stop_after_turn=self._should_stop_after_turn,
            prepare_next_turn=self._prepare_next_turn,
            get_steering_messages=_drain_steering,
            get_follow_up_messages=follow_up_queue.drain,
        )
        context = AgentContext(
            system_prompt=self._system_prompt,
            messages=list(self._messages),
            tools=self._tool_registry.freeze(),
            env=self._env,
        )

        self._is_streaming = True
        self._streaming_message = None
        self._error_message = None

        async def _execute() -> AgentLoopResult:
            try:
                if prompts is None:
                    return await run_agent_loop_continue(
                        context, config, self._process_event, token
                    )
                return await run_agent_loop(
                    prompts, context, config, self._process_event, token
                )
            except asyncio.CancelledError:
                raise
            except JournalError:
                raise
            except Exception as exc:
                # A loop fault still produces a complete terminal event
                # sequence and a typed failed outcome (Pi handleRunFailure).
                return await self._run_failure_outcome(exc, transaction)

        task = asyncio.create_task(_execute(), name=f"qitos-agent-{run_id[:8]}")
        active = _ActiveRun(task=task, token=token)
        self._active = active

        def _settle(_done: "asyncio.Task[AgentLoopResult]") -> None:
            self._is_streaming = False
            self._streaming_message = None
            self._pending_tool_calls = frozenset()
            active.idle.set()
            if self._active is active:
                self._active = None

        task.add_done_callback(_settle)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Caller cancellation aborts cooperatively; the run task stays
            # owned by this Agent and settles through the done callback.
            token.request_cancel("immediate")
            raise

    async def _run_failure_outcome(
        self,
        exc: Exception,
        transaction: Optional[TurnTransactionBoundary],
    ) -> AgentLoopResult:
        failure = AssistantMessage(
            error=str(exc) or "agent run failed",
            model_name=getattr(self._model, "model", None),
            provider=getattr(self._model, "provider_name", None),
        )
        await self._process_event(MessageStart(message=failure))
        await self._process_event(MessageEnd(message=failure))
        await self._process_event(
            TurnEnd(turn=-1, message=failure, tool_results=())
        )
        result = AgentLoopResult(
            status=AgentRunStatus.FAILED,
            messages=(failure,),
            error=failure.error,
        )
        if transaction is not None:
            await transaction.run_terminal(result)
        await self._process_event(AgentEnd(messages=(failure,)))
        return result

    async def _process_event(self, event: AgentEvent) -> None:
        if isinstance(event, (MessageStart, MessageUpdate)):
            self._streaming_message = event.message
        elif isinstance(event, MessageEnd):
            self._streaming_message = None
            self._messages.append(event.message)
        elif isinstance(event, ToolExecutionStart):
            self._pending_tool_calls = self._pending_tool_calls | {
                event.tool_call_id
            }
        elif isinstance(event, ToolExecutionEnd):
            self._pending_tool_calls = self._pending_tool_calls - {
                event.tool_call_id
            }
        elif isinstance(event, TurnEnd):
            if event.message.error:
                self._error_message = event.message.error
        elif isinstance(event, AgentEnd):
            self._streaming_message = None

        for listener in list(self._listeners):
            outcome = listener(event)
            if inspect.isawaitable(outcome):
                await outcome


__all__ = [
    "Agent",
    "AgentBusyError",
    "AgentEventListener",
    "AgentRunRejected",
    "AgentRunResult",
    "AgentRunStatus",
    "QueueMode",
]
