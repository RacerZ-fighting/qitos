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
    cast,
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
    RunFinalizer,
    ShouldStopAfterTurnHook,
    TransformContextHook,
    TurnTransactionBoundary,
    run_agent_loop,
    run_agent_loop_continue,
)
from .cancellation import CancelSignalView, CancelToken
from .env import Env
from .message import AssistantMessage, Message, ToolResultMessage, UserMessage
from .thinking import ThinkingLevel
from .tool_executor import AfterToolCallHook, BeforeToolCallHook
from .tool_registry import ToolRegistry

if TYPE_CHECKING:
    from ..models.base import Model


class QueueMode(str, Enum):
    """How a pending-message queue drains at its safe point."""

    ALL = "all"
    ONE_AT_A_TIME = "one-at-a-time"


class AgentBusyError(RuntimeError):
    """A state-mutating call raced an active run."""


class AgentListenerTimeoutError(RuntimeError):
    """A listener exceeded the active run's absolute deadline."""


@dataclass(frozen=True, slots=True)
class AgentRunRejected:
    """Typed expected rejection for run-entry operations.

    ``task_terminal`` is returned by the Session Harness when a
    task-bearing Session whose Root Task is terminal is prompted for new
    work; continuing requires an explicit new Task
    (``SessionRun.start_follow_up``).
    """

    reason: Literal["busy", "empty_history", "assistant_tail", "task_terminal"]


AgentRunResult = Union[AgentLoopResult, AgentRunRejected]


class _PendingMessageQueue:
    def __init__(self, mode: QueueMode) -> None:
        self._mode = QueueMode.ONE_AT_A_TIME
        self.mode = mode
        self._messages: List[Message] = []

    @property
    def mode(self) -> QueueMode:
        return self._mode

    @mode.setter
    def mode(self, value: QueueMode) -> None:
        if not isinstance(value, QueueMode):
            raise TypeError("queue mode must be a QueueMode")
        self._mode = value

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

    def prepend(self, messages: Sequence[Message]) -> None:
        """Restore an accepted batch that the loop never injected."""

        if messages:
            self._messages = list(messages) + self._messages

    def clear(self) -> None:
        self._messages = []


@dataclass(slots=True)
class _ActiveRun:
    task: asyncio.Task[AgentLoopResult]
    token: CancelToken
    deadline_monotonic: Optional[float] = None
    idle: asyncio.Event = field(default_factory=asyncio.Event)


AgentEventListener = Union[
    Callable[[AgentEvent], Union[Awaitable[None], None]],
    Callable[
        [AgentEvent, CancelSignalView], Union[Awaitable[None], None]
    ],
]


def normalize_prompt_messages(
    message: Union[str, Message, Sequence[Message]]
) -> List[Message]:
    """Normalize prompt input into a non-empty list of typed Messages."""

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


class Agent:
    """One model, one Tool registry, one transcript, one active run.

    The façade hands the live Tool registry to the loop, which re-freezes it
    into an immutable exposure at every turn boundary; mutating the registry,
    system prompt or model between runs affects the next run only, while a
    Tool loaded mid-run becomes visible to the next turn. ``initial_messages``
    and ``turn_base`` restore a recovered transcript and its journaled turn
    numbering (Session resume/fork); ``set_transcript`` replaces the
    transcript between runs and seals the seeded Provider continuations.
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
        thinking_level: Optional[ThinkingLevel] = None,
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
        run_finalizer: RunFinalizer | None = None,
        initial_messages: Sequence[Message] = (),
        turn_base: int = 0,
    ) -> None:
        self._model = model
        self._tool_registry = (
            tool_registry if tool_registry is not None else ToolRegistry()
        )
        self._system_prompt = system_prompt
        self._env = env
        self._tool_execution = tool_execution
        self._max_tool_concurrency = max_tool_concurrency
        self._max_turns = max_turns
        self._run_timeout_s = run_timeout_s
        self._extra_request_options = dict(extra_request_options or {})
        self.thinking_level = thinking_level
        self._runtime_context = dict(runtime_context or {})
        self._transaction_factory = transaction_factory
        self._run_id_factory = run_id_factory or (lambda: uuid.uuid4().hex)
        self._transform_context = transform_context
        self._before_tool_call = before_tool_call
        self._after_tool_call = after_tool_call
        self._should_stop_after_turn = should_stop_after_turn
        self._prepare_next_turn = prepare_next_turn
        self._run_finalizer = run_finalizer
        if (
            isinstance(turn_base, bool)
            or not isinstance(turn_base, int)
            or turn_base < 0
        ):
            raise ValueError("turn_base must be a non-negative integer")
        self._turn_base = turn_base

        self._messages: List[Message] = []
        self._continuation_floor = 0
        self._seed_transcript(initial_messages)
        self._listeners: List[Tuple[AgentEventListener, bool]] = []
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
    def thinking_level(self) -> Optional[ThinkingLevel]:
        """Requested thinking level for runs started after the assignment.

        The value is captured into each run's frozen loop configuration, so
        a mid-run assignment never rewrites an in-flight turn; per-turn
        changes inside one run go through ``prepare_next_turn``.
        """

        return self._thinking_level

    @thinking_level.setter
    def thinking_level(self, value: Optional[ThinkingLevel]) -> None:
        if value is not None and not isinstance(value, ThinkingLevel):
            raise TypeError("thinking_level must be a ThinkingLevel or None")
        self._thinking_level = value

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
    def signal(self) -> Optional[CancelSignalView]:
        """Read-only cancellation signal for the active run, if any."""

        return None if self._active is None else self._active.token.signal

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
        finished. A raising listener is an implementation fault — it
        terminalizes the run's durable records and then propagates, since
        persistence listeners must not silently lose records. Pi-style
        two-argument listeners receive the run's read-only
        :class:`CancelSignalView`; existing one-argument listeners remain
        accepted. Cancellation is visible through that signal but does not
        forcibly cancel a listener; a configured absolute run deadline is the
        bounded escape hatch. Cancelled listener tasks are awaited to
        settlement.
        """

        try:
            signature = inspect.signature(listener)
        except (TypeError, ValueError):
            accepts_signal = False
        else:
            try:
                signature.bind(object(), object())
            except TypeError:
                try:
                    signature.bind(object())
                except TypeError as exc:
                    raise TypeError(
                        "agent listener must accept (event) or (event, signal)"
                    ) from exc
                accepts_signal = False
            else:
                accepts_signal = True
        subscription = (listener, accepts_signal)
        self._listeners.append(subscription)

        def _unsubscribe() -> None:
            try:
                self._listeners.remove(subscription)
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
        self._continuation_floor = 0
        self._clear_run_projection()
        self.clear_all_queues()

    def set_transcript(self, messages: Sequence[Message]) -> None:
        """Replace the transcript between runs (restore or compaction swap).

        Busy runs reject because a run holds its own context snapshot. The
        replacement seals every Provider continuation carried by the seeded
        messages: the next model request is a full request, and only
        assistant messages produced by this façade's own later runs may
        chain a continuation again. Steering/follow-up queues are
        memory-only input, not transcript truth, so they are preserved.
        """

        if self._active is not None:
            raise AgentBusyError(
                "Agent is already processing. Wait for completion before "
                "replacing the transcript."
            )
        self._seed_transcript(messages)
        self._continuation_floor = len(self._messages)
        self._clear_run_projection()

    def _seed_transcript(self, messages: Sequence[Message]) -> None:
        seeded = list(messages)
        for message in seeded:
            if not isinstance(
                message, (UserMessage, AssistantMessage, ToolResultMessage)
            ):
                raise TypeError(
                    "transcript messages must be typed UserMessage, "
                    "AssistantMessage or ToolResultMessage values"
                )
        self._messages = seeded

    def _clear_run_projection(self) -> None:
        self._is_streaming = False
        self._streaming_message = None
        self._pending_tool_calls = frozenset()
        self._error_message = None

    async def prompt(
        self, message: Union[str, Message, Sequence[Message]]
    ) -> AgentRunResult:
        """Start a new run from text, one message or a message batch."""

        if self._active is not None:
            return AgentRunRejected(reason="busy")
        messages = normalize_prompt_messages(message)
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
        drained_steering: List[Message] = []
        drained_follow_up: List[Message] = []
        skip_poll = skip_initial_steering_poll

        def _drain_steering() -> List[Message]:
            nonlocal skip_poll
            if skip_poll:
                skip_poll = False
                return []
            drained = steering_queue.drain()
            drained_steering.extend(drained)
            return drained

        def _drain_follow_up() -> List[Message]:
            drained = follow_up_queue.drain()
            drained_follow_up.extend(drained)
            return drained

        config = AgentLoopConfig(
            model=self._model,
            run_id=run_id,
            tool_execution=self._tool_execution,
            max_tool_concurrency=self._max_tool_concurrency,
            max_turns=self._max_turns,
            deadline_monotonic=deadline,
            extra_request_options=self._extra_request_options,
            thinking_level=self._thinking_level,
            runtime_context=self._runtime_context,
            transaction=transaction,
            turn_base=self._turn_base,
            transform_context=self._transform_context,
            before_tool_call=self._before_tool_call,
            after_tool_call=self._after_tool_call,
            should_stop_after_turn=self._should_stop_after_turn,
            prepare_next_turn=self._prepare_next_turn,
            get_steering_messages=_drain_steering,
            get_follow_up_messages=_drain_follow_up,
            continuation_floor=self._continuation_floor,
            run_finalizer=self._run_finalizer,
        )
        context = AgentContext(
            system_prompt=self._system_prompt,
            messages=list(self._messages),
            # The live registry: the loop re-freezes it at every turn, so a
            # Skill/MCP Tool loaded during one turn is exposed to the next.
            tools=self._tool_registry,
            env=self._env,
        )

        self._is_streaming = True
        self._streaming_message = None
        self._error_message = None

        async def _execute() -> AgentLoopResult:
            # Expected rejections are typed values at the prompt/continue_run
            # boundary and model failures are FAILED results inside the loop;
            # everything else that raises here is an implementation fault
            # (listener, codec, persistence or loop bug) and propagates after
            # the loop has terminalized the run's durable records.
            result: Optional[AgentLoopResult] = None
            try:
                if prompts is None:
                    result = await run_agent_loop_continue(
                        context, config, self._process_event, token
                    )
                else:
                    result = await run_agent_loop(
                        prompts, context, config, self._process_event, token
                    )
                return result
            finally:
                # A queue drain is an internal poll, not delivery. If a turn
                # budget or fault stops the loop before those exact Message
                # objects enter its context/result, restore them ahead of
                # messages accepted later in the same run.
                delivered = list(context.messages)
                if result is not None:
                    delivered.extend(result.messages)

                def _undelivered(drained: Sequence[Message]) -> List[Message]:
                    return [
                        message
                        for message in drained
                        if not any(message is item for item in delivered)
                    ]

                steering_queue.prepend(_undelivered(drained_steering))
                follow_up_queue.prepend(_undelivered(drained_follow_up))

        task = asyncio.create_task(_execute(), name=f"qitos-agent-{run_id[:8]}")
        active = _ActiveRun(task=task, token=token, deadline_monotonic=deadline)
        self._active = active

        def _settle(_done: "asyncio.Task[AgentLoopResult]") -> None:
            self._is_streaming = False
            self._streaming_message = None
            self._pending_tool_calls = frozenset()
            if not _done.cancelled():
                fault = _done.exception()
                if fault is not None:
                    # Event listeners project façade state, but they are not
                    # the canonical transcript owner. If projection fails,
                    # recover the fully paired in-memory loop transcript
                    # before exposing the implementation fault.
                    self._messages = list(context.messages)
                    self._error_message = str(fault) or "agent run failed"
            active.idle.set()
            if self._active is active:
                self._active = None

        task.add_done_callback(_settle)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Caller cancellation aborts cooperatively; the run task stays
            # owned by this Agent and must reach its durable terminal state
            # before the caller observes the cancellation.
            token.request_cancel("immediate")
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    # Caller cancellation remains the public outcome, but only
                    # after the owned run (including durable terminalization
                    # and listener cleanup) has fully settled.
                    break
            while not active.idle.is_set():
                try:
                    await asyncio.shield(active.idle.wait())
                except asyncio.CancelledError:
                    continue
            raise

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

        active = self._active
        if active is None:
            raise RuntimeError("Agent listener invoked outside an active run")
        signal: CancelSignalView = active.token.signal
        for listener, accepts_signal in list(self._listeners):
            if accepts_signal:
                signal_listener = cast(
                    Callable[
                        [AgentEvent, CancelSignalView],
                        Union[Awaitable[None], None],
                    ],
                    listener,
                )
                outcome = signal_listener(event, signal)
            else:
                event_listener = cast(
                    Callable[[AgentEvent], Union[Awaitable[None], None]],
                    listener,
                )
                outcome = event_listener(event)
            if inspect.isawaitable(outcome):
                await self._bounded_listener(listener, outcome)

    async def _bounded_listener(
        self, listener: AgentEventListener, outcome: Awaitable[None]
    ) -> None:
        """Await one listener bounded by the run's cancellation and deadline.

        Listener exceptions propagate as faults (persistence listeners must
        not silently lose records). Cancellation is exposed through the
        listener's read-only signal but does not cancel the listener task: Pi
        listeners settle in subscription order even for an aborted run. The
        absolute run deadline remains the bounded escape hatch. A task that is
        cancelled by its caller is still awaited during cleanup, so no callback
        work is detached from the run.
        """

        active = self._active
        deadline = active.deadline_monotonic if active is not None else None
        if deadline is None:
            await outcome
            return

        async def _await_outcome() -> None:
            await outcome

        task = asyncio.create_task(_await_outcome(), name="qitos-agent-listener")
        remaining = max(0.0, deadline - time.monotonic())
        try:
            done, _pending = await asyncio.wait(
                {task}, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        if task in done:
            task.result()  # listener faults propagate
            return
        task.cancel()
        outcomes = await asyncio.gather(task, return_exceptions=True)
        listener_outcome = outcomes[0]
        if isinstance(listener_outcome, BaseException) and not isinstance(
            listener_outcome, asyncio.CancelledError
        ):
            raise listener_outcome
        raise AgentListenerTimeoutError(
            "agent listener exceeded the run deadline: "
            f"{getattr(listener, '__name__', repr(listener))}"
        )


__all__ = [
    "Agent",
    "AgentBusyError",
    "AgentEventListener",
    "AgentListenerTimeoutError",
    "AgentRunRejected",
    "AgentRunResult",
    "AgentRunStatus",
    "QueueMode",
    "normalize_prompt_messages",
]
