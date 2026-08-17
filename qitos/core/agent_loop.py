"""Minimal async agent loop: Message -> Model -> ToolCall -> ToolResult.

The loop owns only context projection, one immutable model request per turn,
assistant streaming, ToolCall admission/execution/finalization, ordered
ToolResult insertion, steering/follow-up safe points and stop evaluation. It
does not discover resources, construct environments, own Session storage or
reduce application state.

Failure semantics:
- Every admitted or rejected ToolCall receives exactly one terminal ToolResult.
- Model stream failures become terminal assistant messages with ``error`` set;
  they never raise out of the loop.
- Cooperative cancellation flows through ``CancelToken``: ``"immediate"``
  interrupts in-flight work at the next safe point, ``"after_step"`` lets the
  current turn commit and stops the run at the turn boundary; both end with a
  terminal assistant message and an ``ABORTED`` result.
- External task cancellation and faults (persistence failures, listener or
  hook bugs, contract violations) first terminalize started work — the open
  assistant message, its model transaction record, the Tool batch and the
  run terminal record — and only then re-raise. A run never ends without a
  terminal record.
"""

from __future__ import annotations

import asyncio
import dataclasses
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
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Union,
)

from .agent_events import (
    AgentEnd,
    AgentEvent,
    AgentStart,
    EventSink,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    ToolExecutionEnd,
    ToolExecutionStart,
    TurnEnd,
    TurnStart,
    emit_to,
)
from .cancellation import CancelToken
from .env import Env
from .errors import ModelRequestDeadlineExceeded, ModelTransportError
from .message import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolResultMessage,
    assistant_from_response,
    message_to_wire,
)
from .model_request import ModelContinuation, ModelRequest
from .model_response import ModelResponse
from .model_stream import ModelStreamEvent, ModelStreamEventType
from .thinking import ThinkingLevel, clamp_thinking_level
from .tool_executor import (
    AfterToolCallHook,
    AgentContextSnapshot,
    BeforeToolCallHook,
    ToolBatchExecutor,
    ToolExecutionConfig,
    ToolTransactionBoundary,
)
from .tool_registry import ToolExposure, ToolRegistry
from .tool_result import ToolResult

if TYPE_CHECKING:
    from ..models.base import Model


class AgentRunStatus(str, Enum):
    """Terminal status of one loop run."""

    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"
    MAX_TURNS = "max_turns"
    DEADLINE_EXCEEDED = "deadline_exceeded"


@dataclass(frozen=True, slots=True)
class AgentLoopResult:
    """Terminal outcome of one run; ``messages`` are the run's new messages."""

    status: AgentRunStatus
    messages: Tuple[Message, ...]
    error: Optional[str] = None


class TurnTransactionBoundary(ToolTransactionBoundary, Protocol):
    """Durable barriers the loop records around model and Tool side effects."""

    async def model_terminal(
        self, turn: int, request: ModelRequest, message: AssistantMessage
    ) -> None:
        """Record one finalized model transaction (success or failure)."""
        ...

    async def turn_committed(
        self, turn: int, new_messages: Tuple[Message, ...]
    ) -> None:
        """Record one committed turn with the messages it appended."""
        ...

    async def run_terminal(self, result: AgentLoopResult) -> None:
        """Record the run's terminal outcome exactly once."""
        ...


@dataclass(slots=True)
class AgentContext:
    """The loop's working context: system prompt, transcript, tools, env.

    ``tools`` may be a frozen ``ToolExposure`` (fixed for the whole run) or a
    live ``ToolRegistry``, which the loop re-freezes at every turn boundary —
    a Skill/MCP Tool registered during one turn becomes visible to the next
    turn's model request, while the turn in flight keeps its own immutable
    snapshot.
    """

    system_prompt: str = ""
    messages: List[Message] = field(default_factory=list)
    tools: Optional[Union[ToolExposure, ToolRegistry]] = None
    env: Optional[Env] = None


@dataclass(frozen=True, slots=True)
class TurnHookContext:
    """Immutable view handed to turn-boundary hooks."""

    turn: int
    message: AssistantMessage
    tool_results: Tuple[ToolResultMessage, ...]
    new_messages: Tuple[Message, ...]
    context: AgentContextSnapshot


@dataclass(frozen=True, slots=True)
class NextTurnUpdate:
    """Optional per-turn replacement of the loop's runtime state (Pi parity).

    Provided fields replace their counterpart before the next provider
    request; omitted fields keep the current value. Replacing ``messages``
    does not rewrite already-committed journal records — the next turn's
    model transaction record reflects the replacement.
    """

    system_prompt: Optional[str] = None
    model: Optional["Model"] = None
    messages: Optional[Tuple[Message, ...]] = None
    tools: Optional[Union[ToolExposure, ToolRegistry]] = None
    extra_request_options: Optional[Mapping[str, Any]] = None
    thinking_level: Optional[ThinkingLevel] = None


TransformContextHook = Callable[
    [List[Message]], Union[List[Message], Awaitable[List[Message]]]
]
ShouldStopAfterTurnHook = Callable[
    [TurnHookContext], Union[bool, Awaitable[bool]]
]
PrepareNextTurnHook = Callable[
    [TurnHookContext],
    Union[NextTurnUpdate, None, Awaitable[Optional[NextTurnUpdate]]],
]
QueueDrainHook = Callable[
    [], Union[Sequence[Message], Awaitable[Sequence[Message]], None]
]


@dataclass(frozen=True, slots=True)
class AgentLoopConfig:
    """One run's immutable loop configuration.

    Hook contract: hooks must not throw. ``before_tool_call`` and
    ``after_tool_call`` exceptions are converted to Tool error results;
    exceptions from ``transform_context``, ``should_stop_after_turn`` and
    ``prepare_next_turn`` are implementation faults and propagate after the
    run is terminalized. Every hook await is bounded by the run deadline and
    the cancel token, so a hung hook cannot block abort or the deadline.
    """

    model: "Model"
    run_id: str
    tool_execution: Literal["sequential", "parallel"] = "sequential"
    max_tool_concurrency: int = 8
    max_turns: Optional[int] = None
    deadline_monotonic: Optional[float] = None
    extra_request_options: Mapping[str, Any] = field(default_factory=dict)
    thinking_level: Optional[ThinkingLevel] = None
    runtime_context: Mapping[str, Any] = field(default_factory=dict)
    transaction: Optional[TurnTransactionBoundary] = None
    transform_context: Optional[TransformContextHook] = None
    before_tool_call: Optional[BeforeToolCallHook] = None
    after_tool_call: Optional[AfterToolCallHook] = None
    should_stop_after_turn: Optional[ShouldStopAfterTurnHook] = None
    prepare_next_turn: Optional[PrepareNextTurnHook] = None
    get_steering_messages: Optional[QueueDrainHook] = None
    get_follow_up_messages: Optional[QueueDrainHook] = None

    def __post_init__(self) -> None:
        if not getattr(self.model, "provider_name", None) or not getattr(
            self.model, "model", None
        ):
            raise TypeError("AgentLoopConfig.model must be a qitos Model")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id must be non-empty text")
        if self.max_turns is not None and (
            isinstance(self.max_turns, bool) or self.max_turns < 1
        ):
            raise ValueError("max_turns must be a positive integer or None")
        if self.thinking_level is not None and not isinstance(
            self.thinking_level, ThinkingLevel
        ):
            raise TypeError("thinking_level must be a ThinkingLevel or None")


class _StreamAborted(Exception):
    """Internal signal: immediate cancellation fired while streaming."""


class _HookTimedOut(Exception):
    """Internal signal: a loop hook exceeded the remaining run deadline."""


class _EventSinkFault(Exception):
    """Internal wrapper preventing listener faults from becoming model errors."""

    def __init__(self, cause: Exception) -> None:
        super().__init__(str(cause))
        self.cause = cause


async def _emit_during_model_stream(
    emit: EventSink | None, event: AgentEvent
) -> None:
    """Emit a streaming event without classifying listener faults as model faults."""

    try:
        await emit_to(emit, event)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise _EventSinkFault(exc) from exc


async def run_agent_loop(
    prompts: Sequence[Message],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: EventSink | None,
    cancel_token: CancelToken | None = None,
) -> AgentLoopResult:
    """Run the loop with new prompt messages appended to the context."""

    if not prompts:
        raise ValueError("agent loop requires at least one prompt message")
    new_messages: List[Message] = list(prompts)
    context.messages.extend(prompts)
    guard = _RunTerminalGuard()
    try:
        if cancel_token is not None:
            cancel_token.reset_step_event()
        await emit_to(emit, AgentStart())
        await emit_to(emit, TurnStart(turn=0))
        for prompt in prompts:
            await emit_to(emit, MessageStart(message=prompt))
            await emit_to(emit, MessageEnd(message=prompt))
        return await _run_loop(
            context, new_messages, config, emit, cancel_token, guard, turn=0
        )
    except asyncio.CancelledError:
        await guard.terminalize_interrupted(
            context,
            new_messages,
            config,
            emit,
            status=AgentRunStatus.ABORTED,
            error="run task cancelled",
        )
        raise
    except Exception as exc:
        await guard.terminalize_interrupted(
            context,
            new_messages,
            config,
            emit,
            status=AgentRunStatus.FAILED,
            error=str(exc) or "agent run failed",
        )
        raise


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: EventSink | None,
    cancel_token: CancelToken | None = None,
) -> AgentLoopResult:
    """Run the loop from the current context without a new message."""

    if not context.messages:
        raise ValueError("cannot continue: no messages in context")
    if isinstance(context.messages[-1], AssistantMessage):
        raise ValueError("cannot continue from an assistant message")
    new_messages: List[Message] = []
    guard = _RunTerminalGuard()
    try:
        if cancel_token is not None:
            cancel_token.reset_step_event()
        await emit_to(emit, AgentStart())
        await emit_to(emit, TurnStart(turn=0))
        return await _run_loop(
            context, new_messages, config, emit, cancel_token, guard, turn=0
        )
    except asyncio.CancelledError:
        await guard.terminalize_interrupted(
            context,
            new_messages,
            config,
            emit,
            status=AgentRunStatus.ABORTED,
            error="run task cancelled",
        )
        raise
    except Exception as exc:
        await guard.terminalize_interrupted(
            context,
            new_messages,
            config,
            emit,
            status=AgentRunStatus.FAILED,
            error=str(exc) or "agent run failed",
        )
        raise


class _RunTerminalGuard:
    """Write the run terminal record exactly once, on any exit path."""

    def __init__(self) -> None:
        self._written = False
        self._end_attempted = False

    async def _emit_end_once(
        self, emit: EventSink | None, new_messages: Sequence[Message]
    ) -> None:
        if self._end_attempted:
            return
        self._end_attempted = True
        await emit_to(emit, AgentEnd(messages=tuple(new_messages)))

    async def finish(
        self,
        status: AgentRunStatus,
        new_messages: Sequence[Message],
        config: AgentLoopConfig,
        emit: EventSink | None,
        error: Optional[str] = None,
    ) -> AgentLoopResult:
        result = AgentLoopResult(
            status=status, messages=tuple(new_messages), error=error
        )
        self._written = True
        if config.transaction is not None:
            await config.transaction.run_terminal(result)
        await self._emit_end_once(emit, new_messages)
        return result

    async def terminalize_interrupted(
        self,
        context: AgentContext,
        new_messages: Sequence[Message],
        config: AgentLoopConfig,
        emit: EventSink | None,
        *,
        status: AgentRunStatus,
        error: str,
    ) -> None:
        """Terminalize a run torn down by cancellation or a fault."""

        if not self._written:
            self._written = True
            if config.transaction is not None:
                result = AgentLoopResult(
                    status=status, messages=tuple(new_messages), error=error
                )
                await config.transaction.run_terminal(result)
        await self._emit_end_once(emit, new_messages)


def agent_loop(
    prompts: Sequence[Message],
    context: AgentContext,
    config: AgentLoopConfig,
    cancel_token: CancelToken | None = None,
) -> "AgentEventStream":
    """Start a loop run and return its push-based event stream."""

    run_token = cancel_token if cancel_token is not None else CancelToken()

    async def _runner() -> AgentLoopResult:
        return await run_agent_loop(
            prompts, context, config, stream.push, run_token
        )

    stream = AgentEventStream(_runner(), cancel_token=run_token)
    return stream


def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    cancel_token: CancelToken | None = None,
) -> "AgentEventStream":
    """Continue a loop run from the context tail and return its event stream."""

    run_token = cancel_token if cancel_token is not None else CancelToken()

    async def _runner() -> AgentLoopResult:
        return await run_agent_loop_continue(
            context, config, stream.push, run_token
        )

    stream = AgentEventStream(_runner(), cancel_token=run_token)
    return stream


_STREAM_SENTINEL: Any = object()


class AgentEventStream:
    """Async-iterable run events plus exactly one terminal result.

    The producing run is owned by this stream. Cancelling the consuming task
    requests cooperative cancellation on the run; the run still settles and
    its result remains available through :meth:`result`.
    """

    def __init__(
        self,
        runner: Awaitable[AgentLoopResult],
        *,
        cancel_token: CancelToken | None = None,
    ) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._cancel_token = cancel_token
        self._task: asyncio.Task[None] = asyncio.create_task(
            self._drive(runner), name="qitos-agent-loop"
        )

    def push(self, event: AgentEvent) -> None:
        self._queue.put_nowait(event)

    async def _drive(self, runner: Awaitable[AgentLoopResult]) -> None:
        try:
            self._result = await runner
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failure = exc
        finally:
            self._queue.put_nowait(_STREAM_SENTINEL)

    def __aiter__(self) -> "AgentEventStream":
        return self

    async def __anext__(self) -> AgentEvent:
        try:
            item = await self._queue.get()
        except asyncio.CancelledError:
            if self._cancel_token is not None:
                self._cancel_token.request_cancel("immediate")
            while not self._task.done():
                try:
                    await asyncio.shield(self._task)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
            raise
        if item is _STREAM_SENTINEL:
            raise StopAsyncIteration
        return item

    async def result(self) -> AgentLoopResult:
        """Await the run's terminal outcome, re-raising run faults."""

        await self._task
        failure = getattr(self, "_failure", None)
        if failure is not None:
            raise failure
        return self._result  # type: ignore[attr-defined]


async def _drain_queue(
    hook: QueueDrainHook | None,
    token: CancelToken | None,
    deadline_monotonic: Optional[float],
) -> List[Message]:
    """Drain one steering/follow-up queue, bounded by cancel and deadline."""

    if hook is None:
        return []
    drained = await _bounded_await(hook(), token, deadline_monotonic)
    return [message for message in list(drained or [])]


async def _bounded_await(
    value: Any,
    token: CancelToken | None,
    deadline_monotonic: Optional[float],
) -> Any:
    """Await one loop hook bounded by cancellation and the run deadline.

    Hook exceptions propagate as faults; immediate cancellation raises
    :class:`_StreamAborted`; deadline expiry raises :class:`_HookTimedOut`.
    Cancelled hook tasks are awaited to settlement so the loop never leaves
    detached callback work behind. Graceful ``after_step`` cancellation is
    observed only at the committed turn boundary.
    """

    if not inspect.isawaitable(value):
        return value

    async def _await_value() -> Any:
        return await value

    hook_task = asyncio.create_task(_await_value(), name="qitos-loop-hook")
    tasks: Set[asyncio.Task[Any]] = {hook_task}
    watcher: Optional[asyncio.Task[bool]] = None
    if token is not None:
        watcher = asyncio.create_task(token.wait_immediate())
        tasks.add(watcher)
    remaining = (
        None
        if deadline_monotonic is None
        else max(0.0, deadline_monotonic - time.monotonic())
    )
    try:
        done, _pending = await asyncio.wait(
            tasks, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
        )
    except asyncio.CancelledError:
        hook_task.cancel()
        if watcher is not None:
            watcher.cancel()
        await asyncio.gather(
            *[task for task in (hook_task, watcher) if task is not None],
            return_exceptions=True,
        )
        raise
    if hook_task in done:
        if watcher is not None:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        return hook_task.result()
    hook_task.cancel()
    if watcher is not None and not watcher.done():
        watcher.cancel()
    outcomes = await asyncio.gather(
        *[task for task in (hook_task, watcher) if task is not None],
        return_exceptions=True,
    )
    hook_outcome = outcomes[0]
    if isinstance(hook_outcome, BaseException) and not isinstance(
        hook_outcome, asyncio.CancelledError
    ):
        raise hook_outcome
    if watcher is not None and watcher in done:
        raise _StreamAborted
    raise _HookTimedOut


def _turn_exposure(context: AgentContext) -> ToolExposure:
    """Resolve this turn's immutable Tool exposure (per-turn snapshot)."""

    tools = context.tools
    if tools is None:
        return ToolRegistry().freeze()
    if isinstance(tools, ToolRegistry):
        return tools.freeze()
    return tools


def _turn_hook_context(
    *,
    turn: int,
    message: AssistantMessage,
    tool_results: Sequence[ToolResultMessage],
    new_messages: Sequence[Message],
    context: AgentContext,
) -> TurnHookContext:
    """Build a fresh hook view from the current next-turn runtime state."""

    return TurnHookContext(
        turn=turn,
        message=message,
        tool_results=tuple(tool_results),
        new_messages=tuple(new_messages),
        context=AgentContextSnapshot(
            system_prompt=context.system_prompt,
            messages=tuple(context.messages),
            tools=_turn_exposure(context),
            env=context.env,
        ),
    )


def _model_protocol(model: Any) -> str:
    """The model adapter's API identity used for durable request records."""

    capabilities = getattr(model, "capabilities", None)
    api = getattr(capabilities, "api", None)
    value = getattr(api, "value", None)
    if isinstance(value, str) and value:
        return value
    return "legacy"


def _turn_thinking_level(
    model: Any, requested: ThinkingLevel | None
) -> ThinkingLevel | None:
    """Clamp the run's requested level to this turn's model capability."""

    if requested is None:
        return None
    capabilities = getattr(model, "capabilities", None)
    supported = getattr(capabilities, "thinking_levels", None)
    if not isinstance(supported, tuple):
        supported = ()
    return clamp_thinking_level(requested, supported)


async def _run_loop(
    context: AgentContext,
    new_messages: List[Message],
    config: AgentLoopConfig,
    emit: EventSink | None,
    token: CancelToken | None,
    guard: _RunTerminalGuard,
    *,
    turn: int,
) -> AgentLoopResult:
    turn_base = 0

    async def _finish(
        status: AgentRunStatus, error: Optional[str] = None
    ) -> AgentLoopResult:
        return await guard.finish(status, new_messages, config, emit, error=error)

    try:
        pending = await _drain_queue(
            config.get_steering_messages, token, config.deadline_monotonic
        )
    except _StreamAborted:
        return await _finish(AgentRunStatus.ABORTED, error="run aborted")
    except _HookTimedOut:
        return await _finish(
            AgentRunStatus.DEADLINE_EXCEEDED,
            error="run deadline expired while draining steering messages",
        )
    first_iteration = True

    # Outer loop: continue when follow-up messages arrive as the agent stops.
    while True:
        has_more_tool_calls = True
        # Inner loop: process Tool calls and steering messages.
        while has_more_tool_calls or pending:
            if first_iteration:
                first_iteration = False
            else:
                if config.max_turns is not None and turn + 1 >= config.max_turns:
                    return await _finish(
                        AgentRunStatus.MAX_TURNS,
                        error=f"run reached the max turn budget ({config.max_turns})",
                    )
                turn += 1
                if token is not None:
                    token.reset_step_event()
                await emit_to(emit, TurnStart(turn=turn))
                turn_base = len(new_messages)

            for message in pending:
                context.messages.append(message)
                new_messages.append(message)
                await emit_to(emit, MessageStart(message=message))
                await emit_to(emit, MessageEnd(message=message))
            pending = []

            exposure = _turn_exposure(context)
            message, status_override = await _stream_assistant(
                turn, context, new_messages, exposure, config, emit, token
            )

            if status_override is not None or message.failed:
                if config.transaction is not None:
                    await config.transaction.turn_committed(
                        turn, tuple(new_messages[turn_base:])
                    )
                try:
                    await emit_to(
                        emit, TurnEnd(turn=turn, message=message, tool_results=())
                    )
                finally:
                    if token is not None:
                        token.mark_step_complete()
                if status_override is not None:
                    return await _finish(status_override, error=message.error)
                return await _finish(AgentRunStatus.FAILED, error=message.error)

            calls = list(message.tool_calls)
            tool_results: List[ToolResultMessage] = []
            has_more_tool_calls = False
            if calls:
                if message.truncated:
                    # A token-limit stop may carry silently incomplete call
                    # arguments; fail the whole batch without executing.
                    tool_results = await _fail_truncated_calls(
                        calls, turn, context, new_messages, config, emit
                    )
                else:
                    tool_results = await _execute_calls(
                        message,
                        turn,
                        context,
                        new_messages,
                        exposure,
                        config,
                        emit,
                        token,
                    )
                has_more_tool_calls = not _should_terminate_batch(tool_results)

            if config.transaction is not None:
                await config.transaction.turn_committed(
                    turn, tuple(new_messages[turn_base:])
                )
            try:
                await emit_to(
                    emit,
                    TurnEnd(
                        turn=turn,
                        message=message,
                        tool_results=tuple(tool_results),
                    ),
                )
                hook_context = _turn_hook_context(
                    turn=turn,
                    message=message,
                    tool_results=tool_results,
                    new_messages=new_messages,
                    context=context,
                )
                try:
                    if config.prepare_next_turn is not None:
                        update = await _bounded_await(
                            config.prepare_next_turn(hook_context),
                            token,
                            config.deadline_monotonic,
                        )
                        if update is not None:
                            if update.system_prompt is not None:
                                context.system_prompt = update.system_prompt
                            if update.model is not None:
                                config = dataclasses.replace(config, model=update.model)
                            if update.messages is not None:
                                context.messages = list(update.messages)
                            if update.tools is not None:
                                context.tools = update.tools
                            if update.extra_request_options is not None:
                                config = dataclasses.replace(
                                    config,
                                    extra_request_options=dict(
                                        update.extra_request_options
                                    ),
                                )
                            if update.thinking_level is not None:
                                config = dataclasses.replace(
                                    config,
                                    thinking_level=update.thinking_level,
                                )
                    if config.should_stop_after_turn is not None:
                        # prepare_next_turn may replace history, tools, prompt or
                        # request options. Pi's stop hook observes that updated
                        # next-turn context, not the stale pre-update snapshot.
                        hook_context = _turn_hook_context(
                            turn=turn,
                            message=message,
                            tool_results=tool_results,
                            new_messages=new_messages,
                            context=context,
                        )
                        should_stop = await _bounded_await(
                            config.should_stop_after_turn(hook_context),
                            token,
                            config.deadline_monotonic,
                        )
                        if should_stop:
                            return await _finish(AgentRunStatus.COMPLETED)
                except _StreamAborted:
                    return await _finish(
                        AgentRunStatus.ABORTED, error="run aborted"
                    )
                except _HookTimedOut:
                    return await _finish(
                        AgentRunStatus.DEADLINE_EXCEEDED,
                        error="run deadline expired in a turn-boundary hook",
                    )
            finally:
                # A step includes the terminal TurnEnd listener and both
                # Pi-style turn-boundary hooks. Observers waiting for graceful
                # cancellation must not wake while any of them is still live.
                if token is not None:
                    token.mark_step_complete()

            if token is not None and token.is_cancel_requested:
                # A graceful request raised while turn-boundary hooks were
                # settling still belongs to the turn that already committed;
                # it must stop before queue polling or another model request.
                return await _finish(AgentRunStatus.ABORTED, error="run aborted")

            if config.max_turns is not None and turn + 1 >= config.max_turns:
                # Queue hooks are destructive drains. Do not poll them when
                # this run has no capacity for another provider turn: callers
                # retain accepted steering/follow-up for an explicit
                # continuation instead of losing it behind MAX_TURNS.
                if has_more_tool_calls:
                    return await _finish(
                        AgentRunStatus.MAX_TURNS,
                        error=(
                            "run reached the max turn budget "
                            f"({config.max_turns})"
                        ),
                    )
                return await _finish(AgentRunStatus.COMPLETED)

            try:
                pending = await _drain_queue(
                    config.get_steering_messages, token, config.deadline_monotonic
                )
            except _StreamAborted:
                return await _finish(AgentRunStatus.ABORTED, error="run aborted")
            except _HookTimedOut:
                return await _finish(
                    AgentRunStatus.DEADLINE_EXCEEDED,
                    error="run deadline expired while draining steering messages",
                )

        try:
            follow_up = await _drain_queue(
                config.get_follow_up_messages, token, config.deadline_monotonic
            )
        except _StreamAborted:
            return await _finish(AgentRunStatus.ABORTED, error="run aborted")
        except _HookTimedOut:
            return await _finish(
                AgentRunStatus.DEADLINE_EXCEEDED,
                error="run deadline expired while draining follow-up messages",
            )
        if follow_up:
            pending = follow_up
            continue
        break

    return await _finish(AgentRunStatus.COMPLETED)


def _should_terminate_batch(results: Sequence[ToolResultMessage]) -> bool:
    return bool(results) and all(
        item.result.metadata.get("terminate") is True for item in results
    )


async def _fail_truncated_calls(
    calls: Sequence[ToolCall],
    turn: int,
    context: AgentContext,
    new_messages: List[Message],
    config: AgentLoopConfig,
    emit: EventSink | None,
) -> List[ToolResultMessage]:
    messages: List[ToolResultMessage] = []
    for call in calls:
        if config.transaction is not None:
            await config.transaction.tool_started(turn, call)
        result = ToolResult(
            status="error",
            output=None,
            error=(
                f'Tool call "{call.name}" was not executed: the response hit '
                "the output token limit, so its arguments may be truncated. "
                "Re-issue the tool call with complete arguments."
            ),
            metadata={
                "tool_name": call.name,
                "error_category": "arguments_truncated",
                "recoverable": True,
                "started": False,
            },
            call_id=call.id,
        ).frozen()
        if config.transaction is not None:
            await config.transaction.tool_terminal(turn, call, result)
        result_message = ToolResultMessage(
            tool_call_id=call.id,
            tool_name=call.name,
            result=result,
            usage=result.usage,
            added_tool_names=result.added_tool_names,
        )
        messages.append(result_message)
        context.messages.append(result_message)
        new_messages.append(result_message)

    # All call/result pairs are canonical before observational events run, so
    # one broken listener cannot leave the rest of a truncated batch orphaned.
    for call, result_message in zip(calls, messages):
        await emit_to(
            emit,
            ToolExecutionStart(
                tool_call_id=call.id, tool_name=call.name, args=call.arguments
            ),
        )
        await emit_to(
            emit,
            ToolExecutionEnd(
                tool_call_id=call.id,
                tool_name=call.name,
                result=result_message.result,
                is_error=True,
            ),
        )
        await emit_to(emit, MessageStart(message=result_message))
        await emit_to(emit, MessageEnd(message=result_message))
    return messages


async def _execute_calls(
    message: AssistantMessage,
    turn: int,
    context: AgentContext,
    new_messages: List[Message],
    exposure: ToolExposure,
    config: AgentLoopConfig,
    emit: EventSink | None,
    token: CancelToken | None,
) -> List[ToolResultMessage]:
    calls = list(message.tool_calls)
    executor = ToolBatchExecutor(
        exposure,
        ToolExecutionConfig(
            mode=config.tool_execution,
            max_concurrency=config.max_tool_concurrency,
            deadline_monotonic=config.deadline_monotonic,
            cancel_token=token,
            run_id=config.run_id,
            turn=turn,
            env=context.env,
            before_tool_call=config.before_tool_call,
            after_tool_call=config.after_tool_call,
            assistant_message=message,
            agent_context=AgentContextSnapshot(
                system_prompt=context.system_prompt,
                messages=tuple(context.messages),
                tools=exposure,
                env=context.env,
            ),
            extra_runtime_context=_runtime_context(config, context),
        ),
        emit=emit,
        transaction=config.transaction,
    )
    try:
        results = await executor.execute_batch(calls)
    except asyncio.CancelledError:
        # The executor terminalized every admitted call before re-raising;
        # backfill the transcript so ToolCall/ToolResult stay paired, then
        # let the cancellation reach the run terminalization path.
        recovered = executor.last_batch_results
        if recovered is None:
            recovered = [
                ToolResult(
                    status="cancelled",
                    output=None,
                    error="tool call cancelled: caller_cancelled",
                    metadata={
                        "tool_name": call.name,
                        "error_category": "cancelled",
                        "cancel_source": "caller_cancelled",
                        "started": False,
                    },
                ).frozen()
                for call in calls
            ]
        for call, result in zip(calls, recovered):
            result_message = ToolResultMessage(
                tool_call_id=call.id,
                tool_name=call.name,
                result=result,
                usage=result.usage,
                added_tool_names=result.added_tool_names,
            )
            context.messages.append(result_message)
            new_messages.append(result_message)
            try:
                await emit_to(emit, MessageStart(message=result_message))
                await emit_to(emit, MessageEnd(message=result_message))
            except Exception:
                # Caller cancellation remains the externally observable fault;
                # canonical Tool terminals and transcript pairing already won.
                pass
        raise
    except Exception:
        # A Tool event listener may fail after the executor has already made
        # every terminal result durable. Preserve those canonical call/result
        # pairs in the local transcript before propagating the implementation
        # fault; no further observational events are attempted here.
        recovered = executor.last_batch_results
        if recovered is not None:
            for call, result in zip(calls, recovered):
                result_message = ToolResultMessage(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    result=result,
                    usage=result.usage,
                    added_tool_names=result.added_tool_names,
                )
                context.messages.append(result_message)
                new_messages.append(result_message)
        raise
    messages: List[ToolResultMessage] = []
    for call, result in zip(calls, results):
        result_message = ToolResultMessage(
            tool_call_id=call.id,
            tool_name=call.name,
            result=result,
            usage=result.usage,
            added_tool_names=result.added_tool_names,
        )
        messages.append(result_message)
        context.messages.append(result_message)
        new_messages.append(result_message)
    for result_message in messages:
        await emit_to(emit, MessageStart(message=result_message))
        await emit_to(emit, MessageEnd(message=result_message))
    return messages


def _runtime_context(config: AgentLoopConfig, context: AgentContext) -> Dict[str, Any]:
    # Product context may contribute domain values, but an initialized Env is
    # the authority for its permission context. Merge that fact last so a
    # caller cannot replace the frozen turn's Scope/Permission boundary.
    merged: Dict[str, Any] = dict(config.runtime_context)
    env = context.env
    if env is not None:
        permission_context = getattr(env, "tool_permission_context", None)
        if permission_context is not None:
            merged["permission_context"] = permission_context
    return merged


async def _stream_assistant(
    turn: int,
    context: AgentContext,
    new_messages: List[Message],
    exposure: ToolExposure,
    config: AgentLoopConfig,
    emit: EventSink | None,
    token: CancelToken | None,
) -> Tuple[AssistantMessage, Optional[AgentRunStatus]]:
    """Run one model transaction and return its terminal assistant message.

    The terminal message is appended to both the context and the run's new
    messages on every exit path, including abort, deadline, model failure
    and caller cancellation. The second tuple element overrides the run
    status for abort and deadline terminals; ``None`` means normal turn
    processing continues.
    """

    model = config.model
    messages = list(context.messages)
    if config.transform_context is not None:
        try:
            transformed = await _bounded_await(
                config.transform_context(messages),
                token,
                config.deadline_monotonic,
            )
        except _StreamAborted:
            message = AssistantMessage(
                error="run aborted before model admission",
                model_name=model.model,
                provider=model.provider_name,
            )
            await _finalize_model_message(
                message, context, new_messages, emit, config, turn, None
            )
            return message, AgentRunStatus.ABORTED
        except _HookTimedOut:
            message = AssistantMessage(
                error="model request deadline expired before admission",
                model_name=model.model,
                provider=model.provider_name,
            )
            await _finalize_model_message(
                message, context, new_messages, emit, config, turn, None
            )
            return message, AgentRunStatus.DEADLINE_EXCEEDED
        messages = list(transformed)

    wire = [
        message_to_wire(message)
        for message in messages
        if not (isinstance(message, AssistantMessage) and message.failed)
    ]
    if context.system_prompt:
        wire.insert(0, {"role": "system", "content": context.system_prompt})

    options: Dict[str, Any] = dict(config.extra_request_options)
    protocol = _model_protocol(model)
    if len(exposure) > 0:
        built = model.build_tool_schema_request_options(
            exposure.get_all_specs(),
            protocol=protocol,
            delivery="api_parameter",
        )
        if built:
            options.update(built)

    continuation = _tail_continuation(context.messages, config)
    request = ModelRequest(
        run_id=config.run_id,
        transaction_id=f"{config.run_id}:turn:{turn}:{uuid.uuid4().hex[:8]}",
        provider=model.provider_name,
        model=model.model,
        protocol=protocol,
        messages=tuple(wire),
        options=options,
        deadline_monotonic=config.deadline_monotonic,
        continuation=continuation,
        thinking_level=_turn_thinking_level(model, config.thinking_level),
    )

    if token is not None and token.immediate_requested:
        message = AssistantMessage(
            error="run aborted before model admission",
            model_name=model.model,
            provider=model.provider_name,
        )
        await _finalize_model_message(
            message, context, new_messages, emit, config, turn, request
        )
        return message, AgentRunStatus.ABORTED

    if config.deadline_monotonic is not None and (
        time.monotonic() >= config.deadline_monotonic
    ):
        message = AssistantMessage(
            error="model request deadline expired before admission",
            model_name=model.model,
            provider=model.provider_name,
        )
        await _finalize_model_message(
            message, context, new_messages, emit, config, turn, request
        )
        return message, AgentRunStatus.DEADLINE_EXCEEDED

    accumulated_text: List[str] = []
    accumulated_reasoning: List[str] = []
    partial = AssistantMessage(model_name=model.model, provider=model.provider_name)
    started = False
    terminal_seen = False
    terminal_error: Optional[str] = None
    final_usage: Any = None
    final_tool_calls: Optional[List[Dict[str, Any]]] = None
    final_native_items: Optional[List[Dict[str, Any]]] = None
    final_finish_reason: Optional[str] = None
    final_metadata: Dict[str, Any] = {}
    final_continuation: Optional[ModelContinuation] = None
    aborted = False

    stream_iter = model.stream(request)
    iterator = stream_iter.__aiter__()
    try:
        while True:
            try:
                chunk = await _next_chunk(
                    iterator, token, config.deadline_monotonic
                )
            except StopAsyncIteration:
                break
            except _StreamAborted:
                aborted = True
                break
            if not isinstance(chunk, ModelStreamEvent):
                raise TypeError("Model.stream() must yield ModelStreamEvent values")
            if terminal_seen:
                raise ModelTransportError(
                    "model stream emitted an event after its terminal event",
                    attempts=1,
                    retryable=False,
                )
            observable = bool(
                chunk.text
                or chunk.reasoning_content
                or chunk.is_final
                or chunk.tool_calls
                or chunk.native_items
                or chunk.event_type
            )
            if observable and not started:
                started = True
                context.messages.append(partial)
                await _emit_during_model_stream(
                    emit, MessageStart(message=partial)
                )
            if chunk.text:
                accumulated_text.append(chunk.text)
            if chunk.reasoning_content:
                accumulated_reasoning.append(chunk.reasoning_content)
            if chunk.type in (
                ModelStreamEventType.TEXT_DELTA,
                ModelStreamEventType.REASONING_DELTA,
                ModelStreamEventType.TOOL_CALL_DELTA,
                ModelStreamEventType.OUTPUT_ITEM,
            ):
                partial = dataclasses.replace(
                    partial,
                    text="".join(accumulated_text),
                    reasoning_content=(
                        "".join(accumulated_reasoning)
                        if accumulated_reasoning
                        else None
                    ),
                )
                if started:
                    context.messages[-1] = partial
                await _emit_during_model_stream(
                    emit,
                    MessageUpdate(message=partial, stream_event=chunk),
                )
            if chunk.type is ModelStreamEventType.USAGE:
                # Standalone usage events accumulate; the terminal event's
                # usage, when present, wins over earlier reports.
                if chunk.usage is not None:
                    final_usage = chunk.usage
            if chunk.is_final:
                terminal_seen = True
                if chunk.type is ModelStreamEventType.FAILED:
                    terminal_error = str(chunk.error or "model stream failed")
                    continue
                if chunk.usage is not None:
                    final_usage = chunk.usage
                if chunk.tool_calls is not None:
                    final_tool_calls = list(chunk.tool_calls)
                if chunk.native_items is not None:
                    final_native_items = list(chunk.native_items)
                if chunk.finish_reason is not None:
                    final_finish_reason = str(chunk.finish_reason)
                final_continuation = chunk.continuation
                final_metadata = dict(chunk.event_metadata)

        if aborted:
            message = dataclasses.replace(
                partial,
                error="run aborted during model streaming",
            )
        elif not terminal_seen:
            raise ModelTransportError(
                "model stream ended before a terminal event",
                attempts=1,
                retryable=True,
            )
        elif terminal_error is not None:
            message = dataclasses.replace(
                partial,
                error=terminal_error,
                metadata=final_metadata,
            )
        else:
            message = assistant_from_response(
                ModelResponse(
                    text="".join(accumulated_text),
                    usage=final_usage,
                    finish_reason=final_finish_reason,
                    tool_calls=final_tool_calls,
                    model_name=model.model,
                    provider=model.provider_name,
                    metadata=final_metadata,
                    reasoning_content=(
                        "".join(accumulated_reasoning)
                        if accumulated_reasoning
                        else None
                    ),
                    native_items=final_native_items,
                    continuation=final_continuation,
                )
            )
    except asyncio.CancelledError:
        # Caller cancellation of the loop task: close the open model
        # transaction before the cancellation reaches the run boundary.
        message = dataclasses.replace(
            partial,
            error="run cancelled during model streaming",
        )
        await _finalize_model_message(
            message, context, new_messages, emit, config, turn, request,
            started=started,
        )
        raise
    except _EventSinkFault as exc:
        message = dataclasses.replace(
            partial,
            error=str(exc.cause) or "agent event listener failed",
        )
        # The event stream is already faulty. Commit the open model
        # transaction without dispatching further message events, then let the
        # original listener fault reach run terminalization.
        await _finalize_model_message(
            message,
            context,
            new_messages,
            None,
            config,
            turn,
            request,
            started=started,
        )
        try:
            # The start/update event may have reached earlier subscribers
            # before one failed. Still offer the terminal projection after
            # the canonical model record is safe; preserve the first fault.
            await emit_to(emit, MessageEnd(message=message))
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        raise exc.cause
    except Exception as exc:
        message = dataclasses.replace(
            partial,
            error=str(exc) or "model stream failed",
        )
        if isinstance(exc, ModelRequestDeadlineExceeded):
            status: Optional[AgentRunStatus] = AgentRunStatus.DEADLINE_EXCEEDED
        else:
            status = AgentRunStatus.FAILED
        await _finalize_model_message(
            message, context, new_messages, emit, config, turn, request,
            started=started,
        )
        return message, status
    finally:
        close = getattr(stream_iter, "aclose", None)
        if callable(close):
            try:
                await close()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

    await _finalize_model_message(
        message, context, new_messages, emit, config, turn, request,
        started=started,
    )
    if aborted:
        return message, AgentRunStatus.ABORTED
    return message, None


async def _finalize_model_message(
    message: AssistantMessage,
    context: AgentContext,
    new_messages: List[Message],
    emit: EventSink | None,
    config: AgentLoopConfig,
    turn: int,
    request: Optional[ModelRequest],
    *,
    started: bool = False,
) -> None:
    """Land one terminal assistant message in transcript, events and journal."""

    if started:
        context.messages[-1] = message
    else:
        context.messages.append(message)
    new_messages.append(message)
    if request is not None and config.transaction is not None:
        await config.transaction.model_terminal(turn, request, message)
    if not started:
        await emit_to(emit, MessageStart(message=message))
    await emit_to(emit, MessageEnd(message=message))


def _tail_continuation(
    messages: Sequence[Message], config: AgentLoopConfig
) -> Optional[ModelContinuation]:
    """Offer the latest effective assistant continuation for the next request.

    New user/Tool messages may sit on top of the last assistant message, and
    failed assistant messages never reach the wire; the continuation chains
    from the latest non-failed assistant message instead.
    """

    for message in reversed(messages):
        if not isinstance(message, AssistantMessage):
            continue
        if message.failed:
            continue
        continuation = message.continuation
        if continuation is None:
            return None
        model = config.model
        if (
            continuation.run_id == config.run_id
            and continuation.provider == model.provider_name
            and continuation.model == model.model
            and continuation.protocol == _model_protocol(model)
        ):
            return continuation
        return None
    return None


async def _next_chunk(
    iterator: Any,
    token: CancelToken | None,
    deadline_monotonic: Optional[float],
) -> ModelStreamEvent:
    """Await the next stream event, racing immediate cancellation/deadline.

    ``after_step`` never interrupts an in-flight stream; only ``"immediate"``
    cancellation and the absolute deadline do.
    """

    getter = asyncio.create_task(iterator.__anext__())
    watcher: Optional[asyncio.Task[bool]] = None
    if token is not None:
        watcher = asyncio.create_task(token.wait_immediate())
    tasks: set[asyncio.Task[Any]] = {getter}
    if watcher is not None:
        tasks.add(watcher)
    remaining = (
        None
        if deadline_monotonic is None
        else max(0.0, deadline_monotonic - time.monotonic())
    )
    try:
        done, pending = await asyncio.wait(
            tasks, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
        )
    except asyncio.CancelledError:
        getter.cancel()
        if watcher is not None:
            watcher.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    if not done:
        getter.cancel()
        if watcher is not None:
            watcher.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise ModelRequestDeadlineExceeded("model request deadline expired")
    if getter in done:
        if watcher is not None:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        return getter.result()
    getter.cancel()
    await asyncio.gather(getter, return_exceptions=True)
    raise _StreamAborted


__all__ = [
    "AgentContext",
    "AgentEventStream",
    "AgentLoopConfig",
    "AgentLoopResult",
    "AgentRunStatus",
    "NextTurnUpdate",
    "TurnHookContext",
    "TurnTransactionBoundary",
    "agent_loop",
    "agent_loop_continue",
    "run_agent_loop",
    "run_agent_loop_continue",
]
