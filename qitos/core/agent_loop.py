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
- Cancellation is cooperative through ``CancelToken``: aborted runs end with a
  terminal assistant message and an ``ABORTED`` result.
- Persistence faults from the transaction boundary and caller cancellation of
  the loop task propagate as exceptions.
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
from .tool_executor import (
    AfterToolCallHook,
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
    """The loop's working context: system prompt, transcript, tools, env."""

    system_prompt: str = ""
    messages: List[Message] = field(default_factory=list)
    tools: Optional[ToolExposure] = None
    env: Optional[Env] = None


@dataclass(frozen=True, slots=True)
class TurnHookContext:
    """Immutable view handed to turn-boundary hooks."""

    turn: int
    message: AssistantMessage
    tool_results: Tuple[ToolResultMessage, ...]
    new_messages: Tuple[Message, ...]


@dataclass(frozen=True, slots=True)
class NextTurnUpdate:
    """Optional per-turn replacement of the system prompt or model."""

    system_prompt: Optional[str] = None
    model: Optional["Model"] = None


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

    All hooks must not throw; ``before_tool_call``/``after_tool_call``
    exceptions are converted to Tool error results, while
    ``transform_context``/``should_stop_after_turn``/``prepare_next_turn``
    exceptions fail the run.
    """

    model: "Model"
    run_id: str
    tool_execution: Literal["sequential", "parallel"] = "sequential"
    max_tool_concurrency: int = 8
    max_turns: Optional[int] = None
    deadline_monotonic: Optional[float] = None
    extra_request_options: Mapping[str, Any] = field(default_factory=dict)
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


class _StreamAborted(Exception):
    """Internal signal: the cancel token fired while streaming."""


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
    await emit_to(emit, AgentStart())
    await emit_to(emit, TurnStart(turn=0))
    for prompt in prompts:
        await emit_to(emit, MessageStart(message=prompt))
        await emit_to(emit, MessageEnd(message=prompt))
    return await _run_loop(
        context, new_messages, config, emit, cancel_token, turn=0
    )


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
    await emit_to(emit, AgentStart())
    await emit_to(emit, TurnStart(turn=0))
    return await _run_loop(
        context, new_messages, config, emit, cancel_token, turn=0
    )


def agent_loop(
    prompts: Sequence[Message],
    context: AgentContext,
    config: AgentLoopConfig,
    cancel_token: CancelToken | None = None,
) -> "AgentEventStream":
    """Start a loop run and return its push-based event stream."""

    async def _runner() -> AgentLoopResult:
        return await run_agent_loop(
            prompts, context, config, stream.push, cancel_token
        )

    stream = AgentEventStream(_runner(), cancel_token=cancel_token)
    return stream


def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    cancel_token: CancelToken | None = None,
) -> "AgentEventStream":
    """Continue a loop run from the context tail and return its event stream."""

    async def _runner() -> AgentLoopResult:
        return await run_agent_loop_continue(
            context, config, stream.push, cancel_token
        )

    stream = AgentEventStream(_runner(), cancel_token=cancel_token)
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


async def _drain(hook: QueueDrainHook | None) -> List[Message]:
    if hook is None:
        return []
    drained = hook()
    if inspect.isawaitable(drained):
        drained = await drained
    return [message for message in list(drained or [])]


async def _run_loop(
    context: AgentContext,
    new_messages: List[Message],
    config: AgentLoopConfig,
    emit: EventSink | None,
    token: CancelToken | None,
    *,
    turn: int,
) -> AgentLoopResult:
    turn_base = 0
    pending = await _drain(config.get_steering_messages)
    first_iteration = True

    async def _finish(
        status: AgentRunStatus, error: Optional[str] = None
    ) -> AgentLoopResult:
        result = AgentLoopResult(
            status=status, messages=tuple(new_messages), error=error
        )
        if config.transaction is not None:
            await config.transaction.run_terminal(result)
        await emit_to(emit, AgentEnd(messages=tuple(new_messages)))
        return result

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
                await emit_to(emit, TurnStart(turn=turn))
                turn_base = len(new_messages)

            for message in pending:
                await emit_to(emit, MessageStart(message=message))
                await emit_to(emit, MessageEnd(message=message))
                context.messages.append(message)
                new_messages.append(message)
            pending = []

            message, status_override = await _stream_assistant(
                turn, context, config, emit, token
            )
            new_messages.append(message)

            if status_override is not None or message.failed:
                await emit_to(
                    emit, TurnEnd(turn=turn, message=message, tool_results=())
                )
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
                        calls, turn, config, emit
                    )
                else:
                    tool_results = await _execute_calls(
                        calls, turn, context, config, emit, token
                    )
                has_more_tool_calls = not _should_terminate_batch(tool_results)
                for result_message in tool_results:
                    context.messages.append(result_message)
                    new_messages.append(result_message)

            await emit_to(
                emit,
                TurnEnd(
                    turn=turn, message=message, tool_results=tuple(tool_results)
                ),
            )
            if config.transaction is not None:
                await config.transaction.turn_committed(
                    turn, tuple(new_messages[turn_base:])
                )

            hook_context = TurnHookContext(
                turn=turn,
                message=message,
                tool_results=tuple(tool_results),
                new_messages=tuple(new_messages),
            )
            if config.prepare_next_turn is not None:
                update = config.prepare_next_turn(hook_context)
                if inspect.isawaitable(update):
                    update = await update
                if update is not None:
                    if update.system_prompt is not None:
                        context.system_prompt = update.system_prompt
                    if update.model is not None:
                        config = dataclasses.replace(config, model=update.model)
            if config.should_stop_after_turn is not None:
                should_stop = config.should_stop_after_turn(hook_context)
                if inspect.isawaitable(should_stop):
                    should_stop = await should_stop
                if should_stop:
                    return await _finish(AgentRunStatus.COMPLETED)

            pending = await _drain(config.get_steering_messages)

        follow_up = await _drain(config.get_follow_up_messages)
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
    config: AgentLoopConfig,
    emit: EventSink | None,
) -> List[ToolResultMessage]:
    messages: List[ToolResultMessage] = []
    for call in calls:
        await emit_to(
            emit,
            ToolExecutionStart(
                tool_call_id=call.id, tool_name=call.name, args=call.arguments
            ),
        )
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
        )
        if config.transaction is not None:
            await config.transaction.tool_terminal(turn, call, result)
        await emit_to(
            emit,
            ToolExecutionEnd(
                tool_call_id=call.id,
                tool_name=call.name,
                result=result,
                is_error=True,
            ),
        )
        result_message = ToolResultMessage(
            tool_call_id=call.id, tool_name=call.name, result=result
        )
        await emit_to(emit, MessageStart(message=result_message))
        await emit_to(emit, MessageEnd(message=result_message))
        messages.append(result_message)
    return messages


async def _execute_calls(
    calls: Sequence[ToolCall],
    turn: int,
    context: AgentContext,
    config: AgentLoopConfig,
    emit: EventSink | None,
    token: CancelToken | None,
) -> List[ToolResultMessage]:
    if context.tools is None:
        exposure = ToolRegistry().freeze()
    else:
        exposure = context.tools
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
            extra_runtime_context=_runtime_context(config, context),
        ),
        emit=emit,
        transaction=config.transaction,
    )
    results = await executor.execute_batch(calls)
    messages: List[ToolResultMessage] = []
    for call, result in zip(calls, results):
        result_message = ToolResultMessage(
            tool_call_id=call.id, tool_name=call.name, result=result
        )
        await emit_to(emit, MessageStart(message=result_message))
        await emit_to(emit, MessageEnd(message=result_message))
        messages.append(result_message)
    return messages


def _runtime_context(config: AgentLoopConfig, context: AgentContext) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    env = context.env
    if env is not None:
        permission_context = getattr(env, "tool_permission_context", None)
        if permission_context is not None:
            merged["permission_context"] = permission_context
    merged.update(dict(config.runtime_context))
    return merged


async def _stream_assistant(
    turn: int,
    context: AgentContext,
    config: AgentLoopConfig,
    emit: EventSink | None,
    token: CancelToken | None,
) -> Tuple[AssistantMessage, Optional[AgentRunStatus]]:
    """Run one model transaction and return its terminal assistant message.

    The second tuple element overrides the run status for abort and deadline
    terminals; ``None`` means normal turn processing continues.
    """

    model = config.model
    messages = list(context.messages)
    if config.transform_context is not None:
        transformed = config.transform_context(messages)
        if inspect.isawaitable(transformed):
            transformed = await transformed
        messages = list(transformed)

    wire = [
        message_to_wire(message)
        for message in messages
        if not (isinstance(message, AssistantMessage) and message.failed)
    ]
    if context.system_prompt:
        wire.insert(0, {"role": "system", "content": context.system_prompt})

    options: Dict[str, Any] = dict(config.extra_request_options)
    if context.tools is not None and len(context.tools) > 0:
        built = model.build_tool_schema_request_options(
            context.tools.get_all_specs(), protocol=None, delivery="api_parameter"
        )
        if built:
            options.update(built)

    continuation = _tail_continuation(context.messages, config)
    request = ModelRequest(
        run_id=config.run_id,
        transaction_id=f"{config.run_id}:turn:{turn}:{uuid.uuid4().hex[:8]}",
        provider=model.provider_name,
        model=model.model,
        protocol="native",
        messages=tuple(wire),
        options=options,
        deadline_monotonic=config.deadline_monotonic,
        continuation=continuation,
    )

    async def _record(
        message: AssistantMessage,
    ) -> Tuple[AssistantMessage, Optional[AgentRunStatus]]:
        if config.transaction is not None:
            await config.transaction.model_terminal(turn, request, message)
        return message, None

    if token is not None and token.is_cancel_requested:
        message = AssistantMessage(
            error="run aborted before model admission",
            model_name=model.model,
            provider=model.provider_name,
        )
        context.messages.append(message)
        await emit_to(emit, MessageStart(message=message))
        await emit_to(emit, MessageEnd(message=message))
        await _record(message)
        return message, AgentRunStatus.ABORTED

    if config.deadline_monotonic is not None and (
        time.monotonic() >= config.deadline_monotonic
    ):
        message = AssistantMessage(
            error="model request deadline expired before admission",
            model_name=model.model,
            provider=model.provider_name,
        )
        context.messages.append(message)
        await emit_to(emit, MessageStart(message=message))
        await emit_to(emit, MessageEnd(message=message))
        await _record(message)
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
                await emit_to(emit, MessageStart(message=partial))
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
                await emit_to(
                    emit,
                    MessageUpdate(message=partial, stream_event=chunk),
                )
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
        raise
    except Exception as exc:
        message = dataclasses.replace(
            partial,
            error=str(exc) or "model stream failed",
        )
        if isinstance(exc, ModelRequestDeadlineExceeded):
            status: Optional[AgentRunStatus] = AgentRunStatus.DEADLINE_EXCEEDED
        else:
            status = AgentRunStatus.FAILED
        if started:
            context.messages[-1] = message
        else:
            context.messages.append(message)
            await emit_to(emit, MessageStart(message=message))
        await emit_to(emit, MessageEnd(message=message))
        await _record(message)
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

    if started:
        context.messages[-1] = message
    else:
        context.messages.append(message)
        await emit_to(emit, MessageStart(message=message))
    await emit_to(emit, MessageEnd(message=message))
    await _record(message)
    if aborted:
        return message, AgentRunStatus.ABORTED
    return message, None


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
            and continuation.protocol == "native"
        ):
            return continuation
        return None
    return None


async def _next_chunk(
    iterator: Any,
    token: CancelToken | None,
    deadline_monotonic: Optional[float],
) -> ModelStreamEvent:
    """Await the next stream event, racing cancellation and the deadline."""

    getter = asyncio.create_task(iterator.__anext__())
    watcher: Optional[asyncio.Task[bool]] = None
    if token is not None:
        watcher = asyncio.create_task(token.wait_cancelled())
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
