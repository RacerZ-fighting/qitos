"""Async Engine for non-blocking AgentModule execution."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, AsyncIterator, Callable, Generic, List, Optional, TypeVar

from ..core.agent_module import AgentModule
from ..core.runtime_input import RuntimeInput
from ..core.state import StateSchema
from .engine import Engine, EngineResult
from .events import EngineEvent, EngineEventType, EventStream
from .hooks import HookContext
from .states import RuntimeEvent, RuntimePhase, StepRecord
from .stream.transformer import StreamTransformer, TransformerChain, TransformerOutput

StateT = TypeVar("StateT", bound=StateSchema)
ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


class AsyncEngine(Generic[StateT, ObservationT, ActionT]):
    """Async execution kernel for AgentModule workflows.

    Provides the same interface as Engine but with async execution:
    - ``await async_engine.arun(task)`` returns EngineResult
    - ``async for event in async_engine.arun_stream(task)`` yields EngineEvents

    Internally delegates to a sync Engine running on a daemon worker thread.
    Cancellation remains cooperative, while an uncooperative provider or tool
    cannot keep the hosting asyncio loop alive during process shutdown.
    """

    def __init__(
        self,
        agent: AgentModule[StateT, ObservationT, ActionT],
        **engine_kwargs: Any,
    ):
        self._engine: Engine[StateT, ObservationT, ActionT] = Engine(
            agent=agent,
            **engine_kwargs,
        )
        self.agent = self._engine.agent
        self.event_stream: Optional[EventStream] = None

    @property
    def engine(self) -> Engine[StateT, ObservationT, ActionT]:
        return self._engine

    async def arun(self, task: str, **kwargs: Any) -> EngineResult[StateT]:
        """Run the agent loop asynchronously, returning the final result.

        This executes the sync Engine.run() on a daemon thread to avoid
        blocking the event loop or interpreter shutdown.

        If the awaiting task is cancelled, the same cancellation signal is
        propagated to the underlying Engine. Note that a sync tool already
        executing inside the worker thread cannot be forcibly killed by
        Python; it is asked to stop and drains on its own.
        """
        run_task = asyncio.ensure_future(
            self._run_in_daemon_thread(lambda: self._engine.run(task, **kwargs))
        )
        try:
            return await asyncio.shield(run_task)
        except asyncio.CancelledError:
            self.cancel("immediate")
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass
            raise

    def cancel(self, mode: str = "immediate") -> None:
        """Request cancellation of the in-flight run.

        Thread-safe, idempotent, and usable after the run has started.
        Delegates to the underlying Engine's cancellation token.
        """
        self._engine.cancel(mode)

    @property
    def active_run_id(self) -> str:
        """Return the underlying Engine's active run id."""

        return self._engine.active_run_id

    def post_runtime_event(self, event: RuntimeInput, *, run_id: str) -> bool:
        """Post one event to the exact underlying Engine run."""

        return self._engine.post_runtime_event(event, run_id=run_id)

    async def _run_in_daemon_thread(
        self,
        callback: Callable[[], EngineResult[StateT]],
    ) -> EngineResult[StateT]:
        """Run one blocking Engine call without owning loop shutdown."""

        loop = asyncio.get_running_loop()
        result_future: asyncio.Future[EngineResult[StateT]] = loop.create_future()

        def _set_result(result: EngineResult[StateT]) -> None:
            if not result_future.done():
                result_future.set_result(result)

        def _set_exception(exc: BaseException) -> None:
            if not result_future.done():
                result_future.set_exception(exc)

        def _worker() -> None:
            try:
                result = callback()
            except BaseException as exc:
                try:
                    loop.call_soon_threadsafe(_set_exception, exc)
                except RuntimeError:
                    pass
            else:
                try:
                    loop.call_soon_threadsafe(_set_result, result)
                except RuntimeError:
                    pass

        threading.Thread(
            target=_worker,
            name="qitos-engine-run",
            daemon=True,
        ).start()
        return await result_future

    async def _stream_events(
        self,
        task: str,
        **kwargs: Any,
    ) -> AsyncIterator[EngineEvent]:
        """Bridge one Engine run into a cancellation-safe event iterator."""

        stream = EventStream()
        self.event_stream = stream
        hook = _StreamBridgeHook(stream)
        self._engine.hooks.append(hook)

        def _run() -> EngineResult[StateT]:
            try:
                return self._engine.run(task, **kwargs)
            finally:
                stream.close()

        run_task = asyncio.ensure_future(self._run_in_daemon_thread(_run))
        stream_exhausted = False
        try:
            async for event in stream:
                yield event
            stream_exhausted = True
        finally:
            self._engine.hooks = [
                existing for existing in self._engine.hooks if existing is not hook
            ]
            if self.event_stream is stream:
                self.event_stream = None

            if stream_exhausted:
                await run_task
            else:
                if not run_task.done():
                    self.cancel("immediate")
                    run_task.cancel()
                try:
                    await run_task
                except (asyncio.CancelledError, Exception):
                    pass

    async def arun_stream(
        self,
        task: str,
        *,
        transformers: Optional[List[StreamTransformer]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[EngineEvent | TransformerOutput]:
        """Run the agent loop and yield structured events as they occur.

        A sync Engine runs on a daemon worker. Engine hooks bridge events
        into the EventStream for async consumption.

        Parameters
        ----------
        task : str
            The task to execute.
        transformers : list of StreamTransformer, optional
            If provided, each event is passed through the transformer chain
            and TransformerOutput values are yielded instead of raw events.
            Events that produce no TransformerOutput are suppressed.
        """
        chain = TransformerChain(transformers) if transformers else None

        if chain:
            chain.on_run_start()

        try:
            async for event in self._stream_events(task, **kwargs):
                if chain:
                    outputs = await chain.aprocess(event)
                    for output in outputs:
                        yield output
                else:
                    yield event
        finally:
            if chain:
                chain.on_run_end()

    def run(self, task: str, **kwargs: Any) -> EngineResult[StateT]:
        """Synchronous run (delegates to underlying Engine)."""
        return self._engine.run(task, **kwargs)

    async def arun_stream_tokens(
        self, task: str, **kwargs: Any
    ) -> AsyncIterator[EngineEvent]:
        """Run the agent loop with token-level streaming.

        Yields EngineEvents including STEP_STREAM events that carry
        ModelStreamChunk payloads for real-time token display.

        Uses the same event lifecycle as ``arun_stream()`` so cancellation and
        terminal delivery remain identical.
        """
        async for event in self._stream_events(task, **kwargs):
            yield event


class _StreamBridgeHook:
    """Engine hook that bridges RuntimeEvents into an EventStream."""

    def __init__(self, stream: EventStream) -> None:
        self._stream = stream

    def on_run_start(self, task: str, state: Any, engine: Engine) -> None:
        self._stream.emit_sync(
            EngineEvent(
                event_type=EngineEventType.RUN_START,
                payload={"task": task},
            )
        )

    def on_run_end(self, result: EngineResult, engine: Engine) -> None:
        self._stream.emit_sync(
            EngineEvent(
                event_type=EngineEventType.RUN_END,
                step_id=result.step_count,
                payload={
                    "step_count": result.step_count,
                    "runtime_seconds": result.runtime_seconds,
                    "total_tokens": result.total_tokens,
                    "stop_reason": result.state.stop_reason
                    if result.state
                    else None,
                },
            )
        )

    def on_before_step(self, ctx: HookContext, engine: Engine) -> None:
        agent_id = getattr(ctx.record, "agent_id", None) if ctx.record else None
        self._stream.emit_sync(
            EngineEvent(
                event_type=EngineEventType.STEP_START,
                step_id=ctx.step_id,
                agent_id=agent_id,
                phase=ctx.phase,
                payload={"phase": ctx.phase.value if ctx.phase else None},
            )
        )

    def on_after_step(self, ctx: HookContext, engine: Engine) -> None:
        agent_id = getattr(ctx.record, "agent_id", None) if ctx.record else None
        self._stream.emit_sync(
            EngineEvent(
                event_type=EngineEventType.STEP_END,
                step_id=ctx.step_id,
                agent_id=agent_id,
                phase=ctx.phase,
                payload={
                    "phase": ctx.phase.value if ctx.phase else None,
                    "stop_reason": ctx.stop_reason,
                },
            )
        )

    def on_before_decide(self, ctx: HookContext, engine: Engine) -> None:
        self._stream.emit_sync(
            EngineEvent(
                event_type=EngineEventType.DECIDE,
                step_id=ctx.step_id,
                phase=RuntimePhase.DECIDE,
                payload={"stage": "start"},
            )
        )

    def on_after_decide(self, ctx: HookContext, engine: Engine) -> None:
        mode = None
        if ctx.decision is not None:
            mode = getattr(ctx.decision, "mode", None)
        self._stream.emit_sync(
            EngineEvent(
                event_type=EngineEventType.DECIDE,
                step_id=ctx.step_id,
                phase=RuntimePhase.DECIDE,
                payload={"stage": "end", "mode": mode},
            )
        )

    def on_before_act(self, ctx: HookContext, engine: Engine) -> None:
        self._stream.emit_sync(
            EngineEvent(
                event_type=EngineEventType.ACT,
                step_id=ctx.step_id,
                phase=RuntimePhase.ACT,
                payload={"stage": "start"},
            )
        )

    def on_after_act(self, ctx: HookContext, engine: Engine) -> None:
        self._stream.emit_sync(
            EngineEvent(
                event_type=EngineEventType.ACT,
                step_id=ctx.step_id,
                phase=RuntimePhase.ACT,
                payload={"stage": "end"},
            )
        )

    def on_event(
        self, event: RuntimeEvent, state: Any, record: Optional[StepRecord], engine: Engine
    ) -> None:
        phase_val = getattr(event, "phase", None)
        phase_str = phase_val.value if phase_val else ""
        event_type = EngineEventType.PHASE_END

        if "HANDOFF" in phase_str:
            event_type = EngineEventType.HANDOFF
        elif "DELEGATE" in phase_str:
            event_type = EngineEventType.DELEGATE
        elif "FANOUT" in phase_str:
            event_type = EngineEventType.FANOUT

        agent_id = getattr(record, "agent_id", None) if record else None
        self._stream.emit_sync(
            EngineEvent(
                event_type=event_type,
                step_id=getattr(event, "step_id", 0),
                agent_id=agent_id,
                phase=phase_val,
                ok=getattr(event, "ok", True),
                payload=getattr(event, "payload", {}) or {},
                error=getattr(event, "error", None),
            )
        )


__all__ = ["AsyncEngine"]
