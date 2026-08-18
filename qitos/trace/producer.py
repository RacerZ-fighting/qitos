"""Reattach the TraceWriter to the Agent façade event stream.

The producer subscribes to one Agent run and projects its lifecycle into
the existing three-file trace layout (``manifest.json``, ``events.jsonl``,
``steps.jsonl``): every committed turn publishes its staged events with one
``TraceStep`` (``step_id`` = turn) through ``write_transaction``, while
lifecycle events go through ``write_event``. A turn that never commits
(abort or fault mid-turn) is dropped instead of being published as a
misleading partial transaction. ``finalize`` maps the loop's terminal
status onto the manifest vocabulary ``qita`` reads.

Trace artifacts stay an observational projection: nothing in Session
recovery ever reads them, and payloads carry call ids, names and statuses —
never credentials (the writer applies the established trace redaction).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ..core.agent import Agent
from ..core.agent_events import (
    AgentEnd,
    AgentEvent,
    AgentStart,
    MessageEnd,
    ToolExecutionEnd,
    ToolExecutionStart,
    TurnEnd,
)
from ..core.agent_loop import AgentLoopResult, AgentRunStatus
from ..core.message import (
    AssistantMessage,
    ContextMessage,
    TextContent,
    ToolResultMessage,
    UserMessage,
)
from .events import TraceEvent, TraceStep
from .writer import TraceWriter

if TYPE_CHECKING:
    from ..models.base import Model

#: Loop-vocabulary trace phases (the retired Engine's RuntimePhase names do
#: not return; this small set is what the minimal loop can observe).
PHASE_AGENT = "agent"
PHASE_INPUT = "input"
PHASE_CONTEXT = "context"
PHASE_MODEL = "model"
PHASE_TOOL = "tool"
PHASE_TURN = "turn"

_LIFECYCLE_STEP_ID = -1
_FINAL_TEXT_MAX_CHARS = 4_000

_STATUS_BY_RUN_STATUS = {
    AgentRunStatus.COMPLETED: "completed",
    AgentRunStatus.FAILED: "failed",
    AgentRunStatus.ABORTED: "stopped",
    AgentRunStatus.MAX_TURNS: "stopped",
    AgentRunStatus.DEADLINE_EXCEEDED: "stopped",
}


def trace_producer_metadata(model: "Model") -> Dict[str, Any]:
    """Return the manifest producer metadata derivable from the run's model."""

    from .. import __version__

    return {
        "model_id": getattr(model, "model", "unknown"),
        "package_version": __version__,
        "agent_name": "qitos.agent",
        "provenance": {
            "producer": "qitos.trace.producer",
            "provider": getattr(model, "provider_name", "unknown"),
            "context_window": getattr(model, "context_window", None),
            "max_tokens": getattr(model, "max_tokens", None),
        },
    }


class AgentTraceProducer:
    """Project one Agent run's event stream into trace artifacts.

    The producer is a synchronous listener: trace writes are short buffered
    appends serialized by the writer's own lock, and listener settlement is
    part of the run's settlement, so a finalized run's artifacts are
    complete once the run is idle.
    """

    def __init__(self, writer: TraceWriter) -> None:
        if not isinstance(writer, TraceWriter):
            raise TypeError("writer must be a TraceWriter")
        self._writer = writer
        self._staged: List[TraceEvent] = []
        self._staged_turn = 0
        self._staged_model: Dict[str, Any] = {}
        self._event_index = 0
        self._total_tokens = 0
        self._finalized = False

    @property
    def finalized(self) -> bool:
        return self._finalized

    def attach(self, agent: Agent) -> Callable[[], None]:
        """Subscribe to one Agent; return the unsubscribe callback."""

        return agent.subscribe(self.handle)

    def handle(self, event: AgentEvent) -> None:
        if isinstance(event, AgentStart):
            self._write_lifecycle("started")
        elif isinstance(event, MessageEnd):
            self._on_message_end(event)
        elif isinstance(event, ToolExecutionStart):
            self._stage(
                PHASE_TOOL,
                {
                    "call_id": event.tool_call_id,
                    "name": event.tool_name,
                    "args": dict(event.args),
                    "status": "started",
                },
            )
        elif isinstance(event, ToolExecutionEnd):
            self._stage(
                PHASE_TOOL,
                {
                    "call_id": event.tool_call_id,
                    "name": event.tool_name,
                    "status": event.result.status,
                    "is_error": event.is_error,
                    "error": event.result.error,
                },
                ok=not event.is_error,
            )
        elif isinstance(event, TurnEnd):
            self._commit_turn(event)
        elif isinstance(event, AgentEnd):
            self._write_lifecycle("finished", messages=len(event.messages))

    def finalize(self, result: Optional[AgentLoopResult]) -> None:
        """Close the manifest once; staged uncommitted turns are dropped."""

        if self._finalized:
            return
        self._finalized = True
        self._staged = []
        if result is None:
            self._writer.finalize(
                "stopped",
                {
                    "stop_reason": "session_closed",
                    "token_usage": self._total_tokens,
                },
            )
            return
        status = _STATUS_BY_RUN_STATUS.get(result.status, "failed")
        summary: Dict[str, Any] = {
            "stop_reason": result.status.value,
            "final_result": _final_text(result),
            "token_usage": self._total_tokens,
        }
        if result.error:
            summary["failure_report"] = {"error": result.error}
        self._writer.finalize(status, summary)

    # ── internals ─────────────────────────────────────────────────────────

    def _write_lifecycle(self, status: str, **extra: Any) -> None:
        payload: Dict[str, Any] = {"status": status}
        payload.update(extra)
        event = TraceEvent(
            run_id=self._writer.run_id,
            step_id=_LIFECYCLE_STEP_ID,
            phase=PHASE_AGENT,
            payload=payload,
        )
        self._writer.write_event(event)
        self._event_index += 1

    def _on_message_end(self, event: MessageEnd) -> None:
        message = event.message
        if isinstance(message, UserMessage):
            self._stage(PHASE_INPUT, {"content": _user_text(message)})
        elif isinstance(message, ContextMessage):
            self._stage(PHASE_CONTEXT, {"content": message.content})
        elif isinstance(message, AssistantMessage):
            if message.usage is not None and message.usage.total_tokens:
                self._total_tokens += message.usage.total_tokens
            payload: Dict[str, Any] = {
                "text": message.text,
                "finish_reason": message.finish_reason,
                "error": message.error,
                "usage": (
                    message.usage.to_dict() if message.usage is not None else None
                ),
                "tool_calls": [
                    {"id": call.id, "name": call.name}
                    for call in message.tool_calls
                ],
                "model_name": message.model_name,
                "provider": message.provider,
            }
            self._staged_model = {
                key: value
                for key, value in (
                    ("model_name", message.model_name),
                    ("provider", message.provider),
                    ("finish_reason", message.finish_reason),
                )
                if value
            }
            self._stage(PHASE_MODEL, payload, ok=not message.failed)
        elif isinstance(message, ToolResultMessage):
            # ToolExecutionEnd already carries the terminal call evidence;
            # the transcript message would only duplicate it.
            return

    def _stage(self, phase: str, payload: Mapping[str, Any], *, ok: bool = True) -> None:
        self._staged.append(
            TraceEvent(
                run_id=self._writer.run_id,
                step_id=self._staged_turn,
                phase=phase,
                ok=ok,
                payload=dict(payload),
            )
        )

    def _commit_turn(self, event: TurnEnd) -> None:
        self._staged_turn = event.turn
        self._stage(
            PHASE_TURN,
            {
                "turn": event.turn,
                "tool_results": len(event.tool_results),
                "error": event.message.error,
            },
            ok=not event.message.failed,
        )
        step = TraceStep(
            step_id=event.turn,
            event_start_idx=self._event_index,
            event_end_idx=self._event_index + len(self._staged),
            model_response=dict(self._staged_model),
        )
        self._writer.write_transaction(self._staged, step)
        self._event_index += len(self._staged)
        self._staged = []
        self._staged_model = {}
        self._staged_turn = event.turn + 1


def _user_text(message: UserMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "\n".join(
        block.text for block in message.content if isinstance(block, TextContent)
    )


def _final_text(result: AgentLoopResult) -> Optional[str]:
    for message in reversed(result.messages):
        if isinstance(message, AssistantMessage) and not message.error:
            text = message.text.strip()
            if text:
                return text[:_FINAL_TEXT_MAX_CHARS]
    return None


__all__ = [
    "AgentTraceProducer",
    "PHASE_AGENT",
    "PHASE_CONTEXT",
    "PHASE_INPUT",
    "PHASE_MODEL",
    "PHASE_TOOL",
    "PHASE_TURN",
    "trace_producer_metadata",
]
