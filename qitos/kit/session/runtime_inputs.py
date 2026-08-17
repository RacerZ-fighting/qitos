"""Durable runtime-input posting and consumption tracking (D7).

A posted runtime input is durable truth (``runtime_input.posted``); its
delivery is complete only when the steered message enters the committed
transcript, marked by an idempotent ``runtime_input.consumed`` record. The
tracker observes the Agent event stream: the steered message's
``MessageEnd`` means the loop injected it at a turn safe point, and the
following ``TurnEnd`` means that turn's commit now covers it. A run that
ends without injecting the message consumes nothing, so pure recovery can
re-project the input exactly once; consumed inputs and inherited fork facts
are never re-projected.
"""

from __future__ import annotations

from ...core.agent import Agent
from ...core.agent_events import AgentEvent, MessageEnd, TurnEnd
from ...core.journal import JournalRecordType, SessionJournal
from ...core.message import UserMessage
from ...core.runtime_input import RuntimeInput
from ..journal.turn_recorder import encode_runtime_input_consumed


def runtime_input_text(event: RuntimeInput) -> str:
    """Project one runtime input to its steered user-message text."""

    return str(event.payload.get("content") or "").strip()


class SessionRuntimeInputs:
    """Default root runtime-input endpoint bound to one journal and Agent.

    The same instance re-projects recovered unconsumed inputs: re-steering
    writes no duplicate ``runtime_input.posted`` record, and the consumption
    append is idempotent by record id, so crash windows between injection
    and commit never deliver twice.
    """

    def __init__(self, journal: SessionJournal, agent: Agent) -> None:
        self._journal = journal
        self._agent = agent
        # Message identity is the delivery evidence: the loop injects the
        # exact object drained from the steering queue. Values keep a strong
        # reference so an id can never be reused by an unrelated message.
        self._tracked: dict[int, tuple[str, UserMessage]] = {}
        self._injected: list[str] = []
        self._consumed: set[str] = set()
        agent.subscribe(self._observe)

    async def post(self, event: RuntimeInput) -> bool:
        """Persist one input and steer it into the current or next run."""

        if not isinstance(event, RuntimeInput):
            raise TypeError("event must be a RuntimeInput")
        text = runtime_input_text(event)
        if not text:
            return False
        await self._journal.append(
            JournalRecordType.RUNTIME_INPUT_POSTED,
            event.to_dict(),
            record_id=f"{self._journal.run_id}:runtime:{event.event_id}",
        )
        self._steer(event.event_id, text)
        return True

    def project_recovered(self, event: RuntimeInput) -> None:
        """Re-steer one recovered unconsumed input without a new POSTED record."""

        text = runtime_input_text(event)
        if not text:
            return
        self._steer(event.event_id, text)

    def _steer(self, event_id: str, text: str) -> None:
        message = UserMessage(content=text)
        self._tracked[id(message)] = (event_id, message)
        self._agent.steer(message)

    async def _observe(self, event: AgentEvent) -> None:
        if isinstance(event, MessageEnd):
            tracked = self._tracked.pop(id(event.message), None)
            if tracked is not None and tracked[0] not in self._consumed:
                self._injected.append(tracked[0])
        elif isinstance(event, TurnEnd):
            # TurnEnd follows the turn's durable commit, so every input the
            # turn injected is now covered by a step.committed.
            while self._injected:
                event_id = self._injected.pop(0)
                if event_id in self._consumed:
                    continue
                await self._journal.append(
                    JournalRecordType.RUNTIME_INPUT_CONSUMED,
                    encode_runtime_input_consumed(event_id),
                    record_id=(
                        f"{self._journal.run_id}:runtime:{event_id}:consumed"
                    ),
                )
                self._consumed.add(event_id)


__all__ = ["SessionRuntimeInputs", "runtime_input_text"]
