"""MessageBuilder protocol for customizing LLM message construction.

Agents can provide a ``message_builder`` attribute implementing this protocol
to take full control over how messages are assembled before being sent to the
LLM.  When no custom builder is provided, the engine falls back to its
default message construction logic (unchanged behavior).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


class ContextSnapshotConflictError(RuntimeError):
    """Raised when one immutable context revision is reused for other content."""


@dataclass(frozen=True, slots=True)
class ContextSnapshot:
    """One immutable, append-only model projection of current application state."""

    revision: str
    content: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        revision = str(self.revision or "").strip()
        content = str(self.content or "").strip()
        if not revision:
            raise ValueError("context snapshot revision must be non-empty")
        if not content:
            raise ValueError("context snapshot content must be non-empty")
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "content", content)
        object.__setattr__(
            self,
            "digest",
            hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )

    @classmethod
    def from_content(cls, content: str) -> "ContextSnapshot":
        """Use the content digest as a stable revision when no domain revision exists."""

        normalized = str(content or "").strip()
        revision = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return cls(revision=revision, content=normalized)


@dataclass
class MessageBuildRequest:
    """Everything the engine provides to a MessageBuilder."""

    step_id: int
    state: Any  # StateSchema instance
    observation: Any  # Observation instance
    prompt_bundle: Any  # PromptBuildResult instance
    prepared: str  # agent.prepare(state) return value
    history: List[Dict[str, Any]]  # retrieved history messages
    record: Any  # StepRecord


@dataclass
class MessageBuildResult:
    """What a MessageBuilder returns to the engine."""

    messages: List[Dict[str, Any]]
    # Optional entries to append to the engine history.
    # Each dict must have at least: {"role": str, "content": str, "step_id": int}
    # Optional keys: "metadata", "tool_calls", "tool_call_id", "name"
    history_entries: List[Dict[str, Any]] = field(default_factory=list)
    # When the revision or digest changes, the engine appends this projection
    # as a standalone user message and persists exactly what the model saw.
    context_snapshot: Optional[ContextSnapshot] = None


@runtime_checkable
class MessageBuilder(Protocol):
    """Protocol for agents that want full control over message construction."""

    def build_messages(self, request: MessageBuildRequest) -> MessageBuildResult:
        """Build the complete message list sent to the LLM.

        The returned ``messages`` are passed directly to the LLM without
        any further injection or wrapping by the engine.

        The returned ``history_entries`` are appended to the engine's
        history, replacing the engine's default history-append logic.

        ``context_snapshot`` is an immutable application-state projection. The
        engine appends changed snapshots instead of rewriting a prior user or
        tool message, preserving Provider-cacheable request prefixes.
        """
        ...


__all__ = [
    "ContextSnapshot",
    "ContextSnapshotConflictError",
    "MessageBuildRequest",
    "MessageBuildResult",
    "MessageBuilder",
]
