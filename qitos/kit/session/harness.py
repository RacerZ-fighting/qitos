"""Authoritative Session Harness: start, resume, fork, compact, close.

One Run journal is one canonical append-only log and one Agent run; a
Session is the lineage of Run journals linked by fork. The harness owns the
lifecycle the façade cannot own itself: it restores a recovered journal
into a fresh façade (transcript, thinking level, configuration lineage),
verifies the provided model identity and Tool registry coverage with typed
rejections, closes crash-torn Tool calls without re-execution, re-projects
unconsumed runtime inputs exactly once, compacts at idle boundaries, and
reattaches trace production. Corruption and implementation faults raise;
expected rejections are values.

Continuation after any terminal run is an explicit fork that embeds the
committed prefix, so historical records are never rewritten. Inside one
:class:`SessionRun`, prompting again after the current journal settled
advances along that same mechanism automatically; ``SessionRun.run_id``
always names the current journal.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

from ...core.agent import (
    Agent,
    AgentRunRejected,
    AgentRunResult,
    normalize_prompt_messages,
)
from ...core.agent_loop import (
    AgentLoopResult,
    AgentRunStatus,
    model_protocol_identity,
)
from ...core.budget import BudgetLedger
from ...core.journal import (
    JournalError,
    JournalOwnershipError,
    JournalPosition,
    JournalRecord,
    JournalRecordType,
    SessionJournal,
    resolve_inherited_record,
)
from ...core.message import AssistantMessage, Message, UserMessage
from ...core.model_response import ModelPricing, ModelUsage
from ...core.plan import Plan
from ...core.runtime_input import RuntimeInput
from ...core.task import (
    Task,
    TaskBlocker,
    TaskLifecycle,
    TaskStatus,
    validate_task_transition,
)
from ...core.thinking import ThinkingLevel
from ...core.tool_registry import ToolRegistry
from ...trace.producer import AgentTraceProducer, trace_producer_metadata
from ...trace.writer import TraceWriter
from ..journal import (
    InMemoryJournalStore,
    InMemorySessionJournal,
    JsonlSessionJournal,
    JournalTurnTransaction,
    RecoveredSession,
    RecoveredTask,
    close_crashed_tool_calls,
    recover_session,
)
from ..journal._paths import JOURNAL_FILENAME, validate_run_id
from ..journal.turn_recorder import (
    decode_task_transition,
    encode_task_created,
    encode_task_transition,
)
from .compaction import (
    CompactRejected,
    CompactResult,
    CompactionSettings,
    ContextEntry,
    SummarizationError,
    compact_context,
    estimate_context_tokens,
    estimate_tokens,
    is_context_overflow,
    prepare_compaction,
    should_compact,
    usage_context_tokens,
)
from .runtime_inputs import SessionRuntimeInputs

if TYPE_CHECKING:
    from ...models.base import Model

_logger = logging.getLogger(__name__)

_ResumeReason = Literal[
    "not_found", "terminal", "model_mismatch", "tools_missing", "busy"
]
_RESUME_REASONS = frozenset(
    {"not_found", "terminal", "model_mismatch", "tools_missing", "busy"}
)


@dataclass(frozen=True, slots=True)
class ResumeRejected:
    """Typed expected rejection for resume/fork entry operations."""

    reason: _ResumeReason
    missing_tools: tuple[str, ...] = ()
    detail: str = ""

    def __post_init__(self) -> None:
        if self.reason not in _RESUME_REASONS:
            raise ValueError(f"unknown resume rejection reason: {self.reason!r}")
        if not isinstance(self.missing_tools, tuple) or not all(
            isinstance(name, str) and name for name in self.missing_tools
        ):
            raise TypeError("missing_tools must be a tuple of non-empty strings")
        if not isinstance(self.detail, str):
            raise TypeError("detail must be a string")


_TaskTransitionReason = Literal["unknown", "terminal", "invalid"]
_TASK_TRANSITION_REASONS = frozenset({"unknown", "terminal", "invalid"})


@dataclass(frozen=True, slots=True)
class TaskTransitionRejected:
    """Typed expected rejection for Task lifecycle operations.

    ``unknown``: the session lineage holds no Root Task. ``terminal``: the
    Task already committed a terminal status. ``invalid``: the requested
    move is not a legal transition from the folded state. Corruption and
    implementation faults raise instead of returning this value.
    """

    reason: _TaskTransitionReason
    detail: str = ""

    def __post_init__(self) -> None:
        if self.reason not in _TASK_TRANSITION_REASONS:
            raise ValueError(
                f"unknown task transition rejection reason: {self.reason!r}"
            )
        if not isinstance(self.detail, str):
            raise TypeError("detail must be a string")


class SessionHarness:
    """Entry point owning Session journals, recovery verification and policy.

    ``journal_home`` is either a directory of JSONL Run journals or one
    shared :class:`InMemoryJournalStore`. ``compaction`` enables the
    automatic paths (idle-boundary threshold and one-shot overflow
    recovery); manual compaction stays available with the defaults. With a
    ``trace_directory`` every journal run also reattaches a trace producer
    writing the three-file layout ``qita`` discovers.
    """

    def __init__(
        self,
        journal_home: Union[str, Path, InMemoryJournalStore],
        *,
        compaction: Optional[CompactionSettings] = None,
        trace_directory: Union[str, Path, None] = None,
        run_id_factory: Optional[Callable[[], str]] = None,
    ) -> None:
        self._store: InMemoryJournalStore | None = None
        self._root: Path | None = None
        if isinstance(journal_home, InMemoryJournalStore):
            self._store = journal_home
        elif isinstance(journal_home, (str, Path)):
            self._root = Path(journal_home).expanduser().resolve()
        else:
            raise TypeError(
                "journal_home must be a directory or an InMemoryJournalStore"
            )
        if compaction is not None and not isinstance(compaction, CompactionSettings):
            raise TypeError("compaction must be a CompactionSettings or None")
        self._compaction = compaction
        self._trace_directory = (
            Path(trace_directory).expanduser().resolve()
            if trace_directory is not None
            else None
        )
        if run_id_factory is not None and not callable(run_id_factory):
            raise TypeError("run_id_factory must be callable or None")
        self._run_id_factory = run_id_factory or (lambda: uuid.uuid4().hex)

    @property
    def compaction_settings(self) -> Optional[CompactionSettings]:
        """The configured auto-compaction policy, when present."""

        return self._compaction

    @property
    def trace_directory(self) -> Optional[Path]:
        """The trace output directory, when trace reattachment is enabled."""

        return self._trace_directory

    # ── journal plumbing ──────────────────────────────────────────────────

    def _new_journal(self) -> SessionJournal:
        if self._store is not None:
            return InMemorySessionJournal(self._store)
        assert self._root is not None
        return JsonlSessionJournal(self._root)

    def _journal_exists(self, run_id: str) -> bool:
        validate_run_id(run_id)
        if self._store is not None:
            return run_id in self._store.records
        assert self._root is not None
        return (self._root / run_id / JOURNAL_FILENAME).is_file()

    # ── entry points ──────────────────────────────────────────────────────

    async def start(
        self,
        *,
        model: "Model",
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: str = "",
        thinking_level: Optional[ThinkingLevel] = None,
        budget_ledger: Optional[BudgetLedger] = None,
        model_pricing: Optional[ModelPricing] = None,
        post_runtime_event: Optional[Callable[[RuntimeInput], Any]] = None,
        run_metadata: Optional[Mapping[str, Any]] = None,
        task: Optional[Task] = None,
        **agent_kwargs: Any,
    ) -> "SessionRun":
        """Create a fresh journal and an Agent ready for its first run.

        When ``task`` is given it must be a Root Task (no parent): it
        commits as one ``task.created`` record before ``input.accepted``
        and any model or Tool side effect. A fresh lineage cannot already
        hold an unfinished Root Task — the Run id is minted here and journal
        creation fails on collision — so start has no task rejection path.
        """

        if task is not None:
            if not isinstance(task, Task):
                raise TypeError("task must be a Task or None")
            if task.parent_task_id is not None:
                raise ValueError("SessionHarness.start commits a Root Task")
        run_id = self._run_id_factory()
        journal = self._new_journal()
        metadata = {
            "lineage_id": run_id,
            "agent": "qitos.kit.session",
            **dict(run_metadata or {}),
        }
        session_run = self._session_run(
            model=model,
            tool_registry=tool_registry,
            system_prompt=system_prompt,
            thinking_level=thinking_level,
            budget_ledger=budget_ledger,
            model_pricing=model_pricing,
            post_runtime_event=post_runtime_event,
            agent_kwargs=agent_kwargs,
        )
        transferred = False
        try:
            await journal.create(run_id, metadata)
            if task is None:
                session_run._install(journal, (), None)
            else:
                await journal.append(
                    JournalRecordType.TASK_CREATED,
                    encode_task_created(task),
                    record_id=f"{run_id}:task:{task.task_id}:created",
                )
                records = await journal.replay()
                session_run._install(journal, records, recover_session(records))
            transferred = True
            return session_run
        finally:
            if not transferred:
                await journal.close()

    async def resume(
        self,
        run_id: str,
        *,
        model: "Model",
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: str = "",
        budget_ledger: Optional[BudgetLedger] = None,
        model_pricing: Optional[ModelPricing] = None,
        post_runtime_event: Optional[Callable[[RuntimeInput], Any]] = None,
        **agent_kwargs: Any,
    ) -> "SessionRun | ResumeRejected":
        """Continue one unfinished journal in place with a restored façade.

        The recovered transcript, thinking level and configuration lineage
        seed a fresh Agent; a crash window is closed with explicit cancelled
        terminals, and each unconsumed runtime input is re-projected exactly
        once. Terminal runs reject (fork is the explicit continuation), as do
        model identity and Tool registry mismatches against the lineage.
        """

        if not self._journal_exists(run_id):
            return ResumeRejected(
                "not_found", detail=f"no journal exists for run {run_id!r}"
            )
        journal = self._new_journal()
        try:
            await journal.open(run_id)
        except JournalOwnershipError:
            return ResumeRejected(
                "busy", detail=f"run {run_id!r} has an active journal writer"
            )
        transferred = False
        try:
            records = await journal.replay()
            recovered = recover_session(records)
            if recovered.outcome is not None:
                return ResumeRejected(
                    "terminal",
                    detail=(
                        f"run {run_id!r} finished with status "
                        f"{recovered.outcome.status.value}; fork it explicitly"
                    ),
                )
            rejection = _verify_lineage(recovered, model, tool_registry)
            if rejection is not None:
                return rejection
            if recovered.unterminated_calls or recovered.unstarted_calls:
                await close_crashed_tool_calls(journal, recovered)
                records = await journal.replay()
                recovered = recover_session(records)
            session_run = self._session_run(
                model=model,
                tool_registry=tool_registry,
                system_prompt=system_prompt,
                thinking_level=None,
                budget_ledger=budget_ledger,
                model_pricing=model_pricing,
                post_runtime_event=post_runtime_event,
                agent_kwargs=agent_kwargs,
            )
            session_run._install(journal, records, recovered)
            session_run._reproject_inputs(recovered.unconsumed_inputs)
            transferred = True
            return session_run
        finally:
            if not transferred:
                await journal.close()

    async def fork(
        self,
        run_id: str,
        position: Optional[JournalPosition] = None,
        *,
        model: "Model",
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: str = "",
        budget_ledger: Optional[BudgetLedger] = None,
        model_pricing: Optional[ModelPricing] = None,
        post_runtime_event: Optional[Callable[[RuntimeInput], Any]] = None,
        **agent_kwargs: Any,
    ) -> "SessionRun | ResumeRejected":
        """Branch one journal at a committed boundary into a new Run.

        The default position is the journal's latest committed boundary; the
        new journal embeds the inherited prefix and is self-contained
        recovery truth. Forking a terminal run is the explicit way to
        continue finished work, so the terminal check does not apply; model
        identity and Tool coverage are verified exactly like a resume.
        """

        if not self._journal_exists(run_id):
            return ResumeRejected(
                "not_found", detail=f"no journal exists for run {run_id!r}"
            )
        parent = self._new_journal()
        try:
            await parent.open(run_id)
        except JournalOwnershipError:
            return ResumeRejected(
                "busy", detail=f"run {run_id!r} has an active journal writer"
            )
        child: SessionJournal | None = None
        transferred = False
        try:
            records = await parent.replay()
            recovered = recover_session(records)
            rejection = _verify_lineage(recovered, model, tool_registry)
            if rejection is not None:
                return rejection
            if position is None:
                position = _latest_committed_boundary(records)
                if position is None:
                    raise JournalError(
                        f"run {run_id!r} has no committed boundary to fork from"
                    )
            child = await parent.fork(position, self._run_id_factory())
            await parent.close()
            child_records = await child.replay()
            child_recovered = recover_session(child_records)
            session_run = self._session_run(
                model=model,
                tool_registry=tool_registry,
                system_prompt=system_prompt,
                thinking_level=None,
                budget_ledger=budget_ledger,
                model_pricing=model_pricing,
                post_runtime_event=post_runtime_event,
                agent_kwargs=agent_kwargs,
            )
            session_run._install(child, child_records, child_recovered)
            transferred = True
            return session_run
        finally:
            if not transferred:
                if child is not None:
                    await child.close()
                await parent.close()

    # ── shared construction ───────────────────────────────────────────────

    def _session_run(
        self,
        *,
        model: "Model",
        tool_registry: Optional[ToolRegistry],
        system_prompt: str,
        thinking_level: Optional[ThinkingLevel],
        budget_ledger: Optional[BudgetLedger],
        model_pricing: Optional[ModelPricing],
        post_runtime_event: Optional[Callable[[RuntimeInput], Any]],
        agent_kwargs: Mapping[str, Any],
    ) -> "SessionRun":
        return SessionRun(
            self,
            model=model,
            tool_registry=tool_registry,
            system_prompt=system_prompt,
            thinking_level=thinking_level,
            budget_ledger=budget_ledger,
            model_pricing=model_pricing,
            post_runtime_event=post_runtime_event,
            agent_kwargs=dict(agent_kwargs),
        )


def _seed_task_state(recovered: RecoveredSession | None) -> RecoveredTask | None:
    """Seed the lineage's latest Root Task (terminal or unfinished), if any."""

    if recovered is None:
        return None
    latest: RecoveredTask | None = None
    for task in recovered.tasks.values():
        if task.definition.parent_task_id is None:
            latest = task
    return latest


_STATE_CARRY_TYPES = frozenset(
    {
        JournalRecordType.TASK_CREATED,
        JournalRecordType.TASK_TRANSITION,
        JournalRecordType.PLAN_UPDATED,
    }
)


def _verify_lineage(
    recovered: RecoveredSession,
    model: "Model",
    tool_registry: Optional[ToolRegistry],
) -> ResumeRejected | None:
    """Verify the provided runtime against the journaled lineage (D5/D6)."""

    identity = (
        model.provider_name,
        model.model,
        model_protocol_identity(model),
    )
    if recovered.model_identity is not None and recovered.model_identity != identity:
        return ResumeRejected(
            "model_mismatch",
            detail=(
                f"journal lineage model {recovered.model_identity!r} does "
                f"not match the provided model {identity!r}"
            ),
        )
    registry = tool_registry if tool_registry is not None else ToolRegistry()
    missing = tuple(
        name
        for name in recovered.active_tool_names or ()
        if registry.get(name) is None
    )
    if missing:
        return ResumeRejected(
            "tools_missing",
            missing_tools=missing,
            detail=(
                "journal lineage activated Tools absent from the provided "
                f"registry: {', '.join(missing)}"
            ),
        )
    return None


def _latest_committed_boundary(
    records: Sequence[JournalRecord],
) -> JournalPosition | None:
    for record in reversed(records):
        if resolve_inherited_record(record).type is JournalRecordType.STEP_COMMITTED:
            return record.position
    return None


def _latest_compaction_timestamp(records: Sequence[JournalRecord]) -> float | None:
    latest: float | None = None
    for record in records:
        effective = resolve_inherited_record(record)
        if effective.type is not JournalRecordType.COMPACTION:
            continue
        try:
            latest = datetime.fromisoformat(effective.timestamp).timestamp()
        except ValueError:
            continue
    return latest


def _context_entries(recovered: RecoveredSession) -> list[ContextEntry]:
    """Align the recovered context with its transcript record ids.

    Recovery projects the context from the same message objects it places
    in the canonical transcript (summaries are fresh objects), so identity
    matching in canonical order is exact.
    """

    transcript = recovered.transcript
    record_ids = recovered.transcript_record_ids
    context = recovered.context_messages
    if len(context) == len(transcript) and all(
        context_message is transcript_message
        for context_message, transcript_message in zip(context, transcript)
    ):
        return list(zip(record_ids, context))
    entries: list[ContextEntry] = []
    tail = 0
    for message in context:
        if tail < len(transcript) and message is transcript[tail]:
            entries.append((record_ids[tail], message))
            tail += 1
        else:
            entries.append((None, message))
    return entries


class SessionRun:
    """One live Session leg: current journal, Agent, and run policy.

    ``prompt``/``continue_run`` delegate to the current Agent; once the
    current journal settles, the next call advances along an explicit fork
    (the terminal continuation boundary), so one ``SessionRun`` carries a
    whole interactive session while every journal stays append-only. The
    one-shot overflow recovery guard is scoped to this object.
    """

    def __init__(
        self,
        harness: SessionHarness,
        *,
        model: "Model",
        tool_registry: Optional[ToolRegistry],
        system_prompt: str,
        thinking_level: Optional[ThinkingLevel],
        budget_ledger: Optional[BudgetLedger],
        model_pricing: Optional[ModelPricing],
        post_runtime_event: Optional[Callable[[RuntimeInput], Any]],
        agent_kwargs: dict[str, Any],
    ) -> None:
        self._harness = harness
        self._model = model
        self._tool_registry = tool_registry
        self._system_prompt = system_prompt
        self._thinking_level = thinking_level
        self._ledger_source = budget_ledger
        self._model_pricing = model_pricing
        self._post_runtime_event = post_runtime_event
        self._agent_kwargs = agent_kwargs

        self._journal: SessionJournal
        self._recorder: JournalTurnTransaction
        self._agent: Agent
        self._recovered: RecoveredSession | None = None
        self._context_entries: list[ContextEntry] = []
        self._latest_compaction_ts: float | None = None
        self._budget_ledger: BudgetLedger | None = None
        self._runtime_inputs: SessionRuntimeInputs | None = None
        self._trace_producer: AgentTraceProducer | None = None
        self._task_state: RecoveredTask | None = None
        self._plan: Plan | None = None
        self._run_started = False
        self._leg_prepared = False
        self._leg_has_input = False
        self._overflow_attempted = False
        self._closed = False
        self._last_result: AgentLoopResult | None = None
        self._leg_lock = asyncio.Lock()
        self._task_lock = asyncio.Lock()

    # ── views ─────────────────────────────────────────────────────────────

    @property
    def run_id(self) -> str:
        return self._journal.run_id

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def journal(self) -> SessionJournal:
        return self._journal

    @property
    def budget_ledger(self) -> BudgetLedger | None:
        """The ledger attached to the current journal, when configured."""

        return self._budget_ledger

    @property
    def task(self) -> Task | None:
        """The lineage's current Root Task definition, when task-bearing."""

        state = self._task_state
        return state.definition if state is not None else None

    @property
    def task_lifecycle(self) -> TaskLifecycle | None:
        """The folded lifecycle of the current Root Task, when task-bearing."""

        state = self._task_state
        return state.lifecycle if state is not None else None

    @property
    def plan(self) -> Plan | None:
        """The latest committed Plan in this Session lineage, when present."""

        return self._plan

    # ── run control ───────────────────────────────────────────────────────

    async def prompt(
        self, message: Union[str, Message, Sequence[Message]]
    ) -> AgentRunResult:
        """Run the next leg from text, one message or a message batch.

        One Run journal accepts one initial input. On a journal that
        already committed its ``input.accepted`` (a resumed run), the new
        prompt enters as steering before the next turn instead of writing
        a second input record. In a task-bearing session whose Root Task is
        terminal, prompting for new work rejects with
        ``AgentRunRejected("task_terminal")``: continue explicitly with
        :meth:`start_follow_up`. A blocked Task does not block prompting;
        only :meth:`unblock_task` returns it to active.
        """

        self._require_open()
        async with self._leg_lock:
            state = self._task_state
            if state is not None and state.lifecycle.status.terminal:
                return AgentRunRejected("task_terminal")
            await self._prepare_next_leg()
            return await self._prompt_locked(message)

    async def _prompt_locked(
        self, message: Union[str, Message, Sequence[Message]]
    ) -> AgentRunResult:
        if self._leg_has_input:
            for prompt_message in normalize_prompt_messages(message):
                self._agent.steer(prompt_message)
            try:
                result = await self._agent.continue_run()
            except BaseException:
                self._run_started = True
                raise
            return await self._settle_leg(result)
        try:
            result = await self._agent.prompt(message)
        except BaseException:
            self._run_started = True
            raise
        return await self._settle_leg(result)

    async def continue_run(self) -> AgentRunResult:
        """Continue from the current transcript tail (user or Tool result)."""

        self._require_open()
        async with self._leg_lock:
            await self._prepare_next_leg()
            try:
                result = await self._agent.continue_run()
            except BaseException:
                self._run_started = True
                raise
            return await self._settle_leg(result)

    def abort(self) -> None:
        """Cooperatively abort the active run, if any."""

        self._agent.abort()

    async def wait_for_idle(self) -> None:
        """Resolve when the active run and its listeners have settled."""

        await self._agent.wait_for_idle()

    async def post_runtime_event(self, event: RuntimeInput) -> bool:
        """Default root endpoint: durable post plus steering (D7)."""

        if self._closed or self._run_started:
            return False
        tracker = self._runtime_inputs
        if tracker is None:
            return False
        return await tracker.post(event)

    # ── task lifecycle ────────────────────────────────────────────────────

    async def complete_task(self, reason: str) -> TaskLifecycle | TaskTransitionRejected:
        """Commit the current Root Task's ``completed`` terminal transition."""

        return await self._transition_task(TaskStatus.COMPLETED, reason=reason)

    async def fail_task(self, reason: str) -> TaskLifecycle | TaskTransitionRejected:
        """Commit the current Root Task's ``failed`` terminal transition."""

        return await self._transition_task(TaskStatus.FAILED, reason=reason)

    async def cancel_task(self, reason: str) -> TaskLifecycle | TaskTransitionRejected:
        """Commit the current Root Task's ``cancelled`` terminal transition."""

        return await self._transition_task(TaskStatus.CANCELLED, reason=reason)

    async def block_task(
        self, blocker: TaskBlocker
    ) -> TaskLifecycle | TaskTransitionRejected:
        """Commit the current Root Task's active → blocked transition."""

        if not isinstance(blocker, TaskBlocker):
            raise TypeError("blocker must be a TaskBlocker")
        return await self._transition_task(TaskStatus.BLOCKED, blocker=blocker)

    async def unblock_task(self) -> TaskLifecycle | TaskTransitionRejected:
        """Commit the only blocked → active path: explicit caller input or
        an observed external-state change, delivered by the application."""

        return await self._transition_task(TaskStatus.ACTIVE)

    async def start_follow_up(
        self,
        task: Task,
        prompt: Union[str, Message, Sequence[Message]],
    ) -> AgentRunResult | TaskTransitionRejected:
        """Start a new Root Task on a terminal-task lineage, then prompt.

        The leg advances along the existing explicit-fork machinery and the
        new ``task.created`` commits before the prompt's ``input.accepted``.
        When the current leg never accepted input there is nothing to fork,
        so the new Task commits in place instead. A taskless session or an
        unfinished current Task returns a typed rejection; corruption
        raises.
        """

        self._require_open()
        if not isinstance(task, Task):
            raise TypeError("task must be a Task")
        if task.parent_task_id is not None:
            raise ValueError("a terminal follow-up starts a new Root Task")
        async with self._leg_lock:
            state = self._task_state
            if state is None:
                return TaskTransitionRejected(
                    "unknown",
                    detail=(
                        "session lineage holds no Root Task; "
                        "prompt() runs taskless work"
                    ),
                )
            if not state.lifecycle.status.terminal:
                return TaskTransitionRejected(
                    "invalid",
                    detail=(
                        "the current Root Task is not terminal; complete, "
                        "fail or cancel it first, or use prompt()"
                    ),
                )
            if self._run_started or self._leg_has_input:
                await self._advance(task)
            else:
                await self._commit_task_created_in_place(task)
            if not self._leg_prepared:
                self._leg_prepared = True
                await self._auto_compact_threshold()
            return await self._prompt_locked(prompt)

    async def _transition_task(
        self,
        to_status: TaskStatus,
        *,
        reason: str | None = None,
        blocker: TaskBlocker | None = None,
    ) -> TaskLifecycle | TaskTransitionRejected:
        self._require_open()
        async with self._task_lock:
            state = self._task_state
            if state is None:
                return TaskTransitionRejected(
                    "unknown", detail="session lineage holds no Root Task"
                )
            current = state.lifecycle
            if current.status.terminal:
                return TaskTransitionRejected(
                    "terminal",
                    detail=(
                        f"task {state.definition.task_id!r} is already "
                        f"{current.status.value}"
                    ),
                )
            try:
                validate_task_transition(current.status, to_status)
            except ValueError as exc:
                return TaskTransitionRejected("invalid", detail=str(exc))
            if self._run_started:
                # A settled leg keeps its run terminal as its last record,
                # so the transition commits into the next leg, ahead of
                # that leg's input and model side effects.
                async with self._leg_lock:
                    if self._run_started:
                        await self._advance()
                state = self._task_state
                if state is None or state.lifecycle.status is not current.status:
                    raise JournalError("task state diverged across the leg advance")
                current = state.lifecycle
            usage: ModelUsage | None = None
            ledger = self._budget_ledger
            if ledger is not None:
                snapshot = ledger.snapshot()
                usage = ModelUsage.from_mapping(
                    {
                        "total_tokens": snapshot.total_tokens,
                        "cost_usd": float(snapshot.total_cost_usd),
                    }
                )
            lifecycle = TaskLifecycle(
                status=to_status,
                usage=usage,
                blocker=blocker,
                terminal_reason=reason,
            )
            task_id = state.definition.task_id
            payload = encode_task_transition(
                task_id=task_id,
                from_status=current.status,
                to_status=to_status,
                reason=reason,
                blocker=blocker,
                usage=usage,
            )
            run_id = self._journal.run_id
            sequence = await self._own_transition_count(run_id, task_id)
            await self._journal.append(
                JournalRecordType.TASK_TRANSITION,
                payload,
                record_id=f"{run_id}:task:{task_id}:transition:{sequence}",
            )
            records = await self._journal.replay()
            self._task_state = _seed_task_state(recover_session(records))
            return lifecycle

    async def _own_transition_count(self, run_id: str, task_id: str) -> int:
        records = await self._journal.replay()
        return sum(
            1
            for record in records
            if record.type is JournalRecordType.TASK_TRANSITION
            and record.run_id == run_id
            and record.payload.get("task_id") == task_id
        )

    async def _commit_task_created_in_place(self, task: Task) -> None:
        """Commit a new Root Task into a leg that never accepted input."""

        journal = self._journal
        await journal.append(
            JournalRecordType.TASK_CREATED,
            encode_task_created(task),
            record_id=f"{journal.run_id}:task:{task.task_id}:created",
        )
        producer = self._trace_producer
        if producer is not None and not producer.finalized:
            producer.finalize(self._last_result)
        records = await journal.replay()
        self._install(journal, records, recover_session(records))

    async def compact(self) -> CompactResult | CompactRejected:
        """Manually compact at idle; terminal legs advance first."""

        self._require_open()
        if self._leg_lock.locked() or self._agent.is_streaming:
            return CompactRejected("busy")
        async with self._leg_lock:
            if self._run_started:
                await self._advance()
            settings = self._harness.compaction_settings or CompactionSettings()
            result = await self._apply_compaction(settings)
            if result is None:
                return CompactRejected("nothing_to_compact")
            self._leg_prepared = True
            return result

    async def close(self) -> None:
        """Idle wait, trace finalize, journal close; idempotent."""

        if self._closed:
            return
        self._closed = True
        self._agent.abort()
        await self._agent.wait_for_idle()
        producer = self._trace_producer
        if producer is not None and not producer.finalized:
            producer.finalize(self._last_result)
        await self._journal.close()

    # ── leg machinery ─────────────────────────────────────────────────────

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("session run is closed")

    def _install(
        self,
        journal: SessionJournal,
        records: Sequence[JournalRecord],
        recovered: RecoveredSession | None,
    ) -> None:
        """Bind one journal and its restored façade to this Session run."""

        self._budget_ledger = self._attach_ledger(journal, records)
        self._recorder = JournalTurnTransaction(
            journal,
            recovered=recovered.recorder_state if recovered is not None else None,
            budget_ledger=self._budget_ledger,
            model_pricing=self._model_pricing,
        )
        self._journal = journal
        self._recovered = recovered
        self._task_state = _seed_task_state(recovered)
        self._plan = recovered.plan if recovered is not None else None
        self._context_entries = (
            _context_entries(recovered) if recovered is not None else []
        )
        self._latest_compaction_ts = _latest_compaction_timestamp(records)
        self._leg_has_input = any(
            record.type is JournalRecordType.INPUT_ACCEPTED for record in records
        )
        thinking_level = (
            recovered.thinking_level
            if recovered is not None
            else self._thinking_level
        )
        passthrough = {
            key: value
            for key, value in self._agent_kwargs.items()
            if key != "runtime_context"
        }
        runtime_context = dict(self._agent_kwargs.get("runtime_context") or {})
        runtime_context.setdefault("journal", journal)
        if self._budget_ledger is not None:
            # The Session ledger is authoritative for Tool-launched descendants;
            # a caller-provided runtime context cannot substitute another owner.
            runtime_context["budget_ledger"] = self._budget_ledger
        if self._task_state is not None:
            # Publish the current Root Task identity for Tools that bind
            # their work to it (the Agent Tool reads these keys for its
            # Subagent launch request).
            runtime_context["task_id"] = self._task_state.definition.task_id
            if self._task_state.definition.plan_assignment is not None:
                runtime_context["plan_assignment"] = (
                    self._task_state.definition.plan_assignment
                )
        runtime_context["post_runtime_event"] = (
            self._post_runtime_event or self.post_runtime_event
        )
        self._agent = Agent(
            model=self._model,
            tool_registry=self._tool_registry,
            system_prompt=self._system_prompt,
            thinking_level=thinking_level,
            initial_messages=(
                recovered.context_messages if recovered is not None else ()
            ),
            turn_base=recovered.next_turn if recovered is not None else 0,
            transaction_factory=lambda _run_id: self._recorder,
            run_id_factory=lambda: journal.run_id,
            runtime_context=runtime_context,
            **passthrough,
        )
        self._runtime_inputs = SessionRuntimeInputs(journal, self._agent)
        self._trace_producer = self._attach_trace(journal)
        self._run_started = False
        self._leg_prepared = False

    def _attach_trace(self, journal: SessionJournal) -> AgentTraceProducer | None:
        directory = self._harness.trace_directory
        if directory is None:
            return None
        writer = TraceWriter(
            str(directory),
            journal.run_id,
            metadata=trace_producer_metadata(self._model),
        )
        producer = AgentTraceProducer(writer)
        producer.attach(self._agent)
        return producer

    def _attach_ledger(
        self,
        journal: SessionJournal,
        records: Sequence[JournalRecord],
    ) -> BudgetLedger | None:
        source = self._ledger_source
        if source is None:
            return None
        try:
            source.attach(journal, root_run_id=journal.run_id, records=records)
            return source
        except RuntimeError:
            # The lineage moved to a new journal: the ledger follows it, and
            # the replay of the fork's inherited budget commits rebuilds the
            # same totals under identical limits.
            snapshot = source.snapshot()
            fresh = BudgetLedger(
                max_steps=snapshot.max_steps,
                max_tokens=snapshot.max_tokens,
                max_cost_usd=snapshot.max_cost_usd,
            )
            fresh.attach(journal, root_run_id=journal.run_id, records=records)
            self._ledger_source = fresh
            return fresh

    def _reproject_inputs(self, inputs: Sequence[RuntimeInput]) -> None:
        tracker = self._runtime_inputs
        if tracker is None:
            return
        for event in inputs:
            tracker.project_recovered(event)

    async def _prepare_next_leg(self) -> None:
        if self._run_started:
            await self._advance()
        if not self._leg_prepared:
            self._leg_prepared = True
            await self._auto_compact_threshold()

    async def _settle_leg(self, result: AgentRunResult) -> AgentRunResult:
        if isinstance(result, AgentRunRejected):
            return result
        self._run_started = True
        self._last_result = result
        await self._refresh_from_journal()
        producer = self._trace_producer
        if producer is not None:
            producer.finalize(result)
        if (
            result.status is AgentRunStatus.FAILED
            and self._harness.compaction_settings is not None
            and self._harness.compaction_settings.enabled
            and self._overflow_detected()
        ):
            retry = await self._overflow_recovery()
            if retry is not None:
                return retry
        return result

    async def _refresh_from_journal(self) -> None:
        records = await self._journal.replay()
        recovered = recover_session(records)
        self._recovered = recovered
        self._plan = recovered.plan
        self._context_entries = _context_entries(recovered)
        self._latest_compaction_ts = _latest_compaction_timestamp(records)
        self._leg_has_input = any(
            record.type is JournalRecordType.INPUT_ACCEPTED for record in records
        )

    async def _advance(self, task: Task | None = None) -> None:
        """Fork the settled journal at its latest committed boundary.

        Task facts committed after that boundary (between the last turn
        commit and the run terminal, or in a leg whose run never reached an
        own commit) would be truncated by the fork, so they are carried
        into the new leg ahead of its input; ``task`` is an optional new
        Root Task committed right after them, before any of the new leg's
        side effects.
        """

        journal = self._journal
        records = await journal.replay()
        recover_session(records)  # fault early on corruption, before forking
        position = _latest_committed_boundary(records)
        if position is None:
            raise JournalError(
                "cannot continue a run without a committed boundary"
            )
        carried = [
            record
            for record in records[position.seq :]
            if record.type in _STATE_CARRY_TYPES
        ]
        child = await journal.fork(position, self._harness._run_id_factory())
        await journal.close()
        transition_counts: dict[str, int] = {}
        plan_sequence = 0
        for record in carried:
            if record.type is JournalRecordType.TASK_CREATED:
                carried_id = str(record.payload.get("task_id") or "")
                await child.append(
                    JournalRecordType.TASK_CREATED,
                    record.payload,
                    record_id=f"{child.run_id}:task:{carried_id}:created",
                )
                continue
            if record.type is JournalRecordType.TASK_TRANSITION:
                task_id = decode_task_transition(record.payload)[0]
                sequence = transition_counts.get(task_id, 0)
                transition_counts[task_id] = sequence + 1
                await child.append(
                    JournalRecordType.TASK_TRANSITION,
                    record.payload,
                    record_id=(
                        f"{child.run_id}:task:{task_id}:transition:{sequence}"
                    ),
                )
                continue
            await child.append(
                JournalRecordType.PLAN_UPDATED,
                record.payload,
                record_id=f"{child.run_id}:plan:carried:{plan_sequence}",
            )
            plan_sequence += 1
        if task is not None:
            await child.append(
                JournalRecordType.TASK_CREATED,
                encode_task_created(task),
                record_id=f"{child.run_id}:task:{task.task_id}:created",
            )
        child_records = await child.replay()
        child_recovered = recover_session(child_records)
        self._install(child, child_records, child_recovered)

    # ── compaction ────────────────────────────────────────────────────────

    async def _apply_compaction(
        self, settings: CompactionSettings
    ) -> CompactResult | None:
        """Summarize, persist the durable record and swap the context."""

        entries = self._context_entries
        preparation = prepare_compaction(entries, settings)
        if preparation is None:
            return None
        result = await compact_context(self._model, preparation, settings)
        await self._recorder.compaction(
            summary=result.summary,
            first_kept_transcript_id=result.first_kept_transcript_id,
            tokens_before=result.tokens_before,
            usage=result.usage,
        )
        kept_index = next(
            index
            for index, (record_id, _message) in enumerate(entries)
            if record_id == result.first_kept_transcript_id
        )
        summary_message = UserMessage(content=result.summary)
        self._agent.set_transcript(
            [summary_message, *(message for _id, message in entries[kept_index:])]
        )
        self._context_entries = [
            (None, summary_message),
            *entries[kept_index:],
        ]
        self._latest_compaction_ts = time.time()
        return result

    async def _auto_compact_threshold(self) -> None:
        """Pi's idle-boundary threshold check before the next model request."""

        settings = self._harness.compaction_settings
        if settings is None or not settings.enabled:
            return
        context_window = getattr(self._model, "context_window", None)
        if not isinstance(context_window, int) or context_window <= 0:
            return
        tokens = self._threshold_context_tokens()
        if tokens is None or not should_compact(tokens, context_window, settings):
            return
        try:
            await self._apply_compaction(settings)
        except SummarizationError as exc:
            # Auto compaction is best-effort: the leg proceeds uncompacted
            # and the model outcome (success or overflow) stays authoritative.
            _logger.warning(
                "auto compaction for run %s failed: %s", self.run_id, exc
            )

    def _threshold_context_tokens(self) -> int | None:
        """Context tokens for the threshold rule, with Pi's staleness guard.

        The latest assistant usage is only trustworthy when the assistant
        answered after the latest compaction; kept pre-compaction usage
        reflects the old, larger context and would retrigger compaction
        immediately.
        """

        messages = [message for _record_id, message in self._context_entries]
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if (
                not isinstance(message, AssistantMessage)
                or message.failed
                or message.usage is None
            ):
                continue
            tokens = usage_context_tokens(message.usage)
            if tokens <= 0:
                continue
            if (
                self._latest_compaction_ts is not None
                and message.timestamp <= self._latest_compaction_ts
            ):
                return None
            return tokens + sum(
                estimate_tokens(item) for item in messages[index + 1 :]
            )
        if not messages:
            return None
        return estimate_context_tokens(messages)

    # ── overflow recovery ─────────────────────────────────────────────────

    def _overflow_detected(self) -> bool:
        tail = self._agent.messages[-1] if self._agent.messages else None
        if not isinstance(tail, AssistantMessage):
            return False
        context_window = getattr(self._model, "context_window", None)
        return is_context_overflow(
            tail, context_window if isinstance(context_window, int) else None
        )

    async def _overflow_recovery(self) -> AgentLoopResult | None:
        """Compact-and-continue once after a context-overflow failure."""

        if self._overflow_attempted:
            return None
        self._overflow_attempted = True
        settings = self._harness.compaction_settings
        assert settings is not None
        await self._advance()
        try:
            applied = await self._apply_compaction(settings)
        except SummarizationError as exc:
            _logger.warning(
                "overflow recovery compaction for run %s failed: %s",
                self.run_id,
                exc,
            )
            return None
        if applied is None:
            # Nothing to summarize: a retry would re-issue the identical
            # request and overflow again (Pi makes the same call).
            return None
        # The failed assistant tail stays durable in the forked journal's
        # inherited prefix, but it does not join the compact-and-retry
        # context (Pi drops it from agent state the same way).
        messages = list(self._agent.messages)
        if (
            messages
            and isinstance(messages[-1], AssistantMessage)
            and messages[-1].failed
        ):
            self._agent.set_transcript(messages[:-1])
            self._context_entries = self._context_entries[:-1]
        retry = await self._agent.continue_run()
        if isinstance(retry, AgentRunRejected):
            return None
        settled = await self._settle_leg(retry)
        if isinstance(settled, AgentRunRejected):
            return None
        return settled


__all__ = [
    "ResumeRejected",
    "SessionHarness",
    "SessionRun",
    "TaskTransitionRejected",
]
