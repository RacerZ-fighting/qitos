"""Pure replay of one Run journal into recoverable Session state.

``recover_session`` is a total function over one journal's replay (INHERITED
wrappers resolved): it rebuilds the typed transcript, the compaction-projected
context, the configuration lineage, open Tool operations, unconsumed runtime
inputs and the terminal outcome, and it fails closed with
``JournalCorruptionError`` on any contradiction instead of guessing. A Tool
call admitted but never terminated, or never admitted at all, is the
legitimate crash window when no commit or run terminal covers it;
``close_crashed_tool_calls`` closes that window with explicit cancelled
terminal records and never re-executes anything.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Sequence

from ...core.agent_loop import AgentRunStatus
from ...core.journal import (
    JournalCorruptionError,
    JournalRecord,
    JournalRecordRef,
    JournalRecordType,
    SessionJournal,
    resolve_inherited_record,
)
from ...core.message import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from ...core.plan import Plan, validate_plan_transition
from ...core.runtime_input import RuntimeInput
from ...core.task import Task, TaskLifecycle, TaskStatus
from ...core.thinking import ThinkingLevel
from ...core.tool_result import ToolResult
from .turn_recorder import (
    RecoveredRecorderState,
    decode_compaction,
    decode_input_accepted,
    decode_model_change,
    decode_model_completed,
    decode_plan_updated,
    decode_run_terminal,
    decode_runtime_input_consumed,
    decode_step_committed,
    decode_task_created,
    decode_task_transition,
    decode_thinking_change,
    decode_tool_started,
    decode_tool_terminal,
    decode_tools_change,
    decode_transcript_message,
    encode_step_committed,
    encode_tool_terminal,
    encode_transcript_message,
)


@dataclass(frozen=True, slots=True)
class RecoveredRunOutcome:
    """Terminal outcome decoded from one run journal's run terminal record.

    ``messages`` are this run's own transcript messages; an inherited fork
    prefix is never reported as new run output.
    """

    status: AgentRunStatus
    error: str | None
    messages: tuple[Message, ...]


@dataclass(frozen=True, slots=True)
class CrashedToolCall:
    """One Tool call torn by a crash, with its admission evidence.

    ``started_record_id`` is ``None`` when the call never reached Tool
    admission (provably no side effects). ``transcript_record_id`` names a
    torn tool transcript entry whose content is already durable; the crash
    closure links it instead of synthesizing a replacement.
    ``evidence_timestamp`` gives synthesized transcript entries a
    deterministic timestamp (the admission record's, or the tail assistant
    message's), so a repeated closure is an idempotent append.
    """

    turn: int
    call: ToolCall
    started_record_id: str | None
    transcript_record_id: str | None
    evidence_timestamp: float


@dataclass(frozen=True, slots=True)
class RecoveredTask:
    """One Task definition plus its lifecycle folded from the journal."""

    definition: Task
    lifecycle: TaskLifecycle


@dataclass(frozen=True, slots=True)
class RecoveredSession:
    """Everything a harness needs to resume, fork or close one Run journal.

    ``transcript`` is the canonical conversation order: committed entries in
    commit order, then the uncommitted tail in journal order.
    ``uncommitted_transcript_record_ids`` / ``uncommitted_terminal_record_ids``
    name the tail records no ``step.committed`` covers yet; the crash closure
    folds them into its closing commit. ``tasks`` projects every
    ``task.created`` definition with its lifecycle folded from the
    ``task.transition`` records through the fork lineage. ``plans`` folds
    Task-bound ``plan.updated`` replacements without treating one Task's
    strategy as another Task's state.
    """

    run_id: str
    transcript: tuple[Message, ...]
    transcript_record_ids: tuple[str, ...]
    context_messages: tuple[Message, ...]
    next_turn: int
    model_identity: tuple[str, str, str] | None
    thinking_level: ThinkingLevel | None
    active_tool_names: tuple[str, ...] | None
    unterminated_calls: tuple[CrashedToolCall, ...]
    unstarted_calls: tuple[CrashedToolCall, ...]
    unconsumed_inputs: tuple[RuntimeInput, ...]
    outcome: RecoveredRunOutcome | None
    recorder_state: RecoveredRecorderState
    crash_turn: int | None
    crash_turn_transcript_entries: int
    uncommitted_transcript_record_ids: tuple[str, ...]
    uncommitted_terminal_record_ids: tuple[str, ...]
    tasks: Mapping[str, RecoveredTask]
    plans: Mapping[str, Plan]

    @property
    def unfinished_root(self) -> RecoveredTask | None:
        """The lineage's single unfinished Root Task, when one exists."""

        for task in self.tasks.values():
            if (
                task.definition.parent_task_id is None
                and not task.lifecycle.status.terminal
            ):
                return task
        return None

    @property
    def plan(self) -> Plan | None:
        """The Plan bound to the latest Task in this journal lineage."""

        if not self.tasks:
            return None
        task_id = next(reversed(self.tasks))
        return self.plans.get(task_id)


@dataclass(frozen=True, slots=True)
class _CompactionFact:
    record_index: int
    transcript_position: int
    summary: str
    first_kept_transcript_id: str
    timestamp: float


_TRANSCRIPT_ID_PATTERN = re.compile(r"^(.+):turn:(\d+):transcript:(\d+)$")


def _epoch(record: JournalRecord) -> float:
    try:
        return datetime.fromisoformat(record.timestamp).timestamp()
    except ValueError as exc:
        raise JournalCorruptionError("journal record timestamp is invalid") from exc


def _parse_transcript_turn(record_id: str) -> tuple[str, int]:
    match = _TRANSCRIPT_ID_PATTERN.match(record_id)
    if match is None:
        raise JournalCorruptionError(
            "transcript.message record id is not a loop record id"
        )
    return match.group(1), int(match.group(2))


def recover_session(records: Sequence[JournalRecord]) -> RecoveredSession:
    """Replay one journal into recoverable Session state, failing closed.

    The replay resolves INHERITED wrappers, so one forked journal is
    self-contained recovery truth. Contradictions raise
    ``JournalCorruptionError``: dangling transcript references, tool
    transcript entries without their ``tool.terminal`` outside the crash
    window, commits or run terminals covering open calls, unknown commit
    references, conflicting run terminals, records after a run terminal,
    regressing turns, invalid compaction cuts and invalid payload shapes.
    """

    resolved: list[tuple[JournalRecord, JournalRecord, bool]] = []
    seen_record_ids: set[str] = set()
    for record in records:
        effective = resolve_inherited_record(record)
        if effective.record_id in seen_record_ids:
            raise JournalCorruptionError("journal contains a duplicate record id")
        seen_record_ids.add(effective.record_id)
        resolved.append(
            (record, effective, record.type is not JournalRecordType.INHERITED)
        )
    if not resolved:
        raise JournalCorruptionError("journal replay is empty")
    _first_record, first_effective, first_own = resolved[0]
    if not first_own or first_effective.type is not JournalRecordType.RUN_STARTED:
        raise JournalCorruptionError("journal does not start with run.started")
    run_id = first_effective.run_id

    transcript: list[Message] = []
    transcript_ids: list[str] = []
    transcript_index: dict[str, int] = {}
    transcript_turns: dict[str, int] = {}
    own_transcript_ids: set[str] = set()
    # Tool-call state is keyed by (owning run, call id): call ids are only
    # unique within one run, and a fork prefix is a sequence of run segments.
    started: dict[tuple[str, str], tuple[int, ToolCall, str, float, bool]] = {}
    terminals: dict[tuple[str, str], tuple[int, ToolCall, str, str, bool]] = {}
    terminal_record_ids: set[str] = set()
    covered_terminal_ids: set[str] = set()
    covered_entry_ids: set[str] = set()
    commit_entry_ids: list[str] = []
    unreferenced_tool_entries: dict[tuple[str, str], list[tuple[str, int]]] = {}
    model_identity: tuple[str, str, str] | None = None
    thinking_level: ThinkingLevel | None = None
    tools_change_names: tuple[str, ...] | None = None
    tools_change_index = -1
    added_tool_entries: list[tuple[int, tuple[str, ...]]] = []
    compactions: list[_CompactionFact] = []
    posted_inputs: dict[str, RuntimeInput] = {}
    consumed_inputs: set[str] = set()
    tasks: dict[str, RecoveredTask] = {}
    plans: dict[str, Plan] = {}
    # Run segments (own or inherited) that already folded a model, Tool,
    # transcript or input side effect; a root task.created must precede them.
    segment_side_effects: set[str] = set()
    outcome_status: AgentRunStatus | None = None
    outcome_error: str | None = None
    input_accepted_seen = False
    last_turn = -1
    fork_seen = False
    prefix_closed = False
    last_inherited_type: JournalRecordType | None = None
    previous_top_type: JournalRecordType | None = None
    previous_effective_type: JournalRecordType | None = None
    current_segment_run = run_id

    def _turn_barrier(turn: int) -> None:
        nonlocal last_turn
        if turn < last_turn:
            raise JournalCorruptionError("journal turns must not regress")
        for key, (started_turn, _c, _r, _e, _o) in started.items():
            if key[0] == current_segment_run and started_turn < turn and key not in terminals:
                raise JournalCorruptionError(
                    "journal continued past an unterminated tool call"
                )
        for (segment, _call_id), entries in unreferenced_tool_entries.items():
            if segment != current_segment_run:
                continue
            for _record_id, entry_turn in entries:
                if entry_turn < turn:
                    raise JournalCorruptionError(
                        "journal continued past a torn tool transcript entry"
                    )
        last_turn = turn

    def _commit_barrier(kind: str) -> None:
        for key in started:
            if key[0] == current_segment_run and key not in terminals:
                raise JournalCorruptionError(f"{kind} covers an unterminated tool call")
        torn = [
            entries
            for (segment, _call_id), entries in unreferenced_tool_entries.items()
            if segment == current_segment_run
        ]
        if any(torn):
            raise JournalCorruptionError(
                f"{kind} covers a torn tool transcript entry"
            )

    def _close_segment(kind: str) -> None:
        # A fork prefix is a sequence of closed run segments: every segment
        # ends at its committed boundary, and the younger run restarts its
        # own turn numbering. Tool-call state stays keyed by run, since call
        # ids are only unique within one run.
        nonlocal last_turn
        _commit_barrier(kind)
        last_turn = -1

    def _close_inherited_prefix() -> None:
        # A forked run restarts its own turn numbering after the inherited
        # prefix, and the prefix itself is always one committed boundary.
        nonlocal prefix_closed
        if last_inherited_type is not JournalRecordType.STEP_COMMITTED:
            raise JournalCorruptionError(
                "inherited prefix does not end at a committed boundary"
            )
        _close_segment("inherited prefix")
        prefix_closed = True

    for index, (record, effective, is_own) in enumerate(resolved):
        if outcome_status is not None:
            raise JournalCorruptionError("journal has a record after its run terminal")
        if is_own and effective.run_id != run_id:
            raise JournalCorruptionError("journal mixes records from another run")
        record_type = effective.type
        payload = effective.payload
        if not is_own and effective.run_id != current_segment_run:
            # Inside one nested fork prefix, each run's records form one
            # contiguous segment. A new segment either opens a younger run's
            # header (right after its run.forked) or follows the older
            # segment's committed boundary.
            if previous_effective_type is JournalRecordType.RUN_FORKED:
                if record_type is not JournalRecordType.RUN_STARTED:
                    raise JournalCorruptionError(
                        "inherited segment does not start with run.started"
                    )
            elif previous_effective_type is JournalRecordType.STEP_COMMITTED:
                _close_segment("inherited prefix")
            else:
                raise JournalCorruptionError(
                    "inherited prefix is not a sequence of committed run segments"
                )
            current_segment_run = effective.run_id
        if not is_own:
            if not fork_seen:
                raise JournalCorruptionError(
                    "journal inherits records without a fork"
                )
            if prefix_closed or previous_top_type not in (
                JournalRecordType.RUN_FORKED,
                JournalRecordType.INHERITED,
            ):
                raise JournalCorruptionError(
                    "journal inherited prefix is not contiguous"
                )
            last_inherited_type = record_type
        else:
            if record_type is JournalRecordType.RUN_FORKED:
                if fork_seen or index != 1:
                    raise JournalCorruptionError(
                        "run.forked is not the first run event"
                    )
                fork_seen = True
            elif (
                record_type is not JournalRecordType.RUN_STARTED
                and last_inherited_type is not None
                and not prefix_closed
            ):
                _close_inherited_prefix()
        previous_top_type = record.type
        previous_effective_type = record_type
        try:
            if record_type is JournalRecordType.RUN_STARTED:
                if is_own and index != 0:
                    raise JournalCorruptionError("journal contains a second run.started")
            elif record_type is JournalRecordType.RUN_FORKED:
                pass
            elif record_type is JournalRecordType.TRANSCRIPT_MESSAGE:
                message = decode_transcript_message(payload)
                segment_side_effects.add(effective.run_id)
                entry_run, entry_turn = _parse_transcript_turn(effective.record_id)
                if is_own:
                    if entry_run != run_id:
                        raise JournalCorruptionError(
                            "transcript.message belongs to another run"
                        )
                    own_transcript_ids.add(effective.record_id)
                _turn_barrier(entry_turn)
                transcript_index[effective.record_id] = len(transcript)
                transcript_ids.append(effective.record_id)
                transcript_turns[effective.record_id] = entry_turn
                transcript.append(message)
                if isinstance(message, ToolResultMessage):
                    unreferenced_tool_entries.setdefault(
                        (entry_run, message.tool_call_id), []
                    ).append((effective.record_id, entry_turn))
                    if message.added_tool_names:
                        added_tool_entries.append((index, message.added_tool_names))
            elif record_type is JournalRecordType.MODEL_COMPLETED:
                turn, _request, message_record_id = decode_model_completed(payload)
                segment_side_effects.add(effective.run_id)
                _turn_barrier(turn)
                entry_position = transcript_index.get(message_record_id)
                if entry_position is None:
                    raise JournalCorruptionError(
                        "model.completed references an unknown transcript entry"
                    )
                if not isinstance(transcript[entry_position], AssistantMessage):
                    raise JournalCorruptionError(
                        "model.completed must reference an assistant message"
                    )
            elif record_type is JournalRecordType.MODEL_CHANGE:
                model_identity = decode_model_change(payload)
            elif record_type is JournalRecordType.THINKING_CHANGE:
                thinking_level = decode_thinking_change(payload)
            elif record_type is JournalRecordType.TOOLS_CHANGE:
                tools_change_names = decode_tools_change(payload)
                tools_change_index = index
            elif record_type is JournalRecordType.TOOL_STARTED:
                turn, call = decode_tool_started(payload)
                segment_side_effects.add(effective.run_id)
                _turn_barrier(turn)
                call_key = (effective.run_id, call.id)
                if call_key in started:
                    raise JournalCorruptionError(
                        "journal contains a duplicate tool.started"
                    )
                if call_key in terminals:
                    raise JournalCorruptionError("tool.started occurs after tool.terminal")
                started[call_key] = (
                    turn,
                    call,
                    effective.record_id,
                    _epoch(effective),
                    is_own,
                )
            elif record_type is JournalRecordType.TOOL_TERMINAL:
                turn, call, message_record_id = decode_tool_terminal(payload)
                segment_side_effects.add(effective.run_id)
                _turn_barrier(turn)
                call_key = (effective.run_id, call.id)
                if call_key in terminals:
                    raise JournalCorruptionError(
                        "journal contains a duplicate tool.terminal"
                    )
                entry_position = transcript_index.get(message_record_id)
                if entry_position is None:
                    raise JournalCorruptionError(
                        "tool.terminal references an unknown transcript entry"
                    )
                entry = transcript[entry_position]
                if not isinstance(entry, ToolResultMessage):
                    raise JournalCorruptionError(
                        "tool.terminal must reference a tool message"
                    )
                if entry.tool_call_id != call.id or entry.tool_name != call.name:
                    raise JournalCorruptionError(
                        "tool.terminal does not match its transcript entry"
                    )
                entries = unreferenced_tool_entries.get(call_key, [])
                unreferenced_tool_entries[call_key] = [
                    item for item in entries if item[0] != message_record_id
                ]
                terminals[call_key] = (
                    turn,
                    call,
                    effective.record_id,
                    message_record_id,
                    is_own,
                )
                terminal_record_ids.add(effective.record_id)
            elif record_type is JournalRecordType.STEP_COMMITTED:
                turn, commit_transcript_ids, commit_terminal_ids = (
                    decode_step_committed(payload)
                )
                _turn_barrier(turn)
                _commit_barrier("step.committed")
                for record_id in commit_transcript_ids:
                    if record_id not in transcript_index:
                        raise JournalCorruptionError(
                            "step.committed references an unknown transcript entry"
                        )
                    if record_id in covered_entry_ids:
                        raise JournalCorruptionError(
                            "transcript entry was committed twice"
                        )
                for record_id in commit_terminal_ids:
                    if record_id not in terminal_record_ids:
                        raise JournalCorruptionError(
                            "step.committed references an unknown tool terminal"
                        )
                    if record_id in covered_terminal_ids:
                        raise JournalCorruptionError("tool terminal was committed twice")
                covered_entry_ids.update(commit_transcript_ids)
                covered_terminal_ids.update(commit_terminal_ids)
                commit_entry_ids.extend(commit_transcript_ids)
            elif record_type is JournalRecordType.INPUT_ACCEPTED:
                segment_side_effects.add(effective.run_id)
                if is_own:
                    input_accepted_seen = True
                for record_id in decode_input_accepted(payload):
                    if record_id not in transcript_index:
                        raise JournalCorruptionError(
                            "input.accepted references an unknown transcript entry"
                        )
            elif record_type is JournalRecordType.COMPACTION:
                if len(covered_entry_ids) != len(transcript_index):
                    raise JournalCorruptionError(
                        "compaction follows uncommitted transcript entries"
                    )
                summary, first_kept, _tokens, _usage = decode_compaction(payload)
                compactions.append(
                    _CompactionFact(
                        record_index=index,
                        transcript_position=len(transcript),
                        summary=summary,
                        first_kept_transcript_id=first_kept,
                        timestamp=_epoch(effective),
                    )
                )
            elif record_type is JournalRecordType.RUNTIME_INPUT_POSTED:
                if is_own:
                    try:
                        runtime_input = RuntimeInput.from_dict(payload)
                    except (TypeError, ValueError) as exc:
                        raise JournalCorruptionError(
                            "runtime_input.posted is not decodable"
                        ) from exc
                    if runtime_input.event_id in posted_inputs:
                        raise JournalCorruptionError(
                            "journal contains a duplicate runtime input"
                        )
                    posted_inputs[runtime_input.event_id] = runtime_input
            elif record_type is JournalRecordType.RUNTIME_INPUT_CONSUMED:
                if is_own:
                    event_id = decode_runtime_input_consumed(payload)
                    if event_id not in posted_inputs:
                        raise JournalCorruptionError(
                            "runtime_input.consumed references an unknown input"
                        )
                    if event_id in consumed_inputs:
                        raise JournalCorruptionError("runtime input was consumed twice")
                    consumed_inputs.add(event_id)
            elif record_type is JournalRecordType.TASK_CREATED:
                task = decode_task_created(payload)
                existing = tasks.get(task.task_id)
                if existing is not None and existing.definition != task:
                    raise JournalCorruptionError(
                        "task.created conflicts with the recorded task definition"
                    )
                if existing is None:
                    if task.parent_task_id is None:
                        if effective.run_id in segment_side_effects:
                            raise JournalCorruptionError(
                                "root task.created follows the run's model or "
                                "transcript side effects"
                            )
                        if any(
                            recorded.definition.parent_task_id is None
                            and not recorded.lifecycle.status.terminal
                            for recorded in tasks.values()
                        ):
                            raise JournalCorruptionError(
                                "journal holds a second unfinished root task"
                            )
                    tasks[task.task_id] = RecoveredTask(
                        definition=task,
                        lifecycle=TaskLifecycle(status=TaskStatus.ACTIVE),
                    )
                # An identical duplicate settles as an idempotent append.
            elif record_type is JournalRecordType.TASK_TRANSITION:
                (
                    task_id,
                    from_status,
                    to_status,
                    reason,
                    blocker,
                    usage,
                ) = decode_task_transition(payload)
                current = tasks.get(task_id)
                if current is None:
                    raise JournalCorruptionError(
                        "task.transition references an unknown task"
                    )
                folded = current.lifecycle
                if folded.status.terminal:
                    raise JournalCorruptionError(
                        "task.transition follows a terminal task status"
                    )
                if folded.status is not from_status:
                    raise JournalCorruptionError(
                        "task.transition from_status does not match the folded state"
                    )
                tasks[task_id] = RecoveredTask(
                    definition=current.definition,
                    lifecycle=TaskLifecycle(
                        status=to_status,
                        usage=usage,
                        blocker=blocker,
                        terminal_reason=reason,
                    ),
                )
            elif record_type is JournalRecordType.PLAN_UPDATED:
                task_id, proposed = decode_plan_updated(payload)
                if task_id not in tasks:
                    raise JournalCorruptionError(
                        "plan.updated references an unknown task"
                    )
                validate_plan_transition(plans.get(task_id), proposed)
                plans[task_id] = proposed
            elif record_type in (
                JournalRecordType.RUN_COMPLETED,
                JournalRecordType.RUN_INTERRUPTED,
            ):
                if not is_own:
                    raise JournalCorruptionError(
                        "inherited prefix contains a run terminal"
                    )
                _commit_barrier("run terminal")
                outcome_status, outcome_error = decode_run_terminal(
                    record_type, payload
                )
            elif record_type is JournalRecordType.BUDGET_COMMITTED:
                pass
            elif record_type in (
                JournalRecordType.PROCESS_STARTED,
                JournalRecordType.PROCESS_TERMINAL,
                JournalRecordType.CHILD_STARTED,
                JournalRecordType.CHILD_TERMINAL,
            ):
                pass
            else:
                raise JournalCorruptionError("journal record type is not recoverable")
        except ValueError as exc:
            raise JournalCorruptionError(str(exc)) from exc

    # ── canonical conversation order ──────────────────────────────────────
    # Committed entries follow their step.committed reference lists (that is
    # the order the messages entered the conversation); the uncommitted tail
    # follows journal order, which is conversation order there because the
    # crash window's entries are journaled eagerly.

    if fork_seen and last_inherited_type is None:
        raise JournalCorruptionError("run.forked is not followed by its prefix")
    if last_inherited_type is not None and not prefix_closed:
        _close_inherited_prefix()

    tail_entry_ids = [
        record_id
        for record_id in transcript_ids
        if record_id not in covered_entry_ids
    ]
    conversation_ids = (*commit_entry_ids, *tail_entry_ids)
    message_by_id = {
        record_id: transcript[position]
        for record_id, position in transcript_index.items()
    }
    conversation = tuple(message_by_id[record_id] for record_id in conversation_ids)
    conversation_index = {
        record_id: position for position, record_id in enumerate(conversation_ids)
    }
    transcript = list(conversation)
    transcript_ids = list(conversation_ids)
    transcript_index = conversation_index
    uncommitted_terminal_ids = tuple(
        terminal[2] for terminal in terminals.values() if terminal[2] not in covered_terminal_ids
    )

    # ── cross-record consistency ──────────────────────────────────────────

    unterminated: list[CrashedToolCall] = []
    for call_key, (turn, call, record_id, timestamp, is_own) in started.items():
        if call_key in terminals:
            continue
        if not is_own:
            raise JournalCorruptionError(
                "inherited prefix contains an unterminated tool call"
            )
        torn = unreferenced_tool_entries.get(call_key, [])
        if len(torn) > 1:
            raise JournalCorruptionError(
                "tool call has multiple torn transcript entries"
            )
        unterminated.append(
            CrashedToolCall(
                turn=turn,
                call=call,
                started_record_id=record_id,
                transcript_record_id=torn[0][0] if torn else None,
                evidence_timestamp=timestamp,
            )
        )

    unstarted: list[CrashedToolCall] = []
    tail_assistant_position = (
        len(transcript) - 1
        if transcript and isinstance(transcript[-1], AssistantMessage)
        else None
    )
    for position, message in enumerate(transcript):
        if not isinstance(message, AssistantMessage):
            continue
        entry_run = _parse_transcript_turn(transcript_ids[position])[0]
        open_calls = [
            call
            for call in message.tool_calls
            if (entry_run, call.id) not in started
            and (entry_run, call.id) not in terminals
        ]
        if not open_calls or message.failed:
            # Failed assistant messages stop before Tool admission by
            # contract; their calls are protocol-failure evidence.
            continue
        if (
            position != tail_assistant_position
            or transcript_ids[position] in covered_entry_ids
        ):
            # A committed assistant with never-admitted calls is a
            # contradiction: the turn finished, so the calls cannot be a
            # crash window.
            raise JournalCorruptionError(
                "assistant tool calls were neither admitted nor closed"
            )
        if transcript_ids[position] not in own_transcript_ids:
            raise JournalCorruptionError(
                "an inherited tail assistant cannot hold unstarted calls"
            )
        for call in open_calls:
            torn = unreferenced_tool_entries.get((entry_run, call.id), [])
            if len(torn) > 1:
                raise JournalCorruptionError(
                    "tool call has multiple torn transcript entries"
                )
            unstarted.append(
                CrashedToolCall(
                    turn=transcript_turns[transcript_ids[position]],
                    call=call,
                    started_record_id=None,
                    transcript_record_id=torn[0][0] if torn else None,
                    evidence_timestamp=message.timestamp,
                )
            )

    crash_turn: int | None = None
    if unterminated or unstarted:
        crash_turns = {crashed.turn for crashed in (*unterminated, *unstarted)}
        if len(crash_turns) != 1:
            raise JournalCorruptionError("crash window spans multiple turns")
        crash_turn = crash_turns.pop()
    crash_turn_entries = (
        sum(
            1
            for record_id in own_transcript_ids
            if transcript_turns[record_id] == crash_turn
        )
        if crash_turn is not None
        else 0
    )

    # ── configuration lineage ─────────────────────────────────────────────

    active_tool_names: tuple[str, ...] | None = None
    if tools_change_names is not None:
        merged = list(tools_change_names)
        for entry_index, names in added_tool_entries:
            if entry_index <= tools_change_index:
                continue
            for name in names:
                if name not in merged:
                    merged.append(name)
        active_tool_names = tuple(merged)

    # ── compaction projection ─────────────────────────────────────────────

    for fact in compactions:
        kept = transcript_index.get(fact.first_kept_transcript_id)
        if kept is None or kept >= fact.transcript_position:
            raise JournalCorruptionError(
                "compaction first_kept_transcript_id is unresolvable"
            )
        if isinstance(transcript[kept], ToolResultMessage):
            raise JournalCorruptionError(
                "compaction cut must not land on a tool message"
            )
    context_messages: tuple[Message, ...]
    if compactions:
        latest = compactions[-1]
        kept_from = transcript_index[latest.first_kept_transcript_id]
        summaries: dict[int, list[_CompactionFact]] = {}
        for fact in compactions[:-1]:
            if fact.transcript_position >= kept_from:
                summaries.setdefault(fact.transcript_position, []).append(fact)
        context: list[Message] = [
            UserMessage(content=latest.summary, timestamp=latest.timestamp)
        ]
        for position in range(kept_from, len(transcript)):
            for fact in summaries.get(position, []):
                context.append(
                    UserMessage(content=fact.summary, timestamp=fact.timestamp)
                )
            context.append(transcript[position])
        for fact in summaries.get(len(transcript), []):
            context.append(UserMessage(content=fact.summary, timestamp=fact.timestamp))
        context_messages = tuple(context)
    else:
        context_messages = tuple(transcript)

    next_turn = last_turn + 1
    outcome: RecoveredRunOutcome | None = None
    if outcome_status is not None:
        outcome = RecoveredRunOutcome(
            status=outcome_status,
            error=outcome_error,
            messages=tuple(
                message
                for record_id, message in zip(transcript_ids, transcript)
                if record_id in own_transcript_ids
            ),
        )
    return RecoveredSession(
        run_id=run_id,
        transcript=tuple(transcript),
        transcript_record_ids=tuple(transcript_ids),
        context_messages=context_messages,
        next_turn=next_turn,
        model_identity=model_identity,
        thinking_level=thinking_level,
        active_tool_names=active_tool_names,
        unterminated_calls=tuple(unterminated),
        unstarted_calls=tuple(unstarted),
        unconsumed_inputs=tuple(
            runtime_input
            for event_id, runtime_input in posted_inputs.items()
            if event_id not in consumed_inputs
        ),
        outcome=outcome,
        recorder_state=RecoveredRecorderState(
            next_turn=next_turn,
            model_identity=model_identity,
            thinking_level=thinking_level,
            active_tool_names=active_tool_names,
            recorded_message_count=len(transcript),
            input_accepted=input_accepted_seen,
        ),
        crash_turn=crash_turn,
        crash_turn_transcript_entries=crash_turn_entries,
        uncommitted_transcript_record_ids=tuple(tail_entry_ids),
        uncommitted_terminal_record_ids=uncommitted_terminal_ids,
        tasks=MappingProxyType(tasks),
        plans=MappingProxyType(plans),
    )


def recover_run_outcome(
    records: Sequence[JournalRecord],
) -> RecoveredRunOutcome | None:
    """Decode the run terminal record of one loop journal.

    Returns ``None`` when the run never reached a terminal record (the
    process exited mid-run). Raises ``ValueError`` when the journal is not a
    consistent loop journal — records written by the retired Engine path and
    every other contradiction fail closed here, never guessed.
    """

    try:
        session = recover_session(records)
    except JournalCorruptionError as exc:
        raise ValueError("run journal is not a recoverable loop journal") from exc
    return session.outcome


async def close_crashed_tool_calls(
    journal: SessionJournal,
    recovered: RecoveredSession,
) -> tuple[JournalRecordRef, ...]:
    """Close one recovered crash window with explicit cancelled terminals.

    Admitted-but-unterminated calls and never-admitted calls each receive a
    ``tool.terminal`` plus its transcript entry (a torn durable entry is
    linked, never replaced), followed by one closing ``step.committed``.
    Nothing is re-executed and side effects are never guessed: every
    synthesized result is ``cancelled`` with an explicit unknown-outcome
    error that distinguishes the two categories. All record ids and payloads
    are deterministic functions of the recovered journal, so a repeated
    closure settles as idempotent appends and a fresh recovery afterwards
    finds no crash window at all.
    """

    if not isinstance(recovered, RecoveredSession):
        raise TypeError("recovered must be a RecoveredSession")
    if journal.run_id != recovered.run_id:
        raise ValueError("journal is not open for the recovered run")
    crashed = (*recovered.unterminated_calls, *recovered.unstarted_calls)
    if not crashed:
        return ()
    if recovered.crash_turn is None:
        raise ValueError("recovered crash window has no owning turn")
    turn = recovered.crash_turn
    run_id = recovered.run_id
    transcript_seq = recovered.crash_turn_transcript_entries

    transcript_record_ids: list[str] = []
    tool_terminal_record_ids: list[str] = []
    closed: list[JournalRecordRef] = []
    for crashed_call in crashed:
        call = crashed_call.call
        transcript_record_id = crashed_call.transcript_record_id
        if transcript_record_id is None:
            if crashed_call.started_record_id is not None:
                error = (
                    f'Tool call "{call.name}" was interrupted: the process '
                    "exited after the tool started; its outcome is unknown "
                    "and the call is closed as cancelled without re-execution."
                )
            else:
                error = (
                    f'Tool call "{call.name}" was never executed: the process '
                    "exited before tool admission; the call is closed as "
                    "cancelled and had no side effects."
                )
            result = ToolResult(
                status="cancelled",
                output=None,
                error=error,
                metadata={
                    "tool_name": call.name,
                    "error_category": "cancelled",
                    "cancel_source": "crash_recovery",
                    "started": crashed_call.started_record_id is not None,
                },
                call_id=call.id,
            ).frozen()
            message = ToolResultMessage(
                tool_call_id=call.id,
                tool_name=call.name,
                result=result,
                usage=result.usage,
                added_tool_names=result.added_tool_names,
                timestamp=crashed_call.evidence_timestamp,
            )
            transcript_record_id = f"{run_id}:turn:{turn}:transcript:{transcript_seq}"
            transcript_seq += 1
            await journal.append(
                JournalRecordType.TRANSCRIPT_MESSAGE,
                encode_transcript_message(message),
                record_id=transcript_record_id,
            )
            transcript_record_ids.append(transcript_record_id)
        terminal_record_id = f"{run_id}:turn:{turn}:tool:{call.id}:terminal"
        await journal.append(
            JournalRecordType.TOOL_TERMINAL,
            encode_tool_terminal(turn, call, transcript_record_id),
            record_id=terminal_record_id,
        )
        tool_terminal_record_ids.append(terminal_record_id)
        closed.append(JournalRecordRef(run_id, terminal_record_id))
    # The closing commit folds the crash turn's uncommitted tail (its
    # assistant message and any torn entries) ahead of the closure entries,
    # keeping every Tool call ahead of its result in conversation order.
    await journal.append(
        JournalRecordType.STEP_COMMITTED,
        encode_step_committed(
            turn,
            [
                *recovered.uncommitted_transcript_record_ids,
                *transcript_record_ids,
            ],
            [
                *recovered.uncommitted_terminal_record_ids,
                *tool_terminal_record_ids,
            ],
        ),
        record_id=f"{run_id}:turn:{turn}:committed",
    )
    return tuple(closed)


__all__ = [
    "CrashedToolCall",
    "RecoveredRunOutcome",
    "RecoveredSession",
    "RecoveredTask",
    "close_crashed_tool_calls",
    "recover_run_outcome",
    "recover_session",
]
