# QitOS Agent runtime architecture

## Status

Accepted target architecture on 2026-08-16. On the migration branch, the minimal loop,
the `Agent` façade, the authoritative Session/Harness, the goal-bearing Task and the
progress-checklist Plan are the only execution and recovery path. Final integration gates
and merge remain separate release steps.

This document owns the final QitOS runtime boundaries. The
[migration plan](../plans/pi-aligned-agent-runtime.md) owns the remaining sequencing.

## 1. Architecture

```text
Provider APIs
└── Model / Provider
    └── Message / ModelEvent / Usage / continuation
        │
        ▼
minimal async Agent loop
└── Message → Model → ToolCall → ToolResult → next turn
        │
        ▼
Agent façade / Session Harness
├── prompt / steer / follow-up / abort / idle
├── goal-bearing Task + transcript + operation records
├── compact / resume / fork / pure recovery
└── Tool / Runtime / Skill / MCP / Plan / Subagent primitives
        │
        ▼
application composition
```

The architecture preserves proven durable Tool transactions, absolute deadlines,
cancellation, ordered results, traceability and recovery. It does not preserve
`Observation → Decision → Action → reduce` or a large application-policy façade as the
public execution model.

## 2. Model and Provider

The Model boundary owns provider-neutral messages, model identity, usage,
reasoning/thinking, continuation and discriminated stream events. OpenAI Responses and
Anthropic Messages are first-class; Chat Completions is a compatibility transport.
Provider SDK objects stop at the adapter edge.

Continuation is an optimization. Canonical local transcript is sufficient to resume,
fork or switch Provider. Retry is allowed only before observable output or external
side effects make replay unsafe.

## 3. Minimal Agent loop

The loop owns only:

- context projection and one immutable model request;
- assistant text/reasoning streaming;
- ToolCall validation, execution and finalization;
- ordered ToolResult insertion;
- steering/follow-up safe points and stop evaluation.

It does not discover resources, construct environments, load products, evaluate
benchmarks, own Session storage or reduce application domain state.

The loop freezes one request view, then optionally asks the application for typed
`ContextMessage` values. It appends those values to live history and commits the
turn's new prompt, steering and context input before model admission. This is a narrow
request boundary, not a state reducer or extension bus. Request-only
`transform_context` remains a provider-view transformation and must not be used as a
durable application-state owner.

The transaction boundary records accepted input, per-turn input, model terminal, Tool
admission, Tool terminal and turn commit around side effects. Every uniquely identified
call that reaches Tool admission, including a per-call rejection, receives exactly one
terminal ToolResult. Duplicate raw call ids remain assistant protocol-failure evidence
and the ambiguous batch is rejected before Tool admission or side effects.

## 4. Agent façade and Session Harness

The façade presents context, event subscription and the current Session/Harness.
The Harness owns prompt, queue, abort, idle, compact, resume and fork operations.
Expected rejections use typed results; corruption and implementation faults use
exceptions.

One Run may carry one frozen async resource finalizer composed by the application.
The Run owner awaits it exactly once after model/Tool work has settled and before
the canonical Run terminal append, including fault, deadline and cancellation
paths. A cleanup failure is retained as a bounded typed finalization diagnostic;
it does not replace the primary Run status or error. The finalizer receives only
the current leg's stable Run id and quiesces resources owned by that Run. It does
not close reusable Env, Tool or MCP owners shared across Session legs; the outer
application performs full teardown exactly once. This is not an extension Hook list.

Ordinary faults and caller cancellation terminalize started work before the Run
terminal when canonical appends settle. An append failure or unknown durable outcome
stops the writer: it never guesses a Tool terminal or crosses an open Tool boundary
with a Run terminal. The owner must close and replay the writer, then recovery closes
the missing Tool terminals in input order without re-executing handlers.

### Canonical log

One Run journal is one canonical append-only log; a Session is the lineage of
Run journals linked by fork. Every journal embeds its inherited fork prefix, so
each journal is self-contained recovery truth. Resuming an unfinished Run
continues the same journal; continuing from a terminal Run explicitly forks a
new journal (and, per §5, a new Task). One journal accepts at most one active
writer and one unfinished Run.

The log carries two typed record families in one sequence:

- Transcript entries own message content: `transcript.message` (user, runtime
  context, assistant and Tool-result messages) and `compaction` (summary plus
  the first kept transcript reference). `ContextMessage` is model-visible
  history, not a user instruction or an application-state truth source.
- Operation records own lifecycle and side-effect boundaries: `run.started`,
  `input.accepted`, `turn_input.committed` (ordered prompt, steering and
  context references durable before sampling), `model.completed` (exact
  request audit plus the assistant transcript reference), `tool.started` /
  `tool.terminal` (call plus terminal transcript reference), `step.committed`
  (turn commit marker listing the remaining transcript and Tool-terminal
  references), `run.completed` /
  `run.interrupted` (primary status/error plus an optional resource-finalization
  diagnostic), `model.change` / `thinking.change` / `tools.change`
  (per-turn configuration freeze diffs), `budget.step_reserved` (atomic
  Root-lineage model admission), `budget.committed` (settled token/cost usage),
  `process.*`, `subagent.*`, `runtime_input.posted` / `runtime_input.consumed`,
  `task.*`, `plan.updated`, `run.forked` and `journal.inherited`.

Records that carry references resolve fail closed through `journal.inherited`
wrappers. Message content appears in exactly one record (its transcript
entry); operation records reference it by record id.

### Recovery

Pure recovery replays the log into the transcript, the configuration lineage
(model, thinking level, active Tools), open Tool operations, Task/Plan state,
unconsumed runtime inputs and the terminal outcome, and fails closed on
contradictions. A Tool call admitted but never terminated (crash window) is
closed with an explicit cancelled terminal record; recovery never re-executes
Tools and never guesses side effects. Runtime inputs posted by the current Run
and never consumed are re-projected exactly once; inherited fork facts and
foreground results are never redelivered. Queued steering/follow-up messages
are memory-only and are not recovery truth.

Full replay occurs when a Session starts, resumes or forks. During a live Run,
the Agent owns its in-memory transcript and per-turn snapshots; successful
appends update that live state incrementally. The JSONL recovery contract does
not imply reconstructing runtime state from the entire journal before every
model request.

Memory and JSONL journal implementations share one conformance suite; a
SQLite index remains a disposable projection. Authoritative snapshots come
from Session state. Progress events, trace files and UI projections are
observational and cannot mutate recovery truth.

### Compaction

Compaction replaces history before a cut point with a model-generated summary
injected as a user message. The cut never lands between a Tool call and its
result, so ToolCall/ToolResult pairing survives compaction, resume and fork.
The compaction entry is durable; context rebuild keeps the latest compaction
plus entries after its cut. Historical records are never rewritten or deleted.
Manual compaction runs at idle; automatic compaction evaluates a token
threshold at idle boundaries, and a one-shot overflow recovery compacts and
continues once after a context-overflow model failure. Compaction invalidates
Provider continuation, which is only an optimization.

### Configuration lineage

The transaction boundary writes `model.change`, `thinking.change` and
`tools.change` when a per-turn freeze observes a diff, so configuration cannot
diverge from the log through façade setters. Resume restores the transcript,
thinking level and configuration lineage into a fresh façade, verifies the
provided Model identity and Tool registry coverage against the lineage
(including Tool names activated by earlier Tool results), and rejects with
typed values on mismatch. Tool objects themselves are never reconstructed
from the journal; application composition owns Tool construction.

## 5. Goal-bearing Task

Every Root execution begins with one Task committed before the first model request or
Tool side effect:

```text
Task
├── immutable definition
│   ├── task id / optional parent task id
│   ├── objective + success criteria
│   ├── constraints + stable resource/context references
│   └── budget + creation provenance
└── durable lifecycle
    ├── active | blocked | completed | failed | cancelled
    ├── usage
    └── typed blocker or terminal reason
```

One Session has at most one unfinished Root Task. Resume and forks from non-terminal
boundaries retain Task identity; continuing from a terminal Task creates a new Task
explicitly. Blocked is durable and resumable only after explicit caller input or an
observed external-state change. Completed, failed and cancelled commit once.

The Task definition commits as one `task.created` record before `input.accepted` and
the first model or Tool side effect; every lifecycle change commits as one
`task.transition` record. Task state is rebuilt by replaying these records through
the fork lineage; recovery fails closed on a second unfinished Root Task or a
transition after a terminal state.

Benchmark resources, environment probing, evaluation metrics and free-form metadata do
not belong to Task. They remain at recipe/application boundaries and reference Task by
id. The existing Task schema is replaced in place; there is no Goal alias, TaskV2 or
compatibility mirror.

## 6. Plan and Subagent

Plan is an optional progress checklist. Each `PlanItem` carries one concise `step`
description and a `pending | in_progress | completed` display status. A model may add,
remove, rewrite or reorder items; QitOS validates the bounded shape and at most one item
in progress. Every accepted replacement commits as one `plan.updated` record with its
owning Task id in the Run journal. Plans are replayed per Task through the fork lineage,
so an explicit terminal follow-up Task does not inherit the previous Task's strategy.
TODO Markdown is a deterministic projection of the committed Plan, never an editable
second truth.

Plan does not express dependencies, readiness, Subagent ownership or assignment. It is
not a scheduler or Task completion gate. Root and Subagent use the same optional
contract, while Parent and Subagent Plans remain independent and are never merged.

Root and Subagent use the same Agent implementation. A Subagent has its own Task, Session,
optional Plan, context and cancellation domain; authorization and budgets only narrow.
Root and every descendant reserve each model step through one durable, async-safe
lineage ledger before the provider request. A Subagent's `max_steps` is only a local cap
within the shared remaining total. Concurrent admission is serialized, and deterministic
reservation ids make resume/fork replay idempotent without resetting or double-charging
the lineage.

Each active Subagent also owns one process-local claim on a shared step for its final
natural-language answer. Work-step admission counts all live claims, so concurrent
Subagents cannot consume one another's conclusion capacity. At the local/shared step
boundary, or after settled usage crosses a token/cost limit, the Subagent receives one
same-context follow-up turn with an empty Tool exposure. If it already returned useful
final text, its unused claim is released immediately. The model's final assistant text
is the canonical conclusion: a typed conclusion projector may attach stable evidence,
resource, failure, unknown and next-step references, but cannot author or replace the
summary. Text attached to a Tool call is progress commentary, not a terminal answer.
Only text that crossed the durable model-terminal barrier can survive deadline or
cancellation settlement, so live results and recovery choose the same conclusion.
An empty final turn remains an explicit missing conclusion; the Supervisor
does not synthesize prose from Tool results, Plan state or Journal records. Claims are
not persisted because a crash interrupts the live Agent; only provider admissions and
settled usage remain durable. Explicit cancellation and an expired absolute deadline
remain hard boundaries.
Subagent launch is independent of Plan state. `SubagentLaunchRequest` carries explicit success criteria,
inherited Task constraints/references and the frozen parent Permission context; the
built-in factory rejects conflicting Permission values. Independent Agent Tool calls
may execute concurrently through the bounded, concurrency-safe Tool path while results
remain in input order. A typed conclusion factory runs before Subagent cleanup and can
project committed application facts into evidence/resource refs, failure paths,
unknowns and next steps while preserving the model-authored summary. Parent control uses
stable handles and bounded conclusions, never live Agents or Subagent transcripts.

## 7. Tool, Runtime and extension boundary

Tool definition keeps strict input schema and handler together. Registration, exposure,
admission, execution, persistence and model projection are distinct responsibilities
without requiring a class for each one. One frozen exposure drives both model schema
and dispatch for a turn.

QitOS owns reusable file, Shell/PTY, managed process, Artifact, Skill, MCP, Permission,
Plan and Subagent primitives. Product-specific drivers, pentest semantics and product
plugin systems stay in applications. QitOS does not provide a general product plugin
loader or arbitrary loop/Session hooks.

## 8. Cross-cutting invariants

- One prompt enters one Model/loop/Session path.
- A turn uses one immutable model, history, Tool exposure, capability, deadline and
  budget snapshot.
- No model, Tool, user input or cleanup waits while a Session lock or storage
  transaction is held.
- Admitted ToolCall and ToolResult remain paired across compact, resume, fork and
  interruption; ambiguous duplicate ids never reach Tool admission.
- Provider continuation, SQLite index, trace and projections are never recovery truth.
- Cancellation is re-raised after started work reaches durable terminal records when
  canonical appends settle; an append failure or unknown outcome instead requires
  close-and-replay recovery and never permits a Run terminal across an open Tool call.
- Runtime and Subagent concurrency are bounded and leave no detached tasks.
- Concurrent Root/Subagent model requests cannot oversubscribe the durable lineage step
  budget; token/cost totals settle exactly once after each model terminal.
- The outermost CLI/application is the only sync-to-async boundary.
