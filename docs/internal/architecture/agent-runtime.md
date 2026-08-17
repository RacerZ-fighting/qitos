# QitOS Agent runtime architecture

## Status

Accepted target architecture on 2026-08-16. The minimal loop and `Agent` façade are
the only execution path; authoritative Session/Harness and Task/Plan work remains in
the migration plan.

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
└── Tool / Runtime / Skill / MCP / Plan / Child primitives
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

The loop receives a transaction boundary that can record model terminal, Tool admission,
Tool terminal and turn commit around side effects. Every uniquely identified call that
reaches Tool admission, including a per-call rejection, receives exactly one terminal
ToolResult. Duplicate raw call ids remain assistant protocol-failure evidence and the
ambiguous batch is rejected before Tool admission or side effects.

## 4. Agent façade and Session Harness

The façade presents context, event subscription and the current Session/Harness. The
Harness owns prompt, queue, abort, idle, compact, resume and fork operations. Expected
rejections use typed results; corruption and implementation faults use exceptions.

Session keeps one canonical log with typed transcript entries and operation records.
Memory and JSONL implementations share conformance behavior; a SQLite index is
disposable. Pure recovery reconstructs open operations, incomplete Tools, queue, Task
lifecycle and terminal outcomes, and fails closed on contradictions.

Authoritative snapshots come from Session state. Progress events, trace files and UI
projections are observational and cannot mutate recovery truth.

## 5. Goal-bearing Task

Every Root execution begins with one Task committed before the first model request or
Tool side effect:

```text
Task
├── immutable definition
│   ├── task id / optional parent task id
│   ├── objective + success criteria
│   ├── constraints + stable resource/context references
│   ├── budget + creation provenance
│   └── optional parent Plan assignment reference
└── durable lifecycle
    ├── active | blocked | completed | failed | cancelled
    ├── usage
    └── typed blocker or terminal reason
```

One Session has at most one unfinished Root Task. Resume and forks from non-terminal
boundaries retain Task identity; continuing from a terminal Task creates a new Task
explicitly. Blocked is durable and resumable only after explicit caller input or an
observed external-state change. Completed, failed and cancelled commit once.

Benchmark resources, environment probing, evaluation metrics and free-form metadata do
not belong to Task. They remain at recipe/application boundaries and reference Task by
id. The existing Task schema is replaced in place; there is no Goal alias, TaskV2 or
compatibility mirror.

## 6. Plan and Child

Plan is a dependency-aware graph with stable node identity, owner and explicit state.
Readiness derives from dependency completion. Multiple owners may hold independent
nodes in progress. QitOS validates graph, transition, reservation, budget and
concurrency; applications or models choose the schedule.

Root and Child use the same Agent implementation. A Child has its own Task, Session,
Plan, context and cancellation domain; authorization and budgets only narrow. Launch
creates the Child Task and durable parent Plan assignment before runtime construction.
Parent control uses stable handles and bounded conclusions, never live Agents or Child
transcripts.

## 7. Tool, Runtime and extension boundary

Tool definition keeps strict input schema and handler together. Registration, exposure,
admission, execution, persistence and model projection are distinct responsibilities
without requiring a class for each one. One frozen exposure drives both model schema
and dispatch for a turn.

QitOS owns reusable file, Shell/PTY, managed process, Artifact, Skill, MCP, Permission,
Plan and Child primitives. Product-specific drivers, pentest semantics and product
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
- Cancellation is re-raised only after started work reaches durable terminal records.
- Runtime and Child concurrency are bounded and leave no detached tasks.
- The outermost CLI/application is the only sync-to-async boundary.
