# QitOS architecture

## Status

The minimal async Agent loop and `Agent` façade are the only execution path. The
retired `AgentModule + Engine` lifecycle and its `Observation -> Decision -> Action`
model are removed. Authoritative Session/Harness and Task/Plan work remains in the
accepted migration below; no parallel V2 runtime or compatibility façade will be
published.

Detailed target contracts live in
[`docs/internal/architecture/agent-runtime.md`](docs/internal/architecture/agent-runtime.md).
Migration order and deletion boundaries live in
[`docs/internal/plans/pi-aligned-agent-runtime.md`](docs/internal/plans/pi-aligned-agent-runtime.md).

## Runtime narrative

```text
Provider APIs
└── Model / Provider
    └── provider-neutral Message / ModelEvent / Usage / continuation
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

The loop stays small; durable control and recovery belong to Session/Harness; product
policy belongs to applications. The current loop preserves Tool transaction, deadline,
cancellation and result-ordering behavior. Session/Harness recovery is the next
migration boundary. The old `Observation → Decision → Action → reduce`
hierarchy is not a compatibility path.

## Ownership

### Stable contracts

`qitos.core` owns provider-neutral cross-module values such as Message, Task, Tool and
Session records. Contracts remain small, typed and independent of a particular Provider,
benchmark or product.

### Provider adapters

`qitos.models` owns concrete Provider transports, model capabilities, reasoning/usage,
stream events and continuation. SDK objects stop at the adapter edge.

### Agent loop

The loop owns immutable model request construction, streaming, ToolCall validation and
execution, ordered ToolResult insertion, safe-point input and stop evaluation. It does
not load products, probe environments, evaluate benchmarks or own persistence.

### Target Session/Harness

Session/Harness is the accepted owner for queue, abort/idle, canonical transcript and
operations, compact, resume, fork and pure recovery. The current façade owns in-memory
queue/run control and can journal loop transactions; authoritative recovery remains a
migration gate. Trace and UI projections are never recovery truth.

### Reusable concrete capabilities

`qitos.kit` owns reusable storage, Runtime, file/Shell/managed-process Tools, Skill,
MCP, Artifact, Permission, Plan and Child implementations. Concrete capability does not
automatically become a stable core interface.

### Applications and benchmarks

Application composition owns prompts, completion policy, domain state and product
plugins. Benchmark execution adapters and recipes are not part of the QitOS runtime;
result-file evaluation and application-owned benchmark integration stay outside the
Agent loop, Task, Session and Tool executor. Product-grade agents live outside the
QitOS kernel repository.

## Target Task, Plan and Child

The remaining Task/Plan migration will make every Root execution commit one
goal-bearing Task before external side effects. Task owns objective, success criteria,
constraints, budget and lifecycle. Plan is a revisable dependency graph, never a second
Task/Goal truth.

Root and Child use the same Agent implementation. A Child has an independent Task,
Session, Plan and cancellation domain while authorization and budget only narrow. The
parent stores a stable handle and bounded conclusion, not a live Agent or transcript.
Every descendant reserves model steps from one Root-lineage ledger before provider
admission; a Child's local step budget is a cap within that shared total, not an
additional allowance.

## Repository invariants

- One public runtime architecture; no V2, legacy alias, wrapper or mirror DTO.
- One immutable turn snapshot for model, history, Tool exposure, runtime, deadline and
  budget.
- Exactly one terminal ToolResult for every uniquely identified call that reaches Tool
  admission; duplicate raw ids fail before admission.
- No external wait while holding Session locks or storage transactions.
- Provider continuation, index, trace and rendering are not recovery truth.
- Imports do not start threads/tasks, connect services or mutate global registries.
- Read-only qita/trace consumers never become a second replay or Session owner.
- Superseded callers, exports, tests, examples, dependencies and docs are deleted with
  each completed migration slice.

## Repository map

- `qitos/`: framework implementation
- `tests/`: behavior, contract, recovery and lifecycle tests
- `examples/`: small runnable reference paths
- `docs/`: shipped user documentation
- `docs/internal/architecture/`: accepted target contracts
- `docs/internal/plans/`: active migration plans only; Git history is the archive
