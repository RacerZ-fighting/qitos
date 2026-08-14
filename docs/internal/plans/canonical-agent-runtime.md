# Canonical agent runtime refactor

## Goal

Make the existing `AgentModule + Engine` pipeline a predictable native tool-using
runtime without adding a parallel Engine. The model-facing loop must have one action
protocol, a transaction-safe transcript, bounded reliability behavior, and explicit
contracts for context, reasoning, child Agents, and tools.

## Constraints

- Preserve the single `AgentModule + Engine` public mental model.
- Keep stable contracts in `qitos.core`, loop mechanics in `qitos.engine`, and concrete
  policies or built-ins in `qitos.kit`.
- Native provider calls are authoritative when enabled. Text parsers remain opt-in
  compatibility behavior, not an automatic second protocol.
- Preserve trace and qita auditability.
- Do not move benchmark-specific or PentestAgent-specific state into the Engine.
- Add observable behavior tests before changing each contract.
- Use QitOS's own uv-managed Python 3.11 dev environment and quality gates.

## Milestones

### 1. Conformance baseline and protocol normalization

- [x] Add deterministic tests for exact provider options, native call/result parity,
  final text, parser-only agents, cancellation placeholders, and parallel ordering.
- [x] Stop injecting a text output contract when API-native tool delivery is selected.
- [x] Skip parser interpretation for completed native-mode text responses; keep parser
  fallback only when explicitly configured as the action protocol.
- [x] Remove PentestAgent-specific/benchmark-specific salvage from the generic native
  request path after compatibility tests demonstrate it is unnecessary.

Done when a native request advertises tools once, accepts only provider-native calls as
actions, and retains exactly one result per admitted call id.

Progress (2026-08-08): provider-native calls now bypass the agent text interpreter;
valid, policy-blocked, loop-blocked, and malformed calls share one ordered result-commit
path. Malformed calls retain their call id and return a recoverable error without tool
execution. Parser-only final-text compatibility remains for a later explicit-lane cut.

### 2. Deadlines, cancellation, concurrency, and retries

- [x] Propagate absolute deadline and cancellation accessors through Engine and tool
  runtime context.
- [x] Clamp tool admission, tool timeout, runtime wait, and retry backoff to remaining
  time.
- [x] Make `ToolSpec.retry_policy` the sole tool retry owner and share one absolute
  action deadline across admission, attempts, backoff, and concurrent drain.
- [x] Clamp model attempts and provider transport timeouts to remaining time.
- [x] Preserve action status and attempts through ToolResult, history, hooks, and trace.
- [x] Make async-native stream shutdown request Engine cancellation and guarantee a terminal
  stream event even when the bounded event queue is full.
- [x] Audit OpenAI model adapters so the provider transport has one retry owner.
- [x] Preserve provider finish reasons and typed reasoning/tool-call stream events
  through the canonical model chunk and Engine response path.

Done when deterministic tests cover cancellation before admission, queued actions,
running actions, partial streams, provider retry, tool retry, and queue saturation.

Progress (2026-08-12): Engine runs now resolve relative and absolute limits into one
effective monotonic deadline. Tools receive live deadline/cancellation accessors; tool
admission, timeout, retry backoff, and runtime wait all use the same remaining-time
calculation. `Engine.arun()` and `Engine.astep()` are the canonical execution paths;
early stream close requests cooperative cancellation, duplicated stream cleanup is
removed, and full event queues retain a terminal marker. End-to-end child/process cleanup
remains a separate follow-up.

Progress (2026-08-08): one fail-closed `ToolResult` classifier now preserves executor
timeout, cancellation, denial, input/approval, partial, and background states through
Engine records, observations, history, events, summaries, and success metrics. Legacy
status aliases and flattened result fields were removed; domain outcomes no longer
occupy the execution-status field in maintained built-ins.

Progress (2026-08-08): action-level retry/timeout knobs and the duplicate interceptor
middleware were removed. Admission runs once, explicit invocation retries share the
tool/runtime deadline, HTTP transport retries stay disabled, and bounded daemon action
workers detach blocked admission or concurrent work instead of waiting indefinitely.

Progress (2026-08-12): one Engine-scoped model-request deadline now governs provider
connection, stream-idle, and retry waits plus immediate cancellation. Official providers
implement one asynchronous stream contract with SDK retries disabled where QitOS owns
the budget; streams close on every exit path, and late output cannot update callbacks,
history, actions, or final state.

Progress (2026-08-12): Chat, Responses, and Anthropic streams now retain their real
finish reason, reasoning/tool-call deltas, completed calls, and usage through
discriminated `ModelStreamEvent` values. Providers reuse the canonical accumulator,
incomplete streams no longer fabricate completion, and rich handlers receive optional
normalized event/error callbacks without changing the required stream-handler protocol.

### 3. Context, compression, and reasoning

- [x] Define complete transcript transactions and make all history policies compact or
  trim only at their boundaries.
- [x] Preserve opaque provider continuation/reasoning items needed by the next request.
- [x] Bound tool projections and record truncation/compaction metadata without rewriting
  retained messages.
- [x] Resolve reasoning effort through one provider capability path for all streaming
  requests.

Progress (2026-08-08): provider presets resolve one reasoning request default for all
OpenAI-compatible invocation paths. GPT-5.6 keeps `max`; forced Chat tool calls remove
conflicting controls. Official Responses requests retain encrypted item fields across
stream completion and stateless replay while summaries omit the opaque payload.

Done when full and compacted runs produce equivalent tool/final behavior in a scripted
model fixture, and provider request tests cover every supported reasoning level.

### 4. Child Agent lifecycle

- [x] Specify parent/child identity, depth, budgets, permission inheritance, workspace
  isolation, cancellation propagation, and terminal delivery.
- [x] Bound concurrent children and guarantee all children are reaped on parent stop.
- [x] Keep child model history and reasoning private while returning bounded tool-backed
  evidence and usage metadata.
- [x] Keep spawn/delegation attached to the existing Engine action lifecycle.

Done when tests cover concurrent children, recursive-depth rejection, parent
cancellation, forced conclusion, partial child evidence, and no stale writes.

### 5. Tool contract and built-ins

- [x] Validate model arguments before tool invocation and preserve structured validation
  failures as non-retryable terminal results.
- [x] Define truthful status projection for success, error, skipped/denied, timed out,
  cancelled, partial, and background/running results.
- [x] Keep one exact active registry selected by application role policy; do not add a
  second hidden/deferred exposure state to the generic registry.
- [x] Replace blanket parallel booleans with explicit `concurrency_safe` declarations
  and exclusive barriers for tools that share mutable resources.
- [x] Bound nested outputs and retain truncation/artifact metadata.
- [x] Audit built-in coding, shell, web, MCP, and Agent tools against the contract.

Progress (2026-08-08): the truthful lifecycle projection is complete, including
unknown-status rejection and removal of the older executor/runtime status guesses.
Schema validation, exposure, keyed concurrency, nested output bounds, and the remaining
built-in audit stay as independent slices.

Done when schema, status, environment, concurrency, output-bound, and exposure tests pass
for every maintained built-in tool family.

### 6. PentestAgent migration and evaluation

- [x] Centralize parent and child Engine construction in the PentestAgent adapter.
- [x] Pass Worker deadlines and canonical runtime metadata into QitOS.
- [x] Make role profiles select the exact advertised tool surface.
- [x] Remove redundant protocol wiring and text-tool salvage after QitOS conformance is
  proven, while retaining branch-result compatibility.
- [x] Add a deterministic same-spec A/B harness and semantic trace-closure checks before
  running optional live benchmark slices.

Progress (2026-08-08): qita now derives tool statistics from its canonical per-call
action/result pairing. Exact lifecycle counts and unmatched action/result evidence make
trace-closure gaps visible without assigning a step-level failure to unrelated calls.

Progress (2026-08-09): qita compares only stable configuration fields, records the
effective step/runtime/token budget without persisting process-local deadline values,
and rejects mismatched or incomplete provenance before outcome deltas can be read as a
same-spec repeat. A scripted model test executes the same traced run twice and verifies
the comparison contract; separate tests reject parser drift and missing prompt identity.

Progress (2026-08-09): PentestAgent removed its final-answer text-tool salvage path and
its legacy-only tests. Worker and child traces now attach the application version,
parent repository revision, clean-source state, capability, and scope; qita rejects
dirty or incomplete application provenance. Role policy constructs one exact active
registry instead of adding a second hidden/deferred exposure mechanism.

Done when PentestAgent keeps its Planner/Facts/AuthSession/Artifact/Shell boundaries,
passes `make check`, and produces a closed QitOS transcript with no parser salvage on a
native-capable Worker.

## Verification

After each QitOS slice, run focused tests first, then the independent QitOS gates:

```bash
uv run --no-project --python 3.11 --with-editable . --with 'pytest>=7' pytest -q
uv run --no-project --python 3.11 --with-editable . --with 'flake8>=6' flake8 qitos/core qitos/engine qitos/models qitos/trace
uv run --no-project --python 3.11 --with-editable . --with 'mypy>=1' mypy qitos/core qitos/engine qitos/models qitos/trace
```

PentestAgent uses its own `make check`; it never substitutes for the commands above.

## Known baseline issues

- Trace replay is a read-only artifact viewer, not deterministic Engine replay.
  Executable forks use the Session Journal. Conformance tests use scripted models and
  tools rather than claiming live provider reproducibility.
