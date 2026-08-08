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

- [ ] Add deterministic tests for exact provider options, native call/result parity,
  final text, parser-only agents, cancellation placeholders, and parallel ordering.
- [x] Stop injecting a text output contract when API-native tool delivery is selected.
- [ ] Skip parser interpretation for completed native-mode text responses; keep parser
  fallback only when explicitly configured as the action protocol.
- [ ] Remove PentestAgent-specific/benchmark-specific salvage from the generic native
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
- [ ] Clamp model attempts and provider transport timeouts to remaining time.
- [x] Preserve action status and attempts through ToolResult, history, hooks, and trace.
- [x] Make async stream shutdown request Engine cancellation and guarantee a terminal
  stream event even when the bounded event queue is full.
- [ ] Audit legacy model adapters so the provider transport has one retry owner.

Done when deterministic tests cover cancellation before admission, queued actions,
running actions, partial streams, provider retry, tool retry, and queue saturation.

Progress (2026-08-08): Engine runs now resolve relative and absolute limits into one
effective monotonic deadline. Tools receive live deadline/cancellation accessors; tool
admission, timeout, retry backoff, and runtime wait all use the same remaining-time
calculation. Async Engine execution no longer relies on asyncio's non-daemon default
executor, duplicated stream cleanup is removed, early stream close requests cooperative
cancellation, and full event queues retain a terminal marker. Provider-call clamping and
end-to-end child cleanup remain open.

Progress (2026-08-08): one fail-closed `ToolResult` classifier now preserves executor
timeout, cancellation, denial, input/approval, partial, and background states through
Engine records, observations, history, events, summaries, and success metrics. Legacy
status aliases and flattened result fields were removed; domain outcomes no longer
occupy the execution-status field in maintained built-ins.

### 3. Context, compression, and reasoning

- [ ] Define complete transcript transactions and make all history policies compact or
  trim only at their boundaries.
- [x] Preserve opaque provider continuation/reasoning items needed by the next request.
- [ ] Bound tool projections and record truncation/compaction metadata without rewriting
  retained messages.
- [x] Resolve reasoning effort through one provider capability path for sync, async, and
  streaming requests.

Progress (2026-08-08): provider presets resolve one reasoning request default for all
OpenAI-compatible invocation paths. GPT-5.6 keeps `max`; forced Chat tool calls remove
conflicting controls. Official Responses requests retain encrypted item fields across
stream completion and stateless replay while summaries omit the opaque payload.

Done when full and compacted runs produce equivalent tool/final behavior in a scripted
model fixture, and provider request tests cover every supported reasoning level.

### 4. Child Agent lifecycle

- [ ] Specify parent/child identity, depth, budgets, permission inheritance, workspace
  isolation, cancellation propagation, and terminal delivery.
- [ ] Bound concurrent children and guarantee all children are reaped on parent stop.
- [ ] Keep child model history and reasoning private while returning bounded tool-backed
  evidence and usage metadata.
- [ ] Keep spawn/delegation attached to the existing Engine action lifecycle.

Done when tests cover concurrent children, recursive-depth rejection, parent
cancellation, forced conclusion, partial child evidence, and no stale writes.

### 5. Tool contract and built-ins

- [ ] Validate model arguments before tool invocation and preserve structured validation
  failures as non-retryable terminal results.
- [x] Define truthful status projection for success, error, skipped/denied, timed out,
  cancelled, partial, and background/running results.
- [ ] Separate registered tools from direct/deferred/hidden model exposure.
- [ ] Replace blanket parallel booleans with safe defaults and explicit keyed/exclusive
  policy where shared resources require it.
- [ ] Bound nested outputs and retain truncation/artifact metadata.
- [ ] Audit built-in coding, shell, web, MCP, and Agent tools against the contract.

Progress (2026-08-08): the truthful lifecycle projection is complete, including
unknown-status rejection and removal of the older executor/runtime status guesses.
Schema validation, exposure, keyed concurrency, nested output bounds, and the remaining
built-in audit stay as independent slices.

Done when schema, status, environment, concurrency, output-bound, and exposure tests pass
for every maintained built-in tool family.

### 6. PentestAgent migration and evaluation

- [ ] Centralize parent and child Engine construction in the PentestAgent adapter.
- [ ] Pass Worker deadlines and canonical runtime metadata into QitOS.
- [ ] Make role profiles select the exact advertised tool surface.
- [ ] Remove redundant protocol wiring and text-tool salvage after QitOS conformance is
  proven, while retaining branch-result compatibility.
- [ ] Add a deterministic same-spec A/B harness and semantic trace-closure checks before
  running optional live benchmark slices.

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

- Full QitOS test collection in its independent uv environment currently stops on
  existing `qitos_zoo`/PentAGI import failures (`AuditAgent` and `pentagi`). The focused
  native-runtime surface is therefore the acceptance baseline until those unrelated
  collection failures are repaired.
- Existing trace replay is an artifact viewer/fork helper, not deterministic Engine
  replay. Initial conformance tests use scripted models and tools rather than claiming
  live provider reproducibility.
