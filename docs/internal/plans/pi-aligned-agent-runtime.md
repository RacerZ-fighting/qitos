# Pi-aligned Agent runtime migration

## Status

Accepted migration plan on 2026-08-16. The target contracts are defined by the
[QitOS Agent runtime architecture](../architecture/agent-runtime.md). Current releases
still execute `AgentModule + Engine`; a milestone is complete only after callers and
superseded paths are removed.

## 1. Current gaps

- Public execution still centers on `AgentModule`, Observation, Decision, Action and
  Engine rather than Model, minimal loop and Session/Harness.
- Engine still owns application parser/critic/search/handoff, environment setup,
  checkpoint compatibility and benchmark concerns.
- Task mixes objective identity with benchmark resources, environment probing, metrics
  and free-form metadata.
- WorkPlan is a flat single-active checklist rather than an owner/dependency graph.
- Application composition subclasses the old Agent lifecycle, so deleting historical
  runtime surfaces is not yet possible.

Proven Tool transaction, cancellation, absolute deadline, result ordering, trace and
recovery behavior is the conformance baseline. Legacy type names and package layout are
not compatibility requirements.

## 2. Milestones

### 2.1 Freeze behavioral conformance

Done on `feat/pi-aligned-agent-loop` (commits `af8f8ef`, `306d3bc`): typed
messages (`core/message.py`), loop events (`core/agent_events.py`),
`AgentLoopResult`/rejection types, the minimal loop (`core/agent_loop.py`),
the `Agent` façade (`core/agent.py`) and `ToolBatchExecutor`
(`core/tool_executor.py`) with conformance tests in
`tests/core/test_agent_{message,loop,facade}.py`, `test_tool_executor.py` and
`tests/journal/test_turn_recorder.py`.

Review-hardening dispositions recorded on the same branch:

- `ToolExecutionUpdate` is implemented, wired to the Tool `emit_progress`
  runtime callback (Pi parity); earlier notes claiming "no data source" were
  wrong.
- Pi's custom `AgentMessage` + `convertToLlm` extension point is explicitly
  excluded at the QitOS canonical transcript boundary: the transcript stays a
  closed typed set (`UserMessage`/`AssistantMessage`/`ToolResultMessage`)
  with a fail-closed codec; application-specific projections belong in
  journal records, events and metadata, not in LLM-bound messages.
- `before_tool_call` may return `updated_args`; QitOS re-validates them
  against the same input schema and permission before execution instead of
  copying Pi's unvalidated mutation channel.
- Cancellation closes the loop: external task cancellation and faults
  terminalize started work plus the run record before re-raising;
  `CancelToken` `after_step` stops at turn boundaries and never interrupts
  in-flight streams or Tool calls; duplicate Tool-call ids are rejected at
  batch admission; terminal `ToolResult`s are deeply immutable before they
  cross journal, event and Message boundaries.

### 2.2 Replace loop and façade

Done on `feat/pi-aligned-agent-loop`. The minimal loop and small `Agent`
façade are the only execution path: C2 (commit `e28b23e`) moved child agents
onto the façade and removed the delegate/fanout tools, kit handoff tool
chain, shared memory and repl; C3 (commit `9ad277e`) removed the zoo
submodule, demo, recipes, benchmark execution adapters and the Engine-era
examples; C4 (this slice) deletes `qitos/engine/`, the `core` old-lifecycle
modules, checkpoint, protocols, render, and the Engine-era kit
parser/critic/planning/prompts/history packages. No composition subclasses
AgentModule, and the old Engine is absent from exports, examples and tests.

### 2.3 Make Session/Harness authoritative

- Separate transcript entries from operation records in one canonical storage contract.
- Move queue, compact, recovery, resume, fork and expected rejection out of Engine.
- Provide memory/JSONL conformance and pure recovery tests.
- Reattach the trace writer to the loop/façade so new runs emit trace
  artifacts (the Engine-era producer was removed with C4).

Done when recovery never branches through a live Engine or guesses side effects.

### 2.4 Replace Task and Plan

- Commit Root Task before side effects and persist lifecycle/usage.
- Remove benchmark/environment/evaluation/free-metadata fields from canonical Task.
- Replace flat WorkPlan with the dependency-aware owner graph.
- Bind Child Task and parent assignment durably before launch.

Done when Task, Session, Run, Plan and Child identities stay distinct across recovery,
and no TaskV2/Goal mirror or compatibility Plan remains.

### 2.5 Migrate application composition

- Let PentestAgent compose the new façade directly.
- Preserve product-owned Engagement, Scope, domain state, plugin system and completion
  policy outside QitOS.
- Keep Root and Child on the same Agent implementation with narrowing-only authority.

Done when QitOS contains no PentestAgent concepts and PentestAgent contains no QitOS
runtime wrapper.

### 2.6 Remove historical surface

Done in the C2/C3 slices: delegate/fanout kit tools, the kit handoff tool
chain and shared-memory multi-Agent paths; the `qitos_zoo` submodule,
`qitos.demo`, the pattern recipes, the whole `qitos.recipes` package, and the
recipe-coupled benchmark execution layer (`qitos.benchmark` adapter
subpackages and `runner.py`, `qit bench run`/`list`).

Delete final callers, packages, exports, dependencies, tests, examples and docs for:

- AgentModule, Observation, Decision and Action public lifecycle;
- checkpoint as a second persistence owner;
- the Engine-internal handoff policy and its event names;
- Engine critic, search, branch selector, text parser and prompt-injected protocol;
- planning/workflow runtimes that compete with Task + Plan + Child;
- unconsumed environment, memory, parser, ToolSet, renderer and demo surfaces;
- sync bridges below the outermost application/CLI boundary.

Do not keep V2 modules, legacy aliases, wrappers or mirror DTOs.

## 3. Verification

Each slice needs focused behavior tests, followed by QitOS's independent pytest,
flake8 and mypy checks. The final matrix covers terminal ToolResult pairing,
cancellation, absolute deadline, ordered concurrency, queue safe points, Task/Plan
transitions, compact/resume/fork, corruption, Child cleanup and no leaked tasks.

README, public shipped-behavior docs and Changelog change in the same slice. A
PentestAgent quality gate does not validate QitOS.
