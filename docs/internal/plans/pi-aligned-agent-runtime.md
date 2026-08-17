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

- Express provider/message, Tool transaction, cancellation, deadline, ordering and
  recovery tests without legacy class names.
- Resolve the public Message, ToolCall, ToolResult, AgentEvent and failure types.

Done when the suite can run against a replacement loop without importing AgentModule,
Observation, Decision or Action.

Progress on `feat/pi-aligned-agent-loop`: typed messages (`core/message.py`),
loop events (`core/agent_events.py`), `AgentLoopResult`/rejection types, the
minimal loop (`core/agent_loop.py`), the `Agent` façade (`core/agent.py`) and
`ToolBatchExecutor` (`core/tool_executor.py`) landed with conformance tests in
`tests/core/test_agent_{message,loop,facade}.py`, `test_tool_executor.py` and
`tests/journal/test_turn_recorder.py`. Child agents now run on the façade:
`kit/child/agent_engine.py` drives one child run per invocation through `Agent`
with narrowed tools/budgets and a `JournalTurnTransaction` journal, supervisor
recovery rebuilds terminal Child facts from the child's loop journal, and the
Engine-coupled `DelegateTool`/`FanOutTool`, the kit `HandoffTool` chain,
`core/shared_memory.py` and `kit/repl` are removed. The showcase layer is
gone: the `qitos_zoo` submodule, `qitos.demo` and `qit demo`, the whole
`qitos.recipes` package, the recipe-coupled `qitos.benchmark` adapters and
`runner.py` (`qit bench` keeps the engine-free `eval`/`replay`/`export`/
`presets`), and the Engine-era examples/tutorial course; the quickstart
example composes the façade directly. Old-lifecycle callers remain until the
rest of milestone 2.2.

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

- Introduce the minimal loop and small Agent façade as the only execution path.
- Move all application callers in the same slices that delete the corresponding old
  lifecycle; do not publish two runtimes.

Done when application composition no longer subclasses AgentModule and old Engine is
absent from public exports, examples and runtime tests.

### 2.3 Make Session/Harness authoritative

- Separate transcript entries from operation records in one canonical storage contract.
- Move queue, compact, recovery, resume, fork and expected rejection out of Engine.
- Provide memory/JSONL conformance and pure recovery tests.

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
