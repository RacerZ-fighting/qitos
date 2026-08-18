# Pi-aligned Agent runtime migration

## Status

Accepted migration plan on 2026-08-16. The target contracts are defined by the
[QitOS Agent runtime architecture](../architecture/agent-runtime.md). The minimal loop,
façade, retired-runtime deletion slices, S1 parity items, the authoritative
Session/Harness (S2), goal-bearing Task (S3a), and dependency-aware Plan (S3b) are
implemented on the migration branch. The branch has passed its independent pytest,
flake8 and mypy gates for S1-S3. Cross-repository integration hardening has since added
Subagent product hooks, authorization/task binding, committed transaction queries and a
trace serialization fix; the branch has passed a fresh full pytest, flake8 and mypy gate
and now awaits review and merge.

## 1. Current gaps

- QitOS runtime migration S1-S3 and the QitOS side of GeneralAgent integration
  hardening are complete on the feature branch. PentestAgent's product composition is
  implemented and independently verified on its own feature branch; QitOS mainline merge
  and gitlink movement remain separate cross-repository steps.

Proven Tool transaction, cancellation, absolute deadline, result ordering, trace and
recovery behavior is the conformance baseline. Legacy type names and package layout are
not compatibility requirements.

## 2. Milestones

### 2.1 Freeze behavioral conformance

Foundation implemented on `feat/pi-aligned-agent-loop` (commits `af8f8ef`,
`306d3bc`); S1 lands the remaining parity items. Typed
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
- Pi's open-ended custom `AgentMessage` + `convertToLlm` extension point is
  excluded at the QitOS canonical transcript boundary. The transcript stays a
  closed typed set (`UserMessage`/`ContextMessage`/`AssistantMessage`/
  `ToolResultMessage`) with a fail-closed codec. The narrow `ContextMessage`
  addition follows Codex's durable developer-context behavior: products own
  their state and delta, while QitOS only preserves the model-visible history
  projection before sampling.
- `before_tool_call` keeps Pi's `block` / `reason` / `terminate` surface. Tool
  argument rewriting remains owned by the existing permission pipeline; the
  hook does not introduce a second mutation channel.
- Cancellation closes the loop: external task cancellation and ordinary faults
  terminalize started work plus the run record before re-raising when canonical
  appends settle; append failure or uncertainty stops before an open Tool boundary,
  which close-and-replay recovery closes in input order without re-execution;
  `CancelToken` `after_step` stops at turn boundaries and never interrupts
  in-flight streams or Tool calls; duplicate Tool-call ids remain in the
  assistant transcript as protocol-failure evidence and the malformed batch
  is rejected before Tool admission; terminal `ToolResult`s are deeply immutable before they
  cross journal, event and Message boundaries.
- Run terminalization now has one frozen, optional async resource-quiescence
  finalizer. The Run owner awaits it exactly once on success, failure, deadline
  and cancellation before appending the terminal record; cleanup faults are a
  bounded typed diagnostic and never replace the primary status/error. It receives
  only the current leg's stable Run id, quiesces that Run's resources, and cannot
  close Env, Tool or MCP owners reused by later legs; the outer application owns
  their exactly-once teardown. This is a narrow lifecycle barrier, not a Hook list
  or extension bus. Parallel Tool
  handlers still overlap, while every Tool terminal transaction and end event
  commits deterministically in assistant-call input order.

S1 closes the remaining parity items:

- provider-neutral typed thinking level (`off|minimal|low|medium|high|xhigh|max`,
  Pi's exact values) owned by the Model boundary: a typed `ModelRequest` field,
  adapter translation keyed by declared model capability with Pi's
  nearest-up-then-down clamping, and a per-turn `NextTurnUpdate` override;
- typed `ToolResult.usage` (`ModelUsage`) and `ToolResult.added_tool_names`,
  carried onto `ToolResultMessage` and through the durable codecs; the Agent
  (Subagent) Tool moves its Subagent token/cost accounting out of the untyped output
  payload into these fields;
- restore of initial façade messages and Tool activation is part of S2, where
  the authoritative Session/Harness path exists.

### 2.2 Replace loop and façade

Done on `feat/pi-aligned-agent-loop`. The minimal loop and small `Agent`
façade are the only execution path: C2 (commit `e28b23e`) moved subagents
onto the façade and removed the delegate/fanout tools, kit handoff tool
chain, shared memory and repl; C3 (commit `9ad277e`) removed the zoo
submodule, demo, recipes, benchmark execution adapters and the Engine-era
examples; C4 (commit `88da5bf`) deletes `qitos/engine/`, the `core` old-lifecycle
modules, checkpoint, protocols, render, and the Engine-era kit
parser/critic/planning/prompts/history packages. No composition subclasses
AgentModule, and the old Engine is absent from exports, examples and tests.

### 2.3 Make Session/Harness authoritative (S2)

Done on `feat/pi-aligned-agent-loop`. Scope per the architecture document §4;
concretely:

- (done in S2a) Transcript entries (`transcript.message`, `compaction`) own
  message content; operation records reference them by record id.
  `model.completed` keeps the exact request audit plus the assistant
  transcript reference; `tool.terminal` keeps the call plus the terminal
  transcript reference; `step.committed` is a pure commit marker of record
  references; `run.completed`/`run.interrupted` no longer embed messages.
- (done in S2a) `input.accepted` gains a writer and commits, with the initial
  prompt transcript entries, before the first model side effect.
- A later hardening slice adds `turn_input.committed`: prompt, steering and
  typed runtime-context entries are durable in their model-visible order before
  each sampling request. Normal live turns use the in-memory Agent context;
  full Journal replay remains a start/resume/fork operation.
- (done in S2a) Pure recovery (`qitos.kit.journal.recovery.recover_session`)
  replays the log into transcript, configuration lineage, open Tool
  operations, unconsumed runtime inputs and terminal outcome; crash-torn Tool
  admissions close with explicit cancelled terminals
  (`close_crashed_tool_calls`); contradictions fail closed.
- (done in S2a) Memory and JSONL journal implementations share one
  conformance suite (`InMemorySessionJournal`,
  `tests/journal/test_session_journal_conformance.py`).
- (done in S2b) Resume/fork restore the façade (transcript, thinking level,
  configuration lineage) and verify the provided Model identity and Tool
  registry coverage with typed rejections; the run's turn counter continues
  from recovery via the loop's `turn_base` and the recovered recorder seed.
  `qitos.kit.session.SessionHarness` owns start/resume/fork and the
  `SessionRun` leg machinery (one journal per Agent run; terminal
  continuations advance along explicit forks).
- (done in S2b) Compaction: manual at idle, automatic token-threshold at
  idle boundaries, and one-shot overflow recovery; Pi's cut-point rule
  (never between a Tool call and its result), split-turn prefix merge and
  summary-as-user-message projection land in
  `qitos.kit.session.compaction`, and the context swap seals Provider
  continuation through `Agent.set_transcript` plus the loop's
  `continuation_floor`.
- (done in S2b) Unconsumed background Subagent completion inputs re-project
  from own-run posted facts exactly once (the default root
  `post_runtime_event` and `AgentSubagentEngine` both append
  `runtime_input.consumed` once the steered message is covered by a
  `step.committed`); inherited fork facts and foreground results are never
  redelivered.
- (done in S2b) The trace writer is reattached to the loop/façade event
  stream via `qitos.trace.AgentTraceProducer`; new runs emit the three-file
  layout again and `qita` reads them unchanged.
- (done in S2a) The run catalog reads the new payload shapes; Engine-era
  payload readers (`stop_reason`, `reason`, `task`, terminal flags, legacy
  usage fallback) are removed in favor of fail-closed decoders.
- (done in S2b) Recovery replays nested fork prefixes as a sequence of
  closed per-run segments: turn barriers and Tool-call pairing are scoped
  by owning run, since call ids are unique only within one run.

Crash recovery is demonstrated at the model terminal, tool started, tool
terminal and state commit boundaries, and recovery never branches through a
live Engine or guesses side effects.

### 2.4 Replace Task and Plan (S3)

S3a (Task), done on `feat/pi-aligned-agent-loop`:

- (done) Replaced `core/task.py` in place with the goal-bearing Task of
  architecture §5: immutable definition (task id, optional parent task id,
  objective, success criteria, string constraints, stable typed references,
  budget, creation provenance, optional parent Plan assignment reference)
  plus durable lifecycle (`active|blocked|completed|failed|cancelled`,
  usage, typed blocker/terminal reason) committed as `task.created` /
  `task.transition`.
- (done) Removed `TaskResource`, `TaskResult`, `TaskCriterionResult`,
  `TaskResourceBinding`, `TaskValidationIssue`, environment probing
  (`resolve_resources`, `validate_structured`) and free-form metadata from
  the canonical Task; migrated `RepoEnv` (Task coupling and
  `required_missing` probing removed), the evaluation context/DSL exposure
  (new `task.to_dict()` shape) and the Subagent budget consumers.
- (done) Root Task commits before `input.accepted`; one unfinished Root Task
  per Session lineage; blocked/terminal semantics per architecture §5
  (`unblock_task` is the only blocked → active path; run termination never
  auto-transitions the Task; `start_follow_up` starts a new Task explicitly
  on a terminal-task lineage); Subagent launch commits the narrowed Subagent Task
  into the Subagent journal before its `input.accepted`. Remainder for S3b:
  binding `plan_assignment` to a real parent Plan node and parent-side
  narrowing enforcement of the Subagent Task.

S3b (Plan), done on `feat/pi-aligned-agent-loop`:

- (done) Replaced `core/work_plan.py` in place with the dependency-aware graph:
  stable node ids, dependencies, owners, explicit state, derived readiness,
  per-owner in-progress limits, cycle/reference/transition validation.
- (done) Kept one optional Plan contract for Root and Subagent: Root normally uses the full
  dependency/owner graph; a simple Subagent may have no Plan or a flat graph whose
  deterministic projection is a TODO. Parent and Subagent Plans never merge.
- (done) Every accepted update commits one Task-bound `plan.updated` record; recovery
  replays each Task's latest committed update through the lineage, so terminal
  follow-up starts without the previous Task's Plan; TODO Markdown is the deterministic
  topological projection.
- (done) Reworked the `update_plan` tool onto the graph contract and bound Subagent launch
  to a parent Plan assignment.

Done when Task, Session, Run, Plan and Subagent identities stay distinct across
recovery, and no TaskV2/Goal mirror or compatibility Plan remains.

### 2.5 Harden application composition boundary

Done on the two feature branches, pending final gates and cross-repository landing:

- PentestAgent composes the façade directly and keeps Engagement, Scope, investigation
  state, plugin system and completion policy outside QitOS.
- Root and Subagent use the same Agent implementation and product hooks. Subagent launch
  carries explicit success criteria, inherited Task constraints/references and frozen
  Permission; the built-in factory persists them on the Subagent Task before model work.
- A typed conclusion factory runs before invocation cleanup, allowing the application to
  project committed evidence, resource, failure, unknown and next-step facts without
  copying a transcript.
- Independent Agent Tool calls use the executor's bounded concurrency-safe path, while a
  single Root-owned limiter remains shared by recursive descendants.
- `committed_tool_transactions` exposes a read-only, fail-closed application fold over
  terminals referenced by `step.committed`; trace serialization handles deeply frozen
  runtime mappings without mutating them.

This slice now has no PentestAgent concepts in QitOS, no QitOS runtime wrapper in
PentestAgent, and both independent gates pass.

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
- planning/workflow runtimes that compete with Task + Plan + Subagent;
- unconsumed environment, memory, parser, ToolSet, renderer and demo surfaces;
- sync bridges below the outermost application/CLI boundary.

Do not keep V2 modules, legacy aliases, wrappers or mirror DTOs.

## 3. Session/Harness dispositions

Decisions taken while designing S1–S3, recorded for review. Pi remains the
primary reference; where QitOS deliberately deviates, the reason is stated.

- D1 — Session = Run journal lineage, not a multi-Run file. Pi v3 appends
  every Run of a session to one file; QitOS keeps its proven per-Run journal
  (writer lease, tail repair, idempotent append) and links Runs by fork:
  every journal embeds its inherited prefix, so each one is self-contained
  recovery truth. Continuing a terminal Run always forks explicitly, which
  keeps the side-effect boundary and the storage boundary identical.
- D2 — Queues stay memory-only (Pi v3 parity). Pi v4's journaled queues live
  in a harness whose runtime is still a stub in the Pi tree; QitOS does not
  adopt an unproven design. Undelivered steering/follow-up messages are not
  recovery truth.
- D3 — Message content lives in exactly one record. Transcript entries carry
  messages; `model.completed`/`tool.terminal`/`step.committed` reference
  record ids instead of embedding copies. The exact model request audit
  (including the message prefix) stays in `model.completed.request`; its
  growth is the known storage concern owned by the application-side
  long-context acceptance item, not by this contract.
- D4 — Compaction follows Pi v3's proven algorithm: token estimation,
  keep-recent cut search, cut never at a Tool result, split-turn handling,
  summary injected as a user message, `firstKeptEntryId`-style reference
  (resolved fail closed through `journal.inherited`). Pi v4's `retainedTail`
  embedding is not adopted; reference resolution is already the journal's
  established mechanism.
- D5 — Configuration entries are written by the transaction boundary on
  per-turn freeze diffs, never by façade setters, so the log cannot diverge
  from the effective configuration. Restore verifies the provided Model
  identity and Tool registry coverage and rejects mismatches with typed
  values.
- D6 — Tool activation restore is lineage plus verification, not object
  reconstruction: `ToolResult.added_tool_names` and `tools.change` make the
  activation order durable; application composition re-registers the Tool
  objects; resume fails closed with the missing names otherwise.
- D7 — Runtime input re-projection scans only records originating in the
  current Run (never `journal.inherited` facts) and treats an input as
  consumed only when a `runtime_input.consumed` record exists; consumption
  commits when the steered message enters the committed transcript.
  Foreground Subagent results are delivered by their ToolResult and are never
  re-projected.
- D8 — Engine-era payload readers are deleted, not tolerated: the run
  catalog, the budget ledger and the transaction decoders fail closed on old
  payload shapes instead of silently skipping them.
- D9 — Thinking level uses Pi's exact seven values and clamping rule.
  Construction-time provider reasoning kwargs remain adapter defaults; the
  typed per-turn field is the only runtime mutation channel.
- D10 — Plan updates remain whole-graph replacements (the model-facing shape
  models already handle), validated as legal transitions; committed updates
  are the Plan truth and TODO Markdown stays a deterministic projection.

## 4. Verification

Each slice needs focused behavior tests, followed by QitOS's independent pytest,
flake8 and mypy checks. The final matrix covers terminal ToolResult pairing,
cancellation, absolute deadline, ordered concurrency, queue safe points, Task/Plan
transitions, compact/resume/fork, corruption, Subagent cleanup and no leaked tasks.

README, public shipped-behavior docs and Changelog change in the same slice. A
PentestAgent quality gate does not validate QitOS.
