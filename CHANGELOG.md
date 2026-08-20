# Changelog

This project keeps a human-curated changelog so users and contributors can see how QitOS evolves over time.

Format:
- `Added`: new features and capabilities
- `Changed`: behavior changes, refactors, and structural improvements
- `Fixed`: bug fixes
- `Deprecated`: old paths or APIs that will be removed later
- `Removed`: deleted features
- `Breaking`: upgrade notes for incompatible changes

How to update:
- Add high-signal entries under `Unreleased` while work is in progress
- Move `Unreleased` notes into a dated or versioned section when publishing a release
- Prefer user-facing changes, upgrade notes, and important engineering changes over low-level edit logs

## Unreleased

### Fixed

- Subagent launch admission now narrows each requested `max_steps` to the
  shared lineage budget's remaining steps instead of oversubscribing it, and a
  new `SubagentSupervisor` `min_remaining_step_reserve` option rejects launches
  once the remaining lineage steps fall to the configured reserve, keeping
  capacity for the parent's own turns.

- `subagent_wait` may now omit `subagent_id` to wait for the next Subagent
  terminal state in the Run through the new event-driven
  `SubagentSupervisor.wait_any`, and its `timeout_seconds` ceiling is raised
  from 60 to 600 seconds, so a parent can take one long wait instead of
  polling in a tight loop.

- Subagent mailboxes now queue parent messages instead of awaiting turn
  admission. `AgentSubagentEngine.apost_runtime_event` accepts a message for
  any active Run — including while a model request or Tool call is in flight
  and between turns — and returns once the message is queued; journal
  persistence and steering-queue delivery still settle at the next
  turn-boundary safe point. Steering a busy Subagent therefore no longer
  times out behind long turns. Messages still queued when a Run terminalizes
  are rejected at the terminal boundary without a `runtime_input.posted`
  record, so acceptance means queued, not guaranteed delivery.

- Live Subagent status projections now carry the run's committed steps, token
  usage and elapsed time: `SubagentSupervisor._current_result` previously left
  them at zero for running Subagents, so a parent's `subagent_status` poll saw
  no observable progress and could mistake healthy work for a stall.

- The `subagent` launch Tool accepts an optional `max_steps` argument so the
  model can size each Subagent's step budget to its front; the effective budget
  always narrows to the smaller of the requested value and the configured
  Subagent budget.

- Cancelled, failed and budget-exhausted Subagent terminal results now keep
  the run's committed step count and elapsed time instead of reporting zeros,
  matching the token usage those paths already reported.

- qita now derives live event and step counts plus the latest update time from
  committed trace JSONL files while a Run is still active. Finalized Runs keep
  using their validated manifest summary.
- Subagent runs attach their own trace producer when a trace directory is
  configured: every Subagent Run writes its events, steps and manifest linked to
  the immediate parent Run id, finalized on completion, failure and interrupt.
- Root and recursively launched Subagents now reserve model steps from one durable
  lineage budget before provider admission. Concurrent Subagents cannot oversubscribe
  the remaining allowance, and each active Subagent holds one slot for a final
  tool-free, same-context answer. A useful answer releases the unused hold; a step
  boundary or settled token/cost crossing consumes it for the conclusion turn. The
  model's final text remains canonical, and typed projections cannot replace it with
  Tool, Plan, or Journal-derived prose. Resume/fork restores only durable usage, not
  process-local holds for Agents interrupted by a crash.
- Subagent runs now receive the same typed pre-sampling context hook as Root runs.
  Deadline, cancellation, and transaction-barrier races preserve only a terminal
  natural-language answer that crossed the durable model boundary; text attached to
  Tool calls remains progress commentary. Live results and Journal recovery therefore
  select the same conclusion without retrying the model past a hard boundary.
- OpenAI-compatible Chat Completions transports that do not claim the official OpenAI
  provider contract now project typed runtime context as tagged user-role content.
  Kimi and similar endpoints therefore keep the canonical non-user `ContextMessage`
  without receiving the unsupported `developer` wire role; official OpenAI Chat and
  Responses continue to use `developer`.
- Bundled Skills whose exact names are already visible may now be loaded directly;
  the model-facing Tool guidance reserves `list_skills` for catalog discovery or
  truncated summaries instead of requiring a redundant listing call.
- Parallel Tool handlers now settle independently but append their canonical
  ToolResult transcript/terminal records and end events strictly in ToolCall
  input order, eliminating concurrent transcript record-id reuse. A failed or
  uncertain terminal append stops the ordered prefix; recovery closes that
  call and later candidates in input order without re-running handlers.
- Runs may compose one frozen async resource finalizer. It is awaited exactly
  once before `run.completed` / `run.interrupted` on normal, fault, deadline
  and cancellation paths; cleanup failures are preserved as bounded typed
  diagnostics without replacing the primary outcome. The finalizer receives
  only the stable Run id and quiesces that Run's resources; reusable Env,
  Tool and MCP owners remain open for later Session legs and are closed once
  by the outer application.

### Breaking

- `Plan` is now a replaceable progress checklist of `PlanItem(step, status)` values,
  matching the model-facing `update_plan` shape. Dependency nodes, readiness, owners,
  transition rules and `plan_assignment` are removed from the current API; Plan no
  longer schedules Subagents or gates Task completion. Existing graph snapshots and
  Task/Subagent payloads with `plan_assignment` have one-way read migration into the
  new projection and are never emitted again.
- Custom `TurnTransactionBoundary` implementations must implement
  `turn_input_committed(turn, messages)`. Exhaustive handlers for the closed
  `Message` union must also handle `ContextMessage`.

### Added

- Added provider-neutral `ContextMessage` and a `turn_input.committed` request
  barrier. Applications can project dynamic state as durable model history
  immediately before sampling without impersonating a user message or
  rebuilding runtime state from the Journal every turn. OpenAI Responses and official
  OpenAI Chat keep the developer role; compatibility Chat, Anthropic and Gemini
  preserve it as explicitly tagged contextual user content. Recovery, compaction, trace, fork and continuation
  validation use the same canonical entries.
- **Minimal agent loop and Agent façade (Pi-aligned).** New
  `qitos.core.agent_loop` (`agent_loop` / `run_agent_loop`), `qitos.core.agent`
  (`Agent`), `qitos.core.message` (typed `UserMessage` / `ContextMessage` /
  `AssistantMessage` / `ToolResultMessage` / `ToolCall` with wire and durable codecs),
  `qitos.core.agent_events`, and `qitos.core.tool_executor`
  (`ToolBatchExecutor`) implement the `Message -> Model -> ToolCall ->
  ToolResult` mainline. The loop freezes one
  model/Tool exposure/deadline snapshot per turn (a live `ToolRegistry` is
  re-frozen per turn, so Tools loaded mid-run become visible to the next
  turn), injects steering before each model request, revives on follow-up
  messages, fails truncated Tool-call batches without execution, and records
  model/Tool/turn/run barriers through a `TurnTransactionBoundary`;
  `qitos.kit.journal.JournalTurnTransaction` persists them into the canonical
  JSONL journal, including committed-Tool-transaction query linkage.
  Tool execution keeps the proven invariants: serial by default; in
  `parallel` mode validation/permission/`before_tool_call` run sequentially
  in input order (Pi's preflight) and only prepared handlers share bounded
  concurrent segments; results in input order; exactly one terminal
  `ToolResult` per admitted call; duplicate Tool-call ids remain in the failed
  assistant message as protocol evidence and are rejected before Tool
  admission or any side effect (the executor also rejects a direct malformed
  batch before admission); terminal results are deeply
  immutable snapshots; absolute downward-propagated deadlines; and
  cooperative `CancelToken` cancellation (now `qitos.core.cancellation`)
  where `immediate` interrupts in-flight work and `after_step` stops the run
  after the current turn commits. External task cancellation and ordinary
  faults terminalize started work and the run record before re-raising when
  canonical appends settle. If an append fails or its outcome is unknown, the
  writer stops before crossing an open Tool boundary; close, replay and
  recovery establish the missing terminals without re-execution. Hook
  contracts mirror Pi: `before_tool_call` receives validated arguments and an
  immutable agent-context snapshot and may block the call;
  `after_tool_call` applies a field-level `AfterToolCallOverride`;
  progress reported by Tools surfaces as `ToolExecutionUpdate` events. Tool
  hooks observe the run cancellation/deadline boundary. Event listeners settle
  in subscription order, receive a read-only cancellation signal, and are not
  forcibly cancelled by `abort()`; a configured absolute run deadline bounds
  a listener that ignores that signal. Without a deadline, listener settlement
  remains part of `prompt()` / `wait_for_idle()` settlement. Error `ToolResult`s
  project their error text and an `is_error` flag into provider payloads
  (including Anthropic `tool_result.is_error`). The façade returns typed
  values for expected rejections only; listener/codec/persistence bugs are
  faults that propagate after the run is terminalized.

- **Façade-driven subagents.** `qitos.kit.subagent.AgentSubagentEngine` implements
  the `SubagentEngine` contract over the new `Agent` façade, and
  `qitos.kit.subagent.build_agent_subagent_invocation_factory` wires one Subagent run
  per invocation with a narrowed Tool registry (filtered to the request's
  allowed groups) and a value snapshot of the parent's effective Tool
  permission/scope policy, budgets tightened to the tightest of factory defaults,
  request budget and the parent's remaining deadline, and a per-run
  `JournalTurnTransaction` journal (`journal_directory` or an explicit
  SessionJournal factory). Parent messages enter the Subagent run's steering
  queue with journaled acceptance (`runtime_input.posted`) only while the run
  can guarantee a following turn safe point; starting and terminal-settlement
  windows reject instead of reporting false acceptance. Subagent control Tools
  use the executor-owned Run identity, and every abort
  still ends in a durable `run.interrupted` record. Subagent recovery now
  rebuilds a started Subagent's terminal `SubagentResult` from its own loop journal
  when that journal holds a run terminal record
  (`qitos.kit.journal.recover_run_outcome`), and closes it as `interrupted`
  only when no terminal record exists; journals written by the retired Engine
  taxonomy are rejected instead of being guessed.
- **Typed provider-neutral thinking level.** `qitos.core.thinking` defines
  `ThinkingLevel` (Pi's exact `off|minimal|low|medium|high|xhigh|max` values)
  with `clamp_thinking_level` implementing Pi's nearest-up-then-down rule.
  `ModelCapabilities.thinking_levels` declares the levels an adapter can
  translate (empty means no typed support); the OpenAI Responses, OpenAI Chat
  Completions and Anthropic Messages adapters declare the full range. The
  `Agent` façade (`thinking_level=...`), `AgentLoopConfig.thinking_level` and
  `NextTurnUpdate.thinking_level` feed one immutable level per turn; the loop
  clamps it against the turn's model capability and stores the result on
  `ModelRequest.thinking_level` (validated, durable codec, included in
  `request_digest`). Adapters translate the typed field through the same wire
  encoding the harness reasoning policy uses (`qitos.core.thinking.thinking_request_options`;
  `qitos.harness` now delegates to it): Responses emits `reasoning.effort`,
  Chat Completions emits `reasoning_effort`, Anthropic emits the manual
  `thinking` budget config (or the Kimi Messages `thinking` +
  `output_config.effort` variant, selected by provider identity), and `off`
  is an explicit disable signal (`effort: "none"`, `thinking:
  {"type": "disabled"}`). A typed `ModelRequest.thinking_level` overrides
  construction-time and per-request reasoning kwargs for exactly those wire
  keys; `None` leaves configured defaults untouched.
- **Typed Tool-result usage and activated Tool names.** `ToolResult` gains
  `usage` (`ModelUsage`, token/cost accounting for work the Tool itself
  performed) and `added_tool_names` (names of Tools this result activated,
  Pi's `addedToolNames`), both validated, deeply immutable and carried by the
  exact fail-closed codec (never by `to_model_dict` — they are not provider
  wire data). `ToolResultMessage` mirrors both fields, the loop propagates
  them onto every committed message, and they survive the journal
  `tool.terminal` / `step.committed` round trip.
- **Canonical transcript/operation journal split, pure recovery and the
  in-memory journal (S2a).** The Run journal now carries two record families
  in one sequence: transcript entries (`transcript.message`, `compaction`)
  own message content exactly once, and operation records reference them by
  record id. `JournalTurnTransaction` writes one `transcript.message` per
  accepted prompt followed by `input.accepted` before the first model side
  effect, per-turn freeze diffs (`model.change` / `thinking.change` /
  `tools.change`, full trio on the first turn), the assistant transcript
  entry plus `model.completed` (exact request audit plus
  `message_record_id`), the Tool-result transcript entry plus
  `tool.terminal` (call plus `message_record_id`), a pure `step.committed`
  commit marker of record references, and a `run.completed` /
  `run.interrupted` carrying only `{status, error}`. The loop's
  `TurnTransactionBoundary` gains `input_accepted` and `turn_frozen`
  barriers with a typed `TurnConfigSnapshot`.
  `qitos.kit.journal.recovery.recover_session` is a pure, total replay of one
  journal (INHERITED wrappers resolved) into the typed transcript in
  canonical conversation order, the compaction-projected context, the
  configuration lineage (including Tools activated by earlier Tool results),
  open Tool operations, unconsumed runtime inputs and the terminal outcome,
  failing closed with `JournalCorruptionError` on contradictions.
  `close_crashed_tool_calls` closes the legitimate crash window (admitted but
  never terminated, or never admitted calls) with explicit cancelled
  terminals and one closing commit — never re-executing, with deterministic
  record ids so closing twice never double-appends.
  `InMemorySessionJournal` implements the full `SessionJournal` contract
  against a process-local `InMemoryJournalStore` and shares one conformance
  suite with the JSONL implementation
  (`tests/journal/test_session_journal_conformance.py`). The recorder accepts
  a `RecoveredRecorderState` seed (next turn, last journaled config, recorded
  message count) so a resumed run continues without rewriting history, and
  can commit `budget.committed` per model terminal through an attached
  `BudgetLedger` with the same idempotency-key scheme the Subagent boundary
  uses. `runtime_input.consumed` records consumption of a posted runtime
  input; recovery folds own-run records only and never redelivers inherited
  fork facts.
- **Authoritative Session Harness (S2b).** `qitos.kit.session.SessionHarness`
  owns start, resume, fork, compaction and trace reattachment over the
  canonical Run journals. `start` creates the journal and an `Agent` whose
  transaction boundary is the seeded `JournalTurnTransaction`; `resume`
  replays the journal through pure recovery, closes any crash window with
  explicit cancelled terminals, and restores the façade (context messages,
  thinking level, configuration lineage) with typed `ResumeRejected`
  values (`not_found` / `terminal` / `model_mismatch` / `tools_missing`
  with the missing names / `busy`) — corruption raises
  `JournalCorruptionError` instead. `fork` branches at a committed boundary
  (default: the latest) into a new self-contained journal; forking a
  terminal run is the explicit continuation, and one `SessionRun` advances
  along the same mechanism when the caller prompts again after a leg
  settles. The default root `post_runtime_event` endpoint persists
  `runtime_input.posted`, steers the message and appends
  `runtime_input.consumed` once the steered message is covered by a
  `step.committed`; unconsumed inputs are re-projected exactly once on
  resume, and the Subagent engine now marks consumption the same way. Manual
  `compact()` runs at idle only (typed `CompactRejected` for `busy` /
  `nothing_to_compact`); automatic compaction evaluates Pi's token
  threshold at idle boundaries against the model's `context_window`, and a
  one-shot overflow recovery compacts and continues once after a
  context-overflow model failure (conservative provider error patterns plus
  usage-based silent-overflow detection, ported from Pi). Compaction
  follows Pi v3's algorithm (chars/4 estimation, keep-recent cut search
  that never splits a Tool call from its result, split-turn prefix merge,
  previous-summary iteration), persists the durable `compaction` record
  with summarization usage when available, swaps the context through
  `Agent.set_transcript` and seals Provider continuation so the next
  request is full. `qitos.trace.AgentTraceProducer` reattaches the trace
  writer to the façade event stream: each committed turn publishes its
  events with one `TraceStep` (step id = turn), lifecycle events write
  directly, and the manifest finalizes with the terminal status `qita`
  reads; trace artifacts remain observational and are never recovery truth.
  The `Agent` façade restores `initial_messages`, replaces its transcript
  between runs via `set_transcript` (busy runs reject), and the loop counts
  turns from a recovered `turn_base` so journaled turn numbering continues
  across resume. Journal recovery now replays nested fork prefixes as a
  sequence of closed per-run segments (turn barriers and Tool-call pairing
  are scoped per segment, since call ids are unique only within one run).
- **Goal-bearing Task with a durable journal lifecycle (S3a).** `qitos.core.task`
  is replaced in place with the architecture §5 contract: an immutable `Task`
  definition (`task_id`, optional `parent_task_id`, `objective`,
  `success_criteria`, string `constraints`, stable typed `references`
  (`TaskReference`, no filesystem probing), `budget`, `created_at` UTC
  provenance and `created_by_run_id`) plus
  `TaskStatus` / `TaskBlocker` / `TaskLifecycle` lifecycle values with
  invariant validation (blocker exactly while blocked, terminal reason
  exactly at a terminal status) and exact fail-closed codecs throughout.
  The journal gains `task.created` (the definition, committed before
  `input.accepted` and any model/Tool side effect) and `task.transition`
  records; `recover_session` folds them through the fork lineage into
  `RecoveredSession.tasks` (with an `unfinished_root` accessor) and fails
  closed on a second unfinished Root Task, terminal-once violations,
  from-status mismatches, unknown-task transitions, conflicting duplicate
  creations and root creations that follow model/transcript side effects.
  `SessionHarness.start` accepts the Root Task and commits it first;
  `SessionRun` gains `complete_task` / `fail_task` / `cancel_task` /
  `block_task` / `unblock_task` (durable before returning, with a budget
  usage snapshot when a ledger is attached, typed `TaskTransitionRejected`
  for expected misuse) and `start_follow_up(task, prompt)` for continuing a
  terminal-task lineage with an explicit new Task. A settled leg keeps its
  run terminal last, so post-run transitions commit into the advanced leg;
  leg advances carry Task facts that would otherwise be truncated by the
  fork boundary. Prompting a task-bearing Session whose Root Task is
  terminal rejects with `AgentRunRejected("task_terminal")`; taskless
  Sessions are unchanged. Subagent launch durably commits the narrowed Subagent Task,
  whose `SubagentLaunchRequest` carries `parent_task_id`, before `input.accepted`.
- **Durable progress-checklist Plan (S3b).** `qitos.core.plan` provides immutable
  `Plan` / `PlanItem` / `PlanStatus` contracts matching `update_plan`: each item has a
  concise step description and display status, with at most one item in progress.
  Task-bound `plan.updated` records are the sole Plan truth and pure recovery folds each
  Task's latest replacement through fork lineage, so a terminal follow-up starts without
  the previous Task's Plan. Root and Subagent use this one optional contract; Plan does
  not express dependencies, ownership, assignment, scheduling or completion truth.
- **Subagent product-binding and conclusion contracts.** `SubagentLaunchRequest` now carries
  explicit success criteria, Task constraints/references and a frozen
  `ToolPermissionContext`; all fields use the strict durable codec and the façade Subagent
  engine commits them on the narrowed Subagent Task. `SubagentInvocation` accepts a typed
  async conclusion factory that settles before resource cleanup, and
  `AgentConclusion.resource_refs` carries stable reusable-resource references without
  copying a Subagent transcript. Recursive Agent Tools can share one Root-owned
  `SubagentRunLimiter`, while independent concurrency-safe Agent calls remain ordered at
  the parent ToolResult boundary.
- **Independent active and cumulative Subagent limits.** `SubagentRunLimiter` accepts
  `max_subagents=None` as an unbounded cumulative launch budget while continuing to
  enforce `max_active_subagents`. Terminal Subagents immediately return active capacity;
  an explicit cumulative limit still includes restored launch history.
- **Committed Tool-transaction projection.**
  `qitos.kit.journal.committed_tool_transactions` returns only canonical Tool terminals
  referenced by `step.committed`, follows inherited record origins and fails closed on
  missing or duplicate references so application state can be rebuilt without replaying
  handlers.

### Breaking

- **Flat WorkPlan replaced in place (S3b).** `WorkPlanState`, `WorkPlanItem`,
  `WorkPlanStatus`, `WorkPlanUpdate`, `UpdateWorkPlanTool` and their codecs/reducer
  are removed rather than aliased. Use `Plan`, `PlanItem`, `PlanStatus`,
  `PlanUpdate`, `UpdatePlanTool` and the Plan codecs. `SubagentLaunchRequest.from_dict`
  now requires the canonical task/product-binding keys, including `success_criteria`,
  `constraints`, `references`, `permission_context` and explicit `null` values.
  The unused `qitos.kit.state` package is removed.

- **`Task` schema replaced in place (S3a).** `Task` loses `id` (renamed
  `task_id`), `inputs`, `resources`, `env_spec`, `metadata`, `validate`,
  `validate_structured` and `resolve_resources`; `TaskResource`,
  `TaskResult`, `TaskCriterionResult`, `TaskResourceBinding` and
  `TaskValidationIssue` are deleted with no aliases or mirrors. Benchmark
  resources, environment probing, metrics and free-form metadata stay at
  application boundaries and reference the Task by id. The evaluation DSL
  `task` variable now exposes the new `task.to_dict()` shape, and `RepoEnv`
  no longer accepts or probes a Task (the `required_missing` observation key
  is gone). `TaskBudget.to_dict` / `from_dict` is the canonical budget codec
  (unchanged six-key payload).

### Changed

- **Breaking:** The public Agent-delegation vocabulary is now consistently `Subagent`:
  `qitos.core.subagent`, `qitos.kit.subagent`, `qitos.kit.tool.subagent`,
  `SubagentTool`, `SubagentSupervisor`, `SubagentHandle`, `SubagentResult`, and the
  `subagent_*` control Tools replace the old names. New journals and runtime inputs use
  `subagent.started` / `subagent.terminal` and `agent.subagent.*`. Strict decoders keep
  one-way read migration for already-persisted `child.*` data; no old Python API alias
  remains.

- Trace DTO serialization now traverses dataclass fields and immutable mappings without
  `deepcopy`, so deeply frozen Tool arguments remain serializable without weakening the
  runtime snapshot boundary.
- `CancelToken` / `CancelMode` moved from the retired Engine lifecycle to
  `qitos.core.cancellation`; the compatibility export is removed.
- `SubagentTool` (and the Subagent completion/runtime projections in
  `qitos.core.runtime_input.subagent_result_payload`) no longer report
  `total_tokens` / `total_cost_usd` inside the untyped output payload. Subagent
  token/cost accounting rides the typed `ToolResult.usage` carrier
  (`cost_usd` as a lossless detail key) on terminal Subagent results; status,
  conclusion, completeness flags (`usage_complete` / `cost_complete`), steps
  and elapsed time stay in `output`. The Agent Tool now returns a typed
  `ToolResult` on the success path instead of a plain payload mapping. The
  durable `subagent.terminal` journal record still carries the full typed
  `SubagentResult` facts.

### Removed

- **Breaking:** Removed the retired `AgentModule`/`Engine` lifecycle and its
  exclusive surface: the whole `qitos/engine/` package (action executor,
  journal/control/turn runtimes, handoff runtime, hooks, streaming, recovery,
  stop criteria, interrupt), the `qitos/core` old-lifecycle modules
  (`agent_module`, `observation`, `decision`, `action`, `state`,
  `state_delta`, `field_reducers`, `channel`, `agent_spec`, `message_builder`,
  `completion`), `qitos/checkpoint/`, `qitos/protocols.py` (prompt-injected
  protocol registry), `qitos/render/` (Engine hook renderer), the Engine-era
  kit packages (`qitos.kit.{parser,critic,planning,prompts,history}`), and
  `qitos/prompting.py` / `qitos/core/_json_repair.py` (zero surviving
  consumers), plus the now-unreferenced `qitos/core/history.py` and
  `qitos/core/turn.py`. `qitos/core/errors.py` keeps only the model-facing error
  types; `core/spec.py` (`RunSpec`/`ExperimentSpec`/`BenchmarkRunResult`) stays
  because result-file evaluation and the leaderboard store consume it. The
  `Agent` façade over the minimal agent loop is the only execution path.
- **Breaking:** `ToolTransaction` decodes only the loop's journal schema
  (`turn` + typed `ToolCall`); the retired Engine payload fields
  (`step_id`/`action_index`/`action`) fail closed on decode, consistent with
  the no-cross-path-resume rule. `ToolResult` validates its status against
  the local `ToolResultStatus` literal set instead of the deleted
  `ActionStatus` enum.
- **Breaking:** `qitos.harness` no longer builds Engine parsers or
  prompt-injected protocol objects: `HarnessPolicy` carries `protocol_id` /
  `fallback_protocol_ids` identities instead of `protocol`/`parser`
  instances, and `build_model_for_preset` attaches protocol identity as
  metadata only. Family presets, adapters, `resolve_family_preset`, and
  reasoning policies are unchanged.
- **Breaking:** `qit` drops no further commands in this slice; `qita`, the
  trace readers, and the leaderboard store are untouched and read existing
  run directories and result files as before. Note the loop-side trace writer
  is not yet reattached, so new façade runs journal but do not emit
  engine-era trace manifests yet.
- **Breaking:** Removed the benchmark execution layer: `qitos.recipes` is gone
  entirely (`benchmarks/`, `desktop/`, and the pattern recipes), together with
  the recipe-coupled `qitos.benchmark` adapter subpackages (`cybench`,
  `desktop`, `gaia`, `osworld`, `tau_bench`), `qitos/benchmark/runner.py`, the
  `qit bench run` / `qit bench list` commands, the `qitos[benchmarks]` extra,
  `examples/benchmarks/`, and `examples/real/openai_cua_agent.py`.
  The orphaned `BenchmarkAdapter`, `BenchmarkRuntimeHook`,
  `BenchmarkEvaluator`, `BenchmarkScorer`, and `PreparedBenchmarkTask`
  extension protocols are removed as part of the same deletion boundary;
  they had no implementation or caller after the family adapters left.
  `qitos.benchmark` keeps only `BenchmarkRunResult` file read/write and result
  aggregation, so `qit bench eval`, `replay`, `export`, and `presets` plus the
  leaderboard store keep working on result files and run directories produced
  elsewhere. Stale `.[benchmarks]` installation references are removed.
  Benchmark docs move to git history; run inspection stays with `qita`.
- **Breaking:** Removed the pattern method recipes
  (`qitos.recipes.{lats,magentic_one,moa,reflexion,self_refine}`): AgentModule
  Agent + Critic templates with no caller outside their own tests. The
  surviving composition path is the Agent façade plus subagents; the
  method-templates guide and glossary entries leave with them.
- **Breaking:** Removed the `qitos_zoo` submodule (gitlink and `.gitmodules`).
  The fork carries no product showcase agents; the zoo's coder/cyber
  applications were Engine-era compositions with no remaining consumer.
- **Breaking:** Removed `qitos.demo` and the `qit demo` command. The packaged
  minimal coding demo was an `AgentModule + Engine` composition; the
  quickstart example now composes the `Agent` façade directly.
- **Breaking:** Removed the Engine-coupled example set: `examples/patterns/`
  `react.py`, `planact.py`, `reflexion.py`, `tot.py` and `handoff.py`, all
  `AgentModule`-based `examples/real/` agents (the env-only
  `desktop_env_smoke.py` stays), the Engine-based `examples/zh/` agents, and
  the tutorial course that taught them (`docs/tutorials/react|planact|
  claude-code|code-security-audit|multi-agent` and `docs/guides/agent-patterns`,
  English and Chinese). `examples/quickstart/minimal_agent.py` is ported to
  the Agent façade (`Agent(model=..., tool_registry=...)` +
  `await agent.prompt(...)`); engine-free examples
  (`patterns/function_tool_custom.py`, `patterns/embedding_vectorstore.py`,
  `zh/embedding_vectorstore.py`) remain.
- **Breaking:** Removed the Engine-coupled delegation tools
  `qitos.kit.tool.delegate` (`DelegateTool`) and `qitos.kit.tool.fanout`
  (`FanOutTool`) together with `AgentRegistry.get_delegate_tools()` /
  `get_fanout_tool()` and their examples. Subagents now run through the
  façade-driven `AgentSubagentEngine` (see Added); the retired delegate/fanout
  event names are removed with the Engine lifecycle.
- **Breaking:** Removed the kit-side handoff tool chain:
  `qitos.kit.tool.handoff_tool` (`HandoffTool`),
  `AgentRegistry.get_handoff_tools()`, and the Engine's `handoff_targets`
  auto-registration. The internal Decision-handoff policy is removed with the
  Engine lifecycle.
- **Breaking:** Removed `qitos.core.shared_memory` (`SharedMemory`,
  `InMemorySharedMemory`, `FileSharedMemory`, `SharedMemoryNamespace`,
  `SharedMemoryManager`) and its wiring: the `Engine(shared_memory=...)` /
  `ActionExecutor(shared_memory=...)` parameters, the
  `ExecutionConfig.has_shared_memory` projection, the `shared_memory` runtime
  context entry, `AgentSpec.shared_memory`, and the handoff namespace setup.
  The blackboard was consumed only by the retired delegation tools and the
  Engine multi-agent path; parent/Subagent collaboration uses typed Subagent
  results and messages instead.
- **Breaking:** Removed the `qitos.kit.repl` package (`AgentREPL`). It drove
  the Engine stepping API interactively and had no consumer outside the zoo
  coder CLI, which leaves with the zoo removal slice.
- **Breaking:** Removed the deprecated `qitos.cache` package and
  `Engine(cache_backend=...)`. The implicit wrapper mutated an Agent's model during
  Engine construction and replayed complete prior transactions with stale usage,
  continuation, deadline, and trace semantics. Provider-native prompt caching remains
  available through each model adapter and its recorded usage fields.
- **Breaking:** Removed `qitos.config`, `qitos.experiment`, and the `qit experiment`
  command. The YAML builder and parameter-sweep runner formed a parallel runtime that
  bypassed canonical Agent runs, Journals, and traces while ignoring most declared
  settings. Construct models through provider classes or `ModelFactory`, and run
  reproducible benchmark tasks through `qit bench` and canonical Run specs.
- **Breaking:** Removed the fixed `qitos.kit.patterns` Manager/Worker,
  Planner/Executor, Proposer/Verifier, Debate, MoA, and DAG workflow templates.
  They formed a synchronous parallel orchestration layer with no production caller
  inside QitOS. Applications should express coordination policy through the typed
  `SubagentTool` and `SubagentControlToolSet` lifecycle or in application-owned recipes.
- **Breaking:** Removed the deprecated product-level `SecurityAuditAgent` template and
  redundant security-audit Tool/ToolSet forwarding packages. Security research remains
  available from the explicit `qitos.kit.tool.experimental.security_research` owner,
  which is no longer loaded as a side effect of importing the generic kit.
- **Breaking:** The Run journal record contract moved to the canonical
  transcript/operation split. `state.snapshot` is deleted as a record type
  (journals containing it now fail at envelope decode); `step.committed` no
  longer embeds `messages` or legacy `terminal_record_ids`;
  `model.completed` no longer embeds `message`; `tool.terminal` no longer
  embeds `result`; `run.completed` / `run.interrupted` no longer embed
  `messages`; `input.accepted` now carries `transcript_record_ids` instead of
  a task payload. Journals written before this change fail closed in the
  catalog, the transaction index, the budget ledger and pure recovery
  instead of being silently skipped: the Engine-era payload readers
  (`stop_reason`, `reason`, `task`, commit terminal flags, the legacy
  aggregate budget fallback) are removed. The migration branch has never
  been released, so no journal migration path is provided. `RunHandle.task`
  reports empty until the goal-bearing Task slice lands.
- Removed the unexported `WorktreeManager` helper and its self-only tests. No runtime
  called it, and its fallback created an invalid pseudo-worktree instead of an isolated
  repository. Applications that need isolation should own a real workspace runtime.
- **Breaking:** Removed the unused network-backed SkillHub installer, mutable Skill
  registry, `SkilledAgent` mixin, `qit skill` command, runtime install/search tools,
  and their compatibility loader. `SkillToolSet` now exposes only application-owned
  bundled `SKILL.md` discovery, full loading, and bounded relative-resource reads.
- **Breaking:** Removed the orphaned `SearchBackend` hierarchy and its DuckDuckGo,
  Google CSE, Perplexity, SearXNG, Sploitus, Tavily, and Traversaal adapters. Managed
  public search remains available through `WebSearchCapability` and
  `ManagedWebSearchTool`.
- **Breaking:** Removed deprecated compatibility import paths under `qitos.kit.tool`
  (`taskboard`, `report_toolset`, `text_web_browser`, `skill_tools`,
  `network_toolset`, `web_test_toolset`, and `tools`). Use the canonical tool and
  toolset packages instead; the experimental security-research modules remain
  available only from their explicit opt-in package.
- **Breaking:** Removed the unused `qitos.func` decorator and composition package.
  Its `@agent`/`@task` callables bypassed Engine transactions, while its advertised
  `AgentModule` conversion never executed the wrapped function. Applications now use
  the `Agent` façade for Agents and the maintained function-tool decorator for ordinary
  callable Tools.
- **Breaking:** Removed `qitos.checkpoint.fork`, including `fork_checkpoint()` and
  `list_fork_history()`. Copying a checkpoint snapshot did not create a canonical Run
  transaction or resumable lineage. Checkpoint persistence and snapshot resume remain;
  executable branching uses `SessionJournal.fork()` at a committed position.
- **Breaking:** Removed the unused `qitos.engine.run_state` whole-result snapshot,
  Snowl-specific adapters/docs, and the source-only template/scaffold CLI. No Engine
  resume path consumed `RunState`, and packaged distributions never contained the
  templates required by `qit new` or `qit list-templates`. Canonical recovery remains
  owned by Session Journals or checkpoint stores. The completed temporary zoo migration
  staging and its no-op workflow were also removed now that product applications live
  in the independent `qitos-zoo` repository.
- **Breaking:** Removed the process-global `qitos.tracing` provider/processor
  hierarchy, the `Engine(..., tracing_provider=...)` argument, and the W&B/MLflow
  extras. Canonical runtime events, `TraceWriter`, the Session Journal, and `qita`
  now form the single observability and replay path. `EngineConfig` reports this
  path through `has_trace_writer`.
- **Breaking:** Removed the deprecated `qitos.debug` replay/inspector package and
  qita's trace-file mutation endpoint. `qita` is now a read-only artifact viewer;
  executable forks use the canonical Session Journal and a committed continuation
  boundary.

### Changed

- Adopted the Pi-aligned provider-neutral Model, minimal async Agent loop, and `Agent`
  façade as the only execution path. The remaining plan makes Session/Harness
  authoritative and replaces Task/Plan in place; it removes benchmark, environment,
  metrics, and free-form metadata concerns from canonical Task.
- Consolidated internal architecture history into one target architecture and one
  active migration plan, removing superseded Engine, handoff, WorkPlan, release-roadmap
  and completed task plans. Contributor instructions now use one short risk-based
  workflow instead of requiring plans, full checks, README news and changelog edits for
  every non-trivial change.
- Long-running model input is now append-only between explicit compactions. The old
  recent-round projection is gone; Engine warns at 80%, compacts at 85%, records stable
  cache-affinity/prefix telemetry, and exposes an immutable `ContextSnapshot` contract
  for application state without rewriting ToolResults or prior messages.
- Model-visible Tool output now has one Engine-owned envelope: 8,000 characters per
  result and 16,000 per batch. Complete oversized output is persisted first and replaced
  by a deterministic head/tail receipt with Artifact path, size, and read reference.
- `RuntimeBudget.terminal_synthesis` can reserve the final step for a tool-free
  conclusion with a deterministic Agent fallback and durable resume boundary.
  `PermissionMode.AUTONOMOUS` lets authorized applications suppress QitOS approval and
  shell-shape heuristics while retaining explicit deny rules and runtime boundaries.
- Foreground commands now yield a stable managed-process handle when they outlive their
  initial wait. Process wait/read cancellation and deadline results preserve the latest
  bounded handle snapshot so resume and mailbox completion remain actionable.
- Managed `web_fetch` now projects its requested URL into the canonical Tool permission
  scope, including the direct `BaseTool` permission path, so application network rules
  are applied consistently before a fetch begins.
- An admitted managed `web_search` schema now prefers the tested provider-hosted
  contract on the official OpenAI Responses endpoint, Qwen Responses/Chat, and the
  official Anthropic Messages endpoint. Native output items and citations remain canonical. Only an
  explicit request-time unsupported response can retry with the managed schema;
  provider output already published to the Run is never replayed. Kimi-compatible
  Messages endpoints continue to receive an ordinary managed Tool.
- Docker runtime-profile command probes now execute in the selected container and the
  resulting verified commands are retained in its immutable capability snapshot.
- CI, documentation validation, and contribution checks now run only when explicitly
  dispatched from GitHub Actions. The manual jobs retain the supported-Python test
  matrix, coverage, packaging, lint, type, audit, and focused contribution checks.
- Text Web, coding Web fetch, HTML extraction, and EPUB reading now share one
  Beautiful Soup parser policy. The required dependency replaces three unreachable
  optional-dependency branches and their regex fallbacks; text search results are also
  selected from the parsed DOM instead of matching HTML with a regular expression.
- QitOS now uses `httpx` as its only direct HTTP client dependency. Text Web, desktop
  controllers, OSWorld setup/probes, and streamed VM downloads preserve redirects and
  timeout behavior while sharing the same SOCKS-capable transport dependency as MCP
  and managed Web capabilities.
- Run handles now expose a stable lineage id, immediate parent, latest committed
  position, and latest non-terminal continuation position. `Engine.arun()` can bind a
  product lineage id, forks inherit it, legacy Journals derive a stable root id, and a
  terminal Run can be continued only through an explicit resumable fork. Terminal
  steps carry an explicit marker so even a crash before the terminal snapshot cannot
  expose completed state as a continuation boundary.
- Environments now publish one typed, immutable `RuntimeCapabilitySnapshot` with
  backend identity, working directory, operation groups, optional facilities,
  verified commands, and stable limitations. Engine captures that exact snapshot in
  each turn, filters Tools whose operation groups are unavailable, and passes the same
  snapshot to Tool execution. Environment initialization and health probes now run
  through awaited lifecycle methods; local host and ordinary-container processes share
  the managed file, foreground, background, PTY, stdin, and shutdown contracts.
- Root Engines and every descendant can now share one journal-backed `BudgetLedger`.
  Completed model transactions atomically settle token and cost usage into the Root
  JSONL, while `EngineResult` and `SubagentResult` preserve local totals and usage
  completeness for audit. Local Subagent ceilings further narrow, rather than duplicate,
  the shared Run allowance.
- Canonical `ToolResult` now carries the model ToolCall `call_id`, assigned by the
  Engine before application finalization and reconstructed for older Journal records.
  Subagent invocation factories may complete async resource construction while existing
  synchronous factories remain supported.
- Engine-owned MCP catalogs now refresh atomically at pre-turn safe points after an
  explicit request or `notifications/tools/list_changed`. Discovery follows bounded
  cursor pagination, preserves official SDK Tool models, retains the last complete catalog
  on failure, and uses the same Tool exposure, permission, deadline, cancellation,
  terminal-result, Journal, and cleanup path as native Tools. Stdio and Streamable HTTP
  now use the official `mcp` Python SDK for protocol models, initialization, transport,
  notification, and shutdown behavior. A dedicated per-server task owns every SDK
  context from enter through exit while active requests remain independently cancellable.
  Expired sessions recover only at safe discovery boundaries and never replay a failed
  side-effecting Tool call. Run startup discovers independent servers concurrently under
  a fixed limit and timeout, then publishes successful catalogs in factory order.
  Interactive sessions lazily start and refresh MCP through `astep()` on one owning
  event loop; synchronous `step()` rejects MCP-backed sessions explicitly.
- Host PTY creation now uses `ptyprocess` for the platform PTY/fork/exec boundary while
  QitOS retains async incremental I/O, process-group termination, Journal recovery,
  terminal notification, and owner-scoped cleanup.
- Bundled Skill roots now use recursive nearest-root discovery, deterministic
  first-root-wins precedence, and typed non-fatal diagnostics. Explicit refresh
  replaces the catalog atomically; bundle revisions cover both `SKILL.md` and resource
  bytes, stale resources fail instead of mixing revisions, and an optional immutable
  requirement inventory gates full Skill loading.
- Context recovery coverage now exercises compaction, cancellation, committed-boundary
  fork, and resume as one Run. It verifies ordered ToolCall/ToolResult parity in the
  canonical Journal while resume alone may retain a matching Provider continuation.
- Canonical `write_file` and `edit_file` now commit through the selected environment's
  atomic replacement operation. Reads expose a complete-file SHA-256 revision;
  explicit and implicit compare-and-swap guards turn concurrent changes into a stable
  `file_revision_conflict` result instead of a lost update.
- Subagent invocation factories now receive the already-journaled `SubagentHandle` in their
  runtime context, so product runtimes can correlate independent Subagent sessions and
  traces without deriving identity from mutable task data.

### Breaking

- Custom `WebSearchCapability` implementations must provide async `aclose()`. The
  managed Tool now uses that typed lifecycle directly instead of probing for an
  optional close method at runtime.
- `CapabilityEnv(attestation=...)` has been replaced by
  `CapabilityEnv(snapshot=RuntimeCapabilitySnapshot(...))`.
  `TurnRuntimeCapabilities` now stores the complete Runtime snapshot; its
  `environment_ops` property remains a derived read-only view. Environment subclasses
  that allocate asynchronously may override `ainitialize()` and `ahealth_check()`;
  legacy synchronous hooks run in a worker thread by default.
- Live MCP transports no longer belong to `AgentModule.mcp_servers`. Applications pass
  `Engine(mcp_server_factory=...)` a construction-only factory that returns fresh,
  unconnected transports for each Run, preventing Root, Subagent, resumed, or repeated
  Runs from sharing a process, HTTP client, or session.

- `MCPServer.list_tools()` and `MCPServer.call_tool()` now use the official
  `mcp.types.Tool` and `mcp.types.CallToolResult` models directly. The redundant
  `MCPToolInfo`, `MCPToolAnnotations`, and `MCPCallToolResult` mirrors were removed;
  callers should import protocol models from `mcp.types`. MCP bridges still project
  remote results into regular QitOS `ToolResult` values with stable classifications.
- Subagent invocation factories are now async-only and must resolve to a
  `SubagentInvocation`. This makes cancellation and partial-construction cleanup part of
  the same awaited ownership chain instead of a synchronous pre-run side effect.
- Custom `FileSystemCapability` implementations must provide `write_text_atomic()`.
  The method owns same-path mutation ordering, optional SHA-256 preconditions, and the
  final replace boundary; `write_text()` remains the unconditional convenience path.
- `AgentRequest`, `AgentInvocation`, and `AgentResult` have been replaced by the
  immutable core `SubagentLaunchRequest`, `SubagentInvocation`, `SubagentHandle`, `SubagentResult`,
  `SubagentStatus`, and `AgentConclusion` contracts. `SubagentTool` now accepts a complete
  `TaskBudget`, returns parent-scoped handles, and exposes typed `subagent_result(handle)`
  and `cancel_subagent(handle)` lifecycle operations. Tool execution status remains
  separate from the Subagent task's `subagent_status`.
- `CommandCapability.start()` has been replaced by the async typed
  `astart()`/`apoll()`/`aread()`/`awrite()`/`await_process()`/`aterminate()` lifecycle.
  Environment cleanup now awaits `Env.ateardown()` so managed subprocess readers and
  watchers settle before a Run closes. Docker's untracked shell-detach implementation
  was removed; a remote environment must implement the same managed async contract
  before exposing background execution.
- `ModelStreamChunk` has been replaced by the discriminated `ModelStreamEvent`.
  Provider implementations must emit exactly one explicit event kind and terminate
  with `COMPLETED` or `FAILED`; only `COMPLETED` is a successful transaction.
- `Model.stream()` now accepts one immutable `ModelRequest` instead of mutable
  `messages + **kwargs`. Provider implementations read isolated message and option
  projections from that request; the Engine is the only owner that assembles it.
- Class-based tools now implement `async execute(args, runtime_context)`. Managed Web
  capabilities and their concrete adapters are async as well. Synchronous decorated
  functions remain supported through one explicit compatibility boundary, but code
  inside an active event loop must await Tool and Engine APIs.
- Runtime components now await `Engine.apost_runtime_event()`. The synchronous
  `post_runtime_event()` entry remains only for callers outside the Engine event loop
  and schedules onto that loop without creating a temporary loop.

### Fixed

- Concrete kit Tools now return explicit typed error results for execution failures;
  ordinary mappings, including mappings with domain fields named `status`, remain
  successful Tool output. HTTP 4xx/5xx, capability failures, invalid input, and
  unavailable resources therefore keep their error lifecycle without making the
  executor guess application payloads.
- Durable Message, ToolResult, and Model usage codecs now reject non-string mapping
  keys and non-finite numbers instead of accepting values that JSONL cannot reproduce.
  Tool runtime contexts also keep the frozen Env permission context authoritative, so
  caller-supplied context cannot replace the policy used at admission.
- The public one-at-a-time queue value now matches Pi's `one-at-a-time` spelling and
  invalid raw queue modes fail closed. Reaching `max_turns` no longer consumes steering
  or follow-up input that cannot be injected, and the façade restores any input drained
  immediately before a run rejection.
- Subagent launch now intersects requested Tool groups with the parent's frozen exposure,
  carries a value snapshot of the parent's permission/scope authority, and applies one
  absolute deadline across limiter admission, factory construction, execution, and
  cleanup. Subagent control Tools prefer the executor-owned parent Run identity, while
  mailbox posting rejects startup, between-turn, and terminal-settlement windows where
  no later safe point can be guaranteed. Posts reserved during a live turn settle in
  FIFO order at Pi's existing prepare-next-turn boundary, before steering is drained;
  cancellation and persistence faults preserve committed, rolled-back, and unknown
  journal outcomes instead of returning a rejection beside a durable ghost input.
- Canonical Tool results now use an exact Journal decoder, so terminal Runs containing
  an empty error string can be resumed without rewriting or rejecting durable records.
- Terminal-synthesis Runs now commit a non-empty cancellation conclusion and
  `run.completed` boundary instead of leaving resumable work at `run.interrupted`.
  Controlled `Engine.cancel()` returns that terminal result; direct caller Task
  cancellation propagates only after Tool terminals, the terminal step, and Journal
  completion are durable. Immediate and after-step cancellation remain distinct stop
  reasons.
- The declared development environment now installs `pytest-asyncio`, so the complete
  async suite executes in CI instead of producing unknown-mark warnings and hundreds
  of unsupported-coroutine failures. Build and audit environments require a
  `setuptools` release containing the fix for `PYSEC-2026-3447`.
- MCP startup, Subagent waits, and their behavior tests now use structured primitives
  available across the declared Python 3.10+ range. Python 3.10 cancellation tests
  verify the canonical Journal outcome when its runtime normalizes a cancelled Task's
  `CancelledError` subclass.
- Manual contribution checks now import the canonical `ToolSpec`, and the Chinese
  documentation set includes the multi-agent tutorial required by the bilingual
  navigation validation.
- Foreground Subagent supervision now distinguishes `SubagentInvocationCancelled` from an
  actual caller Task cancellation. Both outcomes persist one cancelled Subagent terminal,
  but only `asyncio.CancelledError` from the caller aborts the parent.
- Caller cancellation after Tool execution now reuses the executor's typed terminal
  `cancel_source` instead of inspecting Python-version-specific Task internals. The
  Engine still commits one terminal result per ToolCall before propagating cancellation.
- Resume now derives idempotent background Subagent and process completion inputs from local
  terminal facts. A crash between `subagent.terminal` / `process.terminal` and mailbox
  acceptance no longer loses the notification; foreground Subagent results are not
  redelivered, consumed ids stay consumed, and forks do not receive inherited completions.
- Cancelling MCP shutdown now waits for active requests and asks the task that entered
  the official SDK contexts to finish their transport cleanup before cancellation
  propagates. SDK validation, remote errors, timeouts, closed transports, and expired
  sessions map back to stable QitOS error categories.
- Engine now rejects native and text-salvaged ToolCalls from known incomplete, length-
  limited, failed, filtered, or cancelled model terminals. The calls remain visible as
  typed `invalid_tool_calls` diagnostics but can never reach a Tool handler.

- Journal recovery now retains the latest Provider continuation when a complete Tool
  batch reached durable terminal records but crashed before `step.committed`; the
  recovered step still resumes from the canonical local transcript if the Provider
  later rejects that optimization.
- Session Journal payloads now cross one strict JSON boundary before append, keeping
  in-memory replay, reopened replay, stable record IDs, JSONL, and projection digests
  consistent. Unsupported schema versions now have a dedicated upgrade error, and a
  failed source close during fork no longer leaks the unreturned child's writer lease.
- Nested Journal forks now resolve inherited records to their canonical origin across
  every ancestor. A fork-of-fork can resume independently without replaying a
  completed Tool, and inconsistent inherited identity fails closed.
- Plain-text Task coercion now derives a stable task ID from content instead of wall
  time. Trace provenance preserves explicit structured Task IDs in an independent
  `task_hash` without mixing task data into `run_config_hash`; qita requires the task
  fingerprint for same-spec comparisons while keeping older manifests readable.
- The `SubagentRunResult.records` contract is now a read-only sequence view, so concrete
  typed `EngineResult` values structurally satisfy the Subagent Engine protocol.
- Synchronous Engine hooks can now defer runtime input for durable async acceptance at
  the next turn safe point instead of calling the external sync mailbox bridge from
  the Engine event loop.
- Inferred model protocols are now resolved from each immutable turn snapshot instead
  of being cached for the whole Run, so a model change takes effect on the next turn.
- Native tool calls now require a complete protocol terminal before execution. Chat
  drops calls on output-limit/non-tool finishes; Anthropic keeps interleaved block
  arguments isolated and records, but does not replay or execute, unclosed, malformed,
  non-object, or non-`tool_use` terminal blocks.
- Responses streams now preserve lifecycle/item events and refusals, backfill text or
  reasoning that appears only in terminal output, and keep interleaved function-call
  argument state separate. Incomplete responses cannot publish executable tool calls;
  malformed or conflicting terminal events fail explicitly.
- Engine context telemetry now preserves a provider-reported zero input count,
  distinguishes provider usage from local estimates and absent usage, and projects
  cache-read, cache-write, and reasoning counts without adding those subset counts to
  cumulative totals.
- A family preset can now select a different explicit wire adapter without carrying
  request defaults from the preset's original adapter. This lets Kimi K3 use either
  compatible Chat Completions or Anthropic Messages while preserving the Kimi family
  identity and mapping Messages reasoning to `thinking` plus `output_config.effort`.
- Local trace events now retain the completed model transaction's provider, model,
  finish reason, typed usage, and usage source while recursively redacting nested
  credential fields.
- Anthropic presets now build the native Messages adapter, preserve default request
  options through every construction path, and map Claude 4.5 reasoning effort to a
  bounded manual thinking budget that reaches the provider payload. Presets prefer
  native API tool schemas and typed tool calls instead of duplicate prompt injection.
- Journal recovery now closes interrupted tool calls with explicit terminal results:
  started calls retain unknown-side-effect status, unstarted calls are cancelled, and
  neither path replays a handler. Engines also reject Journal/checkpoint dual writes.
- Multi-action Journal batches now reduce each durable terminal before finalizing the
  next result, so later state-aware finalizers observe earlier tool outcomes.
- Journal-backed finalizer and reducer failures now abort the uncommitted transaction
  instead of entering legacy in-memory ACT recovery and publishing invalid state.
- MCP tools now execute on the transport's owning event loop and are removed from
  shared registries during Engine cleanup, preventing cross-loop failures and stale
  registrations when an Agent or registry is reused.
- Resuming a terminal checkpoint now finalizes a newly configured empty trace with
  the persisted stop reason and result instead of leaving its manifest running.
- `run_command` now executes after the caller's normal tool admission instead of
  applying a second command-permission decision inside the handler.
- Resuming a terminal checkpoint now returns its persisted state without issuing
  another model request, executing tools, or writing a descendant checkpoint.

### Added

- Added `QwenWebSearchCapability`, which uses DashScope's native Chat
  `enable_search` option and preserves structured `search_info.search_results` as
  bounded `WebSource` values. It remains available as the managed fallback when a
  model transport cannot accept hosted Web Tools.
- Added immutable `RunHandle`/`RunStatus` contracts and a lease-free
  `JsonlRunCatalog` for inspect, deterministic listing, validated lineage, and direct
  children. Catalog reads use only an exact read-only SQLite projection or canonical
  JSONL prefix and never acquire ownership, repair, or rebuild storage.
- `SubagentTool` composition can now populate the Subagent profile, allowed Tool groups, and
  working directory in each immutable launch request. `SubagentInvocation.cleanup` gives
  the supervisor explicit async ownership of per-Subagent model and trace resources across
  success, failure, and cancellation.
- Added a reusable Run-owned `SubagentSupervisor`. It admits fresh Engines under one async
  concurrency limit, supports non-destructive wait and bounded interrupt, stores
  terminal results before parent delivery, and continues to own delivery tasks until
  shutdown drains them. `SubagentTool` is now only the model-facing launch projection.
- Subagent supervision now journals `subagent.started` before constructing an Engine and
  `subagent.terminal` before parent delivery. Recovery marks a started Subagent without a
  terminal as `interrupted` and never replays it; forked Runs do not acquire authority
  over inherited parent handles.
- Added `subagent_status`, `subagent_wait`, `subagent_message`, and `subagent_interrupt` tools over
  the shared supervisor. Message delivery uses the active Subagent's async durable mailbox;
  wait timeout preserves execution, interrupt awaits cleanup, and unknown or foreign
  handles return stable state.
- Added canonical Subagent contracts that keep persisted launch intent, stable
  identity, lifecycle status, bounded conclusions, evidence references, and live Engine
  ownership separate. The contracts round-trip without serializing a runnable Agent.
- Added durable run-scoped mailbox acceptance. Journal-backed events append
  `runtime_input.posted` before waking the Engine, bind to the next completed model
  transaction, and are redelivered exactly once on resume if that transaction was not
  durable. Final completion is linearized with mailbox acceptance, so input accepted
  during a final turn is processed on a following turn instead of being silently lost.
- Added a Run-owned host process supervisor with opaque `ProcessHandle` values,
  immutable snapshots, incremental bounded UTF-8 output, complete workspace logs,
  stdin and POSIX PTY interaction, active-process limits, and graceful process-group
  shutdown with forced escalation. Journal-enabled background starts persist exactly
  one `process.started` and `process.terminal` record, and a failed started-record
  write reaps the process before returning an error. The shell profile now exposes
  process list/read/write/wait/terminate tools through that same capability. Resume
  closes an interrupted start as `lost` without replaying or reattaching it; forked
  Runs receive no authority over inherited parent handles. Terminal watchers now post
  one bounded `process.completed` RuntimeInput only after `process.terminal` is
  durable, so an active Agent wakes through the existing turn-safe mailbox rather than
  receiving an out-of-band state mutation.
- Added durable model request snapshots and guarded OpenAI Responses continuation.
  `model.completed` records the exact credential-redacted Provider input plus an
  optional Run-bound handle. Resume reuses a handle only when provider, model,
  protocol, request settings, and canonical input prefix still match; fork, Provider
  changes, compaction drift, and rejected/expired handles fall back to the complete
  local transcript without replaying a tool.
- Added one immutable `TurnSnapshot` for every model transaction. It freezes the model
  reference, protocol, transaction-complete History view, Tool definitions, runtime
  capabilities, absolute deadline, explicit pricing, and remaining step/token/cost,
  Tool-concurrency, and Subagent budgets used by both model projection and dispatch.
- Added typed completion assessment. Product agents can accept a proposed final answer,
  reject it with durable feedback for another turn, or classify it as blocked; new
  runs distinguish `completed`, `blocked`, step/time/token/cost budget exhaustion,
  cancellation, and failure.
- Added explicit `ModelPricing` plus Run and Task limits for cost, Tool concurrency, and
  Subagent count. Journal resume restores accumulated provider token usage and calculated
  cost before another turn is admitted.

- Added one process-safe writer lease per Run and a disposable SQLite Journal read
  projection. JSONL remains the only canonical source; stale, corrupt, missing, or
  unsupported projections rebuild without changing canonical recovery semantics.
- Added revisioned, read-only per-turn `ToolExposure` snapshots. The model projection
  and action dispatcher now share one exact tool surface, while completed Journal
  model transactions retain its names, schema digest, and application selection
  metadata for replay audits.
- Added stable `JournalRecordRef` locators and read-only committed Tool transaction
  lookup. Open JSONL journals rebuild the query view from canonical records, return
  only terminals published by `step.committed`, and preserve origin references across
  forks without creating another persistence source.
- Added immutable `ModelCapabilities` snapshots for configured adapters. OpenAI
  Responses, Anthropic Messages, and compatible Chat Completions now report only
  tested transport facts such as native tools, reasoning replay, usage/cache
  reporting, and multimodal input. Responses additionally reports its guarded
  continuation contract; hosted tools remain disabled until their runtime contracts
  are complete. Completed model transactions now
  normalize token counts into typed `ModelUsage` while retaining the lossless
  provider usage mapping for cache, trace, and compatibility consumers.
- Added a durable per-Run JSONL journal for canonical model/tool transactions,
  incremental state commits, terminal resume, and independent committed-boundary
  forks. Tool execution is permitted only after `tool.started` is durable, while
  terminal results are finalized once before persistence and reduction.
- Added typed `AgentModule.finalize_action_result` and `reduce_action_result`
  contracts, including ordered access to prior durable action results for products
  that build evidence-backed domain state.
- `AgentModule.mcp_servers` now forms a complete opt-in Engine lifecycle: an empty
  list is inert, while configured servers connect after preflight, expose bounded
  `mcp__server__tool` names for the first model turn, and close at run end.
- Added the read-only `Engine.last_checkpoint_id` lifecycle property so hooks can
  detect durable checkpoint advancement without depending on Engine internals.
- Added read-only bundled Skill roots to `SkillToolSet`, with atomic validation,
  stable bounded `list_skills` summaries, and exact-name full-content `load_skill`
  disclosure independent of provider search and installation.
- Bundled Skills now expose immutable content revisions and relative resources.
  `read_skill_resource` provides root-bounded UTF-8 reads with content-bound paging
  cursors so applications can persist and revalidate progressive Skill disclosure.
- Added an optional explicit process-environment snapshot for host command
  capabilities and `RunCommand`, applied consistently to shell, argv, and background
  subprocess paths while preserving inherited-environment behavior by default.
- Added a typed lightweight `WorkPlanState`, pure ordered reducer, checkpoint codec,
  deterministic Markdown projection, and mutation-free `update_plan` tool. The coding
  preset now uses this contract instead of unchecked `todo_write` metadata.
- Added typed `ArtifactRef`/`ArtifactStore` contracts and a workspace-relative
  `FileArtifactStore` for complete oversized tool outputs.
- Added a qita same-spec comparison preflight that checks stable model, prompt, tool,
  environment, context, budget, source, run-spec, and experiment provenance before
  presenting outcome deltas as repeat-comparable.
- Added absolute monotonic run deadlines and live `remaining_seconds`,
  `deadline_monotonic`, and `agent_cancelled` accessors to tool runtime context.
- Added backend-neutral `CapabilityEnv` composition plus bounded filesystem and
  fixed-argv process contracts for tools that run on host, container, or remote
  application providers without per-tool backend adapters.
- Added a compact environment-backed coding workspace profile with bounded reads,
  exact edits, `rg` glob/grep, binary hex inspection, listings, and directory creation.
- Added provider-neutral managed `web_fetch` capability/tool support with an optional,
  explicitly configured Kimi adapter, public-initial-URL validation, and bounded text
  results. The selected model never implies a fetch service endpoint.
- Added run-scoped `RuntimeInput` delivery and explicit idle wait/wakeup. Background
  work can wake an Engine at the next model-safe boundary without polling, advancing
  steps while idle, or fabricating a second tool result.
- Added a bounded asynchronous model transport policy with one retry owner,
  provider retry-hint handling, typed exhaustion errors, absolute deadlines, and
  stream event-idle timeouts.
- Added generic `model_summary` projection for native tool-call history and
  model-visible observations. Tools can now retain full structured evidence
  for reducers and replay while supplying a bounded readable result to models.
- Added transient runtime-context delivery to `MessageBuildResult`. Custom
  agents can fold authoritative controller state into the final real tool
  result without persisting a synthetic user turn.
- Added first-class OpenAI Responses API support alongside Anthropic Messages and the
  compatible Chat Completions channel, including structured output-item preservation,
  `call_id` tool-result correlation, stateless tool-round replay, and privacy-safe trace
  summaries.
- Added `AgentSpec.tool_name` so delegate workers can expose task-oriented model-facing tool names while keeping the registry agent name stable.
- Added qita's trajectory analysis workbench with diagnosis-first run pages, derived failure insights, focus navigation, critical-step guidance, an inspector panel, and expandable full-content evidence views for long thoughts, observations, parser diagnostics, actions, and critic outputs.
- Added qita `step_interactions`, a derived action-observation view that pairs each action with its complete arguments, invocation metadata, model-visible result, and canonical raw result while separating environment-only and unmatched evidence.
- Added a qita light/dark theme system with a persistent toolbar toggle across board, run detail, replay, and comparison pages.

### Changed

- Journal replay now always parses and validates canonical JSONL before consulting the
  disposable SQLite projection. A projection is retained only when its record digests
  match the validated JSONL; drift rebuilds the projection and can never override replay.
- `Engine.arun()` and `Engine.astep()` now share one immutable turn transaction instead
  of duplicating decide/act/reduce behavior. Run lifecycle stays in Engine, parser and
  compatibility interpretation live in a composed decision runtime, and optional
  critic/handoff policies no longer occupy the outer loop.
- The canonical action path is now async from Engine through Tool, Mailbox, MCP, and
  Subagent execution. Blocking host and Docker commands use asyncio subprocesses with
  process-group cleanup; explicitly concurrency-safe calls remain parallel and commit
  terminal results in source order.
- Cancellation now drains every started Tool handler before publishing its terminal
  result. A direct caller `Task.cancel()` is re-raised only after terminal persistence
  and runtime cleanup, while controlled Engine cancellation returns an auditable
  cancelled result without leaving event-loop tasks behind.
- `ToolExposure` now freezes each Tool definition while retaining one live handler
  owner, so stateful Tool lifecycle and Run-scoped Subagent accounting cannot be copied
  away between lookups or turns.
- Runtime input is accepted through a thread-safe async inbox and projected only at the
  next pre-turn safe point. MCP transports and background Subagent tasks stay on their
  owning event loop; the temporary-loop MCP bridge and daemon action pool were removed.
- Journal records now deep-copy nested payloads at construction and projection
  boundaries. Replay and committed-tool queries return isolated values.
- Engine run and resume entry points now own and close their configured Journal on
  success, failure, or cancellation. Journal fork closes the source and transfers an
  open child Journal to the caller.
- Applications can assign tools to exposure `group` values and override
  `AgentModule.build_tool_exposure()` to select the current turn's tools. A configured
  `PermissionPipeline` now owns parameter-level allow/deny/ask admission instead of
  also applying the static `needs_approval` fallback.
- Tool calls now use one strict JSON Schema at model projection and execution.
  Generated argument objects reject undeclared fields by default; missing fields,
  wrong types, enums, and bounds fail before custom validation or handler execution
  and commit an `invalid_tool_arguments` terminal result with `executed=false`.
  Tools that intentionally accept arbitrary root keys must declare
  `additionalProperties` explicitly.
- Model transports now publish non-terminal text, reasoning, and tool-call deltas
  immediately. Retries remain available before the first provider event, but an
  observable attempt is never replayed; its terminal chunk is committed only after
  provider EOF confirms that no late event follows it.
- Synchronous history retrieval no longer calls the async `Model` contract as a
  function during compaction. It uses the bounded heuristic projection and records
  `summarizer_mode=heuristic_async_model`; explicit synchronous summarizers keep their
  existing failure and circuit-breaker semantics.
- Journal-backed action execution is async-native at the Engine boundary and retains
  explicit `concurrency_safe` parallel segments with input-ordered terminal commits.
- Journal state persistence uses deterministic JSON Patch deltas plus full snapshots
  at initial, periodic, terminal, and fork-safe boundaries. Darwin uses
  `F_FULLFSYNC`; other POSIX systems prefer `fdatasync`.
- **Breaking:** Removed the `read_only` and `allow_destructive` model arguments from
  `run_command`. Applications that need command admission enforce it before execution.
- Oversized tool results now retain canonical output for reducers and traces while
  History, hooks, and model-visible observations share one durable bounded replacement.
  Checkpoint resume reuses that recorded replacement instead of truncating again.
- Fresh Engine runs now persist an `input` checkpoint after task, state, and history
  initialization but before the first provider request. Step checkpoints descend from
  that boundary, and resuming it retries step zero without creating a second input
  checkpoint.
- **Breaking:** Checkpoints now have one asynchronous persistence owner. Engine waits
  for state, task, complete model history, and lineage to reach the configured store at
  each safe step boundary; SQLite operations run off the event loop and settle before
  cancellation propagates. Fork and checkpoint-store APIs are async-only.
- **Breaking:** Removed the legacy JSON `CheckpointManager`, background durability
  queue, unused pending-write manager, ineffective state-version tracker, and legacy
  tracing bridge. Persistence failures are no longer dropped or reported as success.
- Trace runtime events now commit through the existing `TraceWriter` at one completed
  step boundary: event payloads flush before the step marker, while cancellation and
  other lifecycle events remain immediately visible.
- Checkpoint history snapshots now detach nested provider items and reject orphan or
  incomplete tool transactions before they reach durable storage. SQLite updates keep
  one checkpoint row per id and enforce thread-scoped reads, listing, and deletion.
- Memory and SQLite checkpoint stores now share one JSON boundary, return independent
  values, and reject cross-event-loop reuse.
- **Breaking:** `SubagentTool` now requires one explicit `invocation_factory` and uses only
  the canonical `execution_mode`. Removed the class registry, generic model/workspace
  construction, hidden worktree argument, `allow_background` alias, and the separate
  `CodingToolSet.agent_spawn` loop; applications own fresh Subagent Engine construction.

- Model calls now use one async-native Engine path and one Engine-scoped absolute
  request deadline. Provider connection, stream-idle, and retry waits are clamped to
  live remaining time; immediate cancellation propagates through the active task and
  closes the provider stream. Official providers implement the same asynchronous model
  contract instead of maintaining synchronous and asynchronous transports.
- Tool actions now have one absolute budget covering admission, invocation retries, and
  backoff. `ToolSpec.retry_policy` is the only tool retry owner, validation and
  permission checks run once, HTTP transport retries are disabled, and bounded daemon
  workers prevent blocked admission or concurrent-drain paths from owning process exit.
- Built-in coding search now uses one fixed-argv `rg` boundary with NUL-delimited file
  paths, stable ordering, strict result limits, explicit hidden/ignored-file controls,
  reconstructable context records, and structured exit, timeout, and launch failures.
- Reasoning effort now resolves through model-specific preset capabilities across sync,
  async, and streaming request defaults. GPT-5.6 accepts `max`; older OpenAI models keep
  their existing `xhigh` ceiling.
- Context control now measures the complete provider input, including native tool
  schemas and response schemas, and forces transaction-safe compaction at 80% of
  the provider-safe input budget. `CompactHistory` uses three bounded levels
  (microcompact, recent-round summary, and all-but-latest-round resummary), applies
  summaries only to immutable projections, reuses exact-prefix summary checkpoints,
  and bounds repeated summary failures with a three-attempt circuit. Overflow recovery
  uses `.70/.50/.35` history budgets without mutating canonical history. Responses
  text/tool payloads and generic/native mirrors now keep one complete, accurately
  counted call/result transaction.
- `Engine.arun()` and `Engine.astep()` are now the canonical execution paths. The
  synchronous `run()` and `step()` entry points only bridge at the process boundary and
  reject calls from an active event loop; the daemon-thread `AsyncEngine` bridge has
  been removed.
- Background `SubagentTool` runs now snapshot parent history at launch but defer model,
  Engine, and trace construction until an execution slot opens. Event-loop-owned tasks,
  awaited close, canonical cancellation stops, and terminal-before-wakeup ordering keep
  Subagent teardown structured and observable. Terminal
  Subagents retain their queryable result while active task, request, Engine, and
  cancellation records are reaped immediately.
- Clarified the generic `SubagentTool` model contract: independent multi-step tasks can be
  delegated in one response for concurrent execution, while dependent steps and cheap
  mechanical variants remain in the parent. Explicit tool guidance is no longer replaced
  by the `execute()` implementation docstring during initialization.
- Model calls use one live streaming transaction. Connection and pre-event failures
  may retry within the QitOS-owned attempt budget and absolute request deadline;
  observable attempts are never replayed, and active streams use a bounded event-idle
  timeout.
- OpenAI-compatible clients now disable OpenAI SDK retries on paths where QitOS owns the
  retry budget, preventing multiplicative retry delays.
- Raised the optional OpenAI SDK floor to `openai>=1.66.0` and taught compact history to preserve active Responses function-call rounds atomically.
- Strengthened the CyberGym PoC agent's task bootstrap with lightweight structured task-spec extraction and more relevant repo evidence ranking.
- Clarified candidate provenance and lightweight failure taxonomy handling in the CyberGym agent without changing its single-agent runtime architecture.
- Improved qita diagnostics for CyberGym-style traces so budget stops are marked as review-needed, `submit_poc` verification failures are promoted as critical inspection steps, and low-frequency metadata stays out of the default attention path.
- Redesigned qita step stories around `Input -> Thought -> Action Calls -> Environment Observation`: multi-action calls now render as numbered paired units with status, latency, parameters, and their own result; failed calls expand by default, successful calls fold, and all long evidence remains available in wrapped, copyable code views and call-aware Inspector tabs.

### Fixed

- Fixed the documented contributor pytest gate so asynchronous tests and OpenAI
  Responses, transport retry, and reasoning tests install their required checker and
  optional SDK dependencies instead of failing on a clean environment.
- Fixed workspace path validation so lexical parent traversal is rejected while
  intentional workspace-owned symlinks retain their documented behavior.
- Fixed streaming completion being flattened to a synthetic `stop`. Chat,
  Responses, and Anthropic adapters now preserve provider finish reasons, async Chat
  retains incremental and completed tool calls, incomplete streams fail explicitly,
  and rich Engine handlers can observe normalized chunks or failures without receiving
  a false normal-end callback.
- Fixed qita tool statistics attributing one failed result to every action in the same
  step. Statistics now use the canonical action/result pairing, retain exact lifecycle
  counts, and expose unmatched actions or results as trace-closure gaps.
- Fixed model requests and streams outliving the Engine deadline, late responses being
  accepted as successful decisions, async Responses completion bypassing QitOS retry,
  Azure retaining SDK retries, and provider attempts reusing stale timeout values.
- Fixed action-level timeouts being resolved before approval and permission work, retry
  attempts each receiving a fresh timeout, and parallel execution waiting indefinitely
  for a non-daemon executor to drain.
- Fixed forced compatible-Chat tool calls sending contradictory reasoning controls,
  official Responses requests omitting encrypted continuation fields, streamed output
  item data being overwritten by `response.completed`, and reasoning-only content being
  promoted to a visible final answer.
- Fixed runtime deadlines being checked only at Engine step boundaries. The effective
  deadline now clamps tool admission, execution timeout, retry backoff, and runtime
  waits. Synchronous compatibility tools run off-loop and are awaited to a known
  side-effect boundary. Saturated event queues now report dropped deliveries, retain
  exactly one priority `run_end` before the close marker, and terminate late subscribers.
- Fixed native-capable agent turns so provider `tool_calls` are authoritative before
  custom text interpreters or parsers, API-delivered tools no longer receive a second
  framework text action contract, malformed native arguments produce a paired recoverable
  error without executing the tool, and mixed executable/blocked batches commit exactly
  one tool result per call id in the model's original order.
- Fixed lower-level OpenAI-compatible transport errors such as read-timeout and TLS
  exceptions being treated as non-retryable; they now use the existing bounded model
  retry policy instead of immediately terminating the agent loop. Unsupported stream
  options now fall back only after an explicit provider rejection and remain disabled
  for the rest of that logical request.
- Fixed structured tool lifecycle results being flattened to success or generic error.
  Canonical results now retain partial, running, skipped/denied, input/approval, timeout,
  and cancellation through observations, history, traces, summaries, and metrics;
  unknown and legacy alias statuses fail closed, flattened legacy fields are removed,
  and domain outcomes no longer overload execution status.
- Fixed bound function tools and fail-closed coding profiles mutating shared permission
  metadata across toolset instances.
- Fixed immediate cancellation finalization so the END event, canonical State, `TaskResult`/`EngineResult`, and trace manifest all report `cancelled_immediate`; cancelled manifests now use the existing terminal `stopped` status instead of `completed`.
- Fixed native text fallback so malformed structured action output enters parser recovery instead of being misreported as a successful final answer, while ordinary natural-language conclusions still use `native_text_final`.
- Fixed message-window trimming so native tool results whose declaring assistant call has been evicted are removed before provider dispatch, while complete tool chains and existing interrupted-call recovery remain unchanged.
- Fixed direct `Engine(agent=...)` construction so models created with `build_model_for_preset(...)` retain their declared protocol and native API tool-schema delivery, including provider aliases such as Kimi K3 that cannot be inferred from the model name alone.
- Fixed empty model responses with neither usable text nor tool calls being misclassified as parser `wait` decisions. The Engine now records them as `model_error`, retries once through bounded recovery, and stops with `unrecoverable_error` if the empty response repeats while preserving response diagnostics in traces.

- Fixed native response text extraction so OpenAI-compatible messages with null content no longer become repr-string final answers.
- Fixed OpenAI-compatible forced tool-call requests so conflicting thinking options are disabled, and repaired JSON/tool-call parsing for bare control characters inside string values.
- Fixed JSON-like object extraction so apostrophes in surrounding natural-language text no longer hide valid JSON payloads.
- Fixed `DelegateTool` context delivery so the optional tool-call `context` object is passed into the subagent via `Engine.run(..., context=...)`.
- Fixed OpenAI-compatible tool schema generation for postponed or string annotations so CyberGym tools no longer emit invalid JSON Schema types.
- Fixed CyberGym batch trace/result/render redaction so API keys and auth token markers are scrubbed before persisted artifacts are written.
- Fixed CyberGym PoC generation runs so benchmark-local Bash commands can run without interactive command review while the default coding toolset review guard remains intact.
- Fixed tool registration with name overrides so CyberGym uppercase aliases do not mutate source tool specs shared with ordinary coding toolsets.

### Removed

- Removed `AsyncEngine`, `AsyncModel`, synchronous provider calls, `call_raw`, provider
  import-time registration, global model registries, and the parallel sync/async model
  class hierarchy. Model construction is explicit and instance-scoped.
- Removed the duplicate LM Studio and vLLM transports. Compatible endpoints configure
  the canonical OpenAI-compatible adapter; Ollama and Azure keep the transport behavior
  required by their native asynchronous clients.
- Removed the legacy official OpenAI sync/async request implementations, including their
  hard-coded three-attempt loop, implicit SDK retries, and error-as-success strings.
- Removed provider-failure-as-model-text paths from the Anthropic, Gemini, LiteLLM,
  Ollama, LM Studio, and vLLM adapters; transport failures now enter Engine recovery.
- Removed legacy Cookiecutter mocks that depended on an undeclared optional package and
  a Docker write test for the deleted shell-string transport.
- Removed unused `ActionKind`, action-level timeout/retry/idempotency/classification
  fields, integer `max_retries` tool metadata, nonfunctional functional-task retry and
  timeout options, and the duplicate ToolInterceptor middleware (including broken cache
  and retry interceptors). Engine hooks and runtime events remain the observation path.
- Removed class-tool `run`/`call`/callable adapters, `ToolRegistry.call()`, automatic
  short and separator aliases, normalized-name dispatch, duck-typed executor fallbacks,
  callable approval flags, read-only concurrency inference, and the historical
  concurrency-safe name list. Class tools now implement only
  `execute(args, runtime_context)`; the registry performs exact-name lookup and the
  executor owns the complete invocation lifecycle.
- Removed duplicate uppercase, `*_v2`, and historical editor/search built-in tools.
  Coding profiles now expose only the lowercase canonical names such as `read_file`,
  `edit_file`, `glob`, `grep`, `run_command`, and `web_fetch`.
- Removed the compatibility-only compact-history message-slicing option. Recent history
  retention is configured only through complete rounds; explicit count eviction remains
  available through `hard_window`.

### Breaking

- Custom `SessionJournal` implementations must provide async `close()`. A concrete
  Journal instance is terminal after close and cannot be reopened or mutated.
- Model implementations must provide the asynchronous `Model.stream(...)` contract.
  Applications in an event loop must call `Engine.arun()` or `Engine.astep()`; the
  synchronous wrappers no longer emulate async execution with worker threads.
- Direct class-tool callers must pass an argument dictionary to `execute(...)`.
  Registry callers must resolve an exact canonical name with `get()` or execute through
  `Engine`/`ActionExecutor`. Tools run in parallel only when their spec explicitly sets
  `concurrency_safe=True`. A `FunctionTool` remains directly callable as the ordinary
  host-side behavior of `@function_tool`; agent execution never uses that shortcut.
- `CodingToolSet` no longer accepts `expose_legacy_aliases` or
  `expose_modern_names`. Callers must use the canonical lowercase tool names.
- `CompactConfig` and `CompactHistory` no longer accept the obsolete message-count
  retention keyword. Callers must configure transaction-safe retention with
  `keep_last_rounds`.

## v0.6.0 (2026-05-28)

### Added

- **WebBrowserEnv**: Playwright-backed web browser environment (`qitos.kit.env.web`) with `MockBrowserProvider` and `PlaywrightBrowserProvider`, extending desktop GUI actions with `navigate`, `go_back`, `go_forward`, `switch_tab`, `close_tab`. Optional dep: `pip install qitos[web]`
- **qita Screenshot Strip**: Interactive horizontal thumbnail strip at the top of run detail pages, showing one thumbnail per step with screenshot. Click thumbnail to scroll to step card. Grounding failure and critic retry indicators.
- **qita Action Overlay**: Click/action markers on screenshots with coordinate labels. Red markers for grounding failures, green for success. Navigate actions shown with URL badge.
- **qita Observation Pack Viewer**: Expandable per-step panel showing DOM, accessibility tree, OCR spans, UI candidates, and grounding metadata. Toggle with "observation pack" button.
- **qita Branch Comparison**: `/compare-branches/{run_id}/{step_id}` route for side-by-side branch candidate comparison with grounding failure banner.
- **MultimodalCapabilityProfile**: Model-aware observation adaptation in `qitos.models.profile_registry`. Vision models receive screenshots; text-only models receive DOM + OCR fallback.
- **AgentSpec.model_override / tools_override**: Override the sub-agent's model and tool registry for delegation.
- **AgentSpec.__post_init__ validation**: Empty name raises ValueError.
- **AgentRegistry.get_handoff_tools()**: Returns `HandoffTool` instances for Decision-mode handoff.
- **DelegateTool nested delegation fix**: `_build_sub_engine()` now passes `agent_registry` enabling depth-2+ delegation.
- **DelegateEventInterceptor**: First-class `DELEGATE_START`/`DELEGATE_END` events in `EngineResult.events` when `agent_registry` is provided.
- **Sub-trace writer depth-aware run_id**: `f"{parent_run_id}__delegate_{agent_name}_depth{depth}"` prevents collisions.
- **ReviewerAgent** in delegate example demonstrating multi-delegation with `ContextStrategy.SUMMARY`.
- **v0.7 handoff scope document**: Documents what is in v0.6 vs v0.7 scope for handoff/Decision mode.

### Changed

- `DelegateTool._build_sub_engine()`: now passes `agent_registry`, applies `model_override`/`tools_override` from `AgentSpec`.
- `DelegateTool._build_sub_trace_writer()`: includes `current_depth` in sub-run-id for uniqueness.
- `qita renderActionOverlay()`: now shows grounding failure banners inline.
- Engine auto-registers `DelegateEventInterceptor` when `agent_registry` is provided.

## v0.5.0 (2026-05-27)

### Added

- Added `CORE_BOUNDARY.md`, a core governance audit, a dependency audit, and a staged `qitos-zoo` migration manifest for product-grade agents.
- Added regression tests for public API and examples governance.
- Added `FamilyPreset.override()` for programmatic preset customization and `recommended_models`, `recommended_protocol`, `recommended_parser` advisory fields.
- Added `MaxTokensCriteria` stop criterion so engines can halt when accumulated output tokens exceed a budget.
- Added `CriticTrace` and `HandoffTrace` export APIs for programmatic access to critic decisions and multi-agent handoff data.
- Added `EngineConfig` export API for inspecting engine configuration outside the engine runtime.
- Added `ToolPermissionSpec` for declarative tool permission policies.
- Added `WandbTraceProcessor` for W&B experiment tracking integration (`pip install qitos[wandb]`).
- Added `MlflowTraceProcessor` for MLflow experiment tracking integration (`pip install qitos[mlflow]`).
- Added qita cost panel showing token usage and cost metrics in the run overview.
- Added `qit --version` and `qita --version` CLI flags.
- Added `qit new --template <name>` CLI for scaffolding new agent projects from built-in cookiecutter templates.
- Added `qit list-templates` CLI for listing built-in scaffold and method templates.
- Added 5 method template recipe implementations:
  - `qitos.recipes.self_refine` — Self-Refine pattern (generate → critique → refine)
  - `qitos.recipes.reflexion` — Reflexion pattern (act → reflect → retry with memory)
  - `qitos.recipes.lats` — LATS pattern (Monte Carlo tree search with UCB1 scoring and reflection)
  - `qitos.recipes.moa` — MoA pattern (parallel proposals + aggregation layers)
  - `qitos.recipes.magentic_one` — Magentic-One pattern (orchestrator + specialist workers with stall detection)
- Added 12 method template directories under `templates/` with `paper.md`, `config.yaml`, `agent.py`, and `__init__.py`:
  - react, plan_act, swe_agent, voyager, debate, manager_worker, planner_executor, self_refine, reflexion, lats, moa, magentic_one
- Added eval config YAML files for LATS, MoA, and Magentic-One under `qitos/recipes/benchmarks/eval_configs/`.
- Added bilingual method-templates guide covering all 12 templates with quickstart code, parameters, and state fields.
- Added LATS, MoA, and Magentic-One terms to bilingual glossary.
- Added `cookiecutter` optional extra (`pip install qitos[cookiecutter]`).

### Changed

- Tightened QitOS public/default surfaces around kernel-first contracts and moved product-grade agent positioning toward `qitos-zoo`.
- Updated examples policy so canonical examples are teaching-first and product-like agents are marked for migration.
- Refreshed README.md with v0.5.0 content: 12 method templates table, `qit --version` in quickstart, Beta status, optional extras, and method-templates guide link.

### Fixed

- Restored engine final/wait lifecycle behavior so reduce, parser feedback, hooks, checkpoints, and memory records are preserved.
- Fixed `_TEMPLATES_DIR` path resolution in `qit new` so template directories at repo root are found correctly.

## v0.4.0 (2026-05-13)

### Added

- Added `qitos.cache` package with `CacheBackend` ABC, `InMemoryCache` (LRU + TTL), `DiskCache` (file-per-key), and `CachedModel` wrapper that transparently caches any `Model` instance — zero Engine changes required.
- Added `qitos.config` package with `AgentConfig`, `ModelConfig`, `DatasetItem`, `load_agent_config()` for YAML-driven agent setup with `${ENV_VAR}` resolution, and `build_model()`, `build_run_spec()`, `build_tool_registry()` builders.
- Added `qitos.checkpoint` package with `CheckpointData` and `CheckpointManager` for run persistence and resume support. Engine auto-saves checkpoints at configurable intervals.
- Added `qitos.experiment` package with `ExperimentRunner`, `ExperimentResult`, `SweepSpec`, and `sweep_product()` for parameter-sweep experiments with concurrent execution, resume support, and result persistence.
- Added `EngineResult.run_id` field so callers can track run identity after engine execution completes.
- Added `qit experiment run --config <yaml>` CLI subcommand for experiment execution from YAML configs.
- Added `AsyncEngine` with `arun()` and `arun_stream()` methods for non-blocking agent execution inside `asyncio` event loops.
- Added `EngineEvent`, `EngineEventType`, and `EventStream` for structured real-time event streaming from engine runs.
- Added `AsyncOpenAICompatibleModel` and `AsyncOpenAIModel` with `_acall_api()` and `acall_raw()` using `openai.AsyncOpenAI`.
- Added SSE endpoint `/api/stream/{run_id}` to qita for streaming run events as Server-Sent Events.
- Added "live stream" button to qita run detail page for real-time event viewing.
- Added bilingual third-party benchmark integration guidance explaining the official `framework / benchmark / recipe` boundary, required family package structure, normalized result expectations, and qita/trace compatibility rules for future benchmark contributors.
- Added a new `qitos.benchmark.osworld` family with dataset adapter, runtime hook, evaluator bridge, scorer, and built-in runner entrypoints for the real OSWorld benchmark path.
- Added a new `qitos.recipes.desktop.osworld_starter` recipe layer so the canonical desktop baseline can be reused by examples, benchmark runners, and docs without depending on `examples/`.
- Added the first official `desktop` benchmark family as an OSWorld-compatible starter path, including committed starter tasks and built-in `qit bench` support.
- Added lightweight `ActionSpace` and `EnvironmentAdapter` multimodal abstractions so the desktop benchmark path is backed by stable framework types instead of example-local glue.
- Added a benchmark-grade upgrade for `examples/real/openai_cua_agent.py`, including planner/grounding/action-selector workflow guidance, a desktop grounding critic, and richer family-first harness integration.
- Added qita screenshot timelines, replay screenshot previews, basic action overlays, grounding visibility, and step-level visual summaries for desktop runs.
- Added bilingual v0.5 desktop benchmark docs, qita GUI-failure tutorials, and a short release note explaining the OSWorld-compatible starter positioning.
- Added a native tool-call decision lane for OpenAI-compatible family presets so Qwen-class endpoints can execute structured `tool_calls` before falling back to text parsers.
- Added bilingual Qwen best-practice docs explaining the native-lane-first harness strategy for `qwen-plus` and other OpenAI-compatible Qwen endpoints.
- Added the first v0.5 multimodal core slice with shared `ContentBlock` / `ObservationPack` abstractions, screenshot-first environment support, and an OpenAI-compatible visual input path for `chat.completions`.
- Added a minimal `ScreenshotEnv`, visual trace asset metadata, qita visual-asset inspection, and a new `examples/real/visual_inspect_agent.py` baseline for screenshot-driven agent workflows.
- Added an OSWorld-inspired desktop/computer-use substrate with `DesktopEnv`, mock and container-first desktop providers, provider-neutral GUI action tools, `ComputerUseToolSet`, and new desktop action protocols.
- Added `examples/real/openai_cua_agent.py` and `examples/real/desktop_env_smoke.py` as the first QitOS-native desktop/computer-use baselines.
- Added a run-scoped structured audit board memory for `examples/real/whitzard_agent.py`, giving the long-running security auditor durable target ranking, failed-search recall, focused-read tracking, and phase-aware convergence hints.

### Changed

- Migrated GAIA, Tau-Bench, and CyBench onto the same `qitos.benchmark.* + qitos.recipes.*` architecture as the desktop starter and OSWorld paths, leaving `examples/benchmarks/*.py` as thin wrappers instead of canonical implementations.
- Changed the canonical starter benchmark name from `desktop` to `desktop-starter` while keeping `desktop` as a compatibility alias.
- Split the desktop / OSWorld story into three explicit layers: framework (`DesktopEnv`, qita, multimodal contracts), benchmark (`qitos.benchmark.*`), and recipe (`qitos.recipes.*`).
- Moved the real implementation behind `examples/real/openai_cua_agent.py` into `qitos.recipes.desktop.osworld_starter`, leaving the example file as a thin wrapper.
- Changed `AgentModule.run()` so structured `Task.env_spec` environments are no longer accidentally overridden by an implicit `HostEnv` when `workspace` is set.
- Changed the desktop runtime to validate GUI actions against a formal action space before execution and to distinguish `executed`, `accepted`, `approval_required`, and failed validation outcomes.
- Changed the unified benchmark summary layer to aggregate desktop failure-tag distributions in addition to stop reasons.
- Upgraded the `qwen` family preset from generic JSON-first compatibility to native-tool-call-first behavior with text parser fallback.
- Preserved OpenAI-compatible raw responses inside the Engine runtime instead of flattening them to strings too early, while keeping direct text-oriented model calls available for existing authoring paths.
- Collapsed the canonical coding tool surface onto one traditional naming scheme, removed duplicated `*_v2` registry aliases, and standardized file-edit parameter names around `path` and `content`.
- Upgraded `examples/real/whitzard_agent.py` to the same preset-first family switching path as the flagship coding example, so long-running security audits can swap model families and harness policies without rewriting the agent.
- Tightened `examples/real/whitzard_agent.py` around a precision-first audit workflow with `CompactHistory`, deterministic target ranking, regex-recovery guidance, and stronger transitions from broad search to focused code reads.
- Upgraded the Engine and prompt/runtime chain so current-step screenshots can flow from task resources or environment observations into multimodal user messages without changing existing parser or tool-schema behavior.
- Extended the multimodal lane into a provider-neutral desktop action path, keeping image input on the OpenAI-compatible multimodal request shape while moving GUI action scaffolding into QitOS protocols and prompt helpers instead of a provider-specific computer-use API.

### Fixed

- Fixed the desktop benchmark path so built-in runs now resolve to the desktop protocol/parser pair instead of inheriting the generic `react_text_v1` CLI defaults.
- Fixed a prompt-plumbing bug where agents overriding `build_system_prompt()` could silently drop API-level tool schemas, causing OpenAI-compatible models to guess tool argument names instead of receiving the real schema.
- Fixed qita step inspection so screenshot-backed runs can display visual assets and model-input modality summaries instead of hiding multimodal state inside raw JSON only.
- Fixed `examples/real/whitzard_agent.py` so family presets remain the protocol authority while inventory results now advance audit progress correctly and the agent no longer exposes `list_files` as an easy low-value fallback during long-running audits.

## 0.3.0 - 2026-04-08

### Added

- Added PR/push CI gates covering tests, packaging validation, stable-surface linting, and stable-surface type checking.
- Added dedicated maturity docs for architecture, development workflow, security reporting, community conduct, and environment configuration.
- Added an explicit `qitos.kit.tool.experimental.security_research` namespace for opt-in security research tool imports and registry builders.
- Added thin module boundaries for `qita` data/server/views and `render` terminal/themes façades to make future maintenance easier.
- Added a root-level changelog to document ongoing project evolution.
- Added a dedicated `requirements-dev.txt` entrypoint for full contributor installs from a local clone.
- Added stable `RunSpec`, `ExperimentSpec`, and `BenchmarkRunResult` public contracts to anchor reproducible-run metadata and normalized benchmark outputs.
- Added a first-pass unified `qit bench` CLI with `run`, `eval`, `replay`, and `export` subcommands.
- Added qita compare/diff views and export routes for summary-level run comparison.
- Added official-run and glossary docs, plus new reproducibility tutorials for benchmark runs and failed-run replay in both English and Chinese.
- Added a blog entry on why reproducible runs matter in QitOS.
- Added a first-class `qitos.harness` layer with `FamilyPreset`, `HarnessPolicy`, `ModelAdapter`, `ToolPolicy`, `ContextPolicy`, `build_harness_policy(...)`, and `build_model_for_preset(...)`.
- Added built-in gold presets for Qwen, Kimi, MiniMax, `gpt-oss`, and Gemma 4, plus bilingual docs for family presets, preset authoring, the model-family matrix, and same-example switching.
- Added `qit demo minimal`, a packaged minimal coding-agent demo that configures a real model, fixes a tiny workspace bug, and leaves behind a qita-ready trace.
- Added release notes for the first formal GitHub release package. Those notes were
  later consolidated into this changelog when completed internal plans were removed.

### Changed

- Dropped Python 3.9 support and aligned CI, packaging metadata, README, and installation docs around Python 3.10+.
- Normalized the class-based tool contract around `execute(args, runtime_context)` while keeping `run(...)` as a compatibility path.
- Removed deprecated editor/codebase/file/shell compatibility shims in favor of the canonical `CodingToolSet` surface.
- Tightened default public exports from `qitos.kit` and `qitos.kit.tool` so experimental and higher-risk tool families are no longer part of the default surface.
- Preserved old security research import paths as short-term deprecation shims instead of keeping them as primary public entrypoints.
- Extracted shared coding-tool helper logic into internal utility modules to reduce coupling inside the canonical coding toolset.
- Slimmed `qita` and `render` entry modules so public behavior stays the same while implementation can evolve behind clearer boundaries.
- Reworked root installation guidance so `requirements.txt` is now a lightweight repo install path instead of a drifting copy of runtime and dev dependencies.
- Added coverage, dependency audit, and pre-commit tooling to the standard contributor workflow.
- Removed legacy root planning/audit scratch files, obsolete MkDocs configuration, and local phase-artifact directories so the repository surface matches the current Mintlify-based docs flow.
- Extended trace manifests with normalized run-spec, experiment-spec, benchmark, parser, and reproducibility metadata instead of keeping benchmark context in ad hoc side channels.
- Reworked benchmark example scripts so GAIA, Tau-Bench, and CyBench wrappers now emit the unified `BenchmarkRunResult` shape and route through the official v0.3 runner contract.
- Surfaced official-run and best-effort replay metadata inside qita board, run detail, and diff views.
- Updated benchmark, tracing, and CLI docs to position `qit bench` as the canonical benchmark path while keeping `examples/benchmarks` as thin wrappers.
- Refactored the flagship `examples/real/claude_code_agent.py` example into a preset-first showcase so the same agent can switch across supported model families without rewriting the agent implementation.
- Moved model-profile defaults onto preset-derived family data and extended context inference for the new v0.4 target families.
- Reworked README, quickstart, installation, CLI reference, and first-agent docs around the minimal coding-agent path so the public “minimal agent” story now matches the QitOS mindset: model config, workspace actions, verification, and qita inspection.
- Updated package metadata and contributor guidance so PyPI, docs, and release materials all describe QitOS as the torch-flavor framework for agent researchers.

### Fixed

- Fixed compatibility issues in direct `.run(...)` calls after the tool execution contract was normalized.
- Fixed the known undefined `target` reference in the exploit payload generation flow.
- Fixed stable-surface lint and mypy failures across `qitos/core`, `qitos/engine`, `qitos/models`, and `qitos/trace`.

### Deprecated

- Deprecated legacy security research import paths under `qitos.kit.tool.*_toolset` and `qitos.kit.tool.security_audit` in favor of explicit imports from `qitos.kit.tool.experimental.security_research`.

### Breaking

- Default root exports from `qitos.kit` and `qitos.kit.tool` no longer include advanced/security-audit convenience surfaces; import those explicitly from their module paths when needed.
