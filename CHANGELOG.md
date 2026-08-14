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

### Removed

- **Breaking:** Removed the fixed `qitos.kit.patterns` Manager/Worker,
  Planner/Executor, Proposer/Verifier, Debate, MoA, and DAG workflow templates.
  They formed a synchronous parallel orchestration layer with no production caller
  inside QitOS. Applications should express coordination policy through the typed
  `AgentTool` and `ChildControlToolSet` lifecycle or in application-owned recipes.
- **Breaking:** Removed the deprecated product-level `SecurityAuditAgent` template and
  redundant security-audit Tool/ToolSet forwarding packages. Security research remains
  available from the explicit `qitos.kit.tool.experimental.security_research` owner,
  which is no longer loaded as a side effect of importing the generic kit.
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
  `AgentModule + Engine` for Agents and the maintained function-tool decorator for
  ordinary callable Tools.
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
  JSONL, while `EngineResult` and `ChildResult` preserve local totals and usage
  completeness for audit. Local Child ceilings further narrow, rather than duplicate,
  the shared Run allowance.
- Canonical `ToolResult` now carries the model ToolCall `call_id`, assigned by the
  Engine before application finalization and reconstructed for older Journal records.
  Child invocation factories may complete async resource construction while existing
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
- YAML Agent configuration now uses private Pydantic input models for strict nested
  validation, unknown-field rejection, positive runtime limits, and standard structured
  validation errors. Public runtime configuration remains framework-neutral dataclasses.
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
- Child invocation factories now receive the already-journaled `ChildHandle` in their
  runtime context, so product runtimes can correlate independent Child sessions and
  traces without deriving identity from mutable task data.

### Breaking

- `CapabilityEnv(attestation=...)` has been replaced by
  `CapabilityEnv(snapshot=RuntimeCapabilitySnapshot(...))`.
  `TurnRuntimeCapabilities` now stores the complete Runtime snapshot; its
  `environment_ops` property remains a derived read-only view. Environment subclasses
  that allocate asynchronously may override `ainitialize()` and `ahealth_check()`;
  legacy synchronous hooks run in a worker thread by default.
- Live MCP transports no longer belong to `AgentModule.mcp_servers`. Applications pass
  `Engine(mcp_server_factory=...)` a construction-only factory that returns fresh,
  unconnected transports for each Run, preventing Root, Child, resumed, or repeated
  Runs from sharing a process, HTTP client, or session.

- `MCPServer.list_tools()` and `MCPServer.call_tool()` now use the official
  `mcp.types.Tool` and `mcp.types.CallToolResult` models directly. The redundant
  `MCPToolInfo`, `MCPToolAnnotations`, and `MCPCallToolResult` mirrors were removed;
  callers should import protocol models from `mcp.types`. MCP bridges still project
  remote results into regular QitOS `ToolResult` values with stable classifications.
- Child invocation factories are now async-only and must resolve to a
  `ChildInvocation`. This makes cancellation and partial-construction cleanup part of
  the same awaited ownership chain instead of a synchronous pre-run side effect.
- Custom `FileSystemCapability` implementations must provide `write_text_atomic()`.
  The method owns same-path mutation ordering, optional SHA-256 preconditions, and the
  final replace boundary; `write_text()` remains the unconditional convenience path.
- `AgentRequest`, `AgentInvocation`, and `AgentResult` have been replaced by the
  immutable core `ChildLaunchRequest`, `ChildInvocation`, `ChildHandle`, `ChildResult`,
  `ChildStatus`, and `AgentConclusion` contracts. `AgentTool` now accepts a complete
  `TaskBudget`, returns parent-scoped handles, and exposes typed `child_result(handle)`
  and `cancel_child(handle)` lifecycle operations. Tool execution status remains
  separate from the child task's `child_status`.
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

- Foreground Child supervision now distinguishes `ChildInvocationCancelled` from an
  actual caller Task cancellation. Both outcomes persist one cancelled Child terminal,
  but only `asyncio.CancelledError` from the caller aborts the parent.
- Caller cancellation after Tool execution now reuses the executor's typed terminal
  `cancel_source` instead of inspecting Python-version-specific Task internals. The
  Engine still commits one terminal result per ToolCall before propagating cancellation.
- Resume now derives idempotent background Child and process completion inputs from local
  terminal facts. A crash between `child.terminal` / `process.terminal` and mailbox
  acceptance no longer loses the notification; foreground Child results are not
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
- The `ChildRunResult.records` contract is now a read-only sequence view, so concrete
  typed `EngineResult` values structurally satisfy the Child Engine protocol.
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

- Added immutable `RunHandle`/`RunStatus` contracts and a lease-free
  `JsonlRunCatalog` for inspect, deterministic listing, validated lineage, and direct
  children. Catalog reads use only an exact read-only SQLite projection or canonical
  JSONL prefix and never acquire ownership, repair, or rebuild storage.
- `AgentTool` composition can now populate the Child profile, allowed Tool groups, and
  working directory in each immutable launch request. `ChildInvocation.cleanup` gives
  the supervisor explicit async ownership of per-Child model and trace resources across
  success, failure, and cancellation.
- Added a reusable Run-owned `ChildSupervisor`. It admits fresh Engines under one async
  concurrency limit, supports non-destructive wait and bounded interrupt, stores
  terminal results before parent delivery, and continues to own delivery tasks until
  shutdown drains them. `AgentTool` is now only the model-facing launch projection.
- Child supervision now journals `child.started` before constructing an Engine and
  `child.terminal` before parent delivery. Recovery marks a started child without a
  terminal as `interrupted` and never replays it; forked Runs do not acquire authority
  over inherited parent handles.
- Added `child_status`, `child_wait`, `child_message`, and `child_interrupt` tools over
  the shared supervisor. Message delivery uses the active Child's async durable mailbox;
  wait timeout preserves execution, interrupt awaits cleanup, and unknown or foreign
  handles return stable state.
- Added canonical Child Agent contracts that keep persisted launch intent, stable
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
  Tool-concurrency, and Child budgets used by both model projection and dispatch.
- Added typed completion assessment. Product agents can accept a proposed final answer,
  reject it with durable feedback for another turn, or classify it as blocked; new
  runs distinguish `completed`, `blocked`, step/time/token/cost budget exhaustion,
  cancellation, and failure.
- Added explicit `ModelPricing` plus Run and Task limits for cost, Tool concurrency, and
  Child count. Journal resume restores accumulated provider token usage and calculated
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
  Child execution. Blocking host and Docker commands use asyncio subprocesses with
  process-group cleanup; explicitly concurrency-safe calls remain parallel and commit
  terminal results in source order.
- Cancellation now drains every started Tool handler before publishing its terminal
  result. A direct caller `Task.cancel()` is re-raised only after terminal persistence
  and runtime cleanup, while controlled Engine cancellation returns an auditable
  cancelled result without leaving event-loop tasks behind.
- `ToolExposure` now freezes each Tool definition while retaining one live handler
  owner, so stateful Tool lifecycle and Run-scoped Child accounting cannot be copied
  away between lookups or turns.
- Runtime input is accepted through a thread-safe async inbox and projected only at the
  next pre-turn safe point. MCP transports and background Child tasks stay on their
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
- **Breaking:** `AgentTool` now requires one explicit `invocation_factory` and uses only
  the canonical `execution_mode`. Removed the class registry, generic model/workspace
  construction, hidden worktree argument, `allow_background` alias, and the separate
  `CodingToolSet.agent_spawn` loop; applications own fresh child Engine construction.

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
- Background `AgentTool` runs now snapshot parent history at launch but defer model,
  Engine, and trace construction until an execution slot opens. Event-loop-owned tasks,
  awaited close, canonical cancellation stops, and terminal-before-wakeup ordering keep
  child teardown structured and observable. Terminal
  children retain their queryable result while active task, request, Engine, and
  cancellation records are reaped immediately.
- Clarified the generic `AgentTool` model contract: independent multi-step tasks can be
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
- Fixed `DelegateTool` context delivery so the optional tool-call `context` object is passed into the child agent via `Engine.run(..., context=...)`.
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
- Added release notes for the first formal GitHub release package under `plans/releases/v0.3.0.md`.

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
