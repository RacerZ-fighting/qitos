# QitOS

<img src="assets/logo.png" alt="QitOS Logo" width="75%">

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-qitor.mintlify.app-0A66C2)](https://qitor.mintlify.app/)
[![PyPI](https://img.shields.io/pypi/v/qitos.svg)](https://pypi.org/project/qitos/)
[![Repo](https://img.shields.io/badge/github-Qitor%2Fqitos-black)](https://github.com/Qitor/qitos)

QitOS is a provider-neutral runtime for building Tool-using agents.

Current releases run the minimal async Agent loop (`Message -> Model ->
ToolCall -> ToolResult`) behind the `Agent` façade. `qita` remains a read-only
viewer for compatible trace artifacts; new façade runs do not yet emit those
artifacts automatically. The retired `AgentModule + Engine` lifecycle,
checkpoint store, prompt-injected parsers/critics, recipes, and benchmark
execution adapters are removed; the changelog records the migration.


[Quickstart](https://qitor.mintlify.app/quickstart) · [Tutorial Track](https://qitor.mintlify.app/tutorials) · [CLI Reference](https://qitor.mintlify.app/reference/cli) · [Changelog](CHANGELOG.md) · [Chinese README](README.zh.md)

## Current runtime

- **Pi-aligned runtime is the mainline**: the `Agent` façade drives the minimal
  async `Message -> Model -> ToolCall -> ToolResult` loop with typed messages,
  one immutable model/Tool snapshot per turn, journaled transactions, and
  façade-driven subagents. The retired `AgentModule + Engine` lifecycle is
  removed. See the
  [target architecture](docs/internal/architecture/agent-runtime.md) and
  [migration plan](docs/internal/plans/pi-aligned-agent-runtime.md).

- **Typed Provider and Tool transactions**: OpenAI Responses, Anthropic
  Messages, and compatible Chat Completions enter one provider-neutral Message
  stream. Tool execution is serial by default, preserves input order under
  bounded parallelism, and produces one terminal `ToolResult` for every call.
- **Explicit persistence boundary**: loop transactions can be written to the
  canonical Session Journal. Session/Harness owns compact, resume, fork, and
  pure recovery; trace artifacts and `qita` remain observational rather than
  recovery truth.
- **Subagents finish with their own words**: every active Subagent reserves one
  shared step for a tool-free final answer. Step, token, or cost exhaustion can
  trigger that same-context conclusion turn; a useful earlier answer releases
  the reservation. The Supervisor transports the model's final text and never
  synthesizes prose from Tool, Plan, or Journal facts. Explicit cancellation
  and expired absolute deadlines remain hard boundaries.

## Historical pre-migration highlights

The entries below describe behavior from the retired v0.6 runtime. They remain
as migration context; names such as `Engine`, `AgentModule`, `Decision`, parser,
critic, checkpoint, and recipe are not current execution APIs.

- **Cache-stable long runs**: canonical model history stays append-only between explicit
  compactions. Engine warns at 80%, compacts at 85%, records Provider cache/prefix facts,
  and appends changed application `ContextSnapshot` revisions without rewriting an old
  ToolResult. Oversized Tool output is persisted first, with an 8K/result and 16K/batch
  model receipt plus a readable Artifact reference.
- **A real final step for bounded autonomous runs**: applications can reserve the last
  budget step for a tool-free conclusion with a deterministic durable fallback.
  Immediate and after-step cancellation use the same durable terminal boundary, so a
  terminal resume returns the saved conclusion without replaying a model or Tool.
  `PermissionMode.AUTONOMOUS` removes approval and shell-shape prompts for authorized
  products while explicit denies, Runtime capabilities, and owner boundaries remain.
- **Provider-hosted Web with a managed fallback**: an admitted `web_search` Tool now
  uses the official OpenAI Responses endpoint, Qwen Responses/Chat, or the official
  Anthropic Messages server-side search contract when that transport declares support. Native search
  items and citations remain in the model transaction. An explicit request-time
  rejection can fall back to the same managed Tool, but observable provider output is
  never replayed. Kimi keeps its `/search` and `$web_search` managed paths, and Qwen
  has a matching `enable_search` managed capability for providers without hosted Tool
  delivery. Managed `web_fetch` projects the requested URL into the same permission
  scope before network access.
- **Reproducible on-demand quality gates**: CI, documentation, and contribution
  validation now run only through explicit GitHub Actions dispatch. The declared
  development environment includes async-test support, and build/audit jobs require a
  `setuptools` release fixed for `PYSEC-2026-3447`.
- **Provider-owned model caching**: the deprecated local response-cache package and
  implicit local model mutation are gone. QitOS no longer replays
  a previous provider transaction as a new model call; provider-native prompt caching,
  cache usage, deadlines, continuation, and trace facts remain authoritative.
- **One runtime configuration path**: the disconnected `qitos.config` YAML builder,
  parallel `qitos.experiment` runner, and `qit experiment` command are gone. They
  bypassed canonical Agent runs, Journals, and traces while silently ignoring most
  declared configuration. Applications now construct Providers and Agents explicitly;
  reproducible benchmark runs continue through `qit bench` and canonical Run specs.
- **One maintained HTML parser**: Text Web, coding Web fetch, standalone extraction,
  and EPUB reading now use the required Beautiful Soup parser through one shared
  policy. Optional-dependency branches and regex HTML fallbacks are gone, and search
  result links are selected from the parsed DOM.
- **One general Subagent lifecycle**: the fixed `qitos.kit.patterns`
  Manager/Worker, Planner/Executor, Proposer/Verifier, Debate, MoA, and DAG workflow
  templates are gone. Applications now compose coordination policy from the typed,
  async `SubagentTool` and `SubagentControlToolSet` lifecycle or keep it in explicit
  application-owned recipes instead of entering a second orchestration runtime.
- **One Skill path and one HTTP client**: QitOS now keeps only progressive, read-only
  bundled `SKILL.md` disclosure in the Agent runtime. The old SkillHub installer,
  mutable registry, self-install Tools, and orphaned search adapters are gone; all
  maintained sync/async HTTP paths use SOCKS-capable `httpx` directly.
- **Canonical tool imports only**: seven deprecated `qitos.kit.tool.*` forwarding
  modules, the product-level security-audit Agent template, and its redundant Tool and
  ToolSet shims are gone. Applications now import maintained tools from their canonical
  owners; experimental security research remains available only through its explicit
  opt-in package and is not loaded by generic kit imports.
- **Removed dead compatibility surfaces**: the superseded in-repo zoo staging,
  Snowl-specific adapters, unused `RunState` snapshot, source-only scaffold CLI, and
  unwired pseudo-worktree helper are gone. Product applications live in the independent
  `qitos-zoo` repository; QitOS recovery uses only canonical Session Journals or
  checkpoint stores.
- **One executable Agent path**: the unused `qitos.func` decorator/compose package is
  gone. Its direct calls bypassed canonical transactions, and its apparent lifecycle
  adapter never ran the wrapped function. Agents now use the `Agent` façade;
  ordinary Python callables continue to use the maintained function-tool decorator.
- **One canonical observability path**: the duplicate process-global tracing provider
  and its W&B/MLflow processors are gone. Runtime events, `TraceWriter`, the Session
  Journal, and read-only `qita` now form one explicit trace and replay path; handoffs
  remain visible through canonical `HANDOFF_START`/`HANDOFF_END` events. The deprecated
  debug package, trace-file “fork,” and snapshot-copy checkpoint fork are gone;
  resumable forks use committed Journal lineage.
- **Truthful Runtime capability snapshots**: initialized environments now expose one
  immutable backend, working-directory, operation-group, facility, verified-command,
  and limitation snapshot. Engine freezes it with each turn, removes Tools whose ops
  are unavailable, and gives execution the same facts. Local host and ordinary-
  container runs use the same managed file, foreground, background, PTY, stdin, and
  cleanup primitives; Docker exec advertises foreground-only process support instead
  of silently detaching untracked work. Runtime-profile probes for Docker now execute
  inside the selected container, and the verified commands are frozen into that
  backend's snapshot.
- **Official MCP lifecycle for full and interactive Runs**: independent servers now
  connect and discover concurrently under fixed limits, then publish successful
  catalogs in deterministic factory order. `Engine.astep()` lazily owns the same MCP
  lifecycle for interactive sessions, while synchronous `step()` rejects MCP-backed
  sessions. Stdio and Streamable HTTP now use the official Python SDK and protocol
  models; one dedicated task owns each SDK context through shutdown while QitOS keeps
  bounded catalog publication and stable ToolResult errors. PTYs use `ptyprocess`, and
  official MCP protocol models retain their Pydantic validation. ToolCalls from
  incomplete or failed model terminals are retained only as diagnostics and never
  execute.
- **Deterministic completion projections**: Background Subagent and managed-process completion
  inputs are deterministic projections of canonical Journal terminals. The retired
  Engine recovery path that re-projected unconsumed inputs has been removed; closing
  the terminal-to-mailbox crash window now remains an explicit authoritative
  Session/Harness migration gate. Canonical
  `ToolResult` values also carry their model `call_id`, and Subagent factories may finish
  async resource construction safely. Foreground Subagent-local cancellation returns a
  terminal Subagent result without being misclassified as parent Task cancellation.
- **Lease-free Run discovery and lineage**: `JsonlRunCatalog` now returns immutable
  typed summaries, deterministic listings, validated ancestors, and direct children
  while an Engine still owns the writer lease. Reads never repair canonical JSONL or
  rebuild its disposable SQLite projection. Inherited committed boundaries support
  nested forks, whose Engine recovery no longer depends on ancestor files or replays
  completed tools. Stable lineage ids survive every fork, and terminal handles expose
  the latest non-terminal continuation boundary so explicit follow-up forks become
  resumable instead of inheriting a completed state.
- **Revision-safe atomic file tools**: bounded reads now return a SHA-256 revision for
  the complete UTF-8 file. Host and Docker filesystem capabilities replace files
  atomically, serialize same-path mutations per environment, and support compare-and-
  swap writes. `edit_file` automatically guards the revision it read, so concurrent
  edits fail explicitly instead of silently overwriting one another.
- **Run-owned managed processes**: host background commands now use one asyncio
  supervisor with opaque typed handles, incremental bounded output, complete UTF-8
  logs, stdin/PTY interaction, process-group cleanup, and exactly one terminal state.
  Runs await every reader and watcher during shutdown, while Journal-enabled starts
  persist `process.started` and `process.terminal` around the live lifecycle. The shell
  profile exposes list/read/write/wait/terminate controls; resume marks interrupted
  ownership as `lost`, and forks do not inherit live handles. A terminal watcher posts
  one durable `process.completed` input after the terminal record, waking an active
  Agent only through its next turn safe point.
- **Discriminated model stream events**: every Provider event now declares whether it
  is text, reasoning, a ToolCall delta, a native output item, usage, lifecycle,
  successful completion, or failure. The ambiguous `ModelStreamChunk` field bag is
  gone; Engine treats `FAILED` as an error terminal and commits only `COMPLETED`.
- **Recoverable model requests and guarded continuation**: the Engine now sends one
  immutable `ModelRequest` and journals its exact credential-redacted snapshot.
  Responses reuses `previous_response_id` only when the Run, Provider, model,
  protocol, request settings, and canonical input prefix all match. Resume may keep
  that optimization; fork, Provider changes, compaction drift, or an expired handle
  automatically use the complete local transcript. Recovery coverage now follows a
  multi-turn Run through compaction, cancellation, committed-boundary fork, and resume
  while proving that the canonical ToolCall/ToolResult transcript remains complete.
  A crash after every Tool terminal but before the step commit now also retains the
  matching continuation when the step is reconstructed.
- **One async turn transaction**: every model turn now captures an immutable model,
  protocol, complete History, Tool exposure, capability, deadline, pricing, and budget
  view. Full runs and interactive steps share this one transaction; parser and optional
  critic/handoff policies compose around it instead of adding another loop. Engine,
  Tool, Mailbox, MCP, and Subagent calls stay on the caller's event loop; cancellation
  drains started handlers and journals one ordered terminal result per call before it
  propagates. Model changes re-resolve inferred protocol only for the next turn.
- **Product-owned completion and bounded Runs**: `AgentModule.assess_completion()` can
  accept a final answer, request another evidence-gathering turn, or classify a concrete
  blocker. Runtime and Task budgets now cover steps, time, tokens, cost, Tool
  concurrency, and Subagent count, with shared step/token/cost usage restored on Journal
  resume.
- **One Root/Subagent budget ledger**: product runtimes can pass one `BudgetLedger` to
  descendant Engines. Every model request first reserves one step in the Root JSONL;
  every completed model transaction then settles token and cost exactly once. The
  reservation is atomic across concurrent Subagents, so the lineage cannot issue more
  provider requests than its shared step limit. Subagent budgets remain local caps inside
  that allowance. Each live Subagent also holds one step for its final natural-language
  answer, so ordinary Root or sibling work cannot consume the last chance to report its
  result. Results expose both shared and local totals plus completeness. Token and cost
  enforcement still settles after a response; unknown usage is not pre-reserved, but a
  crossing may use the held tool-free conclusion turn.
- **Single-owner Session journals**: each Run now has one process-safe JSONL writer
  lease and an explicit terminal lifecycle. Replay always validates canonical JSONL;
  the disposable SQLite read projection is retained only when it matches JSONL and
  rebuilds after drift or corruption. Payloads cross one strict JSON boundary before
  append, unsupported schemas report an upgrade error, and failed forks do not leak an
  unreturned child writer.
- **Frozen per-turn tool exposure**: each model request and its action dispatcher now
  share one revisioned, read-only `ToolExposure`. Applications can select by tool
  group or policy, later registry changes wait until the next turn, and the Journal
  records exact names plus a schema digest for replay audits.
- **Strict tool inputs share one authority**: the exact JSON Schema projected to the
  model is now enforced before every handler. Generated root objects reject unknown
  fields by default, and missing fields, wrong types, enums, or bounds produce a
  durable unexecuted terminal result instead of reaching custom tool code.
- **Resolvable committed tool transactions**: canonical Journal terminals now have
  stable `JournalRecordRef` locators. An open JSONL Journal can reconstruct a fresh,
  typed `ToolTransaction` for a committed terminal; forks retain the origin locator,
  and uncommitted or unrelated records fail closed without a second database.
- **Local model latency telemetry**: every completed Engine model transaction now
  records monotonic time to first provider event, first text/reasoning/tool content,
  and terminal completion. The typed timing travels with `ModelResponse` and its local
  trace summary without an exporter or an extra provider request.
- **Complete-before-execute tool calls**: compatible Chat and Anthropic Messages now
  publish native calls only after their protocol-specific terminal proves completion.
  Output-limit, unclosed, or malformed calls remain diagnostic and never reach a tool
  handler or provider replay.
- **Complete Responses stream lifecycle**: Responses adapters retain lifecycle and
  item events, refusals, terminal-only text/reasoning, and independently interleaved
  function arguments. Incomplete or contradictory terminal responses cannot become
  executable tool calls.
- **Source-aware token accounting**: Engine context telemetry now distinguishes
  provider counts, estimates, and absent usage. Provider zeroes remain authoritative;
  cache read/write and reasoning subsets stay visible without being double-counted in
  cumulative totals.
- **Truthful model capability snapshots**: configured adapters expose immutable
  `ModelCapabilities`. Responses, Anthropic Messages, and compatible Chat report
  tested native-tool, reasoning/replay, usage/cache, and multimodal behavior without
  claiming unfinished hosted-tool support. Responses also declares its guarded
  continuation behavior. Their terminal token
  counts are normalized into typed `ModelUsage` without discarding provider details.
- **Family and wire can be selected independently**: one Kimi family configuration can
  use compatible Chat Completions or Anthropic Messages without leaking wire-specific
  request defaults across adapters. Local traces retain model identity, finish state,
  typed usage, and usage source while nested credentials remain redacted.
- **Async-safe context fallback**: synchronous `CompactHistory.retrieve()` no longer
  invokes an async `Model` as a callable. It uses a bounded heuristic summary and
  records that mode, preventing context compaction from failing merely because the
  main model follows QitOS's async-native contract.
- **Actually live model deltas**: model transports now publish text, reasoning, and
  tool-call deltas as they arrive. Retries are limited to failures before the first
  provider event, preventing duplicate visible output or tool calls; success remains
  transactional because the terminal chunk is held until provider EOF.
- **Native Anthropic preset reasoning**: Anthropic family presets now build the
  official Messages adapter. Claude 4.5 reasoning effort resolves to a bounded manual
  thinking budget, request defaults reach the wire payload, and thinking requests omit
  incompatible temperature overrides. Native API tool delivery is the preset default.
- **Canonical per-Run journals**: opt an Engine into one durable append-only JSONL
  source for complete model/tool transactions, committed state deltas, crash recovery,
  terminal resume, and committed-boundary fork. Tools execute only after a durable
  permit; ordered terminal results are reduced one by one, and unknown in-flight side
  effects are closed explicitly without replay.
- **Run-scoped MCP tools**: pass `Engine` an `mcp_server_factory` that creates fresh
  transports for each Run, and Engine will connect, discover, expose
  `mcp__server__tool` names, execute calls on the caller's event loop, and unregister
  and close everything at run end. Catalog changes publish atomically at the next
  turn safe point; annotations, pagination, typed remote errors, cancellation, and
  the last good catalog are preserved. HTTP JSON/SSE, resumable GET notifications,
  isolated reconnect cursors, and cancellation-safe cleanup do not replay
  side-effecting Tool calls. The empty default has no startup cost.
- **Public checkpoint lifecycle state**: hooks can read
  `Engine.last_checkpoint_id` after a successful durable commit instead of reaching
  into private Engine fields.
- **Progressive bundled Skills**: applications can point `SkillToolSet` at read-only
  asset roots, expose a bounded catalog, load one exact `SKILL.md` in full, and page
  linked UTF-8 resources without invoking a provider or writing an installation
  registry. Recursive discovery uses explicit root precedence and typed diagnostics;
  one bundle revision covers the instructions and resources, while an optional frozen
  requirement set prevents loading workflows unavailable in the current runtime.
- **Application-owned command environments**: host command capabilities and
  `RunCommand` can now receive one explicit environment snapshot for shell, argv, and
  background processes, while existing callers continue to inherit by default.
- **Typed work plans**: `WorkPlanState` and `update_plan` provide one validated,
  checkpoint-friendly checklist with a pure reducer and deterministic Markdown
  projection. The coding preset no longer mutates free-form todo metadata.
- **Terminal resume traces close cleanly**: resuming an already terminal checkpoint
  creates no model or tool step and now finalizes its empty trace manifest.
- **One shell admission boundary**: `run_command` now executes after normal tool
  admission instead of applying a second permission decision inside its handler.
- **Readable long tool outputs**: configure `FileArtifactStore` to save complete
  oversized results before the Engine creates a bounded model preview. Canonical
  `ToolResult.output` remains available to reducers and traces, checkpoints retain the
  exact model replacement, and a workspace-relative artifact path can be paged with the
  existing `read_file` tool.
- **Recoverable async checkpoints**: Engine now awaits one `CheckpointStore` at safe
  boundaries, including an initialized input snapshot before the first provider
  request, and persists the original task, full model-history prefix, state, and fork
  lineage. Resuming a terminal checkpoint returns that state without another model or
  tool step. SQLite work stays off the event loop and settles before
  cancellation propagates; the old JSON manager, lossy durability thread, empty
  pending-write layer, and trace bridge are gone. `TraceWriter` commits each
  completed step's event range before its step marker, while cancellation remains
  observable immediately. Memory and SQLite stores use the same JSON boundary.
- **One async-native model runtime**: `Engine.arun()` and `Engine.astep()` now own the
  model path from request through terminal response. OpenAI Responses, Anthropic
  Messages, compatible Chat Completions, Gemini, LiteLLM, and Ollama implement the same
  asynchronous stream contract; the former sync/async class hierarchy, `call_raw`,
  import-time registration, and daemon-thread `AsyncEngine` bridge are gone.
- **Same-spec qita comparisons**: compare views now verify recorded model, prompt,
  semantic task, tools, environment, context policy, budget, source revision, and
  experiment provenance first. Plain-text tasks use content-derived stable IDs, while
  task identity and runtime configuration use separate fingerprints, so wall-clock
  task wrapping cannot create false mismatches.
  Mismatched or incomplete pairs are explicitly descriptive rather than causal;
  matching pairs remain subject to provider and environment nondeterminism.
- **Truthful typed model streams**: Chat, Responses, and Anthropic streams retain
  provider finish reasons, reasoning and tool-call deltas, completed tool calls, and
  usage. Incomplete streams fail instead of fabricating completion, and Engine handlers
  no longer receive `on_end` after an error.
- **Call-accurate qita tool statistics**: tool counts and failures now come from the
  canonical action/result pairing instead of applying one step-level error to every
  call. Exact lifecycle counts and unmatched trace evidence remain visible for audits.
- **One bounded model-request lifecycle**: every Engine model call receives the run's
  absolute deadline and immediate cancellation signal. Provider connection, stream-idle,
  and QitOS-owned retry waits use live remaining time; cancellation closes the active
  asynchronous stream and late responses cannot commit.
- **One bounded tool-action lifecycle**: one absolute deadline now covers interceptor-
  free admission, approval, permission checks, invocation retries, and backoff.
  `ToolSpec.retry_policy` is the sole retry owner; validation and authorization run once,
  HTTP client retries are disabled, and event-loop-owned tasks drain before the batch
  returns. Dead Action execution knobs and the duplicate interceptor middleware have
  been removed.
- **One class-tool execution contract**: class tools now expose only
  `execute(args, runtime_context)`. `ToolRegistry` performs exact canonical-name lookup,
  while `ActionExecutor` alone owns validation, permissions, timeout/retry handling,
  invocation, and result normalization. Old `run`/`call` adapters, registry execution,
  automatic registry name aliases, duck-typed fallbacks, and implicit concurrency
  whitelists are gone; parallelism requires an explicit `concurrency_safe=True`
  declaration.
- **Truthful tool lifecycle results**: one canonical `ToolResult` projection now
  preserves success, partial, running, error, skipped/denied, input/approval, timeout,
  and cancellation across execution records, observations, history, traces, summaries,
  and success metrics. Unknown and legacy alias statuses fail closed; domain outcomes
  use a separate field instead of overloading execution status.
- **Provider-consistent reasoning continuation**: model presets now resolve GPT-5.6
  `max` without changing older OpenAI capability limits, forced compatible-tool calls
  cannot send contradictory thinking controls, and official Responses streams preserve
  encrypted reasoning items for stateless replay without exposing them in trace
  summaries or visible answers.
- **Run-scoped deadlines and bounded async shutdown**: relative runtime budgets and
  caller-supplied monotonic deadlines now resolve to one effective deadline shared by
  the Engine, tool admission, tool timeouts, retry backoff, and runtime waits. Async
  cancellation requests cooperative Engine shutdown. Legacy synchronous decorated
  functions run outside the event loop and are awaited to a known side-effect boundary.
- **One native tool-call lane**: when a model preset prefers provider-native tools,
  typed calls now bypass text interpreters and parsers, API requests omit the duplicate
  framework action contract, and every accepted, rejected, or malformed call commits one
  ordered result with the original call id. Malformed arguments never execute a tool.
- **Typed Subagent lifecycle**: immutable launch, handle, status, result, budget, and
  conclusion contracts separate persisted Subagent identity from live Engines. A Run-owned
  async supervisor handles admission, wait, interrupt, terminal state, parent delivery,
  and teardown; durable started/terminal records prevent recovery from replaying a Subagent,
  and forks cannot control inherited handles. Launch policy carries narrowed profile,
  Tool groups, workspace, and budget, while invocation cleanup owns fresh model
  resources. Shared status, wait, message, and interrupt tools use the same supervisor;
  `SubagentTool` is now a thin launch projection.
- **Environment-backed coding tools**: named Env capability groups now let the same
  bounded workspace tools run against host, container, or remote providers. The compact
  workspace profile exposes one lowercase surface (`read_file`, `write_file`,
  `edit_file`, `glob`, `grep`, and related tools). Search uses fixed-argv `rg`, stable
  bounded results, NUL-safe paths, and explicit hidden/ignored-file controls without
  per-tool backend adapters.
- **Managed public web fetch**: a provider-neutral `web_fetch` tool now accepts
  host-injected providers. An explicitly configured Kimi managed-fetch adapter adds
  public-initial-URL validation, bounded results, and provider failure categories; QitOS
  does not guess a service URL from the selected model.
- **Runtime input and idle wait**: background work can post a small event to an exact
  Engine run through an async, Journal-first mailbox. Explicit runtime waits sleep
  without model polling or step growth and wake on input, cancellation, or the run
  deadline. Accepted input survives restart until a completed model transaction binds
  it, and input racing a final answer is handled on the next turn.
- **Live OpenAI-compatible streams**: Engine calls use one explicit QitOS retry budget
  with SDK retries disabled. Connection and pre-event failures can retry within a
  300-second recovery window by default; after the first provider event, deltas stay
  live and failures stop without replay. An event-idle timeout detects stalled streams
  without cutting off healthy long responses.
- **Readable tool evidence**: tools can now project a compact `model_summary`
  into native tool-call history without discarding their full structured result
  from reducers, traces, or replay.
- **Transaction-safe context compaction**: complete provider inputs, including native
  tool schemas, now force compaction at 80% of the provider-safe input budget. Three
  bounded levels preserve complete tool exchanges, and failed or raced summaries never
  mutate canonical history. Sync retrieval uses a bounded heuristic when given an async
  model instead of creating an invalid sync-to-async bridge. The obsolete
  message-slicing compatibility option is gone; recent retention is expressed only in
  complete rounds.
- **Modern CyberGym tool turns**: authoritative per-step runtime state is now folded into the final real tool result instead of creating a trailing user turn, preserving native `assistant -> tool` chains for compatible providers.
- **qita trajectory workbench**: Run pages now open in a diagnosis-first view with a Focus Navigator, Agent Behavior Story, and right-side Inspector. Each step follows `Input -> Thought -> Action Calls -> Environment Observation`; every action is paired with its complete parameters, status, latency, and model-visible result, while canonical raw and unmatched evidence stays auditable in the Inspector. Failed calls expand by default, successful calls fold, and long content is wrapped and never available only as a truncated preview. CyberGym budget stops and `submit_poc` verification failures are promoted as review targets. Persistent light/dark themes cover board, run, replay, and compare pages.
- **Consistent immediate cancellation traces**: once the Engine observes an immediate cancellation, State, task/result objects, END events, and trace manifests now agree on `cancelled_immediate`; qita sees the manifest as `stopped` rather than a normal completion.
- **No false completion for structured action text**: when a native-tool model emits malformed action fields as text instead of `tool_calls`, QitOS now keeps the parser recovery path rather than treating that text as a final answer; ordinary natural-language conclusions remain unchanged.
- **Window-safe native tool history**: model requests now discard orphan tool results when a message window evicts their assistant declaration, preventing long-running parallel-tool agents from sending invalid `tool_call_id` chains while preserving complete rounds and existing recovery behavior.
- **Preset-aware direct Engine construction**: `Engine(agent=...)` now honors protocols attached by `build_model_for_preset(...)`, so provider aliases such as Kimi K3 keep JSON/native API tool delivery instead of silently falling back to text ReAct.
- **Bounded empty-response recovery**: model responses with neither usable text nor tool calls are now classified as traceable `model_error` failures, retried once, and stopped cleanly if they repeat instead of consuming the full agent step budget as parser waits.
- **Optional OpenAI Responses API transport**: construct a model with
  `api_mode="responses"` to preserve typed output items, parallel function calls,
  `call_id` tool results, streaming events, and replayable tool context. Existing Chat
  Completions behavior remains the default.
- **Native response extraction hardening**: null-content OpenAI-compatible messages no longer surface SDK repr strings as final answers.
- **OpenAI-compatible request hardening**: forced tool-call requests now avoid provider thinking-mode conflicts, and JSON/tool-call parsing repairs bare control characters inside string values.
- **More robust JSON salvage**: JSON-like parser recovery now ignores apostrophes in surrounding prose, so contractions before a valid payload no longer hide the object.
- **Cleaner delegate tools**: `AgentSpec.tool_name` lets multi-agent systems expose task-oriented tool names, and `DelegateTool` now delivers structured `context` payloads into subagents.
- **CyberGym integration hardening**: v0.6 integration runs now preserve valid OpenAI-compatible tool schemas, redact persisted secrets across traces/results/render artifacts, and keep CyberGym PoC-generation shell commands out of the interactive review path while preserving the default coding-tool guard.
- **Lighter-weight CyberGym bootstrap guidance**: the CyberGym PoC agent now derives a compact task-spec summary, ranks likely parser/harness/sample paths more aggressively, tracks richer candidate provenance, and records a lightweight internal failure taxonomy without changing the single-agent runtime.

## Historical v0.5.0 highlights

These features describe the v0.5 release and include APIs removed by the
current breaking migration. Use the current runtime section and changelog for
the supported surface.

- **Method recipes**: Self-Refine, Reflexion, LATS, MoA, and Magentic-One are available
  as importable `qitos.recipes` implementations.
- **Export APIs**: `EngineConfig`, `ToolPermissionSpec`, `CriticTrace`, and `HandoffTrace` for programmatic access to engine configuration and trace data.
- **FamilyPreset extensibility**: `override()`, `recommended_*` advisory fields, and `MaxTokensCriteria` stop criterion.
- **qita cost panel**: Token usage and cost metrics in the run overview.

See [CHANGELOG.md](CHANGELOG.md) for the full list.

## Live Terminal of QitOS for Code Review

<p align="center">
  <img src="demo.gif" alt="QitOS long-running agent demo" width="92%">
</p>

## Who QitOS is For

- **Agent researchers** who want to compare models, Tool policies, and context
  strategies without replacing the runtime.
- **Long-running agent debuggers** who care about trajectory review, replay, diff, and context-collapse diagnosis instead of app scaffolding alone.

## Run QitOS in 2 Minutes

The minimal agent in QitOS composes the `Agent` façade directly: one model, one
Tool registry, one prompt.

```bash
git clone https://github.com/Qitor/qitos.git
cd qitos
pip install ".[models]"
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.siliconflow.cn/v1/"
export QITOS_MODEL="Qwen/Qwen3-8B"
python examples/quickstart/minimal_agent.py
```

The quickstart drives the `Message -> Model -> ToolCall -> ToolResult` loop through
`agent.prompt(...)` and prints the final assistant message.

Then go deeper:

- Want the building blocks? See the [kit reference](https://qitor.mintlify.app/reference/kit)

## Why QitOS

| If you want... | QitOS gives you... |
|---|---|
| reproducible agent research | typed loop transactions and canonical journals |
| artifact inspection | read-only `qita` board, replay, export, and diff for compatible traces |
| less framework glue code | one canonical execution loop |

## Tooling Layout

QiTOS separates tool imports into three layers:

- `qitos.kit`: the simplest curated entrypoint for common toolsets
- `qitos.kit.toolset`: scenario-oriented presets and registry builders
- `qitos.kit.tool.<domain>`: advanced atomic capability imports

Default composition is list-first:

```python
from qitos import ToolRegistry
from qitos.kit.tool.file import ReadFile
from qitos.kit.toolset import coding_tools

registry = ToolRegistry().include_toolset(
    [
        ReadFile(workspace_root="."),
        coding_tools(workspace_root="."),
    ]
)
```

Security-sensitive tools are explicit opt-in imports and are not part of `qitos`, `qitos.kit`, or the quickstart path.

## Documentation Map

- Start here: [Introduction](https://qitor.mintlify.app/introduction)
- First successful run: [Quickstart](https://qitor.mintlify.app/quickstart)
- Install options: [Installation](https://qitor.mintlify.app/installation)
- Compose your own agent: [Kit Reference](https://qitor.mintlify.app/reference/kit)
- Inspect compatible traces: [Observability](https://qitor.mintlify.app/guides/observability)
- Follow the course: [Tutorials](https://qitor.mintlify.app/tutorials)
- Check commands: [CLI Reference](https://qitor.mintlify.app/reference/cli)

## Preview

<table>
  <tr>
    <td align="center"><strong>QitOS CLI</strong></td>
    <td align="center"><strong>qita Board</strong></td>
    <td align="center"><strong>qita Trajectory View</strong></td>
  </tr>
  <tr>
    <td align="center">
      <a href="assets/qitos_cli_snapshot.png">
        <img src="assets/qitos_cli_snapshot.png" alt="QitOS CLI" width="100%" />
      </a>
    </td>
    <td align="center">
      <a href="assets/qita_board_snapshot.png">
        <img src="assets/qita_board_snapshot.png" alt="qita Board" width="100%" />
      </a>
    </td>
    <td align="center">
      <a href="assets/qita_traj_snapshot.png">
        <img src="assets/qita_traj_snapshot.png" alt="qita Trajectory View" width="100%" />
      </a>
    </td>
  </tr>
</table>

## Status

QitOS is currently **Beta**.

- Stable behavior: the minimal Agent loop and façade, canonical durable
  journals, and qita inspection of compatible trace artifacts.
- Likely to evolve: public Agent/runtime APIs, higher-level convenience APIs, some
  `kit` modules, and experimental toolsets.
- If you are evaluating adoption, start from the kernel and examples, not assumptions about frozen surface area.
- For ongoing project evolution and upgrade notes, see [CHANGELOG.md](CHANGELOG.md).

## Installation and Versions

- Supported Python version: **3.10+**
- User install: `pip install "qitos[models]"`
- Version check: `qit --version`
- Minimal agent example (from a source checkout): `python examples/quickstart/minimal_agent.py`
- Optional provider config: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `QITOS_MODEL`
- Core-only install: `pip install qitos`
- Repo source install: `pip install -r requirements.txt`
- Full contributor install: `pip install -r requirements-dev.txt`
- Optional extras: `qitos[models]`, `qitos[hf]`, `qitos[all]`
- Installation guide: [Installation](https://qitor.mintlify.app/installation)

## Contributing

Contributions are welcome, especially around the Agent runtime, tools, provider adapters,
Session journals, qita UX, and documentation. Start with
[CONTRIBUTING.md](CONTRIBUTING.md) for the PR process,
[DEVELOPMENT.md](DEVELOPMENT.md) for the local workflow,
[ARCHITECTURE.md](ARCHITECTURE.md) for system design,
[SECURITY.md](SECURITY.md) for disclosure guidance, and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations.

## License

MIT. See [LICENSE](LICENSE).
