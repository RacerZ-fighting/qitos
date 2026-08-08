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

### Added

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
- Added a bounded transport policy for synchronous OpenAI-compatible calls and async
  Responses streams, with one visible retry owner, provider retry-hint handling, typed
  exhaustion errors, and stream event-idle timeouts.
- Added generic `model_summary` projection for native tool-call history and
  model-visible observations. Tools can now retain full structured evidence
  for reducers and replay while supplying a bounded readable result to models.
- Added transient runtime-context delivery to `MessageBuildResult`. Custom
  agents can fold authoritative controller state into the final real tool
  result without persisting a synthetic user turn.
- Added opt-in OpenAI Responses API support for synchronous, asynchronous, and typed streaming calls, including structured output-item preservation, `call_id` tool-result correlation, stateless tool-round replay, and privacy-safe trace summaries. Chat Completions remains the default.
- Added `AgentSpec.tool_name` so delegate workers can expose task-oriented model-facing tool names while keeping the registry agent name stable.
- Added qita's trajectory analysis workbench with diagnosis-first run pages, derived failure insights, focus navigation, critical-step guidance, an inspector panel, and expandable full-content evidence views for long thoughts, observations, parser diagnostics, actions, and critic outputs.
- Added qita `step_interactions`, a derived action-observation view that pairs each action with its complete arguments, invocation metadata, model-visible result, and canonical raw result while separating environment-only and unmatched evidence.
- Added a qita light/dark theme system with a persistent toolbar toggle across board, run detail, replay, and comparison pages.

### Changed

- Model calls now run under one Engine-scoped absolute request deadline. Provider
  timeouts and retry backoff are clamped to live remaining time, immediate cancellation
  stops waiting, and uncooperative synchronous calls remain on bounded daemon workers.
  Official sync/async OpenAI models are thin specializations of the canonical
  OpenAI-compatible adapter instead of maintaining separate transports.
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
  uses `.70/.50/.35` history budgets without mutating canonical history. The unsafe
  `keep_last_messages` slicing option is deprecated;
  it remains accepted as a compatibility-only no-op while callers migrate to complete
  round retention. Responses text/tool payloads and generic/native mirrors now keep one
  complete, accurately counted call/result transaction.
- Async Engine runs now use a daemon worker with cooperative cancellation, and the two
  duplicated stream cleanup paths share one lifecycle implementation. Early consumer
  close requests Engine cancellation; normal completion still propagates Engine errors.
- Background `AgentTool` runs now snapshot parent history at launch but defer model,
  Engine, and trace construction until an execution slot opens. A bounded daemon pool,
  repeatable bounded close, canonical cancellation stops, and terminal-before-wakeup
  ordering keep child teardown from extending process lifetime indefinitely.
- Clarified the generic `AgentTool` model contract: independent multi-step tasks can be
  delegated in one response for concurrent execution, while dependent steps and cheap
  mechanical variants remain in the parent. Explicit tool guidance is no longer replaced
  by the `execute()` implementation docstring during initialization.
- OpenAI-compatible Engine calls now use transactional streaming by default. Retryable
  mid-stream failures discard partial text and tool calls before retrying within the
  existing QitOS-owned attempt budget and a configurable 300-second recovery window,
  while active streams use an idle deadline instead of the ordinary request timeout.
- OpenAI-compatible clients now disable OpenAI SDK retries on paths where QitOS owns the
  retry budget, preventing multiplicative retry delays.
- Raised the optional OpenAI SDK floor to `openai>=1.66.0` and taught compact history to preserve active Responses function-call rounds atomically.
- Strengthened the CyberGym PoC agent's task bootstrap with lightweight structured task-spec extraction and more relevant repo evidence ranking.
- Clarified candidate provenance and lightweight failure taxonomy handling in the CyberGym agent without changing its single-agent runtime architecture.
- Improved qita diagnostics for CyberGym-style traces so budget stops are marked as review-needed, `submit_poc` verification failures are promoted as critical inspection steps, and low-frequency metadata stays out of the default attention path.
- Redesigned qita step stories around `Input -> Thought -> Action Calls -> Environment Observation`: multi-action calls now render as numbered paired units with status, latency, parameters, and their own result; failed calls expand by default, successful calls fold, and all long evidence remains available in wrapped, copyable code views and call-aware Inspector tabs.

### Fixed

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
  waits. Timed-out synchronous tools use daemon workers so an orphan cannot block
  interpreter shutdown, and a saturated event queue still retains its terminal marker.
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

### Breaking

- Direct class-tool callers must pass an argument dictionary to `execute(...)`.
  Registry callers must resolve an exact canonical name with `get()` or execute through
  `Engine`/`ActionExecutor`. Tools run in parallel only when their spec explicitly sets
  `concurrency_safe=True`. A `FunctionTool` remains directly callable as the ordinary
  host-side behavior of `@function_tool`; agent execution never uses that shortcut.
- `CodingToolSet` no longer accepts `expose_legacy_aliases` or
  `expose_modern_names`. Callers must use the canonical lowercase tool names.

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
