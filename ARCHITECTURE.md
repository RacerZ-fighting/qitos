# QitOS Architecture

## High-Level Design

QitOS is organized around one core execution narrative:

`AgentModule -> Engine -> Decision -> ActionExecutor -> Env/Tools -> Trace/qita`

The project is intentionally kernel-first. Higher-level patterns, examples, and benchmark adapters are expected to build on the same runtime loop rather than bypass it.

## Core Components

### `qitos.core`

Defines the framework contracts:

- `AgentModule`
- `StateSchema`
- `Decision`
- `Action`
- `BaseTool` and tool specs
- task, memory, history, and error abstractions

### `qitos.engine`

Owns the runtime:

- state initialization and control flow
- decision normalization and parser integration
- action execution
- hook dispatch
- env capability checks
- trace and context telemetry

### `qitos.kit`

Provides curated building blocks on top of the core:

- parsers
- prompts
- planning helpers
- env implementations
- memory/history implementations
- canonical toolsets such as `CodingToolSet`

### `qitos.trace` and `qitos.qita`

Provide observability:

- structured run artifacts
- event and step records
- replay and run inspection
- browser-friendly trace views

### `qitos.benchmark`

Contains adapters and runtime support for benchmark-oriented workflows such as GAIA, Tau-Bench, and CyBench.

## Key Design Decisions

### Kernel First

The stable surface is the execution kernel, not every convenience wrapper in `kit`.

### Decision-Centric Runtime

The engine executes `Decision` objects, even when the model output originates as text or provider-specific response payloads.

### Canonical Tool Surface

`CodingToolSet` is the standard file/shell/codebase tool bundle. Deprecated compatibility wrappers were removed to keep one primary authoring path.

### Opt-In Experimental Security Research

Higher-risk security research tools live under `qitos.kit.tool.experimental.security_research` and are not part of the default public surface.

### Observability As A First-Class Feature

Trace artifacts and qita views are part of the framework contract, not an afterthought.

## Typical Flow

1. An `AgentModule` produces or delegates a `Decision`.
2. The `Engine` normalizes the decision and records observability.
3. `ActionExecutor` validates and executes actions through tools and env ops.
4. The agent reduces the observation into updated state.
5. Trace artifacts and qita views expose the full run.

## Repository Structure

- `qitos/`: framework source
- `tests/`: regression and behavior tests
- `docs/`: Mintlify documentation content
- `examples/`: canonical patterns and reference agents

## Repository Layers

### Stable Core Framework Surface

The stable surface is:

- `qitos.core`
- `qitos.engine`
- `qitos.trace`
- `qitos.qita`
- basic model/provider abstractions needed by the engine
- public contracts such as `AgentModule`, `StateSchema`, `Decision`, `Action`, `BaseTool`, `ToolRegistry`, trace artifacts, run specs, and hook contracts

Top-level `qitos` imports should stay limited to these kernel contracts and compatibility-critical public contracts.

### Curated Framework Extensions

The following may remain in this repository when they are generic, reusable, and not tied to one product agent:

- `qitos.kit`
- `qitos.kit.tool`
- `qitos.kit.toolset`
- `qitos.prompting`
- `qitos.protocols`
- `qitos.models`
- `qitos.render`

These APIs are useful framework extensions, but they are not all equally stable. They should avoid product naming, product workflows, and high-risk default exports.

### Recipes And Benchmarks

`qitos.recipes` may contain reusable research baselines or canonical benchmark methods. Recipes should be callable from thin examples instead of duplicated inside examples.

`qitos.benchmark` (deprecated, migrating to `qitos.recipes.benchmarks`) may contain framework-level adapters, runners, scorers, and dataset-neutral glue. They must not vendor benchmark datasets, large external assets, or product-specific workflows.

### Examples Policy

`examples/` is a small canonical learning path, not an app gallery. Examples should:

- teach one QitOS concept at a time
- run locally or with one standard model-provider path
- be short enough to read as documentation
- demonstrate canonical authoring flow
- avoid heavy hidden dependencies
- avoid standalone product behavior

Full applications live in `qitos-zoo`.

### qitos-zoo

Move or prepare to move examples and apps that:

- reproduce mature agent products
- require many files, configs, prompts, workflows, or assets
- are showcase-grade instead of teaching-first
- are cybersecurity/pentesting product agents
- are Claude Code-style product agents
- have their own roadmap, UI, CLI, benchmark harness, or workflow state

Recommended names:

- `qitos_coder`: Claude Code-inspired coding agent built with QitOS.
- `qitos_cyber`: PentAGI-inspired cybersecurity agent built with QitOS.
- `qitos_auditor`: DeepAudit-inspired code security audit agent built with QitOS.

#### qitos_zoo structure

```
qitos_zoo/
  __init__.py
  qitos_coder/       — coding agent + tests/ + README.md
  qitos_cyber/       — cyber agent + tests/ + README.md
  qitos_auditor/     — audit agent + tests/ + README.md
  experimental/      — product candidates needing hardening
  docs/              — adding_a_new_agent.md, app_template.md, safety_and_scope.md
```

#### E2E test ownership

Agent application e2e tests belong in `qitos_zoo/<app>/tests/`. The core `tests/` directory only contains framework-level tests. No agent-specific e2e tests should reside in the core repository.

#### Zero duplication

No file may exist in both `examples/` and `qitos_zoo/` with identical content. Once code is migrated to `qitos_zoo/`, the original must be removed from `examples/`.

## Security-Sensitive Rule

Cybersecurity research tooling must never be part of the default public surface. It may exist only as explicit experimental modules with clear warnings, or as qitos-zoo applications with controlled documented use.

Do not export offensive or high-risk tools from `qitos.__init__`, `qitos.kit` default imports, `qit demo`, or quickstart examples.

## What Must Not Enter Core

- Claude Code-style product agents
- PentAGI-style cybersecurity agents
- full SWE, desktop, EPUB, or SkillHub product workflows
- offensive or high-risk security tools in default imports or demos
- benchmark datasets, large generated artifacts, local absolute paths, or secrets
- dependencies needed only by product apps

## Promotion Rule

Code may move from qitos-zoo into QitOS only if it is generic, tested, documented, and needed by at least two independent apps. Product-specific code stays in qitos-zoo.

## Decision Table

| Item | Belongs in core? | Decision |
| --- | --- | --- |
| Engine loop | Yes | Stable core |
| Agent contracts | Yes | Stable core |
| Trace/qita artifacts | Yes | Stable observability |
| Generic tool protocol | Yes | Stable core or kit |
| `CodingToolSet` | Maybe | Keep only if generic and minimal |
| Claude Code-style agent | No | qitos-zoo |
| PentAGI-style cyber agent | No | qitos-zoo |
| SWE product workflow | No | qitos-zoo unless reduced to minimal recipe |
| OpenAI CUA product clone | No | qitos-zoo unless reduced to minimal desktop smoke test |
| Benchmark adapters | Yes | Only if thin and dataset-neutral |
| Benchmark datasets/assets | No | External or qitos-zoo |
| Experimental security tools | Opt-in only | Never default-exported |

## Non-Goals

- hiding execution semantics behind opaque abstractions
- maintaining multiple long-term public APIs for the same tool workflow
- treating examples as production-only integrations detached from the kernel
