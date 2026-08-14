# Qitos Roadmap

> Goal: make Qitos the leading open-source framework for AI Agent researchers.  
> Strategy: do **not** compete as the broadest agent platform; compete as the most compelling **research-first, benchmark-native, trace-native, reproducibility-first** framework for building and studying agents.

---

## 1. Product Thesis

### 1.1 Positioning

**Qitos = the research-first, benchmark-native, trace-native framework for reproducible AI agents.**

In practice, this means Qitos should become the default choice when a researcher wants to:

- prototype a new agent method quickly,
- run a fair benchmark comparison across model families,
- replay and inspect failed trajectories,
- study multimodal / computer-use agents,
- build and evaluate multi-agent systems,
- reproduce a paper or publish a benchmark report.

### 1.2 What Qitos should win on

Qitos should aim to win the following mindshare category:

> **“The PyTorch for AI agent research.”**

That means winning on:

- reproducibility,
- experimental clarity,
- benchmark integration,
- trace inspection,
- extensibility for new methods,
- clean abstractions for agent research.

### 1.3 What Qitos should *not* optimize for first

Qitos should **not** prioritize the following as its primary identity:

- the largest number of integrations,
- no-code / low-code agent builders,
- enterprise workflow orchestration,
- “all-in-one” RAG infrastructure,
- vendor breadth as the main growth story.

Those are adjacent opportunities, but they are not the shortest path to becoming dominant among AI agent researchers.

---

## 2. Core Design Principles

### 2.1 One Kernel

The same `AgentModule + Engine` runtime should power:

- toy examples,
- benchmark runners,
- product-style agents,
- multimodal agents,
- multi-agent systems.

Qitos should avoid splitting into multiple parallel runtimes.

### 2.2 Trace as Artifact

A trace is not just a debug log. It is a research artifact.

Every run should natively support:

- replay,
- export,
- diff,
- failure analysis,
- benchmark submission,
- report / appendix generation.

### 2.3 Benchmarks are First-Class

Benchmarks should not live as special-case demos. They should be first-class framework citizens with:

- a unified entrypoint,
- a unified config model,
- a unified artifact format,
- a unified judge / evaluation interface,
- a unified comparison workflow.

### 2.4 Fair Baselines by Default

Qitos should help researchers produce serious default baselines.

Users should not need to manually rebuild prompt protocol, parsing, retry logic, context policy, and evaluation conventions from scratch every time.

### 2.5 Research Extensibility First

Qitos should be optimized for:

- inserting a new method,
- adding a new environment,
- adapting a new benchmark,
- extending new observation / action schemas,
- analyzing new failure modes.

### 2.6 Opinionated Where It Matters

Qitos should be explicit and opinionated about:

- prompt protocol,
- parser pairing,
- harness presets,
- trace schema,
- benchmark contracts,
- run artifact shape.

That is how Qitos becomes a real research framework instead of a loose collection of utilities.

---

## 3. The Five Strategic Pillars

### Pillar A — Reproducibility Core

Build the foundation that makes experiments reproducible, comparable, and exportable.

Key components:

- `ExperimentSpec` / `RunSpec`
- artifact schema
- deterministic replay
- structured run metadata
- qita diff
- benchmark-grade outputs

### Pillar B — Optimized Harness

Build default optimized execution layers for major model families.

Key components:

- `ModelAdapter`
- `FamilyPreset`
- `PromptProtocol`
- `ToolPolicy`
- `ContextPolicy`
- harness benchmark reports

### Pillar C — Multimodal / Computer-Use Agents

Upgrade Qitos into a true research platform for multimodal and GUI agents.

Key components:

- unified observation schema
- unified action schema
- browser / desktop / mobile environment adapters
- visual replay in qita
- grounding-aware inspection tools

### Pillar D — Native Multi-Agent Runtime

Treat multi-agent as a research object, not only an orchestration feature.

Key components:

- local sub-agent runtime
- handoff / delegation
- message bus
- shared vs private memory
- canonical collaboration templates
- multi-agent tracing and evaluation

### Pillar E — Ecosystem and Interoperability

Turn Qitos from a strong repo into a durable research ecosystem.

Key components:

- benchmark adapter SDK
- recipes / reproductions
- MCP bridge
- A2A bridge
- contribution paths by module type
- recurring benchmark reports

---

## 4. Priority Order

The recommended build order is:

1. **Reproducibility Core**
2. **Optimized Harness**
3. **Multimodal / Computer-Use**
4. **Native Multi-Agent**
5. **Interop / Ecosystem Scaling**

Why this order:

- Without reproducibility, benchmark claims will be weak.
- Without optimized harnesses, comparisons across model families will be noisy and unfair.
- Without multimodal support, Qitos misses the frontier of current agent research.
- Without a native multi-agent runtime, Qitos cannot become the default for studying collaboration patterns.
- Interop matters, but only after the core research semantics are stable.

---

## 5. Roadmap by Phase

## Phase 0 — Research Kernel Hardening (now → 2026-05)

### Objective

Upgrade Qitos from a promising framework into a framework with explicit research semantics.

### Key Deliverables

#### 5.0.1 `ExperimentSpec` / `RunSpec`

Define a unified schema that fully describes a run:

- model family / model name
- prompt protocol
- parser
- toolset
- environment
- benchmark split
- seed
- stop criteria
- judge config
- git SHA
- package version
- cost / latency / token stats

#### 5.0.2 Artifact Schema v1

Standardize outputs for every run:

- trajectory
- state snapshots
- observations
- tool calls
- model raw outputs
- parser outputs
- judge results
- replay assets
- final metrics

#### 5.0.3 qita v2 baseline

At minimum:

- run replay
- run export
- side-by-side diff
- token / cost / latency comparison
- first-failure-step highlight
- HTML export

#### 5.0.4 Unified benchmark CLI

Examples:

- `qitos bench run`
- `qitos bench eval`
- `qitos bench replay`
- `qitos bench export`

### Exit Criteria

- any benchmark run can be replayed from config + seed + git SHA,
- two runs can be diffed structurally,
- examples and benchmark runners no longer feel like separate products,
- docs clearly explain the experiment and artifact model.

---

## Phase 1 — Optimized Harness as a Flagship Capability (2026-05 → 2026-07)

### Objective

Make “default optimized harness” the first major reason researchers adopt Qitos.

### Product Definition

Qitos should provide more than provider adapters. It should provide serious **model-family presets**.

A preset should encode:

- prompt protocol,
- parser strategy,
- tool schema behavior,
- retry / repair logic,
- context compaction policy,
- stop rules,
- optional reflection / critic settings,
- benchmark-specific tuning notes.

### Key Deliverables

#### 5.1.1 Harness Abstraction Layer

Core objects:

- `ModelAdapter`
- `FamilyPreset`
- `HarnessPolicy`
- `PromptProtocol`
- `ToolPolicy`
- `ContextPolicy`

#### 5.1.2 Gold Presets

Ship strong default presets for 3–5 major model families.

Each preset should specify:

- recommended benchmark targets,
- default protocol,
- parser strategy,
- retry behavior,
- context policy,
- recommended temperatures / stop rules.

#### 5.1.3 Harness Benchmark Report

For every meaningful harness release, publish:

- family A vs B vs C comparisons,
- success / cost / latency tables,
- common failure taxonomy,
- preset tuning notes,
- reproducible config references.

### Exit Criteria

- researchers can switch model families with minimal code changes,
- Qitos presets become credible baselines,
- benchmark comparisons feel fairer and more repeatable.

---

## Phase 2 — Multimodal / Computer-Use Platformization (2026-07 → 2026-10)

### Objective

Make Qitos a serious framework for multimodal and GUI agent research.

### Core Insight

Do not just add “image input.” Build a unified environment abstraction.

### Key Deliverables

#### 5.2.1 Unified Observation Layer

Support:

- text
- screenshot
- DOM
- accessibility tree
- UI candidates
- OCR / grounding metadata

#### 5.2.2 Unified Action Layer

Support:

- click
- double click
- type
- hotkey
- scroll
- drag
- select region
- wait
- tool call

#### 5.2.3 Environment Adapter SDK

A standard way to integrate new benchmarks and execution environments.

#### 5.2.4 qita Visual Replay

Add:

- screenshot timeline,
- action overlay,
- observation pack viewer,
- branch comparison,
- grounding failure annotation.

#### 5.2.5 Computer-Use Starter Pack

A canonical multimodal baseline agent including:

- planner,
- grounding step,
- action selector,
- critic / retry loop,
- visual replay support.

### Exit Criteria

- the same agent policy can run across at least two GUI / web environment families,
- qita can clearly reveal where GUI agents fail,
- researchers begin to think of Qitos as a computer-use research framework.

---

## Phase 3 — Native Multi-Agent Semantics (2026-10 → 2026-12)

### Objective

Treat multi-agent behavior as something to study, measure, and compare.

### Priority Rule

Build **local multi-agent runtime first**, then remote interop.

### Key Deliverables

#### 5.3.1 Local Sub-Agent Runtime

Support:

- handoff,
- delegation,
- role contracts,
- message bus,
- shared vs private memory,
- stop / timeout / arbitration.

#### 5.3.2 Composable Multi-Agent Experiments

Keep role and topology policy in explicit recipes or applications. Reuse the same
typed Child launch, control, result, budget, cancellation, and Journal contracts for
manager-worker, planner-executor, proposer-verifier, actor-critic, debate, and
self-play experiments instead of shipping fixed constructors as another runtime.

#### 5.3.3 Multi-Agent Trace and Evaluation

Support:

- per-agent token / cost breakdown,
- handoff timeline,
- conflict / loop detection,
- contribution estimates,
- role-level attribution.

### Exit Criteria

- multi-agent experiments are no longer just ad-hoc nested agent calls,
- Qitos can support systematic studies of handoff strategy, memory partitioning, and role design,
- replay / diff / benchmark workflows still work in multi-agent settings.

---

## Phase 4 — Interop and Ecosystem Flywheel (2026-12 → 2027-Q1)

### Objective

Turn Qitos into a durable research ecosystem.

### Key Deliverables

#### 5.4.1 MCP / A2A Bridges

Build:

- MCP client,
- MCP server wrapper,
- A2A bridge,
- remote-agent wrapper.

#### 5.4.2 `qitos-recipes`

Create a separate recipes layer for:

- canonical agent patterns,
- benchmark baselines,
- representative multimodal methods,
- representative multi-agent methods.

#### 5.4.3 Modular Contribution Paths

Make it easy for contributors to work on:

- benchmark adapters,
- environment wrappers,
- harness presets,
- qita panels,
- recipes,
- docs and tutorials.

#### 5.4.4 Monthly Benchmark Reporting

Establish a recurring growth loop:

- one benchmark update per month,
- one technical blog per release,
- artifact and trace sharing,
- failure case writeups.

### Exit Criteria

- external contributors begin contributing by module type,
- Qitos develops a visible reproduction and benchmark culture,
- Qitos becomes recognizable as infrastructure for papers and public benchmark reports.

---

## 6. Version Narrative

A strong version narrative helps the community understand what each release *means*.

- **v0.3** — Reproducible Runs Foundation
- **v0.4** — Fair Baselines via Optimized Harness
- **v0.5** — Multimodal / Visual-Web Core
- **v0.6** — Computer-Use and Visual Replay
- **v0.7** — Native Multi-Agent Core
- **v0.8** — Safety / Judge / Robustness Benchmarks
- **v0.9** — Interop + Recipes + Community Preparation
- **v1.0** — Research OS for AI Agents

---

## 7. 12-Month OKRs

### O1 — Establish Research Credibility

- 100% of official benchmark runs are replayable,
- artifact schema is stable and documented,
- at least 3 benchmarks have robust runners.

### O2 — Establish Baseline Credibility

- 3–5 model families have gold presets,
- at least 2 public harness benchmark reports are published.

### O3 — Establish Frontier Capability

- Qitos supports at least two major multimodal / GUI benchmark families,
- qita supports visual replay and grounding failure analysis.

### O4 — Establish Multi-Agent Research Capability

- at least 3 canonical multi-agent templates are shipped,
- handoff / shared memory / role attribution traces are supported.

### O5 — Establish a Community Flywheel

- `qitos-recipes` is launched,
- every release ships with docs + examples + report,
- external contributions arrive through adapters / presets / recipes.

---

## 8. Key Risks and Mitigations

### Risk 1 — Too many directions, unstable abstractions

**Mitigation:** strictly sequence work as: reproducibility → harness → multimodal → multi-agent → interop.

### Risk 2 — Many features, weak identity

**Mitigation:** every release must tell one clear story:

- v0.3 = reproducible runs
- v0.4 = fair baselines
- v0.5 = multimodal research
- v0.7 = multi-agent science
- v1.0 = research OS

### Risk 3 — Multi-agent complexity explodes too early

**Mitigation:** local runtime first, remote interop second.

### Risk 4 — Benchmark integrations become inconsistent

**Mitigation:** enforce the environment adapter SDK and artifact schema.

### Risk 5 — Docs lag behind framework evolution

**Mitigation:** every phase must ship with:

- tutorial,
- example,
- benchmark guide,
- migration note.

---

## 9. Final Standard of Success

Qitos does **not** win when it has the longest feature page.

Qitos wins when this becomes true:

> When an AI agent researcher wants to build a new method, run a benchmark, analyze a failure, or reproduce a paper, their first instinct is to use Qitos.
