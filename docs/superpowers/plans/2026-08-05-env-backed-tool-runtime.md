# Env-Backed Tool Runtime Implementation Plan

**Goal:** Make QitOS environment-dependent tools consume declared Env capability
groups so one environment provider can target a host, container, remote runner, or
other execution backend without per-tool adapters.

**Architecture:** Keep QitOS tool schemas and behavior in QitOS. The Engine resolves
each tool's `required_ops` from the current `Env`; concrete providers implement the
small filesystem, process, terminal, browser, and service contracts. Preserve
controller-local tools and the existing `AgentModule + Engine` lifecycle. PentestAgent
will supply one attempt-scoped Env whose filesystem and process providers target the
same Worker container.

**Constraints:**

- QitOS must not import PentestAgent or Docker-specific policy.
- PentestAgent must adapt capability groups, not individual QitOS tools.
- Existing host-backed QitOS usage remains supported.
- Registered tools fail preflight when a required capability is unavailable.
- ToolRegistry allowlists remain separate from environment capability availability.

## Task 1: Lock the generic Env composition contract

- [x] Add focused tests for a mapping-backed Env, capability preflight, lifecycle,
  and shared-provider cleanup.
- [x] Implement the smallest reusable concrete Env that exposes arbitrary named ops
  groups without encoding tool names.
- [x] Keep stable contracts in `qitos.core` and the concrete implementation in
  `qitos.kit`.

## Task 2: Fill QitOS filesystem and process capability gaps

- [x] Define the bounded filesystem primitives required by standard file/coding
  tools, including target-side path resolution and binary-safe reads/writes.
- [x] Add fixed-argv process execution while retaining the existing command
  compatibility path.
- [x] Implement host providers and focused contract tests.
- [x] Ensure errors remain real tool/runtime failures rather than successful payloads
  containing error text.

## Task 3: Route standard QitOS workspace tools through Env ops

- [x] Migrate the standard file, codebase, and shell implementations used by the
  workspace profile. Notebook, task, and HTTP tools remain controller-local and are not
  part of this profile.
- [x] Declare environment ops for each environment-bound tool family and centralize ops
  lookup so implementations contain no backend-specific branches.
- [x] Preserve pure controller tools such as planning, agent coordination, and user
  interaction.
- [x] Exercise environment selection, host fallback, fail-closed mode, fixed argv, and
  structured action errors through behavior tests.

## Task 4: Bind PentestAgent Workers to one attempt Env

- [x] Implement attempt filesystem/process providers over the existing
  `ShellRuntime` and `ContainerRuntime` lifecycle.
- [x] Add fixed-argv execution and bounded target-side workspace operations without
  mounts or host fallback.
- [x] Pass the attempt Env to each Worker `AsyncEngine` and register the QitOS tool
  surface without per-tool PentestAgent wrappers.
- [x] Keep redaction, audit, deadline, cancellation, artifacts, and network policy in
  PentestAgent.

## Task 5: Document and verify

- [x] Update QitOS architecture docs, changelog, and README news.
- [x] Update PentestAgent ADR/TODO/README for the Env-backed tool boundary.
- [ ] Run focused QitOS tests and stable-surface static checks.
- [ ] Run PentestAgent `make check` and review both repository diffs.

## Implementation Record

- QitOS defines generic file/process contracts and `CapabilityEnv`; it contains no
  PentestAgent or Docker-specific imports.
- `CodingToolSet(profile="workspace", allow_local_fallback=False)` is the fail-closed
  application surface. No PentestAgent per-tool wrappers are required.
- PentestAgent injects its filesystem helper at attempt startup through Docker's control
  plane, avoiding host mounts and worker-image rebuilds.
- Final QitOS and PentestAgent checks plus real-container acceptance remain pending.
