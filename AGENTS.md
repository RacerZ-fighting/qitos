# QitOS development instructions

This file contains repository-wide rules. Keep detailed module contracts in code/docs
and add nested instructions only when a directory has a genuinely different workflow.

## 1. Architecture authority

QitOS is a reusable Agent runtime with one target path:

```text
Model / Provider
→ minimal async Message / ToolCall / ToolResult loop
→ Agent façade / Session Harness
→ application composition
```

The contracts are in `docs/internal/architecture/agent-runtime.md`. The minimal loop,
the `Agent` façade, the journaled child path and the façade quickstart are the shipped
mainline; the retired `AgentModule / Observation / Decision / Action` lifecycle, its
Engine, checkpoint store, prompt-injected protocols/parsers/critics, recipes, and
benchmark execution adapters are removed. Remaining migration work (authoritative
Session/Harness, Task/Plan replacement) is tracked in
`docs/internal/plans/pi-aligned-agent-runtime.md`.

Hard rules:

- Maintain one implementation of each core concept. Do not add V2/legacy aliases,
  wrappers, mirror DTOs, second codecs or parallel runtimes.
- Replace old behavior in place. A replacement is complete only when callers, exports,
  tests, examples and affected docs move, and the superseded path is removed.
- Preserve verified Tool transaction, deadline, cancellation, ordering, trace and
  recovery behavior.
- New product capability must target the Model/loop/Session contracts.
- Benchmark, evaluation, recipe, renderer and product/plugin policy stay outside the
  Agent loop, Task, Session and Tool executor.

## 2. Ownership boundaries

- `qitos.core`: stable provider-neutral contracts such as Message, Task, Tool and
  Session records.
- `qitos.models`: Provider adapters and model-runtime configuration.
- Agent loop: model/Tool turn mechanics and transaction barriers only.
- Session/Harness: queue, compact, resume, fork, abort, idle and pure recovery.
- `qitos.kit`: concrete reusable Runtime, storage, Tool, Skill, MCP, Artifact, Plan and
  Child implementations.
- `examples`: runnable product-surface examples, not alternate framework layers.
- `docs`: public shipped behavior or explicitly labelled internal target architecture.

QitOS does not own PentestAgent concepts or a general product plugin loader. Products
compose QitOS primitives explicitly without arbitrary loop/Session hooks.

## 3. Standard workflow

For a non-trivial change:

1. Inspect the current owner, public callers, behavior tests and directly relevant doc.
2. State the observable change, failure semantics, migration impact and success test.
3. If the change affects architecture, a public contract, persistence, Provider
   semantics or the default runtime path, update the active plan and confirm the
   tradeoff before implementation.
4. Implement the smallest complete slice in place. Move real callers before extracting
   a new abstraction; do not prebuild a registry/factory/hook without a second use.
5. Add or update behavior tests, then run the checks appropriate to the changed layer.
6. Update only the affected docs/history surfaces, inspect the final diff and remove
   superseded code/documents from the same slice.

Small isolated fixes and docs-only edits do not require a new plan document.

## 4. Runtime and contract rules

- Async is native below the outermost application/CLI boundary. Do not call
  `asyncio.run()` or create a temporary loop inside running async code.
- Re-raise `asyncio.CancelledError` after durable terminalization and cleanup.
- Propagate absolute deadlines; child calls receive only remaining time.
- Use structured concurrency for bounded parallel work. No detached unowned tasks.
- Every ToolCall receives exactly one terminal ToolResult, including invalid, denied,
  timeout, cancelled and failed calls. Parallel results commit in input order.
- A turn freezes model, history, Tool exposure, runtime capability, deadline and budget.
- Do not wait for Model, Tool, user or cleanup while holding a Session lock or storage
  transaction.
- Imports must not start threads/tasks, connect MCP, probe environments or mutate a
  process-global registry.
- Public/cross-module objects use explicit types; validate external dict/SDK payloads at
  the boundary and preserve exception causality.

Task rules:

- Commit one goal-bearing Root Task before the first model or Tool side effect.
- A Session has at most one unfinished Root Task. Resume and non-terminal fork preserve
  Task identity; terminal follow-up creates a new Task explicitly.
- Blocked is resumable only after explicit caller input or observed external-state
  change. Completed, failed and cancelled are terminal once.
- Child launch creates a narrowed Task linked to parent Task and Plan assignment before
  runtime construction.
- Benchmark resources, environment probing, metrics and free-form metadata do not enter
  canonical Task.

Tool rules:

- Class Tools expose only `execute(args, runtime_context)`; function Tools use the
  canonical decorator path.
- Registry conflicts fail. Registration and exposure are instance-scoped and explicit.
- Model schema and execution admission use the same frozen ToolSpec/input schema.
- Tool retry, permission, timeout and concurrency behavior has one owner. Do not stack
  transport/executor/handler retry loops or infer concurrency safety from a name.
- Env-backed operations use Env capabilities and never silently fall back to the host.

## 5. Verification by change type

Run commands through `uv`, not bare Python/pip or a parent repository environment.
Use Python 3.11 for contributor checks.

Docs/comments only:

- `git diff --check`
- verify changed local links and current-vs-target wording

Focused behavior change:

- run the smallest relevant pytest selection first;
- run flake8/mypy when touching their stable surfaces.

Core/public contract, Provider, persistence, concurrency or broad refactor:

```bash
uv run --no-project --python 3.11 \
  --with-editable . \
  --with 'pytest>=7' \
  --with 'pytest-asyncio>=0.23' \
  --with 'openai>=1.66.0' \
  pytest -q

uv run --no-project --python 3.11 --with-editable . \
  --with 'flake8>=6' flake8 qitos/core qitos/models qitos/trace

uv run --no-project --python 3.11 --with-editable . \
  --with 'mypy>=1' mypy qitos/core qitos/models qitos/trace
```

Packaging/release metadata:

```bash
uv build
uv run --no-project --python 3.11 --with 'twine>=5.1.1' twine check dist/*
```

Do not claim a check passed unless it ran. Report skipped checks and why.

## 6. Documentation and history

- Update `CHANGELOG.md` for user/developer-visible behavior, API, dependency,
  performance, deprecation or removal changes.
- Update the relevant public doc when behavior/workflow changes; update internal
  architecture/plan docs when the target or migration changes.
- Update README news only for meaningful user-visible or roadmap-level progress, not
  every internal edit.
- Keep English/Chinese public docs aligned when both exist.
- Remove completed/superseded internal plan documents; Git history is the archive.
- Examples and commands must remain runnable and credentials come from environment
  variables.

## 7. Repository hygiene

- Keep changes scoped; preserve unrelated user work.
- Do not add production dependencies without a demonstrated default-path need.
- Never hide flaky behavior with retries or broad exception handling.
- Do not use destructive Git commands or amend commits unless explicitly requested.
- Review the final diff for credentials, machine-local paths, generated artifacts,
  compatibility leftovers and unrelated rewrites.
