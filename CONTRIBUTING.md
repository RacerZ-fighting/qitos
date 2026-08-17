# Contributing to QitOS

Participation in this project is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Ways to contribute

- Improve documentation, examples, and walkthroughs.
- Add or refine Tools, environment integrations, and Provider adapters.
- Fix bugs or tighten Agent, Tool, journal, and recovery contracts.
- Improve tests, release hardening, and qita observability.

Product-grade agents and application-specific orchestration belong in the independent
`qitos-zoo` repository. QitOS owns reusable runtime primitives, not product policy or a
second application workflow engine.

## Development setup

For a source checkout:

```bash
git clone https://github.com/Qitor/qitos.git
cd qitos
pip install -r requirements-dev.txt
pre-commit install
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for the local workflow and troubleshooting.

## Branches and commits

Use short branches such as `feat/<topic>`, `fix/<topic>`, `docs/<topic>`,
`refactor/<topic>`, or `chore/<topic>`.

Use imperative Conventional Commit summaries, for example:

- `feat: add model response summary to qita`
- `fix: preserve terminal tool results on cancellation`
- `docs: clarify tool composition`

## Pull requests

Before opening a PR:

- Keep changes on the Model → minimal Agent loop → Session/Harness architecture in
  [ARCHITECTURE.md](ARCHITECTURE.md).
- Do not reintroduce the retired AgentModule/Engine lifecycle, recipes, parsers,
  critics, checkpoint owner, or benchmark execution adapters.
- Keep Provider, Tool, Session, application, and observability responsibilities
  separate.
- Add behavior tests for public changes and update affected English and Chinese docs.
- Update [CHANGELOG.md](CHANGELOG.md) for user-visible behavior or API changes.
- Avoid unrelated cleanup and new dependencies.

Reviewers check correctness, failure and cancellation semantics, persistence and
recovery boundaries, public surface size, documentation accuracy, and regression risk.

Run the repository checks from [AGENTS.md](AGENTS.md) before requesting review. The
current core checks cover `qitos/core`, `qitos/models`, and `qitos/trace`; deleted
packages must not be kept in lint, type-check, coverage, or packaging commands.

## Contribution boundaries

### Core runtime

Core changes should preserve one provider-neutral async path and its typed messages,
events, Tool transactions, cancellation, deadlines, immutable turn snapshots, and
fault propagation. New loop hooks or parallel runtimes require an established caller
and an architecture decision.

### Tools and environments

Tools use the canonical asynchronous contract and declare strict input schemas. Model
exposure and execution admission use the same frozen `ToolSpec`. Environment-backed
operations use verified environment capabilities and never silently fall back to the
host.

For a Tool contribution, include tests for schema validation, permission denial,
success, failure, timeout, cancellation, and cleanup where applicable. Security-
sensitive Tools must be explicit opt-in imports.

### Providers

Provider adapters translate SDK data into QitOS message and stream-event types. They
must preserve provider identity, usage, reasoning/thinking data, Tool call identity,
continuation, terminal failures, and cancellation without leaking SDK objects into
canonical state.

### Journals and recovery

Journal changes need round-trip and corruption tests. Recovery is pure: it must not
construct a live Agent, replay side effects, repair canonical JSONL during reads, or
guess missing Tool results.

### Documentation

QitOS maintains English documentation in `docs/` and Chinese documentation in
`docs/zh/`. Keep paired pages aligned, update `docs/docs.json` when navigation changes,
and verify every published command against a source checkout or installed package as
documented.

## Reporting issues

Bug reports should include the QitOS version, Python version, minimal reproduction,
expected and actual behavior, traceback, and relevant environment details with secrets
redacted.

Feature requests should explain the use case, required observable behavior, and any
alternatives considered. Security issues follow [SECURITY.md](SECURITY.md).
