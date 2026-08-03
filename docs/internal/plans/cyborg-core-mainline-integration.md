# Cyborg Core Mainline Integration

## Goal

Create a reviewable QitOS branch that keeps the Cyborg `core` runtime and its
canonical tool-history delivery while incorporating the fixes currently unique
to `main`.

## Baseline

- Common ancestor: `e82ef2f173951fdb04d8cdd62e8336f17aac9b44`
- Runtime baseline: `core@05a1c821f627bb31b6eda6e7fc5d3770b9167cba`
- Fix source: `main@9b67dc8917760f1f7a4e0144ba9834dab6618ab8`
- Integration branch: `feat/cyborg-core-mainline`

## Success criteria

- Replay the 12 commits unique to `main` onto `core` without dropping either
  side's runtime behavior.
- Preserve `MessageBuilder`, `runtime_context_delivery="merge_tool"`, and
  parser-action projection as canonical `assistant(tool_calls) -> tool`
  history.
- Preserve the OpenAI-compatible provider and JSON repair fixes from `main`.
- Run the QitOS test suite and stable-surface static checks, with any
  pre-existing optional-component or static-analysis blockers recorded.
- Update PentestAgent to the validated gitlink and pass its `make check` gate.

## Execution

- [x] Rebase `main` commits onto `core` and resolve conflicts surgically.
- [x] Run focused message-history and provider tests.
- [x] Run QitOS full validation.
- [x] Update and validate PentestAgent integration.
- [x] Push the validated QitOS branch to the PentestAgent-owned fork.
- [ ] Push the PentestAgent update.

## Publication

- Fork: `https://github.com/RacerZ-fighting/qitos`
- Branch: `feat/cyborg-core-mainline`

## Conflict policy

For runtime conflicts, retain the `core` architecture and integrate the narrow
bug fix from `main`. Do not restore the older message assembly path or
benchmark-specific runtime coupling removed by `core`.

## Verification

- Focused runtime, provider, Responses API, history, protocol, recovery, and
  cancellation suites: `154 passed`.
- Full collection is blocked by the out-of-sync optional `qitos_zoo` auditor
  package and an unavailable PentAGI package.
- Remaining collectable suite: `1661 passed, 50 skipped, 121 failed`; failures
  are the existing optional zoo/PentAGI surface plus four core/environment
  baseline failures (`python` command availability, missing `setuptools`, the
  workspace path-guard expectation, and `concurrency_safe`'s default).
- `flake8` is unavailable in the integration environment. Stable-surface
  `mypy` was run and reports the existing core baseline debt (`161 errors`),
  including the known incomplete `_EngineProtocol` typing surface.
- PentestAgent integration gate: Ruff lint/format, ty, and mypy passed; pytest
  completed with `579 passed, 27 skipped`.
