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
- Pass the QitOS test suite and stable-surface static checks.
- Update PentestAgent to the validated gitlink and pass its `make check` gate.

## Execution

- [ ] Rebase `main` commits onto `core` and resolve conflicts surgically.
- [ ] Run focused message-history and provider tests.
- [ ] Run QitOS full validation.
- [ ] Update and validate PentestAgent integration.
- [ ] Push the validated QitOS branch and PentestAgent update.

## Conflict policy

For runtime conflicts, retain the `core` architecture and integrate the narrow
bug fix from `main`. Do not restore the older message assembly path or
benchmark-specific runtime coupling removed by `core`.
