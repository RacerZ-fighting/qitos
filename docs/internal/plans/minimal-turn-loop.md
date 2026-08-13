# Minimal turn loop

## Goal

Make one transactional turn the canonical execution unit for both `Engine.astep()` and
`Engine.arun()`. Keep the loop readable as safe-point input, immutable turn capture,
provider decision, tool lifecycle, reducer commit, and stop assessment. Legacy critic
and handoff behavior remains available through composed control runtimes instead of
being reimplemented in the loop.

The boundary follows the small sampling/tool loop in `pi:` and the request-scoped
turn/step snapshots in `codex:`. Those projects remain design references, not runtime
dependencies.

## Change layers

1. Add an internal turn transaction result with explicit `continue`, `stop`, `wait`,
   `handoff`, `retry`, and `recovered` outcomes.
2. Move one-turn orchestration out of `Engine._arun_impl()` and make the run loop consume
   that result without duplicating decide/act/reduce/commit behavior.
3. Keep parser, critic, handoff, tracing, environment, and compatibility mechanics in
   their existing focused runtimes; the transaction only invokes their contracts.
4. Make `Engine.astep()` use the same turn transaction so public single-step and full-run
   behavior cannot drift.

## Success conditions

- `Engine._arun_impl()` contains lifecycle setup/cleanup plus a small loop over a single
  turn executor; it does not implement tool or critic transactions inline.
- Each turn receives one immutable `TurnSnapshot`; model dispatch and tool execution use
  that exact value.
- Journal order, cancellation terminal completion, final completion assessment, wait,
  critic retry, handoff, recovery, checkpoint, trace, and hook behavior stay covered by
  behavior tests.
- No temporary event loop, mirror public API, compatibility wrapper, or new plugin
  abstraction is introduced.

## Verification

- Run focused Engine, Journal, handoff, critic, wait, cancellation, and turn-policy tests.
- Run the complete QitOS pytest suite and stable-surface flake8/mypy checks.
- Update Engine concepts, changelog, and README news after the code boundary is stable.

## Progress

- [x] Define the internal turn transaction contract.
- [x] Route full runs and public single-step execution through it.
- [x] Preserve legacy control behavior through existing composed runtimes.
- [x] Pass focused and full QitOS validation.
- [x] Merge into QitOS `main` and update PentestAgent.
