# WorkPlan contract implementation plan

## Goal

Provide one typed, checkpoint-friendly checklist for any QitOS Agent without using
free-form state metadata or TaskBoard coordination semantics.

## Contract

- `qitos.core` owns immutable WorkPlan items, validation, serialization, Markdown
  rendering, and the pure reducer for ordered `update_plan` results.
- `qitos.kit` owns the concrete `update_plan` tool. The tool validates and returns one
  complete replacement; it never mutates state or writes a projection.
- Agent products embed `WorkPlanState` in their own `StateSchema`, call the reducer in
  `commit_action_results`, and decide where disposable projections live.
- The coding preset exposes `update_plan`; the unchecked `todo_write` metadata path is
  removed. TaskBoard remains separate.

## Verification

1. Contract tests cover bounds, uniqueness, one active item, serialization, ordered
   reduction, deterministic Markdown, and checkpoint JSON round-trip.
2. Tool tests cover schema, normalized output, invalid input, and no direct mutation.
3. Existing coding-preset tests use `update_plan` and confirm `todo_write` is absent.
4. Run the complete QitOS test suite plus stable-surface flake8 and mypy checks.

## Repository surfaces

Update the public core/kit exports, API documentation, English and Chinese coding
tutorials, changelog, and README news in the same change.
