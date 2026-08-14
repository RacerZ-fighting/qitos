# Remove the legacy debug and trace-mutation path

## Goal

Keep `qita` as a read-only projection of canonical trace artifacts and keep
`SessionJournal` as the only owner of executable resume/fork lineage. Remove the
deprecated `qitos.debug` package and qita's endpoint that copied and modified trace
files without creating a real Run.

## Contract changes

- Remove `qitos.debug`, including its separate replay session, breakpoint, and
  inspector models.
- Remove qita's `POST /api/fork/{run_id}/{step_id}` endpoint and replay-page fork
  buttons. The endpoint did not fork an Engine or Journal and produced artifacts
  whose lineage and completion state were not trustworthy.
- Preserve qita board, run inspection, replay, browser-local breakpoints, compare,
  and export. Real resumable forks continue through `SessionJournal.fork()` at a
  committed continuation boundary.

This is an intentional breaking removal. A compatibility module would retain the
second replay model and is therefore not provided.

## Implementation

- [x] Delete `qitos.debug` and its qita import.
- [x] Delete the trace-mutation HTTP route and replay-page UI action.
- [x] Update documentation, changelog, README news, and architecture ledgers.
- [x] Verify qita remains read-only and no removed API references remain.

## Validation

- Focused qita, trace, Journal catalog, and fork behavior tests.
- Complete QitOS pytest suite and stable-surface Flake8/mypy checks.
- Python 3.10 import, package build/Twine checks, docs navigation, and wheel-content
  inspection.
- Consuming PentestAgent `make check` after merge and gitlink update.

## Result

The deprecated debug package and qita mutation path were removed. Qita's existing
read-only board, replay, compare, and export behavior remains available. Requests to
the removed POST endpoint fail without creating artifacts, while executable lineage
continues to be owned by `SessionJournal` committed continuation boundaries.

The focused and complete test suites, stable-surface lint and type checks, Python
3.10 import check, package build and Twine validation, documentation navigation, and
wheel-content inspection all passed. PentestAgent validation remains a separate
consumer step after the QitOS commit is merged and its gitlink is updated.
