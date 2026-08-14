# Remove the legacy tracing pipeline

## Goal

Keep one observability path. `qitos.trace.TraceWriter`, canonical runtime events,
the Session Journal, and `qita` remain the supported trace and replay surfaces.
The process-global `qitos.tracing` provider/processor hierarchy is removed instead
of being maintained as a second, partial event system.

## Contract changes

- Remove `qitos.tracing`, the `Engine(..., tracing_provider=...)` argument, and the
  W&B/MLflow trace processor extras.
- Preserve `HANDOFF_START` and `HANDOFF_END` runtime events. They remain the
  canonical source for `EngineResult.handoff_traces` and persisted traces.
- Rename `EngineConfig.has_tracing_provider` to `has_trace_writer` so exported
  configuration describes the actual Engine observability owner.
- Move shared output redaction into `qitos.trace.redaction`; trace, render, and
  benchmark writers continue to redact the same sensitive fields.

This is an intentional breaking removal. No compatibility import, alias, or adapter
is retained because that would preserve the duplicate architecture.

## Implementation

- [x] Move and test the shared redaction helper.
- [x] Remove the provider integration from Engine and handoff execution.
- [x] Delete the legacy tracing package and its processor-specific tests.
- [x] Remove obsolete optional dependencies and documentation routes.
- [x] Update canonical tracing/handoff docs, changelog, README news, and the kernel
  alignment ledger.
- [x] Verify there are no live references to the removed API.

## Validation

- Focused redaction, Engine configuration, handoff, trace, and packaging tests.
- Complete QitOS pytest suite.
- Black formatting for the new module, stable-surface Flake8 and mypy, Python 3.10
  import, package build, and Twine metadata checks. Repository-wide Black is not a
  configured QitOS gate and is outside this change.
- Consuming PentestAgent `make check` after QitOS is merged to the fork `main` and
  the submodule gitlink is updated.

## Result

- `1953 passed, 47 skipped` on the complete QitOS test suite.
- Stable-surface Flake8 and mypy passed on Python 3.11.
- Python 3.10 imports, wheel/sdist builds, Twine metadata, wheel contents, and docs
  navigation passed.
