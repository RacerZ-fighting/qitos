# Remove the legacy checkpoint fork helper

## Goal

Keep `SessionJournal.fork()` as the only executable Run-branching authority. Preserve
checkpoint stores and checkpoint resume, but remove the helper that deep-copied a state
snapshot into a second checkpoint lineage without canonical model/tool transactions.

## Evidence and contract change

- QitOS production code, PentestAgent, examples, and zoo code do not call
  `fork_checkpoint()` or `list_fork_history()`.
- The only code consumer is one checkpoint-store test; the only documented consumers
  are the English and Chinese checkpoint tutorials.
- The helper copies `state_data`, task data, and History into another checkpoint. It
  does not create a Session Journal, committed continuation boundary, Run handle,
  terminal ToolResult pairing, or recoverable Engine lineage.
- `CheckpointStore`, automatic checkpoint persistence, and
  `Engine.aresume_from_checkpoint()` remain supported and unchanged.

This is an intentional breaking removal. Executable branching uses a journal-backed
Run and a committed `JournalPosition`; no compatibility alias is provided.

## Implementation

- [x] Delete `qitos.checkpoint.fork` and its package exports.
- [x] Retain store listing coverage while removing the implementation-coupled fork
  assertions.
- [x] Rewrite both checkpoint tutorials to distinguish snapshot resume from canonical
  Journal fork and lineage inspection.
- [x] Update the changelog, README news, and architecture ledgers.
- [x] Verify no removed API or packaged-wheel reference remains.

## Validation

- Focused checkpoint, Journal fork, and Run catalog tests.
- Complete QitOS pytest suite.
- Stable-surface Flake8 and mypy checks.
- Python 3.10 import check.
- Package build, Twine metadata validation, documentation navigation, and wheel-content
  inspection.
- Consuming PentestAgent `make check` after merge and gitlink update.

## Result

The snapshot-copy fork helper and exports were removed while checkpoint persistence,
listing, and Engine snapshot resume remain unchanged. The public tutorials now direct
executable branching to committed Session Journal positions and use the read-only Run
catalog for lineage inspection.

Focused checkpoint, Journal, catalog, and tutorial checks passed with 79 tests. The
complete suite passed with 1,922 tests and 47 environment-gated skips. Stable-surface
Flake8 and mypy, Python 3.10 import, package build, Twine metadata, documentation
navigation, and wheel/sdist content checks also passed. PentestAgent validation remains
a separate consumer step after merge and gitlink update.
