# Run Session catalog

## Goal

Make a persisted QitOS Run discoverable and inspectable without acquiring its writer
lease or constructing an Agent. A caller should be able to locate the latest durable
position, the latest forkable committed boundary, terminal status, and immediate
fork lineage before deciding whether to resume or fork.

Pi provides the small append-only Session manager and explicit parent Session behavior.
Codex provides the separation between canonical rollout history and a read/list store,
plus fail-closed lineage and committed snapshot semantics. QitOS keeps JSONL canonical
and does not adopt either reference's product UI or storage layout.

## Change layers

1. Add immutable core Run summary and fork-origin types with explicit lifecycle state.
2. Extract one shared strict JSONL reader so the writer may repair a torn tail while a
   catalog reader observes the same prefix without mutating it.
3. Add a concrete asynchronous JSONL Run catalog that can inspect, list, traverse
   ancestors, and find children without a writer lease.
4. Reuse a current SQLite projection read-only when it exactly matches the canonical
   file; otherwise read JSONL without rebuilding or changing either artifact.
5. Allow a forked Run to be forked again when its local boundary is an inherited
   committed record. Keep the immediate parent and exact local cutoff in the child.

## Success conditions

- A catalog can inspect a Run while its Engine writer remains active.
- Reads never create, repair, truncate, rebuild, or update Journal artifacts.
- A torn final line is omitted from a read snapshot; corruption before it fails closed.
- Handles expose the exact latest and last committed `JournalPosition`, lifecycle,
  stop reason, task input, timestamps, and optional immediate fork origin.
- Listing is deterministic; lineage rejects cycles and missing parents instead of
  inventing ancestry.
- Nested forks remain independently replayable after their ancestors are unavailable.
- Terminal resume still executes no Provider or Tool; continued work starts from an
  explicit committed-boundary fork.
- No second transcript, state store, Session manager, or Engine path is introduced.

## Verification

- Focused Journal/catalog/fork/recovery tests, including concurrent writer reads.
- Complete QitOS pytest plus stable flake8/mypy checks.
- PentestAgent consumer tests and `make check` after its gitlink is updated.
- Configured live model protocol matrix and isolated remote Docker/Kali smoke before
  this feature branch enters QitOS `main`.

## Progress

- [x] Inspect Pi/Codex Session, lineage, store, and fork boundaries.
- [x] Add immutable Run summary contracts.
- [x] Share strict read/repair parsing without weakening corruption behavior.
- [x] Implement the lease-free catalog and read-only projection path.
- [x] Support nested committed-boundary forks.
- [x] Add behavior tests and public documentation.
- [ ] Pass local and remote acceptance before merge.

## Stable lineage handle extension (2026-08-15)

PentestAgent now has a concrete product consumer that must identify one stable Session
across committed-boundary forks. QitOS will extend the existing Run contract in place:

1. `Engine.arun()` accepts an optional non-empty `lineage_id`; when omitted, the Root
   Run id is the lineage id.
2. `run.started` persists that value. A fork inherits it from the source metadata.
3. `RunHandle` exposes the typed `lineage_id` and immediate `parent_run_id`; catalog
   lineage and children remain the only authority for ancestry traversal.
4. Terminal Runs remain non-resumable. Continued work is an explicit fork whose new
   handle is resumable and shares the lineage id.

Success requires behavior tests for explicit/default ids, fork inheritance, legacy
journals, terminal resume, catalog projections, and corrupt lineage metadata. Public
docs, English/Chinese concepts, README news, and CHANGELOG must remain aligned.

### Extension progress

- [x] Persist explicit and default lineage ids without adding a second Session store.
- [x] Preserve lineage and immediate parent identity across nested forks.
- [x] Keep audit and non-terminal continuation positions distinct.
- [x] Cover completed Runs, pre-snapshot terminal commits, legacy Journals, and corrupt
  lineage metadata with behavior tests.
- [x] Pass the complete QitOS test suite plus Black, Flake8, mypy, and Python 3.10
  import checks.
- [x] Update public concepts, README news, and CHANGELOG.
