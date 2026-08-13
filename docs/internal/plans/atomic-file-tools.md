# Atomic file tools

## Goal

Make QitOS's canonical file tools safe under concurrent agent activity without adding
a second filesystem abstraction or a patch language.

The behavior follows the per-environment, per-canonical-path mutation ordering used by
`pi:` and the validate-then-commit boundary used by `codex:`. QitOS keeps its existing
exact-text edit surface and environment capability boundary.

## Success criteria

- [x] A bounded text read returns a stable SHA-256 revision for the complete file.
- [x] Host and Docker file capabilities expose one atomic UTF-8 replacement operation.
- [x] `write_file` can require an expected revision and never leaves a partial target.
- [x] `edit_file` uses the revision it read as an implicit compare-and-swap guard.
- [x] Mutations to one canonical path are serialized within an environment instance.
- [x] Traversal and symlink escapes fail without writing outside the capability root.
- [x] Contract tests cover create, replace, conflict, concurrent edits, UTF-8, line
      endings, and failed commit cleanup.
- [x] Public docs, changelog, and README news describe the behavior.

## Boundaries

- Keep `FileSystemCapability` as the only generic filesystem contract.
- Do not add a new patch DSL, filesystem manager, registry, or PentestAgent adapter.
- Keep the existing exact-text `edit_file` input. Preserve `expected_mtime` only as a
  compatibility check while exposing SHA-256 as the durable precondition.
- Atomicity is scoped to one filesystem backend operation. Cross-host distributed
  locking is out of scope.

## Verification

Run focused filesystem and coding-tool tests first, then the repository pytest,
flake8, and mypy gates from `AGENTS.md`.
