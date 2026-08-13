# Stable trace task fingerprint

## Goal

Keep trace comparison provenance stable when two runs execute the same semantic
task under different run-instance identifiers, while still rejecting comparisons of
different tasks or different runtime configurations.

## Modification layers

1. Task coercion: give plain-text tasks a content-derived stable task ID instead of a
   wall-clock-derived ID; the trace Run remains the execution-instance identity.
2. Trace runtime: derive the task fingerprint from the complete serialized `Task`, and
   derive the run-configuration fingerprint from runtime metadata without task data.
3. Trace manifest: persist the task fingerprint as its own provenance field while
   retaining `task_id` in the summary metadata.
4. Qita comparison: require and compare the task fingerprint independently of the
   run-configuration fingerprint.
5. Tests and documentation: cover stable semantic identity, configuration changes,
   missing provenance, and runs whose generated task IDs differ.

## Success criteria

- [x] Equivalent plain-text tasks have the same generated task ID and fingerprint
  across different wall-clock times.
- [x] Explicit structured `Task.id` values remain part of task identity and the task
  fingerprint.
- [x] A semantic task change changes the task fingerprint.
- [x] A runtime-configuration change preserves the task fingerprint and changes the
  configuration fingerprint.
- [x] New manifests expose `task_hash`; legacy manifests without it remain readable, but
  qita marks them as incomplete comparison provenance.
- [x] Two equivalent runs started at different wall-clock times compare as `same_spec`
  without relying on scheduler timing.
- [x] Focused tests, the full QitOS suite, flake8, and mypy pass.

## Assumptions

- `Task.id` is the stable task/sample identity. It is part of the canonical Task
  package and remains significant for benchmark cases. Trace and Journal `run_id`
  values identify execution instances.
- Trace schema `v1` already exists in stored artifacts, so adding `task_hash` to the
  writer must not make old manifests unreadable. Comparison eligibility can safely be
  stricter than basic manifest readability.
- Hashes are provenance fingerprints, not authentication or secrecy boundaries.

## Reference notes

- `pi:` keeps Session identity and creation time in Session metadata, gives each
  operation its own run identity, and records model/thinking configuration changes as
  separate entries.
- `codex:` stores canonical thread/session identity separately from timestamps and
  turn configuration. QitOS applies that separation to trace provenance rather than
  copying either reference project's storage format.

## Validation

- Focused trace-runtime, harness comparison, qita, and trace-writer tests.
- QitOS full pytest suite.
- QitOS stable flake8 and mypy gates.
