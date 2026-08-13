# Managed process runtime plan

## Scope and ownership

QitOS owns the reusable process lifecycle. The stable contract belongs in
`qitos.core`; the local asyncio implementation and shell tools belong in
`qitos.kit`. Product-specific authorization, environment profiles, and command
policy remain outside this runtime.

The implementation follows three proven behavioral boundaries:

- Pi's shell path keeps the tool contract small, streams output, propagates
  cancellation, and makes truncation explicit.
- Codex unified exec gives one owner responsibility for spawn, bounded output,
  per-process interaction serialization, terminal observation, and cleanup.
- Hermes confirms the Python-specific need for opaque handles, foreground and
  background reuse, process-tree cleanup, incremental reads, and terminal
  retention. Its thread-based global registry and broad policy surface are not
  adopted.

No reference project is a runtime dependency and no source is copied.

## Success conditions

1. Foreground and background host commands use asyncio subprocess primitives.
2. Every background process has one opaque handle, one owner Run, a bounded
   incremental output view, and a persisted full-output log.
3. Reads and writes for the same process are serialized without holding the
   registry lock across I/O.
4. Timeout, cancellation, explicit termination, and Run shutdown reap the
   process group and await reader/watcher cleanup.
5. A process publishes exactly one terminal snapshot. No Task remains pending
   after runtime shutdown.
6. `process.started` and `process.terminal` are durable Journal records. Resume
   marks an unpaired started record as `lost`; fork does not inherit a live
   handle.
7. Shell tools expose start, list, read/poll, write, wait, and terminate through
   the environment process capability. Tool results retain input order and the
   existing action Journal remains authoritative for each invocation.

## Implementation slices

### 1. Typed process kernel

- Add immutable process handle, status, output, and snapshot types.
- Add async managed-process operations to the command capability.
- Implement a Run-scoped host process manager with bounded buffers, log files,
  process-group cleanup, and deterministic terminal state.
- Test incremental output, UTF-8 chunk boundaries, stdin, timeout,
  cancellation, process limits, and shutdown.

### 2. Journal and tools

- Add process lifecycle record types and idempotent replay projection.
- Pass the active Run and Journal through the existing Tool runtime context.
- Replace detached `Popen` background execution with the manager.
- Add process list/read/write/wait/terminate tools that share the same
  capability instance.
- Test start-record failure cleanup, one terminal record, resume loss recovery,
  fork isolation, and Engine shutdown.

### 3. Product adoption

- Update PentestAgent to compose QitOS process tools directly.
- Keep only engagement authorization, Kali/runtime selection, and domain audit
  facts in PentestAgent.
- Remove replaced local background-process ownership after contract tests pass.

## Validation

Each QitOS slice runs focused async tests, then the complete QitOS pytest suite,
Flake8, and mypy. QitOS commits land on this feature branch before merging into
the local fork `main`. PentestAgent then updates its gitlink, runs focused
consumer tests, and finishes with `make check`.
