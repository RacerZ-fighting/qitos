# Bounded child-agent lifecycle

## Goal

Make background `AgentTool` work obey its owner's concurrency, cancellation, and
process-lifecycle boundaries without introducing a second agent loop.

## Scope

- [x] Snapshot launch-time parent history without constructing the child invocation.
- [x] Construct each model, Engine, and trace only after `execution_scope` admission.
- [x] Replace non-daemon executor workers with one small bounded daemon pool.
- [x] Propagate queued and running cancellation with canonical stop reasons.
- [x] Store terminal result state before attempting parent event delivery.
- [x] Let a later bounded `close()` wait after an earlier `close(wait_seconds=0)`.
- [x] Run focused child-agent tests and changed-surface lint.

## Non-goals

- No persistent child task service, resume protocol, or cross-process worker manager.
- No event retry subsystem; Engine event IDs already provide duplicate rejection, and
  completed results remain queryable by task id.
- No forced Python thread termination. Cancellation remains cooperative, while daemon
  ownership guarantees an unresponsive child cannot own process exit.

## Acceptance

A queued child allocates no invocation, parent history remains launch-stable, parent
cancellation reaches queued and running work, terminal queries cannot race event delivery,
and an unfinished synchronous child does not prevent interpreter shutdown.
