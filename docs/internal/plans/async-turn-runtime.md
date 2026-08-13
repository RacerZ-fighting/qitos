# Async turn runtime

## Goal

Make the canonical `AgentModule + Engine` loop async from provider dispatch through
tool, mailbox, and child execution. Capture one immutable turn view and preserve
durable tool-terminal and completion semantics across cancellation and resume.

The design follows the minimal turn boundary represented by `pi:` and the explicit
turn-context, ordered execution, cancellation, and recovery invariants represented by
`codex:`. QitOS remains the implementation owner; the reference projects are not
runtime dependencies.

## Change layers

1. Core contracts: async tool and environment operations, immutable `TurnSnapshot`,
   typed completion assessment, and run budget fields.
2. Engine runtime: await model/tools/mailbox/children on one event loop, propagate one
   absolute deadline, submit ordered terminal results, and apply external input only at
   turn safe points.
3. Persistence: restore token and cost usage from journaled completed model records and
   never replay a handler whose durable `tool.started` record exists.
4. Product composition: let PentestAgent assess completion from investigation state and
   configure run-wide step, time, token, cost, tool, and child limits without wrapping
   QitOS contracts.

## Success conditions

- No temporary event loop or executor-owned thread boundary exists in the Engine action
  path. Existing synchronous function tools use one isolated compatibility boundary.
- Parent cancellation waits for every started handler to clean up and produces one
  terminal result per call in source order.
- A turn carries the exact model, protocol, history, tool exposure, runtime capability,
  deadline, and remaining budget view used by both model and action phases.
- A proposed final answer can be accepted, rejected with feedback for another turn, or
  classified as blocked. Stop reasons distinguish completion, blocking, budget,
  cancellation, and failure.
- Journal resume restores accumulated token and cost usage before the next turn.
- Focused async, cancellation, ordering, snapshot, completion, budget, and journal tests
  pass, followed by the full QitOS quality gates.

## Assumptions and constraints

- Class-based tools adopt the async contract directly. Decorated synchronous functions
  remain compatible through `asyncio.to_thread()` and cannot be force-cancelled; the
  runtime waits for their side effects to reach a known terminal boundary.
- Tool calls are serial unless their frozen spec explicitly declares concurrency safety.
- Cost limits require explicit pricing or provider-reported cost data. The runtime does
  not guess prices from model names.
- Configuration and mailbox changes observed during a turn affect only the following
  turn.

## Verification

- Run focused pytest modules while changing each contract.
- Run the complete QitOS pytest suite.
- Run QitOS flake8 and mypy stable-surface checks.
- Review the final diff for temporary artifacts, reference-machine paths, and unrelated
  changes.

## Progress

- [x] Capture immutable turn inputs and remove Agent `_runtime_*` turn attributes.
- [x] Await ActionExecutor and mailbox on the caller's event loop.
- [x] Journal mailbox acceptance and restore only events not bound to a completed turn.
- [x] Let synchronous Engine hooks defer runtime input for async persistence at the
  next turn safe point without spawning an unowned Task.
- [x] Add async host and Docker command operations with process-group cleanup.
- [x] Complete tool and child lifecycle cleanup and migrate tests.
- [x] Add completion assessment, complete budget accounting, and journal restoration.
- [x] Update docs, changelog, and README news.
- [x] Pass focused and full QitOS validation.
- [x] Merge the QitOS feature branch and update the PentestAgent consumer.
