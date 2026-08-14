# Distinguish foreground Child and caller cancellation

## Problem

A foreground Child's product-owned construction or cleanup can terminate without the
parent Task being cancelled. Reporting that outcome as `CancelledError` makes
`ChildSupervisor.launch()` misclassify it as caller cancellation and abort the Root Run.
Background Children already keep the terminal Child result separate from the parent.

## Change

- Add a typed `ChildInvocationCancelled` signal for product invocation factories.
- Reduce that signal to an already-persisted `ChildStatus.CANCELLED` result.
- Continue to re-raise `asyncio.CancelledError` exclusively as caller cancellation.
- Cover both Child-local cancellation and real caller cancellation behavior.

## Acceptance

- Typed Child-local cancellation produces one terminal result without cancelling the
  parent.
- Real caller cancellation still propagates after terminal persistence.
- QitOS and PentestAgent quality gates pass independently.
