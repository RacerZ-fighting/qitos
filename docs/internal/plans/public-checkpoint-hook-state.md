# Public checkpoint state for hooks

## Goal

Let product hooks observe whether a durable checkpoint advanced without reading an
Engine private field. This supports post-commit projections while keeping the
checkpoint store and commit ordering inside QitOS.

## Contract

1. `Engine.last_checkpoint_id` is a read-only property.
2. It is `None` before the first successful checkpoint and becomes the exact committed
   checkpoint id only after `CheckpointStore.put()` succeeds.
3. Resume initializes it to the checkpoint being resumed.
4. No setter, receipt type, or second checkpoint lifecycle is added.

## Verification

- Exercise the property against a real Engine and `InMemoryCheckpointStore`.
- Run the checkpoint boundary tests, stable static checks, and the QitOS test suite.
