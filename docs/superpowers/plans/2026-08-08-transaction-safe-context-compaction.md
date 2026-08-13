# Transaction-Safe Context Compaction

## Goal

Keep long-running QitOS agents below a predictable provider input boundary without
splitting tool exchanges, silently deleting canonical history, or relying on a
provider overflow as the normal trigger.

The design combines the transaction boundaries and bounded shrink retries used by
Kimi Code, the clone/project-then-replace discipline used by Codex, and the staged
compaction behavior verified in the local Claude Code reference. QitOS does not copy
or depend on those projects.

## Success criteria

- [x] Define 80% against the provider-safe input budget, not the raw context window.
- [x] Count complete wire input, including native tool and response schemas.
- [x] Keep assistant calls and their results atomic across every built-in history window.
- [x] Apply three bounded compaction levels without arbitrary message slicing.
- [x] Keep canonical history immutable during compaction and provider projection repair.
- [x] Reject stale summary work when the source history version changes.
- [x] Reuse bounded summary checkpoints only when their exact source remains a prefix.
- [x] Use `.70/.50/.35` transaction-safe projections for bounded reactive recovery.
- [x] Emit budget source, occupancy, compaction level, source version, and digest telemetry.
- [x] Run the repository's focused tests and changed-surface lint gate; record the
      repository's unrelated collection and static-analysis baselines separately.

## Runtime contract

1. Resolve the hard input budget from an explicit model `max_input_tokens`, or derive
   it from `context_window - max_output_tokens - safety_reserve`.
2. Build tool request options before history retrieval so schema tokens reduce the
   history allowance.
3. Ask the configured history strategy for a projection whose history plus pending
   content fits the 80% total-input boundary.
4. Assemble and count the complete provider packet. At or above 80%, do not call the
   model; retry the projection with the bounded reactive factors.
5. If the latest complete transaction and fixed anchors cannot fit after all retries,
   stop with an explicit context-overflow reason.

## Three levels

1. **Microcompact** only large content in older complete rounds.
2. **Summarize older prefix** while preserving `keep_last_rounds` verbatim.
3. **Resummarize all older context** while preserving the latest complete round.

No level edits the stored message list. A summary is a request projection tied to a
source version and digest. A concurrent write invalidates it before exposure. Later
projections reuse the checkpoint only after verifying the exact canonical prefix, then
summarize only newly covered complete rounds.

## Verification focus

- Exact 79%/80% boundary behavior.
- Tool and response schema token accounting.
- Generic, parallel, cross-step, and Responses API tool transactions.
- Summary failure, empty output, cache reuse, and concurrent append rollback.
- Three overflow retries without canonical history mutation.
- Focused pytest and flake8 checks for the changed stable runtime surface.
