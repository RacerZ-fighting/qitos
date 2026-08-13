# Typed child lifecycle

## Goal

Give QitOS one explicit Child Agent vocabulary before moving lifecycle supervision out
of the model-facing `AgentTool`. A Child is an independent Engine Run with a stable,
parent-scoped handle; it is not a live Agent object stored in a request or result.

The boundary follows the small delegation surface in `pi:` and the independent child
session, stable handle, explicit status, and narrowed launch configuration in `codex:`.
Those projects remain design references, not runtime dependencies.

## Change layers

1. Define immutable core launch, invocation, handle, status, conclusion, and result
   contracts with strict persistence codecs.
2. Migrate `AgentTool` to those contracts and remove its superseded request/result
   types and string task identifiers.
3. Extract Run-owned async supervision so model-facing create and control tools share
   one lifecycle implementation.
4. Add explicit wait, message/follow-up, and interrupt operations, then journal the
   recoverable Child projection without serializing or reattaching live Engines.

## Success conditions

- Each child has independent Engine, History, budget, cancellation, and Run identity.
- Child authorization, tool groups, budget, depth, and working directory can only be
  narrowed by the owning product composition.
- Every started child reaches one typed terminal status and remains queryable even if
  parent mailbox delivery fails.
- Closing a Run cancels and drains every owned child task; no temporary event loop or
  detached task is created.
- Resume and fork preserve facts but never imply authority over a live parent process.

## Verification

- Run core codec and AgentTool lifecycle tests, including ordering, admission,
  cancellation, completion delivery, and shutdown.
- Run the complete QitOS pytest suite and stable-surface flake8/mypy checks.
- Update the public kit reference, changelog, and README news for each completed layer.

## Progress

- [x] Define and test canonical immutable Child contracts.
- [x] Migrate `AgentTool` to handles, typed statuses, conclusions, and `TaskBudget`.
- [ ] Extract one Run-owned async supervisor.
- [ ] Add wait, message/follow-up, and interrupt projections.
- [ ] Journal lifecycle facts and define resume/fork behavior.
- [ ] Adopt the canonical lifecycle directly in PentestAgent `GeneralAgent`.
