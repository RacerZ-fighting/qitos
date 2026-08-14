# Remove the legacy functional Agent API

## Goal

Keep `AgentModule + Engine` and the canonical function-tool decorator as QitOS's only
execution paths. Remove `qitos.func`, whose `@agent`, `@task`, inference, and composition
helpers execute ordinary Python functions outside Engine transactions.

## Evidence and contract change

- QitOS production code, PentestAgent, examples, and zoo code do not import
  `qitos.func`; its only consumer is its own test module.
- `AgentFunction.__call__()` and `TaskFunction.__call__()` invoke host functions
  directly, without Provider, Tool, Journal, deadline, cancellation, permission, or
  trace ownership.
- `_FunctionAgent.reduce()` never executes the wrapped function, so converting an
  `AgentFunction` to an `AgentModule` does not provide the behavior advertised by the
  package.
- `_Composer` owns a `ThreadPoolExecutor` without a lifecycle and implements a second
  concurrency policy outside the canonical Tool executor.

This is an intentional breaking removal. Agent implementations use `AgentModule` with
`Engine`; ordinary callable Tools use the maintained function-tool decorator. A shim
would preserve the parallel execution model and is therefore not provided.

## Implementation

- [x] Delete the `qitos.func` package and its implementation-coupled tests.
- [x] Record the public removal in the changelog, README news, and architecture ledger.
- [x] Verify no production, documentation, example, or packaged-wheel reference
  remains.

## Validation

- Complete QitOS pytest suite.
- Stable-surface Flake8 and mypy checks.
- Python 3.10 import check.
- Package build, Twine metadata validation, documentation navigation, and wheel-content
  inspection.
- Consuming PentestAgent `make check` after merge and gitlink update.

## Result

The decorator-driven Agent/task package and its private concurrency path were removed.
The canonical `AgentModule + Engine` runtime and maintained `@function_tool` decorator
remain available and unchanged.

The complete QitOS suite passed with 1,922 tests and 47 environment-gated skips. The
stable-surface Flake8 and mypy checks, Python 3.10 import, package build, Twine metadata,
documentation navigation, and wheel/sdist content checks also passed. PentestAgent
validation remains a separate consumer step after the QitOS commit is merged and its
gitlink is updated.
