# Repair and manually trigger the CI quality gate

## Decision

Keep the existing validation jobs, but run the CI, docs, and contribution workflows
only through GitHub Actions `workflow_dispatch`. Repair the declared test and audit
environments so a manual run executes async tests and audits a non-vulnerable build
toolchain. Release-triggered package publication remains unchanged.

## Failure evidence

- The Actions test environment installs `.[dev,models,benchmarks]`, while the `dev`
  extra omits `pytest-asyncio`. Async tests therefore emit unknown-mark warnings and
  are not guaranteed to execute under their intended event-loop integration.
- The audit job scans its complete installed environment. GitHub's Python 3.11 image
  retained `setuptools==79.0.1`, which is affected by `PYSEC-2026-3447`; the fixed
  release starts at 83.0.0.
- Once async tests execute, Python 3.10 reaches MCP and Child paths that used
  Python-3.11-only `asyncio.TaskGroup` and `asyncio.timeout`. Directly cancelled Tasks
  also normalize `CancelledError` subclasses at their outer Task boundary on 3.10.
- The contribution schema job imports `ToolSpec` from an obsolete module, and the docs
  parity job detects a missing Chinese mirror for the multi-agent tutorial.
- Push and pull-request triggers spend the full matrix on every branch update. The
  repository owner requires this gate to be explicitly dispatched instead.

## Scope

1. Declare `pytest-asyncio` in the development test dependencies.
2. Require fixed `setuptools` for builds and explicitly install it in the audited
   environment before the editable project install.
3. Replace automatic push and pull-request validation triggers with
   `workflow_dispatch`; remove pull-request-only conditions from the contribution
   workflow so manual jobs remain executable.
4. Preserve the existing test, coverage, package, lint, type, audit, docs, and focused
   contribution jobs.
5. Express MCP startup cleanup and Child deadlines with structured asyncio primitives
   supported on Python 3.10, and verify durable cancellation outcomes across the
   version-specific Task boundary behavior.
6. Repair the contribution schema import and complete the documented bilingual mirror
   so both manually dispatched workflows validate their intended contracts.

## Verification

- Collect and execute the affected async tests without unknown-mark warnings.
- Run the test suite under the supported Python matrix.
- Reproduce the audit job and confirm `pip-audit` exits successfully.
- Run the stable-surface flake8 and mypy checks and validate package artifacts.

## Progress

- [x] Identify the undeclared async test plugin and vulnerable build-tool version.
- [x] Update dependency metadata and the workflow trigger/install steps.
- [x] Pass the equivalent test, coverage, contribution, docs, audit, static, and
  package checks.
