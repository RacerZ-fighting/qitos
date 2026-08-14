# Remove the parallel experiment runtime

## Decision

Remove `qitos.experiment`, `qit experiment`, and the disconnected `qitos.config`
surface. Keep canonical benchmark execution, `ExperimentSpec`, `BenchmarkRunResult`,
model factories, provider constructors, Skill YAML manifests, and MCP's official
Pydantic protocol models.

## Evidence

- `qit experiment` is the only runtime caller of `qitos.experiment` and the YAML
  loader; `qitos-zoo` has no caller.
- The runner bypasses `AgentModule.run()`, canonical Run specs, Journal ownership, and
  trace persistence by constructing `Engine` directly.
- YAML fields for model, tools, protocol, parser, environment, seed, expected output,
  and dataset metadata do not affect execution. Dotted model sweep fields are declared
  but ignored.
- Concurrent runs share and mutate one `AgentModule`; failures are returned as result
  objects and then counted as completed; resume skips those failures.
- `build_run_spec()` and `build_tool_registry()` have no production caller. The latter
  still advertises a removed Web Tool path.
- `ModelConfig` and `build_model()` duplicate explicit `ModelFactory` composition and
  are used only by their own tests and documentation.

## Scope

1. Delete `qitos/experiment`, `qitos/config`, and their self-only tests.
2. Remove `qit experiment` routing and implementation.
3. Point model construction documentation at `builtin_model_factory()` and provider
   constructors; remove YAML Agent configuration claims.
4. Preserve the `qit bench` path and core benchmark/spec contracts unchanged.
5. Record the breaking removal in the changelog and bilingual README news.

## Verification

- Repository search has no remaining import or public documentation for the removed
  packages or command.
- CLI tests prove the supported top-level command set and reject `experiment`.
- Full Python 3.11 tests, stable-surface Flake8/mypy, and distribution checks pass.
- Built wheel and sdist contain neither `qitos/config` nor `qitos/experiment`.

## Progress

- [x] Remove the packages, CLI route, and self-only tests.
- [x] Point model construction docs at explicit providers and `ModelFactory`.
- [x] Record the public removal and retained benchmark path.
- [x] Pass full tests, static checks, and distribution validation.
