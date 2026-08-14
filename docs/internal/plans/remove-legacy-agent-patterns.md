# Remove legacy agent pattern templates

## Objective

Remove the fixed `qitos.kit.patterns` orchestration layer so QitOS exposes one
general Child-Agent lifecycle instead of maintaining prebuilt Manager/Worker,
Planner/Executor, Proposer/Verifier, Debate, MoA, and DAG workflow runtimes.

## Modification layer

- Remove only the fixed template package, its exclusive tests, flat imports, and
  documentation that tells applications to use those constructors.
- Keep the canonical typed Child contracts, `AgentTool`, `ChildControlToolSet`, and
  their lifecycle, persistence, cancellation, and budget behavior unchanged.
- Keep `AgentRegistry`, `DelegateTool`, `FanOutTool`, and handoff behavior in this
  change because examples and independent zoo applications still consume them.
  Their migration requires a separate caller-by-caller change.

## Success conditions

1. No tracked source, test, recipe, or user documentation imports
   `qitos.kit.patterns` or recommends its constructors.
2. Removing the package does not remove or weaken the general nested Child-Agent
   lifecycle.
3. English and Chinese release surfaces describe the breaking removal and direct
   applications to the general Child primitives or application-owned recipes.
4. QitOS passes its supported Python test matrix, static checks, and packaging checks.

## Assumptions and decisions

- Fixed collaboration policies belong in applications or explicit recipes, not in
  the generic kernel or its default kit exports.
- A language or implementation difference does not justify keeping a second runtime;
  the stable boundary is typed Child launch/control/result behavior.
- Historical design plans may record earlier intent, but active roadmap text must not
  ask maintainers to restore the removed parallel architecture.

## Plan

- [x] Remove the template package, exclusive tests, and flat imports.
- [x] Remove or rewrite every live documentation and recipe reference.
- [x] Update the changelog and English/Chinese README news.
- [x] Run reference scans, the Python 3.11 and 3.12 suites, static checks, and package
      validation.
- [x] Review the final diff and mark this plan complete.
