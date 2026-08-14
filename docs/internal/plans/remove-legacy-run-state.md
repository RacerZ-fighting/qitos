# Remove the legacy RunState snapshot format

## Goal

Keep the Session Journal and checkpoint store as the only Engine recovery owners.
Remove `qitos.engine.run_state`, an unused whole-result JSON snapshot that no Engine
resume path can consume, together with superseded zoo staging and a source-only
template CLI that packaged distributions could not execute. Preserve optional
evaluation export as explicitly read-only `EngineResult.to_dict()` payloads.

## Evidence and contract change

- QitOS production code, Engine, PentestAgent, examples, and packaged zoo code do not
  import `RunState`; only its own tests and Snowl scaffold/staging adapters use it.
- `RunState.from_engine_result()` guesses fields through `Any`, flattens State, records,
  events, budget, and trace metadata, and labels the result resumable even though no
  Engine API restores it.
- The format duplicates Journal transcript/state and checkpoint snapshots without
  stable Tool transactions, continuation, lineage, writer ownership, or commit
  boundaries.
- The temporary zoo migration staging has already been superseded by the independent
  `qitos-zoo` repository. Its Snowl adapters are the remaining non-test consumers of
  `RunState`; the QitOS workflow targeting the old staging path cannot run meaningful
  validation.
- Generic `EngineResult`, trace, benchmark recipe, configuration, permission, critic,
  and handoff export types remain available. No current QitOS runtime integration
  requires a Snowl-specific recovery format or generated compatibility shim.
- `setup.py` excluded `templates*` from distributions, so installed `qit new` and
  `qit list-templates` could never access their advertised assets. The Cookiecutter
  scaffold also encoded an obsolete Agent contract. Importable `qitos.recipes` remain
  the maintained method implementations.

This is an intentional breaking removal. Existing `RunState` JSON is not accepted as
an Engine recovery source. Resume uses a Session Journal or a configured checkpoint
store.

## Implementation

- [x] Delete `qitos.engine.run_state` and its implementation-only tests.
- [x] Remove the superseded zoo migration staging, its no-op workflow, and stale links;
  keep the independent `qitos-zoo` repository as the product owner.
- [x] Remove Snowl-specific scaffold files and documentation that advertise an
  integration QitOS does not implement; keep generic export contracts intact.
- [x] Remove the unavailable template CLI, template tree, Cookiecutter extra, and stale
  contribution guidance while preserving maintained method recipes.
- [x] Update the changelog, README news, and architecture ledgers.
- [x] Verify no removed API or packaged-wheel reference remains.

## Validation

- Focused Engine result, CLI, Journal, and checkpoint tests.
- Complete QitOS pytest suite.
- Stable-surface Flake8 and mypy checks.
- Python 3.10 import check.
- Package build, Twine metadata validation, documentation navigation, and wheel-content
  inspection.
- Consuming PentestAgent `make check` after merge and gitlink update.

## Result

- QitOS suite: 1,874 passed, 47 skipped.
- Stable Flake8 and mypy surfaces passed; Python 3.10 import passed.
- Mintlify navigation JSON parsed, `qit --help` exposed only shipped commands, and
  Twine accepted both wheel and sdist.
- Wheel and sdist contain no `RunState`, Snowl adapter, Cookiecutter dependency, or
  removed template tree.
