# HTTP and Skill Convergence

## Goal

Keep one maintained HTTP client dependency and one canonical Skill runtime. QitOS
should use `httpx` directly for synchronous and asynchronous HTTP work, while Agent
Skills remain read-only `SKILL.md` bundles discovered and loaded progressively by the
application.

## Change layers

1. Remove the unused provider-backed SkillHub installer, mutable Skill registry,
   `SkilledAgent` mixin, compatibility loader, `qit skill` command, and Agent-facing
   install/search tools. Keep the bundled Skill manifest, discovery snapshot, resource
   reader, and the three canonical disclosure tools.
2. Remove the orphaned legacy `SearchBackend` hierarchy and its seven backend modules.
   The maintained `WebSearchCapability`/`ManagedWebSearchTool` path remains unchanged.
3. Migrate the remaining text-web, desktop, and OSWorld synchronous requests directly
   to `httpx`, preserving redirects, timeouts, response bodies, and streaming download
   behavior. Remove `requests` from runtime dependencies.
4. Update public exports, examples, CLI/docs, changelog, README news, and behavior tests.

## Assumptions

- The `qitos-zoo` gitlink is an independent product repository and is outside this
  QitOS package change.
- PentestAgent consumes only bundled Skill disclosure and the maintained managed Web
  capability; it does not consume remote Skill installation or legacy search backends.
- Provider or plugin installation belongs outside the Agent Skill execution surface.

## Success conditions

- QitOS production code and package metadata have no direct `requests` dependency.
- `SkillToolSet.tools()` exposes exactly the bundled disclosure tools when roots are
  configured and exposes no install/search surface otherwise.
- No production caller or public export refers to the removed SkillHub or legacy
  `SearchBackend` types.
- Text-web, desktop-controller, OSWorld setup/probe, and streamed VM download behavior
  remain covered by tests using local fakes or servers.
- Full pytest, stable Flake8/mypy, Python 3.10 compatibility, wheel/sdist, and Twine
  checks pass before the branch is merged to QitOS `main`.

## Status

- [x] Inventory production callers and reference behavior.
- [x] Remove the superseded Skill and search paths.
- [x] Migrate remaining synchronous HTTP calls.
- [x] Update tests and repository-facing documentation.
- [ ] Run the complete QitOS quality and packaging gates.
