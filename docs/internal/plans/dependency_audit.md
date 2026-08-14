# Dependency Audit

## Current Packaging

`setup.py` currently uses maintained protocol and validation libraries directly:

- `httpx[socks]`
- `mcp`
- `pydantic`
- `ptyprocess` on non-Windows platforms
- `beautifulsoup4`
- `jsonschema`
- `rich`
- `pyyaml`

Extras:

- `models`: `openai`, `litellm`
- `benchmarks`: `datasets`, `huggingface_hub`
- `dev`: build/test/lint/type/audit tools

## Classification

| Dependency | Classification | Notes |
| --- | --- | --- |
| `pyyaml` | core runtime dependency | Used by config/skill manifests; currently small enough to keep. |
| `rich` | core/runtime UX dependency | Used by render and REPL helpers. Keep unless render becomes optional. |
| `httpx[socks]` | core runtime dependency | Canonical sync/async HTTP client for MCP, managed Web, controller, and benchmark traffic. |
| `mcp` | core runtime dependency | Official MCP SDK and protocol model owner. |
| `pydantic` | core runtime dependency | Strict external configuration boundary validation. |
| `ptyprocess` | platform runtime dependency | Official PTY process primitive on supported POSIX platforms. |
| `beautifulsoup4` | optional browser/tool dependency | Used for web extraction. Candidate for future optional split. |
| `openai` | optional model/provider dependency | Already in `[models]`. |
| `litellm` | optional model/provider dependency | Already in `[models]`. |
| `datasets` | optional benchmark dependency | Already in `[benchmarks]`. |
| `huggingface_hub` | optional benchmark dependency | Already in `[benchmarks]`. |
| `pytest`, `black`, `flake8`, `mypy`, `build`, `twine`, `pre-commit`, `pip-audit` | docs/dev dependency | Correctly isolated in `[dev]`. |
| desktop/browser GUI dependencies | optional desktop/browser dependency | Current code avoids hard dependency; future GUI packages should go into `[desktop]`. |
| security research dependencies such as `scapy` | should move to qitos-zoo or optional security extra | Do not add to core runtime. Current imports are lazy inside explicit experimental modules. |
| product app dependencies | should move to qitos-zoo | qitos-coder and qitos-cyber-agent should own app-specific deps. |

## Packaging Changes

- `qitos.examples*` is excluded from installable packages.
- The completed temporary zoo migration staging is no longer part of this repository.

## Current convergence

The convergence recorded in
[`httpx-skill-convergence.md`](httpx-skill-convergence.md) removed the duplicate HTTP
dependency and old network-backed Skill/Search paths. A later packaging change may
still consider:

- `qitos[models]`: provider SDKs.
- `qitos[benchmarks]`: benchmark runners and dataset SDKs.
- `qitos[desktop]`: desktop/browser controller dependencies.
- `qitos[web]`: web extraction/search dependencies if core install needs to become smaller.

Cybersecurity product dependencies should live in `qitos-zoo`, not QitOS core.
