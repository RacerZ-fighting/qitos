# Pi/Codex kernel alignment

## Goal

Keep QitOS on one small `AgentModule + Engine` mainline while bringing the reusable
coding-Agent capabilities that already have a real PentestAgent consumer up to the
behavior demonstrated by `pi:` and `codex:`. Pi is the primary reference for the
minimal loop, Session, CLI-first Tool, Skill, and extension boundaries. Codex is the
primary reference for Provider-native behavior, canonical rollout persistence,
derived indexes, context recovery, process ownership, and trace boundaries.

This is a behavior and ownership alignment, not a source-tree copy. Reference projects
remain read-only and are not dependencies.

## Scope rules

- QitOS owns reusable Provider, Tool, Journal, History, Runtime, Skill, MCP, Child, and
  observability primitives.
- Product policy, authorization scope, investigation state, and Root/Child composition
  stay outside QitOS.
- Existing capabilities are corrected in place. Do not add a second Engine, store,
  registry, gateway, or compatibility DTO.
- Add a boundary only when a current cross-module consumer needs it.
- One module is implemented, tested, documented, and committed before the next starts.
- A QitOS feature branch is not merged into `main` until its own complete gate, the
  consuming application gate, configured live Provider contracts, and relevant remote
  runtime acceptance all pass.

## Module matrix

| Module | Current evidence | Next action |
| --- | --- | --- |
| Turn loop and completion | One immutable async turn, durable Tool lifecycle, deadlines, safe points, typed stop reasons | Complete; preserve with conformance tests |
| Provider and context | Typed stream/request events, native Responses/Messages/Chat, guarded continuation, canonical transcript | Complete baseline; split oversized implementation without semantic drift after functional gaps |
| Journal and Session | Canonical JSONL, writer lease, derived per-Run SQLite index, lease-free typed catalog, lineage, resume/fork | Complete; keep the journal authoritative and indexes disposable |
| File and process runtime | Revision-safe files and Run-owned host background processes | Add the same managed contract for a real remote/container backend when its owner is available |
| Child and Mailbox | Typed Child supervisor and durable safe-point input | Complete baseline; retain one product factory boundary |
| Skill and resources | Deterministic recursive catalog, bounded diagnostics, explicit refresh, whole-bundle revisions, and requirement admission | Complete; preserve the small progressive-disclosure surface |
| MCP | Run-scoped async connect/discover/execute/close exists | Add safe refresh and unified exposure/permission handling only where current callers require it |
| Runtime capabilities | Environment ops exist but capability identity/preflight is incomplete | Define one immutable runtime capability snapshot and implement a container consumer |
| Permission and scope | Tool admission and frozen exposure exist | Close command/path/network scope gaps at the existing admission boundary |
| Observability | Trace and qita exist; model timing is local | Connect Session, Tool, Child, process, usage, and stop lineage without making trace recovery authority |
| Package modularity | Focused runtimes exist, but Engine/model/tool files remain oversized | Extract proven responsibility clusters after their behavior contracts are closed |

## Validation ledger

Each module plan records:

1. focused deterministic behavior tests;
2. complete QitOS pytest plus stable flake8/mypy checks;
3. complete consuming PentestAgent `make check` when the gitlink or consumer changes;
4. every locally configured live Provider protocol using the same Agent/Tool scenario;
5. the relevant Docker/Kali acceptance in an isolated remote worktree or exact
   container scope;
6. clean diffs, documentation, changelog, and README news before merge.

## Progress

- [x] Turn transaction, completion policy, model/context recovery, atomic files,
  managed host processes, durable Mailbox, and typed Child lifecycle.
- [x] Lease-free Run catalog, lineage, status, and nested fork.
- [x] Skill/resource alignment.
- [ ] MCP refresh and admission alignment.
- [ ] Runtime capability and managed container alignment.
- [ ] Permission/scope alignment.
- [ ] Cross-component observability alignment.
- [ ] Behavior-preserving modular decomposition of remaining oversized files.
