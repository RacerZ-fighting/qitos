# Skill and resource alignment

## Goal

Keep `SkillToolSet` as the small progressive-disclosure surface while making its
application-owned Skill catalog deterministic, diagnosable, refreshable, and safe to
use against one explicit runtime-requirement snapshot.

Pi is the primary behavior reference for recursive `SKILL.md` discovery, root-first
precedence, collision diagnostics, and explicit reload. Codex is the secondary
reference for dependency metadata, owner-held catalog snapshots, bounded warnings,
and separating discovery identity from runtime availability. The references remain
read-only and are not dependencies.

## Ownership and scope

- QitOS owns generic Skill discovery, immutable bundle identity, resource reads,
  requirement admission, and tool projection.
- Applications own the ordered roots and the set of requirements verified in their
  current runtime.
- Earlier configured roots win name collisions. Invalid or shadowed entries produce
  typed diagnostics instead of invalidating unrelated Skills.
- A directory containing `SKILL.md` is one Skill root; discovery does not reinterpret
  nested content as another Skill.
- Refresh is explicit and atomically replaces the previous catalog. No watcher,
  package manager, plugin namespace, or installation behavior is added.
- `SKILL.md` plus every discovered resource contributes to one bundle revision.
  Resource reads must match that snapshot or fail stale; they never silently mix old
  instructions with new files.
- Requirement identifiers stay opaque to QitOS. An application may use conventions
  such as `command:nmap` or `tool:run_command`, but QitOS only compares exact values.

## Success conditions

- [ ] Nested discovery stops at the nearest Skill root and is deterministic.
- [ ] Missing roots, invalid manifests, unreadable assets, and name collisions return
  typed, bounded diagnostics while valid siblings remain available.
- [ ] Root order defines stable first-wins precedence and duplicate physical roots are
  scanned once.
- [ ] Explicit refresh publishes one new snapshot without retaining removed or invalid
  entries.
- [ ] Bundle revisions include resource identity and resource reads reject unrefreshed
  changes or path escapes.
- [ ] A checked runtime-requirement snapshot marks catalog availability and prevents
  full loading when requirements are missing; an omitted snapshot remains explicitly
  unchecked for compatibility.
- [ ] Existing provider-backed search/install tools remain behaviorally separate.
- [ ] Focused tests, full QitOS checks, consuming PentestAgent checks, configured live
  Provider contracts, and relevant isolated runtime acceptance pass before merge.
