"""Immutable discovery snapshots for application-owned bundled Skills."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .manifest import SkillManifest


class BundledSkillDiagnosticCode(str, Enum):
    """Stable reason codes produced while refreshing a bundled Skill catalog."""

    ROOT_UNAVAILABLE = "root_unavailable"
    SKILL_INVALID = "skill_invalid"
    NAME_COLLISION = "name_collision"


@dataclass(frozen=True, slots=True)
class BundledSkillDiagnostic:
    """One non-fatal discovery problem with an optional collision winner."""

    code: BundledSkillDiagnosticCode
    message: str
    source: str
    name: Optional[str] = None
    winner_source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "source": self.source,
            "name": self.name,
            "winner_source": self.winner_source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BundledSkillDiagnostic":
        raw_name = payload.get("name")
        raw_winner = payload.get("winner_source")
        if raw_name is not None and not isinstance(raw_name, str):
            raise TypeError("bundled Skill diagnostic name must be a string or null")
        if raw_winner is not None and not isinstance(raw_winner, str):
            raise TypeError(
                "bundled Skill diagnostic winner_source must be a string or null"
            )
        return cls(
            code=BundledSkillDiagnosticCode(
                _required_string(payload, "code", "bundled Skill diagnostic")
            ),
            message=_required_string(
                payload, "message", "bundled Skill diagnostic"
            ),
            source=_required_string(payload, "source", "bundled Skill diagnostic"),
            name=raw_name,
            winner_source=raw_winner,
        )


@dataclass(frozen=True, slots=True)
class BundledSkillSnapshot:
    """Stable identity for one validated Skill document and its resources."""

    name: str
    description: str
    source: str
    content_sha256: str
    requires: Tuple[str, ...] = ()
    resources: Tuple[str, ...] = ()
    bundle_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.bundle_sha256:
            object.__setattr__(self, "bundle_sha256", self.content_sha256)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "content_sha256": self.content_sha256,
            "bundle_sha256": self.bundle_sha256,
            "requires": list(self.requires),
            "resources": list(self.resources),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BundledSkillSnapshot":
        content_sha256 = _required_string(
            payload, "content_sha256", "bundled Skill snapshot"
        )
        raw_bundle_sha256 = payload.get("bundle_sha256", content_sha256)
        if not isinstance(raw_bundle_sha256, str) or not raw_bundle_sha256.strip():
            raise TypeError(
                "bundled Skill snapshot bundle_sha256 must be a non-empty string"
            )
        return cls(
            name=_required_string(payload, "name", "bundled Skill snapshot"),
            description=_required_string(
                payload, "description", "bundled Skill snapshot"
            ),
            source=_required_string(payload, "source", "bundled Skill snapshot"),
            content_sha256=content_sha256,
            bundle_sha256=raw_bundle_sha256,
            requires=_string_tuple(payload, "requires", "bundled Skill snapshot"),
            resources=_string_tuple(payload, "resources", "bundled Skill snapshot"),
        )


@dataclass(frozen=True, slots=True)
class BundledSkillEntry:
    """Internal catalog entry used by the progressive-disclosure tools."""

    manifest: SkillManifest
    skill_path: Path
    content: str
    snapshot: BundledSkillSnapshot
    resource_sha256: Tuple[Tuple[str, str], ...]

    def expected_resource_sha256(self, path: str) -> Optional[str]:
        return next(
            (digest for candidate, digest in self.resource_sha256 if candidate == path),
            None,
        )


def normalize_bundled_roots(roots: Iterable[str | Path]) -> Tuple[Path, ...]:
    """Resolve and de-duplicate roots while preserving application precedence."""

    normalized: List[Path] = []
    seen: set[Path] = set()
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if root in seen:
            continue
        seen.add(root)
        normalized.append(root)
    return tuple(normalized)


def discover_bundled_skills(
    roots: Iterable[Path],
) -> tuple[dict[str, BundledSkillEntry], Tuple[BundledSkillDiagnostic, ...]]:
    """Build one deterministic first-root-wins catalog without mutating old state."""

    entries: dict[str, BundledSkillEntry] = {}
    diagnostics: List[BundledSkillDiagnostic] = []
    for root in roots:
        skill_paths, root_diagnostics = _discover_skill_paths(root)
        diagnostics.extend(root_diagnostics)
        for skill_path in skill_paths:
            entry, entry_diagnostics = _load_entry(skill_path)
            diagnostics.extend(entry_diagnostics)
            if entry is None:
                continue
            existing = entries.get(entry.snapshot.name)
            if existing is not None:
                diagnostics.append(
                    BundledSkillDiagnostic(
                        code=BundledSkillDiagnosticCode.NAME_COLLISION,
                        message=(
                            f"Bundled Skill name {entry.snapshot.name!r} is shadowed "
                            "by an earlier configured root"
                        ),
                        source=str(skill_path),
                        name=entry.snapshot.name,
                        winner_source=str(existing.skill_path),
                    )
                )
                continue
            entries[entry.snapshot.name] = entry
    return entries, tuple(diagnostics)


def read_bundled_resource(entry: BundledSkillEntry, path: str) -> tuple[bytes, str]:
    """Read a catalogued resource only if it still matches the captured revision."""

    expected_sha256 = entry.expected_resource_sha256(path)
    if expected_sha256 is None:
        raise ValueError(
            f"Resource {path!r} was not found in bundled skill {entry.snapshot.name!r}"
        )
    skill_root = entry.skill_path.parent.resolve()
    try:
        resource_path = (skill_root / Path(path)).resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            f"Resource {path!r} was not found in bundled skill {entry.snapshot.name!r}"
        ) from exc
    if not resource_path.is_relative_to(skill_root) or not resource_path.is_file():
        raise ValueError("Skill resource path escapes the selected Skill")
    try:
        data = resource_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Could not read Skill resource: {path}") from exc
    content_sha256 = hashlib.sha256(data).hexdigest()
    if content_sha256 != expected_sha256:
        raise ValueError(
            "Bundled Skill resource changed after catalog refresh; "
            "refresh and load the Skill again"
        )
    return data, content_sha256


def _discover_skill_paths(
    root: Path,
) -> tuple[Tuple[Path, ...], Tuple[BundledSkillDiagnostic, ...]]:
    diagnostics: List[BundledSkillDiagnostic] = []
    if not root.is_dir():
        diagnostics.append(
            BundledSkillDiagnostic(
                code=BundledSkillDiagnosticCode.ROOT_UNAVAILABLE,
                message="Bundled Skill root is not an accessible directory",
                source=str(root),
            )
        )
        return (), tuple(diagnostics)

    discovered: List[Path] = []
    pending = [root]
    visited: set[Path] = set()
    while pending:
        candidate = pending.pop()
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            diagnostics.append(
                BundledSkillDiagnostic(
                    code=BundledSkillDiagnosticCode.SKILL_INVALID,
                    message=f"Could not resolve bundled Skill path: {exc}",
                    source=str(candidate),
                )
            )
            continue
        if not resolved.is_relative_to(root):
            diagnostics.append(
                BundledSkillDiagnostic(
                    code=BundledSkillDiagnosticCode.SKILL_INVALID,
                    message="Bundled Skill path escapes its configured root",
                    source=str(candidate),
                )
            )
            continue
        if resolved in visited:
            continue
        visited.add(resolved)
        skill_path = resolved / "SKILL.md"
        if skill_path.is_file():
            try:
                resolved_skill_path = skill_path.resolve(strict=True)
            except OSError as exc:
                diagnostics.append(
                    BundledSkillDiagnostic(
                        code=BundledSkillDiagnosticCode.SKILL_INVALID,
                        message=f"Could not resolve SKILL.md: {exc}",
                        source=str(skill_path),
                    )
                )
                continue
            if not resolved_skill_path.is_relative_to(root):
                diagnostics.append(
                    BundledSkillDiagnostic(
                        code=BundledSkillDiagnosticCode.SKILL_INVALID,
                        message="SKILL.md escapes its configured root",
                        source=str(skill_path),
                    )
                )
                continue
            discovered.append(skill_path)
            continue
        try:
            children = sorted(resolved.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            diagnostics.append(
                BundledSkillDiagnostic(
                    code=BundledSkillDiagnosticCode.SKILL_INVALID,
                    message=f"Could not scan bundled Skill directory: {exc}",
                    source=str(resolved),
                )
            )
            continue
        directories = [
            child
            for child in children
            if child.name != "node_modules"
            and not child.name.startswith(".")
            and child.is_dir()
        ]
        pending.extend(reversed(directories))
    return tuple(discovered), tuple(diagnostics)


def _load_entry(
    skill_path: Path,
) -> tuple[Optional[BundledSkillEntry], Tuple[BundledSkillDiagnostic, ...]]:
    try:
        content = skill_path.read_text(encoding="utf-8")
        manifest = SkillManifest.from_string(content, source=str(skill_path.parent))
    except (OSError, UnicodeError, ValueError) as exc:
        return None, (
            BundledSkillDiagnostic(
                code=BundledSkillDiagnosticCode.SKILL_INVALID,
                message=f"Could not load bundled Skill: {exc}",
                source=str(skill_path),
            ),
        )
    issues = manifest.validate()
    if issues:
        return None, (
            BundledSkillDiagnostic(
                code=BundledSkillDiagnosticCode.SKILL_INVALID,
                message="; ".join(issues),
                source=str(skill_path),
                name=manifest.name,
            ),
        )

    resource_sha256, resource_diagnostics = _resource_revisions(skill_path.parent)
    content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    bundle_sha256 = _bundle_revision(content_sha256, resource_sha256)
    snapshot = BundledSkillSnapshot(
        name=manifest.name,
        description=manifest.description,
        source=str(skill_path),
        content_sha256=content_sha256,
        bundle_sha256=bundle_sha256,
        requires=tuple(manifest.requires),
        resources=tuple(path for path, _ in resource_sha256),
    )
    return (
        BundledSkillEntry(
            manifest=manifest,
            skill_path=skill_path,
            content=content,
            snapshot=snapshot,
            resource_sha256=resource_sha256,
        ),
        resource_diagnostics,
    )


def _resource_revisions(
    skill_root: Path,
) -> tuple[Tuple[Tuple[str, str], ...], Tuple[BundledSkillDiagnostic, ...]]:
    resolved_root = skill_root.resolve()
    revisions: List[Tuple[str, str]] = []
    diagnostics: List[BundledSkillDiagnostic] = []
    try:
        candidates = sorted(
            skill_root.rglob("*"),
            key=lambda path: path.relative_to(skill_root).as_posix(),
        )
    except OSError as exc:
        return (), (
            BundledSkillDiagnostic(
                code=BundledSkillDiagnosticCode.SKILL_INVALID,
                message=f"Could not scan Skill resources: {exc}",
                source=str(skill_root),
            ),
        )
    for candidate in candidates:
        relative = candidate.relative_to(skill_root)
        if relative == Path("SKILL.md"):
            continue
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            diagnostics.append(
                BundledSkillDiagnostic(
                    code=BundledSkillDiagnosticCode.SKILL_INVALID,
                    message=f"Could not resolve Skill resource: {exc}",
                    source=str(candidate),
                )
            )
            continue
        if not candidate.is_file():
            continue
        if not resolved.is_relative_to(resolved_root):
            diagnostics.append(
                BundledSkillDiagnostic(
                    code=BundledSkillDiagnosticCode.SKILL_INVALID,
                    message="Skill resource escapes the selected Skill",
                    source=str(candidate),
                )
            )
            continue
        try:
            digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        except OSError as exc:
            diagnostics.append(
                BundledSkillDiagnostic(
                    code=BundledSkillDiagnosticCode.SKILL_INVALID,
                    message=f"Could not read Skill resource: {exc}",
                    source=str(candidate),
                )
            )
            continue
        revisions.append((relative.as_posix(), digest))
    return tuple(revisions), tuple(diagnostics)


def _bundle_revision(
    content_sha256: str,
    resource_sha256: Tuple[Tuple[str, str], ...],
) -> str:
    payload = {
        "content_sha256": content_sha256,
        "resources": [
            {"path": path, "sha256": digest} for path, digest in resource_sha256
        ],
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _required_string(payload: Mapping[str, Any], key: str, owner: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{owner} {key} must be a non-empty string")
    return value


def _string_tuple(
    payload: Mapping[str, Any], key: str, owner: str
) -> Tuple[str, ...]:
    value = payload.get(key, [])
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise TypeError(f"{owner} {key} must be an array of strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{owner} {key} must not contain duplicates")
    return tuple(value)


__all__ = [
    "BundledSkillDiagnostic",
    "BundledSkillDiagnosticCode",
    "BundledSkillSnapshot",
]
