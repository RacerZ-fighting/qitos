"""Progressive disclosure tools for application-owned bundled Skills."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Mapping, Optional, Tuple

from qitos.core.function_tool_decorator import function_tool
from qitos.core.tool import FunctionTool
from qitos.kit.skill.bundled import (
    BundledSkillDiagnostic,
    BundledSkillEntry,
    BundledSkillSnapshot,
    discover_bundled_skills,
    normalize_bundled_roots,
    read_bundled_resource,
)

_DEFAULT_BUNDLED_SKILL_LIMIT = 20
_MAX_BUNDLED_SKILL_LIMIT = 100
_MAX_BUNDLED_DESCRIPTION_CHARS = 500
_MAX_BUNDLED_DIAGNOSTICS = 100
_MAX_BUNDLED_RESOURCE_RESPONSE_BYTES = 512 * 1024


class SkillToolSet:
    """Read-only progressive disclosure for application-owned Skill bundles."""

    name = ""
    version = "2.0"

    def __init__(
        self,
        bundled_roots: Optional[Iterable[str | Path]] = None,
        available_requirements: Optional[Iterable[str]] = None,
    ):
        self.bundled_roots = normalize_bundled_roots(bundled_roots or ())
        self.available_requirements = _requirement_snapshot(available_requirements)
        self._bundled_skills: dict[str, BundledSkillEntry] = {}
        self._bundled_diagnostics: Tuple[BundledSkillDiagnostic, ...] = ()
        self.refresh_bundled_skills()

    def tools(self) -> List[FunctionTool]:
        if not self.bundled_roots:
            return []
        return [self.list_skills, self.load_skill, self.read_skill_resource]

    def refresh_bundled_skills(self) -> Tuple[BundledSkillDiagnostic, ...]:
        """Atomically replace the catalog and return its typed diagnostics."""

        refreshed, diagnostics = discover_bundled_skills(self.bundled_roots)
        self._bundled_skills = refreshed
        self._bundled_diagnostics = diagnostics
        return diagnostics

    def bundled_skill_snapshots(self) -> Tuple[BundledSkillSnapshot, ...]:
        """Return the current name-sorted immutable bundle snapshot."""

        return tuple(
            entry.snapshot
            for _, entry in sorted(
                self._bundled_skills.items(), key=lambda item: item[0]
            )
        )

    def bundled_skill_diagnostics(self) -> Tuple[BundledSkillDiagnostic, ...]:
        """Return all diagnostics captured by the latest explicit refresh."""

        return self._bundled_diagnostics

    @function_tool(name="list_skills", read_only=True, concurrency_safe=True)
    def list_skills(
        self,
        limit: int = _DEFAULT_BUNDLED_SKILL_LIMIT,
    ) -> dict[str, Any]:
        """List bundled Skills available for full loading.

        Use this bounded catalog when the exact Skill name is not already visible or
        when the available summaries were truncated. Descriptions explain when a Skill
        applies; source identifies the configured read-only asset.

        :param limit: Maximum number of stable, name-sorted entries to return.
        """

        resolved_limit = int(limit)
        if not 1 <= resolved_limit <= _MAX_BUNDLED_SKILL_LIMIT:
            raise ValueError(f"limit must be between 1 and {_MAX_BUNDLED_SKILL_LIMIT}")
        ordered = sorted(self._bundled_skills.items(), key=lambda item: item[0])
        selected = ordered[:resolved_limit]
        diagnostics = self._bundled_diagnostics[:_MAX_BUNDLED_DIAGNOSTICS]
        return {
            "skills": [
                {
                    "name": name,
                    "description": _bounded_description(entry.manifest.description),
                    "source": str(entry.skill_path),
                    "requires": list(entry.snapshot.requires),
                    **_availability_payload(
                        entry.snapshot.requires,
                        self.available_requirements,
                    ),
                }
                for name, entry in selected
            ],
            "returned_count": len(selected),
            "total_count": len(ordered),
            "truncated": len(selected) < len(ordered),
            "diagnostics": [diagnostic.to_dict() for diagnostic in diagnostics],
            "diagnostic_count": len(self._bundled_diagnostics),
            "diagnostics_truncated": len(diagnostics) < len(self._bundled_diagnostics),
        }

    @function_tool(name="load_skill", read_only=True, concurrency_safe=True)
    def load_skill(self, name: str) -> dict[str, Any]:
        """Load one bundled Skill's complete validated `SKILL.md` snapshot.

        Pass an exact name from the visible Skill summaries or `list_skills`. The
        complete content may describe workflows, tools, failure semantics, and
        relative resources; loading it does not install software or grant permissions.

        :param name: Exact bundled Skill name from a visible summary or catalog.
        """

        try:
            entry = self._bundled_skills[name]
        except KeyError as exc:
            raise ValueError(f"Bundled skill {name!r} was not found") from exc
        availability = _availability_payload(
            entry.snapshot.requires,
            self.available_requirements,
        )
        missing = availability["missing_requirements"]
        if missing:
            raise ValueError(
                f"Bundled skill {name!r} is unavailable; missing requirements: "
                + ", ".join(missing)
            )
        return {
            **entry.snapshot.to_dict(),
            **availability,
            "content": entry.content,
        }

    @function_tool(name="read_skill_resource", read_only=True, concurrency_safe=True)
    def read_skill_resource(
        self, name: str, path: str, cursor: str = ""
    ) -> dict[str, Any]:
        """Read one UTF-8 resource from a bundled Skill.

        Use a relative path listed by `load_skill`. Absolute paths, parent traversal,
        directories, and resources outside the selected Skill are rejected. Large
        text resources are paged; pass `next_cursor` back unchanged to continue.

        :param name: Exact bundled Skill name returned by `list_skills`.
        :param path: Skill-relative resource path returned by `load_skill`.
        :param cursor: Opaque continuation returned by an earlier read.
        """

        try:
            entry = self._bundled_skills[name]
        except KeyError as exc:
            raise ValueError(f"Bundled skill {name!r} was not found") from exc
        candidate_text = path.strip()
        posix_path = PurePosixPath(candidate_text)
        windows_path = PureWindowsPath(candidate_text)
        if (
            not candidate_text
            or posix_path.is_absolute()
            or windows_path.is_absolute()
            or bool(windows_path.drive)
            or ".." in posix_path.parts
            or ".." in windows_path.parts
        ):
            raise ValueError("Skill resource path must be a non-empty relative path")
        requested = Path(candidate_text)
        normalized = requested.as_posix()
        if normalized not in entry.snapshot.resources:
            raise ValueError(
                f"Resource {normalized!r} was not found in bundled skill {name!r}"
            )
        try:
            data, content_sha256 = read_bundled_resource(entry, normalized)
            content = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Skill resource is not valid UTF-8 text") from exc
        start = _parse_resource_cursor(cursor, content_sha256, len(content))
        return _page_resource(
            {
                "name": entry.snapshot.name,
                "path": normalized,
                "content_sha256": content_sha256,
            },
            content,
            start,
        )


def _bounded_description(description: str) -> str:
    if len(description) <= _MAX_BUNDLED_DESCRIPTION_CHARS:
        return description
    return description[: _MAX_BUNDLED_DESCRIPTION_CHARS - 3] + "..."


def _requirement_snapshot(
    available_requirements: Optional[Iterable[str]],
) -> Optional[frozenset[str]]:
    if available_requirements is None:
        return None
    normalized: set[str] = set()
    for requirement in available_requirements:
        if not isinstance(requirement, str) or not requirement.strip():
            raise TypeError("available requirements must be non-empty strings")
        normalized.add(requirement.strip())
    return frozenset(normalized)


def _availability_payload(
    requires: Tuple[str, ...],
    available_requirements: Optional[frozenset[str]],
) -> Dict[str, Any]:
    if available_requirements is None:
        return {
            "availability": "unchecked",
            "missing_requirements": [],
        }
    missing = sorted(set(requires) - available_requirements)
    return {
        "availability": "unavailable" if missing else "available",
        "missing_requirements": missing,
    }


def _parse_resource_cursor(cursor: str, content_sha256: str, length: int) -> int:
    if not cursor:
        return 0
    digest, separator, raw_offset = cursor.partition(":")
    if separator != ":" or digest != content_sha256:
        raise ValueError("Skill resource cursor is invalid or stale")
    try:
        offset = int(raw_offset)
    except ValueError as exc:
        raise ValueError("Skill resource cursor is invalid or stale") from exc
    if offset < 0 or offset > length:
        raise ValueError("Skill resource cursor is invalid or stale")
    return offset


def _page_resource(
    metadata: Dict[str, Any], content: str, start: int
) -> Dict[str, Any]:
    def response(end: int, next_cursor: Optional[str]) -> Dict[str, Any]:
        return {
            **metadata,
            "content": content[start:end],
            "next_cursor": next_cursor,
        }

    complete = response(len(content), None)
    if _serialized_size(complete) <= _MAX_BUNDLED_RESOURCE_RESPONSE_BYTES:
        return complete

    low = start + 1
    high = len(content)
    best: Optional[Dict[str, Any]] = None
    while low <= high:
        end = (low + high) // 2
        candidate = response(
            end,
            f"{metadata['content_sha256']}:{end}",
        )
        if _serialized_size(candidate) <= _MAX_BUNDLED_RESOURCE_RESPONSE_BYTES:
            best = candidate
            low = end + 1
        else:
            high = end - 1
    if best is None:
        raise ValueError("Skill resource metadata leaves no room for content")
    return best


def _serialized_size(payload: Mapping[str, Any]) -> int:
    return len(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
