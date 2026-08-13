"""Provider-aware tools for agents to self-manage skills at runtime."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Mapping, Optional, Tuple

from qitos.core.function_tool_decorator import function_tool
from qitos.kit.skill import SkillManager, SkillManifest, SkillRegistry

_DEFAULT_BUNDLED_SKILL_LIMIT = 20
_MAX_BUNDLED_SKILL_LIMIT = 100
_MAX_BUNDLED_DESCRIPTION_CHARS = 500
_MAX_BUNDLED_RESOURCE_RESPONSE_BYTES = 512 * 1024


@dataclass(frozen=True)
class BundledSkillSnapshot:
    """Stable metadata for one validated, application-owned Skill bundle."""

    name: str
    description: str
    source: str
    content_sha256: str
    requires: Tuple[str, ...] = ()
    resources: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "content_sha256": self.content_sha256,
            "requires": list(self.requires),
            "resources": list(self.resources),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BundledSkillSnapshot":
        return cls(
            name=_required_snapshot_string(payload, "name"),
            description=_required_snapshot_string(payload, "description"),
            source=_required_snapshot_string(payload, "source"),
            content_sha256=_required_snapshot_string(payload, "content_sha256"),
            requires=_snapshot_string_tuple(payload, "requires"),
            resources=_snapshot_string_tuple(payload, "resources"),
        )


@dataclass(frozen=True)
class _BundledSkillEntry:
    manifest: SkillManifest
    skill_path: Path
    content: str
    snapshot: BundledSkillSnapshot


class SkillToolSet:
    """Toolset for bundled Skill disclosure and provider-backed management."""

    name = ""
    version = "2.0"

    def __init__(
        self,
        registry: Optional[SkillRegistry] = None,
        manager: Optional[SkillManager] = None,
        workspace_root: Optional[str] = None,
        default_provider: str = "skillhub",
        bundled_roots: Optional[Iterable[str | Path]] = None,
    ):
        self.workspace_root = workspace_root
        self.registry = registry
        self._manager = manager
        self.default_provider = default_provider
        self.bundled_roots = tuple(
            Path(root).expanduser().resolve() for root in bundled_roots or ()
        )
        self._bundled_skills: dict[str, _BundledSkillEntry] = {}
        self.refresh_bundled_skills()

    @property
    def manager(self) -> SkillManager:
        if self._manager is None:
            self._manager = SkillManager(
                workspace_root=self.workspace_root,
                registry=self.registry,
                default_provider=self.default_provider,
            )
        return self._manager

    def tools(self) -> List[Any]:
        tools = [
            self.check_skill_hub,
            self.install_skill_hub,
            self.search_skills,
            self.install_skill,
            self.activate_skill,
            self.list_installed_skills,
            self.get_skill_info,
        ]
        if self.bundled_roots:
            tools.extend([self.list_skills, self.load_skill, self.read_skill_resource])
        return tools

    def refresh_bundled_skills(self) -> None:
        """Reload configured read-only roots as one validated catalog snapshot."""

        refreshed: dict[str, _BundledSkillEntry] = {}
        for root in self.bundled_roots:
            if not root.is_dir():
                raise FileNotFoundError(
                    f"Bundled skill root is not a directory: {root}"
                )
            for skill_dir in sorted(root.iterdir(), key=lambda path: path.name):
                skill_path = skill_dir / "SKILL.md"
                if not skill_dir.is_dir() or not skill_path.is_file():
                    continue
                content = skill_path.read_text(encoding="utf-8")
                try:
                    manifest = SkillManifest.from_string(
                        content,
                        source=str(skill_dir),
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid bundled skill at {skill_path}: {exc}"
                    ) from exc
                issues = manifest.validate()
                if issues:
                    raise ValueError(
                        f"Invalid bundled skill at {skill_path}: {'; '.join(issues)}"
                    )
                if manifest.name in refreshed:
                    previous_path = refreshed[manifest.name].skill_path
                    raise ValueError(
                        f"Duplicate bundled skill name {manifest.name!r}: "
                        f"{previous_path} and {skill_path}"
                    )
                snapshot = BundledSkillSnapshot(
                    name=manifest.name,
                    description=manifest.description,
                    source=str(skill_path),
                    content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    requires=tuple(manifest.requires),
                    resources=_bundled_resource_paths(skill_dir),
                )
                refreshed[manifest.name] = _BundledSkillEntry(
                    manifest=manifest,
                    skill_path=skill_path,
                    content=content,
                    snapshot=snapshot,
                )
        self._bundled_skills = refreshed

    def bundled_skill_snapshots(self) -> Tuple[BundledSkillSnapshot, ...]:
        """Return the current name-sorted immutable bundle snapshot."""

        return tuple(
            entry.snapshot
            for _, entry in sorted(
                self._bundled_skills.items(), key=lambda item: item[0]
            )
        )

    @function_tool(name="list_skills", read_only=True, concurrency_safe=True)
    def list_skills(
        self,
        limit: int = _DEFAULT_BUNDLED_SKILL_LIMIT,
    ) -> dict[str, Any]:
        """List bundled Skills available for full loading.

        Use this bounded catalog before loading a Skill. Descriptions explain when a
        Skill applies; source identifies the configured read-only asset.

        :param limit: Maximum number of stable, name-sorted entries to return.
        """

        resolved_limit = int(limit)
        if not 1 <= resolved_limit <= _MAX_BUNDLED_SKILL_LIMIT:
            raise ValueError(f"limit must be between 1 and {_MAX_BUNDLED_SKILL_LIMIT}")
        ordered = sorted(self._bundled_skills.items(), key=lambda item: item[0])
        selected = ordered[:resolved_limit]
        return {
            "skills": [
                {
                    "name": name,
                    "description": _bounded_description(entry.manifest.description),
                    "source": str(entry.skill_path),
                }
                for name, entry in selected
            ],
            "returned_count": len(selected),
            "total_count": len(ordered),
            "truncated": len(selected) < len(ordered),
        }

    @function_tool(name="load_skill", read_only=True, concurrency_safe=True)
    def load_skill(self, name: str) -> dict[str, Any]:
        """Load one bundled Skill's complete validated `SKILL.md` snapshot.

        Call `list_skills` first, then pass one exact returned name. The complete
        content may describe workflows, tools, failure semantics, and relative
        resources; loading it does not install software or grant permissions.

        :param name: Exact bundled Skill name returned by `list_skills`.
        """

        try:
            entry = self._bundled_skills[name]
        except KeyError as exc:
            raise ValueError(f"Bundled skill {name!r} was not found") from exc
        return {**entry.snapshot.to_dict(), "content": entry.content}

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
        skill_root = entry.skill_path.parent.resolve()
        try:
            resource_path = (skill_root / requested).resolve(strict=True)
        except OSError as exc:
            raise ValueError(
                f"Resource {normalized!r} was not found in bundled skill {name!r}"
            ) from exc
        if not resource_path.is_relative_to(skill_root) or not resource_path.is_file():
            raise ValueError("Skill resource path escapes the selected Skill")
        try:
            data = resource_path.read_bytes()
            content = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Skill resource is not valid UTF-8 text") from exc
        except OSError as exc:
            raise ValueError(f"Could not read Skill resource: {normalized}") from exc
        content_sha256 = hashlib.sha256(data).hexdigest()
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

    @function_tool(name="check_skill_hub", read_only=True)
    def check_skill_hub(self, runtime_context: Optional[dict[str, Any]] = None) -> str:
        """
        Report whether the configured skill provider is available for use.

        :param runtime_context: Optional runtime context with env information.
        """
        manager = self._manager_from_runtime(runtime_context)
        return (
            f"Skill provider '{manager.default_provider}' is configured and ready. "
            "Use search_skills or install_skill to work with third-party skills."
        )

    @function_tool(name="install_skill_hub", needs_approval=True)
    def install_skill_hub(
        self, hub_url: str, runtime_context: Optional[dict[str, Any]] = None
    ) -> str:
        """
        Install a skill hub manifest from a local or remote provider URL.

        :param hub_url: Provider manifest location.
        :param runtime_context: Optional runtime context with env information.
        """
        manager = self._manager_from_runtime(runtime_context)
        installed = manager.install(f"local:{hub_url}", activate=False)
        return f"Installed hub manifest '{installed.manifest.name}' from {hub_url}."

    @function_tool(name="search_skills", read_only=True)
    def search_skills(
        self,
        query: str,
        provider: str = "",
        limit: int = 5,
        runtime_context: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Search the configured skill provider for installable skills.

        :param query: Search query text.
        :param provider: Optional provider override.
        :param limit: Maximum number of results to return.
        :param runtime_context: Optional runtime context with env information.
        """
        manager = self._manager_from_runtime(runtime_context)
        results = manager.search(query=query, provider=provider or None, limit=limit)
        if not results:
            return f"No skills found for query '{query}'."
        lines = []
        for result in results:
            version = result.version or "-"
            lines.append(f"{result.ref} (v{version}): {result.description}")
        return "\n".join(lines)

    @function_tool(name="install_skill", needs_approval=True)
    def install_skill(
        self,
        skill_ref: str = "",
        skill_name: str = "",
        activate: bool = True,
        runtime_context: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Install one skill by reference and optionally activate it immediately.

        :param skill_ref: Fully qualified skill reference.
        :param skill_name: Alternate plain skill name if no reference is provided.
        :param activate: Whether the installed skill should be activated.
        :param runtime_context: Optional runtime context with env information.
        """
        manager = self._manager_from_runtime(runtime_context)
        resolved_ref = skill_ref or skill_name
        installed = manager.install(resolved_ref, activate=activate)
        state = "activated" if installed.active else "installed"
        return f"Successfully {state} skill '{installed.key}' v{installed.package.version}. {installed.manifest.description}"

    @function_tool(name="activate_skill", needs_approval=True)
    def activate_skill(
        self,
        skill_ref: str,
        runtime_context: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Activate one already-installed skill so it can be injected into agents.

        :param skill_ref: Installed skill reference.
        :param runtime_context: Optional runtime context with env information.
        """
        manager = self._manager_from_runtime(runtime_context)
        if manager.activate(skill_ref):
            return f"Activated skill '{skill_ref}'."
        return f"Skill '{skill_ref}' is not installed."

    @function_tool(name="list_installed_skills", read_only=True)
    def list_installed_skills(
        self, runtime_context: Optional[dict[str, Any]] = None
    ) -> str:
        """
        List all skills installed in the current workspace context.

        :param runtime_context: Optional runtime context with env information.
        """
        manager = self._manager_from_runtime(runtime_context)
        installed = manager.list_installed()
        if not installed:
            return "No skills are currently installed."
        lines = ["Installed skills:"]
        for item in installed:
            active = " [active]" if item.active else ""
            lines.append(f"- {item.key}{active}: {item.manifest.description}")
        return "\n".join(lines)

    @function_tool(name="get_skill_info", read_only=True)
    def get_skill_info(
        self,
        skill_ref: str = "",
        skill_name: str = "",
        runtime_context: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Describe one installed or discoverable skill by reference.

        :param skill_ref: Fully qualified skill reference.
        :param skill_name: Alternate plain skill name if no reference is provided.
        :param runtime_context: Optional runtime context with env information.
        """
        manager = self._manager_from_runtime(runtime_context)
        resolved_ref = skill_ref or skill_name
        installed = manager.get_installed(resolved_ref)
        if installed is not None:
            lines = [
                f"Skill: {installed.key}",
                f"Version: {installed.package.version}",
                f"Description: {installed.manifest.description}",
                f"Active: {installed.active}",
                f"Install Path: {installed.install_path}",
            ]
            if installed.package.homepage:
                lines.append(f"Homepage: {installed.package.homepage}")
            return "\n".join(lines)
        described = manager.describe(resolved_ref)
        if described is None:
            return f"Skill '{resolved_ref}' was not found."
        lines = [
            f"Skill: {described.ref}",
            f"Version: {described.version or '-'}",
            f"Description: {described.description}",
        ]
        if described.homepage:
            lines.append(f"Homepage: {described.homepage}")
        return "\n".join(lines)

    def _manager_from_runtime(
        self, runtime_context: Optional[dict[str, Any]]
    ) -> SkillManager:
        runtime_context = runtime_context or {}
        env = runtime_context.get("env")
        workspace_root = self.workspace_root
        if workspace_root is None and env is not None:
            workspace_root = getattr(env, "workspace_root", None)
        if self._manager is None or (
            workspace_root
            and Path(workspace_root).resolve()
            != Path(self.manager.workspace_root or ".").resolve()
        ):
            self._manager = SkillManager(
                workspace_root=workspace_root,
                registry=self.registry,
                default_provider=self.default_provider,
            )
        return self._manager


def _bounded_description(description: str) -> str:
    if len(description) <= _MAX_BUNDLED_DESCRIPTION_CHARS:
        return description
    return description[: _MAX_BUNDLED_DESCRIPTION_CHARS - 3] + "..."


def _bundled_resource_paths(skill_root: Path) -> Tuple[str, ...]:
    resolved_root = skill_root.resolve()
    resources: List[str] = []
    for candidate in sorted(
        skill_root.rglob("*"), key=lambda path: path.relative_to(skill_root).as_posix()
    ):
        relative = candidate.relative_to(skill_root)
        if relative == Path("SKILL.md"):
            continue
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if candidate.is_file() and resolved.is_relative_to(resolved_root):
            resources.append(relative.as_posix())
    return tuple(resources)


def _required_snapshot_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"bundled Skill snapshot {key} must be a non-empty string")
    return value


def _snapshot_string_tuple(payload: Mapping[str, Any], key: str) -> Tuple[str, ...]:
    value = payload.get(key, [])
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise TypeError(f"bundled Skill snapshot {key} must be an array of strings")
    return tuple(value)


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
