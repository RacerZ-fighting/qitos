"""Strict metadata parsing for application-owned ``SKILL.md`` bundles."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class SkillManifest:
    """Parsed SKILL.md content."""

    name: str
    description: str
    requires: tuple[str, ...] = ()
    instructions: str = ""
    source_path: str | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "SkillManifest":
        path = Path(path)
        if path.is_dir():
            path = path / "SKILL.md"
        content = path.read_text(encoding="utf-8")
        return cls.from_string(content, source=str(path.parent))

    @classmethod
    def from_string(cls, content: str, source: str | None = None) -> "SkillManifest":
        pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
        match = re.match(pattern, content, re.DOTALL)
        if not match:
            raise ValueError("Invalid SKILL.md format: missing YAML frontmatter")

        frontmatter_str = match.group(1)
        instructions = match.group(2).strip()
        try:
            frontmatter = yaml.safe_load(frontmatter_str) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML frontmatter: {exc}") from exc

        if not isinstance(frontmatter, dict):
            raise ValueError("Invalid YAML frontmatter: expected an object")

        name = str(frontmatter.get("name") or "").strip()
        description = str(frontmatter.get("description") or "").strip()
        if not name or not description:
            raise ValueError(
                "SKILL.md must have 'name' and 'description' in frontmatter"
            )

        requires = _coerce_list(frontmatter.get("requires"))
        return cls(
            name=name,
            description=description,
            requires=requires,
            instructions=instructions,
            source_path=source,
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not re.match(r"^[A-Za-z0-9._-]+$", self.name):
            issues.append(f"Invalid skill name '{self.name}': must be filesystem-safe")
        if len(self.description) < 5:
            issues.append("Description too short (minimum 5 characters)")
        if not self.instructions.strip():
            issues.append("Missing instructions content")
        return tuple(issues)


def _coerce_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    text = str(value).strip()
    return (text,) if text else ()
