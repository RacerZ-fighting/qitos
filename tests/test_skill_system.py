from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from qitos import Action, Decision, ToolRegistry
from qitos.cli import main as qit_main
from qitos.kit.skill import (
    BundledSkillDiagnostic,
    BundledSkillDiagnosticCode,
    SkillHubProvider,
    SkillManager,
    SkillRegistry,
    SkilledAgent,
)
from qitos.kit.tool import BundledSkillSnapshot
from qitos.kit.tool.skill import SkillToolSet


def _write_skill_dir(
    path: Path,
    *,
    name: str,
    description: str,
    instructions: str = "Use carefully.",
    tags: list[str] | None = None,
    requires: list[str] | None = None,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    tag_line = json.dumps(tags or [name])
    requirement_line = (
        f"requires: {json.dumps(requires)}\n" if requires is not None else ""
    )
    (path / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\nversion: 1.0.0\n"
        f"tags: {tag_line}\n{requirement_line}---\n\n{instructions}\n",
        encoding="utf-8",
    )
    (path / "_meta.json").write_text(
        json.dumps({"slug": name, "version": "1.0.0"}), encoding="utf-8"
    )
    return path


def _zip_skill_dir(path: Path) -> Path:
    archive = path.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w") as zf:
        for file_path in path.iterdir():
            zf.write(file_path, arcname=file_path.name)
    return archive


class _FakeResponse:
    def __init__(self, *, json_data: Any = None, text: str = "", content: bytes = b""):
        self._json_data = json_data
        self.text = text
        self.content = content

    def json(self) -> Any:
        return self._json_data

    def raise_for_status(self) -> None:
        return None


def test_skillhub_provider_search_and_download(tmp_path: Path):
    skill_dir = _write_skill_dir(
        tmp_path / "github", name="github", description="GitHub CLI workflow."
    )
    archive = _zip_skill_dir(skill_dir)
    provider = SkillHubProvider()

    def fake_get(url: str, params: dict[str, Any] | None = None, timeout: int = 20):
        _ = timeout
        if "api/v1/search" in url:
            return _FakeResponse(
                json_data={
                    "results": [
                        {
                            "slug": "github",
                            "displayName": "Github",
                            "summary": "GitHub CLI workflow.",
                            "version": "1.0.0",
                        }
                    ]
                }
            )
        if url.endswith("skills.json"):
            return _FakeResponse(
                json_data={
                    "skills": [
                        {
                            "slug": "github",
                            "name": "Github",
                            "description": "GitHub CLI workflow.",
                            "version": "1.0.0",
                        }
                    ]
                }
            )
        if url.endswith("github.zip"):
            return _FakeResponse(content=archive.read_bytes())
        raise AssertionError(f"unexpected url: {url} params={params}")

    provider._session.get = fake_get  # type: ignore[method-assign]

    results = provider.search("github")
    assert results[0].slug == "github"

    download = provider.download("github", cache_dir=tmp_path / "cache")
    assert download.path.exists()
    assert download.is_archive is True


def test_skill_manager_installs_workspace_scoped_and_activates(tmp_path: Path):
    skill_dir = _write_skill_dir(
        tmp_path / "local-skill", name="github", description="GitHub CLI workflow."
    )
    archive = _zip_skill_dir(skill_dir)

    manager = SkillManager(workspace_root=str(tmp_path / "workspace"))
    installed = manager.install(str(archive), activate=True)

    assert installed.active is True
    assert installed.package.provider == "local"
    assert Path(installed.install_path).exists()
    assert str(tmp_path / "workspace" / ".qitos" / "skills") in installed.install_path
    assert (tmp_path / "workspace" / ".qitos" / "skills" / "registry.json").exists()


def test_prompt_selection_prefers_matching_skill(tmp_path: Path):
    workspace = tmp_path / "workspace"
    manager = SkillManager(workspace_root=str(workspace))
    github_dir = _write_skill_dir(
        tmp_path / "github",
        name="github",
        description="Interact with GitHub PRs.",
        instructions="Use gh pr checks.",
    )
    weather_dir = _write_skill_dir(
        tmp_path / "weather",
        name="weather",
        description="Get forecasts.",
        instructions="Use weather APIs.",
    )

    manager.install(str(_zip_skill_dir(github_dir)), activate=True)
    manager.install(str(_zip_skill_dir(weather_dir)), activate=False)

    registry = SkillRegistry(workspace_root=str(workspace))
    from qitos.kit.skill.injector import SkillPromptBuilder

    prompt = (
        SkillPromptBuilder(registry)
        .with_skills_for_task("Investigate failed GitHub PR checks")
        .build("BASE")
    )
    assert "github" in prompt.lower()
    assert "gh pr checks" in prompt
    assert "weather APIs" not in prompt


class _SkillState:
    def __init__(self, task: str, max_steps: int = 2):
        self.task = task
        self.max_steps = max_steps
        self.current_step = 0


class _BootstrapAgent(SkilledAgent[_SkillState, dict[str, Any], Action]):
    def __init__(
        self, workspace_root: str, skill_sources: list[str], active_skills: list[str]
    ):
        registry = ToolRegistry()
        super().__init__(
            tool_registry=registry,
            workspace_root=workspace_root,
            skill_sources=skill_sources,
            active_skills=active_skills,
        )

    def init_state(self, task: str, **kwargs: Any) -> _SkillState:
        return _SkillState(task=task, max_steps=int(kwargs.get("max_steps", 2)))

    def decide(
        self, state: _SkillState, observation: dict[str, Any]
    ) -> Decision[Action]:
        _ = observation
        return Decision.final(state.task)

    def reduce(
        self,
        state: _SkillState,
        observation: dict[str, Any],
        decision: Decision[Action],
    ) -> _SkillState:
        _ = observation
        _ = decision
        return state


def test_skilled_agent_bootstraps_code_configured_skills(tmp_path: Path):
    skill_dir = _write_skill_dir(
        tmp_path / "github", name="github", description="GitHub CLI workflow."
    )
    archive = _zip_skill_dir(skill_dir)
    agent = _BootstrapAgent(
        workspace_root=str(tmp_path / "workspace"),
        skill_sources=[str(archive)],
        active_skills=["github"],
    )
    assert agent.get_skill("github") is not None
    prompt = agent.build_prompt_with_skills(
        "BASE", task="Check GitHub PRs", auto_select=True
    )
    assert "ACTIVE SKILLS" in prompt
    assert "github" in prompt.lower()


def test_skill_toolset_and_qit_cli(tmp_path: Path, capsys):
    skill_dir = _write_skill_dir(
        tmp_path / "github", name="github", description="GitHub CLI workflow."
    )
    archive = _zip_skill_dir(skill_dir)
    manager = SkillManager(workspace_root=str(tmp_path / "workspace"))
    toolset = SkillToolSet(manager=manager, workspace_root=str(tmp_path / "workspace"))

    install_result = toolset.install_skill(skill_ref=str(archive))
    assert "Successfully activated skill" in install_result
    assert "github" in toolset.list_installed_skills()

    rc = qit_main(["skill", "--workspace", str(tmp_path / "workspace"), "list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "github" in out


def test_bundled_skill_tools_list_stably_and_load_complete_content(
    tmp_path: Path,
) -> None:
    bundled_root = tmp_path / "bundled"
    second_content = "Use the second workflow.\n\nKeep every evidence line."
    _write_skill_dir(
        bundled_root / "z-directory",
        name="second-skill",
        description="Second bundled workflow.",
        instructions=second_content,
    )
    first_path = _write_skill_dir(
        bundled_root / "a-directory",
        name="first-skill",
        description="First bundled workflow.",
        instructions="Use the first workflow.",
    )
    workspace = tmp_path / "workspace"
    toolset = SkillToolSet(
        workspace_root=str(workspace),
        bundled_roots=[bundled_root],
    )

    catalog = toolset.list_skills(limit=1)
    loaded = toolset.load_skill(name="second-skill")

    assert len(catalog["skills"]) == 1
    first = catalog["skills"][0]
    assert first["name"] == "first-skill"
    assert first["description"] == "First bundled workflow."
    assert first["source"] == str(first_path / "SKILL.md")
    assert first["availability"] == "unchecked"
    assert first["missing_requirements"] == []
    assert catalog["returned_count"] == 1
    assert catalog["total_count"] == 2
    assert catalog["truncated"] is True
    assert second_content in loaded["content"]
    assert loaded["content"].startswith("---\nname: second-skill\n")
    assert loaded["source"].endswith("z-directory/SKILL.md")
    assert loaded["content_sha256"]
    assert loaded["bundle_sha256"]
    assert isinstance(loaded["resources"], list)
    assert catalog["diagnostic_count"] == 0
    assert not workspace.exists()
    tool_names = {tool.spec.name for tool in toolset.tools()}
    assert "list_skills" in tool_names
    assert "load_skill" in tool_names
    assert "read_skill_resource" in tool_names


def test_bundled_skill_snapshot_and_relative_resource_round_trip(
    tmp_path: Path,
) -> None:
    skill_dir = _write_skill_dir(
        tmp_path / "bundled" / "skill",
        name="resource-skill",
        description="Workflow with a linked reference.",
    )
    reference = skill_dir / "references" / "guide.md"
    reference.parent.mkdir()
    reference.write_text("evidence workflow", encoding="utf-8")
    toolset = SkillToolSet(bundled_roots=[tmp_path / "bundled"])

    loaded = toolset.load_skill(name="resource-skill")
    snapshot = BundledSkillSnapshot.from_dict(loaded)
    resource = toolset.read_skill_resource(
        name="resource-skill", path="references/guide.md"
    )

    assert toolset.bundled_skill_snapshots() == (snapshot,)
    assert "references/guide.md" in snapshot.resources
    assert snapshot.bundle_sha256
    assert resource["content"] == "evidence workflow"
    assert resource["next_cursor"] is None


def test_bundled_skill_snapshot_reads_pre_bundle_revision_payload() -> None:
    restored = BundledSkillSnapshot.from_dict(
        {
            "name": "legacy-skill",
            "description": "Legacy persisted workflow.",
            "source": "legacy/SKILL.md",
            "content_sha256": "content-revision",
            "requires": [],
            "resources": [],
        }
    )

    assert restored.bundle_sha256 == restored.content_sha256


def test_bundled_skill_resource_pages_large_utf8_content(tmp_path: Path) -> None:
    skill_dir = _write_skill_dir(
        tmp_path / "bundled" / "skill",
        name="paged-skill",
        description="Workflow with a large reference.",
    )
    reference = skill_dir / "references" / "large.md"
    reference.parent.mkdir()
    expected = "证据" * 120_000
    reference.write_text(expected, encoding="utf-8")
    toolset = SkillToolSet(bundled_roots=[tmp_path / "bundled"])

    first = toolset.read_skill_resource(name="paged-skill", path="references/large.md")
    second = toolset.read_skill_resource(
        name="paged-skill",
        path="references/large.md",
        cursor=first["next_cursor"],
    )

    assert first["next_cursor"]
    assert first["content"] + second["content"] == expected
    assert second["next_cursor"] is None


@pytest.mark.parametrize(
    "path",
    ["../outside.md", "/outside.md", r"C:\\outside.md"],
)
def test_bundled_skill_resource_rejects_escaping_paths(
    tmp_path: Path, path: str
) -> None:
    skill_dir = _write_skill_dir(
        tmp_path / "bundled" / "skill",
        name="bounded-skill",
        description="Workflow with bounded resources.",
    )
    reference = skill_dir / "reference.md"
    reference.write_text("inside", encoding="utf-8")
    toolset = SkillToolSet(bundled_roots=[tmp_path / "bundled"])

    with pytest.raises(ValueError, match="relative path|not found"):
        toolset.read_skill_resource(name="bounded-skill", path=path)


def test_bundled_skill_resource_rechecks_symlink_boundary(tmp_path: Path) -> None:
    skill_dir = _write_skill_dir(
        tmp_path / "bundled" / "skill",
        name="symlink-skill",
        description="Workflow with a replaceable resource.",
    )
    reference = skill_dir / "reference.md"
    reference.write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    toolset = SkillToolSet(bundled_roots=[tmp_path / "bundled"])
    reference.unlink()
    reference.symlink_to(outside)

    with pytest.raises(ValueError, match="escapes"):
        toolset.read_skill_resource(name="symlink-skill", path="reference.md")


def test_bundled_skill_refresh_changes_content_revision(tmp_path: Path) -> None:
    skill_dir = _write_skill_dir(
        tmp_path / "bundled" / "skill",
        name="revision-skill",
        description="Workflow with a stable revision.",
    )
    toolset = SkillToolSet(bundled_roots=[tmp_path / "bundled"])
    before = toolset.bundled_skill_snapshots()[0]
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        skill_path.read_text(encoding="utf-8") + "\nUpdated instructions.\n",
        encoding="utf-8",
    )

    toolset.refresh_bundled_skills()

    after = toolset.bundled_skill_snapshots()[0]
    assert after.name == before.name
    assert after.content_sha256 != before.content_sha256
    assert after.bundle_sha256 != before.bundle_sha256


def test_bundled_skill_refresh_drops_entries_that_are_no_longer_valid(
    tmp_path: Path,
) -> None:
    skill_dir = _write_skill_dir(
        tmp_path / "bundled" / "skill",
        name="removed-skill",
        description="Workflow removed by the next snapshot.",
    )
    toolset = SkillToolSet(bundled_roots=[tmp_path / "bundled"])
    (skill_dir / "SKILL.md").write_text(
        "---\nname: removed-skill\ndescription: Invalid now.\n---\n",
        encoding="utf-8",
    )

    diagnostics = toolset.refresh_bundled_skills()

    assert toolset.bundled_skill_snapshots() == ()
    assert any(
        diagnostic.code is BundledSkillDiagnosticCode.SKILL_INVALID
        for diagnostic in diagnostics
    )


def test_bundled_skill_resource_change_requires_explicit_refresh(
    tmp_path: Path,
) -> None:
    skill_dir = _write_skill_dir(
        tmp_path / "bundled" / "skill",
        name="revision-skill",
        description="Workflow with a revision-bound resource.",
    )
    resource_path = skill_dir / "reference.md"
    resource_path.write_text("before", encoding="utf-8")
    toolset = SkillToolSet(bundled_roots=[tmp_path / "bundled"])
    before = toolset.bundled_skill_snapshots()[0]

    resource_path.write_text("after", encoding="utf-8")

    with pytest.raises(ValueError, match="changed after catalog refresh"):
        toolset.read_skill_resource(name="revision-skill", path="reference.md")

    toolset.refresh_bundled_skills()
    after = toolset.bundled_skill_snapshots()[0]
    resource = toolset.read_skill_resource(
        name="revision-skill", path="reference.md"
    )
    assert resource["content"] == "after"
    assert after.content_sha256 == before.content_sha256
    assert after.bundle_sha256 != before.bundle_sha256


def test_bundled_skill_discovery_stops_at_nearest_skill_root(
    tmp_path: Path,
) -> None:
    bundled_root = tmp_path / "bundled"
    parent = _write_skill_dir(
        bundled_root / "group" / "parent",
        name="parent-skill",
        description="Parent workflow owns nested resources.",
    )
    _write_skill_dir(
        parent / "nested",
        name="nested-skill",
        description="Nested workflow should be a resource.",
    )
    _write_skill_dir(
        bundled_root / "other" / "deep" / "child",
        name="child-skill",
        description="Recursively discovered child workflow.",
    )

    toolset = SkillToolSet(bundled_roots=[bundled_root])

    assert {snapshot.name for snapshot in toolset.bundled_skill_snapshots()} == {
        "child-skill",
        "parent-skill",
    }
    parent_snapshot = next(
        snapshot
        for snapshot in toolset.bundled_skill_snapshots()
        if snapshot.name == "parent-skill"
    )
    assert "nested/SKILL.md" in parent_snapshot.resources


def test_bundled_skill_requirements_gate_full_loading(tmp_path: Path) -> None:
    bundled_root = tmp_path / "bundled"
    _write_skill_dir(
        bundled_root / "skill",
        name="runtime-skill",
        description="Workflow with explicit runtime requirements.",
        requires=["command:scanner", "tool:run_command"],
    )
    toolset = SkillToolSet(
        bundled_roots=[bundled_root],
        available_requirements=["tool:run_command"],
    )

    catalog_entry = toolset.list_skills()["skills"][0]

    assert catalog_entry["availability"] == "unavailable"
    assert catalog_entry["missing_requirements"] == ["command:scanner"]
    with pytest.raises(ValueError, match="missing requirements"):
        toolset.load_skill(name="runtime-skill")

    available = SkillToolSet(
        bundled_roots=[bundled_root],
        available_requirements=["command:scanner", "tool:run_command"],
    )
    loaded = available.load_skill(name="runtime-skill")
    assert loaded["availability"] == "available"
    assert loaded["missing_requirements"] == []


def test_bundled_skill_missing_root_is_a_diagnostic(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    toolset = SkillToolSet(bundled_roots=[missing, missing])

    diagnostics = toolset.bundled_skill_diagnostics()
    assert len(diagnostics) == 1
    assert diagnostics[0].code is BundledSkillDiagnosticCode.ROOT_UNAVAILABLE
    assert toolset.list_skills()["total_count"] == 0


def test_bundled_skill_load_requires_an_exact_catalog_name(tmp_path: Path) -> None:
    bundled_root = tmp_path / "bundled"
    _write_skill_dir(
        bundled_root / "skill",
        name="exact-name",
        description="Exact name workflow.",
    )
    toolset = SkillToolSet(bundled_roots=[bundled_root])

    with pytest.raises(ValueError, match="was not found"):
        toolset.load_skill(name="Exact-Name")


def test_bundled_skill_catalog_bounds_descriptions_and_limit(tmp_path: Path) -> None:
    bundled_root = tmp_path / "bundled"
    _write_skill_dir(
        bundled_root / "skill",
        name="bounded-skill",
        description="x" * 600,
    )
    toolset = SkillToolSet(bundled_roots=[bundled_root])

    catalog = toolset.list_skills()

    assert len(catalog["skills"][0]["description"]) == 500
    assert catalog["skills"][0]["description"].endswith("...")
    with pytest.raises(ValueError, match="limit must be between"):
        toolset.list_skills(limit=101)


def test_bundled_skill_root_precedence_reports_collisions(tmp_path: Path) -> None:
    preferred_root = tmp_path / "preferred"
    shadowed_root = tmp_path / "shadowed"
    preferred_path = _write_skill_dir(
        preferred_root / "skill",
        name="duplicate-name",
        description="Preferred bundled workflow.",
    )
    shadowed_path = _write_skill_dir(
        shadowed_root / "skill",
        name="duplicate-name",
        description="Shadowed bundled workflow.",
    )
    toolset = SkillToolSet(bundled_roots=[preferred_root, shadowed_root])

    diagnostics = toolset.bundled_skill_diagnostics()

    assert toolset.list_skills()["total_count"] == 1
    assert toolset.load_skill(name="duplicate-name")["description"] == (
        "Preferred bundled workflow."
    )
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert isinstance(diagnostic, BundledSkillDiagnostic)
    assert diagnostic.code is BundledSkillDiagnosticCode.NAME_COLLISION
    assert diagnostic.source == str(shadowed_path / "SKILL.md")
    assert diagnostic.winner_source == str(preferred_path / "SKILL.md")
    assert BundledSkillDiagnostic.from_dict(diagnostic.to_dict()) == diagnostic


def test_bundled_skill_refresh_reports_invalid_sibling_without_hiding_valid_skill(
    tmp_path: Path,
) -> None:
    bundled_root = tmp_path / "bundled"
    invalid = bundled_root / "invalid"
    invalid.mkdir(parents=True)
    (invalid / "SKILL.md").write_text(
        "---\nname: invalid-skill\ndescription: Invalid workflow.\n---\n",
        encoding="utf-8",
    )
    _write_skill_dir(
        bundled_root / "valid",
        name="valid-skill",
        description="Valid sibling workflow.",
    )

    toolset = SkillToolSet(bundled_roots=[bundled_root])
    diagnostics = toolset.refresh_bundled_skills()

    assert {snapshot.name for snapshot in toolset.bundled_skill_snapshots()} == {
        "valid-skill"
    }
    assert len(diagnostics) == 1
    assert diagnostics[0].code is BundledSkillDiagnosticCode.SKILL_INVALID
    assert diagnostics[0].source == str(invalid / "SKILL.md")
    projected = toolset.list_skills()
    assert projected["diagnostic_count"] == 1
    assert projected["diagnostics"][0]["code"] == diagnostics[0].code.value


def test_skill_toolset_without_bundled_roots_preserves_provider_tool_surface() -> None:
    toolset = SkillToolSet()

    names = {tool.spec.name for tool in toolset.tools()}

    assert "list_skills" not in names
    assert "load_skill" not in names
    assert "read_skill_resource" not in names
