from pathlib import Path

import pytest

from qitos.core.memory import MemoryRecord
from qitos.kit import MemdirMemory
from qitos.kit.tool import WorkspaceAwareMixin


def test_memdir_memory_roundtrip(tmp_path: Path) -> None:
    memory = MemdirMemory(memory_dir=str(tmp_path / ".memdir"))
    memory.append(
        MemoryRecord(
            role="feedback",
            content="Use taint-flow tracing before writing PoC.",
            step_id=3,
            metadata={"type": "feedback"},
        )
    )
    items = memory.retrieve({"type": "feedback"})
    assert items
    assert "taint-flow tracing" in str(items[-1].content)
    summary = memory.summarize(max_items=20)
    assert "feedback" in summary
    assert (tmp_path / ".memdir" / "MEMORY.md").exists()


def test_workspace_aware_mixin_path_guard_and_recent_files(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    file_path = tmp_path / "src" / "app.py"
    file_path.write_text("print('ok')\n", encoding="utf-8")
    helper = WorkspaceAwareMixin(workspace_root=str(tmp_path))
    resolved = helper.resolve_path("src/app.py")
    assert resolved.endswith("src/app.py")
    helper.note_recent_file("src/app.py")
    summary = helper.workspace_summary(max_entries=10, max_depth=2)
    assert "src/app.py" in list(summary.get("sample_files", []))
    assert "src/app.py" in list(summary.get("recent_files", []))
    with pytest.raises(PermissionError):
        helper.resolve_path("../outside.txt")


def test_workspace_aware_mixin_allows_workspace_owned_symlink(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    target = outside / "shared.txt"
    target.write_text("shared\n", encoding="utf-8")
    (tmp_path / "shared").symlink_to(outside, target_is_directory=True)
    helper = WorkspaceAwareMixin(workspace_root=str(tmp_path))

    assert helper.resolve_path("shared/shared.txt") == str(target)
    assert helper.resolve_path(str(tmp_path / "shared" / "shared.txt")) == str(target)
    with pytest.raises(PermissionError):
        helper.resolve_path(str(outside / "shared.txt"))
