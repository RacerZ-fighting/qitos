"""Shared workspace path helpers for tool implementations."""

from __future__ import annotations

from pathlib import Path


def resolve_workspace_path(root_dir: str, path: str) -> Path:
    """Resolve one workspace-relative path and reject parent traversal.

    Supports symlinks inside the workspace whose targets resolve outside:
    if the *unresolved* path is inside the workspace, the access is allowed
    (the symlink is considered intentional, e.g. Level 1 task isolation).
    """

    root = Path(root_dir).expanduser().resolve()
    requested = Path(path or ".").expanduser()
    if requested.is_absolute():
        try:
            relative = requested.relative_to(root)
        except ValueError as exc:
            raise PermissionError(
                f"Access denied: '{path}' is outside workspace '{root}'"
            ) from exc
    else:
        relative = requested
    if ".." in relative.parts:
        raise PermissionError(
            f"Access denied: '{path}' is outside workspace '{root}'"
        )

    raw_target = root / relative
    target = raw_target.resolve()

    if target == root or root in target.parents:
        return target

    # A workspace-owned symlink may intentionally point outside the root.
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return target

    raise PermissionError(f"Access denied: '{path}' is outside workspace '{root}'")
