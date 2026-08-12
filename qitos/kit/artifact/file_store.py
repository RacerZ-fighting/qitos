"""File-backed Artifact storage."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path

from ...core.artifact import ArtifactRef, ArtifactStoreError

_SAFE_STEM = re.compile(r"[^A-Za-z0-9._-]+")


class FileArtifactStore:
    def __init__(
        self,
        directory: str | Path,
        *,
        reference_root: str | Path | None = None,
    ) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.reference_root = (
            Path(reference_root).expanduser().resolve()
            if reference_root is not None
            else None
        )
        if self.reference_root is not None:
            try:
                self.directory.relative_to(self.reference_root)
            except ValueError as exc:
                raise ValueError("artifact directory must be inside reference_root") from exc

    def write_text(
        self,
        *,
        artifact_id: str,
        content: str,
        media_type: str,
    ) -> ArtifactRef:
        if not artifact_id:
            raise ValueError("artifact_id must not be empty")
        if not isinstance(content, str):
            raise TypeError("artifact content must be text")
        encoded = content.encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        stem = _SAFE_STEM.sub("-", artifact_id).strip("-._")[:120] or "artifact"
        suffix = ".json" if media_type == "application/json" else ".md"
        target = self.directory / f"{stem}-{digest[:16]}{suffix}"
        temp_path: Path | None = None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            descriptor, raw_temp_path = tempfile.mkstemp(
                dir=self.directory,
                prefix=".artifact-",
            )
            temp_path = Path(raw_temp_path)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, target)
        except OSError as exc:
            raise ArtifactStoreError(f"failed to persist artifact {artifact_id}") from exc
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
        return ArtifactRef(
            artifact_id=artifact_id,
            path=(
                str(target.relative_to(self.reference_root))
                if self.reference_root is not None
                else str(target)
            ),
            media_type=media_type,
            size_bytes=len(encoded),
            sha256=digest,
        )


__all__ = ["FileArtifactStore"]
