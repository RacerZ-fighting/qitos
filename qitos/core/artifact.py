"""Durable references for outputs kept outside model history."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    artifact_id: str
    path: str
    media_type: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("artifact_id must not be empty")
        if not self.path:
            raise ValueError("artifact path must not be empty")
        if not self.media_type:
            raise ValueError("artifact media_type must not be empty")
        if self.size_bytes < 0:
            raise ValueError("artifact size_bytes must not be negative")
        if len(self.sha256) != 64:
            raise ValueError("artifact sha256 must be a hexadecimal digest")
        try:
            int(self.sha256, 16)
        except ValueError as exc:
            raise ValueError("artifact sha256 must be a hexadecimal digest") from exc

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "path": self.path,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ArtifactRef":
        try:
            artifact_id = payload["artifact_id"]
            path = payload["path"]
            media_type = payload["media_type"]
            size_bytes = payload["size_bytes"]
            sha256 = payload["sha256"]
        except KeyError as exc:
            raise ValueError(f"artifact reference is missing {exc.args[0]}") from exc
        if not all(isinstance(value, str) for value in (artifact_id, path, media_type, sha256)):
            raise ValueError("artifact reference text fields must be strings")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool):
            raise ValueError("artifact size_bytes must be an integer")
        return cls(
            artifact_id=artifact_id,
            path=path,
            media_type=media_type,
            size_bytes=size_bytes,
            sha256=sha256,
        )


class ArtifactStoreError(RuntimeError):
    pass


class ArtifactStore(Protocol):
    def write_text(
        self,
        *,
        artifact_id: str,
        content: str,
        media_type: str,
    ) -> ArtifactRef:
        ...


__all__ = ["ArtifactRef", "ArtifactStore", "ArtifactStoreError"]
