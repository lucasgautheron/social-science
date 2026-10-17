"""Immutable logical-path pins for reproducible artifact hydration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .manifest import ARTIFACT_SCHEMA, atomic_json, read_json

DEFAULT_LOCK_PATH = Path(".openalex-artifacts.lock.json")


@dataclass(frozen=True)
class ArtifactPin:
    path: str
    manifest_sha256: str
    stage: str | None = None
    s3_uri: str | None = None

    @classmethod
    def from_dict(cls, path: str, payload: dict[str, Any]) -> "ArtifactPin":
        return cls(
            path=normalize_logical_path(path),
            manifest_sha256=str(payload["manifest_sha256"]),
            stage=str(payload["stage"]) if payload.get("stage") else None,
            s3_uri=str(payload["s3_uri"]) if payload.get("s3_uri") else None,
        )

    def to_dict(self) -> dict[str, str]:
        payload = {"manifest_sha256": self.manifest_sha256}
        if self.stage:
            payload["stage"] = self.stage
        if self.s3_uri:
            payload["s3_uri"] = self.s3_uri
        return payload


def normalize_logical_path(value: str | Path) -> str:
    path = PurePosixPath(Path(value).as_posix())
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Artifact path must be a safe relative path: {value!r}")
    return path.as_posix()


class ArtifactLock:
    def __init__(self, path: str | Path = DEFAULT_LOCK_PATH):
        self.path = Path(path)
        self.pins: dict[str, ArtifactPin] = {}
        if self.path.is_file():
            payload = read_json(self.path)
            entries = payload.get("artifacts", {})
            if not isinstance(entries, dict):
                raise ValueError(f"Invalid artifact lockfile: {self.path}")
            self.pins = {
                normalize_logical_path(key): ArtifactPin.from_dict(key, value)
                for key, value in entries.items()
            }

    def get(self, logical_path: str | Path) -> ArtifactPin | None:
        return self.pins.get(normalize_logical_path(logical_path))

    def pin(self, artifact: ArtifactPin) -> None:
        self.pins[normalize_logical_path(artifact.path)] = artifact

    def write(self) -> None:
        atomic_json(
            self.path,
            {
                "schema": ARTIFACT_SCHEMA,
                "artifacts": {
                    key: self.pins[key].to_dict() for key in sorted(self.pins)
                },
            },
        )
