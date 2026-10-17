"""Resolve pinned artifacts from local paths, cache, or S3."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .cache import ArtifactCache
from .lockfile import ArtifactLock, ArtifactPin, normalize_logical_path
from .manifest import (
    MANIFEST_FILENAME,
    ArtifactValidationError,
    ValidationResult,
    manifest_sha256,
    validate_artifact,
)
from .s3_store import S3ArtifactStore


@dataclass(frozen=True)
class ResolvedArtifact:
    path: Path
    manifest_sha256: str
    source: str
    validation: ValidationResult


class ArtifactResolver:
    def __init__(
        self,
        *,
        cache: ArtifactCache,
        lock: ArtifactLock,
        store: S3ArtifactStore | None = None,
        base_dir: str | Path = ".",
    ):
        self.cache = cache
        self.lock = lock
        self.store = store
        self.base_dir = Path(base_dir).resolve()

    def _destination(self, logical_path: str | Path) -> tuple[str, Path]:
        logical = normalize_logical_path(logical_path)
        return logical, self.base_dir.joinpath(*Path(logical).parts)

    def _desired_digest(
        self,
        logical: str,
        destination: Path,
        expected_digest: str | None,
    ) -> tuple[str | None, ArtifactPin | None]:
        pin = self.lock.get(logical)
        if expected_digest is not None:
            return expected_digest, pin
        if pin is not None:
            return pin.manifest_sha256, pin
        manifest_path = destination / MANIFEST_FILENAME
        if manifest_path.is_file():
            return manifest_sha256(manifest_path), None
        return None, None

    def resolve(
        self,
        logical_path: str | Path,
        *,
        expected_digest: str | None = None,
        validator: Callable[[Path], None] | None = None,
    ) -> ResolvedArtifact:
        logical, destination = self._destination(logical_path)
        digest, _pin = self._desired_digest(logical, destination, expected_digest)

        try:
            local = validate_artifact(destination, expected_digest=digest)
            if validator is not None:
                validator(destination)
        except (OSError, ValueError):
            local = None
        if local is not None:
            return ResolvedArtifact(destination, local.digest, "local", local)
        if digest is None:
            raise ArtifactValidationError(
                f"No immutable artifact pin is available for {logical}. "
                "Pin a local export or a historical run before hydrating it."
            )

        cached = self.cache.validate(digest)
        if cached is not None and validator is not None:
            try:
                validator(self.cache.path_for(digest))
            except (OSError, ValueError):
                shutil.rmtree(self.cache.path_for(digest), ignore_errors=True)
                cached = None
        if cached is None:
            if self.store is None:
                raise ArtifactValidationError(
                    f"Artifact {logical} ({digest[:12]}) is not local or cached, "
                    "and no S3 artifact store is configured."
                )
            with tempfile.TemporaryDirectory(
                prefix=f"openalex-{digest[:12]}-"
            ) as temp_dir:
                downloaded = Path(temp_dir) / digest
                self.store.fetch(digest, downloaded)
                self.cache.promote(downloaded, digest)
            cached = self.cache.validate(digest)
            if cached is None:  # pragma: no cover - defensive
                raise ArtifactValidationError(
                    f"Downloaded artifact failed cache validation: {digest}"
                )
        self.cache.materialize(digest, destination)
        validation = validate_artifact(destination, expected_digest=digest)
        if validator is not None:
            validator(destination)
        return ResolvedArtifact(destination, digest, "cache", validation)

    def resolve_many(self, paths: Iterable[str | Path]) -> list[ResolvedArtifact]:
        return [self.resolve(path) for path in paths]

    def publish_and_pin(self, logical_path: str | Path) -> ArtifactPin:
        if self.store is None:
            raise ArtifactValidationError("No S3 artifact store is configured")
        logical, destination = self._destination(logical_path)
        result = validate_artifact(destination)
        published = self.store.publish(destination)
        pin = ArtifactPin(
            path=logical,
            manifest_sha256=result.digest,
            stage=str(result.manifest.get("stage") or "") or None,
            s3_uri=published["s3_uri"],
        )
        self.lock.pin(pin)
        self.lock.write()
        try:
            self.cache.promote(destination, result.digest)
        except shutil.Error:
            pass
        return pin
