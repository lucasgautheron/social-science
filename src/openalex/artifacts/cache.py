"""Atomic content-addressed local artifact cache."""

from __future__ import annotations

import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .manifest import ValidationResult, validate_artifact

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "openalex" / "artifacts"


@contextmanager
def digest_lock(cache_dir: Path, digest: str) -> Iterator[None]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / f".{digest}.lock"
    with lock_path.open("a+b") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - Windows fallback
            pass
        yield


class ArtifactCache:
    def __init__(self, root: str | Path = DEFAULT_CACHE_DIR):
        self.root = Path(root).expanduser()

    def path_for(self, digest: str) -> Path:
        return self.root / digest

    def validate(self, digest: str) -> ValidationResult | None:
        path = self.path_for(digest)
        try:
            return validate_artifact(path, expected_digest=digest)
        except (FileNotFoundError, ValueError):
            return None

    def promote(self, source: str | Path, digest: str) -> Path:
        source_path = Path(source).resolve()
        validate_artifact(source_path, expected_digest=digest)
        destination = self.path_for(digest)
        with digest_lock(self.root, digest):
            if self.validate(digest) is not None:
                return destination
            temporary = self.root / f".{digest}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
            shutil.rmtree(temporary, ignore_errors=True)
            self.root.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_path, temporary, symlinks=False)
            validate_artifact(temporary, expected_digest=digest)
            shutil.rmtree(destination, ignore_errors=True)
            os.replace(temporary, destination)
        return destination

    def materialize(self, digest: str, destination: str | Path) -> Path:
        source = self.path_for(digest)
        validate_artifact(source, expected_digest=digest)
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.parent / f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        backup = target.parent / f".{target.name}.old-{os.getpid()}-{uuid.uuid4().hex}"
        shutil.rmtree(temporary, ignore_errors=True)
        shutil.copytree(source, temporary, symlinks=False)
        validate_artifact(temporary, expected_digest=digest)
        had_target = target.exists()
        if had_target:
            os.replace(target, backup)
        try:
            os.replace(temporary, target)
        except Exception:
            if had_target and backup.exists():
                os.replace(backup, target)
            raise
        shutil.rmtree(backup, ignore_errors=True)
        return target
