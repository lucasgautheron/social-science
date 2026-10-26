"""Common manifest and payload-inventory support for pipeline artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ARTIFACT_SCHEMA = "openalex.artifact/1"
INVENTORY_FILENAME = "_artifact_inventory.json"
MANIFEST_FILENAME = "manifest.json"


class ArtifactValidationError(ValueError):
    """Raised when a local or downloaded artifact is incomplete or corrupt."""


@dataclass(frozen=True)
class ValidationResult:
    root: Path
    manifest: dict[str, Any]
    digest: str
    legacy: bool
    file_count: int
    total_size: int


def file_sha256(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sha256(path: str | Path) -> str:
    return file_sha256(path)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"Expected a JSON object in {path}")
    return payload


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def build_inventory(root: str | Path) -> dict[str, Any]:
    artifact = Path(root)
    files: list[dict[str, Any]] = []
    for path in sorted(artifact.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(artifact).as_posix()
        if relative in {MANIFEST_FILENAME, INVENTORY_FILENAME}:
            continue
        if path.name.endswith(".tmp") or ".tmp-" in path.name:
            continue
        files.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "schema": ARTIFACT_SCHEMA,
        "files": files,
        "file_count": len(files),
        "total_size": sum(int(item["size"]) for item in files),
    }


def write_inventory(root: str | Path) -> tuple[Path, dict[str, Any]]:
    artifact = Path(root)
    inventory = build_inventory(artifact)
    path = artifact / INVENTORY_FILENAME
    atomic_json(path, inventory)
    return path, inventory


def complete_manifest(
    root: str | Path,
    manifest: Mapping[str, Any],
    *,
    stage: str,
    inputs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Write a completed manifest with a checksummed payload inventory."""
    artifact = Path(root)
    artifact.mkdir(parents=True, exist_ok=True)
    inventory_path, inventory = write_inventory(artifact)
    payload = dict(manifest)
    recipe_sha256 = os.environ.get("OPENALEX_ARTIFACT_RECIPE_SHA256")
    if recipe_sha256:
        payload.setdefault("recipe_sha256", recipe_sha256)
    payload.update(
        {
            "artifact_schema": ARTIFACT_SCHEMA,
            "stage": stage,
            "complete": True,
            "inputs": [dict(item) for item in inputs],
            "inventory": {
                "path": inventory_path.name,
                "sha256": file_sha256(inventory_path),
                "file_count": inventory["file_count"],
                "total_size": inventory["total_size"],
            },
        }
    )
    atomic_json(artifact / MANIFEST_FILENAME, payload)
    return payload


def _safe_relative_path(value: object) -> Path:
    text = str(value)
    path = Path(text)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ArtifactValidationError(f"Unsafe artifact payload path: {text!r}")
    return path


def validate_inventory(root: str | Path, inventory: Mapping[str, Any]) -> tuple[int, int]:
    artifact = Path(root)
    records = inventory.get("files")
    if not isinstance(records, list):
        raise ArtifactValidationError("Artifact inventory has no files list")
    total_size = 0
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ArtifactValidationError("Artifact inventory contains an invalid record")
        relative = _safe_relative_path(record.get("path"))
        key = relative.as_posix()
        if key in seen:
            raise ArtifactValidationError(f"Duplicate inventory path: {key}")
        seen.add(key)
        path = artifact / relative
        if not path.is_file():
            raise ArtifactValidationError(f"Artifact payload is missing: {key}")
        expected_size = int(record.get("size", -1))
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ArtifactValidationError(
                f"Artifact payload size mismatch for {key}: {actual_size} != {expected_size}"
            )
        expected_digest = str(record.get("sha256", ""))
        actual_digest = file_sha256(path)
        if actual_digest != expected_digest:
            raise ArtifactValidationError(f"Artifact payload checksum mismatch: {key}")
        total_size += actual_size
    return len(records), total_size


def validate_artifact(
    root: str | Path,
    *,
    expected_digest: str | None = None,
    require_complete: bool = True,
) -> ValidationResult:
    """Validate new artifacts fully and accept intact legacy manifests."""
    artifact = Path(root)
    manifest_path = artifact / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ArtifactValidationError(f"Artifact manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    digest = manifest_sha256(manifest_path)
    if expected_digest is not None and digest != expected_digest:
        raise ArtifactValidationError(
            f"Artifact manifest digest mismatch: {digest} != {expected_digest}"
        )
    if require_complete and manifest.get("complete") is False:
        raise ArtifactValidationError(f"Artifact is explicitly incomplete: {manifest_path}")

    inventory_meta = manifest.get("inventory")
    legacy = not isinstance(inventory_meta, dict)
    if legacy and not (artifact / INVENTORY_FILENAME).is_file():
        return ValidationResult(artifact, manifest, digest, True, 0, 0)
    inventory_relative = _safe_relative_path(
        INVENTORY_FILENAME
        if legacy
        else inventory_meta.get("path", INVENTORY_FILENAME)
    )
    inventory_path = artifact / inventory_relative
    if not inventory_path.is_file():
        raise ArtifactValidationError(f"Artifact inventory not found: {inventory_path}")
    expected_inventory_digest = (
        "" if legacy else str(inventory_meta.get("sha256", ""))
    )
    if expected_inventory_digest and file_sha256(inventory_path) != expected_inventory_digest:
        raise ArtifactValidationError("Artifact inventory checksum mismatch")
    inventory = read_json(inventory_path)
    file_count, total_size = validate_inventory(artifact, inventory)
    return ValidationResult(artifact, manifest, digest, legacy, file_count, total_size)
