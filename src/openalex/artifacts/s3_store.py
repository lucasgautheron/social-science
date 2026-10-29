"""S3-backed immutable artifact store."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from .manifest import (
    INVENTORY_FILENAME,
    MANIFEST_FILENAME,
    ArtifactValidationError,
    atomic_json,
    build_inventory,
    manifest_sha256,
    read_json,
    validate_artifact,
    validate_inventory,
)


def _prefix(value: str) -> str:
    return value.strip("/")


class S3ArtifactStore:
    def __init__(self, s3: Any, bucket: str, prefix: str = ""):
        self.s3 = s3
        self.bucket = bucket
        self.prefix = _prefix(prefix)

    def artifact_prefix(self, digest: str) -> str:
        value = f"artifacts/{digest}"
        return f"{self.prefix}/{value}" if self.prefix else value

    def artifact_uri(self, digest: str) -> str:
        return f"s3://{self.bucket}/{self.artifact_prefix(digest)}/"

    def _key(self, digest: str, relative: str) -> str:
        return f"{self.artifact_prefix(digest)}/{relative}"

    def _exists_with_size(
        self, key: str, size: int, expected_digest: str
    ) -> bool:
        try:
            response = self.s3.head_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise
        metadata = response.get("Metadata") or {}
        return (
            int(response.get("ContentLength", -1)) == size
            and metadata.get("sha256") == expected_digest
        )

    def _upload_file(self, path: Path, key: str, digest: str) -> None:
        self.s3.upload_file(
            str(path),
            self.bucket,
            key,
            ExtraArgs={"Metadata": {"sha256": digest}},
        )

    def _remote_inventory(self, digest: str, temporary: Path) -> dict[str, Any] | None:
        path = temporary / INVENTORY_FILENAME
        try:
            self.s3.download_file(
                self.bucket, self._key(digest, INVENTORY_FILENAME), str(path)
            )
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise
        try:
            return read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def is_complete(
        self,
        digest: str,
        inventory: dict[str, Any],
        *,
        manifest_size: int,
    ) -> bool:
        inventory_size = len(
            (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )
        inventory_digest = hashlib.sha256(
            (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode("utf-8")
        ).hexdigest()
        sentinels_exist = self._exists_with_size(
            self._key(digest, INVENTORY_FILENAME),
            inventory_size,
            inventory_digest,
        ) and self._exists_with_size(
            self._key(digest, MANIFEST_FILENAME), manifest_size, digest
        )
        return sentinels_exist and all(
            self._exists_with_size(
                self._key(digest, str(record["path"])),
                int(record["size"]),
                str(record["sha256"]),
            )
            for record in inventory.get("files", [])
        )

    def remote_is_complete(self, digest: str) -> bool:
        """Return whether every checksummed object for an artifact is present."""
        try:
            manifest_response = self.s3.get_object(
                Bucket=self.bucket,
                Key=self._key(digest, MANIFEST_FILENAME),
            )
            manifest_payload = manifest_response["Body"].read()
            if hashlib.sha256(manifest_payload).hexdigest() != digest:
                return False
            manifest = json.loads(manifest_payload.decode("utf-8"))

            inventory_response = self.s3.get_object(
                Bucket=self.bucket,
                Key=self._key(digest, INVENTORY_FILENAME),
            )
            inventory_payload = inventory_response["Body"].read()
            inventory = json.loads(inventory_payload.decode("utf-8"))
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            if isinstance(exc, (KeyError, TypeError, ValueError, json.JSONDecodeError)):
                return False
            raise

        if (
            not isinstance(manifest, dict)
            or manifest.get("complete") is False
            or not isinstance(inventory, dict)
        ):
            return False
        records = inventory.get("files")
        if not isinstance(records, list):
            return False
        seen: set[str] = set()
        try:
            for record in records:
                if not isinstance(record, dict):
                    return False
                relative = PurePosixPath(str(record["path"]))
                if (
                    relative.is_absolute()
                    or not relative.parts
                    or any(part in {"", ".", ".."} for part in relative.parts)
                    or relative.as_posix() in seen
                ):
                    return False
                seen.add(relative.as_posix())
                int(record["size"])
                str(record["sha256"])
        except (KeyError, TypeError, ValueError):
            return False

        inventory_meta = manifest.get("inventory")
        if isinstance(inventory_meta, dict):
            try:
                metadata_matches = (
                    inventory_meta.get("path") == INVENTORY_FILENAME
                    and str(inventory_meta.get("sha256", ""))
                    == hashlib.sha256(inventory_payload).hexdigest()
                    and int(inventory_meta.get("file_count", -1)) == len(records)
                    and int(inventory_meta.get("total_size", -1))
                    == sum(int(record["size"]) for record in records)
                )
            except (TypeError, ValueError):
                return False
            if not metadata_matches:
                return False
        return self.is_complete(
            digest,
            inventory,
            manifest_size=len(manifest_payload),
        )

    def publish(self, root: str | Path) -> dict[str, str]:
        artifact = Path(root).resolve()
        result = validate_artifact(artifact)
        digest = result.digest
        if result.legacy:
            inventory = build_inventory(artifact)
            atomic_json(artifact / INVENTORY_FILENAME, inventory)
        else:
            inventory = read_json(
                artifact / str(result.manifest["inventory"]["path"])
            )
        if self.is_complete(
            digest,
            inventory,
            manifest_size=(artifact / MANIFEST_FILENAME).stat().st_size,
        ):
            return {
                "manifest_sha256": digest,
                "s3_uri": self.artifact_uri(digest),
            }

        records = sorted(inventory["files"], key=lambda item: str(item["path"]))
        for record in records:
            relative = str(record["path"])
            path = artifact / relative
            self._upload_file(
                path,
                self._key(digest, relative),
                str(record["sha256"]),
            )
        with tempfile.TemporaryDirectory() as temp_dir:
            inventory_path = Path(temp_dir) / INVENTORY_FILENAME
            inventory_path.write_text(
                json.dumps(inventory, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._upload_file(
                inventory_path,
                self._key(digest, INVENTORY_FILENAME),
                hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
            )
        self._upload_file(
            artifact / MANIFEST_FILENAME,
            self._key(digest, MANIFEST_FILENAME),
            digest,
        )
        return {
            "manifest_sha256": digest,
            "s3_uri": self.artifact_uri(digest),
        }

    def fetch(self, digest: str, destination: str | Path) -> Path:
        target = Path(destination)
        target.mkdir(parents=True, exist_ok=True)
        inventory = self._remote_inventory(digest, target)
        if inventory is None:
            raise ArtifactValidationError(
                f"Remote artifact {digest} has no inventory; republish or import it first"
            )
        records = inventory.get("files")
        if not isinstance(records, list):
            raise ArtifactValidationError(f"Remote artifact {digest} has an invalid inventory")
        for record in records:
            relative = Path(str(record["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ArtifactValidationError(f"Unsafe remote artifact path: {relative}")
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(
                self.bucket, self._key(digest, relative.as_posix()), str(path)
            )
        self.s3.download_file(
            self.bucket,
            self._key(digest, MANIFEST_FILENAME),
            str(target / MANIFEST_FILENAME),
        )
        validate_inventory(target, inventory)
        if manifest_sha256(target / MANIFEST_FILENAME) != digest:
            raise ArtifactValidationError(
                f"Downloaded artifact manifest does not match digest {digest}"
            )
        return target
