"""Promote completed manifest-backed run outputs into the immutable S3 store."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .manifest import validate_artifact
from .s3_store import S3ArtifactStore


def publish_tree(root: str | Path, store: S3ArtifactStore) -> list[dict[str, str]]:
    output = Path(root)
    published: list[dict[str, str]] = []
    if not output.is_dir():
        return published
    for manifest_path in sorted(output.rglob("manifest.json")):
        artifact = manifest_path.parent
        try:
            validation = validate_artifact(artifact)
        except (OSError, ValueError):
            continue
        record = store.publish(artifact)
        published.append(
            {
                "path": artifact.relative_to(output.parent).as_posix(),
                "manifest_sha256": validation.digest,
                "s3_uri": record["s3_uri"],
            }
        )
    return published


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - installed on AWS runners
        raise SystemExit("boto3 is required") from exc
    store = S3ArtifactStore(boto3.client("s3"), args.bucket, args.prefix)
    records = publish_tree(args.root, store)
    for record in records:
        print(
            f"Promoted {record['path']} as "
            f"{record['manifest_sha256'][:12]} to {record['s3_uri']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
