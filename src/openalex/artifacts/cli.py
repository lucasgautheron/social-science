"""Inspect, pin, publish, and hydrate pipeline artifacts."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Sequence

from .cache import DEFAULT_CACHE_DIR, ArtifactCache
from .lockfile import DEFAULT_LOCK_PATH, ArtifactLock, ArtifactPin, normalize_logical_path
from .manifest import read_json, validate_artifact
from .resolver import ArtifactResolver
from .s3_store import S3ArtifactStore

DEFAULT_STATE_PATH = Path(".aws_runner_state.json")


def _store(args: argparse.Namespace) -> S3ArtifactStore:
    if not Path(args.state_path).is_file():
        raise SystemExit(
            f"Artifact store state not found: {args.state_path}. Run openalex-aws setup first."
        )
    state = read_json(args.state_path)
    try:
        import boto3
    except ImportError as exc:
        raise SystemExit("boto3 is required for S3 artifact operations") from exc
    session_args = {}
    if args.profile:
        session_args["profile_name"] = args.profile
    region = args.region or state.get("region")
    if region:
        session_args["region_name"] = region
    s3 = boto3.Session(**session_args).client("s3")
    return S3ArtifactStore(s3, str(state["bucket"]), str(state.get("prefix", "")))


def build_resolver(args: argparse.Namespace, *, require_store: bool) -> ArtifactResolver:
    store = _store(args) if require_store or Path(args.state_path).is_file() else None
    return ArtifactResolver(
        cache=ArtifactCache(args.cache_dir),
        lock=ArtifactLock(args.lock_path),
        store=store,
    )


def command_inspect(args: argparse.Namespace) -> int:
    records = []
    lock = ArtifactLock(args.lock_path)
    for value in args.paths:
        path = Path(value)
        try:
            validation = validate_artifact(path)
            record = {
                "path": value,
                "valid": True,
                "manifest_sha256": validation.digest,
                "legacy": validation.legacy,
                "complete": validation.manifest.get("complete"),
                "stage": validation.manifest.get("stage"),
                "file_count": validation.file_count,
                "total_size": validation.total_size,
            }
        except (OSError, ValueError) as exc:
            pin = lock.get(value)
            record = {
                "path": value,
                "valid": False,
                "error": str(exc),
                "pinned_manifest_sha256": (
                    None if pin is None else pin.manifest_sha256
                ),
            }
        records.append(record)
    if args.json:
        print(json.dumps(records, indent=2, sort_keys=True))
    else:
        for record in records:
            if record["valid"]:
                marker = "legacy" if record["legacy"] else "verified"
                print(
                    f"{record['path']}: {marker} "
                    f"{str(record['manifest_sha256'])[:12]}"
                )
            else:
                print(f"{record['path']}: invalid ({record['error']})")
    return 0 if all(record["valid"] for record in records) else 1


def command_pin(args: argparse.Namespace) -> int:
    lock = ArtifactLock(args.lock_path)
    for value in args.paths:
        logical = normalize_logical_path(value)
        manifest_path = Path(logical) / "manifest.json"
        published_uri = args.s3_uri
        if args.from_run:
            store = _store(args)
            relative = Path(logical)
            if not relative.parts or relative.parts[0] != "output":
                raise SystemExit("--from-run paths must be beneath output/")
            run_relative = Path(*relative.parts[1:]).as_posix()
            run_prefix = "/".join(
                part
                for part in (
                    store.prefix,
                    "runs",
                    args.from_run,
                    "output",
                    run_relative,
                )
                if part
            ).rstrip("/")
            with tempfile.TemporaryDirectory(prefix="openalex-run-artifact-") as temp_dir:
                imported = Path(temp_dir) / "artifact"
                imported.mkdir()
                objects = []
                paginator = store.s3.get_paginator("list_objects_v2")
                for page in paginator.paginate(
                    Bucket=store.bucket, Prefix=f"{run_prefix}/"
                ):
                    objects.extend(page.get("Contents", []))
                if not objects:
                    raise SystemExit(
                        f"No historical artifact found at "
                        f"s3://{store.bucket}/{run_prefix}/"
                    )
                for item in objects:
                    key = str(item["Key"])
                    relative_key = Path(key.removeprefix(f"{run_prefix}/"))
                    if relative_key.is_absolute() or ".." in relative_key.parts:
                        raise SystemExit(f"Unsafe historical artifact key: {key}")
                    destination = imported / relative_key
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    store.s3.download_file(store.bucket, key, str(destination))
                validation = validate_artifact(imported)
                published = store.publish(imported)
                digest = validation.digest
                published_uri = published["s3_uri"]
                cache = ArtifactCache(args.cache_dir)
                cache.promote(imported, digest)
                cache.materialize(digest, logical)
                manifest = validation.manifest
        elif args.digest:
            if len(args.paths) != 1:
                raise SystemExit("--digest can only be used with one path")
            digest = args.digest
            manifest = read_json(manifest_path) if manifest_path.is_file() else {}
        else:
            validation = validate_artifact(logical)
            digest = validation.digest
            manifest = validation.manifest
        pin = ArtifactPin(
            logical,
            digest,
            str(manifest.get("stage") or "") or None,
            published_uri,
        )
        lock.pin(pin)
        print(f"Pinned {logical} to {digest}")
    lock.write()
    return 0


def command_hydrate(args: argparse.Namespace) -> int:
    resolver = build_resolver(args, require_store=True)
    for value in args.paths:
        resolved = resolver.resolve(value)
        print(
            f"{value}: {resolved.source} "
            f"{resolved.manifest_sha256[:12]} -> {resolved.path}"
        )
    return 0


def command_publish(args: argparse.Namespace) -> int:
    resolver = build_resolver(args, require_store=True)
    for value in args.paths:
        pin = resolver.publish_and_pin(value)
        print(f"Published {value} as {pin.manifest_sha256} to {pin.s3_uri}")
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lock-path", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--profile", default=None)
    parser.add_argument("--region", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage immutable OpenAlex artifacts.")
    commands = parser.add_subparsers(dest="artifact_command", required=True)

    inspect = commands.add_parser("inspect", help="Validate local artifacts.")
    _common(inspect)
    inspect.add_argument("--json", action="store_true")
    inspect.add_argument("paths", nargs="+")
    inspect.set_defaults(func=command_inspect)

    pin = commands.add_parser("pin", help="Pin logical paths to immutable manifests.")
    _common(pin)
    pin.add_argument("--digest", default=None)
    pin.add_argument("--s3-uri", default=None)
    pin.add_argument(
        "--from-run",
        default=None,
        metavar="RUN_ID",
        help="Import this output path from a historical AWS run without recomputing it.",
    )
    pin.add_argument("paths", nargs="+")
    pin.set_defaults(func=command_pin)

    hydrate = commands.add_parser("hydrate", help="Fetch pinned artifacts from cache or S3.")
    _common(hydrate)
    hydrate.add_argument("paths", nargs="+")
    hydrate.set_defaults(func=command_hydrate)

    publish = commands.add_parser("publish", help="Publish and pin local artifacts.")
    _common(publish)
    publish.add_argument("paths", nargs="+")
    publish.set_defaults(func=command_publish)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)
