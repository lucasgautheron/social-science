"""Unified command-line entry point."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

COMMANDS = {
    "artifacts": "openalex.artifacts.cli",
    "author-embeddings": "openalex.analysis.author_embeddings",
    "author-topics": "openalex.analysis.author_topics",
    "build-reference-index": "openalex.db.build_reference_index",
    "build-website": "openalex.website.build",
    "cluster-events": "openalex.analysis.cluster_events",
    "cluster-trends": "openalex.analysis.cluster_trends",
    "compile": "openalex.db.compile",
    "compile-snapshot": "openalex.db.compile_from_snapshot",
    "download": "openalex.imports.download",
    "embeddings": "openalex.analysis.embeddings",
    "events": "openalex.events",
    "filter-events": "openalex.analysis.filter_events",
    "export-parquet": "openalex.db.export_parquet",
    "network": "openalex.analysis.network",
    "new-links": "openalex.analysis.new_links",
    "random-order": "openalex.db.random_order",
    "topics": "openalex.analysis.topics",
    "vacuum": "openalex.db.vacuum",
    "visualize-cluster-topics": "openalex.visualizations.cluster_topics",
    "visualize-new-links": "openalex.visualizations.new_links",
}


def _auto_hydrate(command: str, argv: Sequence[str]) -> None:
    if os.environ.get("OPENALEX_AUTO_HYDRATE", "1").lower() in {"0", "false", "no"}:
        return
    from openalex.artifacts.cache import DEFAULT_CACHE_DIR, ArtifactCache
    from openalex.artifacts.lockfile import DEFAULT_LOCK_PATH, ArtifactLock
    from openalex.artifacts.resolver import ArtifactResolver
    from openalex.artifacts.s3_store import S3ArtifactStore
    from openalex.artifacts.specs import dependency_paths, validate_dependency

    dependencies = dependency_paths(command, argv)
    missing = []
    for value in dependencies:
        try:
            validate_dependency(command, argv, value)
        except (OSError, ValueError):
            missing.append(value)
    if not missing:
        return
    state_path = Path(
        os.environ.get("OPENALEX_ARTIFACT_STATE", ".aws_runner_state.json")
    )
    if not state_path.is_file():
        return
    with state_path.open(encoding="utf-8") as handle:
        state = json.load(handle)
    try:
        import boto3
    except ImportError as exc:
        raise SystemExit(
            "Automatic artifact hydration requires boto3; install the aws extra"
        ) from exc
    session_kwargs = {}
    if state.get("region"):
        session_kwargs["region_name"] = state["region"]
    store = S3ArtifactStore(
        boto3.Session(**session_kwargs).client("s3"),
        str(state["bucket"]),
        str(state.get("prefix", "")),
    )
    resolver = ArtifactResolver(
        cache=ArtifactCache(
            os.environ.get("OPENALEX_ARTIFACT_CACHE", str(DEFAULT_CACHE_DIR))
        ),
        lock=ArtifactLock(
            os.environ.get("OPENALEX_ARTIFACT_LOCK", str(DEFAULT_LOCK_PATH))
        ),
        store=store,
    )
    for value in missing:
        resolved = resolver.resolve(
            value,
            validator=lambda path: validate_dependency(command, argv, path),
        )
        print(
            f"Hydrated {value} from {resolved.source} "
            f"({resolved.manifest_sha256[:12]}).",
            file=sys.stderr,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openalex",
        description="Build and analyze a local OpenAlex corpus.",
    )
    parser.add_argument("command", nargs="?", choices=sorted(COMMANDS), help="Command to run.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0] in {"-h", "--help"}:
        build_parser().print_help()
        return 0
    command = values.pop(0)
    if command not in COMMANDS:
        build_parser().error(f"invalid command: {command}")
    if command != "artifacts":
        _auto_hydrate(command, values)
    module = importlib.import_module(COMMANDS[command])
    result = module.main(values)
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
