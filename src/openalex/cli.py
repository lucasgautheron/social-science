"""Unified command-line entry point."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence

COMMANDS = {
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
    "visualize-new-links": "openalex.visualizations.new_links",
}


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
    module = importlib.import_module(COMMANDS[command])
    result = module.main(values)
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
