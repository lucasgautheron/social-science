"""Declarative artifact inputs shared by local and AWS command dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .manifest import ArtifactValidationError, validate_artifact


@dataclass(frozen=True)
class InputSpec:
    flag: str
    default: str | None = None


COMMAND_INPUTS: dict[str, tuple[InputSpec, ...]] = {
    "author-embeddings": (InputSpec("--embeddings-dir", "output/embeddings"),),
    "author-topics": (InputSpec("--topics-dir", "output/topics"),),
    "cluster-events": (InputSpec("--events-dir", "output/events"),),
    "new-links": (
        InputSpec("--events-dir"),
        InputSpec("--clusters-dir"),
        InputSpec("--author-embeddings-dir"),
    ),
    "topics": (InputSpec("--embeddings-dir", "output/embeddings"),),
    "visualize-cluster-topics": (
        InputSpec("--events-dir", "output/events"),
        InputSpec("--clusters-dir", "output/event_clusters"),
        InputSpec("--topics-dir", "output/topics"),
        InputSpec("--author-topics-dir", "output/author_topics"),
    ),
    "visualize-new-links": (InputSpec("--new-links-dir", "output/new_links"),),
    "build-website": (InputSpec("--events-dir", "output/events"),),
}


def _option_value(argv: Sequence[str], flag: str) -> str | None:
    for index, value in enumerate(argv):
        if value == flag and index + 1 < len(argv):
            return argv[index + 1]
        prefix = f"{flag}="
        if value.startswith(prefix):
            return value[len(prefix) :]
    return None


def dependency_paths(command: str, argv: Sequence[str]) -> list[str]:
    if command == "topics" and "--export-topic-vectors" in argv:
        return []
    paths: list[str] = []
    for spec in COMMAND_INPUTS.get(command, ()):
        value = _option_value(argv, spec.flag)
        if value is None:
            value = spec.default
        if value and value not in paths:
            paths.append(value)

    if command == "build-website":
        for flag in (
            "--clusters-dir",
            "--filtered-dir",
            "--new-link-visualizations-dir",
            "--cluster-topic-visualizations-dir",
        ):
            value = _option_value(argv, flag)
            if value and value not in paths:
                paths.append(value)
    return paths


def validate_dependency(command: str, argv: Sequence[str], path: str | Path) -> None:
    validation = validate_artifact(path)
    if command != "visualize-cluster-topics":
        return
    topics_path = _option_value(argv, "--topics-dir") or "output/topics"
    if Path(path) != Path(topics_path):
        return
    root = Path(path)
    manifest = validation.manifest
    vector_ids = manifest.get("topic_vector_ids")
    vectors = manifest.get("topic_vectors")
    has_vectors = bool(
        vector_ids
        and vectors
        and (root / str(vector_ids)).is_file()
        and (root / str(vectors)).is_file()
    )
    if not has_vectors and not (root / "bertopic_model").exists():
        raise ArtifactValidationError(
            "Topic artifact cannot provide c-TF-IDF vectors: neither exported "
            "vectors nor bertopic_model is present"
        )


def remote_openalex_invocation(argv: Sequence[str]) -> tuple[str, list[str]] | None:
    values = [value for value in argv if value != "--"]
    if len(values) < 2 or values[0] != "openalex":
        return None
    return values[1], values[2:]
