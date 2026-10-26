"""Declarative artifact graph for the complete static website."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .manifest import (
    ArtifactValidationError,
    ValidationResult,
    complete_manifest,
    read_json,
    validate_artifact,
)


@dataclass(frozen=True)
class StageSpec:
    name: str
    output: str
    worker: str
    artifact_version: int | None
    dependencies: tuple[tuple[str, str], ...]
    command: tuple[str, ...]
    required_payloads: tuple[str, ...] = ()

    @property
    def manifest_backed(self) -> bool:
        return self.artifact_version is not None

    def recipe_sha256(self) -> str:
        payload = json.dumps(
            {"stage": self.name, "command": self.command},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


WEBSITE_STAGES: tuple[StageSpec, ...] = (
    StageSpec(
        "events",
        "output/events",
        "cpu",
        1,
        (),
        (
            "openalex",
            "events",
            "--db-path",
            "articles.db",
            "--output-dir",
            "output/events",
        ),
        (
            "event_vocabulary.npy",
            "event_cooccurrence.npz",
            "event_document_frequency.npy",
        ),
    ),
    StageSpec(
        "embeddings",
        "output/embeddings",
        "gpu",
        1,
        (),
        (
            "openalex",
            "embeddings",
            "--db-path",
            "articles.db",
            "--output-dir",
            "output/embeddings",
            "--resume",
        ),
        ("embeddings.db",),
    ),
    StageSpec(
        "cluster-events",
        "output/event_clusters",
        "cpu",
        1,
        (("events", "events"),),
        (
            "openalex",
            "cluster-events",
            "--events-dir",
            "output/events",
            "--output-dir",
            "output/event_clusters",
        ),
        (
            "keywords.npy",
            "groups_by_level.npy",
            "hierarchy.npz",
            "keyword_groups.csv",
            "clusters.csv",
            "cluster_by_year.csv",
        ),
    ),
    StageSpec(
        "topics",
        "output/topics",
        "cpu",
        3,
        (("embeddings", "embeddings"),),
        (
            "openalex",
            "topics",
            "--db-path",
            "articles.db",
            "--embeddings-dir",
            "output/embeddings",
            "--output-dir",
            "output/topics",
        ),
        (
            "article_topic_classifications.parquet",
            "topic_list.csv",
        ),
    ),
    StageSpec(
        "author-embeddings",
        "output/author_embeddings",
        "cpu",
        1,
        (("embeddings", "embeddings"),),
        (
            "openalex",
            "author-embeddings",
            "--db-path",
            "articles.db",
            "--embeddings-dir",
            "output/embeddings",
            "--output-dir",
            "output/author_embeddings",
            "--resume",
        ),
        ("author_embeddings.db",),
    ),
    StageSpec(
        "author-topics",
        "output/author_topics",
        "cpu",
        1,
        (("topics", "topics"),),
        (
            "openalex",
            "author-topics",
            "--db-path",
            "articles.db",
            "--topics-dir",
            "output/topics",
            "--output-dir",
            "output/author_topics",
            "--resume",
        ),
        ("author_topics.db",),
    ),
    StageSpec(
        "new-links",
        "output/new_links",
        "cpu",
        10,
        (
            ("events", "events"),
            ("clusters", "cluster-events"),
            ("embeddings", "embeddings"),
            ("author_embeddings", "author-embeddings"),
        ),
        (
            "openalex",
            "new-links",
            "--db-path",
            "articles.db",
            "--events-dir",
            "output/events",
            "--clusters-dir",
            "output/event_clusters",
            "--embeddings-dir",
            "output/embeddings",
            "--author-embeddings-dir",
            "output/author_embeddings",
            "--output-dir",
            "output/new_links",
        ),
        (
            "cluster_metadata.npz",
            "cluster_distance_reservoir.npz",
            "cluster_semantic_distance_reservoir.npz",
        ),
    ),
    StageSpec(
        "visualize-new-links",
        "output/new_link_visualizations",
        "cpu",
        1,
        (("new_links", "new-links"),),
        (
            "openalex",
            "visualize-new-links",
            "--new-links-dir",
            "output/new_links",
            "--output-dir",
            "output/new_link_visualizations",
            "--bootstrap-replicates",
            "500",
            "--permutation-replicates",
            "1000",
        ),
        ("cluster_link_distance_summary.csv",),
    ),
    StageSpec(
        "visualize-cluster-topics",
        "output/cluster_topic_visualizations",
        "cpu",
        3,
        (
            ("events", "events"),
            ("clusters", "cluster-events"),
            ("topics", "topics"),
            ("author_topics", "author-topics"),
        ),
        (
            "openalex",
            "visualize-cluster-topics",
            "--events-dir",
            "output/events",
            "--clusters-dir",
            "output/event_clusters",
            "--topics-dir",
            "output/topics",
            "--author-topics-dir",
            "output/author_topics",
            "--db-path",
            "articles.db",
            "--output-dir",
            "output/cluster_topic_visualizations",
        ),
        ("cluster_topic_summary.csv",),
    ),
    StageSpec(
        "build-website",
        "output/website",
        "cpu",
        None,
        (
            ("events", "events"),
            ("clusters", "cluster-events"),
            ("new_link_visualizations", "visualize-new-links"),
            (
                "cluster_topic_visualizations",
                "visualize-cluster-topics",
            ),
        ),
        (
            "openalex",
            "build-website",
            "--events-dir",
            "output/events",
            "--clusters-dir",
            "output/event_clusters",
            "--new-link-visualizations-dir",
            "output/new_link_visualizations",
            "--cluster-topic-visualizations-dir",
            "output/cluster_topic_visualizations",
            "--db-path",
            "articles.db",
            "--output-dir",
            "output/website",
        ),
    ),
)

STAGES = {stage.name: stage for stage in WEBSITE_STAGES}


def manifest_input_digests(manifest: Mapping[str, Any]) -> dict[str, str]:
    records = manifest.get("inputs", [])
    if not isinstance(records, list):
        return {}
    return {
        str(record["role"]): str(record["manifest_sha256"])
        for record in records
        if isinstance(record, dict)
        and record.get("role")
        and record.get("manifest_sha256")
    }


def artifact_is_compatible(
    stage: StageSpec,
    validation: ValidationResult,
    selected_inputs: Mapping[str, str],
) -> bool:
    return manifest_is_compatible(stage, validation.manifest, selected_inputs)


def manifest_is_compatible(
    stage: StageSpec,
    manifest: Mapping[str, Any],
    selected_inputs: Mapping[str, str],
) -> bool:
    if manifest.get("stage") not in {None, stage.name}:
        return False
    if int(manifest.get("artifact_version", 0)) != stage.artifact_version:
        return False
    recorded = manifest_input_digests(manifest)
    if any(recorded.get(role) != digest for role, digest in selected_inputs.items()):
        return False
    recipe = manifest.get("recipe_sha256")
    return recipe is None or recipe == stage.recipe_sha256()


def validate_compatible_artifact(
    stage: StageSpec,
    path: str | Path,
    selected_inputs: Mapping[str, str],
) -> ValidationResult:
    validation = validate_artifact(path)
    if not artifact_is_compatible(stage, validation, selected_inputs):
        raise ArtifactValidationError(
            f"{stage.name} artifact is incompatible with the selected inputs"
        )
    return validation


def local_payload_is_adoptable(stage: StageSpec, root: str | Path) -> bool:
    path = Path(root)
    if not path.is_dir():
        return False
    if not stage.required_payloads:
        return False
    return all((path / relative).exists() for relative in stage.required_payloads)


def adopt_local_artifact(
    stage: StageSpec,
    root: str | Path,
    selected_inputs: Mapping[str, str],
) -> ValidationResult:
    """Add provenance around recognizable legacy output without recomputing it."""
    path = Path(root)
    if not local_payload_is_adoptable(stage, path):
        raise ArtifactValidationError(
            f"{path} does not contain the required {stage.name} payload"
        )
    existing = (
        read_json(path / "manifest.json")
        if (path / "manifest.json").is_file()
        else {}
    )
    manifest = {**_adopted_manifest(stage, path), **existing}
    manifest.update(
        {
            "artifact_version": stage.artifact_version,
            "adopted_local_payload": True,
            "recipe_sha256": stage.recipe_sha256(),
        }
    )
    complete_manifest(
        path,
        manifest,
        stage=stage.name,
        inputs=[
            {"role": role, "manifest_sha256": digest}
            for role, digest in selected_inputs.items()
        ],
    )
    return validate_artifact(path)


def _adopted_manifest(stage: StageSpec, root: Path) -> dict[str, Any]:
    if stage.name == "events":
        return {
            "vocabulary": "event_vocabulary.npy",
            "cooccurrence": "event_cooccurrence.npz",
            "document_frequency": "event_document_frequency.npy",
            "incidence_dir": "incidence",
            "incidence_parts": sorted(
                path.name for path in (root / "incidence").glob("part_*.npz")
            ),
            "incidence_article_ids": True,
            "papers_by_year": {},
        }
    if stage.name == "embeddings":
        database = root / "embeddings.db"
        with sqlite3.connect(database) as connection:
            row = connection.execute(
                "SELECT embedding FROM embeddings ORDER BY article_id LIMIT 1"
            ).fetchone()
            count = int(
                connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            )
        dimension = (
            int(np.asarray(pickle.loads(row[0]), dtype=np.float32).size)
            if row is not None
            else 0
        )
        return {
            "database": database.name,
            "dimension": dimension,
            "rows": count,
            "complete": True,
            "adopted_without_corpus_provenance": True,
        }
    if stage.name in {"author-embeddings", "author-topics"}:
        database_name = (
            "author_embeddings.db"
            if stage.name == "author-embeddings"
            else "author_topics.db"
        )
        database = root / database_name
        with sqlite3.connect(database) as connection:
            metadata = {
                str(key): str(value)
                for key, value in connection.execute("SELECT key, value FROM metadata")
            }
        payload: dict[str, Any] = {
            "database": database_name,
            "complete": True,
            "adopted_without_corpus_provenance": True,
        }
        for key, value in metadata.items():
            payload[key] = int(value) if value.isdigit() else value
        return payload
    if stage.name == "topics":
        payload = {
            "article_assignments": "article_topic_classifications.parquet",
            "complete": True,
            "adopted_without_corpus_provenance": True,
        }
        if (root / "topic_vector_ids.npy").is_file():
            payload["topic_vector_ids"] = "topic_vector_ids.npy"
        if (root / "topic_vectors.npz").is_file():
            payload["topic_vectors"] = "topic_vectors.npz"
        return payload
    if stage.name == "cluster-events":
        groups = np.load(root / "groups_by_level.npy", mmap_mode="r")
        return {
            "keywords": "keywords.npy",
            "groups_by_level": "groups_by_level.npy",
            "hierarchy": "hierarchy.npz",
            "keyword_groups": "keyword_groups.csv",
            "clusters": "clusters.csv",
            "cluster_by_year": "cluster_by_year.csv",
            "coarse_dir": "coarse",
            "coarse_level": 0,
            "levels": int(groups.shape[0]),
            "keyword_count": int(groups.shape[1]),
        }
    if stage.name == "new-links":
        years = sorted(
            int(path.stem)
            for path in (root / "cluster_years").glob("*.npz")
            if path.stem.isdigit()
        )
        return {
            "cluster_metadata": "cluster_metadata.npz",
            "cluster_distance_reservoir": "cluster_distance_reservoir.npz",
            "cluster_semantic_distance_reservoir": (
                "cluster_semantic_distance_reservoir.npz"
            ),
            "cluster_years_dir": "cluster_years",
            "years": years,
            "complete": True,
        }
    if stage.name == "visualize-new-links":
        return {
            "summary": "cluster_link_distance_summary.csv",
            "plots": sorted(path.name for path in root.glob("*.png")),
        }
    if stage.name == "visualize-cluster-topics":
        with (root / "cluster_topic_summary.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            cluster_count = sum(1 for _ in csv.DictReader(handle))
        return {
            "summary": "cluster_topic_summary.csv",
            "cluster_count": cluster_count,
        }
    raise ArtifactValidationError(f"{stage.name} cannot adopt a local payload")


def stage_dependencies(stage: StageSpec) -> tuple[StageSpec, ...]:
    return tuple(STAGES[name] for _role, name in stage.dependencies)


def forced_stage_names(values: Sequence[str]) -> set[str]:
    names = set(values)
    unknown = names - set(STAGES)
    if unknown:
        raise ValueError(f"Unknown pipeline stage(s): {', '.join(sorted(unknown))}")
    return names
