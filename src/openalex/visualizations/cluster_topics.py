"""Summarize BERTopic topics and Rao--Stirling diversity by event cluster."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from openalex.analysis.event_papers import load_event_paper_clusters

logger = logging.getLogger(__name__)

ARTIFACT_VERSION = 1
DEFAULT_OUTPUT_DIR = Path("output/cluster_topic_visualizations")

TopicModelLoader = Callable[[Path], object]


def build_cluster_topic_summary(
    events_dir: str | Path,
    clusters_dir: str | Path,
    topics_dir: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    db_path: str | Path | None = None,
    author_topics_dir: str | Path | None = None,
    topic_model_loader: TopicModelLoader | None = None,
) -> Path:
    """Write per-cluster topic histograms and c-TF-IDF Rao--Stirling diversity."""
    events = Path(events_dir).expanduser().resolve()
    clusters = Path(clusters_dir).expanduser().resolve()
    topics = Path(topics_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()

    topic_manifest_path = topics / "manifest.json"
    topic_manifest = _read_json(topic_manifest_path)
    if int(topic_manifest.get("artifact_version", 0)) < 2:
        raise ValueError("Topic artifacts must use artifact version 2 or newer")
    assignments_path = topics / str(
        topic_manifest.get("article_assignments", "article_topic_classifications.parquet")
    )
    topic_list_path = topics / "topic_list.csv"
    model_path = topics / "bertopic_model"
    if not assignments_path.is_file():
        raise ValueError(f"Topic assignments not found: {assignments_path}")
    if not topic_list_path.is_file():
        raise ValueError(f"Topic list not found: {topic_list_path}")
    if not model_path.exists():
        raise ValueError(
            "BERTopic model not found. Rao--Stirling diversity requires the saved "
            "c-TF-IDF vectors; rebuild topics without --no-save-model."
        )

    article_ids, article_topics = _load_topic_assignments(assignments_path)
    labels = _load_topic_labels(topic_list_path)
    loader = topic_model_loader or _load_bertopic_model
    topic_model = loader(model_path)
    topic_ids, topic_vectors = _normalized_topic_vectors(topic_model)

    (
        event_ids,
        event_offsets,
        event_clusters,
        _event_years,
        cluster_labels,
        cluster_level,
    ) = load_event_paper_clusters(events, clusters)
    rows = summarize_cluster_topics(
        event_ids,
        event_offsets,
        event_clusters,
        cluster_labels,
        article_ids,
        article_topics,
        topic_ids,
        topic_vectors,
        labels,
    )
    if (db_path is None) != (author_topics_dir is None):
        raise ValueError("--db-path and --author-topics-dir must be supplied together")
    author_manifest_path = None
    source_database = None
    if db_path is not None and author_topics_dir is not None:
        source_database = Path(db_path).expanduser().resolve()
        author_topics_root = Path(author_topics_dir).expanduser().resolve()
        author_manifest_path = author_topics_root / "manifest.json"
        author_manifest = _read_json(author_manifest_path)
        if not bool(author_manifest.get("complete")):
            raise ValueError(
                f"Author-topic artifact is incomplete: {author_manifest_path}"
            )
        author_database = author_topics_root / str(
            author_manifest.get("database", "author_topics.db")
        )
        if not source_database.is_file():
            raise ValueError(f"Article database not found: {source_database}")
        if not author_database.is_file():
            raise ValueError(f"Author-topic database not found: {author_database}")
        event_authors = _load_event_authors(source_database, event_ids)
        involved_authors = np.unique(
            np.concatenate(event_authors)
            if event_authors
            else np.empty(0, dtype=np.int64)
        )
        author_profiles = _load_author_topic_weights(
            author_database,
            involved_authors,
        )
        author_rows = summarize_cluster_author_topics(
            event_ids,
            event_offsets,
            event_clusters,
            cluster_labels,
            event_authors,
            article_ids,
            article_topics,
            author_profiles,
            topic_ids,
            topic_vectors,
            labels,
        )
        for row, author_row in zip(rows, author_rows, strict=True):
            row.update(author_row)

    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "cluster_topic_summary.csv"
    _write_summary(summary_path, rows)
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "cluster_count": len(rows),
        "cluster_level": cluster_level,
        "distance_definition": "1 - cosine similarity between BERTopic c-TF-IDF vectors",
        "events_manifest_sha256": _file_sha256(events / "manifest.json"),
        "clusters_manifest_sha256": _file_sha256(clusters / "manifest.json"),
        "topics_manifest_sha256": _file_sha256(topic_manifest_path),
        "summary": summary_path.name,
        "topic_assignment": "hard full-corpus BERTopic classifier label",
        "topic_model": str(model_path),
        "author_diversity": author_manifest_path is not None,
        "author_weighting": "sum(1 / paper_author_count) over cluster papers",
        "author_primary_topic": (
            "largest fractional topic weight after subtracting current-cluster papers"
        ),
    }
    if author_manifest_path is not None and source_database is not None:
        manifest.update(
            {
                "author_topics_manifest_sha256": _file_sha256(author_manifest_path),
                "source_database": str(source_database),
            }
        )
    _atomic_json(output / "manifest.json", manifest)
    logger.info("Wrote cluster-topic summary for %s clusters to %s", len(rows), output)
    return summary_path


def summarize_cluster_topics(
    event_ids: np.ndarray,
    event_offsets: np.ndarray,
    event_clusters: np.ndarray,
    cluster_labels: np.ndarray,
    article_ids: np.ndarray,
    article_topics: np.ndarray,
    topic_ids: np.ndarray,
    topic_vectors: sparse.csr_matrix,
    labels: Mapping[int, str],
) -> list[dict[str, Any]]:
    """Aggregate topic counts and diversity from aligned event-paper memberships."""
    event_ids = np.asarray(event_ids, dtype=np.int64)
    event_offsets = np.asarray(event_offsets, dtype=np.int64)
    event_clusters = np.asarray(event_clusters, dtype=np.int32)
    article_ids = np.asarray(article_ids, dtype=np.int64)
    article_topics = np.asarray(article_topics, dtype=np.int32)
    topic_ids = np.asarray(topic_ids, dtype=np.int32)
    if event_offsets.shape != (event_ids.size + 1,):
        raise ValueError("Event offsets do not align with event paper ids")
    if int(event_offsets[-1]) != event_clusters.size:
        raise ValueError("Event cluster memberships do not align with event offsets")
    if article_ids.shape != article_topics.shape:
        raise ValueError("Topic article ids and assignments do not align")
    if topic_vectors.shape[0] != topic_ids.size:
        raise ValueError("Topic vector rows do not align with topic ids")
    if article_ids.size and np.any(article_ids[1:] <= article_ids[:-1]):
        raise ValueError("Topic assignments must have unique, increasing article ids")
    if topic_ids.size and np.any(topic_ids[1:] <= topic_ids[:-1]):
        raise ValueError("Topic vector ids must be unique and increasing")

    paper_topics = _align_article_topics(event_ids, article_ids, article_topics)
    membership_counts = np.diff(event_offsets)
    membership_topics = np.repeat(paper_topics, membership_counts)
    cluster_count = int(cluster_labels.size)
    if event_clusters.size and (
        int(event_clusters.min()) < 0 or int(event_clusters.max()) >= cluster_count
    ):
        raise ValueError("Event cluster ids are outside cluster metadata")

    vector_lookup = {int(topic): index for index, topic in enumerate(topic_ids)}
    unknown = sorted(
        {
            int(topic)
            for topic in np.unique(membership_topics)
            if int(topic) not in {np.iinfo(np.int32).min, -1}
            and int(topic) not in vector_lookup
        }
    )
    if unknown:
        raise ValueError(
            "Topic assignments have no c-TF-IDF vectors: "
            + ", ".join(str(topic) for topic in unknown)
        )

    rows: list[dict[str, Any]] = []
    missing_sentinel = np.iinfo(np.int32).min
    for cluster in range(cluster_count):
        cluster_topics = membership_topics[event_clusters == cluster]
        paper_count = int(cluster_topics.size)
        missing_count = int(np.count_nonzero(cluster_topics == missing_sentinel))
        outlier_count = int(np.count_nonzero(cluster_topics == -1))
        valid = cluster_topics[
            (cluster_topics != missing_sentinel) & (cluster_topics != -1)
        ]
        counts = Counter(int(topic) for topic in valid)
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        observation_count = int(valid.size)
        distribution = [
            {
                "topic_id": topic,
                "label": labels.get(topic, f"Topic {topic}"),
                "papers": count,
                "share": count / observation_count if observation_count else 0.0,
            }
            for topic, count in ordered
        ]
        diversity = _rao_stirling_diversity(counts, vector_lookup, topic_vectors)
        rows.append(
            {
                "cluster_id": cluster,
                "label": str(cluster_labels[cluster]),
                "paper_count": paper_count,
                "topic_observation_count": observation_count,
                "topic_missing_count": missing_count,
                "topic_outlier_count": outlier_count,
                "rao_stirling_diversity": diversity,
                "topic_distribution": distribution,
            }
        )
    return rows


def summarize_cluster_author_topics(
    event_ids: np.ndarray,
    event_offsets: np.ndarray,
    event_clusters: np.ndarray,
    cluster_labels: np.ndarray,
    event_authors: Sequence[np.ndarray],
    article_ids: np.ndarray,
    article_topics: np.ndarray,
    author_profiles: Mapping[int, Mapping[int, float]],
    topic_ids: np.ndarray,
    topic_vectors: sparse.csr_matrix,
    labels: Mapping[int, str],
) -> list[dict[str, Any]]:
    """Summarize authors by primary topics inferred without current-cluster papers."""
    event_ids = np.asarray(event_ids, dtype=np.int64)
    event_offsets = np.asarray(event_offsets, dtype=np.int64)
    event_clusters = np.asarray(event_clusters, dtype=np.int32)
    if len(event_authors) != event_ids.size:
        raise ValueError("Event authors do not align with event paper ids")
    paper_topics = _align_article_topics(
        event_ids,
        np.asarray(article_ids, dtype=np.int64),
        np.asarray(article_topics, dtype=np.int32),
    )
    cluster_count = int(cluster_labels.size)
    author_weights: list[defaultdict[int, float]] = [
        defaultdict(float) for _ in range(cluster_count)
    ]
    excluded_weights: list[dict[int, defaultdict[int, float]]] = [
        {} for _ in range(cluster_count)
    ]
    missing_sentinel = np.iinfo(np.int32).min
    for paper_index, authors_value in enumerate(event_authors):
        authors = np.asarray(authors_value, dtype=np.int64)
        if authors.size == 0:
            continue
        contribution = 1.0 / authors.size
        topic = int(paper_topics[paper_index])
        clusters = event_clusters[
            event_offsets[paper_index] : event_offsets[paper_index + 1]
        ]
        for cluster_value in clusters:
            cluster = int(cluster_value)
            for author_value in authors:
                author = int(author_value)
                author_weights[cluster][author] += contribution
                if topic not in {missing_sentinel, -1}:
                    by_author = excluded_weights[cluster].setdefault(
                        author, defaultdict(float)
                    )
                    by_author[topic] += contribution

    vector_lookup = {int(topic): index for index, topic in enumerate(topic_ids)}
    rows: list[dict[str, Any]] = []
    for cluster in range(cluster_count):
        topic_weights: defaultdict[int, float] = defaultdict(float)
        topic_authors: Counter[int] = Counter()
        missing_weight = 0.0
        missing_authors = 0
        for author, cluster_weight in author_weights[cluster].items():
            profile = author_profiles.get(author, {})
            excluded = excluded_weights[cluster].get(author, {})
            candidates = []
            for topic, global_weight in profile.items():
                topic = int(topic)
                if topic == -1 or topic not in vector_lookup:
                    continue
                remaining = float(global_weight) - float(excluded.get(topic, 0.0))
                if remaining > 1e-12:
                    candidates.append((topic, remaining))
            if not candidates:
                missing_authors += 1
                missing_weight += cluster_weight
                continue
            primary_topic = min(
                candidates,
                key=lambda item: (-item[1], item[0]),
            )[0]
            topic_weights[primary_topic] += cluster_weight
            topic_authors[primary_topic] += 1

        total_weight = float(sum(author_weights[cluster].values()))
        covered_weight = float(sum(topic_weights.values()))
        ordered = sorted(
            topic_weights.items(),
            key=lambda item: (-item[1], item[0]),
        )
        distribution = [
            {
                "topic_id": topic,
                "label": labels.get(topic, f"Topic {topic}"),
                "authors": int(topic_authors[topic]),
                "weight": float(weight),
                "share": weight / covered_weight if covered_weight else 0.0,
            }
            for topic, weight in ordered
        ]
        rows.append(
            {
                "author_count": len(author_weights[cluster]),
                "author_primary_topic_count": int(sum(topic_authors.values())),
                "author_primary_topic_missing_count": missing_authors,
                "author_weight": total_weight,
                "author_primary_topic_weight": covered_weight,
                "author_primary_topic_missing_weight": missing_weight,
                "author_rao_stirling_diversity": _rao_stirling_diversity(
                    topic_weights,
                    vector_lookup,
                    topic_vectors,
                ),
                "author_topic_distribution": distribution,
            }
        )
    return rows


def _rao_stirling_diversity(
    counts: Mapping[int, int | float],
    vector_lookup: Mapping[int, int],
    topic_vectors: sparse.csr_matrix,
) -> float:
    total = sum(float(count) for count in counts.values())
    if total <= 0:
        return float("nan")
    rows = np.asarray([vector_lookup[int(topic)] for topic in counts], dtype=np.int64)
    weights = np.asarray(
        [float(counts[int(topic)]) / total for topic in counts],
        dtype=np.float64,
    )
    centroid = sparse.csr_matrix(weights.reshape(1, -1)) @ topic_vectors[rows]
    squared_norm = float(centroid.multiply(centroid).sum())
    return min(1.0, max(0.0, 1.0 - squared_norm))


def _align_article_topics(
    requested_ids: np.ndarray,
    article_ids: np.ndarray,
    article_topics: np.ndarray,
) -> np.ndarray:
    positions = np.searchsorted(article_ids, requested_ids)
    matched = positions < article_ids.size
    matched_indices = np.flatnonzero(matched)
    matched[matched_indices] = (
        article_ids[positions[matched_indices]] == requested_ids[matched_indices]
    )
    result = np.full(
        requested_ids.size,
        np.iinfo(np.int32).min,
        dtype=np.int32,
    )
    result[matched] = article_topics[positions[matched]]
    return result


def _load_event_authors(
    database: Path,
    event_ids: np.ndarray,
    *,
    batch_size: int = 800,
) -> list[np.ndarray]:
    authors: list[list[int]] = [[] for _ in event_ids]
    index = {int(article_id): position for position, article_id in enumerate(event_ids)}
    uri = f"file:{database}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only = ON")
        for start in range(0, event_ids.size, batch_size):
            batch = event_ids[start : start + batch_size]
            placeholders = ",".join("?" for _ in batch)
            rows = connection.execute(
                f"""
                SELECT article_id, author_id
                FROM articles_authors
                WHERE article_id IN ({placeholders})
                ORDER BY article_id, author_id
                """,
                [int(article_id) for article_id in batch],
            )
            for article_id, author_id in rows:
                authors[index[int(article_id)]].append(int(author_id))
    return [np.asarray(values, dtype=np.int64) for values in authors]


def _load_author_topic_weights(
    database: Path,
    author_ids: np.ndarray,
    *,
    batch_size: int = 800,
) -> dict[int, dict[int, float]]:
    profiles: dict[int, dict[int, float]] = {}
    uri = f"file:{database}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only = ON")
        for start in range(0, author_ids.size, batch_size):
            batch = author_ids[start : start + batch_size]
            placeholders = ",".join("?" for _ in batch)
            rows = connection.execute(
                f"""
                SELECT author_id, topic, fractional_weight
                FROM author_topics
                WHERE author_id IN ({placeholders})
                ORDER BY author_id, topic
                """,
                [int(author_id) for author_id in batch],
            )
            for author_id, topic, weight in rows:
                profiles.setdefault(int(author_id), {})[int(topic)] = float(weight)
    return profiles


def _normalized_topic_vectors(topic_model: object) -> tuple[np.ndarray, sparse.csr_matrix]:
    vectors = getattr(topic_model, "c_tf_idf_", None)
    get_topics = getattr(topic_model, "get_topics", None)
    if vectors is None or get_topics is None:
        raise ValueError("Saved BERTopic model has no c-TF-IDF topic vectors")
    matrix = sparse.csr_matrix(vectors, dtype=np.float64)
    topic_ids = np.asarray(sorted(int(topic) for topic in get_topics()), dtype=np.int32)
    if matrix.shape[0] != topic_ids.size:
        raise ValueError(
            "BERTopic c-TF-IDF rows do not align with the model's topic ids"
        )
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    if np.any(norms <= 0):
        invalid = topic_ids[norms <= 0]
        raise ValueError(
            "BERTopic has zero-length c-TF-IDF vectors for topics: "
            + ", ".join(str(int(topic)) for topic in invalid)
        )
    normalized = sparse.diags(1.0 / norms) @ matrix
    return topic_ids, normalized.tocsr()


def _load_topic_assignments(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Cluster-topic summaries require pyarrow. Install the topics extras."
        ) from exc
    table = pq.read_table(path, columns=["article_id", "topic"])
    article_ids = table.column("article_id").to_numpy(zero_copy_only=False).astype(
        np.int64, copy=False
    )
    topics = table.column("topic").to_numpy(zero_copy_only=False).astype(
        np.int32, copy=False
    )
    if article_ids.size and np.any(article_ids[1:] <= article_ids[:-1]):
        order = np.argsort(article_ids, kind="stable")
        article_ids = article_ids[order]
        topics = topics[order]
    if article_ids.size and np.any(article_ids[1:] == article_ids[:-1]):
        raise ValueError("Topic assignments contain duplicate article ids")
    return article_ids, topics


def _load_topic_labels(path: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"Topic", "Name"} <= set(reader.fieldnames):
            raise ValueError(f"{path} must contain Topic and Name columns")
        for row in reader:
            labels[int(row["Topic"])] = row["Name"]
    return labels


def _load_bertopic_model(path: Path) -> object:
    try:
        from bertopic import BERTopic
    except ImportError as exc:
        raise RuntimeError(
            "Loading c-TF-IDF vectors requires BERTopic. Install the topics extras."
        ) from exc
    return BERTopic.load(path)


def _write_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "cluster_id",
        "label",
        "paper_count",
        "topic_observation_count",
        "topic_missing_count",
        "topic_outlier_count",
        "rao_stirling_diversity",
        "topic_distribution",
        "author_count",
        "author_primary_topic_count",
        "author_primary_topic_missing_count",
        "author_weight",
        "author_primary_topic_weight",
        "author_primary_topic_missing_weight",
        "author_rao_stirling_diversity",
        "author_topic_distribution",
    ]
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            diversity = float(output["rao_stirling_diversity"])
            output["rao_stirling_diversity"] = (
                diversity if math.isfinite(diversity) else "nan"
            )
            output["topic_distribution"] = json.dumps(
                output["topic_distribution"], ensure_ascii=False, separators=(",", ":")
            )
            output.setdefault("author_count", 0)
            output.setdefault("author_primary_topic_count", 0)
            output.setdefault("author_primary_topic_missing_count", 0)
            output.setdefault("author_weight", 0.0)
            output.setdefault("author_primary_topic_weight", 0.0)
            output.setdefault("author_primary_topic_missing_weight", 0.0)
            author_diversity = float(
                output.get("author_rao_stirling_diversity", float("nan"))
            )
            output["author_rao_stirling_diversity"] = (
                author_diversity if math.isfinite(author_diversity) else "nan"
            )
            output["author_topic_distribution"] = json.dumps(
                output.get("author_topic_distribution", []),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            writer.writerow(output)
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize BERTopic distributions and diversity by event cluster."
    )
    parser.add_argument("--events-dir", default="output/events")
    parser.add_argument("--clusters-dir", default="output/event_clusters")
    parser.add_argument("--topics-dir", default="output/topics")
    parser.add_argument("--db-path", default="articles.db")
    parser.add_argument("--author-topics-dir", default="output/author_topics")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = build_parser().parse_args(argv)
    build_cluster_topic_summary(
        args.events_dir,
        args.clusters_dir,
        args.topics_dir,
        args.output_dir,
        db_path=args.db_path,
        author_topics_dir=args.author_topics_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
