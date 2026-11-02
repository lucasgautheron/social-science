"""Yearly collaborator-topic trends from leave-year-out primary topics."""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from openalex.artifacts.manifest import complete_manifest
from openalex.visualizations.cluster_topics import (
    _file_sha256,
    _load_bertopic_model,
    _load_topic_assignments,
    _load_topic_vectors,
    _normalized_topic_vectors,
    _rao_stirling_diversity,
    _read_json,
)

logger = logging.getLogger(__name__)

ARTIFACT_VERSION = 1
DEFAULT_OUTPUT_DIR = Path("output/global_trends")
DEFAULT_BREAK_YEAR = 2022
MEASURES = (
    "collaborator_topic_count",
    "out_of_topic_share",
    "rao_stirling_diversity",
)
_MISSING_TOPIC = int(np.iinfo(np.int32).min)


def build_global_trends(
    db_path: str | Path,
    topics_dir: str | Path,
    author_topics_dir: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    break_year: int = DEFAULT_BREAK_YEAR,
) -> Path:
    """Write yearly collaborator-topic means and their pre-break extrapolations."""
    database = Path(db_path).expanduser().resolve()
    topics = Path(topics_dir).expanduser().resolve()
    author_topics = Path(author_topics_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not database.is_file():
        raise ValueError(f"Article database not found: {database}")

    topic_manifest_path = topics / "manifest.json"
    topic_manifest = _read_json(topic_manifest_path)
    if int(topic_manifest.get("artifact_version", 0)) < 2:
        raise ValueError("Topic artifacts must use artifact version 2 or newer")
    assignments_path = topics / str(
        topic_manifest.get(
            "article_assignments",
            "article_topic_classifications.parquet",
        )
    )
    if not assignments_path.is_file():
        raise ValueError(f"Topic assignments not found: {assignments_path}")
    article_ids, article_topics = _load_topic_assignments(assignments_path)
    topic_ids, topic_vectors = _topic_vectors(topics, topic_manifest)

    author_manifest_path = author_topics / "manifest.json"
    author_manifest = _read_json(author_manifest_path)
    if not bool(author_manifest.get("complete")):
        raise ValueError(f"Author-topic artifact is incomplete: {author_manifest_path}")
    author_database = author_topics / str(
        author_manifest.get("database", "author_topics.db")
    )
    if not author_database.is_file():
        raise ValueError(f"Author-topic database not found: {author_database}")
    profiles = _load_author_profiles(author_database)
    vector_topics = {int(topic) for topic in topic_ids}
    vector_lookup = {int(topic): index for index, topic in enumerate(topic_ids)}

    years = []
    uri = f"file:{database}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only = ON")
        publication_years = [
            int(row[0])
            for row in connection.execute(
                """
                SELECT DISTINCT publication_year
                FROM articles
                WHERE publication_year IS NOT NULL
                ORDER BY publication_year
                """
            )
        ]
        for year in publication_years:
            summary = _summarize_database_year(
                connection,
                year,
                article_ids,
                article_topics,
                profiles,
                vector_topics,
                vector_lookup,
                topic_vectors,
            )
            years.append(summary)
            logger.info(
                "Year %s: %s eligible authors",
                year,
                summary["eligible_authors"],
            )

    series = _series_payload(years, break_year)
    output.mkdir(parents=True, exist_ok=True)
    series_path = output / "series.json"
    series_path.write_text(
        json.dumps(series, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "break_year": int(break_year),
        "series": series_path.name,
        "source_database": str(database),
        "topics_manifest_sha256": _file_sha256(topic_manifest_path),
        "author_topics_manifest_sha256": _file_sha256(author_manifest_path),
        "year_count": len(series["years"]),
        "primary_topic": (
            "largest fractional topic weight after subtracting papers from the plotted year"
        ),
        "population": (
            "authors with a classified collaborator on a paper from the plotted year"
        ),
    }
    complete_manifest(
        output,
        manifest,
        stage="global-trends",
        inputs=[
            {
                "role": "topics",
                "manifest_sha256": manifest["topics_manifest_sha256"],
            },
            {
                "role": "author_topics",
                "manifest_sha256": manifest["author_topics_manifest_sha256"],
            },
        ],
    )
    logger.info("Wrote global topic trends for %s years to %s", len(years), output)
    return series_path


def leave_year_out_primary_topic(
    topics: Sequence[int] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    year_weights: Mapping[int, float],
    vector_topics: set[int],
) -> int | None:
    """Select the largest positive topic weight after removing one year's papers."""
    best_topic: int | None = None
    best_weight: float | None = None
    for topic, weight in zip(topics, weights, strict=True):
        topic = int(topic)
        if topic == -1 or topic not in vector_topics:
            continue
        remaining = float(weight) - float(year_weights.get(topic, 0.0))
        if remaining <= 1e-12:
            continue
        if (
            best_topic is None
            or best_weight is None
            or remaining > best_weight
            or (remaining == best_weight and topic < best_topic)
        ):
            best_topic = topic
            best_weight = remaining
    return best_topic


def collaborator_topic_metrics(
    focal_topic: int,
    collaborator_topics: Sequence[int],
    vector_lookup: Mapping[int, int],
    topic_vectors: sparse.csr_matrix,
) -> tuple[int, float, float]:
    """Return distinct-topic count, out-of-topic share, and Rao--Stirling diversity."""
    if not collaborator_topics:
        raise ValueError("Collaborator metrics require at least one classified collaborator")
    counts = Counter(int(topic) for topic in collaborator_topics)
    total = len(collaborator_topics)
    out_share = sum(topic != int(focal_topic) for topic in counts.elements()) / total
    diversity = _rao_stirling_diversity(counts, vector_lookup, topic_vectors)
    return len(counts), float(out_share), float(diversity)


def fit_pre_break_trend(
    years: Sequence[int],
    values: Sequence[float],
    break_year: int,
) -> dict[str, float | list[int]] | None:
    """Fit an unweighted line to yearly means from years before the break."""
    points = [
        (int(year), float(value))
        for year, value in zip(years, values, strict=True)
        if int(year) < int(break_year)
    ]
    if len(points) < 2:
        return None
    x = np.asarray([year for year, _value in points], dtype=np.float64)
    y = np.asarray([value for _year, value in points], dtype=np.float64)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    denominator = float(np.sum((x - x_mean) ** 2))
    if denominator <= 0:
        return None
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denominator)
    intercept = float(y_mean - slope * x_mean)
    return {
        "intercept": intercept,
        "slope": slope,
        "years": [year for year, _value in points],
    }


def summarize_publication_year(
    paper_authors: Sequence[Sequence[int]],
    paper_topics: Sequence[int],
    author_ids: np.ndarray,
    profile_offsets: np.ndarray,
    profile_topics: np.ndarray,
    profile_weights: np.ndarray,
    vector_topics: set[int],
    vector_lookup: Mapping[int, int],
    topic_vectors: sparse.csr_matrix,
    *,
    year: int,
) -> dict[str, Any]:
    """Average collaborator-topic measures for one publication year."""
    if len(paper_authors) != len(paper_topics):
        raise ValueError("Paper authors and paper topics must have the same length")
    year_weights: dict[int, dict[int, float]] = {}
    coauthors: dict[int, set[int]] = {}
    for authors, topic in zip(paper_authors, paper_topics, strict=True):
        unique_authors = list(dict.fromkeys(int(author) for author in authors))
        author_count = len(unique_authors)
        if author_count == 0:
            continue
        topic = int(topic)
        if topic not in {_MISSING_TOPIC, -1}:
            share = 1.0 / author_count
            for author in unique_authors:
                weights = year_weights.setdefault(author, {})
                weights[topic] = weights.get(topic, 0.0) + share
        if author_count < 2:
            continue
        for author in unique_authors:
            others = coauthors.setdefault(author, set())
            others.update(
                other for other in unique_authors if other != author
            )

    primary_topics: dict[int, int | None] = {}

    def primary_topic(author: int) -> int | None:
        if author not in primary_topics:
            primary_topics[author] = _profile_primary_topic(
                author,
                year_weights.get(author, {}),
                author_ids,
                profile_offsets,
                profile_topics,
                profile_weights,
                vector_topics,
            )
        return primary_topics[author]

    totals = [0.0, 0.0, 0.0]
    collaborating = 0
    missing_primary = 0
    missing_collaborator = 0
    eligible = 0
    for author, others in coauthors.items():
        if not others:
            continue
        collaborating += 1
        focal = primary_topic(author)
        if focal is None:
            missing_primary += 1
            continue
        classified = [
            topic
            for other in others
            if (topic := primary_topic(other)) is not None
        ]
        if not classified:
            missing_collaborator += 1
            continue
        metrics = collaborator_topic_metrics(
            focal,
            classified,
            vector_lookup,
            topic_vectors,
        )
        eligible += 1
        totals = [total + value for total, value in zip(totals, metrics, strict=True)]

    means: list[float | None]
    if eligible:
        means = [total / eligible for total in totals]
    else:
        means = [None, None, None]
    row: dict[str, Any] = {
        "year": int(year),
        "collaborating_authors": collaborating,
        "authors_without_primary_topic": missing_primary,
        "authors_without_classified_collaborator": missing_collaborator,
        "eligible_authors": eligible,
    }
    for name, value in zip(MEASURES, means, strict=True):
        row[name] = value
    return row


def _series_payload(
    years: Sequence[Mapping[str, Any]],
    break_year: int,
) -> dict[str, Any]:
    rows = [dict(year) for year in years]
    fits: dict[str, Any] = {}
    for measure in MEASURES:
        observed = [
            (int(row["year"]), float(row[measure]))
            for row in rows
            if row[measure] is not None
        ]
        fit = fit_pre_break_trend(
            [year for year, _value in observed],
            [value for _year, value in observed],
            break_year,
        )
        fits[measure] = fit
        for row in rows:
            expected_name = f"expected_{measure}"
            if fit is None or int(row["year"]) < int(break_year):
                row[expected_name] = None
                continue
            row[expected_name] = (
                float(fit["intercept"]) + float(fit["slope"]) * int(row["year"])
            )
    return {
        "break_year": int(break_year),
        "years": rows,
        "ols": fits,
    }


def _summarize_database_year(
    connection: sqlite3.Connection,
    year: int,
    article_ids: np.ndarray,
    article_topics: np.ndarray,
    profiles: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    vector_topics: set[int],
    vector_lookup: Mapping[int, int],
    topic_vectors: sparse.csr_matrix,
) -> dict[str, Any]:
    paper_authors: list[list[int]] = []
    paper_topics: list[int] = []
    current_article: int | None = None
    authors: list[int] = []

    def flush() -> None:
        if current_article is None:
            return
        paper_authors.append(authors.copy())
        paper_topics.append(
            _assigned_topic(current_article, article_ids, article_topics)
        )

    rows = connection.execute(
        """
        SELECT articles.article_id, articles_authors.author_id
        FROM articles
        JOIN articles_authors
          ON articles_authors.article_id = articles.article_id
        WHERE articles.publication_year = ?
        ORDER BY articles.article_id, articles_authors.author_id
        """,
        (int(year),),
    )
    for article_id, author_id in rows:
        article_id = int(article_id)
        if article_id != current_article:
            flush()
            current_article = article_id
            authors = []
        authors.append(int(author_id))
    flush()
    author_ids, offsets, topics, weights = profiles
    return summarize_publication_year(
        paper_authors,
        paper_topics,
        author_ids,
        offsets,
        topics,
        weights,
        vector_topics,
        vector_lookup,
        topic_vectors,
        year=year,
    )


def _profile_primary_topic(
    author: int,
    year_weights: Mapping[int, float],
    author_ids: np.ndarray,
    offsets: np.ndarray,
    topics: np.ndarray,
    weights: np.ndarray,
    vector_topics: set[int],
) -> int | None:
    position = int(np.searchsorted(author_ids, author))
    if position >= author_ids.size or int(author_ids[position]) != author:
        return None
    start = int(offsets[position])
    stop = int(offsets[position + 1])
    return leave_year_out_primary_topic(
        topics[start:stop],
        weights[start:stop],
        year_weights,
        vector_topics,
    )


def _assigned_topic(
    article_id: int,
    article_ids: np.ndarray,
    article_topics: np.ndarray,
) -> int:
    position = int(np.searchsorted(article_ids, article_id))
    if position >= article_ids.size or int(article_ids[position]) != article_id:
        return _MISSING_TOPIC
    return int(article_topics[position])


def _load_author_profiles(
    database: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    author_ids: list[int] = []
    offsets = [0]
    topics: list[int] = []
    weights: list[float] = []
    current: int | None = None
    uri = f"file:{database}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only = ON")
        rows = connection.execute(
            """
            SELECT author_id, topic, fractional_weight
            FROM author_topics
            ORDER BY author_id, topic
            """
        )
        for author_id, topic, weight in rows:
            author_id = int(author_id)
            if author_id != current:
                if current is not None:
                    offsets.append(len(topics))
                author_ids.append(author_id)
                current = author_id
            topics.append(int(topic))
            weights.append(float(weight))
    if current is not None:
        offsets.append(len(topics))
    return (
        np.asarray(author_ids, dtype=np.int64),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(topics, dtype=np.int32),
        np.asarray(weights, dtype=np.float64),
    )


def _topic_vectors(
    topics: Path,
    manifest: Mapping[str, Any],
) -> tuple[np.ndarray, sparse.csr_matrix]:
    vector_ids = manifest.get("topic_vector_ids")
    vectors = manifest.get("topic_vectors")
    if vector_ids and vectors:
        return _load_topic_vectors(topics / str(vector_ids), topics / str(vectors))
    model_path = topics / "bertopic_model"
    if not model_path.exists():
        raise ValueError(
            "Topic artifact has neither reusable c-TF-IDF vectors nor a saved "
            "BERTopic model"
        )
    topic_model = _load_bertopic_model(model_path)
    return _normalized_topic_vectors(topic_model)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate yearly collaborator-topic breadth, out-of-topic share, "
            "and Rao--Stirling diversity."
        )
    )
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument("--topics-dir", type=Path, default=Path("output/topics"))
    parser.add_argument(
        "--author-topics-dir",
        type=Path,
        default=Path("output/author_topics"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--break-year", type=int, default=DEFAULT_BREAK_YEAR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_global_trends(
        args.db_path,
        args.topics_dir,
        args.author_topics_dir,
        args.output_dir,
        break_year=args.break_year,
    )
    return 0
