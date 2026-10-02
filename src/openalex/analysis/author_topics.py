"""Aggregate full-corpus article topics into sparse author distributions."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from openalex.analysis.author_aggregation import (
    atomic_json,
    configure_writable_sqlite,
    connect_readonly,
    fetch_authorships,
    file_sha256,
    flatten_authorships,
    manifest_sha256,
    read_json,
    source_metadata,
)

logger = logging.getLogger(__name__)

ARTIFACT_VERSION = 1
REQUIRED_TOPIC_ARTIFACT_VERSION = 2
DATABASE_NAME = "author_topics.db"
DEFAULT_OUTPUT_DIR = Path("output/author_topics")
DEFAULT_BATCH_SIZE = 50_000
AGGREGATION = (
    "hard MLP topic weighted by 1/article_author_count and normalized per author"
)


def build_author_topics(
    db_path: str | Path,
    topics_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sqlite_cache_mb: int = 2048,
    resume: bool = False,
) -> dict[str, object]:
    """Build sparse normalized author-topic distributions."""
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    source = Path(db_path).expanduser().resolve()
    topics_root = Path(topics_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output == source or source.is_relative_to(output):
        raise ValueError("--output-dir must not contain the source database")
    if (
        output == topics_root
        or output.is_relative_to(topics_root)
        or topics_root.is_relative_to(output)
    ):
        raise ValueError("--output-dir must not overlap --topics-dir")

    topic_manifest_path = topics_root / "manifest.json"
    if not topic_manifest_path.is_file():
        raise FileNotFoundError(
            f"Topic artifact manifest not found: {topic_manifest_path}"
        )
    topic_manifest = read_json(topic_manifest_path)
    if int(topic_manifest.get("artifact_version", 0)) != (
        REQUIRED_TOPIC_ARTIFACT_VERSION
    ):
        raise ValueError(
            "Author topics require topic artifact schema version "
            f"{REQUIRED_TOPIC_ARTIFACT_VERSION}"
        )
    assignments_name = topic_manifest.get("article_assignments")
    assignments_path = topics_root / str(assignments_name)
    if not assignments_path.is_file():
        raise FileNotFoundError(
            f"Topic classification Parquet not found: {assignments_path}"
        )
    topic_list_path = topics_root / "topic_list.csv"
    if not topic_list_path.is_file():
        raise FileNotFoundError(
            f"Canonical topic labels not found: {topic_list_path}"
        )
    logger.info("Fingerprinting source corpus %s", source)
    source_info = source_metadata(source)
    _validate_corpus_provenance(topic_manifest, source, source_info)
    logger.info("Fingerprinting topic assignments %s", assignments_path)
    assignments_sha256 = file_sha256(assignments_path)
    labels_sha256 = file_sha256(topic_list_path)

    manifest_path = output / "manifest.json"
    database_path = output / DATABASE_NAME
    scratch = output / ".scratch"
    accumulator_path = scratch / "author_topic_accumulator.db"
    config = {
        "artifact_version": ARTIFACT_VERSION,
        "aggregation": AGGREGATION,
        "database": DATABASE_NAME,
        "topic_assignments_sha256": assignments_sha256,
        "topic_labels_sha256": labels_sha256,
        "topic_manifest_sha256": manifest_sha256(topic_manifest_path),
        "topic_artifact_version": REQUIRED_TOPIC_ARTIFACT_VERSION,
        **source_info,
    }

    existing = read_json(manifest_path) if manifest_path.is_file() else None
    if existing is not None and resume:
        _validate_resume(existing, config)
        if bool(existing.get("complete")) and database_path.is_file():
            logger.info("Author topics are already complete in %s", output)
            return existing
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"{output} already contains results. Use --resume or a new --output-dir."
        )

    output.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    _initialize_accumulator(accumulator_path)
    last_article_id, processed_articles = _read_progress(accumulator_path)
    expected_articles = int(topic_manifest.get("articles", 0))
    if existing is None:
        existing = {
            **config,
            "article_topic_pairs": 0,
            "authors": 0,
            "complete": False,
            "last_article_id": last_article_id,
            "processed_articles": processed_articles,
            "source_articles": expected_articles,
        }
        atomic_json(manifest_path, existing)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Author-topic aggregation requires pyarrow. Install the topics extras."
        ) from exc

    parquet = pq.ParquetFile(assignments_path)
    expected_schema = {
        "article_id": pa.int64(),
        "topic": pa.int32(),
        "probability": pa.float32(),
    }
    for name, expected_type in expected_schema.items():
        field = parquet.schema_arrow.field(name)
        if field.type != expected_type:
            raise ValueError(
                f"Topic column {name!r} has type {field.type}, "
                f"expected {expected_type}"
            )

    previous_parquet_id: int | None = None
    with connect_readonly(source, cache_mb=sqlite_cache_mb) as source_connection:
        for record_batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=["article_id", "topic"],
        ):
            article_ids = record_batch.column(0).to_numpy(
                zero_copy_only=False
            ).astype(np.int64, copy=False)
            topics = record_batch.column(1).to_numpy(
                zero_copy_only=False
            ).astype(np.int32, copy=False)
            if len(article_ids) == 0:
                continue
            if previous_parquet_id is not None and article_ids[0] <= (
                previous_parquet_id
            ):
                raise ValueError(
                    "Topic classifications must be strictly ordered by article_id"
                )
            if np.any(article_ids[1:] <= article_ids[:-1]):
                raise ValueError(
                    "Topic classifications must be strictly ordered by article_id"
                )
            previous_parquet_id = int(article_ids[-1])
            if last_article_id is not None:
                keep = article_ids > int(last_article_id)
                article_ids = article_ids[keep]
                topics = topics[keep]
            if len(article_ids) == 0:
                continue

            article_id_list = article_ids.tolist()
            memberships = fetch_authorships(
                source_connection, article_id_list
            )
            paper_rows, author_ids, fractional_weights = flatten_authorships(
                article_id_list, memberships
            )
            pair_rows = _aggregate_pairs(
                author_ids,
                topics[paper_rows],
                fractional_weights,
            )
            processed_articles += len(article_ids)
            last_article_id = int(article_ids[-1])
            _commit_accumulator_batch(
                accumulator_path,
                pair_rows,
                last_article_id,
                processed_articles,
            )
            existing.update(
                {
                    "last_article_id": last_article_id,
                    "processed_articles": processed_articles,
                }
            )
            atomic_json(manifest_path, existing)
            logger.info(
                "Aggregated topics for %s/%s articles",
                processed_articles,
                expected_articles,
            )

    author_count, pair_count = _write_final_database(
        database_path,
        accumulator_path,
        topic_list_path,
        config,
    )
    if expected_articles and processed_articles != expected_articles:
        raise RuntimeError(
            f"Processed {processed_articles} topic rows, "
            f"expected {expected_articles}"
        )
    final_manifest = {
        **existing,
        "article_topic_pairs": pair_count,
        "authors": author_count,
        "complete": True,
        "files": [DATABASE_NAME],
    }
    atomic_json(manifest_path, final_manifest)
    shutil.rmtree(scratch)
    return final_manifest


def _initialize_accumulator(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        configure_writable_sqlite(connection)
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS contributions(
                author_id INTEGER NOT NULL,
                topic INTEGER NOT NULL,
                fractional_weight REAL NOT NULL,
                paper_count INTEGER NOT NULL,
                PRIMARY KEY(author_id, topic)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS progress(
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                last_article_id INTEGER,
                processed_articles INTEGER NOT NULL
            );
            INSERT OR IGNORE INTO progress(
                singleton, last_article_id, processed_articles
            ) VALUES (1, NULL, 0);
            """
        )


def _read_progress(path: Path) -> tuple[int | None, int]:
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            """
            SELECT last_article_id, processed_articles
            FROM progress
            WHERE singleton = 1
            """
        ).fetchone()
    return (
        int(row[0]) if row and row[0] is not None else None,
        int(row[1]) if row else 0,
    )


def _aggregate_pairs(
    author_ids: np.ndarray,
    topics: np.ndarray,
    weights: np.ndarray,
) -> list[tuple[int, int, float, int]]:
    if len(author_ids) == 0:
        return []
    order = np.lexsort((topics, author_ids))
    sorted_authors = author_ids[order]
    sorted_topics = topics[order]
    sorted_weights = weights[order]
    boundaries = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(
                (sorted_authors[1:] != sorted_authors[:-1])
                | (sorted_topics[1:] != sorted_topics[:-1])
            )
            + 1,
        )
    )
    reduced_weights = np.add.reduceat(sorted_weights, boundaries)
    ends = np.append(boundaries[1:], len(sorted_authors))
    counts = ends - boundaries
    return [
        (
            int(sorted_authors[index]),
            int(sorted_topics[index]),
            float(reduced_weights[row]),
            int(counts[row]),
        )
        for row, index in enumerate(boundaries)
    ]


def _commit_accumulator_batch(
    path: Path,
    rows: Sequence[tuple[int, int, float, int]],
    last_article_id: int,
    processed_articles: int,
) -> None:
    with sqlite3.connect(path) as connection:
        configure_writable_sqlite(connection)
        connection.executemany(
            """
            INSERT INTO contributions(
                author_id, topic, fractional_weight, paper_count
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(author_id, topic) DO UPDATE SET
                fractional_weight = (
                    contributions.fractional_weight
                    + excluded.fractional_weight
                ),
                paper_count = contributions.paper_count + excluded.paper_count
            """,
            rows,
        )
        connection.execute(
            """
            UPDATE progress
            SET last_article_id = ?, processed_articles = ?
            WHERE singleton = 1
            """,
            (last_article_id, processed_articles),
        )
        connection.commit()


def _write_final_database(
    destination: Path,
    accumulator: Path,
    topic_list_path: Path,
    metadata: dict,
) -> tuple[int, int]:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    for path in (
        temporary,
        Path(f"{temporary}-wal"),
        Path(f"{temporary}-shm"),
    ):
        path.unlink(missing_ok=True)
    with sqlite3.connect(temporary) as connection:
        configure_writable_sqlite(connection)
        connection.executescript(
            """
            CREATE TABLE metadata(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE authors(
                author_id INTEGER PRIMARY KEY,
                total_weight REAL NOT NULL,
                paper_count INTEGER NOT NULL,
                topic_count INTEGER NOT NULL
            );
            CREATE TABLE author_topics(
                author_id INTEGER NOT NULL,
                topic INTEGER NOT NULL,
                probability REAL NOT NULL,
                fractional_weight REAL NOT NULL,
                PRIMARY KEY(author_id, topic)
            ) WITHOUT ROWID;
            CREATE TABLE topics(
                topic INTEGER PRIMARY KEY,
                label TEXT NOT NULL
            );
            """
        )
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("aggregation", AGGREGATION),
                ("artifact_version", str(ARTIFACT_VERSION)),
                ("source", json.dumps(metadata, sort_keys=True)),
            ],
        )
        with topic_list_path.open(newline="", encoding="utf-8") as handle:
            topic_labels = {
                int(row["Topic"]): str(row.get("Name", row["Topic"]))
                for row in csv.DictReader(handle)
            }
        connection.executemany(
            "INSERT INTO topics(topic, label) VALUES (?, ?)",
            sorted(topic_labels.items()),
        )
        connection.execute("ATTACH DATABASE ? AS accumulator", (str(accumulator),))
        connection.execute(
            """
            INSERT INTO authors(
                author_id, total_weight, paper_count, topic_count
            )
            SELECT
                author_id,
                SUM(fractional_weight),
                SUM(paper_count),
                COUNT(*)
            FROM accumulator.contributions
            GROUP BY author_id
            """
        )
        connection.execute(
            """
            INSERT INTO author_topics(
                author_id, topic, probability, fractional_weight
            )
            SELECT
                c.author_id,
                c.topic,
                c.fractional_weight / a.total_weight,
                c.fractional_weight
            FROM accumulator.contributions c
            JOIN authors a ON a.author_id = c.author_id
            """
        )
        invalid = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT author_id
                    FROM author_topics
                    GROUP BY author_id
                    HAVING ABS(SUM(probability) - 1.0) > 1e-9
                )
                """
            ).fetchone()[0]
        )
        if invalid:
            raise RuntimeError(
                f"{invalid} author-topic distributions do not sum to one"
            )
        missing_labels = [
            int(row[0])
            for row in connection.execute(
                """
                SELECT DISTINCT a.topic
                FROM author_topics a
                LEFT JOIN topics t ON t.topic = a.topic
                WHERE t.topic IS NULL
                ORDER BY a.topic
                """
            )
        ]
        if missing_labels:
            raise ValueError(
                "Canonical topic labels are missing topics: "
                + ", ".join(str(topic) for topic in missing_labels)
            )
        connection.execute(
            "CREATE INDEX idx_author_topics_topic ON author_topics(topic)"
        )
        author_count = int(
            connection.execute("SELECT COUNT(*) FROM authors").fetchone()[0]
        )
        pair_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM author_topics"
            ).fetchone()[0]
        )
        connection.commit()
        connection.execute("DETACH DATABASE accumulator")
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("PRAGMA journal_mode = DELETE")
    os.replace(temporary, destination)
    return author_count, pair_count


def _validate_resume(existing: dict, config: dict) -> None:
    for key, expected in config.items():
        if existing.get(key) != expected:
            raise ValueError(
                f"Existing author-topic artifact used "
                f"{key}={existing.get(key)!r}, not {expected!r}"
            )


def _validate_corpus_provenance(
    topic_manifest: dict,
    source: Path,
    source_info: dict[str, int | str],
) -> None:
    upstream_sha256 = topic_manifest.get("source_sha256")
    if upstream_sha256 is not None:
        if upstream_sha256 != source_info["source_sha256"]:
            raise ValueError("The topic artifact was built from a different corpus")
        return
    upstream_source = topic_manifest.get("source_database")
    if upstream_source is None or (
        Path(str(upstream_source)).expanduser().resolve() != source
    ):
        raise ValueError("The topic artifact has no matching corpus provenance")
    upstream_size = topic_manifest.get("source_size")
    if upstream_size is not None and int(upstream_size) != int(
        source_info["source_size"]
    ):
        raise ValueError("The topic artifact was built from a different corpus size")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build fractional-authorship topic distributions."
    )
    parser.add_argument("--db-path", type=Path, default=Path("articles.db"))
    parser.add_argument(
        "--topics-dir",
        type=Path,
        default=Path("output/topics"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--sqlite-cache-mb", type=int, default=2048)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = build_parser().parse_args(argv)
    manifest = build_author_topics(
        args.db_path,
        args.topics_dir,
        args.output_dir,
        batch_size=args.batch_size,
        sqlite_cache_mb=args.sqlite_cache_mb,
        resume=args.resume,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
