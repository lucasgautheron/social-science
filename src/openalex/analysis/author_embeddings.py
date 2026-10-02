"""Aggregate article embeddings into fractional-authorship author embeddings."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from openalex.analysis.author_aggregation import (
    atomic_json,
    author_positions,
    build_author_index,
    configure_writable_sqlite,
    connect_readonly,
    fetch_authorships,
    file_sha256,
    flatten_authorships,
    manifest_sha256,
    read_json,
    source_metadata,
)
from openalex.analysis.embeddings import EmbeddingStore

logger = logging.getLogger(__name__)

ARTIFACT_VERSION = 1
DATABASE_NAME = "author_embeddings.db"
DEFAULT_OUTPUT_DIR = Path("output/author_embeddings")
DEFAULT_BATCH_SIZE = 20_000
DEFAULT_FINALIZE_BATCH_SIZE = 10_000
AGGREGATION = "sum((1/author_count) * embedding) / sum(1/author_count)"
ENCODING = "little-endian-float32"


def build_author_embeddings(
    db_path: str | Path,
    embeddings_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    finalize_batch_size: int = DEFAULT_FINALIZE_BATCH_SIZE,
    sqlite_cache_mb: int = 2048,
    resume: bool = False,
) -> dict[str, object]:
    """Build one normalized fractional-authorship embedding per author."""
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if finalize_batch_size < 1:
        raise ValueError("--finalize-batch-size must be >= 1")

    source = Path(db_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output == source or source.is_relative_to(output):
        raise ValueError("--output-dir must not contain the source database")
    store = EmbeddingStore(embeddings_path)
    if not store.manifest:
        raise ValueError("Author embeddings require a manifest-backed artifact")
    if not bool(store.manifest.get("complete")):
        raise ValueError("The article embedding artifact is incomplete")
    if store.dimension is None:
        raise ValueError("The article embedding dimension is unknown")
    if (
        output == store.root
        or output.is_relative_to(store.root)
        or store.root.is_relative_to(output)
    ):
        raise ValueError("--output-dir must not overlap --embeddings-dir")
    logger.info("Fingerprinting source corpus %s", source)
    source_info = source_metadata(source)
    _validate_corpus_provenance(store.manifest, source, source_info)
    logger.info("Fingerprinting embedding payload %s", store.database_path)
    embedding_database_sha256 = file_sha256(store.database_path)

    manifest_path = output / "manifest.json"
    database_path = output / DATABASE_NAME
    scratch = output / ".scratch"
    config = {
        "artifact_version": ARTIFACT_VERSION,
        "aggregation": AGGREGATION,
        "database": DATABASE_NAME,
        "dimension": int(store.dimension),
        "embedding_encoding": ENCODING,
        "embedding_database_sha256": embedding_database_sha256,
        "embedding_manifest_sha256": manifest_sha256(store.manifest_path),
        "model": store.manifest.get("model"),
        **source_info,
    }

    existing = read_json(manifest_path) if manifest_path.is_file() else None
    if existing is not None and resume:
        _validate_resume(existing, config)
        if bool(existing.get("complete")) and database_path.is_file():
            logger.info("Author embeddings are already complete in %s", output)
            return existing
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"{output} already contains results. Use --resume or a new --output-dir."
        )

    output.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    author_ids_path = scratch / "author_ids.npy"
    sums_path = scratch / "weighted_sums.npy"
    weights_path = scratch / "total_weights.npy"
    counts_path = scratch / "paper_counts.npy"
    pending_batch_path = scratch / "pending_batch.npz"

    if existing is not None and resume:
        for path in (author_ids_path, sums_path, weights_path, counts_path):
            if not path.is_file():
                raise RuntimeError(f"Resume scratch file is missing: {path}")
        author_ids = np.load(author_ids_path, mmap_mode="r")
        weighted_sums = np.load(sums_path, mmap_mode="r+")
        total_weights = np.load(weights_path, mmap_mode="r+")
        paper_counts = np.load(counts_path, mmap_mode="r+")
        last_article_id = existing.get("last_article_id")
        processed_articles = int(existing.get("processed_articles", 0))
        authorships = int(existing.get("authorships", 0))
        _recover_pending_batch(
            pending_batch_path,
            last_article_id,
            weighted_sums,
            total_weights,
            paper_counts,
        )
    else:
        with connect_readonly(source, cache_mb=sqlite_cache_mb) as connection:
            author_ids = build_author_index(connection, author_ids_path)
        author_count = len(author_ids)
        weighted_sums = np.lib.format.open_memmap(
            sums_path,
            mode="w+",
            dtype=np.float32,
            shape=(author_count, int(store.dimension)),
        )
        total_weights = np.lib.format.open_memmap(
            weights_path,
            mode="w+",
            dtype=np.float64,
            shape=(author_count,),
        )
        paper_counts = np.lib.format.open_memmap(
            counts_path,
            mode="w+",
            dtype=np.uint32,
            shape=(author_count,),
        )
        weighted_sums.fill(0)
        total_weights.fill(0)
        paper_counts.fill(0)
        last_article_id = None
        processed_articles = 0
        authorships = 0
        existing = {
            **config,
            "authors": 0,
            "authorships": 0,
            "complete": False,
            "last_article_id": None,
            "processed_articles": 0,
            "source_articles": store.count(),
        }
        _flush_memmaps(weighted_sums, total_weights, paper_counts)
        atomic_json(manifest_path, existing)

    with connect_readonly(source, cache_mb=sqlite_cache_mb) as connection:
        for article_ids, vectors in store.iter_batches(
            batch_size,
            start_after=(
                int(last_article_id) if last_article_id is not None else None
            ),
        ):
            memberships = fetch_authorships(connection, article_ids)
            paper_rows, batch_author_ids, fractional_weights = (
                flatten_authorships(article_ids, memberships)
            )
            if len(batch_author_ids):
                positions = author_positions(author_ids, batch_author_ids)
                (
                    batch_positions,
                    batch_weighted_sums,
                    batch_total_weights,
                    batch_paper_counts,
                ) = _aggregate_author_updates(
                    vectors,
                    paper_rows,
                    positions,
                    fractional_weights,
                )
                old_sums = np.asarray(
                    weighted_sums[batch_positions],
                    dtype=np.float32,
                )
                old_weights = np.asarray(
                    total_weights[batch_positions],
                    dtype=np.float64,
                )
                old_counts = np.asarray(
                    paper_counts[batch_positions],
                    dtype=np.uint32,
                )
                _write_pending_batch(
                    pending_batch_path,
                    previous_last_article_id=(
                        int(last_article_id)
                        if last_article_id is not None
                        else None
                    ),
                    next_last_article_id=int(article_ids[-1]),
                    positions=batch_positions,
                    old_sums=old_sums,
                    old_weights=old_weights,
                    old_counts=old_counts,
                )
                np.add(old_sums, batch_weighted_sums, out=old_sums)
                np.add(old_weights, batch_total_weights, out=old_weights)
                np.add(old_counts, batch_paper_counts, out=old_counts)
                weighted_sums[batch_positions] = old_sums
                total_weights[batch_positions] = old_weights
                paper_counts[batch_positions] = old_counts
                authorships += len(batch_author_ids)
            processed_articles += len(article_ids)
            last_article_id = article_ids[-1]
            _flush_memmaps(weighted_sums, total_weights, paper_counts)
            existing.update(
                {
                    "authorships": authorships,
                    "last_article_id": last_article_id,
                    "processed_articles": processed_articles,
                }
            )
            atomic_json(manifest_path, existing)
            pending_batch_path.unlink(missing_ok=True)
            logger.info(
                "Aggregated %s/%s article embeddings",
                processed_articles,
                existing["source_articles"],
            )

    author_count = _write_author_database(
        database_path,
        author_ids,
        weighted_sums,
        total_weights,
        paper_counts,
        int(store.dimension),
        finalize_batch_size,
        config,
    )
    final_manifest = {
        **existing,
        "authors": author_count,
        "complete": processed_articles == store.count(),
        "files": [DATABASE_NAME],
    }
    if not final_manifest["complete"]:
        raise RuntimeError(
            f"Processed {processed_articles} embeddings, expected {store.count()}"
        )
    atomic_json(manifest_path, final_manifest)
    shutil.rmtree(scratch)
    return final_manifest


def _aggregate_author_updates(
    vectors: np.ndarray,
    paper_rows: np.ndarray,
    positions: np.ndarray,
    fractional_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group a batch into one update per author using sparse matrix multiply."""
    unique_positions, local_author_rows = np.unique(
        positions,
        return_inverse=True,
    )
    author_paper_weights = csr_matrix(
        (
            fractional_weights.astype(np.float32, copy=False),
            (local_author_rows, paper_rows),
        ),
        shape=(len(unique_positions), len(vectors)),
        dtype=np.float32,
    )
    batch_weighted_sums = np.asarray(
        author_paper_weights @ vectors,
        dtype=np.float32,
    )
    batch_total_weights = np.bincount(
        local_author_rows,
        weights=fractional_weights,
        minlength=len(unique_positions),
    ).astype(np.float64, copy=False)
    batch_paper_counts = np.bincount(
        local_author_rows,
        minlength=len(unique_positions),
    ).astype(np.uint32, copy=False)
    return (
        unique_positions,
        batch_weighted_sums,
        batch_total_weights,
        batch_paper_counts,
    )


def _write_author_database(
    destination: Path,
    author_ids: np.ndarray,
    weighted_sums: np.ndarray,
    total_weights: np.ndarray,
    paper_counts: np.ndarray,
    dimension: int,
    batch_size: int,
    metadata: dict,
) -> int:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    for path in (
        temporary,
        Path(f"{temporary}-wal"),
        Path(f"{temporary}-shm"),
    ):
        path.unlink(missing_ok=True)
    written = 0
    with sqlite3.connect(temporary) as connection:
        configure_writable_sqlite(connection)
        connection.executescript(
            """
            CREATE TABLE metadata(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE author_embeddings(
                author_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                paper_count INTEGER NOT NULL,
                total_weight REAL NOT NULL
            );
            """
        )
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("artifact_version", str(ARTIFACT_VERSION)),
                ("aggregation", AGGREGATION),
                ("dimension", str(dimension)),
                ("embedding_encoding", ENCODING),
                ("model", str(metadata.get("model") or "")),
                ("source", json.dumps(metadata, sort_keys=True)),
            ],
        )
        for start in range(0, len(author_ids), batch_size):
            stop = min(start + batch_size, len(author_ids))
            weights = np.asarray(total_weights[start:stop])
            included = np.flatnonzero(weights > 0)
            rows = []
            for local_index in included:
                index = start + int(local_index)
                vector = np.asarray(
                    weighted_sums[index] / total_weights[index],
                    dtype="<f4",
                )
                rows.append(
                    (
                        int(author_ids[index]),
                        vector.tobytes(order="C"),
                        int(paper_counts[index]),
                        float(total_weights[index]),
                    )
                )
            connection.executemany(
                """
                INSERT INTO author_embeddings(
                    author_id, embedding, paper_count, total_weight
                ) VALUES (?, ?, ?, ?)
                """,
                rows,
            )
            connection.commit()
            written += len(rows)
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("PRAGMA journal_mode = DELETE")
    os.replace(temporary, destination)
    return written


def _flush_memmaps(*arrays: np.ndarray) -> None:
    for array in arrays:
        flush = getattr(array, "flush", None)
        if flush is not None:
            flush()


def _write_pending_batch(
    path: Path,
    *,
    previous_last_article_id: int | None,
    next_last_article_id: int,
    positions: np.ndarray,
    old_sums: np.ndarray,
    old_weights: np.ndarray,
    old_counts: np.ndarray,
) -> None:
    """Persist rows needed to roll back a partly checkpointed batch."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            has_previous=np.asarray(
                [previous_last_article_id is not None], dtype=np.bool_
            ),
            previous_last=np.asarray(
                [previous_last_article_id or 0], dtype=np.int64
            ),
            next_last=np.asarray([next_last_article_id], dtype=np.int64),
            positions=np.asarray(positions, dtype=np.int64),
            old_sums=np.asarray(old_sums, dtype=np.float32),
            old_weights=np.asarray(old_weights, dtype=np.float64),
            old_counts=np.asarray(old_counts, dtype=np.uint32),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _recover_pending_batch(
    path: Path,
    manifest_last_article_id: object,
    weighted_sums: np.ndarray,
    total_weights: np.ndarray,
    paper_counts: np.ndarray,
) -> None:
    """Complete or roll back a batch interrupted around its manifest update."""
    if not path.is_file():
        return
    with np.load(path) as pending:
        has_previous = bool(pending["has_previous"][0])
        previous_last = (
            int(pending["previous_last"][0]) if has_previous else None
        )
        next_last = int(pending["next_last"][0])
        manifest_last = (
            int(manifest_last_article_id)
            if manifest_last_article_id is not None
            else None
        )
        if manifest_last == next_last:
            path.unlink()
            return
        if manifest_last != previous_last:
            raise RuntimeError(
                "Pending embedding batch does not match the resume manifest"
            )
        positions = pending["positions"]
        weighted_sums[positions] = pending["old_sums"]
        total_weights[positions] = pending["old_weights"]
        paper_counts[positions] = pending["old_counts"]
    _flush_memmaps(weighted_sums, total_weights, paper_counts)
    path.unlink()


def _validate_resume(existing: dict, config: dict) -> None:
    for key, expected in config.items():
        if existing.get(key) != expected:
            raise ValueError(
                f"Existing author embedding artifact used "
                f"{key}={existing.get(key)!r}, not {expected!r}"
            )


def _validate_corpus_provenance(
    embedding_manifest: dict,
    source: Path,
    source_info: dict[str, int | str],
) -> None:
    upstream_sha256 = embedding_manifest.get("source_sha256")
    if upstream_sha256 is not None:
        if upstream_sha256 != source_info["source_sha256"]:
            raise ValueError(
                "The embedding artifact was built from a different corpus"
            )
        return
    upstream_source = embedding_manifest.get("source_database")
    if upstream_source is None or (
        Path(str(upstream_source)).expanduser().resolve() != source
    ):
        raise ValueError(
            "The embedding artifact has no matching corpus provenance"
        )
    upstream_size = embedding_manifest.get("source_size")
    if upstream_size is not None and int(upstream_size) != int(
        source_info["source_size"]
    ):
        raise ValueError(
            "The embedding artifact was built from a different corpus size"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate article embeddings into author embeddings."
    )
    parser.add_argument("--db-path", type=Path, default=Path("articles.db"))
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("output/embeddings"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--finalize-batch-size",
        type=int,
        default=DEFAULT_FINALIZE_BATCH_SIZE,
    )
    parser.add_argument("--sqlite-cache-mb", type=int, default=2048)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = build_parser().parse_args(argv)
    manifest = build_author_embeddings(
        args.db_path,
        args.embeddings_dir,
        args.output_dir,
        batch_size=args.batch_size,
        finalize_batch_size=args.finalize_batch_size,
        sqlite_cache_mb=args.sqlite_cache_mb,
        resume=args.resume,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
