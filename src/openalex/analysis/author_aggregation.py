"""Shared streaming helpers for author-level article aggregation."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

import numpy as np

SQLITE_IN_LIMIT = 900
DEFAULT_SQLITE_CACHE_MB = 2048


@contextmanager
def connect_readonly(
    database: str | Path,
    *,
    cache_mb: int = DEFAULT_SQLITE_CACHE_MB,
) -> Iterator[sqlite3.Connection]:
    """Open an immutable input database with a bounded SQLite page cache."""
    path = Path(database).expanduser().resolve()
    uri = f"file:{quote(str(path))}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA temp_store = FILE")
    connection.execute(f"PRAGMA cache_size = {-max(cache_mb, 1) * 1024}")
    try:
        yield connection
    finally:
        connection.close()


def fetch_authorships(
    connection: sqlite3.Connection,
    article_ids: Sequence[int],
) -> dict[int, list[int]]:
    """Fetch exact article-author memberships through the composite PK."""
    memberships: dict[int, list[int]] = {}
    for chunk in chunks(article_ids, SQLITE_IN_LIMIT):
        placeholders = ",".join("?" for _ in chunk)
        rows = connection.execute(
            f"""
            SELECT article_id, author_id
            FROM articles_authors
            WHERE article_id IN ({placeholders})
            ORDER BY article_id, author_id
            """,
            [int(article_id) for article_id in chunk],
        )
        for article_id, author_id in rows:
            if author_id is None:
                raise ValueError(f"Article {article_id} has a null author_id")
            article_id = int(article_id)
            author_id = int(author_id)
            authors = memberships.setdefault(article_id, [])
            if authors and authors[-1] == author_id:
                raise ValueError(
                    f"Article {article_id} contains duplicate author {author_id}"
                )
            authors.append(author_id)
    return memberships


def flatten_authorships(
    article_ids: Sequence[int],
    memberships: dict[int, list[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return paper rows, author IDs, and fractional 1/n weights."""
    paper_rows: list[int] = []
    author_ids: list[int] = []
    weights: list[float] = []
    for paper_row, article_id in enumerate(article_ids):
        authors = memberships.get(int(article_id), [])
        if not authors:
            continue
        weight = 1.0 / len(authors)
        paper_rows.extend([paper_row] * len(authors))
        author_ids.extend(authors)
        weights.extend([weight] * len(authors))
    return (
        np.asarray(paper_rows, dtype=np.int64),
        np.asarray(author_ids, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
    )


def build_author_index(
    connection: sqlite3.Connection,
    destination: Path,
    *,
    fetch_size: int = 100_000,
) -> np.memmap:
    """Write sorted corpus author IDs to a disk-backed NPY array."""
    total = int(connection.execute("SELECT COUNT(*) FROM authors").fetchone()[0])
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    author_ids = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.int64,
        shape=(total,),
    )
    cursor = connection.execute("SELECT author_id FROM authors ORDER BY author_id")
    offset = 0
    previous_author_id: int | None = None
    while rows := cursor.fetchmany(fetch_size):
        values = np.fromiter(
            (int(row[0]) for row in rows),
            dtype=np.int64,
            count=len(rows),
        )
        if (
            (previous_author_id is not None and values[0] <= previous_author_id)
            or (len(values) > 1 and np.any(values[1:] <= values[:-1]))
        ):
            raise ValueError(
                "authors.author_id must be unique and strictly increasing"
            )
        author_ids[offset : offset + len(values)] = values
        offset += len(values)
        previous_author_id = int(values[-1])
    if offset != total:
        raise RuntimeError(f"Read {offset} authors, expected {total}")
    author_ids.flush()
    del author_ids
    os.replace(temporary, destination)
    return np.load(destination, mmap_mode="r")


def author_positions(
    sorted_author_ids: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Map author IDs to dense rows and reject broken corpus references."""
    positions = np.searchsorted(sorted_author_ids, values)
    valid = positions < len(sorted_author_ids)
    if np.any(valid):
        valid_indices = np.flatnonzero(valid)
        valid[valid_indices] = (
            sorted_author_ids[positions[valid_indices]]
            == values[valid_indices]
        )
    if not np.all(valid):
        missing = int(values[np.flatnonzero(~valid)[0]])
        raise ValueError(
            f"articles_authors references author {missing}, absent from authors"
        )
    return positions.astype(np.int64, copy=False)


def configure_writable_sqlite(connection: sqlite3.Connection) -> None:
    """Tune a new sidecar database for batched artifact construction."""
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA temp_store = FILE")
    connection.execute("PRAGMA cache_size = -262144")


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def manifest_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a payload incrementally without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def source_metadata(path: Path) -> dict[str, int | str]:
    source = path.expanduser().resolve()
    return {
        "source_database": str(source),
        "source_sha256": file_sha256(source),
        "source_size": source.stat().st_size,
    }


def chunks(values: Sequence[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield [int(value) for value in values[start : start + size]]

