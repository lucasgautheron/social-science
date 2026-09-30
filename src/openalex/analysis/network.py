"""Yearly Newman coauthorship matrices.

Each publication year becomes one symmetric CSR adjacency matrix. For a paper
with n >= 2 authors, every unordered pair gains weight 1/(n-1). Rows and
columns share one sorted author index across years.

The builder streams one year at a time from a read-only SQLite corpus so a
database of tens of millions of papers stays on disk.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)

DEFAULT_MAX_AUTHORS = 16
DEFAULT_MAX_EDGES = 200_000_000
DEFAULT_CACHE_MB = 2048
_FETCH_SIZE = 200_000
_LOG_EVERY_PAPERS = 500_000
_AUTHOR_FLUSH = 100_000
_INT32_MAX = int(np.iinfo(np.int32).max)
WEIGHT_DEFINITION = (
    "w_ij += 1/(n-1) for each paper with n>=2 authors shared by i and j"
)

_YEAR_QUERY = """
SELECT aa.article_id, aa.author_id
FROM articles a
JOIN articles_authors aa ON aa.article_id = a.article_id
WHERE a.publication_year = ?
ORDER BY aa.article_id
"""


def build_yearly_coauthorship(
    db_path: str | Path,
    output_dir: str | Path,
    *,
    from_year: int | None = None,
    to_year: int | None = None,
    max_authors: int = DEFAULT_MAX_AUTHORS,
    max_edges: int = DEFAULT_MAX_EDGES,
    sqlite_cache_mb: int = DEFAULT_CACHE_MB,
    resume: bool = False,
    fetch_size: int = _FETCH_SIZE,
) -> None:
    """Write one shared author index and one CSR matrix per publication year."""
    if max_authors < 0:
        raise ValueError("--max-authors must be >= 0")
    if max_edges < 1:
        raise ValueError("--max-edges must be >= 1")
    if from_year is not None and to_year is not None and from_year > to_year:
        raise ValueError("--from-year cannot be greater than --to-year")
    if fetch_size < 1:
        raise ValueError("fetch_size must be >= 1")

    output_dir = Path(output_dir)
    database = Path(db_path)
    scratch = output_dir / "scratch"
    years_dir = output_dir / "years"
    manifest_path = output_dir / "manifest.json"
    config = {
        "from_year": from_year,
        "max_authors": max_authors,
        "max_edges": max_edges,
        "to_year": to_year,
    }

    existing = _load_manifest(manifest_path) if resume and manifest_path.exists() else None
    if existing is not None:
        _check_config(existing, config)
        if _is_complete(existing, output_dir):
            shutil.rmtree(scratch, ignore_errors=True)
            logger.info("Coauthorship matrices already built in %s", output_dir)
            return
    elif not resume and _output_exists(output_dir):
        raise FileExistsError(
            f"{output_dir} already contains coauthorship files. Use --resume or a new --output-dir."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    years_dir.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)

    if existing is not None and existing.get("phase1_complete") and _author_index_path(output_dir).exists():
        years = [int(year) for year in existing["years"]]
        author_ids = np.load(_author_index_path(output_dir), allow_pickle=False)
        manifest = existing
    else:
        with _connect(database, scratch, sqlite_cache_mb) as connection:
            years = _list_years(connection, from_year, to_year)
            for year in years:
                if _scratch_ready(scratch, year):
                    logger.info("Reusing year %s scratch", year)
                    continue
                _scan_year(connection, year, scratch, max_authors, fetch_size)
        author_ids = _write_author_index(scratch, years, output_dir)
        manifest = _fresh_manifest(config, years, author_ids, scratch)
        _atomic_json(manifest_path, manifest)

    _check_index_limit(author_ids)
    completed = {int(year) for year in manifest.get("completed_years", [])}
    for year in years:
        destination = _year_matrix_path(output_dir, year)
        if year in completed and destination.exists():
            logger.info("Skipping finished year %s", year)
            continue
        if not _scratch_ready(scratch, year):
            with _connect(database, scratch, sqlite_cache_mb) as connection:
                _scan_year(connection, year, scratch, max_authors, fetch_size)
        started = time.perf_counter()
        nnz = _write_year_matrix(scratch, year, author_ids, output_dir, max_edges)
        elapsed = time.perf_counter() - started
        stats = _read_json(_stats_path(scratch, year))
        manifest["nnz"][str(year)] = nnz
        manifest["papers_kept"][str(year)] = int(stats["papers_kept"])
        manifest["solo_papers"][str(year)] = int(stats["solo_papers"])
        manifest["hyperauthored_papers"][str(year)] = int(stats["hyperauthored_papers"])
        completed.add(year)
        manifest["completed_years"] = sorted(completed)
        _atomic_json(manifest_path, manifest)
        logger.info(
            "Year %s: kept=%s solo=%s hyper=%s nnz=%s elapsed=%.1fs",
            year,
            stats["papers_kept"],
            stats["solo_papers"],
            stats["hyperauthored_papers"],
            nnz,
            elapsed,
        )

    shutil.rmtree(scratch, ignore_errors=True)
    logger.info("Wrote %s years for %s authors to %s", len(years), len(author_ids), output_dir)


def load_coauthorship(output_dir: str | Path, year: int) -> tuple[np.ndarray, sparse.csr_matrix]:
    """Load the shared author index and one year's CSR matrix."""
    output_dir = Path(output_dir)
    author_ids = np.load(_author_index_path(output_dir), allow_pickle=False)
    matrix = sparse.load_npz(_year_matrix_path(output_dir, year)).tocsr()
    return author_ids, matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build yearly Newman coauthorship matrices.")
    parser.add_argument("--db-path", default="articles.db", help="Read-only source corpus.")
    parser.add_argument("--output-dir", default="output/coauthorship")
    parser.add_argument("--from-year", type=int, default=None)
    parser.add_argument("--to-year", type=int, default=None)
    parser.add_argument(
        "--max-authors",
        type=int,
        default=DEFAULT_MAX_AUTHORS,
        help="Skip papers with more authors than this. 0 disables the cap.",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=DEFAULT_MAX_EDGES,
        help="Refuse a year whose directed-entry upper bound exceeds this.",
    )
    parser.add_argument("--sqlite-cache-mb", type=int, default=DEFAULT_CACHE_MB)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args(argv)
    build_yearly_coauthorship(
        args.db_path,
        args.output_dir,
        from_year=args.from_year,
        to_year=args.to_year,
        max_authors=args.max_authors,
        max_edges=args.max_edges,
        sqlite_cache_mb=args.sqlite_cache_mb,
        resume=args.resume,
    )
    return 0


@contextmanager
def _connect(db_path: Path, scratch: Path, cache_mb: int) -> Iterator[sqlite3.Connection]:
    scratch.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get("SQLITE_TMPDIR")
    os.environ["SQLITE_TMPDIR"] = str(scratch.resolve())
    uri = f"file:{quote(str(db_path.resolve()))}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute(f"PRAGMA cache_size = {-max(int(cache_mb), 1) * 1024}")
        connection.execute("PRAGMA temp_store = FILE")
        connection.execute("PRAGMA query_only = ON")
        yield connection
    finally:
        connection.close()
        if previous is None:
            os.environ.pop("SQLITE_TMPDIR", None)
        else:
            os.environ["SQLITE_TMPDIR"] = previous


def _list_years(
    connection: sqlite3.Connection,
    from_year: int | None,
    to_year: int | None,
) -> list[int]:
    clauses = ["publication_year IS NOT NULL"]
    parameters: list[int] = []
    if from_year is not None:
        clauses.append("publication_year >= ?")
        parameters.append(from_year)
    if to_year is not None:
        clauses.append("publication_year <= ?")
        parameters.append(to_year)
    where = " AND ".join(clauses)
    rows = connection.execute(
        f"SELECT DISTINCT publication_year FROM articles WHERE {where} ORDER BY publication_year",
        parameters,
    )
    return [int(row[0]) for row in rows]


def _scan_year(
    connection: sqlite3.Connection,
    year: int,
    scratch: Path,
    max_authors: int,
    fetch_size: int,
) -> None:
    for path in (_authors_path(scratch, year), _counts_path(scratch, year), _unique_path(scratch, year)):
        path.unlink(missing_ok=True)
    _stats_path(scratch, year).unlink(missing_ok=True)

    started = time.perf_counter()
    papers_seen = 0
    papers_kept = 0
    solo_papers = 0
    hyperauthored_papers = 0
    edge_bound = 0
    author_buffer: list[int] = []
    count_buffer: list[int] = []

    logger.info("Scanning year %s", year)
    with _authors_path(scratch, year).open("wb") as author_handle, _counts_path(scratch, year).open(
        "wb"
    ) as count_handle:
        cursor = connection.execute(_YEAR_QUERY, (year,))
        for authors in _iter_papers(cursor, fetch_size):
            papers_seen += 1
            unique_authors = list(dict.fromkeys(authors))
            author_count = len(unique_authors)
            if author_count < 2:
                solo_papers += 1
            elif max_authors > 0 and author_count > max_authors:
                hyperauthored_papers += 1
            else:
                author_buffer.extend(unique_authors)
                count_buffer.append(author_count)
                papers_kept += 1
                edge_bound += author_count * (author_count - 1)
                if len(author_buffer) >= _AUTHOR_FLUSH:
                    _flush_buffers(author_handle, count_handle, author_buffer, count_buffer)
            if papers_seen % _LOG_EVERY_PAPERS == 0:
                logger.info("Year %s: scanned %s papers", year, papers_seen)
        _flush_buffers(author_handle, count_handle, author_buffer, count_buffer)
        author_handle.flush()
        count_handle.flush()
        os.fsync(author_handle.fileno())
        os.fsync(count_handle.fileno())

    _write_year_uniques(scratch, year)
    _atomic_json(
        _stats_path(scratch, year),
        {
            "edge_bound": edge_bound,
            "hyperauthored_papers": hyperauthored_papers,
            "papers_kept": papers_kept,
            "solo_papers": solo_papers,
        },
    )
    elapsed = time.perf_counter() - started
    logger.info(
        "Scanned year %s in %.1fs: kept=%s solo=%s hyper=%s",
        year,
        elapsed,
        papers_kept,
        solo_papers,
        hyperauthored_papers,
    )


def _iter_papers(cursor: sqlite3.Cursor, fetch_size: int) -> Iterator[list[int]]:
    current_id: int | None = None
    current_authors: list[int] = []
    while True:
        rows = cursor.fetchmany(fetch_size)
        if not rows:
            break
        for article_id, author_id in rows:
            article_id = int(article_id)
            if author_id is None:
                raise ValueError(f"article {article_id} has a null author_id")
            author_id = int(author_id)
            if current_id is None:
                current_id = article_id
                current_authors = [author_id]
            elif article_id == current_id:
                current_authors.append(author_id)
            else:
                yield current_authors
                current_id = article_id
                current_authors = [author_id]
    if current_id is not None:
        yield current_authors


def _flush_buffers(
    author_handle,
    count_handle,
    author_buffer: list[int],
    count_buffer: list[int],
) -> None:
    if author_buffer:
        np.asarray(author_buffer, dtype=np.int64).tofile(author_handle)
        author_buffer.clear()
    if count_buffer:
        np.asarray(count_buffer, dtype=np.int32).tofile(count_handle)
        count_buffer.clear()


def _write_year_uniques(scratch: Path, year: int) -> None:
    authors = np.fromfile(_authors_path(scratch, year), dtype=np.int64)
    unique = np.unique(authors) if authors.size else np.empty(0, dtype=np.int64)
    temporary = _unique_path(scratch, year).with_suffix(".i64.tmp")
    unique.tofile(temporary)
    os.replace(temporary, _unique_path(scratch, year))


def _write_author_index(scratch: Path, years: Sequence[int], output_dir: Path) -> np.ndarray:
    parts = [np.fromfile(_unique_path(scratch, year), dtype=np.int64) for year in years]
    if not parts:
        author_ids = np.empty(0, dtype=np.int64)
    else:
        author_ids = np.unique(np.concatenate(parts))
    _check_index_limit(author_ids)
    temporary = output_dir / ".author_ids.tmp.npy"
    np.save(temporary, author_ids, allow_pickle=False)
    os.replace(temporary, _author_index_path(output_dir))
    logger.info("Author index has %s ids", len(author_ids))
    return author_ids


def _write_year_matrix(
    scratch: Path,
    year: int,
    author_ids: np.ndarray,
    output_dir: Path,
    max_edges: int,
) -> int:
    counts = np.fromfile(_counts_path(scratch, year), dtype=np.int32)
    authors = np.fromfile(_authors_path(scratch, year), dtype=np.int64)
    authorships = int(counts.sum(dtype=np.int64)) if counts.size else 0
    if authorships != int(authors.size):
        raise RuntimeError(f"Year {year} scratch counts do not match author ids")

    n_authors = int(author_ids.size)
    if counts.size == 0:
        matrix = sparse.csr_matrix((n_authors, n_authors), dtype=np.float64)
        _save_npz(_year_matrix_path(output_dir, year), matrix)
        return 0

    edge_bound = _directed_edge_bound(counts)
    if edge_bound > _INT32_MAX:
        raise ValueError(
            f"Year {year} would add up to {edge_bound} directed entries, "
            "which does not fit in SciPy int32 indices. Lower --max-authors."
        )
    if edge_bound > max_edges:
        raise ValueError(
            f"Year {year} would add up to {edge_bound} directed entries, "
            f"above --max-edges {max_edges}. Lower --max-authors."
        )

    columns = _author_columns(author_ids, authors, year)
    matrix = _coauthorship_matrix(columns, counts, n_authors)
    _save_npz(_year_matrix_path(output_dir, year), matrix)
    return int(matrix.nnz)


def _directed_edge_bound(counts: np.ndarray) -> int:
    counts64 = counts.astype(np.int64, copy=False)
    products = counts64 * (counts64 - 1)
    if np.any(products < 0):
        raise ValueError(
            "A paper's directed-entry count overflowed int64. Lower --max-authors."
        )
    total = int(products.sum(dtype=np.int64))
    if total < 0:
        raise ValueError(
            "A year's directed-entry count overflowed int64. Lower --max-authors."
        )
    return total


def _author_columns(author_ids: np.ndarray, authors: np.ndarray, year: int) -> np.ndarray:
    if authors.size == 0:
        return np.empty(0, dtype=np.int32)
    if author_ids.size == 0:
        raise RuntimeError(f"Year {year} has authors but the author index is empty")
    positions = np.searchsorted(author_ids, authors)
    in_range = positions < author_ids.size
    matched = np.zeros(authors.shape, dtype=bool)
    matched[in_range] = author_ids[positions[in_range]] == authors[in_range]
    if not np.all(matched):
        raise RuntimeError(f"Year {year} scratch contains author ids missing from the index")
    return positions.astype(np.int32, copy=False)


def _coauthorship_matrix(columns: np.ndarray, counts: np.ndarray, n_authors: int) -> sparse.csr_matrix:
    """Return A = B.T @ diag(alpha) @ B with the diagonal removed."""
    n_papers = int(counts.size)
    counts64 = counts.astype(np.int64, copy=False)
    indptr = np.zeros(n_papers + 1, dtype=np.int64)
    np.cumsum(counts64, out=indptr[1:])
    if int(indptr[-1]) > _INT32_MAX:
        raise ValueError(
            "A year's authorships do not fit in SciPy int32 indices. Lower --max-authors."
        )
    indptr32 = indptr.astype(np.int32)
    alpha = np.reciprocal(counts64.astype(np.float64) - 1.0)
    paper_index = np.repeat(np.arange(n_papers, dtype=np.int64), counts64)
    order = np.lexsort((columns, paper_index))
    sorted_columns = np.asarray(columns, dtype=np.int32)[order]
    weighted = sparse.csr_matrix(
        (np.repeat(alpha, counts64)[order], sorted_columns, indptr32),
        shape=(n_papers, n_authors),
        dtype=np.float64,
    )
    binary = sparse.csr_matrix(
        (np.ones(sorted_columns.size, dtype=np.float64), sorted_columns.copy(), indptr32.copy()),
        shape=(n_papers, n_authors),
        dtype=np.float64,
    )
    weighted.has_sorted_indices = True
    binary.has_sorted_indices = True
    product = (binary.T @ weighted).tocsr()
    product.sum_duplicates()
    symmetric = ((product + product.T) * 0.5).tocsr()
    symmetric.sum_duplicates()
    return _zero_diagonal(symmetric)


def _zero_diagonal(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    matrix = matrix.tocsr()
    if matrix.shape[0] != matrix.shape[1]:
        raise RuntimeError("Coauthorship matrix must be square")
    n = int(matrix.shape[0])
    if n == 0 or matrix.nnz == 0:
        return sparse.csr_matrix((n, n), dtype=np.float64)

    indptr = np.asarray(matrix.indptr)
    indices = np.asarray(matrix.indices)
    data = np.asarray(matrix.data, dtype=np.float64)
    counts = np.diff(indptr).astype(np.int64, copy=False)
    rows = np.repeat(np.arange(n, dtype=np.int32), counts)
    keep = indices != rows
    removals = np.bincount(rows[~keep].astype(np.int64, copy=False), minlength=n)
    kept_counts = counts - removals
    new_indptr = np.empty(n + 1, dtype=np.int64)
    new_indptr[0] = 0
    np.cumsum(kept_counts, out=new_indptr[1:])
    if int(new_indptr[-1]) > _INT32_MAX:
        raise ValueError(
            "Coauthorship nnz does not fit in SciPy int32 indices. Lower --max-authors."
        )
    result = sparse.csr_matrix(
        (
            data[keep],
            np.asarray(indices[keep], dtype=np.int32),
            new_indptr.astype(np.int32),
        ),
        shape=(n, n),
        dtype=np.float64,
    )
    result.eliminate_zeros()
    return result


def _fresh_manifest(
    config: dict,
    years: Sequence[int],
    author_ids: np.ndarray,
    scratch: Path,
) -> dict:
    papers_kept = {}
    solo_papers = {}
    hyperauthored_papers = {}
    for year in years:
        stats = _read_json(_stats_path(scratch, year))
        key = str(year)
        papers_kept[key] = int(stats["papers_kept"])
        solo_papers[key] = int(stats["solo_papers"])
        hyperauthored_papers[key] = int(stats["hyperauthored_papers"])
    return {
        "author_count": int(author_ids.size),
        "completed_years": [],
        "diagonal": 0,
        "dtype": "float64",
        "format": "csr",
        "from_year": config["from_year"],
        "hyperauthored_papers": hyperauthored_papers,
        "max_authors": config["max_authors"],
        "max_edges": config["max_edges"],
        "nnz": {},
        "papers_kept": papers_kept,
        "phase1_complete": True,
        "solo_papers": solo_papers,
        "symmetric": True,
        "to_year": config["to_year"],
        "weight": "newman",
        "weight_definition": WEIGHT_DEFINITION,
        "years": [int(year) for year in years],
    }


def _check_config(manifest: dict, config: dict) -> None:
    for key, value in config.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Existing output used {key}={manifest.get(key)!r}, not {value!r}. "
                "Use a new --output-dir or resume with the same parameters."
            )


def _check_index_limit(author_ids: np.ndarray) -> None:
    if int(author_ids.size) > _INT32_MAX:
        raise ValueError(
            f"Author index has {len(author_ids)} ids, which does not fit in SciPy int32 indices."
        )


def _is_complete(manifest: dict, output_dir: Path) -> bool:
    if not manifest.get("phase1_complete") or not _author_index_path(output_dir).exists():
        return False
    completed = {int(year) for year in manifest.get("completed_years", [])}
    years = [int(year) for year in manifest.get("years", [])]
    return all(year in completed and _year_matrix_path(output_dir, year).exists() for year in years)


def _output_exists(output_dir: Path) -> bool:
    if not output_dir.exists():
        return False
    markers = [
        output_dir / "manifest.json",
        output_dir / "author_ids.npy",
        output_dir / "scratch",
        output_dir / "years",
    ]
    for marker in markers:
        if not marker.exists():
            continue
        if marker.is_dir():
            if any(marker.iterdir()):
                return True
        else:
            return True
    return False


def _scratch_ready(scratch: Path, year: int) -> bool:
    return all(
        path.exists()
        for path in (
            _authors_path(scratch, year),
            _counts_path(scratch, year),
            _unique_path(scratch, year),
            _stats_path(scratch, year),
        )
    )


def _author_index_path(output_dir: Path) -> Path:
    return output_dir / "author_ids.npy"


def _year_matrix_path(output_dir: Path, year: int) -> Path:
    return output_dir / "years" / f"{year}.npz"


def _authors_path(scratch: Path, year: int) -> Path:
    return scratch / f"{year}.authors.i64"


def _counts_path(scratch: Path, year: int) -> Path:
    return scratch / f"{year}.counts.i32"


def _unique_path(scratch: Path, year: int) -> Path:
    return scratch / f"{year}.unique.i64"


def _stats_path(scratch: Path, year: int) -> Path:
    return scratch / f"{year}.stats.json"


def _save_npz(path: Path, matrix: sparse.csr_matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    sparse.save_npz(temporary, matrix, compressed=True)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    raise SystemExit(main())
