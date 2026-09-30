"""First coauthorship links, prior-network distances, and event attribution.

The corpus is streamed by publication year. Every unordered author pair enters
the cumulative graph in its first observed year. Output retains weighted
yearly random samples of no-cluster links and links in each attributed cluster,
together with hop distance in the graph through the preceding year. A link is
attributed to one event cluster only when that cluster occurs on every paper
creating the link in its first year and no second cluster does too.
"""

from __future__ import annotations

import argparse
import hashlib
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
from numba import njit
from scipy import sparse

from openalex.website.build import load_event_artifacts

logger = logging.getLogger(__name__)

DEFAULT_MAX_AUTHORS = 16
DEFAULT_MAX_EDGES = 200_000_000
DEFAULT_MAX_EVENT_PAIRS = 100_000_000
DEFAULT_GROUPED_BFS_MIN_TARGETS = 4
DEFAULT_NO_CLUSTER_SAMPLE = 10_000
DEFAULT_CLUSTER_SAMPLE = 10_000
DEFAULT_SAMPLING_SEED = 0
DEFAULT_CACHE_MB = 2048
_FETCH_SIZE = 200_000
_AUTHOR_FLUSH = 100_000
_INT32_MAX = int(np.iinfo(np.int32).max)
NO_CLUSTER = np.int32(-1)
DISCONNECTED = np.int32(-1)

_YEAR_QUERY = """
SELECT aa.article_id, aa.author_id
FROM articles a
JOIN articles_authors aa ON aa.article_id = a.article_id
WHERE a.publication_year = ?
ORDER BY aa.article_id
"""


def build_new_links(
    db_path: str | Path,
    events_dir: str | Path,
    clusters_dir: str | Path,
    output_dir: str | Path,
    *,
    to_year: int | None = None,
    max_authors: int = DEFAULT_MAX_AUTHORS,
    max_edges: int = DEFAULT_MAX_EDGES,
    max_event_pairs: int = DEFAULT_MAX_EVENT_PAIRS,
    grouped_bfs_min_targets: int = DEFAULT_GROUPED_BFS_MIN_TARGETS,
    no_cluster_sample: int = DEFAULT_NO_CLUSTER_SAMPLE,
    cluster_sample: int = DEFAULT_CLUSTER_SAMPLE,
    sampling_seed: int = DEFAULT_SAMPLING_SEED,
    sqlite_cache_mb: int = DEFAULT_CACHE_MB,
    resume: bool = False,
    fetch_size: int = _FETCH_SIZE,
) -> None:
    """Build year-partitioned first-link records from a read-only corpus."""
    _validate_options(
        max_authors,
        max_edges,
        max_event_pairs,
        grouped_bfs_min_targets,
        no_cluster_sample,
        cluster_sample,
        sampling_seed,
        fetch_size,
    )
    database = Path(db_path)
    events = Path(events_dir)
    clusters = Path(clusters_dir)
    output = Path(output_dir)
    scratch = output / "scratch"
    years_dir = output / "years"
    manifest_path = output / "manifest.json"

    existing = _load_manifest(manifest_path) if resume and manifest_path.exists() else None
    if existing is not None and int(existing.get("artifact_version", 0)) != 6:
        raise ValueError(
            "Existing output predates global per-cluster distance reservoirs. "
            "Use a new --output-dir."
        )
    if existing is not None and existing.get("distance_mode") != "exact":
        raise ValueError(
            "Existing output does not contain exact distances. Use a new --output-dir."
        )
    if existing is not None and _is_complete(existing, output):
        shutil.rmtree(scratch, ignore_errors=True)
        logger.info("New-link artifacts already built in %s", output)
        return
    if existing is None and not resume and _output_exists(output):
        raise FileExistsError(
            f"{output} already contains new-link files. Use --resume or a new --output-dir."
        )

    config = {
        "clusters_manifest_sha256": _sha256(clusters / "manifest.json"),
        "events_manifest_sha256": _sha256(events / "manifest.json"),
        "distance_mode": "exact",
        "grouped_bfs_min_targets": grouped_bfs_min_targets,
        "max_authors": max_authors,
        "max_edges": max_edges,
        "max_event_pairs": max_event_pairs,
        "no_cluster_sample": no_cluster_sample,
        "cluster_sample": cluster_sample,
        "sampling_seed": sampling_seed,
        "to_year": to_year,
    }
    if existing is not None:
        _check_config(existing, config)
    _require_event_article_ids(events)

    output.mkdir(parents=True, exist_ok=True)
    years_dir.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)

    if existing is not None and existing.get("phase1_complete") and _author_index_path(output).exists():
        years = [int(year) for year in existing["years"]]
        author_ids = np.load(_author_index_path(output), allow_pickle=False)
        manifest = existing
    else:
        with _connect(database, scratch, sqlite_cache_mb) as connection:
            years = _list_years(connection, to_year)
            for year in years:
                if _scratch_ready(scratch, year):
                    logger.info("Reusing year %s scratch", year)
                    continue
                _scan_year(connection, year, scratch, max_authors, fetch_size)
        author_ids = _write_author_index(scratch, years, output)
        manifest = _fresh_manifest(config, years, author_ids, scratch)
        _atomic_json(manifest_path, manifest)

    _check_index_limit(author_ids)
    (
        event_ids,
        event_offsets,
        event_clusters,
        event_years,
        cluster_labels,
        cluster_level,
    ) = _load_event_paper_clusters(events, clusters)
    _save_cluster_metadata(
        output / "cluster_metadata.npz",
        event_offsets,
        event_clusters,
        event_years,
        cluster_labels,
        years,
    )
    if manifest.get("cluster_level") is None:
        manifest["cluster_level"] = cluster_level
        _atomic_json(manifest_path, manifest)
    elif int(manifest["cluster_level"]) != cluster_level:
        raise ValueError("Cluster level changed since this output was created")

    completed = {int(year) for year in manifest.get("completed_years", [])}
    _check_completed_prefix(years, completed)
    graph, parent, size, seen_keys = _restore_prior_graph(
        scratch, years, completed, int(author_ids.size)
    )
    cluster_reservoirs, cluster_observation_population = _restore_cluster_reservoir(
        scratch,
        years,
        completed,
        int(cluster_labels.size),
        cluster_sample,
    )

    for year in years:
        if year in completed:
            logger.info("Skipping finished year %s", year)
            continue
        started = time.perf_counter()
        (
            arrays,
            distance_stats,
            sampling_stats,
            cluster_observations,
            cluster_reservoirs,
            cluster_observation_population,
            all_left,
            all_right,
        ) = _process_year(
            scratch,
            year,
            author_ids,
            event_ids,
            event_offsets,
            event_clusters,
            graph,
            parent,
            max_edges,
            max_event_pairs,
            grouped_bfs_min_targets,
            no_cluster_sample,
            cluster_sample,
            sampling_seed,
            int(cluster_labels.size),
            cluster_reservoirs,
            cluster_observation_population,
            seen_keys,
        )
        _save_year(_year_path(output, year), arrays)
        _save_cluster_year(
            _cluster_year_path(output, year), cluster_observations
        )
        _save_edges(_edge_path(scratch, year), all_left, all_right)
        _save_cluster_reservoir(
            _reservoir_path(scratch, year),
            cluster_reservoirs,
            cluster_observation_population,
        )

        year_keys = _encode_pairs(all_left, all_right)
        _union_edges(parent, size, all_left, all_right)
        graph = _add_graph_edges(graph, all_left, all_right, int(author_ids.size))
        seen_keys = _merge_sorted_unique(seen_keys, year_keys)

        completed.add(year)
        manifest["completed_years"] = sorted(completed)
        manifest["link_counts"][str(year)] = int(all_left.size)
        manifest["stored_link_counts"][str(year)] = int(arrays["author_i"].size)
        manifest["clustered_link_counts"][str(year)] = sampling_stats[
            "clustered_population"
        ]
        manifest["sampling_stats"][str(year)] = sampling_stats
        manifest["distance_stats"][str(year)] = distance_stats
        _atomic_json(manifest_path, manifest)
        logger.info(
            "Year %s: new_links=%s stored=%s clustered=%s distance=%.1fs searches=%s+%s "
            "visited=%s inspected_edges=%s elapsed=%.1fs",
            year,
            all_left.size,
            arrays["author_i"].size,
            manifest["clustered_link_counts"][str(year)],
            distance_stats["seconds"],
            distance_stats["bidirectional_searches"],
            distance_stats["grouped_searches"],
            distance_stats["visited_nodes"],
            distance_stats["inspected_edges"],
            time.perf_counter() - started,
        )

    _save_cluster_reservoir(
        output / "cluster_distance_reservoir.npz",
        cluster_reservoirs,
        cluster_observation_population,
    )
    shutil.rmtree(scratch, ignore_errors=True)
    logger.info("Wrote first-link records for %s years to %s", len(years), output)


def load_new_links(
    output_dir: str | Path, year: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load the shared author index and one year of first-link columns."""
    output = Path(output_dir)
    author_ids = np.load(_author_index_path(output), allow_pickle=False)
    with np.load(_year_path(output, year), allow_pickle=False) as payload:
        arrays = {
            name: payload[name]
            for name in (
                "author_i",
                "author_j",
                "distance",
                "cluster_id",
                "sampling_weight",
            )
        }
    return author_ids, arrays


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build first coauthorship links with prior distances and event clusters."
    )
    parser.add_argument("--db-path", default="articles.db", help="Read-only source corpus.")
    parser.add_argument("--events-dir", required=True)
    parser.add_argument("--clusters-dir", required=True)
    parser.add_argument("--output-dir", default="output/new_links")
    parser.add_argument("--to-year", type=int, default=None)
    parser.add_argument("--max-authors", type=int, default=DEFAULT_MAX_AUTHORS)
    parser.add_argument("--max-edges", type=int, default=DEFAULT_MAX_EDGES)
    parser.add_argument("--max-event-pairs", type=int, default=DEFAULT_MAX_EVENT_PAIRS)
    parser.add_argument(
        "--grouped-bfs-min-targets",
        type=int,
        default=DEFAULT_GROUPED_BFS_MIN_TARGETS,
        help="Use grouped BFS at this many targets per source; use bidirectional BFS below it.",
    )
    parser.add_argument(
        "--no-cluster-sample",
        type=int,
        default=DEFAULT_NO_CLUSTER_SAMPLE,
        help="Uniform no-cluster links retained per year.",
    )
    parser.add_argument(
        "--cluster-sample",
        type=int,
        default=DEFAULT_CLUSTER_SAMPLE,
        help="Uniform distance samples retained independently per cluster and year.",
    )
    parser.add_argument("--sampling-seed", type=int, default=DEFAULT_SAMPLING_SEED)
    parser.add_argument("--sqlite-cache-mb", type=int, default=DEFAULT_CACHE_MB)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args(argv)
    build_new_links(
        args.db_path,
        args.events_dir,
        args.clusters_dir,
        args.output_dir,
        to_year=args.to_year,
        max_authors=args.max_authors,
        max_edges=args.max_edges,
        max_event_pairs=args.max_event_pairs,
        grouped_bfs_min_targets=args.grouped_bfs_min_targets,
        no_cluster_sample=args.no_cluster_sample,
        cluster_sample=args.cluster_sample,
        sampling_seed=args.sampling_seed,
        sqlite_cache_mb=args.sqlite_cache_mb,
        resume=args.resume,
    )
    return 0


def _validate_options(
    max_authors: int,
    max_edges: int,
    max_event_pairs: int,
    grouped_bfs_min_targets: int,
    no_cluster_sample: int,
    cluster_sample: int,
    sampling_seed: int,
    fetch_size: int,
) -> None:
    if max_authors < 0:
        raise ValueError("--max-authors must be >= 0")
    if max_edges < 1:
        raise ValueError("--max-edges must be >= 1")
    if max_event_pairs < 1:
        raise ValueError("--max-event-pairs must be >= 1")
    if grouped_bfs_min_targets < 2:
        raise ValueError("--grouped-bfs-min-targets must be >= 2")
    if no_cluster_sample < 0:
        raise ValueError("--no-cluster-sample must be >= 0")
    if cluster_sample < 1:
        raise ValueError("--cluster-sample must be >= 1")
    if sampling_seed < 0:
        raise ValueError("--sampling-seed must be >= 0")
    if fetch_size < 1:
        raise ValueError("fetch_size must be >= 1")


@contextmanager
def _connect(db_path: Path, scratch: Path, cache_mb: int) -> Iterator[sqlite3.Connection]:
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
    connection: sqlite3.Connection, to_year: int | None
) -> list[int]:
    clauses = ["publication_year IS NOT NULL"]
    parameters: list[int] = []
    if to_year is not None:
        clauses.append("publication_year <= ?")
        parameters.append(to_year)
    rows = connection.execute(
        f"SELECT DISTINCT publication_year FROM articles WHERE {' AND '.join(clauses)} "
        "ORDER BY publication_year",
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
    for path in (
        _articles_path(scratch, year),
        _authors_path(scratch, year),
        _counts_path(scratch, year),
        _unique_path(scratch, year),
        _stats_path(scratch, year),
    ):
        path.unlink(missing_ok=True)

    article_buffer: list[int] = []
    author_buffer: list[int] = []
    count_buffer: list[int] = []
    papers_kept = 0
    solo_papers = 0
    hyperauthored_papers = 0
    edge_bound = 0
    cursor = connection.execute(_YEAR_QUERY, (year,))
    with (
        _articles_path(scratch, year).open("wb") as article_handle,
        _authors_path(scratch, year).open("wb") as author_handle,
        _counts_path(scratch, year).open("wb") as count_handle,
    ):
        for article_id, authors in _iter_papers(cursor, fetch_size):
            unique_authors = list(dict.fromkeys(authors))
            count = len(unique_authors)
            if count < 2:
                solo_papers += 1
            elif max_authors > 0 and count > max_authors:
                hyperauthored_papers += 1
            else:
                article_buffer.append(article_id)
                author_buffer.extend(unique_authors)
                count_buffer.append(count)
                papers_kept += 1
                edge_bound += count * (count - 1)
                if len(author_buffer) >= _AUTHOR_FLUSH:
                    _flush_scan_buffers(
                        article_handle,
                        author_handle,
                        count_handle,
                        article_buffer,
                        author_buffer,
                        count_buffer,
                    )
        _flush_scan_buffers(
            article_handle,
            author_handle,
            count_handle,
            article_buffer,
            author_buffer,
            count_buffer,
        )
        for handle in (article_handle, author_handle, count_handle):
            handle.flush()
            os.fsync(handle.fileno())

    authors = np.fromfile(_authors_path(scratch, year), dtype=np.int64)
    unique = np.unique(authors) if authors.size else np.empty(0, dtype=np.int64)
    temporary = _unique_path(scratch, year).with_suffix(".i64.tmp")
    unique.tofile(temporary)
    os.replace(temporary, _unique_path(scratch, year))
    _atomic_json(
        _stats_path(scratch, year),
        {
            "edge_bound": edge_bound,
            "hyperauthored_papers": hyperauthored_papers,
            "papers_kept": papers_kept,
            "solo_papers": solo_papers,
        },
    )


def _iter_papers(
    cursor: sqlite3.Cursor, fetch_size: int
) -> Iterator[tuple[int, list[int]]]:
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
            if current_id is None:
                current_id = article_id
                current_authors = [int(author_id)]
            elif article_id == current_id:
                current_authors.append(int(author_id))
            else:
                yield current_id, current_authors
                current_id = article_id
                current_authors = [int(author_id)]
    if current_id is not None:
        yield current_id, current_authors


def _flush_scan_buffers(
    article_handle,
    author_handle,
    count_handle,
    article_buffer: list[int],
    author_buffer: list[int],
    count_buffer: list[int],
) -> None:
    if article_buffer:
        np.asarray(article_buffer, dtype=np.int64).tofile(article_handle)
        article_buffer.clear()
    if author_buffer:
        np.asarray(author_buffer, dtype=np.int64).tofile(author_handle)
        author_buffer.clear()
    if count_buffer:
        np.asarray(count_buffer, dtype=np.int32).tofile(count_handle)
        count_buffer.clear()


def _write_author_index(scratch: Path, years: Sequence[int], output: Path) -> np.ndarray:
    parts = [np.fromfile(_unique_path(scratch, year), dtype=np.int64) for year in years]
    author_ids = np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)
    _check_index_limit(author_ids)
    temporary = output / ".author_ids.tmp.npy"
    np.save(temporary, author_ids, allow_pickle=False)
    os.replace(temporary, _author_index_path(output))
    return author_ids


def _load_event_paper_clusters(
    events_dir: Path, clusters_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    artifacts = load_event_artifacts(events_dir)
    cluster_manifest = _read_json(clusters_dir / "manifest.json")
    level = int(cluster_manifest["coarse_level"])
    keywords = np.load(clusters_dir / cluster_manifest["keywords"], allow_pickle=False).astype(str)
    groups_by_level = np.load(
        clusters_dir / cluster_manifest["groups_by_level"], allow_pickle=False
    )
    if level < 0 or level >= groups_by_level.shape[0]:
        raise ValueError("Cluster manifest coarse_level is outside groups_by_level")
    if groups_by_level.shape[1] != keywords.size:
        raise ValueError("Cluster keywords and group assignments do not align")

    vocabulary = np.asarray(artifacts["vocabulary"]).astype(str)
    vocabulary_lookup = {keyword: index for index, keyword in enumerate(vocabulary)}
    try:
        selected = np.asarray([vocabulary_lookup[keyword] for keyword in keywords], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Cluster keyword {exc.args[0]!r} is absent from event vocabulary") from exc
    groups = np.asarray(groups_by_level[level], dtype=np.int32)
    group_count = int(groups.max()) + 1 if groups.size else 0
    selected_frequencies = np.asarray(artifacts["frequencies"])[selected]
    cluster_labels = np.asarray(
        [
            str(
                keywords[
                    members[
                        np.argmax(selected_frequencies[members])
                    ]
                ]
            )
            for group in range(group_count)
            for members in [np.flatnonzero(groups == group)]
        ],
        dtype=np.str_,
    )
    membership = sparse.csr_matrix(
        (
            np.ones(selected.size, dtype=np.int32),
            (selected, groups),
        ),
        shape=(vocabulary.size, group_count),
        dtype=np.int32,
    )

    paper_parts: list[np.ndarray] = []
    cluster_parts: list[np.ndarray] = []
    year_parts: list[np.ndarray] = []
    root = Path(artifacts["root"])
    incidence_dir = root / artifacts["manifest"]["incidence_dir"]
    for relative in artifacts["manifest"].get("incidence_parts", []):
        matrix_path = incidence_dir / relative
        ids_path = incidence_dir / relative.replace(".npz", "_article_ids.npy")
        years_path = incidence_dir / relative.replace(".npz", "_years.npy")
        if not ids_path.exists():
            raise ValueError(
                f"Event incidence shard {relative} has no article-id sidecar. "
                "Regenerate event artifacts with the current openalex events command."
            )
        article_terms = sparse.load_npz(matrix_path).tocsr()
        article_ids = np.load(ids_path, allow_pickle=False).astype(np.int64, copy=False)
        article_years = np.load(years_path, allow_pickle=False).astype(np.int32, copy=False)
        if article_terms.shape[0] != article_ids.size or article_terms.shape[0] != article_years.size:
            raise ValueError(
                f"Incidence rows, article ids, and years do not match for {relative}"
            )
        if article_terms.shape[1] != vocabulary.size:
            raise ValueError(f"Incidence width does not match the vocabulary for {relative}")
        presence = (article_terms.astype(np.int32) @ membership).tocoo()
        if presence.nnz:
            paper_parts.append(article_ids[presence.row])
            cluster_parts.append(presence.col.astype(np.int32, copy=False))
            year_parts.append(article_years[presence.row])

    if not paper_parts:
        return (
            np.empty(0, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            cluster_labels,
            level,
        )
    paper_rows = np.concatenate(paper_parts)
    cluster_rows = np.concatenate(cluster_parts)
    year_rows = np.concatenate(year_parts)
    order = np.lexsort((cluster_rows, paper_rows))
    paper_rows = paper_rows[order]
    cluster_rows = cluster_rows[order]
    year_rows = year_rows[order]
    keep = np.ones(paper_rows.size, dtype=bool)
    keep[1:] = (paper_rows[1:] != paper_rows[:-1]) | (
        cluster_rows[1:] != cluster_rows[:-1]
    )
    paper_rows = paper_rows[keep]
    cluster_rows = cluster_rows[keep]
    year_rows = year_rows[keep]
    event_ids, counts = np.unique(paper_rows, return_counts=True)
    offsets = np.empty(event_ids.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts.astype(np.int64, copy=False), out=offsets[1:])
    event_years = year_rows[offsets[:-1]]
    if not np.array_equal(np.repeat(event_years, counts), year_rows):
        raise ValueError("One event paper is associated with multiple publication years")
    return event_ids, offsets, cluster_rows, event_years, cluster_labels, level


def _require_event_article_ids(events_dir: Path) -> None:
    manifest = _read_json(events_dir / "manifest.json")
    incidence_dir = events_dir / manifest["incidence_dir"]
    for relative in manifest.get("incidence_parts", []):
        ids_path = incidence_dir / relative.replace(".npz", "_article_ids.npy")
        if not ids_path.exists():
            raise ValueError(
                f"Event incidence shard {relative} has no article-id sidecar. "
                "Regenerate event artifacts with the current openalex events command."
            )


def _process_year(
    scratch: Path,
    year: int,
    author_ids: np.ndarray,
    event_ids: np.ndarray,
    event_offsets: np.ndarray,
    event_clusters: np.ndarray,
    graph: sparse.csr_matrix,
    parent: np.ndarray,
    max_edges: int,
    max_event_pairs: int,
    grouped_bfs_min_targets: int,
    no_cluster_sample: int,
    cluster_sample: int,
    sampling_seed: int,
    cluster_count: int,
    cluster_reservoirs: list[np.ndarray],
    cluster_observation_population: np.ndarray,
    seen_keys: np.ndarray,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, int | float],
    dict[str, int | float | None],
    dict[str, np.ndarray],
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    counts = np.fromfile(_counts_path(scratch, year), dtype=np.int32)
    authors = np.fromfile(_authors_path(scratch, year), dtype=np.int64)
    article_ids = np.fromfile(_articles_path(scratch, year), dtype=np.int64)
    if counts.size != article_ids.size or int(counts.sum(dtype=np.int64)) != authors.size:
        raise RuntimeError(f"Year {year} scratch arrays do not align")
    edge_bound = int(_read_json(_stats_path(scratch, year))["edge_bound"])
    if edge_bound > max_edges:
        raise ValueError(
            f"Year {year} would add up to {edge_bound} directed pair occurrences, "
            f"above --max-edges {max_edges}. Lower --max-authors."
        )
    columns = _author_columns(author_ids, authors, year)
    pair_occurrences = _make_pair_keys(columns, counts)
    pair_keys, paper_counts = np.unique(pair_occurrences, return_counts=True)
    del pair_occurrences

    prior_positions = np.searchsorted(seen_keys, pair_keys)
    known = prior_positions < seen_keys.size
    known[known] &= seen_keys[prior_positions[known]] == pair_keys[known]
    new_keys = pair_keys[~known]
    new_paper_counts = paper_counts[~known].astype(np.int64, copy=False)
    del pair_keys, paper_counts, prior_positions, known

    positions = np.searchsorted(event_ids, article_ids)
    matches = positions < event_ids.size
    matches[matches] &= event_ids[positions[matches]] == article_ids[matches]
    event_positions = np.full(article_ids.size, -1, dtype=np.int64)
    event_positions[matches] = positions[matches]
    event_record_count = _event_record_bound(counts, event_positions, event_offsets)
    if event_record_count > max_event_pairs:
        raise ValueError(
            f"Year {year} would add {event_record_count} pair-event occurrences, "
            f"above --max-event-pairs {max_event_pairs}."
        )
    event_pair_keys, event_pair_clusters = _make_event_pair_records(
        columns,
        counts,
        event_positions,
        event_offsets,
        event_clusters,
        event_record_count,
    )

    cluster_id = _attribute_clusters(
        new_keys, new_paper_counts, event_pair_keys, event_pair_clusters
    )
    all_left, all_right = _decode_pairs(new_keys)
    new_connected = _connected_pairs(parent, all_left, all_right)
    retained, sampling_weight, sampling_stats = _sample_links(
        cluster_id,
        no_cluster_sample,
        cluster_sample,
        sampling_seed,
        year,
    )
    retained_keys = new_keys[retained]
    (
        cluster_observations,
        sampled_event_keys,
        sampled_event_clusters,
        kept_reservoirs,
        cluster_observation_population,
    ) = _sample_cluster_observations(
        event_pair_keys,
        event_pair_clusters,
        seen_keys,
        new_keys,
        new_connected,
        cluster_count,
        cluster_sample,
        sampling_seed,
        year,
        cluster_reservoirs,
        cluster_observation_population,
    )
    attributed = cluster_id >= 0
    cluster_observations["attributed_new_link_population"] = np.bincount(
        cluster_id[attributed], minlength=cluster_count
    ).astype(np.int64)
    cluster_observations["attributed_new_link_connected"] = np.bincount(
        cluster_id[attributed & new_connected], minlength=cluster_count
    ).astype(np.int64)
    cluster_observations["attributed_new_link_disconnected"] = np.bincount(
        cluster_id[attributed & ~new_connected], minlength=cluster_count
    ).astype(np.int64)

    distance_keys = np.union1d(retained_keys, np.unique(sampled_event_keys))
    distance_left, distance_right = _decode_pairs(distance_keys)
    all_distances, distance_stats = _distances(
        graph,
        parent,
        distance_left,
        distance_right,
        grouped_bfs_min_targets,
    )
    left = all_left[retained]
    right = all_right[retained]
    retained_positions = np.searchsorted(distance_keys, retained_keys)
    distance = all_distances[retained_positions]
    cluster_reservoirs = _finish_cluster_reservoir(
        kept_reservoirs,
        sampled_event_keys,
        sampled_event_clusters,
        distance_keys,
        all_distances,
        cluster_count,
    )
    return (
        {
            "author_i": left,
            "author_j": right,
            "cluster_id": cluster_id[retained],
            "distance": distance,
            "sampling_weight": sampling_weight,
        },
        distance_stats,
        sampling_stats,
        cluster_observations,
        cluster_reservoirs,
        cluster_observation_population,
        all_left,
        all_right,
    )


def _sample_links(
    cluster_id: np.ndarray,
    no_cluster_sample: int,
    cluster_sample: int,
    sampling_seed: int,
    year: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    no_cluster = np.flatnonzero(cluster_id == NO_CLUSTER)
    population = int(no_cluster.size)
    sample_size = min(no_cluster_sample, population)
    sampled_no_cluster = _uniform_sample(
        no_cluster,
        sample_size,
        [sampling_seed, int(year), 0],
    )
    selected_parts = [sampled_no_cluster]
    no_cluster_weight = population / sample_size if sample_size else 0.0
    weight_parts = [
        np.full(sampled_no_cluster.size, no_cluster_weight, dtype=np.float64)
    ]
    if sample_size:
        inclusion_probability = sample_size / population
        manifest_weight: float | None = no_cluster_weight
    elif population:
        inclusion_probability = 0.0
        manifest_weight = None
    else:
        inclusion_probability = 1.0
        manifest_weight = 1.0

    cluster_stats: dict[str, dict[str, int | float]] = {}
    clustered_population = 0
    for cluster in np.unique(cluster_id[cluster_id >= 0]):
        members = np.flatnonzero(cluster_id == cluster)
        cluster_population = int(members.size)
        cluster_sample_size = min(cluster_sample, cluster_population)
        sampled = _uniform_sample(
            members,
            cluster_sample_size,
            [sampling_seed, int(year), 1, int(cluster)],
        )
        weight = cluster_population / cluster_sample_size
        selected_parts.append(sampled)
        weight_parts.append(np.full(sampled.size, weight, dtype=np.float64))
        clustered_population += cluster_population
        cluster_stats[str(int(cluster))] = {
            "inclusion_probability": cluster_sample_size / cluster_population,
            "population": cluster_population,
            "sample": cluster_sample_size,
            "sampling_weight": weight,
        }

    retained = np.concatenate(selected_parts)
    weights = np.concatenate(weight_parts)
    order = np.argsort(retained, kind="stable")
    retained = retained[order]
    weights = weights[order]
    stats: dict[str, object] = {
        "cluster_strata": cluster_stats,
        "clustered_population": clustered_population,
        "no_cluster_inclusion_probability": inclusion_probability,
        "no_cluster_population": population,
        "no_cluster_sample": sample_size,
        "no_cluster_sampling_weight": manifest_weight,
        "stored_links": int(retained.size),
    }
    return retained, weights, stats


def _uniform_sample(
    indices: np.ndarray, sample_size: int, seed_components: Sequence[int]
) -> np.ndarray:
    if sample_size == indices.size:
        return indices
    if sample_size == 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(np.random.SeedSequence(list(seed_components)))
    positions = rng.choice(
        indices.size, size=sample_size, replace=False, shuffle=False
    )
    return indices[positions]


def _author_columns(author_ids: np.ndarray, authors: np.ndarray, year: int) -> np.ndarray:
    positions = np.searchsorted(author_ids, authors)
    valid = positions < author_ids.size
    matched = np.zeros(authors.size, dtype=bool)
    matched[valid] = author_ids[positions[valid]] == authors[valid]
    if not np.all(matched):
        raise RuntimeError(f"Year {year} scratch contains author ids missing from the index")
    return positions.astype(np.int32, copy=False)


@njit(cache=True)
def _make_pair_keys(columns: np.ndarray, counts: np.ndarray) -> np.ndarray:
    total = 0
    for count in counts:
        total += int(count) * (int(count) - 1) // 2
    result = np.empty(total, dtype=np.uint64)
    offset = 0
    output = 0
    for count_value in counts:
        count = int(count_value)
        for first in range(count - 1):
            a = columns[offset + first]
            for second in range(first + 1, count):
                b = columns[offset + second]
                low = a if a < b else b
                high = b if a < b else a
                result[output] = (np.uint64(low) << np.uint64(32)) | np.uint64(high)
                output += 1
        offset += count
    return result


@njit(cache=True)
def _event_record_bound(
    counts: np.ndarray,
    event_positions: np.ndarray,
    event_offsets: np.ndarray,
) -> int:
    total = 0
    for paper in range(counts.size):
        position = event_positions[paper]
        if position >= 0:
            clusters = event_offsets[position + 1] - event_offsets[position]
            count = int(counts[paper])
            total += count * (count - 1) // 2 * clusters
    return total


@njit(cache=True)
def _make_event_pair_records(
    columns: np.ndarray,
    counts: np.ndarray,
    event_positions: np.ndarray,
    event_offsets: np.ndarray,
    event_clusters: np.ndarray,
    total: int,
) -> tuple[np.ndarray, np.ndarray]:
    keys = np.empty(total, dtype=np.uint64)
    clusters = np.empty(total, dtype=np.int32)
    offset = 0
    output = 0
    for paper in range(counts.size):
        count = int(counts[paper])
        position = event_positions[paper]
        if position >= 0:
            for cluster_offset in range(
                event_offsets[position], event_offsets[position + 1]
            ):
                cluster = event_clusters[cluster_offset]
                for first in range(count - 1):
                    a = columns[offset + first]
                    for second in range(first + 1, count):
                        b = columns[offset + second]
                        low = a if a < b else b
                        high = b if a < b else a
                        key = (
                            np.uint64(low) << np.uint64(32)
                        ) | np.uint64(high)
                        keys[output] = key
                        clusters[output] = cluster
                        output += 1
        offset += count
    return keys, clusters


def _attribute_clusters(
    new_keys: np.ndarray,
    paper_counts: np.ndarray,
    event_keys: np.ndarray,
    event_clusters: np.ndarray,
) -> np.ndarray:
    result = np.full(new_keys.size, NO_CLUSTER, dtype=np.int32)
    if new_keys.size == 0 or event_keys.size == 0:
        return result
    order = np.lexsort((event_clusters, event_keys))
    keys = event_keys[order]
    clusters = event_clusters[order]
    boundary = np.ones(keys.size, dtype=bool)
    boundary[1:] = (keys[1:] != keys[:-1]) | (clusters[1:] != clusters[:-1])
    starts = np.flatnonzero(boundary)
    combo_keys = keys[starts]
    combo_clusters = clusters[starts]
    combo_counts = np.diff(np.append(starts, keys.size)).astype(np.int64, copy=False)

    positions = np.searchsorted(new_keys, combo_keys)
    matched = positions < new_keys.size
    matched[matched] &= new_keys[positions[matched]] == combo_keys[matched]
    qualifies = matched.copy()
    qualifies[matched] &= combo_counts[matched] == paper_counts[positions[matched]]
    positions = positions[qualifies]
    candidates = combo_clusters[qualifies]
    if positions.size == 0:
        return result
    unique_positions, counts = np.unique(positions, return_counts=True)
    candidate_starts = np.empty(counts.size, dtype=np.int64)
    candidate_starts[0] = 0
    if counts.size > 1:
        np.cumsum(counts[:-1], out=candidate_starts[1:])
    single = counts == 1
    result[unique_positions[single]] = candidates[candidate_starts[single]]
    return result


def _sample_cluster_observations(
    event_keys: np.ndarray,
    event_clusters: np.ndarray,
    seen_keys: np.ndarray,
    new_keys: np.ndarray,
    new_connected: np.ndarray,
    cluster_count: int,
    cluster_sample: int,
    sampling_seed: int,
    year: int,
    cluster_reservoirs: list[np.ndarray],
    cluster_observation_population: np.ndarray,
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    np.ndarray,
]:
    totals = np.bincount(event_clusters, minlength=cluster_count).astype(np.int64)
    prior_positions = np.searchsorted(seen_keys, event_keys)
    known = prior_positions < seen_keys.size
    known[known] &= seen_keys[prior_positions[known]] == event_keys[known]
    new_positions = np.searchsorted(new_keys, event_keys)
    is_new = new_positions < new_keys.size
    is_new[is_new] &= new_keys[new_positions[is_new]] == event_keys[is_new]
    if not np.all(known | is_new):
        raise RuntimeError("Event-paper pair is absent from both prior and new edges")
    record_connected = np.zeros(event_keys.size, dtype=bool)
    record_connected[is_new] = new_connected[new_positions[is_new]]
    new_connected_records = is_new & record_connected
    new_disconnected_records = is_new & ~record_connected
    existing = np.bincount(
        event_clusters[known], minlength=cluster_count
    ).astype(np.int64)
    connected_new = np.bincount(
        event_clusters[new_connected_records], minlength=cluster_count
    ).astype(np.int64)
    disconnected = np.bincount(
        event_clusters[new_disconnected_records], minlength=cluster_count
    ).astype(np.int64)

    sampled_parts: list[np.ndarray] = []
    sampled_cluster_parts: list[np.ndarray] = []
    kept_reservoirs: list[np.ndarray] = []
    accepted_counts = np.zeros(cluster_count, dtype=np.int64)
    updated_population = (
        cluster_observation_population.astype(np.int64, copy=True)
        + connected_new
    )
    for cluster in range(cluster_count):
        candidates = np.flatnonzero(
            new_connected_records & (event_clusters == cluster)
        )
        previous_population = int(cluster_observation_population[cluster])
        new_population = int(candidates.size)
        total_population = previous_population + new_population
        target_size = min(cluster_sample, total_population)
        if total_population <= cluster_sample:
            accepted_new = new_population
        else:
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [sampling_seed, int(year), 2, cluster]
                )
            )
            accepted_new = int(
                rng.hypergeometric(
                    ngood=new_population,
                    nbad=previous_population,
                    nsample=target_size,
                )
            )
        kept_old = target_size - accepted_new
        old_indices = _uniform_sample(
            np.arange(cluster_reservoirs[cluster].size, dtype=np.int64),
            kept_old,
            [sampling_seed, int(year), 2, cluster, 0],
        )
        accepted = _uniform_sample(
            candidates,
            accepted_new,
            [sampling_seed, int(year), 2, cluster, 1],
        )
        kept_reservoirs.append(cluster_reservoirs[cluster][old_indices])
        sampled_parts.append(accepted)
        sampled_cluster_parts.append(
            np.full(accepted_new, cluster, dtype=np.int32)
        )
        accepted_counts[cluster] = accepted_new

    if sampled_parts:
        sampled_indices = np.concatenate(sampled_parts)
        sampled_clusters = np.concatenate(sampled_cluster_parts)
    else:
        sampled_indices = np.empty(0, dtype=np.int64)
        sampled_clusters = np.empty(0, dtype=np.int32)
    observations = {
        "connected_pair_observations": existing + connected_new,
        "disconnected_pair_observations": disconnected,
        "existing_pair_observations": existing,
        "new_connected_pair_observations": connected_new,
        "reservoir_new_acceptances": accepted_counts,
        "total_pair_observations": totals,
    }
    return (
        observations,
        event_keys[sampled_indices],
        sampled_clusters,
        kept_reservoirs,
        updated_population,
    )


def _finish_cluster_reservoir(
    kept_reservoirs: list[np.ndarray],
    sampled_keys: np.ndarray,
    sampled_clusters: np.ndarray,
    distance_keys: np.ndarray,
    distances: np.ndarray,
    cluster_count: int,
) -> list[np.ndarray]:
    if sampled_keys.size:
        positions = np.searchsorted(distance_keys, sampled_keys)
        sampled_distances = distances[positions]
        if np.any(sampled_distances < 0):
            raise RuntimeError("Connected cluster observation has no finite distance")
    else:
        sampled_distances = np.empty(0, dtype=np.int32)
    return [
        np.concatenate(
            (
                kept_reservoirs[cluster],
                sampled_distances[sampled_clusters == cluster],
            )
        ).astype(np.int32, copy=False)
        for cluster in range(cluster_count)
    ]


def _distances(
    graph: sparse.csr_matrix,
    parent: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    grouped_bfs_min_targets: int,
) -> tuple[np.ndarray, dict[str, int | float]]:
    started = time.perf_counter()
    result = np.full(left.size, DISCONNECTED, dtype=np.int32)
    empty_stats: dict[str, int | float] = {
        "bidirectional_searches": 0,
        "connected_pairs": 0,
        "disconnected_pairs": int(left.size),
        "distinct_sources": 0,
        "grouped_searches": 0,
        "inspected_edges": 0,
        "seconds": 0.0,
        "visited_nodes": 0,
    }
    if left.size == 0:
        empty_stats["seconds"] = time.perf_counter() - started
        return result, empty_stats
    connected = _connected_pairs(parent, left, right)
    indices = np.flatnonzero(connected)
    if indices.size == 0:
        empty_stats["seconds"] = time.perf_counter() - started
        return result, empty_stats
    degrees = np.diff(graph.indptr)
    use_left = degrees[left[indices]] <= degrees[right[indices]]
    sources = np.where(use_left, left[indices], right[indices]).astype(np.int32)
    targets = np.where(use_left, right[indices], left[indices]).astype(np.int32)
    order = np.lexsort((targets, sources))
    sorted_sources = sources[order]
    sorted_targets = targets[order]
    boundaries = np.ones(sorted_sources.size, dtype=bool)
    boundaries[1:] = sorted_sources[1:] != sorted_sources[:-1]
    starts = np.flatnonzero(boundaries)
    group_sizes = np.diff(np.append(starts, sorted_sources.size))
    grouped_groups = group_sizes >= grouped_bfs_min_targets
    grouped_mask = np.repeat(grouped_groups, group_sizes)

    indptr = graph.indptr.astype(np.int64, copy=False)
    graph_indices = graph.indices.astype(np.int32, copy=False)
    sorted_distances = np.full(sorted_sources.size, DISCONNECTED, dtype=np.int32)
    visited_nodes = 0
    inspected_edges = 0
    bidirectional_searches = int(np.count_nonzero(~grouped_mask))
    grouped_searches = int(np.count_nonzero(grouped_groups))

    if bidirectional_searches:
        bidirectional, visited, inspected = _bidirectional_exact_bfs_profiled(
            indptr,
            graph_indices,
            sorted_sources[~grouped_mask],
            sorted_targets[~grouped_mask],
        )
        sorted_distances[~grouped_mask] = bidirectional
        visited_nodes += int(visited)
        inspected_edges += int(inspected)
    if grouped_searches:
        grouped, visited, inspected, searches = _grouped_exact_bfs_profiled(
            indptr,
            graph_indices,
            sorted_sources[grouped_mask],
            sorted_targets[grouped_mask],
        )
        if int(searches) != grouped_searches:
            raise RuntimeError("Grouped BFS source accounting is inconsistent")
        sorted_distances[grouped_mask] = grouped
        visited_nodes += int(visited)
        inspected_edges += int(inspected)

    if np.any(sorted_distances == DISCONNECTED):
        raise RuntimeError("Union-Find and cumulative graph connectivity disagree")
    result[indices[order]] = sorted_distances
    stats: dict[str, int | float] = {
        "bidirectional_searches": bidirectional_searches,
        "connected_pairs": int(indices.size),
        "disconnected_pairs": int(left.size - indices.size),
        "distinct_sources": int(starts.size),
        "grouped_searches": grouped_searches,
        "inspected_edges": inspected_edges,
        "seconds": time.perf_counter() - started,
        "visited_nodes": visited_nodes,
    }
    return result, stats


@njit(cache=True)
def _connected_pairs(
    parent: np.ndarray, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    connected = np.empty(left.size, dtype=np.bool_)
    for index in range(left.size):
        connected[index] = _find_root(parent, int(left[index])) == _find_root(
            parent, int(right[index])
        )
    return connected


def _grouped_exact_bfs(
    indptr: np.ndarray,
    graph_indices: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Compatibility wrapper returning exact distances without profiling."""
    return _grouped_exact_bfs_profiled(indptr, graph_indices, sources, targets)[0]


@njit(cache=True)
def _grouped_exact_bfs_profiled(
    indptr: np.ndarray,
    graph_indices: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, int, int, int]:
    output = np.full(sources.size, DISCONNECTED, dtype=np.int32)
    node_count = indptr.size - 1
    seen = np.zeros(node_count, dtype=np.int32)
    depth = np.zeros(node_count, dtype=np.int32)
    queue = np.empty(node_count, dtype=np.int32)
    generation = 0
    visited_total = 0
    inspected_total = 0
    searches = 0
    start = 0
    while start < sources.size:
        searches += 1
        stop = start + 1
        while stop < sources.size and sources[stop] == sources[start]:
            stop += 1
        generation += 1
        source = sources[start]
        queue[0] = source
        seen[source] = generation
        depth[source] = 0
        visited_total += 1
        head = 0
        tail = 1
        found = 0
        while head < tail and found < stop - start:
            node = queue[head]
            head += 1
            node_depth = depth[node]
            next_depth = node_depth + 1
            for edge in range(indptr[node], indptr[node + 1]):
                inspected_total += 1
                neighbor = graph_indices[edge]
                if seen[neighbor] == generation:
                    continue
                seen[neighbor] = generation
                depth[neighbor] = next_depth
                queue[tail] = neighbor
                tail += 1
                visited_total += 1
                position = _binary_search(targets, start, stop, neighbor)
                if position >= 0:
                    output[position] = next_depth
                    found += 1
        start = stop
    return output, visited_total, inspected_total, searches


def _bidirectional_exact_bfs(
    indptr: np.ndarray,
    graph_indices: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Compatibility wrapper returning exact pairwise bidirectional distances."""
    return _bidirectional_exact_bfs_profiled(
        indptr, graph_indices, sources, targets
    )[0]


@njit(cache=True)
def _bidirectional_exact_bfs_profiled(
    indptr: np.ndarray,
    graph_indices: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    output = np.full(sources.size, DISCONNECTED, dtype=np.int32)
    node_count = indptr.size - 1
    marks = np.zeros(node_count, dtype=np.int32)
    depth = np.zeros(node_count, dtype=np.int32)
    forward_queue = np.empty(node_count, dtype=np.int32)
    backward_queue = np.empty(node_count, dtype=np.int32)
    visited_total = 0
    inspected_total = 0

    for pair in range(sources.size):
        generation = pair + 1
        source = sources[pair]
        target = targets[pair]
        if source == target:
            output[pair] = 0
            continue

        marks[source] = generation
        depth[source] = 0
        marks[target] = -generation
        depth[target] = 0
        forward_queue[0] = source
        backward_queue[0] = target
        forward_head = 0
        forward_tail = 1
        forward_level_end = 1
        backward_head = 0
        backward_tail = 1
        backward_level_end = 1
        visited_total += 2

        while forward_head < forward_tail and backward_head < backward_tail:
            best = np.iinfo(np.int32).max
            while forward_head < forward_level_end:
                node = forward_queue[forward_head]
                forward_head += 1
                next_depth = depth[node] + 1
                for edge in range(indptr[node], indptr[node + 1]):
                    inspected_total += 1
                    neighbor = graph_indices[edge]
                    mark = marks[neighbor]
                    if mark == -generation:
                        candidate = next_depth + depth[neighbor]
                        if candidate < best:
                            best = candidate
                    elif mark != generation:
                        marks[neighbor] = generation
                        depth[neighbor] = next_depth
                        forward_queue[forward_tail] = neighbor
                        forward_tail += 1
                        visited_total += 1
            forward_level_end = forward_tail
            if best != np.iinfo(np.int32).max:
                output[pair] = best
                break

            best = np.iinfo(np.int32).max
            while backward_head < backward_level_end:
                node = backward_queue[backward_head]
                backward_head += 1
                next_depth = depth[node] + 1
                for edge in range(indptr[node], indptr[node + 1]):
                    inspected_total += 1
                    neighbor = graph_indices[edge]
                    mark = marks[neighbor]
                    if mark == generation:
                        candidate = next_depth + depth[neighbor]
                        if candidate < best:
                            best = candidate
                    elif mark != -generation:
                        marks[neighbor] = -generation
                        depth[neighbor] = next_depth
                        backward_queue[backward_tail] = neighbor
                        backward_tail += 1
                        visited_total += 1
            backward_level_end = backward_tail
            if best != np.iinfo(np.int32).max:
                output[pair] = best
                break
    return output, visited_total, inspected_total


@njit(cache=True)
def _binary_search(values: np.ndarray, start: int, stop: int, target: int) -> int:
    low = start
    high = stop
    while low < high:
        middle = (low + high) // 2
        if values[middle] < target:
            low = middle + 1
        else:
            high = middle
    if low < stop and values[low] == target:
        return low
    return -1


@njit(cache=True)
def _find_root(parent: np.ndarray, node: int) -> int:
    root = node
    while parent[root] != root:
        root = parent[root]
    while parent[node] != node:
        following = parent[node]
        parent[node] = root
        node = following
    return root


@njit(cache=True)
def _union_edges(
    parent: np.ndarray, size: np.ndarray, left: np.ndarray, right: np.ndarray
) -> None:
    for index in range(left.size):
        first = _find_root(parent, int(left[index]))
        second = _find_root(parent, int(right[index]))
        if first == second:
            continue
        if size[first] < size[second]:
            first, second = second, first
        parent[second] = first
        size[first] += size[second]


def _add_graph_edges(
    graph: sparse.csr_matrix, left: np.ndarray, right: np.ndarray, node_count: int
) -> sparse.csr_matrix:
    if left.size == 0:
        return graph
    rows = np.concatenate((left, right))
    columns = np.concatenate((right, left))
    additions = sparse.csr_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, columns)),
        shape=(node_count, node_count),
        dtype=np.int8,
    )
    combined = (graph + additions).tocsr()
    combined.data[:] = 1
    combined.sort_indices()
    return combined


def _restore_prior_graph(
    scratch: Path, years: Sequence[int], completed: set[int], node_count: int
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    graph = sparse.csr_matrix((node_count, node_count), dtype=np.int8)
    parent = np.arange(node_count, dtype=np.int32)
    size = np.ones(node_count, dtype=np.int64)
    seen = np.empty(0, dtype=np.uint64)
    for year in years:
        if year not in completed:
            break
        path = _edge_path(scratch, year)
        if not path.exists():
            raise RuntimeError(
                f"Completed year {year} is missing resumable full-edge scratch"
            )
        with np.load(path, allow_pickle=False) as payload:
            left = payload["author_i"].astype(np.int32, copy=False)
            right = payload["author_j"].astype(np.int32, copy=False)
        _union_edges(parent, size, left, right)
        graph = _add_graph_edges(graph, left, right, node_count)
        seen = _merge_sorted_unique(seen, _encode_pairs(left, right))
    return graph, parent, size, seen


def _restore_cluster_reservoir(
    scratch: Path,
    years: Sequence[int],
    completed: set[int],
    cluster_count: int,
    cluster_sample: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    completed_years = [year for year in years if year in completed]
    if not completed_years:
        return (
            [np.empty(0, dtype=np.int32) for _ in range(cluster_count)],
            np.zeros(cluster_count, dtype=np.int64),
        )
    path = _reservoir_path(scratch, completed_years[-1])
    if not path.exists():
        raise RuntimeError(
            f"Completed year {completed_years[-1]} is missing reservoir scratch"
        )
    with np.load(path, allow_pickle=False) as payload:
        offsets = payload["offsets"].astype(np.int64, copy=False)
        distances = payload["distances"].astype(np.int32, copy=False)
        population = payload["population"].astype(np.int64, copy=False)
    if offsets.shape != (cluster_count + 1,) or population.shape != (cluster_count,):
        raise RuntimeError("Cluster reservoir does not align with cluster metadata")
    reservoirs = [
        distances[offsets[cluster] : offsets[cluster + 1]].copy()
        for cluster in range(cluster_count)
    ]
    expected_sizes = np.minimum(population, cluster_sample)
    if not np.array_equal(
        np.asarray([values.size for values in reservoirs], dtype=np.int64),
        expected_sizes,
    ):
        raise RuntimeError("Cluster reservoir sample sizes do not match populations")
    return reservoirs, population.copy()


def _encode_pairs(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left.astype(np.uint64) << np.uint64(32)) | right.astype(np.uint64)


def _decode_pairs(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = (keys >> np.uint64(32)).astype(np.int32)
    right = (keys & np.uint64(0xFFFFFFFF)).astype(np.int32)
    return left, right


def _merge_sorted_unique(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if first.size == 0:
        return second.copy()
    if second.size == 0:
        return first
    return _merge_disjoint(first, second)


@njit(cache=True)
def _merge_disjoint(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    result = np.empty(first.size + second.size, dtype=np.uint64)
    i = 0
    j = 0
    output = 0
    while i < first.size and j < second.size:
        if first[i] < second[j]:
            result[output] = first[i]
            i += 1
        elif second[j] < first[i]:
            result[output] = second[j]
            j += 1
        else:
            result[output] = first[i]
            i += 1
            j += 1
        output += 1
    while i < first.size:
        result[output] = first[i]
        i += 1
        output += 1
    while j < second.size:
        result[output] = second[j]
        j += 1
        output += 1
    return result[:output]


def _fresh_manifest(
    config: dict, years: Sequence[int], author_ids: np.ndarray, scratch: Path
) -> dict:
    return {
        **config,
        "artifact_version": 6,
        "author_count": int(author_ids.size),
        "cluster_level": None,
        "cluster_metadata": "cluster_metadata.npz",
        "cluster_distance_reservoir": "cluster_distance_reservoir.npz",
        "cluster_years_dir": "cluster_years",
        "clustered_link_counts": {},
        "completed_years": [],
        "distance_algorithm": "exact hybrid bidirectional/grouped CSR BFS",
        "distance_definition": "exact unweighted hop distance in the graph through year y-1",
        "distance_disconnected": int(DISCONNECTED),
        "distance_dtype": "int32",
        "distance_stats": {},
        "format": "year-partitioned-npz",
        "hyperauthored_papers": {
            str(year): int(_read_json(_stats_path(scratch, year))["hyperauthored_papers"])
            for year in years
        },
        "link_counts": {},
        "no_cluster": int(NO_CLUSTER),
        "sampling_design": (
            "year-stratified simple random samples without replacement for no-cluster "
            "links and each attributed cluster; one global reservoir sample per "
            "cluster for connected new paper-pair observations"
        ),
        "sampling_stats": {},
        "stored_link_counts": {},
        "papers_kept": {
            str(year): int(_read_json(_stats_path(scratch, year))["papers_kept"])
            for year in years
        },
        "phase1_complete": True,
        "solo_papers": {
            str(year): int(_read_json(_stats_path(scratch, year))["solo_papers"])
            for year in years
        },
        "years": [int(year) for year in years],
    }


def _check_config(manifest: dict, config: dict) -> None:
    for key, value in config.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Existing output used {key}={manifest.get(key)!r}, not {value!r}. "
                "Use a new --output-dir or resume with the same parameters."
            )


def _check_completed_prefix(years: Sequence[int], completed: set[int]) -> None:
    expected = set(years[: len(completed)])
    if completed != expected:
        raise ValueError("Completed years are not a contiguous prefix")


def _check_index_limit(author_ids: np.ndarray) -> None:
    if author_ids.size > _INT32_MAX:
        raise ValueError("Author index does not fit in int32 graph indices")


def _is_complete(manifest: dict, output: Path) -> bool:
    years = [int(year) for year in manifest.get("years", [])]
    completed = {int(year) for year in manifest.get("completed_years", [])}
    return (
        bool(manifest.get("phase1_complete"))
        and (output / "cluster_metadata.npz").exists()
        and (output / "cluster_distance_reservoir.npz").exists()
        and all(
            year in completed
            and _year_path(output, year).exists()
            and _cluster_year_path(output, year).exists()
            for year in years
        )
    )


def _output_exists(output: Path) -> bool:
    if not output.exists():
        return False
    return any(output.iterdir())


def _scratch_ready(scratch: Path, year: int) -> bool:
    return all(
        path.exists()
        for path in (
            _articles_path(scratch, year),
            _authors_path(scratch, year),
            _counts_path(scratch, year),
            _unique_path(scratch, year),
            _stats_path(scratch, year),
        )
    )


def _save_year(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _save_edges(path: Path, left: np.ndarray, right: np.ndarray) -> None:
    _save_year(path, {"author_i": left, "author_j": right})


def _save_cluster_metadata(
    path: Path,
    event_offsets: np.ndarray,
    event_clusters: np.ndarray,
    event_years: np.ndarray,
    cluster_labels: np.ndarray,
    years: Sequence[int],
) -> None:
    cluster_count = int(cluster_labels.size)
    analyzed = np.isin(event_years, np.asarray(years, dtype=np.int32))
    flat_analyzed = np.repeat(analyzed, np.diff(event_offsets))
    paper_counts = np.bincount(
        event_clusters[flat_analyzed], minlength=cluster_count
    ).astype(np.int64)
    _save_year(
        path,
        {
            "cluster_id": np.arange(cluster_count, dtype=np.int32),
            "label": cluster_labels,
            "paper_count": paper_counts,
        },
    )


def _save_cluster_year(path: Path, observations: dict[str, np.ndarray]) -> None:
    _save_year(path, observations)


def _save_cluster_reservoir(
    path: Path,
    reservoirs: Sequence[np.ndarray],
    population: np.ndarray,
) -> None:
    sizes = np.asarray([values.size for values in reservoirs], dtype=np.int64)
    offsets = np.empty(sizes.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(sizes, out=offsets[1:])
    distances = (
        np.concatenate(reservoirs).astype(np.int32, copy=False)
        if reservoirs
        else np.empty(0, dtype=np.int32)
    )
    _save_year(
        path,
        {
            "distances": distances,
            "offsets": offsets,
            "population": population.astype(np.int64, copy=False),
        },
    )


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _author_index_path(output: Path) -> Path:
    return output / "author_ids.npy"


def _year_path(output: Path, year: int) -> Path:
    return output / "years" / f"{year}.npz"


def _cluster_year_path(output: Path, year: int) -> Path:
    return output / "cluster_years" / f"{year}.npz"


def _reservoir_path(scratch: Path, year: int) -> Path:
    return scratch / "cluster_reservoirs" / f"{year}.npz"


def _edge_path(scratch: Path, year: int) -> Path:
    return scratch / "completed_edges" / f"{year}.npz"


def _articles_path(scratch: Path, year: int) -> Path:
    return scratch / f"{year}.articles.i64"


def _authors_path(scratch: Path, year: int) -> Path:
    return scratch / f"{year}.authors.i64"


def _counts_path(scratch: Path, year: int) -> Path:
    return scratch / f"{year}.counts.i32"


def _unique_path(scratch: Path, year: int) -> Path:
    return scratch / f"{year}.unique.i64"


def _stats_path(scratch: Path, year: int) -> Path:
    return scratch / f"{year}.stats.json"


def _load_manifest(path: Path) -> dict:
    return _read_json(path)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    raise SystemExit(main())
