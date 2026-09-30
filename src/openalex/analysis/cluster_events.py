"""Cluster event keywords with the keyword dendrogram.

Complete linkage uses cosine distance between L2-normalized co-occurrence
rows, the same similarity the website dendrogram uses. The cosine similarity
threshold is the coarsest cut that still reaches a target number of clusters.

The partition is the grouping used for the coarsened matrix ``H.T @ M @ H``.
Yearly cluster counts read the Boolean paper-keyword incidence and count each
paper once per cluster.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.cluster.hierarchy import fcluster

from openalex.analysis.filter_events import find_filtered_events, genuine_ngrams
from openalex.website.build import cosine_complete_linkage, load_event_artifacts

logger = logging.getLogger(__name__)

DEFAULT_MIN_DOCUMENT_FREQUENCY = 1
DEFAULT_N_CLUSTERS = 20
DENDROGRAM_METHOD = "complete-linkage-cosine"


@dataclass(frozen=True)
class CosineCut:
    """Complete-linkage cosine cut and the flat cluster count it produces."""

    similarity: float
    distance: float
    clusters: int


@dataclass(frozen=True)
class DendrogramFit:
    """Flat keyword partition from the complete-linkage cosine cut."""

    levels: tuple[np.ndarray, ...]
    cluster_similarity: float
    target_clusters: int


def cooccurrence_adjacency(matrix, selected: np.ndarray) -> sparse.csr_matrix:
    """Return the symmetric integer adjacency of the selected keywords.

    The diagonal is removed. Each surviving entry remains a paper count.
    """
    selected = np.asarray(selected, dtype=np.int64)
    if selected.size == 0:
        raise ValueError("No keywords selected")
    if np.any(selected < 0):
        raise ValueError("Keyword indices must be nonnegative")
    block = matrix[selected][:, selected].tocsr()
    if block.shape[0] != block.shape[1]:
        raise ValueError("Co-occurrence matrix must be square")
    block = _integer_counts(block)
    block.sum_duplicates()
    difference = (block - block.T).tocsr()
    difference.eliminate_zeros()
    if difference.nnz:
        raise ValueError("Co-occurrence matrix must be symmetric")
    block.setdiag(0)
    block.eliminate_zeros()
    return block


def project_level(levels: Sequence[np.ndarray], level: int) -> np.ndarray:
    """Project one hierarchy level onto the original keywords."""
    if not levels:
        raise ValueError("Cluster hierarchy is empty")
    if level < 0 or level >= len(levels):
        raise ValueError(f"Hierarchy level {level} is outside 0..{len(levels) - 1}")
    labels = np.asarray(levels[0], dtype=np.int64).copy()
    for depth in range(1, level + 1):
        parent = np.asarray(levels[depth], dtype=np.int64)
        if labels.size and (int(labels.min()) < 0 or int(labels.max()) >= parent.size):
            raise ValueError("Hierarchy level does not index the next level")
        labels = parent[labels]
    return _compact_labels(labels)


def project_hierarchy(levels: Sequence[np.ndarray]) -> np.ndarray:
    """Return compact group ids with one row per hierarchy level."""
    projected = [project_level(levels, level) for level in range(len(levels))]
    if not projected:
        return np.empty((0, 0), dtype=np.int32)
    width = projected[0].shape[0]
    if any(row.shape != (width,) for row in projected):
        raise ValueError("Projected hierarchy levels have different lengths")
    return np.vstack(projected)


def coarsen(adjacency: sparse.csr_matrix, groups: np.ndarray) -> sparse.csr_matrix:
    """Aggregate pairwise counts with ``H.T @ M @ H``."""
    groups = np.asarray(groups)
    if groups.shape != (adjacency.shape[0],):
        raise ValueError("Group labels must align with the adjacency rows")
    count = adjacency.shape[0]
    if count == 0:
        return sparse.csr_matrix((0, 0), dtype=np.int64)
    if groups.size and (int(groups.min()) < 0):
        raise ValueError("Group labels must be nonnegative")
    group_count = int(groups.max()) + 1 if groups.size else 0
    membership = sparse.csr_matrix(
        (
            np.ones(count, dtype=np.int64),
            (np.arange(count, dtype=np.int64), groups.astype(np.int64, copy=False)),
        ),
        shape=(count, group_count),
        dtype=np.int64,
    )
    coarse = (membership.T @ adjacency @ membership).tocsr()
    if int(coarse.sum()) != int(adjacency.sum()):
        raise AssertionError("Coarse matrix did not preserve the total count")
    return coarse


def cut_for_cluster_count(linkage_matrix: np.ndarray, target_clusters: int) -> CosineCut:
    """Return the coarsest complete-linkage cut that still meets `target_clusters`.

    The similarity is the height of the last merge in that partition, so every
    kept join has cosine similarity at least this value. Equal merge heights
    skip some counts; the cut then uses the next finer partition. A target
    above the number of separated keywords uses the finest partition.
    """
    if target_clusters < 1:
        raise ValueError("--n-clusters must be >= 1")
    linkage_matrix = np.asarray(linkage_matrix, dtype=np.float64)
    if linkage_matrix.size == 0:
        chosen = CosineCut(similarity=1.0, distance=0.0, clusters=1)
    else:
        if linkage_matrix.ndim != 2 or linkage_matrix.shape[1] < 3:
            raise ValueError("Linkage matrix must record merge distances")
        distances = linkage_matrix[:, 2]
        if distances.size > 1 and np.any(np.diff(distances) < 0):
            raise ValueError("Linkage distances must be nondecreasing")
        leaf_count = int(distances.shape[0]) + 1
        attainable: list[CosineCut] = []
        if float(distances[0]) > 0.0:
            attainable.append(CosineCut(similarity=1.0, distance=0.0, clusters=leaf_count))
        for included in range(1, leaf_count):
            last = float(distances[included - 1])
            if included < leaf_count - 1 and float(distances[included]) <= last:
                continue
            attainable.append(
                CosineCut(
                    similarity=1.0 - last,
                    distance=last,
                    clusters=leaf_count - included,
                )
            )
        if not attainable:
            raise RuntimeError("Dendrogram produced no attainable cluster count")
        meeting = [cut for cut in attainable if cut.clusters >= target_clusters]
        chosen = (
            min(meeting, key=lambda cut: cut.clusters)
            if meeting
            else max(attainable, key=lambda cut: cut.clusters)
        )
    if chosen.clusters == target_clusters:
        logger.info(
            "Cosine similarity %.6f yields %s clusters",
            chosen.similarity,
            chosen.clusters,
        )
    else:
        logger.warning(
            "Cosine similarity %.6f yields %s clusters; target was %s",
            chosen.similarity,
            chosen.clusters,
            target_clusters,
        )
    return chosen


def fit_dendrogram(matrix, *, n_clusters: int) -> DendrogramFit:
    """Cut complete linkage so the flat partition meets `n_clusters`."""
    count = matrix.shape[0]
    if count == 0:
        raise ValueError("No keywords selected")
    if count == 1:
        linkage_matrix = np.empty((0, 4), dtype=float)
    else:
        linkage_matrix = cosine_complete_linkage(matrix)
    cut = cut_for_cluster_count(linkage_matrix, n_clusters)
    if count == 1:
        labels = np.zeros(1, dtype=np.int32)
    else:
        labels = _compact_labels(
            fcluster(linkage_matrix, t=cut.distance, criterion="distance")
        )
    observed = int(labels.max()) + 1 if labels.size else 0
    if observed != cut.clusters:
        raise AssertionError(
            f"Dendrogram cut produced {observed} clusters, expected {cut.clusters}"
        )
    return DendrogramFit(
        levels=(labels,),
        cluster_similarity=float(cut.similarity),
        target_clusters=int(n_clusters),
    )


def select_keywords(
    artifacts: Mapping[str, object],
    *,
    source: str,
    min_document_frequency: int,
    allowed: Collection[str] | None = None,
) -> np.ndarray:
    """Choose vocabulary rows, in vocabulary order."""
    if min_document_frequency < 1:
        raise ValueError("--min-document-frequency must be >= 1")
    if source not in {"events", "all"}:
        raise ValueError("--keywords must be 'events' or 'all'")
    vocabulary = artifacts["vocabulary"]
    frequencies = np.asarray(artifacts["frequencies"])
    eligible = frequencies >= min_document_frequency
    if source == "all":
        selected = np.flatnonzero(eligible).astype(np.int64, copy=False)
    else:
        wanted = {row["ngram"] for row in _event_rows(Path(artifacts["root"]))}
        if allowed is not None:
            wanted &= set(allowed)
        selected = np.asarray(
            [
                index
                for index, keyword in enumerate(vocabulary)
                if str(keyword) in wanted and bool(eligible[index])
            ],
            dtype=np.int64,
        )
    if selected.size == 0:
        if allowed is not None:
            raise ValueError("No genuine keywords passed the document-frequency filter")
        raise ValueError("No keywords passed the document-frequency filter")
    return selected


def _positive_norm_keywords(matrix, selected: np.ndarray) -> np.ndarray:
    """Drop keywords whose co-occurrence row has zero L2 norm."""
    selected = np.asarray(selected, dtype=np.int64)
    block = matrix[selected][:, selected]
    squared = np.asarray(block.multiply(block).sum(axis=1)).ravel()
    kept = selected[squared > 0]
    dropped = int(selected.size - kept.size)
    if dropped:
        logger.info("Dropping %s keywords with zero co-occurrence norm", dropped)
    if kept.size == 0:
        raise ValueError("Selected keywords have zero co-occurrence norm")
    return kept


def cluster_paper_counts(
    artifacts: Mapping[str, object],
    selected_indices: np.ndarray,
    groups_by_level: np.ndarray,
) -> dict[tuple[int, int, int], int]:
    """Count each paper once per hierarchy level, group, and year."""
    if groups_by_level.size == 0:
        return {}
    vocabulary_size = len(artifacts["vocabulary"])
    level_count, keyword_count = groups_by_level.shape
    if keyword_count != len(selected_indices):
        raise ValueError("Group rows must align with the selected keywords")
    manifest = artifacts["manifest"]
    incidence_parts = list(manifest.get("incidence_parts", []))
    if not incidence_parts:
        logger.warning("Event artifacts have no incidence parts; yearly cluster counts are empty")
        return {}

    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    widths = [int(groups_by_level[level].max()) + 1 for level in range(level_count)]
    offsets = np.cumsum([0, *widths[:-1]])
    column_count = int(sum(widths))
    columns = np.concatenate(
        [
            groups_by_level[level].astype(np.int64, copy=False) + int(offsets[level])
            for level in range(level_count)
        ]
    )
    rows = np.tile(selected_indices, level_count)
    membership = sparse.csr_matrix(
        (np.ones(rows.size, dtype=np.int64), (rows, columns)),
        shape=(vocabulary_size, column_count),
        dtype=np.int64,
    )
    by_year: dict[int, np.ndarray] = {}
    root = Path(artifacts["root"])
    incidence_dir = root / manifest["incidence_dir"]
    for relative in incidence_parts:
        matrix_path = incidence_dir / relative
        years_path = incidence_dir / relative.replace(".npz", "_years.npy")
        article_terms = sparse.load_npz(matrix_path).tocsr()
        years = np.load(years_path, allow_pickle=False).astype(np.int32, copy=False)
        if article_terms.shape[0] != len(years):
            raise ValueError(f"Incidence rows and years do not match for {relative}")
        if article_terms.shape[1] != vocabulary_size:
            raise ValueError(f"Incidence width does not match the vocabulary for {relative}")
        presence = (article_terms.astype(np.int64) @ membership).tocsr()
        presence.data[:] = 1
        for year in np.unique(years):
            values = np.asarray(presence[years == year].sum(axis=0), dtype=np.int64).ravel()
            bucket = by_year.setdefault(int(year), np.zeros(column_count, dtype=np.int64))
            bucket += values

    counts: dict[tuple[int, int, int], int] = {}
    for level, width in enumerate(widths):
        start = int(offsets[level])
        for year, values in by_year.items():
            for group in range(width):
                papers = int(values[start + group])
                if papers:
                    counts[(level, group, year)] = papers
    return counts


def cluster_event_keywords(
    events_dir: str | Path,
    output_dir: str | Path,
    *,
    source: str = "events",
    min_document_frequency: int = DEFAULT_MIN_DOCUMENT_FREQUENCY,
    level: int = 0,
    filtered_dir: str | Path | None = None,
    n_clusters: int = DEFAULT_N_CLUSTERS,
) -> dict[str, object]:
    """Cluster event keywords and write the coarsened event artifacts."""
    events_root = Path(events_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output == events_root:
        raise ValueError("--output-dir must be different from --events-dir")
    if level < 0:
        raise ValueError("--level must be >= 0")
    if n_clusters < 1:
        raise ValueError("--n-clusters must be >= 1")

    artifacts = load_event_artifacts(events_root)
    filtered_root = find_filtered_events(events_root, filtered_dir) if source == "events" else None
    allowed = genuine_ngrams(filtered_root) if filtered_root is not None else None
    if allowed is not None:
        logger.info("Restricting clusters to %s genuine keywords from %s", len(allowed), filtered_root)
    selected = select_keywords(
        artifacts,
        source=source,
        min_document_frequency=min_document_frequency,
        allowed=allowed,
    )
    selected = _positive_norm_keywords(artifacts["matrix"], selected)
    block = artifacts["matrix"][selected][:, selected].tocsr()
    adjacency = cooccurrence_adjacency(artifacts["matrix"], selected)
    logger.info(
        "Clustering %s keywords by complete linkage toward %s clusters",
        len(selected),
        n_clusters,
    )
    fitted = fit_dendrogram(block, n_clusters=n_clusters)
    if level >= len(fitted.levels):
        raise ValueError(
            f"--level {level} is outside the fitted hierarchy of {len(fitted.levels)} levels"
        )
    groups_by_level = project_hierarchy(fitted.levels)
    chosen = groups_by_level[level]
    yearly = cluster_paper_counts(artifacts, selected, groups_by_level)
    _write_outputs(
        output,
        artifacts,
        selected,
        adjacency,
        fitted,
        groups_by_level,
        chosen_level=level,
        yearly=yearly,
        source=source,
        min_document_frequency=min_document_frequency,
        filtered_dir=str(filtered_root) if filtered_root is not None else None,
    )
    summary = {
        "output_dir": str(output),
        "keywords": int(len(selected)),
        "keyword_filter": "genuine" if allowed is not None else source,
        "levels": int(groups_by_level.shape[0]),
        "groups": int(chosen.max()) + 1 if chosen.size else 0,
        "level": int(level),
        "method": DENDROGRAM_METHOD,
        "cluster_similarity": fitted.cluster_similarity,
        "target_clusters": int(fitted.target_clusters),
    }
    logger.info(
        "Wrote %s groups at cosine similarity %.6f (target %s) to %s",
        summary["groups"],
        fitted.cluster_similarity,
        fitted.target_clusters,
        output,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster event keywords by complete linkage on cosine similarity, "
            "choosing the threshold that meets --n-clusters."
        )
    )
    parser.add_argument("--events-dir", type=Path, default=Path("output/events"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/event_clusters"))
    parser.add_argument(
        "--keywords",
        choices=("events", "all"),
        default="events",
        help=(
            "Cluster events.csv keywords, or every keyword in the co-occurrence vocabulary. "
            "The events source uses genuine keywords when filter-events output is available."
        ),
    )
    parser.add_argument(
        "--filtered-dir",
        type=Path,
        default=None,
        help=(
            "filter-events output whose genuine keywords are clustered. "
            "Defaults to a filtered_events directory beside --events-dir when classifications.csv exists. "
            "Ignored with --keywords all."
        ),
    )
    parser.add_argument(
        "--min-document-frequency",
        type=int,
        default=DEFAULT_MIN_DOCUMENT_FREQUENCY,
        help="Drop keywords appearing in fewer papers than this.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=DEFAULT_N_CLUSTERS,
        help=(
            "Target number of clusters. The cosine similarity threshold is the coarsest "
            "complete-linkage cut that still reaches this count "
            f"(default {DEFAULT_N_CLUSTERS})."
        ),
    )
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        help="Partition reported as group in keyword_groups.csv. The dendrogram cut has a single level, 0.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args(argv)
    summary = cluster_event_keywords(
        args.events_dir,
        args.output_dir,
        source=args.keywords,
        min_document_frequency=args.min_document_frequency,
        level=args.level,
        filtered_dir=args.filtered_dir,
        n_clusters=args.n_clusters,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _integer_counts(matrix) -> sparse.csr_matrix:
    data = np.asarray(matrix.data)
    if data.size and np.any(data < 0):
        raise ValueError("Co-occurrence counts must be nonnegative")
    if data.size and not np.all(data == np.floor(data)):
        raise ValueError("Co-occurrence counts must be nonnegative integers")
    converted = matrix.astype(np.int64)
    converted.eliminate_zeros()
    return converted.tocsr()


def _compact_labels(labels: np.ndarray) -> np.ndarray:
    if labels.size == 0:
        return np.array([], dtype=np.int32)
    _unique, inverse = np.unique(labels, return_inverse=True)
    return inverse.astype(np.int32, copy=False)


def _event_rows(events_dir: Path) -> list[dict[str, str]]:
    path = events_dir / "events.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} was not found. Re-run event extraction or pass --keywords all."
        )
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "ngram" not in reader.fieldnames:
            raise ValueError(f"{path} must contain an ngram column")
        return list(reader)


def _write_outputs(
    output: Path,
    artifacts: Mapping[str, object],
    selected: np.ndarray,
    adjacency: sparse.csr_matrix,
    fitted: DendrogramFit,
    groups_by_level: np.ndarray,
    *,
    chosen_level: int,
    yearly: Mapping[tuple[int, int, int], int],
    source: str,
    min_document_frequency: int,
    filtered_dir: str | None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    coarse_dir = output / "coarse"
    if coarse_dir.exists():
        shutil.rmtree(coarse_dir)
    coarse_dir.mkdir()

    vocabulary = artifacts["vocabulary"]
    frequencies = np.asarray(artifacts["frequencies"])
    keywords = np.asarray([str(vocabulary[index]) for index in selected], dtype=np.str_)
    event_fields = _event_field_map(Path(artifacts["root"]))
    papers_by_year = {
        int(year): int(count)
        for year, count in artifacts["manifest"].get("papers_by_year", {}).items()
    }

    for level in range(groups_by_level.shape[0]):
        _save_npz(
            coarse_dir / f"level_{level}.npz",
            coarsen(adjacency, groups_by_level[level]),
        )
    np.save(output / "keywords.npy", keywords, allow_pickle=False)
    np.save(output / "groups_by_level.npy", groups_by_level, allow_pickle=False)
    _save_hierarchy(output / "hierarchy.npz", fitted.levels)
    _write_keyword_groups(
        output / "keyword_groups.csv",
        keywords,
        selected,
        frequencies,
        groups_by_level,
        chosen_level,
        event_fields,
    )
    _write_clusters(output / "clusters.csv", keywords, groups_by_level)
    _write_yearly(output / "cluster_by_year.csv", yearly, papers_by_year)
    manifest = {
        "method": DENDROGRAM_METHOD,
        "keywords": "keywords.npy",
        "groups_by_level": "groups_by_level.npy",
        "hierarchy": "hierarchy.npz",
        "keyword_groups": "keyword_groups.csv",
        "clusters": "clusters.csv",
        "cluster_by_year": "cluster_by_year.csv",
        "coarse_dir": "coarse",
        "coarse_level": int(chosen_level),
        "keyword_source": source,
        "keyword_filter": "genuine" if filtered_dir else source,
        "filtered_dir": filtered_dir,
        "min_document_frequency": int(min_document_frequency),
        "levels": int(groups_by_level.shape[0]),
        "keyword_count": int(len(selected)),
        "alignment": "groups_by_level columns match keywords.npy",
        "diagonal": "removed",
        "coarse_diagonal": "within-group pairs counted once in each direction",
        "edge_weight": "paper co-occurrence count",
        "paper_counts": "a paper is counted once per cluster and year",
        "cluster_similarity": fitted.cluster_similarity,
        "target_clusters": fitted.target_clusters,
        "linkage": "complete",
        "profile": "l2-normalized-cooccurrence-cosine",
    }
    _atomic_json(output / "manifest.json", manifest)


def _event_field_map(events_dir: Path) -> dict[str, dict[str, str]]:
    path = events_dir / "events.csv"
    if not path.is_file():
        return {}
    rows = _event_rows(events_dir)
    kept = ("fold_change", "max_year", "min_year")
    return {
        row["ngram"]: {field: row[field] for field in kept if field in row}
        for row in rows
    }


def _write_keyword_groups(
    path: Path,
    keywords: np.ndarray,
    selected: np.ndarray,
    frequencies: np.ndarray,
    groups_by_level: np.ndarray,
    chosen_level: int,
    event_fields: Mapping[str, Mapping[str, str]],
) -> None:
    level_fields = [f"level_{level}" for level in range(groups_by_level.shape[0])]
    extra_fields = ("fold_change", "max_year", "min_year")
    fieldnames = [
        "keyword",
        "vocabulary_index",
        "document_frequency",
        "group",
        *level_fields,
        *extra_fields,
    ]
    rows = []
    for position, keyword in enumerate(keywords):
        event = event_fields.get(str(keyword), {})
        row = {
            "keyword": str(keyword),
            "vocabulary_index": int(selected[position]),
            "document_frequency": int(frequencies[selected[position]]),
            "group": int(groups_by_level[chosen_level, position]),
        }
        for level, name in enumerate(level_fields):
            row[name] = int(groups_by_level[level, position])
        for field in extra_fields:
            row[field] = event.get(field, "")
        rows.append(row)
    rows.sort(key=lambda item: (item["group"], item["keyword"]))
    _write_csv(path, fieldnames, rows)


def _write_clusters(path: Path, keywords: np.ndarray, groups_by_level: np.ndarray) -> None:
    rows = []
    for level in range(groups_by_level.shape[0]):
        labels = groups_by_level[level]
        for group in range(int(labels.max()) + 1 if labels.size else 0):
            members = [str(keywords[index]) for index in np.flatnonzero(labels == group)]
            rows.append(
                {
                    "level": level,
                    "group": group,
                    "n_keywords": len(members),
                    "keywords": " | ".join(members),
                }
            )
    _write_csv(path, ["level", "group", "n_keywords", "keywords"], rows)


def _write_yearly(
    path: Path,
    yearly: Mapping[tuple[int, int, int], int],
    papers_by_year: Mapping[int, int],
) -> None:
    rows = []
    for (level, group, year), papers in sorted(yearly.items()):
        denominator = papers_by_year.get(year, 0)
        rows.append(
            {
                "level": level,
                "group": group,
                "year": year,
                "papers": papers,
                "share": papers / denominator if denominator else 0.0,
            }
        )
    _write_csv(path, ["level", "group", "year", "papers", "share"], rows)


def _save_hierarchy(path: Path, levels: Sequence[np.ndarray]) -> None:
    payload = {f"level_{index}": np.asarray(level) for index, level in enumerate(levels)}
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def _save_npz(path: Path, matrix: sparse.csr_matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    sparse.save_npz(temporary, matrix, compressed=True)
    os.replace(temporary, path)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
