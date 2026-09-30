"""Build the event-keyword website."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
from scipy import sparse
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

DEFAULT_CLUSTER_SIMILARITY = 0.5
DEFAULT_MIN_DOCUMENT_FREQUENCY = 10
DEFAULT_TOP_KEYWORDS = 100
DEFAULT_MAX_DENDROGRAM_KEYWORDS = 500
SITE_ASSETS = (
    "index.html",
    "dendrogram.html",
    "clusters.html",
    "link-distances.html",
    "app.js",
    "dendrogram.js",
    "clusters.js",
    "link-distances.js",
    "style.css",
)


def load_event_artifacts(events_dir: str | Path) -> dict[str, Any]:
    root = Path(events_dir).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if int(manifest.get("artifact_version", 0)) != 1:
        raise ValueError("Unsupported event artifact version")
    vocabulary = np.load(root / manifest["vocabulary"], allow_pickle=False).astype(str)
    frequencies = np.load(root / manifest["document_frequency"], allow_pickle=False)
    matrix = sparse.load_npz(root / manifest["cooccurrence"]).tocsr()
    expected = (len(vocabulary), len(vocabulary))
    if matrix.shape != expected or frequencies.shape != (len(vocabulary),):
        raise ValueError("Event vocabulary, frequency, and matrix shapes do not match")
    return {
        "root": root,
        "manifest": manifest,
        "vocabulary": vocabulary,
        "frequencies": frequencies.astype(np.int64, copy=False),
        "matrix": matrix,
    }


def select_keywords(
    matrix,
    frequencies: np.ndarray,
    *,
    min_document_frequency: int,
    max_keywords: int,
    vocabulary: np.ndarray | None = None,
    allowed: set[str] | None = None,
) -> np.ndarray:
    """Select frequent words with a nonzero filtered co-occurrence direction."""
    candidates = np.flatnonzero(frequencies >= min_document_frequency)
    if allowed is not None:
        if vocabulary is None:
            raise ValueError("vocabulary is required when filtering keywords")
        candidates = np.asarray(
            [int(index) for index in candidates if str(vocabulary[int(index)]) in allowed],
            dtype=np.int64,
        )
    if not len(candidates):
        return candidates
    ranked = candidates[np.argsort(-frequencies[candidates], kind="stable")]
    ranked = ranked[:max_keywords]
    filtered = matrix[ranked][:, ranked]
    norms = np.sqrt(np.asarray(filtered.multiply(filtered).sum(axis=1)).ravel())
    return ranked[norms > 0]


def l2_normalize_cooccurrence(matrix):
    """L2-normalize each co-occurrence row, as in the keyword dendrogram.

    Cosine similarity of these rows is the similarity clustered by complete
    linkage. Every row must have positive norm; the dendrogram drops the rest
    before calling this.
    """
    filtered = matrix.astype(np.float64).tocsr()
    if filtered.shape[0] != filtered.shape[1]:
        raise ValueError("Co-occurrence matrix must be square")
    norms = np.sqrt(np.asarray(filtered.multiply(filtered).sum(axis=1)).ravel())
    if norms.size and np.any(norms <= 0):
        raise ValueError("Co-occurrence rows must have positive L2 norm")
    return filtered.multiply(1.0 / norms[:, None]).tocsr()


def cosine_complete_linkage(matrix) -> np.ndarray:
    """Complete linkage on cosine distance of L2-normalized co-occurrence rows."""
    count = matrix.shape[0]
    if count <= 1:
        return np.empty((0, 4), dtype=float)
    normalized = l2_normalize_cooccurrence(matrix)
    similarities = (normalized @ normalized.T).toarray()
    distances = np.clip(1.0 - similarities, 0.0, 2.0)
    np.fill_diagonal(distances, 0.0)
    return linkage(squareform(distances, checks=False), method="complete")


def cluster_keywords(
    matrix,
    selected_indices: np.ndarray,
    *,
    cluster_similarity: float,
) -> dict[str, Any]:
    """Normalize rows, complete-link cluster cosine distance, and coarsen M."""
    if not 0 <= cluster_similarity <= 1:
        raise ValueError("cluster_similarity must be between 0 and 1")
    filtered = matrix[selected_indices][:, selected_indices].astype(np.float64).tocsr()
    count = filtered.shape[0]
    if count == 0:
        return {
            "matrix": filtered,
            "linkage": np.empty((0, 4), dtype=float),
            "labels": np.array([], dtype=np.int32),
            "coarse": sparse.csr_matrix((0, 0), dtype=matrix.dtype),
        }
    if count == 1:
        linkage_matrix = np.empty((0, 4), dtype=float)
        labels = np.array([1], dtype=np.int32)
    else:
        linkage_matrix = cosine_complete_linkage(filtered)
        labels = fcluster(
            linkage_matrix,
            t=1.0 - cluster_similarity,
            criterion="distance",
        ).astype(np.int32)
    unique_labels = sorted(set(int(value) for value in labels))
    label_to_column = {label: index for index, label in enumerate(unique_labels)}
    membership = sparse.csr_matrix(
        (
            np.ones(count, dtype=np.int64),
            (
                np.arange(count),
                np.array([label_to_column[int(label)] for label in labels]),
            ),
        ),
        shape=(count, len(unique_labels)),
    )
    coarse = (membership.T @ filtered @ membership).tocsr()
    if not math.isclose(float(coarse.sum()), float(filtered.sum())):
        raise AssertionError("Coarse matrix did not preserve the total count")
    return {
        "matrix": filtered,
        "linkage": linkage_matrix,
        "labels": labels,
        "coarse": coarse,
    }


def build_tree(
    vocabulary: np.ndarray,
    selected_indices: np.ndarray,
    frequencies: np.ndarray,
    linkage_matrix: np.ndarray,
    labels: np.ndarray,
) -> tuple[list[dict[str, Any]], list[list[int]]]:
    """Return laid-out dendrogram nodes and their selected-word positions."""
    leaf_count = len(selected_indices)
    if leaf_count == 0:
        return [], []
    descendants: list[list[int]] = [[index] for index in range(leaf_count)]
    nodes: list[dict[str, Any]] = []
    for leaf, original_index in enumerate(selected_indices):
        nodes.append(
            {
                "id": leaf,
                "left_id": None,
                "right_id": None,
                "is_leaf": True,
                "distance": 0.0,
                "keyword": str(vocabulary[original_index]),
                "keywords": [str(vocabulary[original_index])],
                "document_frequency": int(frequencies[original_index]),
                "cluster": int(labels[leaf]),
            }
        )
    for row_index, row in enumerate(linkage_matrix):
        left, right = int(row[0]), int(row[1])
        positions = descendants[left] + descendants[right]
        descendants.append(positions)
        nodes.append(
            {
                "id": leaf_count + row_index,
                "left_id": left,
                "right_id": right,
                "is_leaf": False,
                "distance": float(row[2]),
                "keyword": None,
                "keywords": [
                    str(vocabulary[selected_indices[position]]) for position in positions
                ],
                "document_frequency": None,
                "cluster": None,
            }
        )

    order = (
        leaves_list(linkage_matrix).tolist()
        if len(linkage_matrix)
        else list(range(leaf_count))
    )
    y_by_leaf = {leaf: float(position) for position, leaf in enumerate(order)}
    for node in nodes:
        if node["is_leaf"]:
            node["x"] = 0.0
            node["y"] = y_by_leaf[node["id"]]
        else:
            left = nodes[node["left_id"]]
            right = nodes[node["right_id"]]
            node["x"] = float(node["distance"])
            node["y"] = (float(left["y"]) + float(right["y"])) / 2.0
    return nodes, descendants


def aggregate_node_years(
    artifacts: dict[str, Any],
    selected_indices: np.ndarray,
    descendants: list[list[int]],
) -> dict[int, dict[int, int]]:
    """Count each paper once per node/year using sparse Boolean incidence."""
    if not descendants:
        return {}
    return _aggregate_group_years(
        artifacts,
        [(selected_indices, descendants)],
    )[0]


def _aggregate_group_years(
    artifacts: dict[str, Any],
    groupings: Sequence[tuple[np.ndarray, Sequence[Any]]],
) -> list[dict[int, dict[int, int]]]:
    """Count papers for every grouping in one pass over the incidence shards.

    Each grouping is a keyword index array plus the member positions of each
    node. A paper contributes once to a node when it contains any member
    keyword. Summing those yearly counts is the node's document total.
    """
    if not groupings:
        return []
    vocabulary_size = len(artifacts["vocabulary"])
    membership, widths = _group_membership(vocabulary_size, groupings)
    counts: dict[int, np.ndarray] = {}
    root = artifacts["root"]
    manifest = artifacts["manifest"]
    incidence_dir = root / manifest["incidence_dir"]
    for relative in manifest.get("incidence_parts", []):
        article_terms = sparse.load_npz(incidence_dir / relative).tocsr()
        years = np.load(
            incidence_dir / relative.replace(".npz", "_years.npy"),
            allow_pickle=False,
        ).astype(np.int32, copy=False)
        if article_terms.shape[0] != len(years):
            raise ValueError(f"Incidence rows and years do not match for {relative}")
        if article_terms.shape[1] != vocabulary_size:
            raise ValueError(
                f"Incidence width does not match the vocabulary for {relative}"
            )
        _add_presence_years(counts, article_terms, years, membership)
    results: list[dict[int, dict[int, int]]] = []
    offset = 0
    for width in widths:
        results.append(
            {
                node: {
                    year: int(values[offset + node])
                    for year, values in sorted(counts.items())
                }
                for node in range(width)
            }
        )
        offset += width
    return results


def _group_membership(
    vocabulary_size: int,
    groupings: Sequence[tuple[np.ndarray, Sequence[Any]]],
) -> tuple[sparse.csr_matrix, list[int]]:
    """Stack each grouping's keyword-to-node indicators into one sparse matrix."""
    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    widths: list[int] = []
    column = 0
    for selected_indices, descendants in groupings:
        selected = np.asarray(selected_indices, dtype=np.int64)
        widths.append(len(descendants))
        for positions in descendants:
            position_array = np.asarray(positions, dtype=np.int64)
            if position_array.size:
                row_parts.append(selected[position_array])
                column_parts.append(np.full(position_array.size, column, dtype=np.int64))
            column += 1
    rows = np.concatenate(row_parts) if row_parts else np.empty(0, dtype=np.int64)
    columns = (
        np.concatenate(column_parts) if column_parts else np.empty(0, dtype=np.int64)
    )
    membership = sparse.csr_matrix(
        (np.ones(rows.size, dtype=np.int64), (rows, columns)),
        shape=(vocabulary_size, column),
        dtype=np.int64,
    )
    return membership, widths


def _add_presence_years(
    counts: dict[int, np.ndarray],
    article_terms,
    years: np.ndarray,
    membership: sparse.csr_matrix,
) -> None:
    """Add one shard's per-node paper counts.

    Incidence is stored as int8. The product uses an int64 accumulator, so a
    node that unions more keywords than int8 can hold still becomes presence
    1 rather than wrapping to zero. Casting the shard to int64 first would
    copy every stored count without changing that result.
    """
    node_count = membership.shape[1]
    if node_count == 0:
        return
    unique_years = np.unique(years)
    for year in unique_years:
        counts.setdefault(int(year), np.zeros(node_count, dtype=np.int64))
    if membership.nnz == 0 or article_terms.shape[0] == 0:
        return
    presence = article_terms @ membership
    if presence.nnz == 0:
        return
    presence = presence.tocoo()
    year_ids = np.searchsorted(unique_years, years[presence.row])
    flat = year_ids.astype(np.int64) * node_count + presence.col.astype(np.int64)
    binned = np.bincount(flat, minlength=int(unique_years.size * node_count))
    binned = binned.reshape(unique_years.size, node_count)
    for index, year in enumerate(unique_years):
        counts[int(year)] += binned[index]


def _sparse_payload(matrix) -> dict[str, Any]:
    coo = matrix.tocoo()
    return {
        "shape": list(matrix.shape),
        "entries": [
            [int(row), int(column), int(value)]
            for row, column, value in zip(coo.row, coo.col, coo.data, strict=True)
        ],
    }


def load_cluster_artifacts(
    clusters_dir: str | Path,
    event_artifacts: dict[str, Any],
) -> dict[str, Any]:
    """Load cluster output and align its keywords to the event vocabulary."""
    root = Path(clusters_dir).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    keywords = np.load(root / manifest["keywords"], allow_pickle=False).astype(str)
    groups = np.load(root / manifest["groups_by_level"], allow_pickle=False)
    if groups.ndim != 2 or groups.shape[1] != len(keywords):
        raise ValueError("Cluster groups do not align with the cluster keyword vocabulary")
    level = int(manifest.get("coarse_level", 0))
    if level < 0 or level >= groups.shape[0]:
        raise ValueError("Cluster manifest coarse_level is outside groups_by_level")
    vocabulary = event_artifacts["vocabulary"]
    vocabulary_index = {str(keyword): index for index, keyword in enumerate(vocabulary)}
    if len(vocabulary_index) != len(vocabulary):
        raise ValueError("Event vocabulary contains duplicate keywords")
    missing = [keyword for keyword in keywords if keyword not in vocabulary_index]
    if missing:
        preview = ", ".join(str(keyword) for keyword in missing[:3])
        raise ValueError(f"Cluster keywords are absent from the event vocabulary: {preview}")
    return {
        "root": root,
        "manifest": manifest,
        "keywords": keywords,
        "groups": groups.astype(np.int32, copy=False),
        "level": level,
        "event_indices": np.asarray(
            [vocabulary_index[str(keyword)] for keyword in keywords],
            dtype=np.int64,
        ),
    }


def count_papers_by_year(db_path: str | Path) -> dict[int, int]:
    """Count articles per publication year.

    One grouped index scan of ``idx_publication_year``. The index holds every
    year, so SQLite never reads the article rows.
    """
    path = Path(db_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Database not found: {path}")
    uri = f"file:{quote(str(path))}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA temp_store = MEMORY")
        connection.execute("PRAGMA cache_size = -262144")
        connection.execute("PRAGMA mmap_size = 268435456")
        indexed = connection.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'index' AND name = 'idx_publication_year'
            """
        ).fetchone()
        indexed_by = " INDEXED BY idx_publication_year" if indexed else ""
        rows = connection.execute(
            f"""
            SELECT publication_year, COUNT(*)
            FROM articles{indexed_by}
            GROUP BY publication_year
            """
        )
        return {
            int(year): int(count)
            for year, count in rows
            if year is not None
        }
    finally:
        connection.close()


def papers_by_year_from_manifest(manifest: Mapping[str, Any]) -> dict[int, int]:
    return {
        int(year): int(count)
        for year, count in manifest.get("papers_by_year", {}).items()
    }


def _prepare_cluster_list(
    artifacts: dict[str, Any],
    clusters_dir: str | Path,
    allowed: set[str] | None,
) -> dict[str, Any] | None:
    """Align cluster members with the event vocabulary, or return nothing to list."""
    clustered = load_cluster_artifacts(clusters_dir, artifacts)
    if allowed is not None:
        keep = np.asarray(
            [str(keyword) in allowed for keyword in clustered["keywords"]],
            dtype=bool,
        )
        if not np.any(keep):
            return None
        clustered = {
            **clustered,
            "keywords": clustered["keywords"][keep],
            "groups": clustered["groups"][:, keep],
            "event_indices": clustered["event_indices"][keep],
        }
    indices = clustered["event_indices"]
    keywords = clustered["keywords"]
    groups = np.asarray(clustered["groups"][clustered["level"]], dtype=np.int32)
    if groups.size == 0:
        return None
    frequencies = np.asarray(artifacts["frequencies"])[indices]
    group_ids = sorted({int(group) for group in groups})
    member_positions = []
    for group in group_ids:
        positions = np.flatnonzero(groups == group)
        positions = positions[np.argsort(-frequencies[positions], kind="stable")]
        member_positions.append(positions)
    return {
        "keywords": keywords,
        "group_ids": group_ids,
        "member_positions": member_positions,
        "indices": indices,
        "descendants": member_positions,
    }


def _cluster_rows(
    prepared: Mapping[str, Any],
    yearly_counts: Mapping[int, Mapping[int, int]],
    papers_by_year: Mapping[int, int],
    total_documents: int,
) -> list[dict[str, Any]]:
    """Build cluster rows from yearly presence counts.

    A cluster's paper total is the sum of its yearly counts: every incidence
    row belongs to one year, and each of those counts is already a Boolean
    union of the member keywords.
    """
    rows = []
    for cluster_index, (group, positions) in enumerate(
        zip(prepared["group_ids"], prepared["member_positions"], strict=True)
    ):
        by_year = yearly_counts.get(cluster_index, {})
        papers = int(sum(by_year.values()))
        members = [str(prepared["keywords"][position]) for position in positions]
        rows.append(
            {
                "group": int(group),
                "keywords": members,
                "papers": papers,
                "share": papers / total_documents if total_documents else 0.0,
                "yearly": _yearly_payload(by_year, papers_by_year),
            }
        )
    rows.sort(key=lambda item: (-item["papers"], item["keywords"]))
    return rows


def build_cluster_list(
    artifacts: dict[str, Any],
    clusters_dir: str | Path,
    *,
    allowed: set[str] | None = None,
    papers_by_year: Mapping[int, int] | None = None,
    total_documents: int | None = None,
) -> list[dict[str, Any]]:
    """List every cluster by the share of documents containing any member keyword."""
    prepared = _prepare_cluster_list(artifacts, clusters_dir, allowed)
    if prepared is None:
        return []
    if papers_by_year is None:
        papers_by_year = papers_by_year_from_manifest(artifacts["manifest"])
    if total_documents is None:
        total_documents = int(sum(papers_by_year.values()))
    yearly_counts = aggregate_node_years(
        artifacts,
        prepared["indices"],
        prepared["descendants"],
    )
    return _cluster_rows(prepared, yearly_counts, papers_by_year, total_documents)


def load_link_distance_summary(
    visualizations_dir: str | Path,
    clusters: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Load new-link scatter data and attach each cluster's temporal series."""
    path = (
        Path(visualizations_dir).expanduser().resolve()
        / "cluster_link_distance_summary.csv"
    )
    yearly_by_group = {
        int(cluster["group"]): cluster
        for cluster in (clusters or [])
    }
    integer_fields = (
        "paper_count",
        "new_link_connected_count",
        "new_link_disconnected_count",
        "all_link_connected_count",
        "all_link_disconnected_count",
        "all_link_existing_count",
        "all_link_observation_count",
    )
    distance_fields = (
        "average_new_link_distance",
        "average_all_link_distance",
    )
    distribution_fields = (
        "new_link_distance_distribution",
        "all_link_distance_distribution",
        "outside_cluster_distance_distribution",
    )
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "cluster_id",
            "label",
            *integer_fields,
            *distance_fields,
            *distribution_fields,
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "New-link summary is missing columns: "
                + ", ".join(sorted(missing))
            )
        for source in reader:
            cluster_id = int(source["cluster_id"])
            cluster = yearly_by_group.get(cluster_id)
            row: dict[str, Any] = {
                "cluster_id": cluster_id,
                "label": source["label"],
                "keywords": [] if cluster is None else list(cluster["keywords"]),
                "yearly": [] if cluster is None else list(cluster["yearly"]),
            }
            row.update({field: int(source[field]) for field in integer_fields})
            for field in distance_fields:
                value = float(source[field])
                row[field] = value if math.isfinite(value) else None
            for field in distribution_fields:
                distribution = json.loads(source[field])
                row[field] = [
                    [int(distance), float(count)]
                    for distance, count in distribution
                ]
            rows.append(row)
    return rows


def _yearly_payload(
    counts: dict[int, int],
    papers_by_year: Mapping[int, int],
) -> list[dict[str, int | float]]:
    return [
        {
            "year": int(year),
            "papers": int(papers),
            "share": papers / papers_by_year.get(year, 0)
            if papers_by_year.get(year, 0)
            else 0.0,
        }
        for year, papers in sorted(counts.items())
    ]


def _genuine_event_keywords(
    events_dir: str | Path,
    filtered_dir: str | Path | None,
) -> set[str] | None:
    from openalex.analysis.filter_events import find_filtered_events, genuine_ngrams

    root = find_filtered_events(events_dir, filtered_dir)
    if root is None:
        return None
    return genuine_ngrams(root)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def build_website(
    events_dir: str | Path = "output/events",
    *,
    clusters_dir: str | Path | None = None,
    output_dir: str | Path = "output/website",
    top_keywords: int = DEFAULT_TOP_KEYWORDS,
    min_document_frequency: int = DEFAULT_MIN_DOCUMENT_FREQUENCY,
    max_dendrogram_keywords: int = DEFAULT_MAX_DENDROGRAM_KEYWORDS,
    cluster_similarity: float = DEFAULT_CLUSTER_SIMILARITY,
    filtered_dir: str | Path | None = None,
    db_path: str | Path | None = None,
    new_link_visualizations_dir: str | Path | None = None,
) -> dict[str, Any]:
    artifacts = load_event_artifacts(events_dir)
    vocabulary = artifacts["vocabulary"]
    frequencies = artifacts["frequencies"]
    allowed = _genuine_event_keywords(events_dir, filtered_dir)
    processed = int(artifacts["manifest"].get("processed_papers") or 0)
    if db_path is None:
        papers_by_year = papers_by_year_from_manifest(artifacts["manifest"])
        yearly_denominator = "processed_papers"
    else:
        papers_by_year = count_papers_by_year(db_path)
        yearly_denominator = "database"
    total_documents = int(sum(papers_by_year.values())) or processed
    ranked = np.argsort(-frequencies, kind="stable")
    if allowed is not None:
        ranked = np.asarray(
            [int(index) for index in ranked if str(vocabulary[int(index)]) in allowed],
            dtype=np.int64,
        )
    top_indices = ranked[:top_keywords]
    selected = select_keywords(
        artifacts["matrix"],
        frequencies,
        min_document_frequency=min_document_frequency,
        max_keywords=max_dendrogram_keywords,
        vocabulary=vocabulary,
        allowed=allowed,
    )
    clustered = cluster_keywords(
        artifacts["matrix"],
        selected,
        cluster_similarity=cluster_similarity,
    )
    nodes, descendants = build_tree(
        vocabulary,
        selected,
        frequencies,
        clustered["linkage"],
        clustered["labels"],
    )
    prepared_clusters = (
        _prepare_cluster_list(artifacts, clusters_dir, allowed)
        if clusters_dir is not None
        else None
    )
    # One incidence read serves the dendrogram and the cluster curves. Cluster
    # document totals are the sums of those yearly presence counts.
    groupings: list[tuple[np.ndarray, Sequence[Any]]] = []
    if descendants:
        groupings.append((selected, descendants))
    if prepared_clusters is not None:
        groupings.append(
            (prepared_clusters["indices"], prepared_clusters["descendants"])
        )
    grouped_years = _aggregate_group_years(artifacts, groupings) if groupings else []
    node_years: dict[int, dict[int, int]] = {}
    next_group = 0
    if descendants:
        node_years = grouped_years[next_group]
        next_group += 1
    for node in nodes:
        yearly = []
        for year, count in node_years.get(int(node["id"]), {}).items():
            denominator = papers_by_year.get(year, 0)
            yearly.append(
                {
                    "year": year,
                    "papers": count,
                    "share": count / denominator if denominator else 0.0,
                }
            )
        node["yearly"] = yearly
    if clusters_dir is None:
        cluster_list = None
    elif prepared_clusters is None:
        cluster_list = []
    else:
        cluster_list = _cluster_rows(
            prepared_clusters,
            grouped_years[next_group],
            papers_by_year,
            total_documents,
        )
    link_distances = (
        load_link_distance_summary(new_link_visualizations_dir, cluster_list)
        if new_link_visualizations_dir is not None
        else None
    )
    payload = {
        "meta": {
            "processed_papers": processed,
            "selected_keywords": len(selected),
            "clusters": len(set(int(value) for value in clustered["labels"])),
            "min_document_frequency": min_document_frequency,
            "cluster_similarity": cluster_similarity,
            "keyword_filter": "genuine" if allowed is not None else "vocabulary",
            "yearly_denominator": yearly_denominator,
            "total_documents": total_documents,
        },
        "top_keywords": [
            {
                "keyword": str(vocabulary[index]),
                "papers": int(frequencies[index]),
                "share": int(frequencies[index]) / processed if processed else 0.0,
            }
            for index in top_indices
        ],
        "dendrogram": {"nodes": nodes},
        "coarse_matrix": _sparse_payload(clustered["coarse"]),
        "cluster_list": cluster_list,
        "link_distances": link_distances,
    }
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected_html = {
        "index.html",
        "dendrogram.html",
        "clusters.html",
        "link-distances.html",
    }
    for stale_html in output.glob("*.html"):
        if stale_html.name not in expected_html:
            stale_html.unlink()
    for retired in ("graph.js",):
        retired_path = output / retired
        if retired_path.is_file():
            retired_path.unlink()
    site = files("openalex.website").joinpath("site")
    for asset in SITE_ASSETS:
        write_text(output / asset, site.joinpath(asset).read_text(encoding="utf-8"))
    write_text(output / ".nojekyll", "")
    write_text(
        output / "data.json",
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )
    return {
        "output_dir": str(output),
        "keywords": len(payload["top_keywords"]),
        "dendrogram_keywords": len(selected),
        "clusters": payload["meta"]["clusters"],
        "event_clusters": (
            0 if payload["cluster_list"] is None else len(payload["cluster_list"])
        ),
        "link_clusters": (
            0 if payload["link_distances"] is None else len(payload["link_distances"])
        ),
        "keyword_filter": payload["meta"]["keyword_filter"],
        "genuine_keywords": None if allowed is None else len(allowed),
        "yearly_denominator": yearly_denominator,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the static event-keyword website.")
    parser.add_argument("--events-dir", type=Path, default=Path("output/events"))
    parser.add_argument(
        "--clusters-dir",
        type=Path,
        default=None,
        help="Optional cluster-events output listed on clusters.html.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/website"))
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help=(
            "Read-only corpus used to divide each yearly frequency by that year's "
            "article count. One index scan of idx_publication_year."
        ),
    )
    parser.add_argument(
        "--filtered-dir",
        type=Path,
        default=None,
        help=(
            "filter-events output whose genuine keywords are shown. "
            "Defaults to a filtered_events directory beside --events-dir when classifications.csv exists."
        ),
    )
    parser.add_argument(
        "--new-link-visualizations-dir",
        type=Path,
        default=None,
        help=(
            "Optional visualize-new-links output used by link-distances.html."
        ),
    )
    parser.add_argument("--top-keywords", type=int, default=DEFAULT_TOP_KEYWORDS)
    parser.add_argument(
        "--min-document-frequency",
        type=int,
        default=DEFAULT_MIN_DOCUMENT_FREQUENCY,
    )
    parser.add_argument(
        "--max-dendrogram-keywords",
        type=int,
        default=DEFAULT_MAX_DENDROGRAM_KEYWORDS,
    )
    parser.add_argument(
        "--cluster-similarity",
        type=float,
        default=DEFAULT_CLUSTER_SIMILARITY,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.top_keywords < 1:
        raise SystemExit("--top-keywords must be positive")
    if args.min_document_frequency < 1:
        raise SystemExit("--min-document-frequency must be positive")
    if args.max_dendrogram_keywords < 1:
        raise SystemExit("--max-dendrogram-keywords must be positive")
    summary = build_website(
        args.events_dir,
        clusters_dir=args.clusters_dir,
        output_dir=args.output_dir,
        top_keywords=args.top_keywords,
        min_document_frequency=args.min_document_frequency,
        max_dendrogram_keywords=args.max_dendrogram_keywords,
        cluster_similarity=args.cluster_similarity,
        filtered_dir=args.filtered_dir,
        db_path=args.db_path,
        new_link_visualizations_dir=args.new_link_visualizations_dir,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
