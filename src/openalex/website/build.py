"""Build the event-keyword website."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import squareform

DEFAULT_CLUSTER_SIMILARITY = 0.5
DEFAULT_MIN_DOCUMENT_FREQUENCY = 10
DEFAULT_TOP_KEYWORDS = 100
DEFAULT_MAX_DENDROGRAM_KEYWORDS = 500
DEFAULT_MAX_GRAPH_KEYWORDS = 5_000
DEFAULT_MAX_GRAPH_EDGES = 10_000
SITE_ASSETS = (
    "index.html",
    "dendrogram.html",
    "graph.html",
    "app.js",
    "dendrogram.js",
    "graph.js",
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
        normalized = l2_normalize_cooccurrence(filtered)
        similarities = (normalized @ normalized.T).toarray()
        distances = np.clip(1.0 - similarities, 0.0, 2.0)
        np.fill_diagonal(distances, 0.0)
        linkage_matrix = linkage(
            squareform(distances, checks=False),
            method="complete",
        )
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
    vocabulary_size = len(artifacts["vocabulary"])
    rows: list[int] = []
    columns: list[int] = []
    for node_index, positions in enumerate(descendants):
        for position in positions:
            rows.append(int(selected_indices[position]))
            columns.append(node_index)
    membership = sparse.csr_matrix(
        (
            np.ones(len(rows), dtype=np.int64),
            (np.asarray(rows), np.asarray(columns)),
        ),
        shape=(vocabulary_size, len(descendants)),
    )
    counts: dict[int, np.ndarray] = {}
    root = artifacts["root"]
    incidence_dir = root / artifacts["manifest"]["incidence_dir"]
    for relative in artifacts["manifest"].get("incidence_parts", []):
        matrix_path = incidence_dir / relative
        years_path = incidence_dir / relative.replace(".npz", "_years.npy")
        article_terms = sparse.load_npz(matrix_path).tocsr()
        years = np.load(years_path, allow_pickle=False).astype(np.int32, copy=False)
        if article_terms.shape[0] != len(years):
            raise ValueError(f"Incidence rows and years do not match for {relative}")
        presence = (article_terms.astype(np.int64) @ membership).tocsr()
        presence.data[:] = 1
        for year in np.unique(years):
            values = np.asarray(presence[years == year].sum(axis=0)).ravel()
            counts.setdefault(int(year), np.zeros(len(descendants), dtype=np.int64))
            counts[int(year)] += values.astype(np.int64)
    return {
        node: {
            year: int(values[node])
            for year, values in sorted(counts.items())
        }
        for node in range(len(descendants))
    }


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
    """Load blockmodel output and align its keywords to the event vocabulary."""
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


def build_graph_payload(
    artifacts: dict[str, Any],
    clusters_dir: str | Path,
    *,
    max_keywords: int = DEFAULT_MAX_GRAPH_KEYWORDS,
    max_edges: int = DEFAULT_MAX_GRAPH_EDGES,
    allowed: set[str] | None = None,
) -> dict[str, Any]:
    """Build keyword and cluster graph views from blockmodel artifacts."""
    if max_keywords < 1:
        raise ValueError("max_keywords must be positive")
    if max_edges < 1:
        raise ValueError("max_edges must be positive")
    clustered = load_cluster_artifacts(clusters_dir, artifacts)
    if allowed is not None:
        keep = np.asarray(
            [str(keyword) in allowed for keyword in clustered["keywords"]],
            dtype=bool,
        )
        if not np.any(keep):
            raise ValueError("No cluster keywords remain after the genuine-event filter")
        clustered = {
            **clustered,
            "keywords": clustered["keywords"][keep],
            "groups": clustered["groups"][:, keep],
            "event_indices": clustered["event_indices"][keep],
        }
    all_indices = clustered["event_indices"]
    all_frequencies = artifacts["frequencies"][all_indices]
    order = np.argsort(-all_frequencies, kind="stable")[:max_keywords]
    event_indices = all_indices[order]
    keywords = clustered["keywords"][order]
    frequencies = all_frequencies[order].astype(np.int64, copy=False)
    groups = clustered["groups"][clustered["level"], order]

    all_matrix = artifacts["matrix"][all_indices][:, all_indices].tocsr()
    all_matrix.setdiag(0)
    all_matrix.eliminate_zeros()
    original_edges = int(sparse.triu(all_matrix, k=1).nnz)

    matrix = artifacts["matrix"][event_indices][:, event_indices].tocsr()
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    difference = (matrix - matrix.T).tocsr()
    difference.eliminate_zeros()
    if difference.nnz:
        raise ValueError("Event co-occurrence matrix must be symmetric")
    processed = int(artifacts["manifest"].get("processed_papers") or 0)
    positive_rows, positive_columns, positive_weights = _positive_npmi_edges(
        matrix,
        frequencies,
        processed,
    )
    layout_matrix = _symmetric_edge_matrix(
        len(keywords),
        positive_rows,
        positive_columns,
        positive_weights,
    )
    coordinates = _spectral_layout(layout_matrix)

    edge_order = np.lexsort(
        (positive_columns, positive_rows, -positive_weights)
    )
    edge_order = edge_order[:max_edges]
    rows = positive_rows[edge_order]
    columns = positive_columns[edge_order]
    weights = positive_weights[edge_order]

    group_ids = sorted(set(int(group) for group in groups))
    group_to_position = {group: position for position, group in enumerate(group_ids)}
    member_positions = [np.flatnonzero(groups == group) for group in group_ids]
    descendants = [[position] for position in range(len(keywords))]
    descendants.extend(positions.tolist() for positions in member_positions)
    yearly_counts = aggregate_node_years(artifacts, event_indices, descendants)
    papers_by_year = {
        int(year): int(count)
        for year, count in artifacts["manifest"].get("papers_by_year", {}).items()
    }

    keyword_nodes = []
    for position, keyword in enumerate(keywords):
        group = int(groups[position])
        keyword_nodes.append(
            {
                "id": f"keyword-{position}",
                "kind": "keyword",
                "keyword": str(keyword),
                "keywords": [str(keyword)],
                "group": group,
                "color": _cluster_color(group),
                "papers": int(frequencies[position]),
                "x": float(coordinates[position, 0]),
                "y": float(coordinates[position, 1]),
                "yearly": _yearly_payload(
                    yearly_counts.get(position, {}),
                    papers_by_year,
                ),
            }
        )

    cluster_nodes = []
    for cluster_position, (group, positions) in enumerate(
        zip(group_ids, member_positions, strict=True)
    ):
        member_frequencies = frequencies[positions].astype(np.float64)
        total_frequency = float(member_frequencies.sum())
        if total_frequency:
            center = np.average(coordinates[positions], axis=0, weights=member_frequencies)
        else:
            center = coordinates[positions].mean(axis=0)
        members = [str(keywords[position]) for position in positions]
        cluster_nodes.append(
            {
                "id": f"cluster-{group}",
                "kind": "cluster",
                "group": group,
                "color": _cluster_color(group),
                "papers": int(total_frequency),
                "keyword_count": len(members),
                "keywords": members,
                "x": float(center[0]),
                "y": float(center[1]),
                "yearly": _yearly_payload(
                    yearly_counts.get(len(keywords) + cluster_position, {}),
                    papers_by_year,
                ),
            }
        )

    membership = sparse.csr_matrix(
        (
            np.ones(len(groups), dtype=np.int64),
            (
                np.arange(len(groups), dtype=np.int64),
                np.asarray([group_to_position[int(group)] for group in groups]),
            ),
        ),
        shape=(len(groups), len(group_ids)),
    )
    cluster_frequencies, cluster_matrix = _cluster_incidence_statistics(
        artifacts,
        event_indices,
        membership,
    )
    cluster_rows, cluster_columns, cluster_weights = _positive_npmi_edges(
        cluster_matrix,
        cluster_frequencies,
        processed,
    )
    positive_cluster_edge_count = len(cluster_weights)
    cluster_order = np.lexsort(
        (cluster_columns, cluster_rows, -cluster_weights)
    )[:max_edges]
    cluster_rows = cluster_rows[cluster_order]
    cluster_columns = cluster_columns[cluster_order]
    cluster_weights = cluster_weights[cluster_order]
    for position, frequency in enumerate(cluster_frequencies):
        cluster_nodes[position]["document_frequency"] = int(frequency)
    cluster_edges = [
        {
            "source": int(row),
            "target": int(column),
            "weight": float(weight),
        }
        for row, column, weight in zip(
            cluster_rows,
            cluster_columns,
            cluster_weights,
            strict=True,
        )
    ]
    keyword_edges = [
        {"source": int(row), "target": int(column), "weight": float(weight)}
        for row, column, weight in zip(rows, columns, weights, strict=True)
    ]
    return {
        "level": int(clustered["level"]),
        "edge_weight": "positive NPMI",
        "layout_edges": "all positive-NPMI keyword edges before the display cap",
        "keyword": {"nodes": keyword_nodes, "edges": keyword_edges},
        "cluster": {"nodes": cluster_nodes, "edges": cluster_edges},
        "limits": {"keywords": int(max_keywords), "edges": int(max_edges)},
        "counts": {
            "original_keywords": int(len(all_indices)),
            "displayed_keywords": int(len(keywords)),
            "original_edges": original_edges,
            "positive_edges": int(len(positive_weights)),
            "displayed_edges": int(len(keyword_edges)),
            "displayed_clusters": int(len(cluster_nodes)),
            "positive_cluster_edges": int(positive_cluster_edge_count),
            "displayed_cluster_edges": int(len(cluster_edges)),
        },
    }


def _positive_npmi_edges(
    matrix: sparse.csr_matrix,
    frequencies: np.ndarray,
    processed_papers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return upper-triangle edges whose normalized PMI is strictly positive."""
    if processed_papers < 1:
        raise ValueError("processed_papers must be positive to compute NPMI")
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if frequencies.shape != (matrix.shape[0],):
        raise ValueError("Frequencies must align with the co-occurrence matrix")
    if np.any(frequencies < 0) or np.any(frequencies > processed_papers):
        raise ValueError("Document frequencies must be between zero and processed_papers")
    upper = sparse.triu(matrix, k=1, format="coo")
    counts = upper.data.astype(np.float64, copy=False)
    if np.any(counts < 0) or np.any(counts > processed_papers):
        raise ValueError("Co-occurrence counts must be between zero and processed_papers")
    endpoint_limits = np.minimum(frequencies[upper.row], frequencies[upper.col])
    if np.any(counts > endpoint_limits):
        raise ValueError("Co-occurrence counts cannot exceed endpoint document frequencies")
    if not len(counts):
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float64),
        )
    joint = counts / float(processed_papers)
    expected = (
        frequencies[upper.row]
        * frequencies[upper.col]
        / float(processed_papers * processed_papers)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(joint / expected)
        denominator = -np.log(joint)
        npmi = np.divide(
            pmi,
            denominator,
            out=np.zeros_like(pmi),
            where=denominator > 0,
        )
    npmi = np.clip(npmi, -1.0, 1.0)
    keep = np.isfinite(npmi) & (npmi > 0)
    return (
        upper.row[keep].astype(np.int32, copy=False),
        upper.col[keep].astype(np.int32, copy=False),
        npmi[keep],
    )


def _symmetric_edge_matrix(
    count: int,
    rows: np.ndarray,
    columns: np.ndarray,
    weights: np.ndarray,
) -> sparse.csr_matrix:
    return sparse.coo_matrix(
        (
            np.concatenate((weights, weights)),
            (np.concatenate((rows, columns)), np.concatenate((columns, rows))),
        ),
        shape=(count, count),
        dtype=np.float64,
    ).tocsr()


def _cluster_incidence_statistics(
    artifacts: dict[str, Any],
    selected_indices: np.ndarray,
    membership: sparse.csr_matrix,
) -> tuple[np.ndarray, sparse.csr_matrix]:
    """Return exact cluster document frequencies and paper co-occurrences."""
    cluster_count = membership.shape[1]
    frequencies = np.zeros(cluster_count, dtype=np.int64)
    cooccurrence = sparse.csr_matrix((cluster_count, cluster_count), dtype=np.int64)
    root = artifacts["root"]
    manifest = artifacts["manifest"]
    incidence_dir = root / manifest["incidence_dir"]
    for relative in manifest.get("incidence_parts", []):
        article_terms = sparse.load_npz(incidence_dir / relative).tocsr()
        if article_terms.shape[1] != len(artifacts["vocabulary"]):
            raise ValueError(f"Incidence width does not match the vocabulary for {relative}")
        presence = (
            article_terms[:, selected_indices].astype(np.int64) @ membership
        ).tocsr()
        presence.data[:] = 1
        frequencies += np.asarray(presence.getnnz(axis=0), dtype=np.int64)
        cooccurrence = (cooccurrence + presence.T @ presence).tocsr()
    return frequencies, cooccurrence


def _yearly_payload(
    counts: dict[int, int],
    papers_by_year: dict[int, int],
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


def _cluster_color(group: int) -> str:
    hue = (float(group) * 137.508) % 360.0
    return f"hsl({hue:.1f} 62% 43%)"


def _spectral_layout(matrix: sparse.csr_matrix) -> np.ndarray:
    """Lay out sparse connected components deterministically and pack them."""
    count = matrix.shape[0]
    if count == 0:
        return np.empty((0, 2), dtype=np.float64)
    component_count, labels = connected_components(matrix, directed=False)
    coordinates = np.zeros((count, 2), dtype=np.float64)
    columns = max(1, math.ceil(math.sqrt(component_count)))
    for component in range(component_count):
        positions = np.flatnonzero(labels == component)
        local = _component_layout(matrix[positions][:, positions])
        row, column = divmod(component, columns)
        coordinates[positions] = local + np.array([3.0 * column, 3.0 * row])
    coordinates -= coordinates.mean(axis=0)
    scale = float(np.max(np.abs(coordinates)))
    if scale:
        coordinates /= scale
    return coordinates


def _component_layout(matrix: sparse.csr_matrix) -> np.ndarray:
    count = matrix.shape[0]
    if count == 1:
        return np.zeros((1, 2), dtype=np.float64)
    if count == 2:
        return np.array([[-1.0, 0.0], [1.0, 0.0]])
    normalized = laplacian(matrix.astype(np.float64), normed=True)
    if count <= 32:
        _values, vectors = np.linalg.eigh(normalized.toarray())
        coordinates = vectors[:, 1:3]
    else:
        start = np.linspace(1.0, 2.0, count, dtype=np.float64)
        values, vectors = eigsh(
            normalized,
            k=3,
            which="SM",
            v0=start / np.linalg.norm(start),
        )
        vectors = vectors[:, np.argsort(values)]
        coordinates = vectors[:, 1:3]
    if coordinates.shape[1] == 1:
        coordinates = np.column_stack((coordinates[:, 0], np.zeros(count)))
    for axis in range(2):
        pivot = int(np.argmax(np.abs(coordinates[:, axis])))
        if coordinates[pivot, axis] < 0:
            coordinates[:, axis] *= -1
    coordinates -= coordinates.mean(axis=0)
    scale = float(np.max(np.abs(coordinates)))
    return coordinates / scale if scale else coordinates


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
    max_graph_keywords: int = DEFAULT_MAX_GRAPH_KEYWORDS,
    max_graph_edges: int = DEFAULT_MAX_GRAPH_EDGES,
    cluster_similarity: float = DEFAULT_CLUSTER_SIMILARITY,
    filtered_dir: str | Path | None = None,
) -> dict[str, Any]:
    artifacts = load_event_artifacts(events_dir)
    vocabulary = artifacts["vocabulary"]
    frequencies = artifacts["frequencies"]
    allowed = _genuine_event_keywords(events_dir, filtered_dir)
    processed = int(artifacts["manifest"].get("processed_papers") or 0)
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
    node_years = aggregate_node_years(artifacts, selected, descendants)
    papers_by_year = {
        int(year): int(count)
        for year, count in artifacts["manifest"].get("papers_by_year", {}).items()
    }
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
    payload = {
        "meta": {
            "processed_papers": processed,
            "selected_keywords": len(selected),
            "clusters": len(set(int(value) for value in clustered["labels"])),
            "min_document_frequency": min_document_frequency,
            "cluster_similarity": cluster_similarity,
            "keyword_filter": "genuine" if allowed is not None else "vocabulary",
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
        "graph": (
            build_graph_payload(
                artifacts,
                clusters_dir,
                max_keywords=max_graph_keywords,
                max_edges=max_graph_edges,
                allowed=allowed,
            )
            if clusters_dir is not None
            else None
        ),
    }
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected_html = {"index.html", "dendrogram.html", "graph.html"}
    for stale_html in output.glob("*.html"):
        if stale_html.name not in expected_html:
            stale_html.unlink()
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
        "graph_keywords": (
            payload["graph"]["counts"]["displayed_keywords"] if payload["graph"] else 0
        ),
        "keyword_filter": payload["meta"]["keyword_filter"],
        "genuine_keywords": None if allowed is None else len(allowed),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the static event-keyword website.")
    parser.add_argument("--events-dir", type=Path, default=Path("output/events"))
    parser.add_argument(
        "--clusters-dir",
        type=Path,
        default=None,
        help="Optional blockmodel output used by graph.html.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/website"))
    parser.add_argument(
        "--filtered-dir",
        type=Path,
        default=None,
        help=(
            "filter-events output whose genuine keywords are shown. "
            "Defaults to a filtered_events directory beside --events-dir when classifications.csv exists."
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
    parser.add_argument(
        "--max-graph-keywords",
        type=int,
        default=DEFAULT_MAX_GRAPH_KEYWORDS,
    )
    parser.add_argument(
        "--max-graph-edges",
        type=int,
        default=DEFAULT_MAX_GRAPH_EDGES,
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
    if args.max_graph_keywords < 1:
        raise SystemExit("--max-graph-keywords must be positive")
    if args.max_graph_edges < 1:
        raise SystemExit("--max-graph-edges must be positive")
    summary = build_website(
        args.events_dir,
        clusters_dir=args.clusters_dir,
        output_dir=args.output_dir,
        top_keywords=args.top_keywords,
        min_document_frequency=args.min_document_frequency,
        max_dendrogram_keywords=args.max_dendrogram_keywords,
        max_graph_keywords=args.max_graph_keywords,
        max_graph_edges=args.max_graph_edges,
        cluster_similarity=args.cluster_similarity,
        filtered_dir=args.filtered_dir,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
