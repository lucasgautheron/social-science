"""Build the two-page event-keyword website."""

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
from scipy.spatial.distance import squareform

DEFAULT_CLUSTER_SIMILARITY = 0.5
DEFAULT_MIN_DOCUMENT_FREQUENCY = 10
DEFAULT_TOP_KEYWORDS = 100
DEFAULT_MAX_DENDROGRAM_KEYWORDS = 500
SITE_ASSETS = ("index.html", "dendrogram.html", "app.js", "dendrogram.js", "style.css")


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
) -> np.ndarray:
    """Select frequent words with a nonzero filtered co-occurrence direction."""
    candidates = np.flatnonzero(frequencies >= min_document_frequency)
    if not len(candidates):
        return candidates
    ranked = candidates[np.argsort(-frequencies[candidates], kind="stable")]
    ranked = ranked[:max_keywords]
    filtered = matrix[ranked][:, ranked]
    norms = np.sqrt(np.asarray(filtered.multiply(filtered).sum(axis=1)).ravel())
    return ranked[norms > 0]


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
    norms = np.sqrt(np.asarray(filtered.multiply(filtered).sum(axis=1)).ravel())
    normalized = filtered.multiply(1.0 / norms[:, None]).tocsr()
    if count == 1:
        linkage_matrix = np.empty((0, 4), dtype=float)
        labels = np.array([1], dtype=np.int32)
    else:
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


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def build_website(
    events_dir: str | Path = "output/events",
    *,
    output_dir: str | Path = "output/website",
    top_keywords: int = DEFAULT_TOP_KEYWORDS,
    min_document_frequency: int = DEFAULT_MIN_DOCUMENT_FREQUENCY,
    max_dendrogram_keywords: int = DEFAULT_MAX_DENDROGRAM_KEYWORDS,
    cluster_similarity: float = DEFAULT_CLUSTER_SIMILARITY,
) -> dict[str, Any]:
    artifacts = load_event_artifacts(events_dir)
    vocabulary = artifacts["vocabulary"]
    frequencies = artifacts["frequencies"]
    processed = int(artifacts["manifest"].get("processed_papers") or 0)
    top_indices = np.argsort(-frequencies, kind="stable")[:top_keywords]
    selected = select_keywords(
        artifacts["matrix"],
        frequencies,
        min_document_frequency=min_document_frequency,
        max_keywords=max_dendrogram_keywords,
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
    }
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected_html = {"index.html", "dendrogram.html"}
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
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the static event-keyword website.")
    parser.add_argument("--events-dir", type=Path, default=Path("output/events"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/website"))
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
        output_dir=args.output_dir,
        top_keywords=args.top_keywords,
        min_document_frequency=args.min_document_frequency,
        max_dendrogram_keywords=args.max_dendrogram_keywords,
        cluster_similarity=args.cluster_similarity,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
