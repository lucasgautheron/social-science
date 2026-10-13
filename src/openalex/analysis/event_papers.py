"""Map event-incidence papers to event clusters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from openalex.website.build import load_event_artifacts


def load_event_paper_clusters(
    events_dir: Path, clusters_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return a compact paper-to-cluster mapping from event incidence shards."""
    artifacts = load_event_artifacts(events_dir)
    cluster_manifest = _read_json(clusters_dir / "manifest.json")
    level = int(cluster_manifest["coarse_level"])
    keywords = np.load(
        clusters_dir / cluster_manifest["keywords"], allow_pickle=False
    ).astype(str)
    groups_by_level = np.load(
        clusters_dir / cluster_manifest["groups_by_level"], allow_pickle=False
    )
    if level < 0 or level >= groups_by_level.shape[0]:
        raise ValueError("Cluster manifest coarse_level is outside groups_by_level")
    if groups_by_level.shape[1] != keywords.size:
        raise ValueError("Cluster keywords and group assignments do not align")

    vocabulary = np.asarray(artifacts["vocabulary"]).astype(str)
    vocabulary_lookup = {
        keyword: index for index, keyword in enumerate(vocabulary)
    }
    try:
        selected = np.asarray(
            [vocabulary_lookup[keyword] for keyword in keywords],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ValueError(
            f"Cluster keyword {exc.args[0]!r} is absent from event vocabulary"
        ) from exc
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
        article_ids = np.load(ids_path, allow_pickle=False).astype(
            np.int64, copy=False
        )
        article_years = np.load(years_path, allow_pickle=False).astype(
            np.int32, copy=False
        )
        if (
            article_terms.shape[0] != article_ids.size
            or article_terms.shape[0] != article_years.size
        ):
            raise ValueError(
                f"Incidence rows, article ids, and years do not match for {relative}"
            )
        if article_terms.shape[1] != vocabulary.size:
            raise ValueError(
                f"Incidence width does not match the vocabulary for {relative}"
            )
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
        raise ValueError(
            "One event paper is associated with multiple publication years"
        )
    return event_ids, offsets, cluster_rows, event_years, cluster_labels, level


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
