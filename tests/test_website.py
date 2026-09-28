import json

import numpy as np
from scipy import sparse

from openalex.website.build import (
    aggregate_node_years,
    build_website,
    cluster_keywords,
    load_event_artifacts,
    select_keywords,
)


def write_event_fixture(root):
    root.mkdir()
    incidence = root / "incidence"
    incidence.mkdir()
    vocabulary = np.array(["alpha", "beta", "gamma", "rare"], dtype=np.str_)
    article_terms = sparse.csr_matrix(
        [
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 1, 1],
        ],
        dtype=np.int8,
    )
    matrix = (article_terms.T @ article_terms).tocsr()
    frequencies = np.asarray(article_terms.getnnz(axis=0), dtype=np.int64)
    sparse.save_npz(root / "event_cooccurrence.npz", matrix)
    np.save(root / "event_vocabulary.npy", vocabulary, allow_pickle=False)
    np.save(root / "event_document_frequency.npy", frequencies, allow_pickle=False)
    sparse.save_npz(incidence / "part_000001_000.npz", article_terms)
    np.save(
        incidence / "part_000001_000_years.npy",
        np.array([2020, 2020, 2021, 2021], dtype=np.int32),
        allow_pickle=False,
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_version": 1,
                "vocabulary": "event_vocabulary.npy",
                "cooccurrence": "event_cooccurrence.npz",
                "document_frequency": "event_document_frequency.npy",
                "incidence_dir": "incidence",
                "incidence_parts": ["part_000001_000.npz"],
                "processed_papers": 4,
                "papers_by_year": {"2020": 2, "2021": 2},
            }
        ),
        encoding="utf-8",
    )


def test_complete_linkage_and_coarse_matrix_preserve_counts(tmp_path):
    event_dir = tmp_path / "events"
    write_event_fixture(event_dir)
    artifacts = load_event_artifacts(event_dir)
    selected = select_keywords(
        artifacts["matrix"],
        artifacts["frequencies"],
        min_document_frequency=2,
        max_keywords=10,
    )
    result = cluster_keywords(
        artifacts["matrix"],
        selected,
        cluster_similarity=0.5,
    )
    assert len(selected) == 3
    assert result["linkage"].shape == (2, 4)
    assert result["coarse"].sum() == result["matrix"].sum()


def test_website_has_two_pages_and_exact_paper_unions(tmp_path):
    event_dir = tmp_path / "events"
    site_dir = tmp_path / "site"
    write_event_fixture(event_dir)
    site_dir.mkdir()
    (site_dir / "progress.html").write_text("stale", encoding="utf-8")
    summary = build_website(
        event_dir,
        output_dir=site_dir,
        min_document_frequency=2,
        max_dendrogram_keywords=10,
        cluster_similarity=0.5,
    )
    assert summary["dendrogram_keywords"] == 3
    assert (site_dir / "index.html").is_file()
    assert (site_dir / "dendrogram.html").is_file()
    assert not (site_dir / "progress.html").exists()
    payload = json.loads((site_dir / "data.json").read_text(encoding="utf-8"))
    root = max(payload["dendrogram"]["nodes"], key=lambda node: len(node["keywords"]))
    assert set(root["keywords"]) == {"alpha", "beta", "gamma"}
    assert root["yearly"] == [
        {"year": 2020, "papers": 2, "share": 1.0},
        {"year": 2021, "papers": 2, "share": 1.0},
    ]


def test_large_cluster_union_does_not_overflow(tmp_path):
    event_dir = tmp_path / "events"
    incidence = event_dir / "incidence"
    incidence.mkdir(parents=True)
    width = 256
    sparse.save_npz(
        incidence / "part.npz",
        sparse.csr_matrix(np.ones((1, width), dtype=np.int8)),
    )
    np.save(
        incidence / "part_years.npy",
        np.array([2024], dtype=np.int32),
        allow_pickle=False,
    )
    artifacts = {
        "root": event_dir,
        "vocabulary": np.array([f"word-{index}" for index in range(width)]),
        "manifest": {
            "incidence_dir": "incidence",
            "incidence_parts": ["part.npz"],
        },
    }
    counts = aggregate_node_years(
        artifacts,
        np.arange(width),
        [list(range(width))],
    )
    assert counts == {0: {2024: 1}}
