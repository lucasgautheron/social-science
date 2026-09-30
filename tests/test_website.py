import json

import numpy as np
import pytest
from scipy import sparse

import openalex.website.build as website_build
from openalex.website.build import (
    aggregate_node_years,
    build_graph_payload,
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
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 0, 0],
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
        np.array([2020, 2020, 2021, 2021, 2021], dtype=np.int32),
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
                "processed_papers": 5,
                "papers_by_year": {"2020": 2, "2021": 3},
            }
        ),
        encoding="utf-8",
    )


def write_cluster_fixture(root):
    root.mkdir()
    np.save(
        root / "keywords.npy",
        np.array(["alpha", "beta", "gamma"], dtype=np.str_),
        allow_pickle=False,
    )
    np.save(
        root / "groups_by_level.npy",
        np.array([[0, 0, 1], [0, 0, 0]], dtype=np.int32),
        allow_pickle=False,
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "method": "degree-corrected-assortative-sbm",
                "keywords": "keywords.npy",
                "groups_by_level": "groups_by_level.npy",
                "coarse_level": 0,
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


def test_website_has_three_pages_and_exact_paper_unions(tmp_path):
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
    assert (site_dir / "graph.html").is_file()
    assert not (site_dir / "progress.html").exists()
    payload = json.loads((site_dir / "data.json").read_text(encoding="utf-8"))
    assert payload["graph"] is None
    root = max(payload["dendrogram"]["nodes"], key=lambda node: len(node["keywords"]))
    assert set(root["keywords"]) == {"alpha", "beta", "gamma"}
    assert root["yearly"] == [
        {"year": 2020, "papers": 2, "share": 1.0},
        {"year": 2021, "papers": 2, "share": 2 / 3},
    ]


def test_graph_modes_colors_sizes_edges_and_barycenters(tmp_path, monkeypatch):
    event_dir = tmp_path / "events"
    clusters_dir = tmp_path / "clusters"
    site_dir = tmp_path / "site"
    write_event_fixture(event_dir)
    write_cluster_fixture(clusters_dir)
    artifacts = load_event_artifacts(event_dir)
    original_layout = website_build._spectral_layout
    layout_nnz = []

    def capture_layout(matrix):
        layout_nnz.append(matrix.nnz)
        return original_layout(matrix)

    monkeypatch.setattr(website_build, "_spectral_layout", capture_layout)
    graph = build_graph_payload(
        artifacts,
        clusters_dir,
        max_keywords=3,
        max_edges=1,
    )

    assert graph["counts"] == {
        "original_keywords": 3,
        "displayed_keywords": 3,
        "original_edges": 2,
        "positive_edges": 2,
        "displayed_edges": 1,
        "displayed_clusters": 2,
        "positive_cluster_edges": 1,
        "displayed_cluster_edges": 1,
    }
    assert layout_nnz == [4]
    assert graph["layout_edges"] == "all positive-NPMI keyword edges before the display cap"
    assert graph["keyword"]["edges"][0]["source"] == 0
    assert graph["keyword"]["edges"][0]["target"] == 1
    assert graph["keyword"]["edges"][0]["weight"] == pytest.approx(
        np.log(1.25) / -np.log(0.4)
    )
    keyword_nodes = graph["keyword"]["nodes"]
    cluster_nodes = graph["cluster"]["nodes"]
    assert keyword_nodes[0]["color"] == keyword_nodes[1]["color"] == cluster_nodes[0]["color"]
    assert keyword_nodes[2]["color"] == cluster_nodes[1]["color"]
    assert cluster_nodes[0]["papers"] == 6
    assert cluster_nodes[0]["document_frequency"] == 4
    assert cluster_nodes[0]["keywords"] == ["beta", "alpha"]
    assert cluster_nodes[0]["yearly"] == [
        {"year": 2020, "papers": 2, "share": 1.0},
        {"year": 2021, "papers": 2, "share": 2 / 3},
    ]
    expected_x = (4 * keyword_nodes[0]["x"] + 2 * keyword_nodes[1]["x"]) / 6
    expected_y = (4 * keyword_nodes[0]["y"] + 2 * keyword_nodes[1]["y"]) / 6
    assert cluster_nodes[0]["x"] == expected_x
    assert cluster_nodes[0]["y"] == expected_y
    assert graph["cluster"]["edges"][0]["weight"] == pytest.approx(
        np.log(1.25) / -np.log(0.4)
    )

    limited = build_graph_payload(
        artifacts,
        clusters_dir,
        max_keywords=2,
        max_edges=1,
    )
    assert [node["keyword"] for node in limited["keyword"]["nodes"]] == ["beta", "alpha"]
    assert limited["counts"]["displayed_keywords"] == 2

    summary = build_website(
        event_dir,
        clusters_dir=clusters_dir,
        output_dir=site_dir,
        min_document_frequency=2,
        max_dendrogram_keywords=10,
        max_graph_keywords=3,
        max_graph_edges=1,
    )
    assert summary["graph_keywords"] == 3
    payload = json.loads((site_dir / "data.json").read_text(encoding="utf-8"))
    assert payload["graph"]["counts"]["displayed_edges"] == 1
    assert "canvas" in (site_dir / "graph.html").read_text(encoding="utf-8")


def test_npmi_discards_zero_and_negative_edges():
    matrix = sparse.csr_matrix(
        [
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=np.int64,
    )
    rows, columns, weights = website_build._positive_npmi_edges(
        matrix,
        np.array([3, 2, 3], dtype=np.int64),
        4,
    )
    assert rows.size == columns.size == weights.size == 0


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
