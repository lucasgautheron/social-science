import json
import sqlite3

import numpy as np
import pytest
from scipy import sparse

from openalex.website.build import (
    aggregate_node_years,
    build_cluster_list,
    build_website,
    cluster_keywords,
    count_papers_by_year,
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


def test_website_pages_and_exact_paper_unions(tmp_path):
    event_dir = tmp_path / "events"
    site_dir = tmp_path / "site"
    write_event_fixture(event_dir)
    site_dir.mkdir()
    (site_dir / "progress.html").write_text("stale", encoding="utf-8")
    (site_dir / "app.js").write_text("stale keywords", encoding="utf-8")
    summary = build_website(
        event_dir,
        output_dir=site_dir,
        min_document_frequency=2,
        max_dendrogram_keywords=10,
        cluster_similarity=0.5,
    )
    assert summary["dendrogram_keywords"] == 3
    index = (site_dir / "index.html").read_text(encoding="utf-8")
    assert "dendrogram.html" in index
    assert "Top keywords" not in index
    assert (site_dir / "dendrogram.html").is_file()
    assert (site_dir / "clusters.html").is_file()
    assert (site_dir / "link-distances.html").is_file()
    assert not (site_dir / "app.js").exists()
    assert not (site_dir / "graph.html").exists()
    assert not (site_dir / "progress.html").exists()
    for page in ("dendrogram.html", "clusters.html", "link-distances.html"):
        assert "Top keywords" not in (site_dir / page).read_text(encoding="utf-8")
    payload = json.loads((site_dir / "data.json").read_text(encoding="utf-8"))
    assert "top_keywords" not in payload
    assert payload["cluster_list"] is None
    root = max(payload["dendrogram"]["nodes"], key=lambda node: len(node["keywords"]))
    assert set(root["keywords"]) == {"alpha", "beta", "gamma"}
    assert root["yearly"] == [
        {"year": 2020, "papers": 2, "share": 1.0},
        {"year": 2021, "papers": 2, "share": 2 / 3},
    ]


def test_link_distance_page_joins_plot_summary_to_temporal_curves(tmp_path):
    event_dir = tmp_path / "events"
    clusters_dir = tmp_path / "clusters"
    plots_dir = tmp_path / "plots"
    site_dir = tmp_path / "site"
    write_event_fixture(event_dir)
    write_cluster_fixture(clusters_dir)
    plots_dir.mkdir()
    (plots_dir / "cluster_link_distance_summary.csv").write_text(
        "cluster_id,label,paper_count,average_new_link_distance,"
        "new_link_connected_count,new_link_disconnected_count,"
        "average_all_link_distance,all_link_connected_count,"
        "all_link_disconnected_count,all_link_existing_count,"
        "all_link_observation_count,new_link_distance_distribution,"
        "all_link_distance_distribution,new_link_baseline_distance_distribution,"
        "all_link_baseline_distance_distribution,"
        "new_link_distance_by_year,all_link_distance_by_year\n"
        '0,beta,4,2.5,8,2,1.25,10,2,2,12,"[[1,8],[2,2]]","[[0,2],[2,8]]",'
        '"[[2,20],[3,5]]","[[1,4],[2,6]]",'
        '"[[2020,2.5],[2021,1.5]]","[[2020,1.25]]"\n'
        '1,gamma,2,nan,0,3,3.0,4,3,1,7,[],"[[0,1],[3,3]]",'
        '"[[3,2]]","[[1,1],[3,3]]",[],[]\n',
        encoding="utf-8",
    )
    (clusters_dir / "cluster_trends.csv").write_text(
        "level,group,compatible\n0,0,shock\n1,1,step\n",
        encoding="utf-8",
    )

    summary = build_website(
        event_dir,
        clusters_dir=clusters_dir,
        new_link_visualizations_dir=plots_dir,
        output_dir=site_dir,
        min_document_frequency=2,
        max_dendrogram_keywords=10,
    )

    assert summary["link_clusters"] == 2
    payload = json.loads((site_dir / "data.json").read_text(encoding="utf-8"))
    first, second = payload["link_distances"]
    assert first["label"] == "beta"
    assert first["keywords"] == ["beta", "alpha"]
    assert first["yearly"] == payload["cluster_list"][0]["yearly"]
    assert first["average_new_link_distance"] == 2.5
    assert first["new_link_distance_distribution"] == [[1, 8.0], [2, 2.0]]
    assert first["new_link_distance_by_year"] == [
        {"year": 2020, "distance": 2.5},
        {"year": 2021, "distance": 1.5},
    ]
    assert first["all_link_distance_by_year"] == [
        {"year": 2020, "distance": 1.25},
    ]
    assert second["new_link_distance_by_year"] == []
    assert first["new_link_baseline_distance_distribution"] == [
        [2, 20.0],
        [3, 5.0],
    ]
    assert first["all_link_baseline_distance_distribution"] == [
        [1, 4.0],
        [2, 6.0],
    ]
    assert second["average_new_link_distance"] is None
    assert first["cluster_type"] == "shock"
    assert second["cluster_type"] is None
    external = tmp_path / "cluster_trends.csv"
    external.write_text(
        "level,group,compatible\n0,0,step\n",
        encoding="utf-8",
    )
    build_website(
        event_dir,
        clusters_dir=clusters_dir,
        new_link_visualizations_dir=plots_dir,
        output_dir=site_dir,
        trends=external,
        min_document_frequency=2,
        max_dendrogram_keywords=10,
    )
    labeled = json.loads((site_dir / "data.json").read_text(encoding="utf-8"))
    assert labeled["link_distances"][0]["cluster_type"] == "step"
    with pytest.raises(ValueError, match="Cluster trends not found"):
        build_website(
            event_dir,
            clusters_dir=clusters_dir,
            new_link_visualizations_dir=plots_dir,
            output_dir=site_dir,
            trends=tmp_path / "missing-trends.csv",
            min_document_frequency=2,
            max_dendrogram_keywords=10,
        )
    script = (site_dir / "link-distances.js").read_text(encoding="utf-8")
    assert "scatter-point" in script
    assert "Connected-distance distribution" in script
    assert "Mean distance by year" in script
    assert "Year-matched new links outside clusters" in script
    assert "wasserstein_q_value" in script
    assert "Year-matched links outside this cluster" in script
    assert "All clusters" not in script
    assert "filteredRows()" in script
    assert "new_link_highlighted" not in script
    page = (site_dir / "link-distances.html").read_text(encoding="utf-8")
    assert 'id="cluster-search"' in page
    for cluster_type in (
        "trend increasing",
        "trend decreasing",
        "step",
        "shock",
        "bump increasing",
        "bump decreasing",
        "bump unknown",
    ):
        assert f'data-cluster-type="{cluster_type}"' in page
    assert "matchesType" in script
    assert "data-type" in script
    assert "observableRows()" in script


def test_cluster_list_counts_documents_once(tmp_path):
    event_dir = tmp_path / "events"
    clusters_dir = tmp_path / "clusters"
    site_dir = tmp_path / "site"
    write_event_fixture(event_dir)
    write_cluster_fixture(clusters_dir)
    artifacts = load_event_artifacts(event_dir)
    clusters = build_cluster_list(artifacts, clusters_dir)
    assert [item["keywords"] for item in clusters] == [["beta", "alpha"], ["gamma"]]
    assert clusters[0]["papers"] == 4
    assert clusters[0]["share"] == pytest.approx(4 / 5)
    assert clusters[0]["yearly"] == [
        {"year": 2020, "papers": 2, "share": 1.0},
        {"year": 2021, "papers": 2, "share": 2 / 3},
    ]
    assert clusters[1]["papers"] == 2
    assert clusters[1]["share"] == pytest.approx(2 / 5)
    assert clusters[1]["yearly"] == [
        {"year": 2020, "papers": 0, "share": 0.0},
        {"year": 2021, "papers": 2, "share": 2 / 3},
    ]

    (site_dir).mkdir()
    (site_dir / "graph.html").write_text("stale graph", encoding="utf-8")
    (site_dir / "graph.js").write_text("stale graph", encoding="utf-8")
    summary = build_website(
        event_dir,
        clusters_dir=clusters_dir,
        output_dir=site_dir,
        min_document_frequency=2,
        max_dendrogram_keywords=10,
    )
    assert summary["event_clusters"] == 2
    payload = json.loads((site_dir / "data.json").read_text(encoding="utf-8"))
    assert payload["cluster_list"][0]["papers"] == 4
    assert "canvas" not in (site_dir / "clusters.html").read_text(encoding="utf-8")
    assert not (site_dir / "graph.html").exists()
    assert not (site_dir / "graph.js").exists()
    page = (site_dir / "clusters.js").read_text(encoding="utf-8")
    assert "y(Number(item.share)" in page


def test_website_uses_genuine_keywords_when_available(tmp_path):
    event_dir = tmp_path / "events"
    clusters_dir = tmp_path / "clusters"
    site_dir = tmp_path / "site"
    write_event_fixture(event_dir)
    write_cluster_fixture(clusters_dir)
    filtered = tmp_path / "filtered_events"
    filtered.mkdir()
    (filtered / "classifications.csv").write_text(
        "ngram,label,reason\n"
        "alpha,genuine,topic\n"
        "beta,genuine,topic\n"
        "gamma,artefact,non-English\n",
        encoding="utf-8",
    )
    summary = build_website(
        event_dir,
        clusters_dir=clusters_dir,
        output_dir=site_dir,
        min_document_frequency=2,
        max_dendrogram_keywords=10,
    )
    assert summary["keyword_filter"] == "genuine"
    assert summary["genuine_keywords"] == 2
    assert summary["dendrogram_keywords"] == 2
    assert summary["event_clusters"] == 1
    payload = json.loads((site_dir / "data.json").read_text(encoding="utf-8"))
    root = max(payload["dendrogram"]["nodes"], key=lambda node: len(node["keywords"]))
    assert set(root["keywords"]) == {"alpha", "beta"}
    assert payload["cluster_list"][0]["keywords"] == ["beta", "alpha"]
    assert payload["cluster_list"][0]["papers"] == 4


def write_article_year_database(path, *, indexed=True):
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE articles (
            article_id INTEGER PRIMARY KEY,
            publication_year INTEGER,
            title TEXT
        )
        """
    )
    if indexed:
        connection.execute(
            "CREATE INDEX idx_publication_year ON articles (publication_year)"
        )
    connection.executemany(
        "INSERT INTO articles (publication_year, title) VALUES (?, ?)",
        [(2020, "paper")] * 10 + [(2021, "paper")] * 20 + [(None, "undated")],
    )
    connection.commit()
    connection.close()


def test_count_papers_by_year_scans_the_year_index(tmp_path):
    database = tmp_path / "articles.db"
    write_article_year_database(database)
    connection = sqlite3.connect(database)
    plan = connection.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT publication_year, COUNT(*)
        FROM articles INDEXED BY idx_publication_year
        GROUP BY publication_year
        """
    ).fetchall()
    connection.close()
    assert count_papers_by_year(database) == {2020: 10, 2021: 20}
    assert any("idx_publication_year" in " ".join(str(value) for value in row) for row in plan)


def test_count_papers_by_year_without_index(tmp_path):
    database = tmp_path / "articles.db"
    write_article_year_database(database, indexed=False)
    assert count_papers_by_year(database) == {2020: 10, 2021: 20}


def test_yearly_curves_use_database_article_counts(tmp_path):
    event_dir = tmp_path / "events"
    clusters_dir = tmp_path / "clusters"
    site_dir = tmp_path / "site"
    database = tmp_path / "articles.db"
    write_event_fixture(event_dir)
    write_cluster_fixture(clusters_dir)
    write_article_year_database(database)
    summary = build_website(
        event_dir,
        clusters_dir=clusters_dir,
        output_dir=site_dir,
        db_path=database,
        min_document_frequency=2,
        max_dendrogram_keywords=10,
    )
    assert summary["yearly_denominator"] == "database"
    payload = json.loads((site_dir / "data.json").read_text(encoding="utf-8"))
    assert payload["meta"]["yearly_denominator"] == "database"
    assert payload["meta"]["total_documents"] == 30
    root = max(payload["dendrogram"]["nodes"], key=lambda node: len(node["keywords"]))
    expected = [
        {"year": 2020, "papers": 2, "share": 0.2},
        {"year": 2021, "papers": 2, "share": 0.1},
    ]
    assert root["yearly"] == expected
    cluster = next(item for item in payload["cluster_list"] if item["group"] == 0)
    assert cluster["yearly"] == expected
    assert cluster["papers"] == 4
    assert cluster["share"] == pytest.approx(4 / 30)
    dendrogram = (site_dir / "dendrogram.js").read_text(encoding="utf-8")
    clusters_js = (site_dir / "clusters.js").read_text(encoding="utf-8")
    assert "y(Number(item.share)" in dendrogram
    assert "y(Number(item.share)" in clusters_js


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


def test_sharded_incidence_counts_each_paper_once(tmp_path):
    event_dir = tmp_path / "events"
    incidence = event_dir / "incidence"
    incidence.mkdir(parents=True)
    sparse.save_npz(
        incidence / "a.npz",
        sparse.csr_matrix([[1, 1], [1, 0], [0, 0]], dtype=np.int8),
    )
    sparse.save_npz(
        incidence / "b.npz",
        sparse.csr_matrix([[0, 1], [1, 1]], dtype=np.int8),
    )
    np.save(
        incidence / "a_years.npy",
        np.array([2020, 2020, 2022], dtype=np.int32),
        allow_pickle=False,
    )
    np.save(
        incidence / "b_years.npy",
        np.array([2021, 2020], dtype=np.int32),
        allow_pickle=False,
    )
    artifacts = {
        "root": event_dir,
        "vocabulary": np.array(["alpha", "beta"]),
        "manifest": {
            "incidence_dir": "incidence",
            "incidence_parts": ["a.npz", "b.npz"],
        },
    }
    counts = aggregate_node_years(
        artifacts,
        np.array([0, 1]),
        [[0, 1], [1]],
    )
    assert counts[0] == {2020: 3, 2021: 1, 2022: 0}
    assert counts[1] == {2020: 2, 2021: 1, 2022: 0}
    assert sum(counts[0].values()) == 4
    assert sum(counts[1].values()) == 3
    parallel = aggregate_node_years(
        artifacts,
        np.array([0, 1]),
        [[0, 1], [1]],
        workers=2,
    )
    assert parallel == counts
