import csv
import hashlib
import json
import sqlite3

import numpy as np
import pytest
from scipy import sparse

from openalex.analysis import new_links
from openalex.analysis.network import build_parser as build_network_parser
from openalex.analysis.new_links import (
    _bidirectional_exact_bfs,
    _distances,
    _finish_cluster_reservoir,
    _grouped_exact_bfs,
    _sample_cluster_observations,
    _sample_links,
    build_new_links,
    load_new_links,
)
from openalex.analysis.new_links import (
    build_parser as build_new_links_parser,
)
from openalex.cli import COMMANDS
from openalex.visualizations.new_links import (
    _benjamini_hochberg,
    _compare_distributions,
    _integer_histogram,
    _resample_histogram,
    _residual_highlights,
    build_cluster_link_plots,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_corpus(path):
    articles = [
        (1, 2019),
        (2, 2019),
        (3, 2020),
        (4, 2020),
        (5, 2020),
        (6, 2021),
        (7, 2021),
        (8, 2021),
        (9, 2021),
        (10, 2021),
        (11, 2019),
        (12, 2019),
        (13, 2019),
        (14, 2020),
        (15, 2021),
    ]
    authors = {
        1: [1, 2],
        2: [2, 3],
        3: [1, 3],
        4: [1, 3],
        5: [4, 5],
        6: [3, 4],
        7: [3, 4],
        8: [1, 4],
        9: [1, 4],
        10: [20, 21, 22],
        11: [10, 11],
        12: [11, 12],
        13: [12, 13],
        14: [10, 13],
        15: [1, 3],
    }
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE articles(
                article_id INTEGER PRIMARY KEY,
                publication_year INTEGER
            );
            CREATE TABLE articles_authors(
                article_id INTEGER,
                author_id INTEGER,
                PRIMARY KEY (article_id, author_id)
            );
            """
        )
        connection.executemany("INSERT INTO articles VALUES (?, ?)", articles)
        connection.executemany(
            "INSERT INTO articles_authors VALUES (?, ?)",
            [
                (article_id, author_id)
                for article_id, paper_authors in authors.items()
                for author_id in paper_authors
            ],
        )


def create_event_artifacts(events, clusters, *, include_article_ids=True):
    events.mkdir()
    incidence = events / "incidence"
    incidence.mkdir()
    vocabulary = np.array(["alpha", "beta"], dtype=np.str_)
    article_ids = np.array([3, 4, 6, 8, 9, 10, 15], dtype=np.int64)
    article_terms = sparse.csr_matrix(
        [
            [1, 1],
            [1, 0],
            [0, 1],
            [1, 1],
            [1, 1],
            [1, 0],
            [1, 0],
        ],
        dtype=np.int8,
    )
    sparse.save_npz(events / "event_cooccurrence.npz", article_terms.T @ article_terms)
    np.save(events / "event_vocabulary.npy", vocabulary, allow_pickle=False)
    np.save(
        events / "event_document_frequency.npy",
        np.asarray(article_terms.getnnz(axis=0), dtype=np.int64),
        allow_pickle=False,
    )
    sparse.save_npz(incidence / "part_000001.npz", article_terms)
    np.save(
        incidence / "part_000001_years.npy",
        np.array([2020, 2020, 2021, 2021, 2021, 2021, 2021], dtype=np.int32),
        allow_pickle=False,
    )
    if include_article_ids:
        np.save(
            incidence / "part_000001_article_ids.npy",
            article_ids,
            allow_pickle=False,
        )
    (events / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_version": 1,
                "vocabulary": "event_vocabulary.npy",
                "cooccurrence": "event_cooccurrence.npz",
                "document_frequency": "event_document_frequency.npy",
                "incidence_dir": "incidence",
                "incidence_parts": ["part_000001.npz"],
            }
        ),
        encoding="utf-8",
    )

    clusters.mkdir()
    np.save(clusters / "keywords.npy", vocabulary, allow_pickle=False)
    np.save(
        clusters / "groups_by_level.npy",
        np.array([[0, 1]], dtype=np.int32),
        allow_pickle=False,
    )
    (clusters / "manifest.json").write_text(
        json.dumps(
            {
                "coarse_level": 0,
                "keywords": "keywords.npy",
                "groups_by_level": "groups_by_level.npy",
            }
        ),
        encoding="utf-8",
    )


def pair_rows(author_ids, arrays):
    return {
        (int(author_ids[first]), int(author_ids[second])): (
            int(distance),
            int(cluster),
        )
        for first, second, distance, cluster in zip(
            arrays["author_i"],
            arrays["author_j"],
            arrays["distance"],
            arrays["cluster_id"],
            strict=True,
        )
    }


def test_new_links_distances_clusters_and_read_only(tmp_path):
    database = tmp_path / "articles.db"
    events = tmp_path / "events"
    clusters = tmp_path / "clusters"
    output = tmp_path / "new_links"
    create_corpus(database)
    create_event_artifacts(events, clusters)
    before = digest(database)

    build_new_links(
        database,
        events,
        clusters,
        output,
        max_authors=2,
        fetch_size=1,
    )
    assert digest(database) == before

    author_ids, links_2020 = load_new_links(output, 2020)
    rows_2020 = pair_rows(author_ids, links_2020)
    assert rows_2020[(1, 3)] == (2, 0)
    assert rows_2020[(4, 5)] == (-1, -1)
    assert rows_2020[(10, 13)] == (3, -1)

    _, links_2021 = load_new_links(output, 2021)
    rows_2021 = pair_rows(author_ids, links_2021)
    assert (1, 2) not in rows_2021
    assert rows_2021[(3, 4)] == (-1, -1)
    assert rows_2021[(1, 4)] == (-1, -1)
    assert 20 not in author_ids

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_version"] == 7
    assert manifest["baseline_sample"] == 2_000
    assert manifest["completed_years"] == [2019, 2020, 2021]
    assert manifest["clustered_link_counts"]["2020"] == 1
    assert manifest["hyperauthored_papers"]["2021"] == 1
    stats_2020 = manifest["distance_stats"]["2020"]
    assert stats_2020["connected_pairs"] == 2
    assert stats_2020["bidirectional_searches"] == 2
    assert stats_2020["grouped_searches"] == 0
    assert stats_2020["visited_nodes"] > 0
    assert stats_2020["inspected_edges"] > 0
    assert manifest["sampling_stats"]["2020"]["no_cluster_population"] == 2
    assert manifest["sampling_stats"]["2020"]["no_cluster_sample"] == 2
    assert np.all(links_2020["sampling_weight"] == 1)
    with np.load(output / "cluster_metadata.npz", allow_pickle=False) as metadata:
        assert metadata["paper_count"].tolist() == [6, 4]
        assert metadata["label"].tolist() == ["alpha", "beta"]
    with np.load(output / "cluster_years" / "2020.npz", allow_pickle=False) as stats:
        assert stats["connected_pair_observations"].tolist() == [2, 1]
        assert stats["disconnected_pair_observations"].tolist() == [0, 0]
        assert stats["new_connected_pair_observations"].tolist() == [2, 1]
        assert stats["reservoir_acceptances"].tolist() == [2, 1]
        assert stats["connected_distance_sum"].tolist() == [4, 2]
        assert stats["existing_pair_observations"].tolist() == [0, 0]
        assert stats["reference_cluster_id"].tolist() == [0, 1, 0]
        assert stats["reference_distance"].tolist() == [2, 2, 2]
        assert stats["reference_sampling_weight"].tolist() == [1, 1, 1]
    with np.load(output / "cluster_years" / "2021.npz", allow_pickle=False) as stats:
        assert stats["connected_pair_observations"].tolist() == [1, 0]
        assert stats["disconnected_pair_observations"].tolist() == [2, 3]
        assert stats["new_connected_pair_observations"].tolist() == [0, 0]
        assert stats["reservoir_acceptances"].tolist() == [1, 0]
        assert stats["connected_distance_sum"].tolist() == [1, 0]
        assert stats["existing_pair_observations"].tolist() == [1, 0]
    with np.load(
        output / "cluster_distance_reservoir.npz", allow_pickle=False
    ) as reservoir:
        assert reservoir["population"].tolist() == [3, 1]
        assert reservoir["offsets"].tolist() == [0, 3, 4]
        assert reservoir["distances"].tolist() == [2, 2, 1, 2]
        assert reservoir["years"].tolist() == [2020, 2020, 2021, 2020]
    assert not (output / "scratch").exists()


def test_resume_complete_needs_no_inputs(tmp_path):
    database = tmp_path / "articles.db"
    events = tmp_path / "events"
    clusters = tmp_path / "clusters"
    output = tmp_path / "new_links"
    create_corpus(database)
    create_event_artifacts(events, clusters)
    build_new_links(database, events, clusters, output, max_authors=2)
    year_path = output / "years" / "2020.npz"
    before = digest(year_path)
    modified = year_path.stat().st_mtime_ns
    database.unlink()

    build_new_links(database, events, clusters, output, max_authors=2, resume=True)

    assert digest(year_path) == before
    assert year_path.stat().st_mtime_ns == modified


def test_old_event_artifacts_explain_required_regeneration(tmp_path):
    database = tmp_path / "articles.db"
    events = tmp_path / "events"
    clusters = tmp_path / "clusters"
    create_corpus(database)
    create_event_artifacts(events, clusters, include_article_ids=False)

    with pytest.raises(ValueError, match="article-id sidecar"):
        build_new_links(
            database,
            events,
            clusters,
            tmp_path / "new_links",
            max_authors=2,
        )


def test_new_links_command_is_registered():
    assert COMMANDS["new-links"] == "openalex.analysis.new_links"
    assert COMMANDS["visualize-new-links"] == "openalex.visualizations.new_links"


def test_exact_bfs_has_no_distance_cap():
    graph = sparse.diags(
        [np.ones(8, dtype=np.int8), np.ones(8, dtype=np.int8)],
        offsets=[-1, 1],
        shape=(9, 9),
        format="csr",
    )
    distance = _grouped_exact_bfs(
        graph.indptr.astype(np.int64),
        graph.indices.astype(np.int32),
        np.array([0], dtype=np.int32),
        np.array([8], dtype=np.int32),
    )
    assert distance.tolist() == [8]
    bidirectional = _bidirectional_exact_bfs(
        graph.indptr.astype(np.int64),
        graph.indices.astype(np.int32),
        np.array([0], dtype=np.int32),
        np.array([8], dtype=np.int32),
    )
    assert bidirectional.tolist() == [8]


def test_hybrid_uses_grouped_bfs_for_many_targets():
    graph = sparse.diags(
        [np.ones(5, dtype=np.int8), np.ones(5, dtype=np.int8)],
        offsets=[-1, 1],
        shape=(6, 6),
        format="csr",
    )
    parent = np.zeros(6, dtype=np.int32)
    distances, stats = _distances(
        graph,
        parent,
        np.zeros(4, dtype=np.int32),
        np.array([2, 3, 4, 5], dtype=np.int32),
        grouped_bfs_min_targets=4,
    )
    assert distances.tolist() == [2, 3, 4, 5]
    assert stats["grouped_searches"] == 1
    assert stats["bidirectional_searches"] == 0


def test_parallel_distances_match_single_worker():
    graph = sparse.diags(
        [np.ones(8, dtype=np.int8), np.ones(8, dtype=np.int8)],
        offsets=[-1, 1],
        shape=(9, 9),
        format="csr",
    )
    parent = np.zeros(9, dtype=np.int32)
    left = np.array([0, 0, 0, 0, 1, 2, 3, 4], dtype=np.int32)
    right = np.array([2, 3, 4, 5, 4, 5, 6, 7], dtype=np.int32)

    single, single_stats = _distances(
        graph, parent, left, right, grouped_bfs_min_targets=4, distance_workers=1
    )
    parallel, parallel_stats = _distances(
        graph, parent, left, right, grouped_bfs_min_targets=4, distance_workers=3
    )

    assert np.array_equal(parallel, single)
    for key in (
        "bidirectional_searches",
        "connected_pairs",
        "disconnected_pairs",
        "distinct_sources",
        "grouped_searches",
        "inspected_edges",
        "visited_nodes",
    ):
        assert parallel_stats[key] == single_stats[key]
    assert parallel_stats["workers"] > 1


def test_bidirectional_bfs_matches_scipy_shortest_paths():
    rng = np.random.default_rng(7)
    upper = sparse.triu(
        sparse.csr_matrix(rng.random((24, 24)) < 0.08, dtype=np.int8),
        k=1,
        format="csr",
    )
    chain = sparse.diags(
        [np.ones(23, dtype=np.int8), np.ones(23, dtype=np.int8)],
        offsets=[-1, 1],
        shape=(24, 24),
        format="csr",
    )
    graph = (upper + upper.T + chain).tocsr()
    graph.data[:] = 1
    sources, targets = np.triu_indices(24, k=1)
    observed = _bidirectional_exact_bfs(
        graph.indptr.astype(np.int64),
        graph.indices.astype(np.int32),
        sources.astype(np.int32),
        targets.astype(np.int32),
    )
    expected = sparse.csgraph.shortest_path(
        graph, directed=False, unweighted=True
    )[sources, targets]
    assert np.array_equal(observed, expected.astype(np.int32))


def test_network_commands_default_to_sixteen_authors():
    assert build_network_parser().parse_args([]).max_authors == 16
    assert (
        build_new_links_parser()
        .parse_args(["--events-dir", "events", "--clusters-dir", "clusters"])
        .max_authors
        == 16
    )
    assert (
        build_new_links_parser()
        .parse_args(["--events-dir", "events", "--clusters-dir", "clusters"])
        .no_cluster_sample
        == 2_000
    )
    assert (
        build_new_links_parser()
        .parse_args(["--events-dir", "events", "--clusters-dir", "clusters"])
        .cluster_sample
        == 2_000
    )
    assert (
        build_new_links_parser()
        .parse_args(["--events-dir", "events", "--clusters-dir", "clusters"])
        .baseline_sample
        == 2_000
    )
    assert (
        build_new_links_parser()
        .parse_args(["--events-dir", "events", "--clusters-dir", "clusters"])
        .distance_workers
        == 16
    )


def test_no_cluster_sampling_is_fixed_size_weighted_and_reproducible():
    clusters = np.array([-1] * 20 + [3, 4], dtype=np.int32)
    first, weights, stats = _sample_links(clusters, 5, 5, 17, 2020)
    second, second_weights, second_stats = _sample_links(
        clusters, 5, 5, 17, 2020
    )

    assert np.array_equal(first, second)
    assert np.array_equal(weights, second_weights)
    assert stats == second_stats
    assert np.count_nonzero(clusters[first] == -1) == 5
    assert set(first[-2:]) == {20, 21}
    assert np.all(weights[clusters[first] == -1] == 4)
    assert np.all(weights[clusters[first] != -1] == 1)
    assert stats["no_cluster_inclusion_probability"] == 0.25
    assert stats["stored_links"] == 7


def test_cluster_link_sampling_is_capped_and_weighted_per_cluster():
    clusters = np.array([0] * 20 + [1] * 6, dtype=np.int32)
    retained, weights, stats = _sample_links(clusters, 0, 5, 11, 2020)

    assert np.count_nonzero(clusters[retained] == 0) == 5
    assert np.count_nonzero(clusters[retained] == 1) == 5
    assert np.all(weights[clusters[retained] == 0] == 4)
    assert np.allclose(weights[clusters[retained] == 1], 1.2)
    assert stats["cluster_strata"]["0"]["population"] == 20
    assert stats["cluster_strata"]["0"]["sample"] == 5
    assert stats["cluster_strata"]["1"]["population"] == 6


def test_cluster_paper_observations_use_reproducible_global_reservoirs():
    event_keys = np.arange(1, 27, dtype=np.uint64)
    event_clusters = np.array([0] * 20 + [1] * 6, dtype=np.int32)
    empty_reservoirs = [
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
    ]
    empty_reservoir_years = [
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
    ]
    observations, keys, clusters, existing, kept, kept_years, populations = (
        _sample_cluster_observations(
            event_keys,
            event_clusters,
            np.empty(0, dtype=np.uint64),
            event_keys,
            np.ones(event_keys.size, dtype=bool),
            cluster_count=2,
            cluster_sample=5,
            sampling_seed=13,
            year=2020,
            cluster_reservoirs=empty_reservoirs,
            cluster_reservoir_years=empty_reservoir_years,
            cluster_observation_population=np.zeros(2, dtype=np.int64),
        )
    )
    assert observations["new_connected_pair_observations"].tolist() == [20, 6]
    assert observations["reservoir_acceptances"].tolist() == [5, 5]
    assert populations.tolist() == [20, 6]
    assert np.bincount(clusters, minlength=2).tolist() == [5, 5]

    reservoirs, reservoir_years = _finish_cluster_reservoir(
        kept,
        kept_years,
        keys,
        clusters,
        existing,
        np.unique(keys),
        np.full(np.unique(keys).size, 2, dtype=np.int32),
        cluster_count=2,
        year=2020,
    )
    assert [values.tolist() for values in reservoirs] == [[2] * 5, [2] * 5]
    assert [values.tolist() for values in reservoir_years] == [
        [2020] * 5,
        [2020] * 5,
    ]

    second_keys = event_keys + 100
    first_result = _sample_cluster_observations(
        second_keys,
        event_clusters,
        np.empty(0, dtype=np.uint64),
        second_keys,
        np.ones(second_keys.size, dtype=bool),
        cluster_count=2,
        cluster_sample=5,
        sampling_seed=13,
        year=2021,
        cluster_reservoirs=reservoirs,
        cluster_reservoir_years=reservoir_years,
        cluster_observation_population=populations,
    )
    second_result = _sample_cluster_observations(
        second_keys,
        event_clusters,
        np.empty(0, dtype=np.uint64),
        second_keys,
        np.ones(second_keys.size, dtype=bool),
        cluster_count=2,
        cluster_sample=5,
        sampling_seed=13,
        year=2021,
        cluster_reservoirs=reservoirs,
        cluster_reservoir_years=reservoir_years,
        cluster_observation_population=populations,
    )
    for first, second in zip(first_result[1:4], second_result[1:4], strict=True):
        assert np.array_equal(first, second)
    for first, second in zip(first_result[4], second_result[4], strict=True):
        assert np.array_equal(first, second)
    for first, second in zip(first_result[5], second_result[5], strict=True):
        assert np.array_equal(first, second)
    assert np.array_equal(first_result[6], second_result[6])

    (
        _,
        new_keys,
        new_clusters,
        new_existing,
        kept,
        kept_years,
        populations,
    ) = first_result
    reservoirs, reservoir_years = _finish_cluster_reservoir(
        kept,
        kept_years,
        new_keys,
        new_clusters,
        new_existing,
        np.unique(new_keys),
        np.full(np.unique(new_keys).size, 2, dtype=np.int32),
        cluster_count=2,
        year=2021,
    )
    assert populations.tolist() == [40, 12]
    assert [values.size for values in reservoirs] == [5, 5]
    assert all(values.size == 5 for values in reservoir_years)
    estimated_sums = [
        population / values.size * values.sum()
        for population, values in zip(populations, reservoirs, strict=True)
    ]
    assert estimated_sums == pytest.approx([80, 24])


def test_year_matched_nonparametric_comparison_is_reproducible():
    cluster = [
        {-1: 1.0, 2: 9.0},
        {-1: 9.0, 4: 1.0},
    ]
    reference = [
        {-1: 5.0, 3: 5.0},
        {-1: 1.0, 5: 9.0},
    ]
    first = _compare_distributions(
        cluster,
        reference,
        np.array([9, 1], dtype=np.int64),
        np.array([1, 9], dtype=np.int64),
        bootstrap_replicates=99,
        permutation_replicates=99,
        seed_components=[17, 0, 0],
    )
    second = _compare_distributions(
        cluster,
        reference,
        np.array([9, 1], dtype=np.int64),
        np.array([1, 9], dtype=np.int64),
        bootstrap_replicates=99,
        permutation_replicates=99,
        seed_components=[17, 0, 0],
    )

    assert first.baseline_histogram == pytest.approx({3: 9.0, 5: 1.0})
    assert first.disconnection_probability == pytest.approx(0.5)
    assert first.baseline_disconnection_probability == pytest.approx(0.3)
    assert first.disconnection_risk_difference == pytest.approx(0.2)
    assert first.mean_distance_shift == pytest.approx(-1.0)
    assert first.wasserstein_distance == pytest.approx(1.0)
    assert first.disconnection_ci == second.disconnection_ci
    assert first.wasserstein_ci == second.wasserstein_ci
    assert first.disconnection_p_value == second.disconnection_p_value
    assert first.wasserstein_p_value == second.wasserstein_p_value
    assert _benjamini_hochberg([0.01, 0.03, 0.2]).tolist() == pytest.approx(
        [0.03, 0.045, 0.2]
    )


def test_inference_uses_exact_rates_and_actual_draw_counts():
    comparison = _compare_distributions(
        [{-1: 1.0, 2: 1.0}],
        [{-1: 50.0, 3: 50.0}],
        np.array([99], dtype=np.int64),
        np.array([1], dtype=np.int64),
        bootstrap_replicates=199,
        permutation_replicates=99,
        seed_components=[23, 0, 0],
    )

    assert comparison.disconnection_risk_difference == pytest.approx(-0.49)
    assert comparison.disconnection_ci[1] < 0
    rng = np.random.default_rng(3)
    assert sum(_resample_histogram(rng, {2: 1_250.0, 3: 1_250.0}).values()) == 2_500
    assert sum(_integer_histogram({2: 2_500.0, 3: 2_500.0}).values()) == 5_000


def test_unsampled_links_still_grow_the_cumulative_graph(tmp_path):
    database = tmp_path / "articles.db"
    events = tmp_path / "events"
    clusters = tmp_path / "clusters"
    output = tmp_path / "new_links"
    create_corpus(database)
    create_event_artifacts(events, clusters)

    build_new_links(
        database,
        events,
        clusters,
        output,
        max_authors=2,
        no_cluster_sample=0,
    )

    author_ids, links_2019 = load_new_links(output, 2019)
    assert links_2019["author_i"].size == 0
    _, links_2020 = load_new_links(output, 2020)
    assert pair_rows(author_ids, links_2020) == {(1, 3): (2, 0)}
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["link_counts"]["2019"] == 5
    assert manifest["stored_link_counts"]["2019"] == 0
    assert manifest["sampling_stats"]["2019"]["no_cluster_population"] == 5


def test_resume_restores_unsampled_edges_from_scratch(tmp_path, monkeypatch):
    database = tmp_path / "articles.db"
    events = tmp_path / "events"
    clusters = tmp_path / "clusters"
    output = tmp_path / "new_links"
    create_corpus(database)
    create_event_artifacts(events, clusters)
    original = new_links._process_year

    def interrupt_on_2020(*args, **kwargs):
        if args[1] == 2020:
            raise RuntimeError("interrupted")
        return original(*args, **kwargs)

    monkeypatch.setattr(new_links, "_process_year", interrupt_on_2020)
    with pytest.raises(RuntimeError, match="interrupted"):
        build_new_links(
            database,
            events,
            clusters,
            output,
            max_authors=2,
            no_cluster_sample=1,
        )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["completed_years"] == [2019]
    assert (output / "scratch" / "completed_edges" / "2019.npz").exists()
    assert (output / "scratch" / "cluster_reservoirs" / "2019.npz").exists()

    monkeypatch.setattr(new_links, "_process_year", original)
    build_new_links(
        database,
        events,
        clusters,
        output,
        max_authors=2,
        no_cluster_sample=1,
        resume=True,
    )
    author_ids, links_2020 = load_new_links(output, 2020)
    assert pair_rows(author_ids, links_2020)[(1, 3)] == (2, 0)
    assert not (output / "scratch").exists()


def test_cluster_link_visualizations_use_paper_counts_and_existing_zeros(tmp_path):
    database = tmp_path / "articles.db"
    events = tmp_path / "events"
    clusters = tmp_path / "clusters"
    links = tmp_path / "new_links"
    plots = tmp_path / "plots"
    create_corpus(database)
    create_event_artifacts(events, clusters)
    build_new_links(database, events, clusters, links, max_authors=2)

    summary = build_cluster_link_plots(
        links,
        plots,
        dpi=50,
        bootstrap_replicates=29,
        permutation_replicates=29,
        inference_seed=7,
    )

    with summary.open(newline="", encoding="utf-8") as handle:
        rows = {int(row["cluster_id"]): row for row in csv.DictReader(handle)}
    assert int(rows[0]["paper_count"]) == 6
    assert float(rows[0]["average_new_link_distance"]) == pytest.approx(2)
    assert float(rows[0]["average_all_link_distance"]) == pytest.approx(5 / 3)
    assert int(rows[0]["all_link_existing_count"]) == 1
    assert int(rows[0]["all_link_new_connected_count"]) == 2
    assert int(rows[0]["all_link_distance_sample_count"]) == 3
    assert int(rows[0]["all_link_disconnected_count"]) == 2
    assert json.loads(rows[0]["new_link_distance_distribution"]) == [[2, 1.0]]
    assert json.loads(rows[0]["new_link_distance_by_year"]) == [[2020, 2.0]]
    assert json.loads(rows[0]["all_link_distance_by_year"]) == [
        [2020, 2.0],
        [2021, 1.0],
    ]
    assert json.loads(rows[0]["all_link_distance_distribution"]) == [
        [1, 1.0],
        [2, 2.0],
    ]
    assert json.loads(rows[0]["new_link_baseline_distance_distribution"]) == [
        [3, 1.0]
    ]
    assert json.loads(rows[0]["all_link_baseline_distance_distribution"]) == []
    assert float(rows[0]["new_link_mean_distance_shift"]) == pytest.approx(-1)
    assert float(rows[0]["new_link_wasserstein_distance"]) == pytest.approx(1)
    assert 0 <= float(rows[0]["new_link_wasserstein_p_value"]) <= 1
    assert 0 <= float(rows[0]["new_link_wasserstein_q_value"]) <= 1
    assert float(rows[0]["all_link_repeat_probability"]) == pytest.approx(1 / 3)
    assert float(rows[1]["average_all_link_distance"]) == pytest.approx(2)
    assert int(rows[1]["all_link_disconnected_count"]) == 3
    assert rows[0]["label"] == "alpha"
    assert rows[0]["new_link_highlighted"] == "True"
    assert (plots / "cluster_new_link_distance.png").stat().st_size > 0
    assert (plots / "cluster_all_link_distance.png").stat().st_size > 0


def test_residual_highlights_choose_both_extremes_in_each_size_quantile():
    sizes = np.arange(1, 41, dtype=np.int64)
    averages = 2 * np.log10(sizes)
    positive = np.array([2, 12, 22, 32])
    negative = np.array([7, 17, 27, 37])
    averages[positive] += 10
    averages[negative] -= 10

    selection = _residual_highlights(sizes, averages)

    assert set(selection.highlighted) == set(np.concatenate((positive, negative)))
    assert set(selection.size_quantile[selection.highlighted]) == {1, 2, 3, 4}
    assert np.all(selection.residuals[positive] > 0)
    assert np.all(selection.residuals[negative] < 0)
