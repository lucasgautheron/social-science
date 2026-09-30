import csv
import json

import numpy as np
import pytest
from scipy import sparse

from openalex.analysis.cluster_events import (
    build_parser,
    cluster_event_keywords,
    cooccurrence_adjacency,
    cut_for_cluster_count,
    fit_dendrogram,
    project_level,
)
from openalex.cli import COMMANDS


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
    (root / "events.csv").write_text(
        "ngram,fold_change,max_year,min_year\n"
        "gamma,4,2021,2020\n"
        "alpha,3,2020,2021\n"
        "beta,2,2021,2020\n",
        encoding="utf-8",
    )


def test_adjacency_drops_diagonal_and_rejects_asymmetric_counts():
    matrix = sparse.csr_matrix(
        [
            [5, 1, 0],
            [1, 4, 2],
            [0, 2, 3],
        ],
        dtype=np.int64,
    )
    adjacency = cooccurrence_adjacency(matrix, np.array([0, 1, 2]))
    assert adjacency.diagonal().tolist() == [0, 0, 0]
    assert adjacency.sum() == 6

    asymmetric = matrix.tolil()
    asymmetric[0, 1] = 9
    with pytest.raises(ValueError, match="symmetric"):
        cooccurrence_adjacency(asymmetric.tocsr(), np.array([0, 1, 2]))


def test_project_level_compacts_nested_block_ids():
    levels = (
        np.array([4, 4, 9], dtype=np.int64),
        np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype=np.int64),
    )
    assert project_level(levels, 0).tolist() == [0, 0, 1]
    assert project_level(levels, 1).tolist() == [0, 0, 0]


def write_genuine_classifications(events_dir, rows, directory=None):
    filtered = events_dir.parent / "filtered_events" if directory is None else directory
    filtered.mkdir()
    lines = ["ngram,label,reason", *[f"{ngram},{label},because" for ngram, label in rows]]
    (filtered / "classifications.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_cluster_events_uses_genuine_keywords_when_available(tmp_path):
    events = tmp_path / "events"
    output = tmp_path / "clusters"
    write_event_fixture(events)
    write_genuine_classifications(
        events,
        [("alpha", "genuine"), ("beta", "genuine"), ("gamma", "artefact")],
    )
    summary = cluster_event_keywords(events, output)
    assert summary["keywords"] == 2
    assert summary["keyword_filter"] == "genuine"
    assert np.load(output / "keywords.npy", allow_pickle=False).astype(str).tolist() == ["alpha", "beta"]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["keyword_filter"] == "genuine"


def test_cluster_events_all_keeps_the_full_vocabulary(tmp_path):
    events = tmp_path / "events"
    output = tmp_path / "clusters"
    write_event_fixture(events)
    write_genuine_classifications(events, [("alpha", "genuine")])
    summary = cluster_event_keywords(events, output, source="all")
    assert summary["keywords"] == 4
    assert summary["keyword_filter"] == "all"


def test_cluster_events_command_is_registered():
    assert COMMANDS["cluster-events"] == "openalex.analysis.cluster_events"


def test_cluster_events_defaults_to_the_dendrogram(tmp_path):
    args = build_parser().parse_args([])
    assert args.n_clusters == 20

    events = tmp_path / "events"
    output = tmp_path / "clusters"
    write_event_fixture(events)
    summary = cluster_event_keywords(events, output)
    assert summary["method"] == "complete-linkage-cosine"
    assert summary["keywords"] == 3
    assert summary["groups"] == 3
    assert summary["target_clusters"] == 20
    assert summary["cluster_similarity"] == pytest.approx(1.0)
    assert summary["levels"] == 1

    with (output / "cluster_by_year.csv").open(newline="", encoding="utf-8") as handle:
        yearly = list(csv.DictReader(handle))
    observed = {
        (int(row["level"]), int(row["group"]), int(row["year"])): int(row["papers"])
        for row in yearly
    }
    keywords = np.load(output / "keywords.npy", allow_pickle=False).astype(str)
    groups = np.load(output / "groups_by_level.npy", allow_pickle=False)[0]
    group_of = {str(keyword): int(group) for keyword, group in zip(keywords, groups, strict=True)}
    assert observed[(0, group_of["alpha"], 2020)] == 2
    assert observed[(0, group_of["beta"], 2020)] == 1
    assert observed[(0, group_of["beta"], 2021)] == 1
    assert observed[(0, group_of["gamma"], 2021)] == 2
    assert len({group_of["alpha"], group_of["beta"], group_of["gamma"]}) == 3


def test_dendrogram_cut_meets_the_requested_cluster_count(tmp_path):
    events = tmp_path / "events"
    output = tmp_path / "clusters"
    write_event_fixture(events)
    summary = cluster_event_keywords(events, output, n_clusters=2)
    assert summary["groups"] == 2
    assert summary["cluster_similarity"] == pytest.approx(4 / np.sqrt(30))

    keywords = np.load(output / "keywords.npy", allow_pickle=False).astype(str)
    groups = np.load(output / "groups_by_level.npy", allow_pickle=False)[0]
    group_of = {str(keyword): int(group) for keyword, group in zip(keywords, groups, strict=True)}
    assert group_of["alpha"] == group_of["beta"]
    assert group_of["gamma"] != group_of["alpha"]

    coarse = sparse.load_npz(output / "coarse" / "level_0.npz")
    assert coarse.toarray().tolist() == [[2, 1], [1, 0]]

    with (output / "cluster_by_year.csv").open(newline="", encoding="utf-8") as handle:
        yearly = list(csv.DictReader(handle))
    observed = {
        (int(row["level"]), int(row["group"]), int(row["year"])): int(row["papers"])
        for row in yearly
    }
    assert observed[(0, group_of["alpha"], 2020)] == 2
    assert observed[(0, group_of["alpha"], 2021)] == 1
    assert observed[(0, group_of["gamma"], 2021)] == 2

    with (output / "keyword_groups.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_keyword = {row["keyword"]: row for row in rows}
    assert by_keyword["alpha"]["fold_change"] == "3"
    assert "rare" not in by_keyword

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["method"] == "complete-linkage-cosine"
    assert manifest["linkage"] == "complete"
    assert manifest["profile"] == "l2-normalized-cooccurrence-cosine"
    assert manifest["target_clusters"] == 2
    assert manifest["diagonal"] == "removed"
    assert "degree_corrected" not in manifest

    merged = cluster_event_keywords(events, tmp_path / "one-cluster", n_clusters=1)
    assert merged["groups"] == 1
    assert merged["cluster_similarity"] == pytest.approx(0.2)


def test_cut_uses_the_next_finer_partition_when_heights_tie():
    linkage = np.array(
        [
            [0, 1, 0.2, 2],
            [2, 3, 0.2, 2],
            [4, 5, 0.9, 4],
        ],
        dtype=float,
    )
    skipped = cut_for_cluster_count(linkage, 3)
    assert skipped.clusters == 4
    assert skipped.similarity == pytest.approx(1.0)
    assert skipped.distance == pytest.approx(0.0)

    exact = cut_for_cluster_count(linkage, 2)
    assert exact.clusters == 2
    assert exact.similarity == pytest.approx(0.8)
    assert exact.distance == pytest.approx(0.2)

    scale = np.sqrt(0.75)
    rows = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, scale, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.5, scale],
        ]
    )
    fitted = fit_dendrogram(sparse.csr_matrix(rows), n_clusters=3)
    assert fitted.cluster_similarity == pytest.approx(1.0)
    assert int(fitted.levels[0].max()) + 1 == 4

    paired = fit_dendrogram(sparse.csr_matrix(rows), n_clusters=2)
    assert paired.cluster_similarity == pytest.approx(0.5)
    labels = paired.levels[0]
    assert int(labels.max()) + 1 == 2
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_dendrogram_rejects_a_nonpositive_target(tmp_path):
    events = tmp_path / "events"
    write_event_fixture(events)
    with pytest.raises(ValueError, match="n-clusters"):
        cluster_event_keywords(events, tmp_path / "clusters", n_clusters=0)
