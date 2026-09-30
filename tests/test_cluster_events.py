import csv
import json

import numpy as np
import pytest
from scipy import sparse

from openalex.analysis.cluster_events import (
    BlockmodelFit,
    cluster_event_keywords,
    cooccurrence_adjacency,
    fit_assortative,
    project_level,
)
from openalex.cli import COMMANDS
from openalex.website.build import load_event_artifacts


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


def fake_fit(adjacency, *, restarts, seed):
    assert restarts == 2
    assert seed == 7
    assert adjacency.shape == (3, 3)
    return BlockmodelFit(
        levels=(
            np.array([4, 4, 9], dtype=np.int64),
            np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype=np.int64),
        ),
        description_length=1.25,
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


def test_cluster_events_count_each_paper_once(tmp_path):
    events = tmp_path / "events"
    output = tmp_path / "clusters"
    write_event_fixture(events)
    summary = cluster_event_keywords(
        events,
        output,
        restarts=2,
        seed=7,
        fit=fake_fit,
    )
    assert summary["keywords"] == 3
    assert summary["groups"] == 2
    assert summary["levels"] == 2
    assert summary["description_length_nats"] == 1.25

    artifacts = load_event_artifacts(events)
    selected = np.array([0, 1, 2])
    adjacency = cooccurrence_adjacency(artifacts["matrix"], selected)
    coarse = sparse.load_npz(output / "coarse" / "level_0.npz")
    assert int(coarse.sum()) == int(adjacency.sum())
    assert coarse.toarray().tolist() == [[2, 1], [1, 0]]
    merged = sparse.load_npz(output / "coarse" / "level_1.npz")
    assert int(merged.sum()) == int(adjacency.sum())

    with (output / "cluster_by_year.csv").open(newline="", encoding="utf-8") as handle:
        yearly = list(csv.DictReader(handle))
    observed = {
        (int(row["level"]), int(row["group"]), int(row["year"])): int(row["papers"])
        for row in yearly
    }
    assert observed[(0, 0, 2020)] == 2
    assert observed[(0, 0, 2021)] == 1
    assert observed[(0, 1, 2021)] == 2
    assert observed[(1, 0, 2020)] == 2
    assert observed[(1, 0, 2021)] == 2

    with (output / "keyword_groups.csv").open(newline="", encoding="utf-8") as handle:
        groups = list(csv.DictReader(handle))
    by_keyword = {row["keyword"]: row for row in groups}
    assert by_keyword["alpha"]["group"] == "0"
    assert by_keyword["beta"]["group"] == "0"
    assert by_keyword["gamma"]["group"] == "1"
    assert by_keyword["alpha"]["fold_change"] == "3"
    assert "rare" not in by_keyword

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["method"] == "degree-corrected-assortative-sbm"
    assert manifest["degree_corrected"] is True
    assert manifest["diagonal"] == "removed"


def test_cluster_events_command_is_registered():
    assert COMMANDS["cluster-events"] == "openalex.analysis.cluster_events"


def test_missing_graph_tool_explains_the_install(monkeypatch):
    real_import = __import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "graph_tool" or name.startswith("graph_tool."):
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", blocked)
    adjacency = sparse.csr_matrix([[0, 1], [1, 0]], dtype=np.int64)
    with pytest.raises(RuntimeError, match="conda-forge"):
        fit_assortative(adjacency, restarts=1, seed=0)
