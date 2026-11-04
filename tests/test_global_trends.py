import gzip
import json
import sqlite3
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy import sparse

from openalex.analysis.global_trends import (
    build_global_trends,
    collaborator_topic_metrics,
    leave_year_out_primary_topic,
    summarize_publication_year,
)
from openalex.artifacts.pipeline import STAGES, adopt_local_artifact
from openalex.artifacts.specs import dependency_paths
from openalex.cli import COMMANDS
from openalex.visualizations.cluster_topics import _rao_stirling_diversity
from openalex.website.build import build_website


def test_leave_year_out_selection_removes_the_plotted_year_and_breaks_ties():
    vectors = {1, 2, 4}
    assert leave_year_out_primary_topic([1, 2], [1.0, 0.6], {1: 0.5}, vectors) == 2
    assert leave_year_out_primary_topic([4, 2], [1.0, 1.0], {}, vectors) == 2
    assert leave_year_out_primary_topic([9, -1, 1], [5.0, 4.0, 1.0], {}, vectors) == 1
    assert leave_year_out_primary_topic([1], [0.5], {1: 0.5}, vectors) is None


def test_collaborator_metrics_match_the_existing_rao_stirling_helper():
    topic_ids = np.asarray([1, 2], dtype=np.int32)
    topic_vectors = sparse.eye(2, format="csr", dtype=np.float64)
    lookup = {1: 0, 2: 1}
    distinct, out_share, diversity = collaborator_topic_metrics(
        1,
        [1, 2, 2],
        lookup,
        topic_vectors,
    )
    assert distinct == 2
    assert out_share == pytest.approx(2 / 3)
    assert diversity == pytest.approx(
        _rao_stirling_diversity(Counter({1: 1, 2: 2}), lookup, topic_vectors)
    )
    assert diversity == pytest.approx(4 / 9)
    assert topic_ids.tolist() == [1, 2]
    overlapping = sparse.csr_matrix(
        np.asarray([[1.0, 0.0], [0.6, 0.8]], dtype=np.float64)
    )
    overlapping_diversity = collaborator_topic_metrics(
        1,
        [1, 2],
        lookup,
        overlapping,
    )[2]
    assert overlapping_diversity == pytest.approx(
        _rao_stirling_diversity(Counter({1: 1, 2: 1}), lookup, overlapping)
    )


def test_global_trends_increase_after_a_flat_pre_2022_line(tmp_path):
    database, topics, author_topics = write_trend_inputs(tmp_path)
    output = tmp_path / "global_trends"
    build_global_trends(database, topics, author_topics, output)

    series = json.loads((output / "series.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_version"] == 1
    assert manifest["complete"] is True
    by_year = {row["year"]: row for row in series["years"]}
    assert by_year[2020]["eligible_authors"] == 2
    assert by_year[2020]["authors_without_primary_topic"] == 1
    assert by_year[2020]["authors_without_classified_collaborator"] == 0
    for year in (2020, 2021):
        assert by_year[year]["collaborator_topic_count"] == pytest.approx(1)
        assert by_year[year]["out_of_topic_share"] == pytest.approx(0)
        assert by_year[year]["rao_stirling_diversity"] == pytest.approx(0)
        assert by_year[year]["expected_collaborator_topic_count"] is None

    topic_vectors = sparse.eye(2, format="csr", dtype=np.float64)
    diversity = _rao_stirling_diversity(
        Counter({1: 1, 2: 2}),
        {1: 0, 2: 1},
        topic_vectors,
    )
    observed = by_year[2022]
    assert observed["eligible_authors"] == 4
    assert observed["collaborator_topic_count"] == pytest.approx(2)
    assert observed["out_of_topic_share"] == pytest.approx(2 / 3)
    assert observed["rao_stirling_diversity"] == pytest.approx(diversity)
    assert observed["expected_collaborator_topic_count"] == pytest.approx(1)
    assert observed["expected_out_of_topic_share"] == pytest.approx(0)
    assert observed["expected_rao_stirling_diversity"] == pytest.approx(0)
    fit = series["ols"]["collaborator_topic_count"]
    assert fit["years"] == [2020, 2021]
    assert fit["slope"] == pytest.approx(0)
    assert fit["intercept"] + fit["slope"] * 2022 == pytest.approx(
        observed["expected_collaborator_topic_count"]
    )

    site = tmp_path / "site"
    write_event_fixture(tmp_path / "events")
    summary = build_website(
        tmp_path / "events",
        output_dir=site,
        global_trends_dir=output,
        min_document_frequency=1,
    )
    assert summary["global_trend_years"] == 3
    with gzip.open(site / "data.json.gz", "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["global_trends"]["years"][-1]["collaborator_topic_count"] == (
        pytest.approx(2)
    )
    page = (site / "global-trends.html").read_text(encoding="utf-8")
    script = (site / "global-trends.js").read_text(encoding="utf-8")
    assert "Leave-year-out primary topic" in page
    assert "subtract the 1/n weight" in page
    assert "Out-of-topic collaborators" in page
    assert "Rao–Stirling diversity" in page
    assert "years before 2022" in page
    assert "without a classified collaborator" in page.lower()
    assert "Observed" in script
    assert "Expected" in script
    assert "break_year" in script
    for name in ("clusters.html", "link-distances.html", "shapes.html", "global-trends.html"):
        assert 'href="global-trends.html"' in (site / name).read_text(encoding="utf-8")


def test_unclassified_collaborator_removes_both_authors_from_the_mean():
    author_ids = np.asarray([1], dtype=np.int64)
    offsets = np.asarray([0, 1], dtype=np.int64)
    topics = np.asarray([1], dtype=np.int32)
    weights = np.asarray([2.0], dtype=np.float64)
    vectors = sparse.eye(1, format="csr", dtype=np.float64)
    row = summarize_publication_year(
        [[1, 5]],
        [1],
        author_ids,
        offsets,
        topics,
        weights,
        {1},
        {1: 0},
        vectors,
        year=2020,
    )
    assert row["eligible_authors"] == 0
    assert row["authors_without_primary_topic"] == 1
    assert row["authors_without_classified_collaborator"] == 1
    assert row["collaborator_topic_count"] is None


def test_global_trends_command_is_wired_into_the_website_stage(tmp_path):
    assert COMMANDS["global-trends"] == "openalex.analysis.global_trends"
    assert dependency_paths("global-trends", []) == [
        "output/topics",
        "output/author_topics",
    ]
    stage = STAGES["global-trends"]
    assert stage.worker == "cpu"
    assert stage.artifact_version == 1
    assert stage.dependencies == (
        ("topics", "topics"),
        ("author_topics", "author-topics"),
    )
    assert ("global_trends", "global-trends") in STAGES["build-website"].dependencies
    assert "--global-trends-dir" in STAGES["build-website"].command

    root = tmp_path / "global_trends"
    root.mkdir()
    (root / "series.json").write_text(
        json.dumps({"break_year": 2022, "years": [{"year": 2020}]}),
        encoding="utf-8",
    )
    adopted = adopt_local_artifact(
        stage,
        root,
        {"topics": "a" * 64, "author_topics": "b" * 64},
    )
    assert adopted.manifest["artifact_version"] == 1
    assert adopted.manifest["year_count"] == 1


def write_trend_inputs(root):
    database = root / "articles.db"
    topics = root / "topics"
    author_topics = root / "author_topics"
    topics.mkdir()
    author_topics.mkdir()
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE articles (
                article_id INTEGER PRIMARY KEY,
                publication_year INTEGER NOT NULL
            );
            CREATE TABLE articles_authors (
                article_id INTEGER NOT NULL,
                author_id INTEGER NOT NULL,
                PRIMARY KEY (article_id, author_id)
            );
            CREATE INDEX idx_publication_year ON articles(publication_year);
            """
        )
        connection.executemany(
            "INSERT INTO articles(article_id, publication_year) VALUES (?, ?)",
            [(10, 2020), (11, 2020), (12, 2021), (13, 2022)],
        )
        connection.executemany(
            "INSERT INTO articles_authors(article_id, author_id) VALUES (?, ?)",
            [
                (10, 1),
                (10, 2),
                (11, 1),
                (11, 5),
                (12, 1),
                (12, 2),
                (13, 1),
                (13, 2),
                (13, 3),
                (13, 4),
            ],
        )
    pq.write_table(
        pa.table(
            {
                "article_id": pa.array([10, 11, 12, 13], type=pa.int64()),
                "topic": pa.array([1, 1, 1, 1], type=pa.int32()),
            }
        ),
        topics / "article_topic_classifications.parquet",
    )
    np.save(
        topics / "topic_vector_ids.npy",
        np.asarray([1, 2], dtype=np.int32),
        allow_pickle=False,
    )
    sparse.save_npz(topics / "topic_vectors.npz", sparse.eye(2, format="csr"))
    (topics / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_version": 2,
                "article_assignments": "article_topic_classifications.parquet",
                "topic_vector_ids": "topic_vector_ids.npy",
                "topic_vectors": "topic_vectors.npz",
            }
        ),
        encoding="utf-8",
    )
    with sqlite3.connect(author_topics / "author_topics.db") as connection:
        connection.execute(
            """
            CREATE TABLE author_topics (
                author_id INTEGER NOT NULL,
                topic INTEGER NOT NULL,
                probability REAL NOT NULL,
                fractional_weight REAL NOT NULL,
                PRIMARY KEY (author_id, topic)
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO author_topics(
                author_id, topic, probability, fractional_weight
            ) VALUES (?, ?, ?, ?)
            """,
            [
                (1, 1, 1.0, 3.0),
                (2, 1, 1.0, 3.0),
                (3, 2, 1.0, 3.0),
                (4, 2, 1.0, 3.0),
                (5, 1, 1.0, 0.5),
            ],
        )
    (author_topics / "manifest.json").write_text(
        json.dumps({"complete": True, "database": "author_topics.db"}),
        encoding="utf-8",
    )
    return database, topics, author_topics


def write_event_fixture(root):
    root.mkdir()
    incidence = root / "incidence"
    incidence.mkdir()
    vocabulary = np.array(["alpha", "beta"], dtype=np.str_)
    article_terms = sparse.csr_matrix([[1, 0], [1, 1], [0, 1]], dtype=np.int8)
    sparse.save_npz(
        root / "event_cooccurrence.npz",
        (article_terms.T @ article_terms).tocsr(),
    )
    np.save(root / "event_vocabulary.npy", vocabulary, allow_pickle=False)
    np.save(
        root / "event_document_frequency.npy",
        np.asarray(article_terms.getnnz(axis=0), dtype=np.int64),
        allow_pickle=False,
    )
    sparse.save_npz(incidence / "part_000001_000.npz", article_terms)
    np.save(
        incidence / "part_000001_000_years.npy",
        np.array([2020, 2021, 2022], dtype=np.int32),
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
                "processed_papers": 3,
                "papers_by_year": {"2020": 1, "2021": 1, "2022": 1},
            }
        ),
        encoding="utf-8",
    )
