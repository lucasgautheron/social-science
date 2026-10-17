import csv
import json
import sqlite3

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy import sparse

from openalex.cli import COMMANDS
from openalex.visualizations.cluster_topics import (
    build_cluster_topic_summary,
    summarize_cluster_topics,
)


class FakeTopicModel:
    def __init__(self):
        self.c_tf_idf_ = sparse.csr_matrix(
            [
                [1.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )

    def get_topics(self):
        return {-1: [], 0: [], 1: []}


def write_inputs(root):
    events = root / "events"
    incidence = events / "incidence"
    clusters = root / "clusters"
    topics = root / "topics"
    author_topics = root / "author_topics"
    database = root / "articles.db"
    events.mkdir()
    incidence.mkdir()
    clusters.mkdir()
    topics.mkdir()
    author_topics.mkdir()

    vocabulary = np.array(["alpha", "beta", "gamma"], dtype=np.str_)
    article_terms = sparse.csr_matrix(
        [
            [1, 1, 0],
            [1, 0, 0],
            [0, 1, 1],
            [0, 0, 1],
            [1, 0, 0],
            [0, 0, 1],
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
    sparse.save_npz(incidence / "part_000001_000.npz", article_terms)
    np.save(
        incidence / "part_000001_000_article_ids.npy",
        np.array([10, 11, 12, 13, 14, 15], dtype=np.int64),
        allow_pickle=False,
    )
    np.save(
        incidence / "part_000001_000_years.npy",
        np.array([2020, 2020, 2021, 2021, 2021, 2022], dtype=np.int32),
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
                "incidence_parts": ["part_000001_000.npz"],
            }
        ),
        encoding="utf-8",
    )

    np.save(clusters / "keywords.npy", vocabulary, allow_pickle=False)
    np.save(
        clusters / "groups_by_level.npy",
        np.array([[0, 0, 1]], dtype=np.int32),
        allow_pickle=False,
    )
    (clusters / "manifest.json").write_text(
        json.dumps(
            {
                "keywords": "keywords.npy",
                "groups_by_level": "groups_by_level.npy",
                "coarse_level": 0,
            }
        ),
        encoding="utf-8",
    )

    pq.write_table(
        pa.table(
            {
                "article_id": pa.array(
                    [10, 11, 12, 13, 15, 20, 21, 22, 23],
                    type=pa.int64(),
                ),
                "topic": pa.array(
                    [0, 1, 0, 1, -1, 1, 0, 1, 0],
                    type=pa.int32(),
                ),
            }
        ),
        topics / "article_topic_classifications.parquet",
    )
    (topics / "topic_list.csv").write_text(
        "Topic,Count,Name\n-1,1,Outlier\n0,2,Topic zero\n1,2,Topic one\n",
        encoding="utf-8",
    )
    (topics / "bertopic_model").mkdir()
    (topics / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_version": 2,
                "article_assignments": "article_topic_classifications.parquet",
            }
        ),
        encoding="utf-8",
    )
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE articles_authors(
                article_id INTEGER NOT NULL,
                author_id INTEGER NOT NULL,
                PRIMARY KEY(article_id, author_id)
            );
            INSERT INTO articles_authors VALUES
                (10, 100), (10, 200),
                (11, 100),
                (12, 200), (12, 300),
                (13, 300),
                (14, 500),
                (15, 400),
                (20, 100), (21, 200), (22, 300), (23, 500);
            """
        )
    with sqlite3.connect(author_topics / "author_topics.db") as connection:
        connection.executescript(
            """
            CREATE TABLE author_topics(
                author_id INTEGER NOT NULL,
                topic INTEGER NOT NULL,
                probability REAL NOT NULL,
                fractional_weight REAL NOT NULL,
                PRIMARY KEY(author_id, topic)
            ) WITHOUT ROWID;
            INSERT INTO author_topics VALUES
                (100, 0, 0.2, 0.5),
                (100, 1, 0.8, 2.0),
                (200, 0, 1.0, 2.0),
                (300, 0, 0.2, 0.5),
                (300, 1, 0.8, 2.0),
                (400, -1, 1.0, 1.0),
                (500, 0, 1.0, 1.0);
            """
        )
    (author_topics / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_version": 1,
                "complete": True,
                "database": "author_topics.db",
            }
        ),
        encoding="utf-8",
    )
    return events, clusters, topics, database, author_topics


def test_cluster_topic_summary_counts_each_paper_once_and_orders_topics(tmp_path):
    events, clusters, topics, database, author_topics = write_inputs(tmp_path)
    output = tmp_path / "cluster_topic_visualizations"
    summary = build_cluster_topic_summary(
        events,
        clusters,
        topics,
        output,
        db_path=database,
        author_topics_dir=author_topics,
        topic_model_loader=lambda _path: FakeTopicModel(),
    )

    with summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    first, second = rows
    assert int(first["paper_count"]) == 4
    assert int(first["topic_observation_count"]) == 3
    assert int(first["topic_missing_count"]) == 1
    assert int(first["topic_outlier_count"]) == 0
    assert float(first["rao_stirling_diversity"]) == pytest.approx(4 / 9)
    assert json.loads(first["topic_distribution"]) == [
        {"topic_id": 0, "label": "Topic zero", "papers": 2, "share": 2 / 3},
        {"topic_id": 1, "label": "Topic one", "papers": 1, "share": 1 / 3},
    ]

    assert int(second["paper_count"]) == 3
    assert int(second["topic_observation_count"]) == 2
    assert int(second["topic_missing_count"]) == 0
    assert int(second["topic_outlier_count"]) == 1
    assert float(second["rao_stirling_diversity"]) == pytest.approx(0.5)
    assert [item["topic_id"] for item in json.loads(second["topic_distribution"])] == [
        0,
        1,
    ]
    assert int(first["author_count"]) == 4
    assert int(first["author_primary_topic_count"]) == 4
    assert int(first["author_primary_topic_missing_count"]) == 0
    assert float(first["author_weight"]) == pytest.approx(4.0)
    assert float(first["author_rao_stirling_diversity"]) == pytest.approx(0.5)
    assert json.loads(first["author_topic_distribution"]) == [
        {
            "topic_id": 0,
            "label": "Topic zero",
            "authors": 2,
            "weight": 2.0,
            "share": 0.5,
        },
        {
            "topic_id": 1,
            "label": "Topic one",
            "authors": 2,
            "weight": 2.0,
            "share": 0.5,
        },
    ]
    assert int(second["author_count"]) == 3
    assert int(second["author_primary_topic_count"]) == 2
    assert int(second["author_primary_topic_missing_count"]) == 1
    assert float(second["author_weight"]) == pytest.approx(3.0)
    assert float(second["author_primary_topic_weight"]) == pytest.approx(2.0)
    assert float(second["author_primary_topic_missing_weight"]) == pytest.approx(1.0)
    assert float(second["author_rao_stirling_diversity"]) == pytest.approx(0.375)
    assert [
        item["topic_id"]
        for item in json.loads(second["author_topic_distribution"])
    ] == [1, 0]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cluster_count"] == 2
    assert manifest["cluster_level"] == 0
    assert manifest["summary"] == "cluster_topic_summary.csv"
    assert manifest["author_diversity"] is True


def test_cluster_topic_summary_uses_exported_vectors_without_saved_model(tmp_path):
    events, clusters, topics, database, author_topics = write_inputs(tmp_path)
    np.save(
        topics / "topic_vector_ids.npy",
        np.array([-1, 0, 1], dtype=np.int32),
        allow_pickle=False,
    )
    vectors = FakeTopicModel().c_tf_idf_
    norms = np.sqrt(np.asarray(vectors.multiply(vectors).sum(axis=1)).ravel())
    sparse.save_npz(
        topics / "topic_vectors.npz",
        sparse.diags(1.0 / norms) @ vectors,
    )
    (topics / "bertopic_model").rmdir()
    manifest = json.loads((topics / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "artifact_version": 3,
            "topic_vector_ids": "topic_vector_ids.npy",
            "topic_vectors": "topic_vectors.npz",
        }
    )
    (topics / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    summary = build_cluster_topic_summary(
        events,
        clusters,
        topics,
        tmp_path / "output",
        db_path=database,
        author_topics_dir=author_topics,
        topic_model_loader=lambda _path: pytest.fail("model loader was used"),
    )

    assert summary.is_file()


def test_summarize_cluster_topics_rejects_topic_without_vector():
    with pytest.raises(ValueError, match="no c-TF-IDF vectors: 7"):
        summarize_cluster_topics(
            np.array([10], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int32),
            np.array(["alpha"]),
            np.array([10], dtype=np.int64),
            np.array([7], dtype=np.int32),
            np.array([0], dtype=np.int32),
            sparse.csr_matrix([[1.0]]),
            {0: "zero"},
        )


def test_visualize_cluster_topics_command_is_registered():
    assert (
        COMMANDS["visualize-cluster-topics"]
        == "openalex.visualizations.cluster_topics"
    )
