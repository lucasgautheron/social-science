import hashlib
import json
import sqlite3

import pytest

from openalex.analysis import author_topics as author_topics_module
from openalex.analysis.author_topics import build_author_topics
from openalex.cli import COMMANDS

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_corpus(path):
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE articles_authors(
                article_id INTEGER NOT NULL,
                author_id INTEGER NOT NULL,
                position INTEGER,
                PRIMARY KEY(article_id, author_id)
            );
            INSERT INTO articles_authors VALUES
                (1, 10, 1),
                (1, 20, 2),
                (2, 10, 1);
            """
        )


def write_topics(path, *, artifact_version=2):
    path.mkdir()
    table = pa.table(
        {
            "article_id": pa.array([1, 2], type=pa.int64()),
            "topic": pa.array([0, 1], type=pa.int32()),
            "probability": pa.array([0.01, 0.99], type=pa.float32()),
        }
    )
    pq.write_table(
        table,
        path / "article_topic_classifications.parquet",
        compression="zstd",
    )
    (path / "topic_list.csv").write_text(
        "Topic,Count,Name\n0,1,zero\n1,1,one\n0,1,updated-zero\n"
    )
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_version": artifact_version,
                "article_assignments": (
                    "article_topic_classifications.parquet"
                ),
                "articles": 2,
            }
        )
    )


def test_author_topics_fractional_hard_assignments_and_normalization(tmp_path):
    corpus = tmp_path / "articles.db"
    topics = tmp_path / "topics"
    output = tmp_path / "author_topics"
    write_corpus(corpus)
    write_topics(topics)
    corpus_digest = digest(corpus)
    parquet_digest = digest(
        topics / "article_topic_classifications.parquet"
    )

    manifest = build_author_topics(
        corpus,
        topics,
        output,
        batch_size=1,
    )

    assert digest(corpus) == corpus_digest
    assert (
        digest(topics / "article_topic_classifications.parquet")
        == parquet_digest
    )
    assert manifest["complete"] is True
    assert manifest["authors"] == 2
    assert manifest["article_topic_pairs"] == 3
    assert not (output / ".scratch").exists()
    with sqlite3.connect(output / "author_topics.db") as connection:
        columns = [
            (row[1], row[2])
            for row in connection.execute("PRAGMA table_info(author_topics)")
        ]
        distributions = connection.execute(
            """
            SELECT author_id, topic, probability, fractional_weight
            FROM author_topics
            ORDER BY author_id, topic
            """
        ).fetchall()
        authors = connection.execute(
            """
            SELECT author_id, total_weight, paper_count, topic_count
            FROM authors
            ORDER BY author_id
            """
        ).fetchall()
        labels = connection.execute(
            "SELECT topic, label FROM topics ORDER BY topic"
        ).fetchall()

    assert columns == [
        ("author_id", "INTEGER"),
        ("topic", "INTEGER"),
        ("probability", "REAL"),
        ("fractional_weight", "REAL"),
    ]
    assert distributions == pytest.approx(
        [
            (10, 0, 1 / 3, 0.5),
            (10, 1, 2 / 3, 1.0),
            (20, 0, 1.0, 0.5),
        ]
    )
    assert authors == pytest.approx(
        [(10, 1.5, 2, 2), (20, 0.5, 1, 1)]
    )
    assert labels == [(0, "updated-zero"), (1, "one")]


def test_author_topics_resume_after_transactional_accumulation(
    tmp_path, monkeypatch
):
    corpus = tmp_path / "articles.db"
    topics = tmp_path / "topics"
    output = tmp_path / "author_topics"
    write_corpus(corpus)
    write_topics(topics)
    original = author_topics_module._write_final_database

    monkeypatch.setattr(
        author_topics_module,
        "_write_final_database",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("stop")),
    )
    with pytest.raises(RuntimeError, match="stop"):
        build_author_topics(corpus, topics, output, batch_size=1)
    assert not (output / "author_topics.db").exists()
    assert json.loads((output / "manifest.json").read_text())[
        "processed_articles"
    ] == 2

    monkeypatch.setattr(
        author_topics_module, "_write_final_database", original
    )
    manifest = build_author_topics(
        corpus, topics, output, batch_size=1, resume=True
    )
    assert manifest["complete"] is True
    assert manifest["article_topic_pairs"] == 3


def test_author_topics_reject_stale_schema_and_unordered_rows(tmp_path):
    corpus = tmp_path / "articles.db"
    topics = tmp_path / "topics"
    write_corpus(corpus)
    write_topics(topics, artifact_version=1)
    with pytest.raises(ValueError, match="schema version 2"):
        build_author_topics(corpus, topics, tmp_path / "output")

    topics_manifest = topics / "manifest.json"
    manifest = json.loads(topics_manifest.read_text())
    manifest["artifact_version"] = 2
    topics_manifest.write_text(json.dumps(manifest))
    pq.write_table(
        pa.table(
            {
                "article_id": pa.array([2, 1], type=pa.int64()),
                "topic": pa.array([1, 0], type=pa.int32()),
                "probability": pa.array([0.9, 0.8], type=pa.float32()),
            }
        ),
        topics / "article_topic_classifications.parquet",
    )
    with pytest.raises(ValueError, match="strictly ordered"):
        build_author_topics(corpus, topics, tmp_path / "output")


def test_author_topic_command_is_registered():
    assert COMMANDS["author-topics"] == "openalex.analysis.author_topics"
