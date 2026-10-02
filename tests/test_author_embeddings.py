import hashlib
import json
import sqlite3

import numpy as np
import pytest

from openalex.analysis import author_embeddings as author_embeddings_module
from openalex.analysis.author_embeddings import build_author_embeddings
from openalex.analysis.embeddings import encode_embedding
from openalex.cli import COMMANDS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_corpus(path):
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE authors(author_id INTEGER PRIMARY KEY);
            CREATE TABLE articles_authors(
                article_id INTEGER NOT NULL,
                author_id INTEGER NOT NULL,
                position INTEGER,
                PRIMARY KEY(article_id, author_id)
            );
            INSERT INTO authors VALUES (10), (20), (30);
            INSERT INTO articles_authors VALUES
                (1, 10, 1),
                (1, 20, 2),
                (2, 10, 1);
            """
        )


def write_embeddings(path, *, complete=True):
    path.mkdir()
    source = path.parent / "articles.db"
    database = path / "embeddings.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE embeddings(
                article_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO embeddings VALUES (?, ?)",
            [
                (1, encode_embedding(np.asarray([1.0, 0.0]))),
                (2, encode_embedding(np.asarray([0.0, 1.0]))),
            ],
        )
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_version": 1,
                "complete": complete,
                "database": "embeddings.db",
                "dimension": 2,
                "model": "test-model",
                "rows": 2,
                "source_database": str(source.resolve()),
                "source_size": source.stat().st_size,
            }
        )
    )


def test_author_embeddings_fractional_weight_schema_and_read_only_inputs(tmp_path):
    corpus = tmp_path / "articles.db"
    embeddings = tmp_path / "embeddings"
    output = tmp_path / "author_embeddings"
    write_corpus(corpus)
    write_embeddings(embeddings)
    source_digest = digest(corpus)
    embedding_digest = digest(embeddings / "embeddings.db")

    manifest = build_author_embeddings(
        corpus,
        embeddings,
        output,
        batch_size=1,
        finalize_batch_size=1,
    )

    assert digest(corpus) == source_digest
    assert digest(embeddings / "embeddings.db") == embedding_digest
    assert manifest["complete"] is True
    assert manifest["authors"] == 2
    assert manifest["authorships"] == 3
    assert not (output / ".scratch").exists()
    with sqlite3.connect(output / "author_embeddings.db") as connection:
        columns = [
            (row[1], row[2])
            for row in connection.execute(
                "PRAGMA table_info(author_embeddings)"
            )
        ]
        rows = connection.execute(
            """
            SELECT author_id, embedding, paper_count, total_weight
            FROM author_embeddings
            ORDER BY author_id
            """
        ).fetchall()
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))

    assert columns == [
        ("author_id", "INTEGER"),
        ("embedding", "BLOB"),
        ("paper_count", "INTEGER"),
        ("total_weight", "REAL"),
    ]
    assert [row[0] for row in rows] == [10, 20]
    np.testing.assert_allclose(
        np.frombuffer(rows[0][1], dtype="<f4"),
        [1 / 3, 2 / 3],
    )
    np.testing.assert_allclose(
        np.frombuffer(rows[1][1], dtype="<f4"),
        [1.0, 0.0],
    )
    assert rows[0][2:] == pytest.approx((2, 1.5))
    assert rows[1][2:] == pytest.approx((1, 0.5))
    assert metadata["dimension"] == "2"
    assert metadata["embedding_encoding"] == "little-endian-float32"
    assert metadata["model"] == "test-model"


def test_author_embeddings_resume_after_atomic_finalize_failure(
    tmp_path, monkeypatch
):
    corpus = tmp_path / "articles.db"
    embeddings = tmp_path / "embeddings"
    output = tmp_path / "author_embeddings"
    write_corpus(corpus)
    write_embeddings(embeddings)
    original = author_embeddings_module._write_author_database

    monkeypatch.setattr(
        author_embeddings_module,
        "_write_author_database",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("stop")),
    )
    with pytest.raises(RuntimeError, match="stop"):
        build_author_embeddings(corpus, embeddings, output, batch_size=1)
    assert not (output / "author_embeddings.db").exists()
    assert json.loads((output / "manifest.json").read_text())[
        "processed_articles"
    ] == 2

    monkeypatch.setattr(
        author_embeddings_module, "_write_author_database", original
    )
    manifest = build_author_embeddings(
        corpus, embeddings, output, batch_size=1, resume=True
    )
    assert manifest["complete"] is True
    assert manifest["authors"] == 2


def test_author_embeddings_roll_back_uncommitted_memmap_batch(
    tmp_path, monkeypatch
):
    corpus = tmp_path / "articles.db"
    embeddings = tmp_path / "embeddings"
    output = tmp_path / "author_embeddings"
    write_corpus(corpus)
    write_embeddings(embeddings)
    original = author_embeddings_module.atomic_json

    def interrupt_first_checkpoint(path, payload):
        if payload.get("processed_articles") == 1:
            raise RuntimeError("interrupted checkpoint")
        original(path, payload)

    monkeypatch.setattr(
        author_embeddings_module,
        "atomic_json",
        interrupt_first_checkpoint,
    )
    with pytest.raises(RuntimeError, match="interrupted checkpoint"):
        build_author_embeddings(corpus, embeddings, output, batch_size=1)
    assert (output / ".scratch" / "pending_batch.npz").is_file()

    monkeypatch.setattr(author_embeddings_module, "atomic_json", original)
    build_author_embeddings(
        corpus, embeddings, output, batch_size=1, resume=True
    )
    with sqlite3.connect(output / "author_embeddings.db") as connection:
        blob, weight = connection.execute(
            """
            SELECT embedding, total_weight
            FROM author_embeddings
            WHERE author_id = 10
            """
        ).fetchone()
    np.testing.assert_allclose(
        np.frombuffer(blob, dtype="<f4"),
        [1 / 3, 2 / 3],
    )
    assert weight == pytest.approx(1.5)


def test_author_embeddings_chunks_more_than_sqlite_parameter_limit(tmp_path):
    corpus = tmp_path / "articles.db"
    embeddings = tmp_path / "embeddings"
    output = tmp_path / "author_embeddings"
    count = 1_001
    with sqlite3.connect(corpus) as connection:
        connection.executescript(
            """
            CREATE TABLE authors(author_id INTEGER PRIMARY KEY);
            CREATE TABLE articles_authors(
                article_id INTEGER NOT NULL,
                author_id INTEGER NOT NULL,
                PRIMARY KEY(article_id, author_id)
            );
            INSERT INTO authors VALUES (10);
            """
        )
        connection.executemany(
            "INSERT INTO articles_authors VALUES (?, 10)",
            [(article_id,) for article_id in range(1, count + 1)],
        )
    embeddings.mkdir()
    with sqlite3.connect(embeddings / "embeddings.db") as connection:
        connection.execute(
            """
            CREATE TABLE embeddings(
                article_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL
            )
            """
        )
        vector = encode_embedding(np.asarray([2.0], dtype=np.float32))
        connection.executemany(
            "INSERT INTO embeddings VALUES (?, ?)",
            [
                (article_id, vector)
                for article_id in range(1, count + 1)
            ],
        )
    (embeddings / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_version": 1,
                "complete": True,
                "database": "embeddings.db",
                "dimension": 1,
                "model": "test-model",
                "rows": count,
                "source_database": str(corpus.resolve()),
                "source_size": corpus.stat().st_size,
            }
        )
    )

    manifest = build_author_embeddings(
        corpus, embeddings, output, batch_size=count
    )

    assert manifest["authorships"] == count
    with sqlite3.connect(output / "author_embeddings.db") as connection:
        blob, paper_count = connection.execute(
            "SELECT embedding, paper_count FROM author_embeddings"
        ).fetchone()
    assert paper_count == count
    np.testing.assert_array_equal(
        np.frombuffer(blob, dtype="<f4"),
        np.asarray([2.0], dtype=np.float32),
    )


def test_author_embeddings_reject_incomplete_or_unmanifested_input(tmp_path):
    corpus = tmp_path / "articles.db"
    embeddings = tmp_path / "embeddings"
    write_corpus(corpus)
    write_embeddings(embeddings, complete=False)

    with pytest.raises(ValueError, match="incomplete"):
        build_author_embeddings(corpus, embeddings, tmp_path / "output")

    (embeddings / "manifest.json").unlink()
    with pytest.raises(ValueError, match="manifest-backed"):
        build_author_embeddings(corpus, embeddings, tmp_path / "output")


def test_author_embeddings_reject_mismatched_corpus_provenance(tmp_path):
    corpus = tmp_path / "articles.db"
    embeddings = tmp_path / "embeddings"
    write_corpus(corpus)
    write_embeddings(embeddings)
    manifest_path = embeddings / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_database"] = str(tmp_path / "another.db")
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="corpus provenance"):
        build_author_embeddings(corpus, embeddings, tmp_path / "output")


def test_author_embeddings_resume_rejects_changed_embedding_payload(
    tmp_path, monkeypatch
):
    corpus = tmp_path / "articles.db"
    embeddings = tmp_path / "embeddings"
    output = tmp_path / "author_embeddings"
    write_corpus(corpus)
    write_embeddings(embeddings)
    monkeypatch.setattr(
        author_embeddings_module,
        "_write_author_database",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("stop")),
    )
    with pytest.raises(RuntimeError, match="stop"):
        build_author_embeddings(corpus, embeddings, output)

    with sqlite3.connect(embeddings / "embeddings.db") as connection:
        connection.execute(
            "UPDATE embeddings SET embedding = ? WHERE article_id = 1",
            (encode_embedding(np.asarray([9.0, 9.0])),),
        )
    with pytest.raises(ValueError, match="embedding_database_sha256"):
        build_author_embeddings(corpus, embeddings, output, resume=True)


def test_author_embedding_command_is_registered():
    assert (
        COMMANDS["author-embeddings"]
        == "openalex.analysis.author_embeddings"
    )
