import hashlib
import json
import pickle
import sqlite3

import numpy as np
import pytest

from openalex.analysis import embeddings as embeddings_module
from openalex.analysis.embeddings import (
    DEFAULT_WORKERS,
    EmbeddingStore,
    build_embeddings,
    build_parser,
)
from openalex.cli import COMMANDS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_corpus(path):
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE articles(
                article_id INTEGER PRIMARY KEY,
                title TEXT
            );
            CREATE TABLE abstracts(
                article_id INTEGER PRIMARY KEY,
                abstract TEXT
            );
            INSERT INTO articles VALUES
                (2, 'Second title'),
                (1, 'First title');
            INSERT INTO abstracts VALUES
                (1, 'First abstract'),
                (2, 'Second abstract');
            """
        )


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **_kwargs):
        self.calls.append(list(texts))
        return np.asarray(
            [[float(len(text)), float(index)] for index, text in enumerate(texts)],
            dtype=np.float32,
        )


def test_embeddings_are_manifest_backed_resumable_and_read_only(tmp_path):
    database = tmp_path / "articles.db"
    output = tmp_path / "embeddings"
    write_corpus(database)
    before = digest(database)
    encoder = FakeEncoder()

    manifest = build_embeddings(
        database,
        output,
        batch_size=1,
        workers=1,
        encoder=encoder,
    )

    assert digest(database) == before
    assert encoder.calls == [
        ["First title . First abstract"],
        ["Second title . Second abstract"],
    ]
    assert manifest["complete"] is True
    assert manifest["rows"] == 2
    assert manifest["dimension"] == 2
    assert json.loads((output / "manifest.json").read_text())["database"] == "embeddings.db"

    store = EmbeddingStore(output)
    assert store.article_ids() == [1, 2]
    assert store.get_embedding(1).tolist() == [28.0, 0.0]
    assert set(store.get_embeddings_batch([2, 999])) == {2}

    resumed_encoder = FakeEncoder()
    resumed = build_embeddings(
        database,
        output,
        batch_size=1,
        workers=1,
        resume=True,
        encoder=resumed_encoder,
    )
    assert resumed["rows"] == 2
    assert resumed_encoder.calls == []


def test_embedding_output_collision_and_resume_configuration(tmp_path):
    database = tmp_path / "articles.db"
    output = tmp_path / "embeddings"
    write_corpus(database)
    build_embeddings(database, output, workers=1, encoder=FakeEncoder())

    with pytest.raises(FileExistsError, match="--resume"):
        build_embeddings(database, output, workers=1, encoder=FakeEncoder())
    with pytest.raises(ValueError, match="model"):
        build_embeddings(
            database,
            output,
            model_name="different-model",
            workers=1,
            resume=True,
            encoder=FakeEncoder(),
        )


def test_embedding_store_reads_legacy_text_ids_and_pickles(tmp_path):
    database = tmp_path / "legacy.db"
    expected = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE embeddings(article_id TEXT UNIQUE, embedding BLOB)"
        )
        connection.execute(
            "INSERT INTO embeddings VALUES (?, ?)",
            ("42", pickle.dumps(expected)),
        )

    store = EmbeddingStore(database)
    assert store.article_ids() == [42]
    np.testing.assert_array_equal(store.get_embedding(42), expected)


def test_embeddings_command_is_registered():
    assert COMMANDS["embeddings"] == "openalex.analysis.embeddings"


def test_embeddings_cli_defaults_to_automatic_runtime_selection():
    assert DEFAULT_WORKERS == 32
    args = build_parser().parse_args([])
    assert args.workers is None
    assert args.device is None
    assert args.encode_batch_size is None


def test_embedding_runtime_uses_32_cpu_workers_without_an_accelerator(monkeypatch):
    monkeypatch.setattr(
        embeddings_module, "_detect_accelerator_devices", lambda: []
    )

    device, devices, workers, encode_batch_size = (
        embeddings_module._resolve_runtime(None, None, None)
    )

    assert device == "cpu"
    assert devices == ["cpu"] * 32
    assert workers == 32
    assert encode_batch_size == 32


def test_embedding_runtime_uses_one_worker_per_visible_gpu(monkeypatch):
    monkeypatch.setattr(
        embeddings_module,
        "_detect_accelerator_devices",
        lambda: ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    )

    device, devices, workers, encode_batch_size = (
        embeddings_module._resolve_runtime(None, None, None)
    )

    assert device == "cuda"
    assert devices == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    assert workers == 4
    assert encode_batch_size == 256


def test_resume_fills_holes_in_a_partial_legacy_database(tmp_path):
    database = tmp_path / "articles.db"
    output = tmp_path / "embeddings"
    write_corpus(database)
    output.mkdir()
    with sqlite3.connect(output / "embeddings.db") as connection:
        connection.execute(
            "CREATE TABLE embeddings(article_id TEXT UNIQUE, embedding BLOB)"
        )
        connection.execute(
            "INSERT INTO embeddings VALUES (?, ?)",
            ("2", pickle.dumps(np.asarray([1.0, 2.0], dtype=np.float32))),
        )
    encoder = FakeEncoder()

    manifest = build_embeddings(
        database,
        output,
        workers=1,
        resume=True,
        encoder=encoder,
    )

    assert manifest["complete"] is True
    assert EmbeddingStore(output).article_ids() == [1, 2]
    assert encoder.calls == [["First title . First abstract"]]


def test_parallel_results_are_saved_before_a_later_chunk_fails(tmp_path, monkeypatch):
    database = tmp_path / "articles.db"
    output = tmp_path / "embeddings"
    write_corpus(database)

    class FailingExecutor:
        def __init__(self, **_kwargs):
            pass

        def map(self, _function, chunks):
            chunks = list(chunks)
            yield np.asarray([[float(len(chunks[0][0])), 0.0]], dtype=np.float32)
            raise RuntimeError("worker failed")

        def shutdown(self):
            pass

    monkeypatch.setattr(embeddings_module, "ProcessPoolExecutor", FailingExecutor)
    with pytest.raises(RuntimeError, match="worker failed"):
        build_embeddings(
            database, output, batch_size=2, workers=2, device="cpu"
        )

    partial = EmbeddingStore(output)
    assert partial.article_ids() == [1]
    assert json.loads((output / "manifest.json").read_text())["rows"] == 1

    encoder = FakeEncoder()
    manifest = build_embeddings(
        database,
        output,
        workers=1,
        resume=True,
        encoder=encoder,
    )
    assert manifest["complete"] is True
    assert EmbeddingStore(output).article_ids() == [1, 2]
    assert encoder.calls == [["Second title . Second abstract"]]
