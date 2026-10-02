import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from openalex.analysis.embeddings import EmbeddingStore, build_embeddings
from openalex.analysis.topics import (
    _write_full_classifications,
    assign_topics,
    select_topic_sample,
    topic_probabilities,
    train_topic_classifier,
)
from openalex.cli import COMMANDS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_corpus(path):
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE articles(article_id INTEGER PRIMARY KEY, title TEXT);
            CREATE TABLE abstracts(article_id INTEGER PRIMARY KEY, abstract TEXT);
            CREATE TABLE articles_order(
                article_id INTEGER,
                random_rank INTEGER PRIMARY KEY
            );
            INSERT INTO articles VALUES
                (1, 'Alpha'),
                (2, 'Beta');
            INSERT INTO abstracts VALUES
                (1, 'First abstract'),
                (2, 'Second abstract');
            INSERT INTO articles_order VALUES
                (1, 1),
                (2, 2);
            """
        )


def write_larger_corpus(path):
    write_corpus(path)
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            INSERT INTO articles VALUES
                (3, 'Gamma'),
                (4, 'Delta');
            INSERT INTO abstracts VALUES
                (3, 'Third abstract'),
                (4, 'Fourth abstract');
            INSERT INTO articles_order VALUES
                (3, 3),
                (4, 4);
            """
        )


class FakeEncoder:
    def encode(self, texts, **_kwargs):
        return np.asarray(
            [[float(index), float(len(text))] for index, text in enumerate(texts)],
            dtype=np.float32,
        )


class FakeFrame:
    def __init__(self, records):
        self.records = records

    def to_dict(self, orient):
        assert orient == "records"
        return list(self.records)

    def to_csv(self, path, index=False):
        assert index is False
        fieldnames = list(self.records[0]) if self.records else []
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)


class FakeFigure:
    def write_html(self, path):
        Path(path).write_text("<html>fake</html>\n", encoding="utf-8")


class FakeTopicModel:
    def __init__(self):
        self.documents = None
        self.embeddings = None
        self.updated = None

    def fit_transform(self, documents, embeddings):
        self.documents = list(documents)
        self.embeddings = np.asarray(embeddings)
        return [0, -1], np.asarray([[0.9, 0.1], [0.2, 0.8]])

    def reduce_outliers(self, documents, topics, **_kwargs):
        assert documents == self.documents
        assert topics == [0, -1]
        return [0, 1]

    def update_topics(self, documents, **kwargs):
        self.updated = (list(documents), kwargs["topics"])

    def get_topic_info(self):
        return FakeFrame(
            [
                {"Topic": 0, "Count": 1, "Name": "0_alpha"},
                {"Topic": 1, "Count": 1, "Name": "1_beta"},
            ]
        )

    def get_topic(self, topic):
        return [(f"word-{topic}", 0.75)]

    def hierarchical_topics(self, documents):
        assert documents == self.documents
        return FakeFrame([{"Parent_ID": 2, "Topics": "0,1"}])

    def visualize_hierarchy(self, **_kwargs):
        return FakeFigure()

    def visualize_barchart(self, **_kwargs):
        return FakeFigure()

    def visualize_topics(self):
        return FakeFigure()

    def visualize_distribution(self, _probabilities):
        return FakeFigure()

    def save(self, path, serialization):
        assert serialization == "pickle"
        Path(path).mkdir()
        (Path(path) / "model.fake").write_text("model\n", encoding="utf-8")


class FakeClassifier:
    classes_ = np.asarray([0, 1], dtype=np.int64)

    def predict_proba(self, embeddings):
        return np.tile(np.asarray([[0.75, 0.25]]), (len(embeddings), 1))


def fake_classifier_trainer(
    _embeddings,
    _topics,
    *,
    cv_folds,
    **_kwargs,
):
    return (
        FakeClassifier(),
        {
            "algorithm": "MLPClassifier",
            "cross_validation_folds": cv_folds,
            "cv_macro_f1": 0.8,
            "held_out_macro_f1": 0.75,
            "held_out_weighted_f1": 0.78,
            "best_params": {"hidden_layer_sizes": [128]},
        },
        [
            {
                "params": {"hidden_layer_sizes": [128]},
                "mean_macro_f1": 0.8,
                "std_macro_f1": 0.01,
                "mean_weighted_f1": 0.82,
                "rank_macro_f1": 1,
            }
        ],
    )


def test_topic_assignment_restores_legacy_outputs_and_keeps_corpus_read_only(tmp_path):
    database = tmp_path / "articles.db"
    embeddings = tmp_path / "embeddings"
    output = tmp_path / "topics"
    write_corpus(database)
    build_embeddings(database, embeddings, workers=1, encoder=FakeEncoder())
    before = digest(database)
    created = []
    models = []

    def model_factory(**kwargs):
        created.append(kwargs)
        model = FakeTopicModel()
        models.append(model)
        return model

    manifest = assign_topics(
        database,
        embeddings,
        output,
        sample_size=2,
        random_seed=7,
        min_cluster_size=2,
        model_factory=model_factory,
        classifier_trainer=fake_classifier_trainer,
    )

    assert digest(database) == before
    assert created[0]["min_cluster_size"] == 2
    assert created[0]["use_keybert_representation"] is True
    assert models[0].updated == (
        ["Alpha . First abstract", "Beta . Second abstract"],
        [0, 1],
    )
    assert manifest["articles"] == 2
    assert manifest["sample_articles"] == 2
    assert manifest["classifier"]["held_out_macro_f1"] == 0.75
    assert manifest["topics"] == 2
    assert manifest["outliers"] == 0
    assert manifest["embedding_manifest_sha256"]
    assert (output / "topic_list.csv").is_file()
    assert (output / "detailed_topics.csv").is_file()
    assert (output / "topic_hierarchy.csv").is_file()
    assert (output / "topic_hierarchy.html").is_file()
    assert (output / "bertopic_model" / "model.fake").is_file()
    assert (output / "topic_classifier.joblib").is_file()
    assert (output / "classifier_metrics.json").is_file()
    assert (output / "classifier_cv_results.csv").is_file()

    with (output / "sample_topic_classifications.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        sample_rows = list(csv.DictReader(handle))
    assert [int(row["article_id"]) for row in sample_rows] == [1, 2]
    assert [int(row["topic"]) for row in sample_rows] == [0, 1]
    assert [float(row["probability"]) for row in sample_rows] == pytest.approx(
        [0.9, 0.8]
    )
    assert [row["topic_label"] for row in sample_rows] == [
        "0_alpha",
        "1_beta",
    ]
    full = pq.read_table(
        output / "article_topic_classifications.parquet"
    ).to_pydict()
    assert full["article_id"] == [1, 2]
    assert full["topic"] == [0, 0]
    assert full["probability"] == pytest.approx([0.75, 0.75])
    parquet = pq.ParquetFile(
        output / "article_topic_classifications.parquet"
    )
    assert str(parquet.schema_arrow.field("article_id").type) == "int64"
    assert str(parquet.schema_arrow.field("topic").type) == "int32"
    assert str(parquet.schema_arrow.field("probability").type) == "float"
    assert parquet.metadata.row_group(0).column(0).compression == "ZSTD"
    assert json.loads((output / "manifest.json").read_text())["artifact_version"] == 2

    with pytest.raises(FileExistsError, match="new --output-dir"):
        assign_topics(
            database,
            embeddings,
            output,
            sample_size=2,
            min_cluster_size=2,
            model_factory=model_factory,
        )


def test_topic_sample_uses_existing_random_order(tmp_path):
    database = tmp_path / "articles.db"
    embeddings_path = tmp_path / "embeddings"
    write_larger_corpus(database)
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            DELETE FROM articles_order;
            INSERT INTO articles_order VALUES
                (4, 1),
                (2, 2),
                (1, 3),
                (3, 4);
            """
        )
    build_embeddings(
        database, embeddings_path, workers=1, encoder=FakeEncoder()
    )

    selected = select_topic_sample(
        database,
        EmbeddingStore(embeddings_path),
        sample_size=2,
    )

    assert selected == [2, 4]


def test_topic_sample_rejects_missing_random_order(tmp_path):
    database = tmp_path / "articles.db"
    embeddings_path = tmp_path / "embeddings"
    write_corpus(database)
    build_embeddings(
        database, embeddings_path, workers=1, encoder=FakeEncoder()
    )
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TABLE articles_order")

    with pytest.raises(ValueError, match="no articles_order"):
        select_topic_sample(
            database,
            EmbeddingStore(embeddings_path),
            sample_size=2,
        )


def test_classifier_extrapolates_sample_topics_to_every_embedding(tmp_path):
    database = tmp_path / "articles.db"
    embeddings = tmp_path / "embeddings"
    output = tmp_path / "topics"
    write_larger_corpus(database)
    build_embeddings(database, embeddings, workers=1, encoder=FakeEncoder())

    manifest = assign_topics(
        database,
        embeddings,
        output,
        sample_size=2,
        min_cluster_size=2,
        hierarchy=False,
        visualizations=False,
        save_model=False,
        model_factory=lambda **_kwargs: FakeTopicModel(),
        classifier_trainer=fake_classifier_trainer,
    )

    full = pq.read_table(
        output / "article_topic_classifications.parquet"
    ).to_pydict()
    assert full["article_id"] == [1, 2, 3, 4]
    assert full["topic"] == [0, 0, 0, 0]
    assert full["probability"] == pytest.approx([0.75] * 4)
    assert manifest["articles"] == 4
    assert manifest["sample_articles"] == 2


def test_topics_command_is_registered():
    assert COMMANDS["topics"] == "openalex.analysis.topics"


def test_real_mlp_reports_f1_and_refits_without_holding_out_rows():
    rng = np.random.default_rng(42)
    embeddings = np.concatenate(
        [
            rng.normal(-1, 0.25, size=(60, 8)),
            rng.normal(1, 0.25, size=(60, 8)),
        ]
    ).astype(np.float32)
    topics = np.concatenate(
        [np.zeros(60, dtype=np.int64), np.ones(60, dtype=np.int64)]
    )

    classifier, metrics, cv_results = train_topic_classifier(
        embeddings,
        topics,
        random_seed=42,
        cv_folds=2,
        jobs=1,
        test_size=0.2,
    )

    assert metrics["cv_macro_f1"] > 0.9
    assert metrics["held_out_macro_f1"] > 0.9
    assert len(cv_results) == 6
    assert classifier.named_steps["mlp"].early_stopping is False


def test_full_classification_parquet_is_published_atomically(tmp_path):
    database = tmp_path / "articles.db"
    embeddings_path = tmp_path / "embeddings"
    output_path = tmp_path / "classifications.parquet"
    write_larger_corpus(database)
    build_embeddings(
        database, embeddings_path, workers=1, encoder=FakeEncoder()
    )
    class FailingClassifier(FakeClassifier):
        def __init__(self):
            self.calls = 0

        def predict_proba(self, embeddings):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("prediction failed")
            return super().predict_proba(embeddings)

    with pytest.raises(RuntimeError, match="prediction failed"):
        _write_full_classifications(
            output_path,
            EmbeddingStore(embeddings_path),
            FailingClassifier(),
            batch_size=2,
        )

    assert not output_path.exists()
    assert not output_path.with_suffix(".parquet.tmp").exists()


def test_topics_reject_an_incomplete_embedding_artifact(tmp_path):
    database = tmp_path / "articles.db"
    embeddings = tmp_path / "embeddings"
    write_corpus(database)
    build_embeddings(database, embeddings, workers=1, encoder=FakeEncoder())
    manifest_path = embeddings / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["complete"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete"):
        assign_topics(
            database,
            embeddings,
            tmp_path / "topics",
            sample_size=2,
            min_cluster_size=2,
            model_factory=lambda **_kwargs: FakeTopicModel(),
        )


def test_outlier_probability_is_blank():
    probabilities = topic_probabilities(
        np.asarray([[0.8, 0.2], [0.1, 0.9]]),
        np.asarray([-1, 1]),
    )
    assert np.isnan(probabilities[0])
    assert probabilities[1] == pytest.approx(0.9)
