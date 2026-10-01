import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from openalex.analysis.embeddings import build_embeddings
from openalex.analysis.topics import (
    assign_topics,
    select_article_ids,
    topic_probabilities,
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
            INSERT INTO articles VALUES
                (1, 'Alpha'),
                (2, 'Beta');
            INSERT INTO abstracts VALUES
                (1, 'First abstract'),
                (2, 'Second abstract');
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
    )

    assert digest(database) == before
    assert created[0]["min_cluster_size"] == 2
    assert created[0]["use_keybert_representation"] is True
    assert models[0].updated == (
        ["Alpha . First abstract", "Beta . Second abstract"],
        [0, 1],
    )
    assert manifest["articles"] == 2
    assert manifest["topics"] == 2
    assert manifest["outliers"] == 0
    assert manifest["embedding_manifest_sha256"]
    assert (output / "topic_list.csv").is_file()
    assert (output / "detailed_topics.csv").is_file()
    assert (output / "topic_hierarchy.csv").is_file()
    assert (output / "topic_hierarchy.html").is_file()
    assert (output / "bertopic_model" / "model.fake").is_file()

    with (output / "article_topic_classifications.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["article_id"]) for row in rows] == [1, 2]
    assert [int(row["topic"]) for row in rows] == [0, 1]
    assert [float(row["probability"]) for row in rows] == pytest.approx([0.9, 0.8])
    assert [row["topic_label"] for row in rows] == ["0_alpha", "1_beta"]
    assert json.loads((output / "manifest.json").read_text())["artifact_version"] == 1

    with pytest.raises(FileExistsError, match="new --output-dir"):
        assign_topics(
            database,
            embeddings,
            output,
            sample_size=2,
            min_cluster_size=2,
            model_factory=model_factory,
        )


def test_article_sampling_is_seeded_and_sorted():
    ids = list(range(1, 101))
    first = select_article_ids(ids, 10, 42)
    second = select_article_ids(ids, 10, 42)
    different = select_article_ids(ids, 10, 43)
    assert first == sorted(first)
    assert first == second
    assert first != different
    assert select_article_ids(ids, None, 42) == ids


def test_topics_command_is_registered():
    assert COMMANDS["topics"] == "openalex.analysis.topics"


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
