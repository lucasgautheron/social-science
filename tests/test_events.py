import hashlib
import sqlite3

import pytest

from openalex import events
from openalex.events import EventExtractor


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_event_reads_do_not_modify_sqlite(tmp_path):
    database = tmp_path / "articles.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE articles(
                article_id INTEGER PRIMARY KEY,
                publication_year INTEGER,
                title TEXT
            );
            CREATE TABLE abstracts(article_id INTEGER PRIMARY KEY, abstract TEXT);
            CREATE TABLE articles_order(article_id INTEGER, random_rank INTEGER PRIMARY KEY);
            INSERT INTO articles VALUES (1, 2020, 'An event');
            INSERT INTO abstracts VALUES (1, 'An abstract');
            INSERT INTO articles_order VALUES (1, 1);
            """
        )
    before = digest(database)
    extractor = EventExtractor(
        f"sqlite:///{database}",
        output_dir=tmp_path / "events",
        checkpoint_path=str(tmp_path / "checkpoint.pkl"),
        n_processes=1,
    )
    frame = extractor.fetch_ordered_article_batch(None, 10)
    extractor.engine.dispose()
    assert frame["article_id"].tolist() == [1]
    assert digest(database) == before


def test_checkpoint_configuration_is_bound_to_output_directory(tmp_path):
    first = EventExtractor(
        f"sqlite:///{tmp_path / 'missing.db'}",
        output_dir=tmp_path / "first",
        checkpoint_path=str(tmp_path / "checkpoint.pkl"),
        n_processes=1,
    )
    second = EventExtractor(
        f"sqlite:///{tmp_path / 'missing.db'}",
        output_dir=tmp_path / "second",
        checkpoint_path=str(tmp_path / "checkpoint.pkl"),
        n_processes=1,
    )
    try:
        assert first._checkpoint_config()["output_dir"] != second._checkpoint_config()["output_dir"]
        legacy_config = first._checkpoint_config()
        legacy_config.pop("output_dir")
        second._validate_checkpoint_config({"version": 1, "config": legacy_config})
        with pytest.raises(ValueError, match="output_dir"):
            second._validate_checkpoint_config(
                {"version": events.CHECKPOINT_VERSION, "config": first._checkpoint_config()}
            )
        second.allow_output_dir_change = True
        second._validate_checkpoint_config(
            {"version": events.CHECKPOINT_VERSION, "config": first._checkpoint_config()}
        )
    finally:
        first.engine.dispose()
        second.engine.dispose()


def test_cooccurrence_counts_do_not_overflow_int8(monkeypatch):
    class SplitTokenizer:
        def __call__(self, text):
            return text.split()

    monkeypatch.setattr(events, "is_english", lambda _text: True)
    monkeypatch.setattr(events, "LemmaTokenizer", SplitTokenizer)
    events.init_cooccurrence_worker((1, 1), ("alpha", "beta"))
    articles = [
        (index, 2020, "alpha beta", "")
        for index in range(256)
    ]
    result = events.extract_cooccurrence_for_articles(articles)
    assert result["cooccurrence"][0, 1] == 256
