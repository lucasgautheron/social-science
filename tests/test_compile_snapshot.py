import gzip
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from openalex.db import compile_from_snapshot


def snapshot_work(work_id: str) -> dict:
    return {
        "id": f"https://openalex.org/{work_id}",
        "title": "A test paper",
        "display_name": "A test paper",
        "publication_year": 2020,
        "publication_date": "2020-01-01",
        "language": "en",
        "type": "article",
        "primary_location": {"source": None},
        "primary_topic": {
            "id": "https://openalex.org/T1",
            "display_name": "Testing",
            "domain": {"id": "https://openalex.org/domains/1"},
            "field": {"id": "https://openalex.org/fields/10"},
            "subfield": {"id": "https://openalex.org/subfields/100"},
        },
        "topics": [
            {
                "id": "https://openalex.org/T1",
                "display_name": "Testing",
                "score": 1.0,
            }
        ],
        "concepts": [],
        "authorships": [],
        "abstract_inverted_index": {"A": [0], "test": [1]},
        "referenced_works": [],
    }


class CompileSnapshotTest(unittest.TestCase):
    def test_replaying_snapshot_file_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot = root / "snapshot"
            work_dir = root / "work"
            snapshot.mkdir()
            work_dir.mkdir()
            source = snapshot / "part_000.gz"
            with gzip.open(source, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(snapshot_work("W1")) + "\n")
            database = root / "articles.db"
            database_url = f"sqlite:///{database}"
            prior_cwd = Path.cwd()
            try:
                os.chdir(work_dir)
                compile_from_snapshot.main(
                    [str(snapshot), "--database-url", database_url]
                )
                (work_dir / "processed").unlink()
                compile_from_snapshot.main(
                    [str(snapshot), "--database-url", database_url]
                )
            finally:
                os.chdir(prior_cwd)

            connection = sqlite3.connect(database)
            try:
                article_count = connection.execute(
                    "SELECT COUNT(*) FROM articles"
                ).fetchone()[0]
                topic_count = connection.execute(
                    "SELECT COUNT(*) FROM articles_topics"
                ).fetchone()[0]
            finally:
                connection.close()

        self.assertEqual(article_count, 1)
        self.assertEqual(topic_count, 1)


if __name__ == "__main__":
    unittest.main()
