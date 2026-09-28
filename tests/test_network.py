import hashlib
import json
import sqlite3

import numpy as np
import pytest

from openalex.analysis.network import build_yearly_coauthorship, load_coauthorship


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_corpus(path):
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE articles(
                article_id INTEGER PRIMARY KEY,
                publication_year INTEGER
            );
            CREATE TABLE articles_authors(
                article_id INTEGER,
                author_id INTEGER,
                PRIMARY KEY (article_id, author_id)
            );
            INSERT INTO articles VALUES
                (1, 2020),
                (2, 2020),
                (3, 2021),
                (4, 2020),
                (5, 2020);
            INSERT INTO articles_authors VALUES
                (1, 1), (1, 2), (1, 3),
                (2, 1), (2, 2),
                (3, 2), (3, 4),
                (4, 5),
                (5, 10), (5, 11), (5, 12), (5, 13);
            """
        )


def test_yearly_weights_alignment_and_read_only(tmp_path):
    database = tmp_path / "articles.db"
    create_corpus(database)
    before = digest(database)
    output = tmp_path / "coauthorship"
    build_yearly_coauthorship(
        database,
        output,
        max_authors=3,
        fetch_size=1,
    )
    assert digest(database) == before

    author_ids, matrix_2020 = load_coauthorship(output, 2020)
    assert author_ids.tolist() == [1, 2, 3, 4]
    _, matrix_2021 = load_coauthorship(output, 2021)
    assert matrix_2020.shape == matrix_2021.shape == (4, 4)
    assert matrix_2020.dtype == np.float64

    index = {int(author_id): position for position, author_id in enumerate(author_ids)}
    dense_2020 = matrix_2020.toarray()
    dense_2021 = matrix_2021.toarray()
    assert dense_2020[index[1], index[2]] == pytest.approx(1.5)
    assert dense_2020[index[1], index[3]] == pytest.approx(0.5)
    assert dense_2020[index[2], index[3]] == pytest.approx(0.5)
    assert dense_2020[index[2], index[4]] == 0
    assert dense_2021[index[2], index[4]] == pytest.approx(1)
    assert dense_2021[index[1], index[2]] == 0
    assert np.array_equal(dense_2020, dense_2020.T)
    assert np.array_equal(dense_2021, dense_2021.T)
    assert np.all(np.diag(dense_2020) == 0)
    assert np.all(np.diag(dense_2021) == 0)
    assert 5 not in index
    assert 10 not in index

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["weight"] == "newman"
    assert manifest["solo_papers"]["2020"] == 1
    assert manifest["hyperauthored_papers"]["2020"] == 1
    assert manifest["papers_kept"]["2020"] == 2
    assert manifest["completed_years"] == [2020, 2021]
    assert not (output / "scratch").exists()


def test_resume_does_not_rewrite_finished_year(tmp_path):
    database = tmp_path / "articles.db"
    create_corpus(database)
    output = tmp_path / "coauthorship"
    build_yearly_coauthorship(database, output, max_authors=3, fetch_size=1)
    matrix_path = output / "years" / "2020.npz"
    before = digest(matrix_path)
    modified = matrix_path.stat().st_mtime_ns
    database.unlink()

    build_yearly_coauthorship(database, output, max_authors=3, fetch_size=1, resume=True)

    assert digest(matrix_path) == before
    assert matrix_path.stat().st_mtime_ns == modified


def test_max_edges_rejects_year(tmp_path):
    database = tmp_path / "articles.db"
    create_corpus(database)
    with pytest.raises(ValueError, match="2020") as caught:
        build_yearly_coauthorship(
            database,
            tmp_path / "coauthorship",
            max_authors=3,
            max_edges=5,
        )
    assert "max-authors" in str(caught.value)


def test_refuses_to_overwrite_without_resume(tmp_path):
    database = tmp_path / "articles.db"
    create_corpus(database)
    output = tmp_path / "coauthorship"
    build_yearly_coauthorship(database, output, max_authors=3)
    with pytest.raises(FileExistsError, match="--resume"):
        build_yearly_coauthorship(database, output, max_authors=3)
