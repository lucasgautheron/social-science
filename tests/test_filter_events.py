import csv
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from openalex.analysis.filter_events import (
    MODEL,
    Classification,
    KeywordContext,
    Neighbor,
    RequestThrottle,
    classification_from_response,
    classify_keyword,
    filter_events,
    render_prompt,
    top_correlated_indices,
)
from openalex.cli import COMMANDS
from openalex.website.build import l2_normalize_cooccurrence


def write_event_fixture(root):
    root.mkdir()
    incidence = root / "incidence"
    incidence.mkdir()
    vocabulary = np.array(["alpha", "beta", "gamma", "delta"], dtype=np.str_)
    matrix = sparse.csr_matrix(
        [
            [5, 4, 0, 0],
            [4, 5, 0, 0],
            [0, 0, 5, 1],
            [0, 0, 1, 4],
        ],
        dtype=np.int64,
    )
    frequencies = np.array([5, 5, 5, 4], dtype=np.int64)
    sparse.save_npz(root / "event_cooccurrence.npz", matrix)
    np.save(root / "event_vocabulary.npy", vocabulary, allow_pickle=False)
    np.save(root / "event_document_frequency.npy", frequencies, allow_pickle=False)
    sparse.save_npz(incidence / "part_000001_000.npz", sparse.csr_matrix((1, 4), dtype=np.int8))
    np.save(
        incidence / "part_000001_000_years.npy",
        np.array([2020], dtype=np.int32),
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
            }
        ),
        encoding="utf-8",
    )
    (root / "events.csv").write_text(
        "ngram,fold_change\n"
        "gamma,4\n"
        "alpha,3\n"
        "beta,2\n",
        encoding="utf-8",
    )


def test_neighbors_match_dendrogram_row_cosine():
    matrix = sparse.csr_matrix(
        [
            [5, 4, 0, 0],
            [4, 5, 0, 0],
            [0, 0, 5, 1],
            [0, 0, 1, 4],
        ],
        dtype=np.int64,
    )
    normalized = l2_normalize_cooccurrence(matrix)
    similarities = (normalized @ normalized.T).toarray()
    ranked = top_correlated_indices(matrix, 3)
    for row, neighbors in enumerate(ranked):
        expected = [
            int(index)
            for index in np.argsort(-similarities[row], kind="stable")
            if index != row
        ][:3]
        assert [index for index, _similarity in neighbors] == expected
        assert neighbors[0][1] == pytest.approx(similarities[row, expected[0]])
    assert [index for index, _similarity in ranked[0]] == [1, 2, 3]


def test_prompt_shows_the_keyword_and_three_neighbors():
    context = KeywordContext(
        ngram="alpha",
        neighbors=(
            Neighbor("beta", 0.975609756),
            Neighbor("gamma", 0.0),
            Neighbor("delta", 0.0),
        ),
        event={},
    )
    prompt = render_prompt(context)
    assert prompt.splitlines() == [
        "Keyword: alpha",
        "Most correlated keywords:",
        "1. beta (0.976)",
        "2. gamma (0.000)",
        "3. delta (0.000)",
    ]


def test_response_parser_ignores_extra_message_text():
    response = SimpleNamespace(
        output_text="working notes, not the verdict",
        output=[
            SimpleNamespace(content=[SimpleNamespace(text="We need output JSON schema.")]),
            SimpleNamespace(
                content=[
                    SimpleNamespace(text='{"label": "artefact", "reason": "Non-English fragment."}')
                ]
            ),
        ],
    )
    verdict = classification_from_response("gamma", response)
    assert verdict.label == "artefact"
    assert verdict.reason == "Non-English fragment."


def test_classify_keyword_sends_neighbors_to_gpt_luna():
    captured = {}

    class Responses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                output=[
                    SimpleNamespace(
                        content=[
                            SimpleNamespace(
                                text='{"label": "genuine", "reason": "A method that spread."}'
                            )
                        ]
                    )
                ]
            )

    context = KeywordContext(
        ngram="alpha",
        neighbors=(Neighbor("beta", 0.9756), Neighbor("gamma", 0.0), Neighbor("delta", 0.0)),
        event={},
    )
    verdict = classify_keyword(context, client=SimpleNamespace(responses=Responses()), max_retries=1)
    assert captured["model"] == MODEL
    assert captured["reasoning"] == {"effort": "low"}
    assert "Keyword: alpha" in captured["input"][1]["content"]
    assert "1. beta (0.976)" in captured["input"][1]["content"]
    assert verdict.label == "genuine"


def test_requests_start_at_least_one_interval_apart():
    starts = []

    class Responses:
        def create(self, **kwargs):
            starts.append(time.monotonic())
            return SimpleNamespace(
                output=[
                    SimpleNamespace(
                        content=[
                            SimpleNamespace(text='{"label": "genuine", "reason": "A topic."}')
                        ]
                    )
                ]
            )

    context = KeywordContext(ngram="alpha", neighbors=(), event={})
    client = SimpleNamespace(responses=Responses())
    throttle = RequestThrottle(0.05)
    classify_keyword(context, client=client, max_retries=1, throttle=throttle)
    classify_keyword(context, client=client, max_retries=1, throttle=throttle)
    assert starts[1] - starts[0] >= 0.05


def test_filter_events_classifies_each_extracted_keyword(tmp_path):
    events = tmp_path / "events"
    output = tmp_path / "filtered"
    write_event_fixture(events)
    seen = []

    def classify(context):
        seen.append(render_prompt(context))
        label = "artefact" if context.ngram == "gamma" else "genuine"
        return Classification(ngram=context.ngram, label=label, reason=f"judged {context.ngram}")

    summary = filter_events(events, output, classify=classify, workers=1)
    assert summary["keywords"] == 3
    assert summary["genuine"] == 2
    assert summary["artefact"] == 1
    assert summary["model"] == MODEL
    assert [line for prompt in seen if prompt.startswith("Keyword: alpha") for line in prompt.splitlines()][2:5] == [
        "1. beta (0.976)",
        "2. gamma (0.000)",
        "3. delta (0.000)",
    ]
    assert "Keyword: delta" not in "\n".join(seen)

    with (output / "classifications.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["ngram"] for row in rows] == ["gamma", "alpha", "beta"]
    assert rows[1]["label"] == "genuine"
    assert rows[1]["correlated_1"] == "beta"
    assert rows[1]["fold_change"] == "3"
    assert rows[0]["label"] == "artefact"

    again = []
    filter_events(
        events,
        output,
        classify=lambda context: again.append(context.ngram),
        workers=1,
        resume=True,
    )
    assert again == []


def test_missing_filtered_directory_is_reported(tmp_path):
    from openalex.analysis.filter_events import find_filtered_events

    with pytest.raises(FileNotFoundError, match="classifications.csv"):
        find_filtered_events(tmp_path / "events", tmp_path / "missing")


def test_filter_events_command_is_registered():
    assert COMMANDS["filter-events"] == "openalex.analysis.filter_events"
