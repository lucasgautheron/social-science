import csv
import json

import numpy as np
import pytest

from openalex.analysis.cluster_trends import (
    DECLINE_CLASSES,
    FIELDNAMES,
    RISE_CLASSES,
    classify,
    classify_cluster_list,
    fit_cluster_trends,
    link_distance_class,
    log_activity,
    topic_adjust,
)
from openalex.cli import COMMANDS

YEARS = np.arange(2010, 2026)


def test_command_is_registered():
    assert COMMANDS["cluster-trends"] == "openalex.analysis.cluster_trends"


def test_constant_series_is_stable():
    row = classify(np.zeros(YEARS.size), YEARS)
    assert row["sub"] == "stable"
    assert row["family"] == "smooth"
    assert row["trend"] == pytest.approx(0.0, abs=1e-8)
    assert row["burstiness"] == pytest.approx(0.0, abs=1e-8)


def test_linear_ramp_is_a_steady_rise():
    row = classify(np.linspace(0.0, 1.5, YEARS.size), YEARS)
    assert row["sub"] == "steady rise"
    assert row["cls"] == "linear"
    assert row["trend"] > 0
    assert row["burstiness"] < 0.15


def test_short_pulse_is_a_transient_shock():
    index = np.arange(YEARS.size, dtype=float)
    onset = 6
    pulse = np.where(index >= onset, np.exp(-1.5 * (index - onset)), 0.0)
    row = classify(1.4 * pulse, YEARS)
    assert row["sub"] == "shock: transient"
    assert row["family"] == "abrupt"
    assert row["onset"] == 2016
    assert row["half_life"] == pytest.approx(np.log(2) / 1.5, rel=0.05)
    assert row["burstiness"] >= 0.15


def test_topic_reference_removes_a_shared_trajectory():
    length = YEARS.size
    drift = np.linspace(-0.8, 0.8, length)
    activity = np.vstack([drift + offset for offset in np.linspace(-0.2, 0.2, 15)])
    distributions = [[{"topic_id": 7, "share": 1.0, "papers": 10}] for _ in range(15)]
    adjusted = topic_adjust(activity, distributions)
    assert np.std(adjusted, axis=1).max() == pytest.approx(0.0, abs=1e-8)

    other = np.vstack([activity, np.zeros(length)])
    other_distributions = [*distributions, []]
    adjusted_other = topic_adjust(other, other_distributions)
    assert np.std(adjusted_other[-1]) > 0.2


def test_log_activity_uses_a_fixed_pseudocount_and_median_corpus_size():
    count = np.array([[0.0, 10.0, 40.0]])
    share = np.array([[0.0, 0.1, 0.2]])
    logged = log_activity(count, share)
    total = np.array([100.0, 100.0, 200.0])
    total[0] = np.median([100.0, 200.0])
    expected = np.log((count + 1.0) / total)
    assert logged == pytest.approx(expected)


def test_link_distance_class_collapses_rise_and_decline():
    assert link_distance_class("steady rise") == "rise"
    assert link_distance_class("accelerating rise") == "rise"
    assert link_distance_class("saturating rise") == "rise"
    assert link_distance_class("steady decline") == "decline"
    assert link_distance_class("accelerating decline") == "decline"
    assert link_distance_class("flattening decline") == "decline"
    assert link_distance_class("shock: transient") == "shock: transient"
    assert RISE_CLASSES.isdisjoint(DECLINE_CLASSES)


def test_quadratic_turn_is_classified_by_net_direction():
    years = np.arange(2000, 2016)
    position = np.linspace(-1.0, 1.0, years.size)
    row = classify(position**2 + 0.2 * position, years)
    assert row["cls"] == "quad"
    assert row["sub"] == "accelerating rise"


def test_short_cluster_list_stays_unclassified():
    clusters = [
        {
            "group": 0,
            "keywords": ["alpha"],
            "yearly": [
                {"year": 2020, "papers": 10, "share": 0.1},
                {"year": 2021, "papers": 12, "share": 0.1},
            ],
        }
    ]
    assert classify_cluster_list(clusters) is None


def test_cluster_list_payload_uses_the_leading_keyword():
    years = list(range(2010, 2026))
    clusters = []
    for group in range(2):
        yearly = []
        for offset, year in enumerate(years):
            papers = 1000 if group == 0 else 1000 + 80 * offset
            yearly.append({"year": year, "papers": papers, "share": papers / 100000})
        clusters.append(
            {
                "group": group,
                "keywords": ["beta", "alpha"] if group == 0 else ["gamma"],
                "yearly": yearly,
            }
        )
    payload = classify_cluster_list(clusters)
    assert payload is not None
    assert payload["years"] == years
    assert payload["series"][0]["label"] == "beta"
    assert payload["series"][0]["keywords"] == ["beta", "alpha"]
    assert payload["series"][0]["sub"] in DECLINE_CLASSES
    assert payload["series"][1]["sub"] in RISE_CLASSES
    assert len(payload["series"][1]["activity"]) == 16


def test_fit_writes_the_shape_table(tmp_path):
    clusters = tmp_path / "event_clusters"
    clusters.mkdir()
    years = list(range(2010, 2026))
    with (clusters / "cluster_by_year.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["level", "group", "year", "papers", "share"],
        )
        writer.writeheader()
        for year in years:
            papers = 500
            writer.writerow(
                {
                    "level": 0,
                    "group": 3,
                    "year": year,
                    "papers": papers,
                    "share": papers / 100000,
                }
            )
    with (clusters / "clusters.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["level", "group", "n_keywords", "keywords"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "level": 0,
                "group": 3,
                "n_keywords": 2,
                "keywords": "delta | epsilon",
            }
        )
    topics = tmp_path / "topics"
    topics.mkdir()
    with (topics / "cluster_topic_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["cluster_id", "topic_distribution"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "cluster_id": 3,
                "topic_distribution": json.dumps(
                    [{"topic_id": 1, "share": 1.0, "papers": 4}]
                ),
            }
        )

    summary = fit_cluster_trends(
        clusters,
        level=0,
        topics_dir=topics,
    )
    destination = clusters / "cluster_trends.csv"
    assert summary["output"] == str(destination.resolve())
    assert summary["clusters"] == 1
    with destination.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == list(FIELDNAMES)
    assert rows[0]["group"] == "3"
    assert rows[0]["keywords"] == "delta; epsilon"
    assert rows[0]["sub"] == "stable"
    assert json.loads(rows[0]["activity"])
    assert not destination.with_name(".cluster_trends.csv.tmp").exists()


def test_fit_rejects_a_short_year_grid(tmp_path):
    clusters = tmp_path / "event_clusters"
    clusters.mkdir()
    with (clusters / "cluster_by_year.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["level", "group", "year", "papers", "share"],
        )
        writer.writeheader()
        for year in (2020, 2021):
            writer.writerow(
                {
                    "level": 0,
                    "group": 0,
                    "year": year,
                    "papers": 10,
                    "share": 0.1,
                }
            )
    with pytest.raises(ValueError, match="8 years"):
        fit_cluster_trends(clusters, level=0)
