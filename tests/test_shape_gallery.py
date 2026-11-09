import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from matplotlib.colors import to_hex

from openalex.analysis.cluster_trends import DECLINE_CLASSES, RISE_CLASSES
from openalex.cli import COMMANDS
from openalex.visualizations.shape_gallery import (
    INK,
    OUTPUT_NAMES,
    ClusterSeries,
    _diversity_figure,
    _gallery_figure,
    _persistent_figure,
    keyword_match,
    load_trends,
    log_share,
    parse_exemplars,
    scale_series,
    select_highlights,
    write_shape_gallery,
)

matplotlib.use("Agg")

YEARS = np.arange(2010, 2026)


def _share(level: float, slope: float = 0.0, spike: int | None = None) -> np.ndarray:
    values = level + slope * np.linspace(-1.0, 1.0, YEARS.size)
    if spike is not None:
        values = np.full(YEARS.size, level)
        values[spike] = level + 0.05
    return values


def _cluster(
    group: int,
    label: str,
    subclass: str,
    share: np.ndarray,
    *,
    keywords: tuple[str, ...] | None = None,
    diversity: float | None = 0.5,
    within: dict[int, int | None] | None = None,
    adjusted_shift: float = 3.0,
) -> ClusterSeries:
    words = keywords if keywords is not None else (label,)
    papers = np.where(share > 0, share * 100.0, 0.0)
    return ClusterSeries(
        group=group,
        label=label,
        keywords=words,
        subclass=subclass,
        share=share,
        log_share=log_share(papers, share),
        adjusted=share + adjusted_shift,
        diversity=diversity,
        within_by_lag={1: 4, 3: 2} if within is None else within,
    )


def _corpus() -> list[ClusterSeries]:
    shock_index = int(np.where(YEARS == 2020)[0][0])
    return [
        _cluster(
            1,
            "llm",
            "accelerating rise",
            _share(0.004, 0.003),
            diversity=0.2,
            within={1: 0, 3: 2},
        ),
        _cluster(
            2,
            "other-rise",
            "steady rise",
            _share(0.002),
            diversity=0.4,
            within={1: 4, 3: 6},
        ),
        _cluster(
            3,
            "nanotube",
            "steady decline",
            _share(0.012, -0.008),
            keywords=("nanotube",),
            diversity=0.5,
            within={1: 10, 3: 8},
        ),
        _cluster(
            4,
            "nanowires",
            "accelerating decline",
            _share(0.004, -0.002),
            keywords=("nanowires",),
            diversity=0.7,
            within={1: 6, 3: 4},
        ),
        _cluster(
            5,
            "nanopowders",
            "flattening decline",
            _share(0.002, -0.00005),
            keywords=("nanopowders",),
            diversity=None,
            within={1: 2, 3: 1},
        ),
        _cluster(
            6,
            "hydroxychloroquine",
            "shock: transient",
            _share(0.001, spike=shock_index),
            diversity=0.6,
            within={1: 0, 3: 3},
        ),
        _cluster(
            7,
            "other-shock",
            "shock: transient",
            _share(0.001, spike=shock_index),
            diversity=0.8,
            within={1: 5, 3: 7},
        ),
        _cluster(
            8,
            "covid",
            "shock: lasting",
            _share(0.003, 0.004),
            diversity=0.9,
            within={1: 9, 3: 5},
        ),
        _cluster(
            9,
            "covid-19",
            "shock: lasting",
            _share(0.002, 0.001),
            keywords=("covid-19",),
            diversity=0.3,
            within={1: 3, 3: 1},
        ),
    ]


def _ink_lines(axis):
    return [line for line in axis.lines if to_hex(line.get_color()) == INK]


def _labels(axis) -> dict[str, object]:
    return {text.get_text(): text for text in axis.texts}


def test_command_is_registered():
    assert COMMANDS["visualize-shapes"] == "openalex.visualizations.shape_gallery"


def test_log_share_uses_the_one_paper_pseudocount():
    logged = log_share(np.array([0.0, 10.0]), np.array([0.0, 0.1]))
    assert logged[0] == pytest.approx(np.log(1.0 / 100.0))
    assert logged[1] == pytest.approx(np.log(11.0 / 100.0))


def test_keyword_match_is_a_whole_keyword_glob():
    assert keyword_match("nano*", ["nanotube"])
    assert not keyword_match("covid", ["covid-19"])
    assert keyword_match("covid", ["COVID"])


def test_nano_keeps_the_two_largest_share_ranges():
    series = _corpus()
    chosen = select_highlights(series, DECLINE_CLASSES, "nano*")
    assert [row.label for row in chosen] == ["nanowires", "nanotube"]
    with pytest.raises(ValueError, match="other classes"):
        select_highlights(series, RISE_CLASSES, "hydroxychloroquine")


def test_rise_and_decline_subclasses_share_a_panel():
    figure = _gallery_figure(
        _corpus(),
        YEARS,
        "share_scaled",
        "Scaled share of publications",
        True,
        parse_exemplars(None),
    )
    counts = [sum(len(collection.get_segments()) for collection in axis.collections) for axis in figure.axes]
    assert counts == [2, 3, 2, 2]
    assert figure.axes[0].get_title().startswith("Rising")
    assert "Declining" in figure.axes[1].get_title()
    figure.clf()


def test_normalized_share_is_not_the_topic_adjusted_series():
    series = _corpus()
    llm = series[0]
    figure = _gallery_figure(
        series,
        YEARS,
        "share_scaled",
        "Scaled share of publications",
        True,
        parse_exemplars(None),
    )
    plotted = _ink_lines(figure.axes[0])[-1].get_ydata()
    assert plotted == pytest.approx(scale_series(llm.share))
    assert plotted != pytest.approx(llm.adjusted)
    figure.clf()


def test_topic_adjusted_panel_uses_stored_activity_on_calendar_years():
    series = _corpus()
    shock = next(row for row in series if row.label == "hydroxychloroquine")
    figure = _gallery_figure(
        series,
        YEARS,
        "adjusted",
        "Topic-adjusted log share",
        False,
        parse_exemplars(None),
    )
    line = _ink_lines(figure.axes[2])[-1]
    assert line.get_xdata() == pytest.approx(YEARS)
    assert line.get_ydata() == pytest.approx(shock.adjusted)
    peak = int(np.argmax(line.get_ydata()))
    assert YEARS[peak] == 2020
    assert peak not in {0, YEARS.size - 1}
    figure.clf()


def test_unnormalized_gallery_plots_log_share():
    series = _corpus()
    llm = series[0]
    figure = _gallery_figure(
        series,
        YEARS,
        "log_share",
        "Log share of publications",
        False,
        parse_exemplars(None),
    )
    plotted = _ink_lines(figure.axes[0])[-1].get_ydata()
    assert plotted == pytest.approx(llm.log_share)
    figure.clf()


def test_highlight_labels_are_left_anchored():
    series = _corpus()
    figure = _gallery_figure(
        series,
        YEARS,
        "share_scaled",
        "Scaled share of publications",
        True,
        parse_exemplars(None),
    )
    rising = _labels(figure.axes[0])
    declining = _labels(figure.axes[1])
    assert rising["llm"].get_ha() == "left"
    assert rising["llm"].get_va() == "top"
    assert declining["nanotube"].get_va() == "bottom"
    assert declining["nanowires"].get_va() == "top"
    assert "nanopowders" not in declining
    assert len(_ink_lines(figure.axes[1])) == 2
    assert _labels(figure.axes[3])["covid"].get_ha() == "left"
    assert "covid-19" not in _labels(figure.axes[3])
    figure.clf()


def test_diversity_mean_matches_the_plotted_clusters():
    figure = _diversity_figure(_corpus())
    labels = _labels(figure.axes[0])
    assert labels["0.300"].get_text() == "0.300"
    assert "0.600" in labels
    tick = figure.axes[0].get_xticklabels()[1].get_text()
    assert "n=2" in tick
    figure.clf()


def test_zero_repeat_counts_are_omitted_from_the_log_axis():
    figure = _persistent_figure(_corpus(), lag=1)
    axis = figure.axes[0]
    assert axis.get_yscale() == "log"
    assert "n=1" in axis.get_xticklabels()[0].get_text()
    assert "n=3" in axis.get_xticklabels()[1].get_text()
    assert "n=1" in axis.get_xticklabels()[2].get_text()
    assert _labels(axis)["4.0"].get_text() == "4.0"
    assert _labels(axis)["6.0"].get_text() == "6.0"
    figure.clf()


def test_write_shape_gallery_writes_seven_pdfs(tmp_path: Path):
    paths = write_shape_gallery(_corpus(), YEARS, tmp_path)
    assert [path.name for path in paths] == list(OUTPUT_NAMES)
    for path in paths:
        assert path.read_bytes().startswith(b"%PDF")


def test_load_trends_rebuilds_the_year_grid(tmp_path: Path):
    clusters = tmp_path / "event_clusters"
    clusters.mkdir()
    with (clusters / "cluster_by_year.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["level", "group", "year", "papers", "share"]
        )
        writer.writeheader()
        writer.writerow(
            {"level": 0, "group": 4, "year": 2010, "papers": 0, "share": 0.0}
        )
        writer.writerow(
            {"level": 0, "group": 4, "year": 2011, "papers": 10, "share": 0.1}
        )
    with (clusters / "cluster_trends.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["level", "group", "keywords", "sub", "activity"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "level": 0,
                "group": 4,
                "keywords": "nanotube; swnts",
                "sub": "steady decline",
                "activity": "[1.0, 2.0]",
            }
        )
    topics = tmp_path / "topics"
    topics.mkdir()
    with (topics / "cluster_topic_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["cluster_id", "author_rao_stirling_diversity"]
        )
        writer.writeheader()
        writer.writerow({"cluster_id": 4, "author_rao_stirling_diversity": "0.4"})
    links = tmp_path / "links"
    links.mkdir()
    with (links / "cluster_link_distance_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["cluster_id", "repeat_link_probability_by_lag"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "cluster_id": 4,
                "repeat_link_probability_by_lag": json.dumps(
                    [{"lag": 1, "within_count": 8}]
                ),
            }
        )
    series, years, persistent = load_trends(
        clusters / "cluster_trends.csv",
        clusters,
        topics_dir=topics,
        links_dir=links,
    )
    assert persistent is True
    assert years.tolist() == [2010, 2011]
    assert series[0].log_share[0] == pytest.approx(np.log(0.01))
    assert series[0].keywords == ("nanotube", "swnts")
    assert series[0].adjusted.tolist() == [1.0, 2.0]
    assert series[0].diversity == pytest.approx(0.4)
    assert series[0].within_by_lag == {1: 8}


def test_missing_repeat_column_skips_link_violins(tmp_path: Path):
    clusters = tmp_path / "event_clusters"
    clusters.mkdir()
    with (clusters / "cluster_by_year.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["level", "group", "year", "papers", "share"]
        )
        writer.writeheader()
        writer.writerow(
            {"level": 0, "group": 4, "year": 2010, "papers": 0, "share": 0.0}
        )
        writer.writerow(
            {"level": 0, "group": 4, "year": 2011, "papers": 10, "share": 0.1}
        )
    with (clusters / "cluster_trends.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["level", "group", "keywords", "sub", "activity"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "level": 0,
                "group": 4,
                "keywords": "nanotube",
                "sub": "steady decline",
                "activity": "[1.0, 2.0]",
            }
        )
    topics = tmp_path / "topics"
    topics.mkdir()
    with (topics / "cluster_topic_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["cluster_id", "author_rao_stirling_diversity"]
        )
        writer.writeheader()
        writer.writerow({"cluster_id": 4, "author_rao_stirling_diversity": "0.4"})
    links = tmp_path / "links"
    links.mkdir()
    with (links / "cluster_link_distance_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cluster_id", "paper_count"])
        writer.writeheader()
        writer.writerow({"cluster_id": 4, "paper_count": "10"})
    _series, _years, persistent = load_trends(
        clusters / "cluster_trends.csv",
        clusters,
        topics_dir=topics,
        links_dir=links,
    )
    assert persistent is False
    _series, _years, missing_file = load_trends(
        clusters / "cluster_trends.csv",
        clusters,
        topics_dir=topics,
        links_dir=tmp_path / "missing",
    )
    assert missing_file is False
    paths = write_shape_gallery(_corpus(), YEARS, tmp_path / "figures", persistent=False)
    assert [path.name for path in paths] == list(OUTPUT_NAMES[:5])
    assert not (tmp_path / "figures" / "persistent_new_links_y_plus_1.pdf").exists()
