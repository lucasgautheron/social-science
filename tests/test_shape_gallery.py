import csv
import json
from collections.abc import Sequence
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
    _glyph_center_shift,
    _persistent_figure,
    _persistent_size_figure,
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
    outside: dict[int, int | None] | None = None,
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
        outside_by_lag={1: 1, 3: 1} if outside is None else outside,
        papers=int(round(float(papers.sum()))),
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
            outside={1: 3, 3: 0},
        ),
        _cluster(
            2,
            "other-rise",
            "steady rise",
            _share(0.002),
            diversity=0.4,
            within={1: 4, 3: 6},
            outside={1: 0, 3: 4},
        ),
        _cluster(
            3,
            "nanotube",
            "steady decline",
            _share(0.012, -0.008),
            keywords=("nanotube",),
            diversity=0.5,
            within={1: 10, 3: 8},
            outside={1: 11, 3: 2},
        ),
        _cluster(
            4,
            "nanowires",
            "accelerating decline",
            _share(0.004, -0.002),
            keywords=("nanowires",),
            diversity=0.7,
            within={1: 6, 3: 4},
            outside={1: 1, 3: 0},
        ),
        _cluster(
            5,
            "nanopowders",
            "flattening decline",
            _share(0.002, -0.00005),
            keywords=("nanopowders",),
            diversity=None,
            within={1: 2, 3: 1},
            outside={1: 0, 3: 5},
        ),
        _cluster(
            6,
            "hydroxychloroquine",
            "shock: transient",
            _share(0.001, spike=shock_index),
            diversity=0.6,
            within={1: 0, 3: 3},
            outside={1: 8, 3: 0},
        ),
        _cluster(
            7,
            "other-shock",
            "shock: transient",
            _share(0.001, spike=shock_index),
            diversity=0.8,
            within={1: 5, 3: 7},
            outside={1: 0, 3: 6},
        ),
        _cluster(
            8,
            "covid",
            "shock: lasting",
            _share(0.003, 0.004),
            diversity=0.9,
            within={1: 9, 3: 5},
            outside={1: 2, 3: 7},
        ),
        _cluster(
            9,
            "covid-19",
            "shock: lasting",
            _share(0.002, 0.001),
            keywords=("covid-19",),
            diversity=0.3,
            within={1: 3, 3: 1},
            outside={1: 4, 3: 1},
        ),
    ]


def _ink_lines(axis):
    return [line for line in axis.lines if to_hex(line.get_color()) == INK]


def _labels(axis) -> dict[str, object]:
    return {text.get_text(): text for text in axis.texts}


def _ink_scatter(axis):
    points = []
    for collection in axis.collections:
        colors = collection.get_facecolors()
        if len(colors) == 0 or to_hex(colors[0]) != INK:
            continue
        points.extend((float(x), float(y)) for x, y in collection.get_offsets())
    return points


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
    panels = [axis for axis in figure.axes if axis.get_label() != "frequency"]
    counts = [
        sum(len(collection.get_segments()) for collection in axis.collections)
        for axis in panels
    ]
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
        "Share of publications",
        False,
        parse_exemplars(None),
    )
    plotted = _ink_lines(figure.axes[0])[-1].get_ydata()
    assert plotted == pytest.approx(llm.log_share)
    tick_labels = [tick.get_text() for tick in figure.axes[0].get_yticklabels() if tick.get_text()]
    assert tick_labels
    assert all(label.startswith(r"$10^{") and label.endswith(r"}$") for label in tick_labels)
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
    assert rising["llm"].xy[0] == pytest.approx(0.0)
    assert rising["llm"].xyann[0] == pytest.approx(4)
    assert rising["llm"].get_va() == "top"
    assert declining["nanotube"].get_va() == "bottom"
    assert declining["nanowires"].get_va() == "top"
    assert "nanopowders" not in declining
    assert len(_ink_lines(figure.axes[1])) == 2
    assert _labels(figure.axes[3])["covid"].get_ha() == "left"
    assert "covid-19" not in _labels(figure.axes[3])
    figure.clf()


def test_frequency_inset_plots_raw_share_in_a_lower_right_square():
    series = _corpus()
    llm = series[0]
    nanowires = next(row for row in series if row.label == "nanowires")
    nanotube = next(row for row in series if row.label == "nanotube")
    figure = _gallery_figure(
        series,
        YEARS,
        "log_share",
        "Share of publications",
        False,
        parse_exemplars(None),
    )
    figure.canvas.draw()
    insets = [axis for axis in figure.axes if axis.get_label() == "frequency"]
    assert len(insets) == 4
    assert all(axis.get_yscale() == "linear" for axis in insets)
    rising = _ink_lines(insets[0])
    declining = _ink_lines(insets[1])
    assert rising[-1].get_ydata() == pytest.approx(llm.share)
    assert declining[0].get_ydata() == pytest.approx(nanowires.share)
    assert declining[1].get_ydata() == pytest.approx(nanotube.share)
    width, height = figure.get_size_inches()
    for parent, inset in zip(figure.axes[:4], insets, strict=True):
        panel = parent.get_position()
        box = inset.get_position()
        assert box.width * width == pytest.approx(box.height * height, rel=0.02)
        assert box.x0 > panel.x0 + 0.45 * panel.width
        assert box.y1 < panel.y0 + 0.55 * panel.height
    figure.clf()


def test_diversity_mean_matches_the_plotted_clusters():
    figure = _diversity_figure(_corpus())
    labels = _labels(figure.axes[0])
    assert labels["0.300"].get_text() == "0.300"
    assert "0.600" in labels
    assert labels["llm"].get_va() == "baseline"
    assert labels["llm"].xyann[1] == pytest.approx(_glyph_center_shift("llm", 15))
    assert labels["llm"].xy[1] == pytest.approx(0.2)
    assert labels["nanowires"].xy[1] == pytest.approx(0.7)
    assert labels["nanotube"].xy[1] == pytest.approx(0.5)
    assert labels["hydroxychloroquine"].xy[1] == pytest.approx(0.6)
    assert labels["covid"].xy[1] == pytest.approx(0.9)
    scatter = _ink_scatter(figure.axes[0])
    assert sorted(point[1] for point in scatter) == pytest.approx([0.2, 0.5, 0.6, 0.7, 0.9])
    centers = [float(line.get_xdata().mean()) for line in figure.axes[0].lines]
    assert all(
        any(point[0] == pytest.approx(center) for center in centers) for point in scatter
    )
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
    assert "llm" not in _labels(axis)
    assert "hydroxychloroquine" not in _labels(axis)
    assert sorted(point[1] for point in _ink_scatter(axis)) == pytest.approx([6.0, 9.0, 10.0])
    centers = [float(line.get_xdata().mean()) for line in axis.lines]
    assert all(
        any(point[0] == pytest.approx(center) for center in centers)
        for point in _ink_scatter(axis)
    )
    figure.clf()


def test_size_scatter_plots_within_counts_against_cluster_size():
    series = _corpus()
    figure = _persistent_size_figure(series, lag=1)
    axis = figure.axes[0]
    assert axis.get_xscale() == "log"
    assert axis.get_yscale() == "log"
    assert axis.get_xlabel() == "Cluster size"
    offsets = [point for collection in axis.collections for point in collection.get_offsets()]
    plotted = {int(round(point[1])) for point in offsets}
    assert plotted == {2, 3, 4, 5, 6, 9, 10}
    assert "llm" not in _labels(axis)
    assert "hydroxychloroquine" not in _labels(axis)
    nanotube = next(row for row in series if row.label == "nanotube")
    nanotube_point = next(point for point in offsets if int(round(point[1])) == 10)
    assert nanotube_point[0] == pytest.approx(nanotube.papers)
    figure.clf()


def test_outside_repeats_use_the_outside_count():
    series = _corpus()
    figure = _persistent_figure(series, lag=1, kind="outside")
    axis = figure.axes[0]
    assert axis.get_ylabel() == "New links repeated outside the bubble"
    assert "n=1" in axis.get_xticklabels()[0].get_text()
    assert "n=2" in axis.get_xticklabels()[1].get_text()
    assert _labels(axis)["8.0"].get_text() == "8.0"
    assert "llm" in _labels(axis)
    assert "hydroxychloroquine" in _labels(axis)
    assert "other-rise" not in _labels(axis)
    figure.clf()
    scatter = _persistent_size_figure(series, lag=1, kind="outside")
    offsets = [
        point for collection in scatter.axes[0].collections for point in collection.get_offsets()
    ]
    assert {int(round(point[1])) for point in offsets} == {1, 2, 3, 4, 8, 11}
    assert scatter.axes[0].get_ylabel() == "New links repeated outside the bubble"
    scatter.clf()


def _slide_names(pdf_names: Sequence[str]) -> list[str]:
    return [name for pdf in pdf_names for name in (pdf, pdf.replace(".pdf", ".png"))]


def _png_dpi(path: Path) -> float:
    data = path.read_bytes()
    offset = 8
    while offset + 12 <= len(data):
        length = int.from_bytes(data[offset : offset + 4], "big")
        kind = data[offset + 4 : offset + 8]
        chunk = data[offset + 8 : offset + 8 + length]
        if kind == b"pHYs":
            return int.from_bytes(chunk[:4], "big") * 0.0254
        offset += 12 + length
    raise AssertionError(f"{path} has no PNG resolution chunk")


def test_write_shape_gallery_writes_pdfs_and_pngs(tmp_path: Path):
    paths = write_shape_gallery(_corpus(), YEARS, tmp_path)
    assert [path.name for path in paths] == _slide_names(OUTPUT_NAMES)
    for path in paths:
        header = path.read_bytes()[:8]
        if path.suffix == ".pdf":
            assert header.startswith(b"%PDF")
        else:
            assert header.startswith(b"\x89PNG\r\n\x1a\n")
            assert _png_dpi(path) == pytest.approx(300, abs=0.1)


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
                    [{"lag": 1, "within_count": 8, "outside_count": 5}]
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
    assert series[0].outside_by_lag == {1: 5}
    assert series[0].papers == 10


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
    assert [path.name for path in paths] == _slide_names(OUTPUT_NAMES[:5])
    assert not (tmp_path / "figures" / "persistent_new_links_y_plus_1.pdf").exists()
    assert not (tmp_path / "figures" / "persistent_new_links_y_plus_1.png").exists()
    assert not (tmp_path / "figures" / "outside_new_links_y_plus_1.pdf").exists()
    assert not (tmp_path / "figures" / "outside_new_links_y_plus_1.png").exists()
