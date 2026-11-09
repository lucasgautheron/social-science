"""Slide figures for cluster shape classes.

Four galleries show yearly publication frequency, its logarithm, and the
topic-adjusted log share, with and without per-curve scaling. Three violins
summarize author primary-topic diversity and the number of new links that
publish again inside the originating cluster at y+1 and y+3.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

import numpy as np

from openalex.analysis.cluster_trends import (
    DECLINE_CLASSES,
    RISE_CLASSES,
    _combined_years,
    _papers_by_year,
    _read_cluster_year,
    default_events_dir,
    log_activity,
)

logger = logging.getLogger(__name__)

INK = "#172126"
SPINE = "#a9b4b1"
COLORS = {
    "steady rise": "#2e7d32",
    "accelerating rise": "#66bb6a",
    "saturating rise": "#a5d6a7",
    "steady decline": "#1565c0",
    "accelerating decline": "#42a5f5",
    "flattening decline": "#90caf9",
    "shock: transient": "#ef6c00",
    "shock: lasting": "#c62828",
}
PANEL_COLORS = ("#2e7d32", "#1565c0", "#ef6c00", "#c62828")
PANELS = (
    ("Rising", RISE_CLASSES, "llm"),
    ("Declining", DECLINE_CLASSES, "nano*"),
    ("Shocks (transient)", frozenset({"shock: transient"}), "hydroxychloroquine"),
    ("Shocks (lasting)", frozenset({"shock: lasting"}), "covid"),
)
PANEL_ALIASES = {
    "rising": "Rising",
    "declining": "Declining",
    "transient": "Shocks (transient)",
    "shocks (transient)": "Shocks (transient)",
    "lasting": "Shocks (lasting)",
    "shocks (lasting)": "Shocks (lasting)",
}
GALLERIES = (
    ("shapes_normalized.pdf", "share_scaled", "Scaled share of publications", True),
    ("shapes_unnormalized.pdf", "log_share", "Share of publications", False),
    (
        "shapes_topic_adjusted_normalized.pdf",
        "adjusted_scaled",
        "Scaled topic-adjusted log share",
        True,
    ),
    ("shapes_topic_adjusted.pdf", "adjusted", "Topic-adjusted log share", False),
)
FIGURE_SIZE = (13.33, 7.5)
OUTPUT_NAMES = (
    *(name for name, *_rest in GALLERIES),
    "author_primary_topic_diversity.pdf",
    "persistent_new_links_y_plus_1.pdf",
    "persistent_new_links_y_plus_3.pdf",
)


@dataclass(frozen=True)
class ClusterSeries:
    """One classified cluster aligned to the shared year grid."""

    group: int
    label: str
    keywords: tuple[str, ...]
    subclass: str
    share: np.ndarray
    log_share: np.ndarray
    adjusted: np.ndarray
    diversity: float | None
    within_by_lag: Mapping[int, int | None]

    @property
    def share_range(self) -> float:
        return float(np.max(self.share) - np.min(self.share))

    @property
    def share_mean(self) -> float:
        return float(np.mean(self.share))

    def values(self, key: str) -> np.ndarray:
        if key == "share_scaled":
            return scale_series(self.share)
        if key == "adjusted_scaled":
            return scale_series(self.adjusted)
        return np.asarray(getattr(self, key), dtype=float)


def scale_series(values: np.ndarray) -> np.ndarray:
    """Subtract the mean and divide by the population standard deviation."""
    observed = np.asarray(values, dtype=float)
    centered = observed - observed.mean()
    deviation = float(np.sqrt(np.mean(centered**2)))
    return centered / max(deviation, 1e-12)


def log_share(papers: np.ndarray, share: np.ndarray) -> np.ndarray:
    """Log frequency with the shapes-page one-paper pseudocount."""
    count = np.asarray(papers, dtype=float).reshape(1, -1)
    weights = np.asarray(share, dtype=float).reshape(1, -1)
    if count.shape != weights.shape:
        raise ValueError("Papers and shares must have the same length")
    logged = log_activity(count, weights)[0]
    if not np.all(np.isfinite(logged)):
        raise ValueError("Log share is not finite")
    return logged


def keyword_match(pattern: str, keywords: Sequence[str]) -> bool:
    """Match a whole keyword, case-insensitively, with ``fnmatch`` globs."""
    needle = pattern.casefold()
    return any(fnmatch(str(keyword).casefold(), needle) for keyword in keywords)


def parse_exemplars(values: Sequence[str] | None) -> dict[str, str]:
    """Return panel title to keyword pattern, starting from the defaults."""
    patterns = {title: pattern for title, _classes, pattern in PANELS}
    for item in values or []:
        if "=" not in item:
            raise ValueError(
                f"Exemplar {item!r} must look like declining=nano*"
            )
        key, pattern = item.split("=", 1)
        title = PANEL_ALIASES.get(key.strip().casefold())
        if title is None:
            known = ", ".join(sorted(PANEL_ALIASES))
            raise ValueError(f"Unknown exemplar panel {key!r}. Use one of: {known}")
        if not pattern.strip():
            raise ValueError(f"Exemplar pattern for {title} is empty")
        patterns[title] = pattern.strip()
    return patterns


def select_highlights(
    series: Sequence[ClusterSeries],
    classes: frozenset[str],
    pattern: str,
) -> list[ClusterSeries]:
    """Return the clusters to draw in black for one panel.

    ``nano*`` on the declining classes keeps the two matches with the largest
    yearly-share range. The lower mean share comes first.
    """
    matched = [
        row
        for row in series
        if row.subclass in classes and keyword_match(pattern, row.keywords)
    ]
    if not matched:
        elsewhere = [
            f"{row.label} ({row.subclass})"
            for row in series
            if keyword_match(pattern, row.keywords)
        ]
        detail = (
            " Matches in other classes: " + ", ".join(elsewhere)
            if elsewhere
            else ""
        )
        raise ValueError(f"No cluster in this panel matches {pattern!r}.{detail}")
    if pattern.casefold() == "nano*" and classes == DECLINE_CLASSES:
        ranked = sorted(
            matched,
            key=lambda row: (-row.share_range, -row.share_mean, row.label),
        )
        matched = sorted(ranked[:2], key=lambda row: (row.share_mean, row.label))
    return matched


def write_shape_gallery(
    series: Sequence[ClusterSeries],
    years: np.ndarray,
    output_dir: str | Path,
    exemplars: Mapping[str, str] | None = None,
    *,
    persistent: bool = True,
) -> list[Path]:
    """Write the slide PDFs and return their paths.

    The two persistent-link violins are omitted when ``persistent`` is false.
    """
    if not series:
        raise ValueError("At least one classified cluster is required")
    year_grid = np.asarray(years, dtype=int)
    if year_grid.ndim != 1 or year_grid.size < 2:
        raise ValueError("At least two years are required")
    for row in series:
        for name in ("share", "log_share", "adjusted"):
            values = np.asarray(getattr(row, name))
            if values.shape != year_grid.shape:
                raise ValueError(
                    f"Cluster {row.group} {name} does not match the year grid"
                )
    patterns = dict(exemplars) if exemplars is not None else parse_exemplars(None)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written = [
        _save(
            _gallery_figure(series, year_grid, key, ylabel, symmetric, patterns),
            destination / name,
        )
        for name, key, ylabel, symmetric in GALLERIES
    ]
    written.append(
        _save(
            _diversity_figure(series, patterns),
            destination / "author_primary_topic_diversity.pdf",
        )
    )
    if persistent:
        written.append(
            _save(
                _persistent_figure(series, lag=1, patterns=patterns),
                destination / "persistent_new_links_y_plus_1.pdf",
            )
        )
        written.append(
            _save(
                _persistent_figure(series, lag=3, patterns=patterns),
                destination / "persistent_new_links_y_plus_3.pdf",
            )
        )
    else:
        logger.warning(
            "Skipping persistent-link violins; repeat counts are not available"
        )
    logger.info("Wrote %s shape figures to %s", len(written), destination)
    return written


def load_trends(
    trends: str | Path,
    clusters_dir: str | Path,
    events_dir: str | Path | None = None,
    topics_dir: str | Path | None = None,
    links_dir: str | Path | None = None,
    *,
    level: int = 0,
) -> tuple[list[ClusterSeries], np.ndarray, bool]:
    """Load classified series from ``cluster_trends.csv`` and its sibling tables."""
    root = Path(clusters_dir).expanduser().resolve()
    observed = _read_cluster_year(root / "cluster_by_year.csv", level)
    if not observed:
        raise ValueError(f"No cluster-year rows at level {level}")
    resolved_events = (
        Path(events_dir).expanduser().resolve()
        if events_dir is not None
        else default_events_dir(root)
    )
    papers_by_year = (
        {}
        if resolved_events is None
        else _papers_by_year(resolved_events / "manifest.json")
    )
    years = _combined_years(observed, papers_by_year)
    diversity = _read_diversity(
        Path(topics_dir or "output/cluster_topic_visualizations")
    )
    repeats = _read_repeats(Path(links_dir or "output/new_link_visualizations"))
    series = []
    with Path(trends).expanduser().resolve().open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"level", "group", "keywords", "sub", "activity"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(
                f"{trends} must contain level, group, keywords, sub, and activity"
            )
        for row in reader:
            if int(row["level"]) != level:
                continue
            group = int(row["group"])
            activity = np.asarray(json.loads(row["activity"]), dtype=float)
            if activity.shape != years.shape:
                raise ValueError(
                    f"Cluster {group} activity length does not match the year grid"
                )
            papers, share = _align_counts(observed, group, years)
            keywords = tuple(
                part.strip() for part in row["keywords"].split(";") if part.strip()
            )
            series.append(
                ClusterSeries(
                    group=group,
                    label=keywords[0] if keywords else str(group),
                    keywords=keywords,
                    subclass=row["sub"],
                    share=share,
                    log_share=log_share(papers, share),
                    adjusted=activity,
                    diversity=diversity.get(group),
                    within_by_lag={} if repeats is None else repeats.get(group, {}),
                )
            )
    if not series:
        raise ValueError(f"No trend rows at level {level}")
    return series, years, repeats is not None


def _align_counts(
    observed: Mapping[tuple[int, int], tuple[int, float]],
    group: int,
    years: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    papers = np.zeros(years.size, dtype=float)
    share = np.zeros(years.size, dtype=float)
    for position, year in enumerate(years.tolist()):
        count, weight = observed.get((group, int(year)), (0, 0.0))
        papers[position] = float(count)
        share[position] = float(weight)
    return papers, share


def _within_by_lag(raw: object) -> dict[int, int | None]:
    repeats = json.loads(raw) if isinstance(raw, str) else raw
    found: dict[int, int | None] = {}
    for item in repeats or []:
        found[int(item["lag"])] = int(item["within_count"])
    return found


def _read_diversity(root: Path) -> dict[int, float | None]:
    path = root.expanduser().resolve() / "cluster_topic_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} was not found. Run visualize-cluster-topics first."
        )
    diversity: dict[int, float | None] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "author_rao_stirling_diversity" not in reader.fieldnames:
            raise ValueError(
                f"{path} has no author_rao_stirling_diversity column. "
                "Rerun visualize-cluster-topics."
            )
        for row in reader:
            raw = row["author_rao_stirling_diversity"]
            if raw is None or raw == "":
                value = None
            else:
                value = float(raw)
                if not np.isfinite(value):
                    value = None
            diversity[int(row["cluster_id"])] = value
    return diversity


def _read_repeats(root: Path) -> dict[int, dict[int, int | None]] | None:
    """Return within-cluster repeat counts, or None when they are unavailable."""
    path = root.expanduser().resolve() / "cluster_link_distance_summary.csv"
    if not path.is_file():
        logger.warning(
            "%s was not found. Persistent-link violins will be skipped.", path
        )
        return None
    repeats: dict[int, dict[int, int | None]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "repeat_link_probability_by_lag" not in reader.fieldnames:
            logger.warning(
                "%s has no repeat_link_probability_by_lag column. "
                "Persistent-link violins will be skipped.",
                path,
            )
            return None
        for row in reader:
            repeats[int(row["cluster_id"])] = _within_by_lag(
                row["repeat_link_probability_by_lag"]
            )
    return repeats


def _gallery_figure(
    series: Sequence[ClusterSeries],
    years: np.ndarray,
    key: str,
    ylabel: str,
    symmetric: bool,
    patterns: Mapping[str, str],
):
    from matplotlib.collections import LineCollection

    plt = _pyplot()
    grouped = []
    for title, classes, _pattern in PANELS:
        members = [row for row in series if row.subclass in classes]
        highlights = select_highlights(series, classes, patterns[title])
        grouped.append((title, members, highlights))
    stacked = np.concatenate(
        [row.values(key) for _title, members, _highlights in grouped for row in members]
    )
    ylim = _shared_limits(stacked, symmetric)
    figure, axes = plt.subplots(
        2, 2, figsize=FIGURE_SIZE, sharex=True, sharey=True, facecolor="white"
    )
    for axis, (title, members, highlights) in zip(axes.ravel(), grouped, strict=True):
        _style_axis(axis)
        axis.set_xlim(float(years[0]) - 0.6, float(years[-1]) + 0.6)
        axis.set_ylim(*ylim)
        if members:
            axis.add_collection(
                LineCollection(
                    [np.column_stack([years, row.values(key)]) for row in members],
                    colors=[COLORS[row.subclass] for row in members],
                    linewidths=1.0,
                    alpha=0.22,
                    zorder=2,
                    capstyle="round",
                )
            )
        for row in highlights:
            values = row.values(key)
            axis.plot(years, values, color="white", lw=5.4, zorder=4, solid_capstyle="round")
            axis.plot(years, values, color=INK, lw=2.6, zorder=5, solid_capstyle="round")
        _place_labels(axis, highlights, key)
        axis.set_title(
            f"{title}   (n={len(members):,})",
            color=INK,
            fontsize=24,
            fontweight="bold",
            pad=10,
        )
    _year_ticks(axes.ravel(), years)
    if key == "log_share":
        _log_share_ticks(axes.ravel())
    figure.supxlabel("Year", fontsize=20, fontweight="bold", color=INK)
    figure.supylabel(ylabel, fontsize=20, fontweight="bold", color=INK)
    figure.tight_layout(rect=(0.03, 0.03, 0.995, 0.985))
    _bold_ticks(figure, size=16)
    return figure


def _place_labels(
    axis,
    highlights: Sequence[ClusterSeries],
    key: str,
) -> None:
    if not highlights:
        return
    from matplotlib.transforms import blended_transform_factory

    transform = blended_transform_factory(axis.transAxes, axis.transData)
    multiple = len(highlights) > 1
    for index, row in enumerate(highlights):
        above = multiple and index == len(highlights) - 1
        axis.annotate(
            row.label,
            (0.0, float(row.values(key)[0])),
            xycoords=transform,
            xytext=(4, 12 if above else -12),
            textcoords="offset points",
            ha="left",
            va="bottom" if above else "top",
            color=INK,
            fontsize=18,
            fontweight="bold",
            annotation_clip=False,
            zorder=6,
        )


def _log_share_ticks(axes) -> None:
    """Label the logged frequency axis as powers of ten."""
    low, high = axes[0].get_ylim()
    start = math.ceil(low - 1e-9)
    stop = math.floor(high + 1e-9)
    span = max(stop - start, 0)
    step = 1 if span <= 8 else 2 if span <= 16 else 5
    ticks = list(range(start, stop + 1, step))
    if len(ticks) < 2:
        ticks = [int(round(low)), int(round(high))]
    labels = [rf"$10^{{{tick}}}$" for tick in ticks]
    for axis in axes:
        axis.set_yticks(ticks)
        axis.set_yticklabels(labels)


def _shared_limits(values: np.ndarray, symmetric: bool) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (-1.0, 1.0)
    if symmetric:
        limit = max(abs(float(finite.min())), abs(float(finite.max()))) * 1.08 or 1.0
        return (-limit, limit)
    pad = 0.06 * (float(finite.max()) - float(finite.min()) or 1.0)
    return (float(finite.min()) - pad, float(finite.max()) + pad)


def _diversity_figure(
    series: Sequence[ClusterSeries],
    patterns: Mapping[str, str] | None = None,
):
    samples = []
    for _title, classes, _pattern in PANELS:
        values = np.asarray(
            [
                row.diversity
                for row in series
                if row.subclass in classes and row.diversity is not None
            ],
            dtype=float,
        )
        samples.append(values[np.isfinite(values)])
    figure = _violin(
        samples,
        "Author primary-topic diversity",
        mean_digits=3,
        log=False,
        exemplars=_exemplar_marks(
            series,
            patterns,
            lambda row: row.diversity,
            positive=False,
        ),
    )
    finite = [sample for sample in samples if sample.size]
    if finite:
        stacked = np.concatenate(finite)
        span = float(stacked.max()) - float(stacked.min()) or 1.0
        low = max(0.0, float(stacked.min()) - 0.08 * span)
        high = min(1.0, float(stacked.max()) + 0.18)
        figure.axes[0].set_ylim(low, max(high, low + 0.05))
    return figure


def _persistent_figure(
    series: Sequence[ClusterSeries],
    *,
    lag: int,
    patterns: Mapping[str, str] | None = None,
):
    samples = []
    for _title, classes, _pattern in PANELS:
        counts = []
        for row in series:
            if row.subclass not in classes:
                continue
            count = row.within_by_lag.get(lag)
            if count is None or count <= 0:
                continue
            counts.append(float(count))
        samples.append(np.asarray(counts, dtype=float))
    return _violin(
        samples,
        "Persistent new links",
        mean_digits=1,
        log=True,
        exemplars=_exemplar_marks(
            series,
            patterns,
            lambda row: row.within_by_lag.get(lag),
            positive=True,
        ),
    )


def _exemplar_marks(
    series: Sequence[ClusterSeries],
    patterns: Mapping[str, str] | None,
    reader,
    *,
    positive: bool,
) -> list[list[tuple[str, float]]]:
    """Return each panel's exemplar labels and plotted values."""
    selected = patterns if patterns is not None else parse_exemplars(None)
    marks = []
    for title, classes, _pattern in PANELS:
        panel = []
        for row in select_highlights(series, classes, selected[title]):
            raw = reader(row)
            if raw is None:
                continue
            value = float(raw)
            if not np.isfinite(value) or (positive and value <= 0):
                continue
            panel.append((row.label, value))
        marks.append(panel)
    return marks


def _draw_exemplar_points(axis, marks: Sequence[Sequence[tuple[str, float]]]) -> None:
    positions = (1, 2, 3, 4)
    for position, panel in zip(positions, marks, strict=True):
        for index, (label, value) in enumerate(panel):
            x = position + 0.22 + 0.07 * index
            axis.scatter(
                [x],
                [value],
                s=86,
                color=INK,
                edgecolor="white",
                linewidths=1.4,
                zorder=8,
            )
            axis.annotate(
                label,
                (x, value),
                xytext=(6, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                color=INK,
                fontsize=15,
                fontweight="bold",
                annotation_clip=False,
                zorder=9,
            )


def _violin(
    samples: Sequence[np.ndarray],
    ylabel: str,
    *,
    mean_digits: int,
    log: bool,
    exemplars: Sequence[Sequence[tuple[str, float]]] | None = None,
):
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=FIGURE_SIZE, facecolor="white")
    _style_axis(axis)
    positions = [1, 2, 3, 4]
    observed = [
        (position, sample)
        for position, sample in zip(positions, samples, strict=True)
        if sample.size
    ]
    # A Gaussian density needs at least two distinct counts. A single cluster
    # still gets the mean marker.
    bodies = [
        (position, sample) for position, sample in observed if sample.size >= 2
    ]
    if bodies:
        parts = axis.violinplot(
            [sample for _position, sample in bodies],
            positions=[position for position, _sample in bodies],
            widths=0.62,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        body_colors = [PANEL_COLORS[position - 1] for position, _sample in bodies]
        for body, color in zip(parts["bodies"], body_colors, strict=True):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.4)
            body.set_linewidth(1.05)
    for position, sample in observed:
        mean = float(sample.mean())
        axis.plot(
            [position - 0.14, position + 0.14],
            [mean, mean],
            color=INK,
            lw=2.4,
            zorder=5,
            solid_capstyle="butt",
        )
        axis.scatter(
            [position],
            [mean],
            s=42,
            color="white",
            edgecolor=INK,
            linewidths=1.6,
            zorder=6,
        )
        axis.annotate(
            f"{mean:.{mean_digits}f}",
            (position, mean),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=INK,
            fontsize=18,
            fontweight="bold",
            zorder=7,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.92,
            },
        )
    names = ["Rising", "Declining", "Shocks\n(transient)", "Shocks\n(lasting)"]
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [
            f"{name}\n(n={sample.size:,})"
            for name, sample in zip(names, samples, strict=True)
        ],
        color=INK,
        fontsize=18,
        fontweight="bold",
    )
    axis.tick_params(axis="x", length=0, pad=6)
    axis.set_xlim(0.4, 4.6)
    axis.set_ylabel(ylabel, color=INK, fontsize=20, fontweight="bold")
    if log and observed:
        axis.set_yscale("log")
    elif observed:
        finite = np.concatenate([sample for _position, sample in observed])
        span = float(finite.max()) - float(finite.min()) or 1.0
        axis.set_ylim(float(finite.min()) - 0.08 * span, float(finite.max()) + 0.22 * span)
    if exemplars:
        _draw_exemplar_points(axis, exemplars)
    figure.tight_layout()
    _bold_ticks(figure, size=16, axis="y")
    return figure


def _style_axis(axis) -> None:
    axis.set_facecolor("white")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(SPINE)
    axis.spines["bottom"].set_color(SPINE)


def _year_ticks(axes, years: np.ndarray) -> None:
    ticks = [int(years[0]), int(years[len(years) // 2]), int(years[-1])]
    if years.size >= 8:
        ticks = [int(year) for year in years if int(year) % 5 == 0]
    for axis in axes:
        axis.set_xticks(ticks)


def _bold_ticks(figure, *, size: int, axis: str = "both") -> None:
    figure.canvas.draw()
    for artist in figure.axes:
        labels = []
        if axis in {"both", "x"}:
            labels.extend(artist.get_xticklabels())
        if axis in {"both", "y"}:
            labels.extend(artist.get_yticklabels())
        for label in labels:
            label.set_fontsize(size)
            label.set_fontweight("bold")
            label.set_color(INK)


def _save(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.pdf")
    figure.savefig(temporary, format="pdf")
    figure.clf()
    _pyplot().close(figure)
    temporary.replace(path)
    return path


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update({"axes.linewidth": 1.15, "pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write slide PDFs of cluster shape galleries, author primary-topic "
            "diversity, and persistent new-link counts."
        )
    )
    parser.add_argument(
        "--trends",
        type=Path,
        default=None,
        help="cluster_trends.csv. Defaults to that file inside --clusters-dir.",
    )
    parser.add_argument("--clusters-dir", type=Path, default=Path("output/event_clusters"))
    parser.add_argument("--events-dir", type=Path, default=None)
    parser.add_argument(
        "--cluster-topic-visualizations-dir",
        type=Path,
        default=Path("output/cluster_topic_visualizations"),
    )
    parser.add_argument(
        "--new-link-visualizations-dir",
        type=Path,
        default=Path("output/new_link_visualizations"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/shape_figures"))
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument(
        "--exemplar",
        action="append",
        default=None,
        help="Override one panel pattern, for example declining=nano*.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args(argv)
    patterns = parse_exemplars(args.exemplar)
    trends = args.trends or (args.clusters_dir / "cluster_trends.csv")
    series, years, persistent = load_trends(
        trends,
        args.clusters_dir,
        args.events_dir,
        args.cluster_topic_visualizations_dir,
        args.new_link_visualizations_dir,
        level=args.level,
    )
    write_shape_gallery(
        series, years, args.output_dir, patterns, persistent=persistent
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
