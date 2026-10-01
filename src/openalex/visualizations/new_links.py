"""Scatter plots of event-cluster size and coauthorship-link distance."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DPI = 180


@dataclass(frozen=True)
class ResidualSelection:
    highlighted: np.ndarray
    residuals: np.ndarray
    size_quantile: np.ndarray
    slope: float
    intercept: float


def build_cluster_link_plots(
    new_links_dir: str | Path,
    output_dir: str | Path,
    *,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Write cluster summaries and two paper-count/distance scatter plots."""
    if dpi < 1:
        raise ValueError("--dpi must be >= 1")

    root = Path(new_links_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    manifest = _read_json(root / "manifest.json")
    if int(manifest.get("artifact_version", 0)) < 6:
        raise ValueError(
            "New-link artifacts lack cluster visualization aggregates; rebuild them."
        )
    years = [int(year) for year in manifest.get("years", [])]
    with np.load(root / manifest["cluster_metadata"], allow_pickle=False) as payload:
        cluster_ids = payload["cluster_id"].astype(np.int32, copy=False)
        labels = payload["label"].astype(str)
        paper_counts = payload["paper_count"].astype(np.int64, copy=False)
    cluster_count = int(cluster_ids.size)
    if not np.array_equal(cluster_ids, np.arange(cluster_count, dtype=np.int32)):
        raise ValueError("Cluster metadata ids must be contiguous from zero")

    new_distance_sum = np.zeros(cluster_count, dtype=np.float64)
    new_connected = np.zeros(cluster_count, dtype=np.int64)
    new_disconnected = np.zeros(cluster_count, dtype=np.int64)
    all_distance_sum = np.zeros(cluster_count, dtype=np.float64)
    all_connected = np.zeros(cluster_count, dtype=np.int64)
    all_disconnected = np.zeros(cluster_count, dtype=np.int64)
    all_existing = np.zeros(cluster_count, dtype=np.int64)
    all_total = np.zeros(cluster_count, dtype=np.int64)
    new_distance_histograms: list[defaultdict[int, float]] = [
        defaultdict(float) for _ in range(cluster_count)
    ]
    outside_cluster_distance_histogram: defaultdict[int, float] = defaultdict(float)

    cluster_years_dir = root / manifest["cluster_years_dir"]
    for year in years:
        with np.load(root / "years" / f"{year}.npz", allow_pickle=False) as links:
            link_clusters = links["cluster_id"].astype(np.int32, copy=False)
            distances = links["distance"].astype(np.int32, copy=False)
            sampling_weight = links["sampling_weight"].astype(
                np.float64, copy=False
            )
        attributed = link_clusters >= 0
        connected = attributed & (distances >= 0)
        new_distance_sum += np.bincount(
            link_clusters[connected],
            weights=distances[connected] * sampling_weight[connected],
            minlength=cluster_count,
        )
        for cluster in np.unique(link_clusters[connected]):
            members = connected & (link_clusters == cluster)
            values, inverse = np.unique(distances[members], return_inverse=True)
            totals = np.bincount(
                inverse,
                weights=sampling_weight[members],
            )
            for distance, total in zip(values, totals, strict=True):
                new_distance_histograms[int(cluster)][int(distance)] += float(total)
        outside_connected = (link_clusters < 0) & (distances >= 0)
        values, inverse = np.unique(
            distances[outside_connected],
            return_inverse=True,
        )
        totals = np.bincount(
            inverse,
            weights=sampling_weight[outside_connected],
        )
        for distance, total in zip(values, totals, strict=True):
            outside_cluster_distance_histogram[int(distance)] += float(total)

        with np.load(cluster_years_dir / f"{year}.npz", allow_pickle=False) as stats:
            new_connected += stats["attributed_new_link_connected"]
            new_disconnected += stats["attributed_new_link_disconnected"]
            all_connected += stats["connected_pair_observations"]
            all_disconnected += stats["disconnected_pair_observations"]
            all_existing += stats["existing_pair_observations"]
            all_total += stats["total_pair_observations"]

    with np.load(
        root / manifest["cluster_distance_reservoir"], allow_pickle=False
    ) as reservoir:
        reservoir_distances = reservoir["distances"].astype(np.int32, copy=False)
        reservoir_offsets = reservoir["offsets"].astype(np.int64, copy=False)
        reservoir_population = reservoir["population"].astype(np.int64, copy=False)
    if (
        reservoir_offsets.shape != (cluster_count + 1,)
        or reservoir_population.shape != (cluster_count,)
    ):
        raise ValueError("Cluster distance reservoir does not align with metadata")
    reservoir_sample_count = np.diff(reservoir_offsets)
    all_distance_histograms: list[defaultdict[int, float]] = [
        defaultdict(float) for _ in range(cluster_count)
    ]
    for cluster in range(cluster_count):
        sample = reservoir_distances[
            reservoir_offsets[cluster] : reservoir_offsets[cluster + 1]
        ]
        if sample.size:
            weight = float(reservoir_population[cluster]) / sample.size
            all_distance_sum[cluster] = (
                weight * float(sample.sum(dtype=np.int64))
            )
            values, counts = np.unique(sample, return_counts=True)
            for distance, count in zip(values, counts, strict=True):
                all_distance_histograms[cluster][int(distance)] += (
                    float(count) * weight
                )
        if all_existing[cluster]:
            # An existing coauthorship is already an edge, so its distance is 1.
            all_distance_histograms[cluster][1] += float(all_existing[cluster])

    average_new = _safe_average(new_distance_sum, new_connected)
    average_all = _safe_average(all_distance_sum + all_existing, all_connected)
    new_selection = _residual_highlights(paper_counts, average_new)
    all_selection = _residual_highlights(paper_counts, average_all)
    rows = _summary_rows(
        cluster_ids,
        labels,
        paper_counts,
        average_new,
        new_connected,
        new_disconnected,
        average_all,
        all_connected,
        all_disconnected,
        all_existing,
        all_total,
        reservoir_population,
        reservoir_sample_count,
        new_selection,
        all_selection,
        new_distance_histograms,
        all_distance_histograms,
        outside_cluster_distance_histogram,
    )

    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "cluster_link_distance_summary.csv"
    _write_summary(summary_path, rows)
    _write_scatter(
        output / "cluster_new_link_distance.png",
        paper_counts,
        average_new,
        labels,
        new_selection,
        title="Event-cluster size and average new-link distance",
        y_label="Average new-link distance (connected only)",
        dpi=dpi,
    )
    _write_scatter(
        output / "cluster_all_link_distance.png",
        paper_counts,
        average_all,
        labels,
        all_selection,
        title="Event-cluster size and average distance across all paper links",
        y_label="Average link distance (connected only; existing links = 1)",
        dpi=dpi,
    )
    logger.info("Wrote cluster-link visualizations to %s", output)
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot event-cluster paper counts against coauthorship-link distances."
    )
    parser.add_argument("--new-links-dir", default="output/new_links")
    parser.add_argument("--output-dir", default="output/new_link_visualizations")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args(argv)
    build_cluster_link_plots(
        args.new_links_dir,
        args.output_dir,
        dpi=args.dpi,
    )
    return 0


def _safe_average(total: np.ndarray, count: np.ndarray) -> np.ndarray:
    result = np.full(total.size, np.nan, dtype=np.float64)
    np.divide(total, count, out=result, where=count > 0)
    return result


def _residual_highlights(
    paper_counts: np.ndarray, averages: np.ndarray
) -> ResidualSelection:
    valid = (paper_counts > 0) & np.isfinite(averages)
    valid_indices = np.flatnonzero(valid)
    residuals = np.full(averages.size, np.nan, dtype=np.float64)
    size_quantile = np.full(averages.size, -1, dtype=np.int8)
    if valid_indices.size == 0:
        return ResidualSelection(
            highlighted=np.empty(0, dtype=np.int64),
            residuals=residuals,
            size_quantile=size_quantile,
            slope=0.0,
            intercept=float("nan"),
        )

    x = np.log10(paper_counts[valid_indices].astype(np.float64))
    y = averages[valid_indices]
    if valid_indices.size > 1 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope = 0.0
        intercept = float(np.mean(y))
    residuals[valid_indices] = y - (slope * x + intercept)

    ranked = valid_indices[
        np.argsort(paper_counts[valid_indices], kind="stable")
    ]
    highlighted: list[int] = []
    for quantile, members in enumerate(np.array_split(ranked, 4), start=1):
        if members.size == 0:
            continue
        size_quantile[members] = quantile
        member_residuals = residuals[members]
        negative = int(members[np.argmin(member_residuals)])
        positive = int(members[np.argmax(member_residuals)])
        highlighted.append(negative)
        if positive != negative:
            highlighted.append(positive)
    return ResidualSelection(
        highlighted=np.asarray(highlighted, dtype=np.int64),
        residuals=residuals,
        size_quantile=size_quantile,
        slope=float(slope),
        intercept=float(intercept),
    )


def _summary_rows(
    cluster_ids: np.ndarray,
    labels: np.ndarray,
    paper_counts: np.ndarray,
    average_new: np.ndarray,
    new_connected: np.ndarray,
    new_disconnected: np.ndarray,
    average_all: np.ndarray,
    all_connected: np.ndarray,
    all_disconnected: np.ndarray,
    all_existing: np.ndarray,
    all_total: np.ndarray,
    reservoir_population: np.ndarray,
    reservoir_sample_count: np.ndarray,
    new_selection: ResidualSelection,
    all_selection: ResidualSelection,
    new_distance_histograms: Sequence[Mapping[int, float]],
    all_distance_histograms: Sequence[Mapping[int, float]],
    outside_cluster_distance_histogram: Mapping[int, float],
) -> list[dict[str, object]]:
    new_highlighted = set(int(index) for index in new_selection.highlighted)
    all_highlighted = set(int(index) for index in all_selection.highlighted)
    rows = []
    for index, cluster_id in enumerate(cluster_ids):
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "label": str(labels[index]),
                "paper_count": int(paper_counts[index]),
                "average_new_link_distance": float(average_new[index]),
                "new_link_connected_count": int(new_connected[index]),
                "new_link_disconnected_count": int(new_disconnected[index]),
                "new_link_residual": float(new_selection.residuals[index]),
                "new_link_size_quantile": int(new_selection.size_quantile[index]),
                "new_link_highlighted": index in new_highlighted,
                "new_link_distance_distribution": _distribution_json(
                    new_distance_histograms[index]
                ),
                "outside_cluster_distance_distribution": _distribution_json(
                    outside_cluster_distance_histogram
                ),
                "average_all_link_distance": float(average_all[index]),
                "all_link_connected_count": int(all_connected[index]),
                "all_link_disconnected_count": int(all_disconnected[index]),
                "all_link_existing_count": int(all_existing[index]),
                "all_link_observation_count": int(all_total[index]),
                "all_link_new_connected_count": int(
                    reservoir_population[index]
                ),
                "all_link_distance_sample_count": int(
                    reservoir_sample_count[index]
                ),
                "all_link_residual": float(all_selection.residuals[index]),
                "all_link_size_quantile": int(all_selection.size_quantile[index]),
                "all_link_highlighted": index in all_highlighted,
                "all_link_distance_distribution": _distribution_json(
                    all_distance_histograms[index]
                ),
            }
        )
    return rows


def _distribution_json(histogram: Mapping[int, float]) -> str:
    return json.dumps(
        [
            [int(distance), float(count)]
            for distance, count in sorted(histogram.items())
            if count > 0
        ],
        separators=(",", ":"),
    )


def _write_summary(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = [
        "cluster_id",
        "label",
        "paper_count",
        "average_new_link_distance",
        "new_link_connected_count",
        "new_link_disconnected_count",
        "new_link_residual",
        "new_link_size_quantile",
        "new_link_highlighted",
        "new_link_distance_distribution",
        "outside_cluster_distance_distribution",
        "average_all_link_distance",
        "all_link_connected_count",
        "all_link_disconnected_count",
        "all_link_existing_count",
        "all_link_observation_count",
        "all_link_new_connected_count",
        "all_link_distance_sample_count",
        "all_link_residual",
        "all_link_size_quantile",
        "all_link_highlighted",
        "all_link_distance_distribution",
    ]
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_scatter(
    path: Path,
    paper_counts: np.ndarray,
    averages: np.ndarray,
    labels: np.ndarray,
    selection: ResidualSelection,
    *,
    title: str,
    y_label: str,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    valid = (paper_counts > 0) & np.isfinite(averages)
    figure, axis = plt.subplots(figsize=(9.5, 6.2), facecolor="white")
    axis.set_facecolor("white")
    if np.any(valid):
        axis.scatter(
            paper_counts[valid],
            averages[valid],
            color="#a7a7a7",
            alpha=0.7,
            edgecolors="none",
            s=30,
            zorder=2,
            label="Other clusters",
        )
        line_x = np.geomspace(
            float(np.min(paper_counts[valid])),
            float(np.max(paper_counts[valid])),
            100,
        )
        line_y = selection.slope * np.log10(line_x) + selection.intercept
        axis.plot(
            line_x,
            line_y,
            color="#555555",
            linewidth=1.1,
            linestyle="--",
            alpha=0.8,
            zorder=1,
            label="Linear fit",
        )
        highlighted = selection.highlighted[valid[selection.highlighted]]
        axis.scatter(
            paper_counts[highlighted],
            averages[highlighted],
            color="#b4232f",
            edgecolors="white",
            linewidths=0.7,
            s=54,
            zorder=3,
            label="Residual extremes",
        )
        for index in highlighted:
            direction = 7 if selection.residuals[index] >= 0 else -10
            axis.annotate(
                str(labels[index]),
                (paper_counts[index], averages[index]),
                xytext=(5, direction),
                textcoords="offset points",
                color="#7f1d1d",
                fontsize=8,
                fontweight="medium",
                ha="left",
                va="bottom" if direction > 0 else "top",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                },
                zorder=4,
            )
        axis.legend(frameon=False, fontsize=8, loc="best")
    else:
        axis.text(
            0.5,
            0.5,
            "No clusters have connected distance observations",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_xscale("log")
    axis.set_xlabel("Cluster size (papers, log scale)")
    axis.set_ylabel(y_label)
    axis.set_title(title, loc="left", fontsize=13, fontweight="medium", pad=12)
    axis.grid(True, which="major", color="#e5e5e5", linewidth=0.8)
    axis.grid(True, which="minor", axis="x", color="#f1f1f1", linewidth=0.5)
    axis.tick_params(colors="#4a4a4a", labelsize=9)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#bdbdbd")
    axis.spines["bottom"].set_color("#bdbdbd")
    figure.tight_layout()
    temporary = path.with_name(f".{path.name}.tmp.png")
    figure.savefig(temporary, dpi=dpi)
    plt.close(figure)
    os.replace(temporary, path)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    raise SystemExit(main())
