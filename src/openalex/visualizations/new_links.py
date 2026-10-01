"""Scatter plots of event-cluster size and coauthorship-link distance."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DPI = 180
DEFAULT_BOOTSTRAP_REPLICATES = 250
DEFAULT_PERMUTATION_REPLICATES = 499
DEFAULT_INFERENCE_SEED = 0
DEFAULT_INFERENCE_WORKERS = 8


@dataclass(frozen=True)
class ResidualSelection:
    highlighted: np.ndarray
    residuals: np.ndarray
    size_quantile: np.ndarray
    slope: float
    intercept: float


@dataclass
class DistributionComparison:
    baseline_histogram: dict[int, float]
    disconnection_probability: float
    baseline_disconnection_probability: float
    disconnection_risk_difference: float
    disconnection_ci: tuple[float, float]
    disconnection_p_value: float
    disconnection_q_value: float = float("nan")
    mean_distance_shift: float = float("nan")
    mean_distance_shift_ci: tuple[float, float] = (float("nan"), float("nan"))
    wasserstein_distance: float = float("nan")
    wasserstein_ci: tuple[float, float] = (float("nan"), float("nan"))
    wasserstein_p_value: float = float("nan")
    wasserstein_q_value: float = float("nan")
    repeat_probability: float = float("nan")
    baseline_repeat_probability: float = float("nan")
    repeat_risk_difference: float = float("nan")
    repeat_risk_difference_ci: tuple[float, float] = (
        float("nan"),
        float("nan"),
    )


def build_cluster_link_plots(
    new_links_dir: str | Path,
    output_dir: str | Path,
    *,
    dpi: int = DEFAULT_DPI,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    permutation_replicates: int = DEFAULT_PERMUTATION_REPLICATES,
    inference_seed: int = DEFAULT_INFERENCE_SEED,
    inference_workers: int = DEFAULT_INFERENCE_WORKERS,
) -> Path:
    """Write cluster summaries and two paper-count/distance scatter plots."""
    if dpi < 1:
        raise ValueError("--dpi must be >= 1")
    if bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be >= 1")
    if permutation_replicates < 1:
        raise ValueError("--permutation-replicates must be >= 1")
    if inference_seed < 0:
        raise ValueError("--inference-seed must be >= 0")
    if inference_workers < 1:
        raise ValueError("--inference-workers must be >= 1")

    root = Path(new_links_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    manifest = _read_json(root / "manifest.json")
    if int(manifest.get("artifact_version", 0)) < 7:
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

    year_count = len(years)
    new_connected_by_year = np.zeros((year_count, cluster_count), dtype=np.int64)
    new_disconnected_by_year = np.zeros((year_count, cluster_count), dtype=np.int64)
    all_connected_by_year = np.zeros((year_count, cluster_count), dtype=np.int64)
    all_disconnected_by_year = np.zeros((year_count, cluster_count), dtype=np.int64)
    all_existing_by_year = np.zeros((year_count, cluster_count), dtype=np.int64)
    all_new_connected_by_year = np.zeros(
        (year_count, cluster_count), dtype=np.int64
    )
    all_total_by_year = np.zeros((year_count, cluster_count), dtype=np.int64)
    new_distance_sum = np.zeros(cluster_count, dtype=np.float64)
    new_distance_histograms: list[defaultdict[int, float]] = [
        defaultdict(float) for _ in range(cluster_count)
    ]
    new_cluster_year_histograms: list[list[defaultdict[int, float]]] = [
        [defaultdict(float) for _ in years] for _ in range(cluster_count)
    ]
    new_reference_year_histograms: list[defaultdict[int, float]] = [
        defaultdict(float) for _ in years
    ]
    all_reference_records: list[tuple[np.ndarray, np.ndarray]] = []
    new_distance_by_year: list[list[list[int | float]]] = [
        [] for _ in range(cluster_count)
    ]
    all_distance_by_year: list[list[list[int | float]]] = [
        [] for _ in range(cluster_count)
    ]

    cluster_years_dir = root / manifest["cluster_years_dir"]
    for year_index, year in enumerate(years):
        with np.load(root / "years" / f"{year}.npz", allow_pickle=False) as links:
            link_clusters = links["cluster_id"].astype(np.int32, copy=False)
            distances = links["distance"].astype(np.int32, copy=False)
            sampling_weight = links["sampling_weight"].astype(
                np.float64, copy=False
            )
        attributed = link_clusters >= 0
        connected = attributed & (distances >= 0)
        year_new_sum = np.bincount(
            link_clusters[connected],
            weights=distances[connected] * sampling_weight[connected],
            minlength=cluster_count,
        )
        new_distance_sum += year_new_sum
        for cluster in np.unique(link_clusters[connected]):
            members = connected & (link_clusters == cluster)
            values, inverse = np.unique(distances[members], return_inverse=True)
            totals = np.bincount(
                inverse,
                weights=sampling_weight[members],
            )
            for distance, total in zip(values, totals, strict=True):
                new_distance_histograms[int(cluster)][int(distance)] += float(total)
        for cluster in np.unique(link_clusters[attributed]):
            members = attributed & (link_clusters == cluster)
            _add_values(
                new_cluster_year_histograms[int(cluster)][year_index],
                distances[members],
            )
        _add_values(
            new_reference_year_histograms[year_index],
            distances[link_clusters < 0],
        )

        with np.load(cluster_years_dir / f"{year}.npz", allow_pickle=False) as stats:
            year_new_connected = stats["attributed_new_link_connected"].astype(
                np.int64, copy=False
            )
            new_connected_by_year[year_index] = year_new_connected
            new_disconnected_by_year[year_index] = stats[
                "attributed_new_link_disconnected"
            ]
            all_connected_by_year[year_index] = stats[
                "connected_pair_observations"
            ]
            all_disconnected_by_year[year_index] = stats[
                "disconnected_pair_observations"
            ]
            all_existing_by_year[year_index] = stats[
                "existing_pair_observations"
            ]
            all_new_connected_by_year[year_index] = stats[
                "new_connected_pair_observations"
            ]
            all_total_by_year[year_index] = stats["total_pair_observations"]
            all_reference_records.append(
                (
                    stats["reference_cluster_id"].astype(np.int32, copy=True),
                    stats["reference_distance"].astype(np.int32, copy=True),
                )
            )
            _append_yearly_means(
                year,
                year_new_sum,
                year_new_connected,
                stats,
                new_distance_by_year,
                all_distance_by_year,
            )

    with np.load(
        root / manifest["cluster_distance_reservoir"], allow_pickle=False
    ) as reservoir:
        reservoir_distances = reservoir["distances"].astype(np.int32, copy=False)
        reservoir_years = reservoir["years"].astype(np.int32, copy=False)
        reservoir_offsets = reservoir["offsets"].astype(np.int64, copy=False)
        reservoir_population = reservoir["population"].astype(np.int64, copy=False)
    if (
        reservoir_offsets.shape != (cluster_count + 1,)
        or reservoir_population.shape != (cluster_count,)
    ):
        raise ValueError("Cluster distance reservoir does not align with metadata")
    if reservoir_years.shape != reservoir_distances.shape:
        raise ValueError("Cluster reservoir years do not align with distances")
    reservoir_sample_count = np.diff(reservoir_offsets)
    all_distance_histograms: list[defaultdict[int, float]] = [
        defaultdict(float) for _ in range(cluster_count)
    ]
    all_cluster_year_histograms: list[list[defaultdict[int, float]]] = [
        [defaultdict(float) for _ in years] for _ in range(cluster_count)
    ]
    year_lookup = {year: index for index, year in enumerate(years)}
    all_distance_sum = np.zeros(cluster_count, dtype=np.float64)
    for cluster in range(cluster_count):
        sample = reservoir_distances[
            reservoir_offsets[cluster] : reservoir_offsets[cluster + 1]
        ]
        sample_years = reservoir_years[
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
            for distance, sample_year in zip(sample, sample_years, strict=True):
                if int(sample_year) in year_lookup:
                    all_cluster_year_histograms[cluster][
                        year_lookup[int(sample_year)]
                    ][int(distance)] += 1.0

    new_connected = new_connected_by_year.sum(axis=0)
    new_disconnected = new_disconnected_by_year.sum(axis=0)
    all_connected = all_connected_by_year.sum(axis=0)
    all_disconnected = all_disconnected_by_year.sum(axis=0)
    all_existing = all_existing_by_year.sum(axis=0)
    all_new_connected = all_new_connected_by_year.sum(axis=0)
    all_total = all_total_by_year.sum(axis=0)
    average_new = _safe_average(new_distance_sum, new_connected)
    average_all = _safe_average(all_distance_sum, all_connected)
    def compare_cluster(
        cluster: int,
    ) -> tuple[DistributionComparison, DistributionComparison]:
        new_comparison = _compare_distributions(
            new_cluster_year_histograms[cluster],
            new_reference_year_histograms,
            new_connected_by_year[:, cluster],
            new_disconnected_by_year[:, cluster],
            bootstrap_replicates,
            permutation_replicates,
            [inference_seed, 0, cluster],
        )
        all_reference_histograms = []
        for reference_clusters, reference_distances in all_reference_records:
            histogram: defaultdict[int, float] = defaultdict(float)
            _add_values(
                histogram,
                reference_distances[reference_clusters != cluster],
            )
            all_reference_histograms.append(histogram)
        all_comparison = _compare_distributions(
            all_cluster_year_histograms[cluster],
            all_reference_histograms,
            all_connected_by_year[:, cluster],
            all_disconnected_by_year[:, cluster],
            bootstrap_replicates,
            permutation_replicates,
            [inference_seed, 1, cluster],
            existing_by_year=all_existing_by_year[:, cluster],
        )
        return new_comparison, all_comparison

    worker_count = min(inference_workers, max(cluster_count, 1))
    if worker_count == 1:
        comparison_pairs = [compare_cluster(cluster) for cluster in range(cluster_count)]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            comparison_pairs = list(executor.map(compare_cluster, range(cluster_count)))
    new_comparisons = [pair[0] for pair in comparison_pairs]
    all_comparisons = [pair[1] for pair in comparison_pairs]
    _assign_q_values(new_comparisons)
    _assign_q_values(all_comparisons)
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
        reservoir_sample_count,
        all_new_connected,
        new_selection,
        all_selection,
        new_distance_histograms,
        all_distance_histograms,
        new_comparisons,
        all_comparisons,
        new_distance_by_year,
        all_distance_by_year,
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
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument(
        "--permutation-replicates",
        type=int,
        default=DEFAULT_PERMUTATION_REPLICATES,
    )
    parser.add_argument(
        "--inference-seed",
        type=int,
        default=DEFAULT_INFERENCE_SEED,
    )
    parser.add_argument(
        "--inference-workers",
        type=int,
        default=DEFAULT_INFERENCE_WORKERS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args(argv)
    build_cluster_link_plots(
        args.new_links_dir,
        args.output_dir,
        dpi=args.dpi,
        bootstrap_replicates=args.bootstrap_replicates,
        permutation_replicates=args.permutation_replicates,
        inference_seed=args.inference_seed,
        inference_workers=args.inference_workers,
    )
    return 0


def _append_yearly_means(
    year: int,
    year_new_sum: np.ndarray,
    year_new_connected: np.ndarray,
    stats: np.lib.npyio.NpzFile,
    new_distance_by_year: list[list[list[int | float]]],
    all_distance_by_year: list[list[list[int | float]]],
) -> None:
    """Record connected-only means for one year.

    First-link means use the same weighted sum and exact connected count as
    the scatter. All-link means use the year's accepted connected observations,
    which are sampled uniformly before entering the global reservoir.
    """
    for cluster, count in enumerate(year_new_connected):
        if int(count) > 0:
            new_distance_by_year[cluster].append(
                [int(year), float(year_new_sum[cluster]) / int(count)]
            )
    if "connected_distance_sum" not in stats:
        return
    distance_sums = stats["connected_distance_sum"].astype(np.float64, copy=False)
    acceptances = stats["reservoir_acceptances"].astype(np.int64, copy=False)
    populations = stats["connected_pair_observations"].astype(np.int64, copy=False)
    if distance_sums.shape != year_new_connected.shape:
        raise ValueError(f"Year {year} distance sums do not align with clusters")
    for cluster in range(year_new_connected.size):
        population = int(populations[cluster])
        sample_count = int(acceptances[cluster])
        if population <= 0:
            continue
        if sample_count <= 0:
            continue
        mean = float(distance_sums[cluster]) / sample_count
        all_distance_by_year[cluster].append([int(year), mean])


def _add_values(
    histogram: defaultdict[int, float],
    values: np.ndarray,
    weights: np.ndarray | None = None,
) -> None:
    if values.size == 0:
        return
    unique, inverse = np.unique(values, return_inverse=True)
    counts = np.bincount(inverse, weights=weights)
    for value, count in zip(unique, counts, strict=True):
        histogram[int(value)] += float(count)


def _poststratified_histogram(
    histograms: Sequence[Mapping[int, float]],
    year_weights: np.ndarray,
    *,
    connected_only: bool,
) -> dict[int, float]:
    result: defaultdict[int, float] = defaultdict(float)
    for histogram, year_weight in zip(histograms, year_weights, strict=True):
        if year_weight <= 0:
            continue
        eligible = {
            int(value): float(count)
            for value, count in histogram.items()
            if count > 0 and (value >= 0 if connected_only else True)
        }
        total = float(sum(eligible.values()))
        if total <= 0:
            return {}
        for value, count in eligible.items():
            result[value] += float(year_weight) * count / total
    return dict(result)


def _poststratified_rate(
    histograms: Sequence[Mapping[int, float]],
    year_weights: np.ndarray,
    predicate,
    eligible=lambda value: True,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for histogram, year_weight in zip(histograms, year_weights, strict=True):
        if year_weight <= 0:
            continue
        total = float(
            sum(
                count
                for value, count in histogram.items()
                if count > 0 and eligible(value)
            )
        )
        if total <= 0:
            return float("nan")
        matching = float(
            sum(
                count
                for value, count in histogram.items()
                if count > 0 and eligible(value) and predicate(value)
            )
        )
        numerator += float(year_weight) * matching / total
        denominator += float(year_weight)
    return numerator / denominator if denominator else float("nan")


def _histogram_mean(histogram: Mapping[int, float]) -> float:
    total = float(sum(histogram.values()))
    if total <= 0:
        return float("nan")
    return sum(value * count for value, count in histogram.items()) / total


def _wasserstein_distance(
    first: Mapping[int, float], second: Mapping[int, float]
) -> float:
    first_total = float(sum(first.values()))
    second_total = float(sum(second.values()))
    if first_total <= 0 or second_total <= 0:
        return float("nan")
    support = sorted(set(first) | set(second))
    if len(support) < 2:
        return 0.0
    first_cdf = 0.0
    second_cdf = 0.0
    distance = 0.0
    for index, value in enumerate(support[:-1]):
        first_cdf += first.get(value, 0.0) / first_total
        second_cdf += second.get(value, 0.0) / second_total
        distance += abs(first_cdf - second_cdf) * (support[index + 1] - value)
    return float(distance)


def _resample_histogram(
    rng: np.random.Generator, histogram: Mapping[int, float]
) -> dict[int, float]:
    values = np.asarray(sorted(value for value, count in histogram.items() if count > 0))
    if values.size == 0:
        return {}
    counts = np.asarray([histogram[int(value)] for value in values], dtype=np.float64)
    sample_size = max(1, int(round(float(counts.sum()))))
    sampled = rng.multinomial(sample_size, counts / counts.sum())
    return {
        int(value): float(count)
        for value, count in zip(values, sampled, strict=True)
        if count
    }


def _integer_histogram(
    histogram: Mapping[int, float],
) -> dict[int, int]:
    values = np.asarray(sorted(value for value, count in histogram.items() if count > 0))
    if values.size == 0:
        return {}
    counts = np.asarray([histogram[int(value)] for value in values], dtype=np.float64)
    sample_size = max(1, int(round(float(counts.sum()))))
    raw = counts / counts.sum() * sample_size
    integers = np.floor(raw).astype(np.int64)
    remainder = sample_size - int(integers.sum())
    if remainder:
        order = np.argsort(-(raw - integers), kind="stable")
        integers[order[:remainder]] += 1
    return {
        int(value): int(count)
        for value, count in zip(values, integers, strict=True)
        if count
    }


def _filtered_histogram(
    histogram: Mapping[int, float], predicate
) -> dict[int, float]:
    return {
        int(value): float(count)
        for value, count in histogram.items()
        if count > 0 and predicate(value)
    }


def _permuted_histograms(
    rng: np.random.Generator,
    first: Mapping[int, float],
    second: Mapping[int, float],
) -> tuple[dict[int, float], dict[int, float]]:
    first_counts = _integer_histogram(first)
    second_counts = _integer_histogram(second)
    support = sorted(set(first_counts) | set(second_counts))
    if not support:
        return {}, {}
    pooled = np.asarray(
        [first_counts.get(value, 0) + second_counts.get(value, 0) for value in support],
        dtype=np.int64,
    )
    draws = sum(first_counts.values())
    selected = np.zeros(pooled.size, dtype=np.int64)
    remaining_draws = draws
    remaining_total = int(pooled.sum())
    for index in range(pooled.size - 1):
        if remaining_draws <= 0:
            break
        selected[index] = rng.hypergeometric(
            int(pooled[index]),
            remaining_total - int(pooled[index]),
            remaining_draws,
        )
        remaining_draws -= int(selected[index])
        remaining_total -= int(pooled[index])
    if pooled.size:
        selected[-1] = remaining_draws
    other = pooled - selected
    return (
        {
            value: float(count)
            for value, count in zip(support, selected, strict=True)
            if count
        },
        {
            value: float(count)
            for value, count in zip(support, other, strict=True)
            if count
        },
    )


def _confidence_interval(values: Sequence[float]) -> tuple[float, float]:
    finite = np.asarray([value for value in values if np.isfinite(value)])
    if finite.size == 0:
        return float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def _compare_distributions(
    cluster_histograms: Sequence[Mapping[int, float]],
    reference_histograms: Sequence[Mapping[int, float]],
    connected_by_year: np.ndarray,
    disconnected_by_year: np.ndarray,
    bootstrap_replicates: int,
    permutation_replicates: int,
    seed_components: Sequence[int],
    *,
    existing_by_year: np.ndarray | None = None,
) -> DistributionComparison:
    full_by_year = connected_by_year + disconnected_by_year
    cluster_finite = _poststratified_histogram(
        cluster_histograms, connected_by_year, connected_only=True
    )
    baseline_finite = _poststratified_histogram(
        reference_histograms, connected_by_year, connected_only=True
    )
    full_total = int(full_by_year.sum())
    observed_disconnect = (
        float(disconnected_by_year.sum()) / full_total
        if full_total
        else float("nan")
    )
    baseline_disconnect = _poststratified_rate(
        reference_histograms, full_by_year, lambda value: value < 0
    )
    disconnect_difference = observed_disconnect - baseline_disconnect
    mean_shift = _histogram_mean(cluster_finite) - _histogram_mean(baseline_finite)
    wasserstein = _wasserstein_distance(cluster_finite, baseline_finite)
    observed_repeat = (
        float(existing_by_year.sum()) / float(connected_by_year.sum())
        if existing_by_year is not None and connected_by_year.sum()
        else float("nan")
    )
    baseline_repeat = (
        _poststratified_rate(
            reference_histograms,
            connected_by_year,
            lambda value: value == 1,
            eligible=lambda value: value >= 0,
        )
        if existing_by_year is not None
        else float("nan")
    )
    repeat_difference = observed_repeat - baseline_repeat

    if not any(
        np.isfinite(value)
        for value in (disconnect_difference, wasserstein, repeat_difference)
    ):
        return DistributionComparison(
            baseline_histogram=baseline_finite,
            disconnection_probability=observed_disconnect,
            baseline_disconnection_probability=baseline_disconnect,
            disconnection_risk_difference=disconnect_difference,
            disconnection_ci=(float("nan"), float("nan")),
            disconnection_p_value=float("nan"),
            mean_distance_shift=mean_shift,
            wasserstein_distance=wasserstein,
            wasserstein_p_value=float("nan"),
            repeat_probability=observed_repeat,
            baseline_repeat_probability=baseline_repeat,
            repeat_risk_difference=repeat_difference,
        )

    rng = np.random.default_rng(np.random.SeedSequence(list(seed_components)))
    bootstrap_disconnect: list[float] = []
    bootstrap_mean: list[float] = []
    bootstrap_wasserstein: list[float] = []
    bootstrap_repeat: list[float] = []
    for _ in range(bootstrap_replicates):
        sampled_cluster = [
            _resample_histogram(rng, histogram) for histogram in cluster_histograms
        ]
        sampled_reference = [
            _resample_histogram(rng, histogram) for histogram in reference_histograms
        ]
        reference_disconnect = _poststratified_rate(
            sampled_reference, full_by_year, lambda value: value < 0
        )
        sampled_cluster_finite = _poststratified_histogram(
            sampled_cluster, connected_by_year, connected_only=True
        )
        sampled_reference_finite = _poststratified_histogram(
            sampled_reference, connected_by_year, connected_only=True
        )
        bootstrap_disconnect.append(observed_disconnect - reference_disconnect)
        bootstrap_mean.append(
            _histogram_mean(sampled_cluster_finite)
            - _histogram_mean(sampled_reference_finite)
        )
        bootstrap_wasserstein.append(
            _wasserstein_distance(
                sampled_cluster_finite, sampled_reference_finite
            )
        )
        if existing_by_year is not None:
            bootstrap_repeat.append(
                observed_repeat
                - _poststratified_rate(
                    sampled_reference,
                    connected_by_year,
                    lambda value: value == 1,
                    eligible=lambda value: value >= 0,
                )
            )

    permuted_disconnect = 0
    permuted_wasserstein = 0
    valid_disconnect = 0
    valid_wasserstein = 0
    for _ in range(permutation_replicates):
        permuted_cluster: list[dict[int, float]] = []
        permuted_reference: list[dict[int, float]] = []
        permuted_cluster_status: list[dict[int, float]] = []
        permuted_reference_status: list[dict[int, float]] = []
        for year_index, (cluster_histogram, reference_histogram) in enumerate(
            zip(cluster_histograms, reference_histograms, strict=True)
        ):
            first, second = _permuted_histograms(
                rng,
                _filtered_histogram(cluster_histogram, lambda value: value >= 0),
                _filtered_histogram(reference_histogram, lambda value: value >= 0),
            )
            permuted_cluster.append(first)
            permuted_reference.append(second)
            cluster_status = {
                -1: float(disconnected_by_year[year_index]),
                0: float(connected_by_year[year_index]),
            }
            reference_status = {
                -1: float(reference_histogram.get(-1, 0.0)),
                0: float(
                    sum(
                        count
                        for value, count in reference_histogram.items()
                        if value >= 0
                    )
                ),
            }
            first_status, second_status = _permuted_histograms(
                rng, cluster_status, reference_status
            )
            permuted_cluster_status.append(first_status)
            permuted_reference_status.append(second_status)
        first_disconnect = _poststratified_rate(
            permuted_cluster_status, full_by_year, lambda value: value < 0
        )
        second_disconnect = _poststratified_rate(
            permuted_reference_status, full_by_year, lambda value: value < 0
        )
        permuted_difference = first_disconnect - second_disconnect
        if np.isfinite(permuted_difference) and np.isfinite(disconnect_difference):
            valid_disconnect += 1
            permuted_disconnect += (
                abs(permuted_difference) >= abs(disconnect_difference)
            )
        first_finite = _poststratified_histogram(
            permuted_cluster, connected_by_year, connected_only=True
        )
        second_finite = _poststratified_histogram(
            permuted_reference, connected_by_year, connected_only=True
        )
        permuted_distance = _wasserstein_distance(first_finite, second_finite)
        if np.isfinite(permuted_distance) and np.isfinite(wasserstein):
            valid_wasserstein += 1
            permuted_wasserstein += permuted_distance >= wasserstein

    return DistributionComparison(
        baseline_histogram=baseline_finite,
        disconnection_probability=observed_disconnect,
        baseline_disconnection_probability=baseline_disconnect,
        disconnection_risk_difference=disconnect_difference,
        disconnection_ci=_confidence_interval(bootstrap_disconnect),
        disconnection_p_value=(permuted_disconnect + 1) / (valid_disconnect + 1)
        if valid_disconnect
        else float("nan"),
        mean_distance_shift=mean_shift,
        mean_distance_shift_ci=_confidence_interval(bootstrap_mean),
        wasserstein_distance=wasserstein,
        wasserstein_ci=_confidence_interval(bootstrap_wasserstein),
        wasserstein_p_value=(permuted_wasserstein + 1) / (valid_wasserstein + 1)
        if valid_wasserstein
        else float("nan"),
        repeat_probability=observed_repeat,
        baseline_repeat_probability=baseline_repeat,
        repeat_risk_difference=repeat_difference,
        repeat_risk_difference_ci=_confidence_interval(bootstrap_repeat),
    )


def _benjamini_hochberg(values: Sequence[float]) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    finite_indices = np.asarray(
        [index for index, value in enumerate(values) if np.isfinite(value)],
        dtype=np.int64,
    )
    if finite_indices.size == 0:
        return result
    order = finite_indices[
        np.argsort(np.asarray(values)[finite_indices], kind="stable")
    ]
    adjusted = np.empty(order.size, dtype=np.float64)
    running = 1.0
    for reverse_rank in range(order.size - 1, -1, -1):
        rank = reverse_rank + 1
        candidate = float(values[order[reverse_rank]]) * order.size / rank
        running = min(running, candidate)
        adjusted[reverse_rank] = running
    result[order] = np.minimum(adjusted, 1.0)
    return result


def _assign_q_values(comparisons: Sequence[DistributionComparison]) -> None:
    disconnect_q = _benjamini_hochberg(
        [comparison.disconnection_p_value for comparison in comparisons]
    )
    wasserstein_q = _benjamini_hochberg(
        [comparison.wasserstein_p_value for comparison in comparisons]
    )
    for index, comparison in enumerate(comparisons):
        comparison.disconnection_q_value = float(disconnect_q[index])
        comparison.wasserstein_q_value = float(wasserstein_q[index])


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
    reservoir_sample_count: np.ndarray,
    all_new_connected: np.ndarray,
    new_selection: ResidualSelection,
    all_selection: ResidualSelection,
    new_distance_histograms: Sequence[Mapping[int, float]],
    all_distance_histograms: Sequence[Mapping[int, float]],
    new_comparisons: Sequence[DistributionComparison],
    all_comparisons: Sequence[DistributionComparison],
    new_distance_by_year: Sequence[Sequence[Sequence[int | float]]],
    all_distance_by_year: Sequence[Sequence[Sequence[int | float]]],
) -> list[dict[str, object]]:
    new_highlighted = set(int(index) for index in new_selection.highlighted)
    all_highlighted = set(int(index) for index in all_selection.highlighted)
    rows = []
    for index, cluster_id in enumerate(cluster_ids):
        new_comparison = new_comparisons[index]
        all_comparison = all_comparisons[index]
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
                "new_link_distance_by_year": _series_json(
                    new_distance_by_year[index]
                ),
                "new_link_baseline_distance_distribution": _distribution_json(
                    new_comparison.baseline_histogram
                ),
                **_comparison_fields("new_link", new_comparison),
                "average_all_link_distance": float(average_all[index]),
                "all_link_connected_count": int(all_connected[index]),
                "all_link_disconnected_count": int(all_disconnected[index]),
                "all_link_existing_count": int(all_existing[index]),
                "all_link_observation_count": int(all_total[index]),
                "all_link_new_connected_count": int(all_new_connected[index]),
                "all_link_distance_sample_count": int(
                    reservoir_sample_count[index]
                ),
                "all_link_residual": float(all_selection.residuals[index]),
                "all_link_size_quantile": int(all_selection.size_quantile[index]),
                "all_link_highlighted": index in all_highlighted,
                "all_link_distance_distribution": _distribution_json(
                    all_distance_histograms[index]
                ),
                "all_link_distance_by_year": _series_json(
                    all_distance_by_year[index]
                ),
                "all_link_baseline_distance_distribution": _distribution_json(
                    all_comparison.baseline_histogram
                ),
                **_comparison_fields("all_link", all_comparison),
            }
        )
    return rows


def _comparison_fields(
    prefix: str, comparison: DistributionComparison
) -> dict[str, float]:
    return {
        f"{prefix}_disconnection_probability": comparison.disconnection_probability,
        f"{prefix}_baseline_disconnection_probability": (
            comparison.baseline_disconnection_probability
        ),
        f"{prefix}_disconnection_risk_difference": (
            comparison.disconnection_risk_difference
        ),
        f"{prefix}_disconnection_ci_low": comparison.disconnection_ci[0],
        f"{prefix}_disconnection_ci_high": comparison.disconnection_ci[1],
        f"{prefix}_disconnection_p_value": comparison.disconnection_p_value,
        f"{prefix}_disconnection_q_value": comparison.disconnection_q_value,
        f"{prefix}_mean_distance_shift": comparison.mean_distance_shift,
        f"{prefix}_mean_distance_shift_ci_low": comparison.mean_distance_shift_ci[
            0
        ],
        f"{prefix}_mean_distance_shift_ci_high": comparison.mean_distance_shift_ci[
            1
        ],
        f"{prefix}_wasserstein_distance": comparison.wasserstein_distance,
        f"{prefix}_wasserstein_ci_low": comparison.wasserstein_ci[0],
        f"{prefix}_wasserstein_ci_high": comparison.wasserstein_ci[1],
        f"{prefix}_wasserstein_p_value": comparison.wasserstein_p_value,
        f"{prefix}_wasserstein_q_value": comparison.wasserstein_q_value,
        f"{prefix}_repeat_probability": comparison.repeat_probability,
        f"{prefix}_baseline_repeat_probability": (
            comparison.baseline_repeat_probability
        ),
        f"{prefix}_repeat_risk_difference": comparison.repeat_risk_difference,
        f"{prefix}_repeat_risk_difference_ci_low": (
            comparison.repeat_risk_difference_ci[0]
        ),
        f"{prefix}_repeat_risk_difference_ci_high": (
            comparison.repeat_risk_difference_ci[1]
        ),
    }


def _series_json(points: Sequence[Sequence[int | float]]) -> str:
    return json.dumps(
        [[int(year), float(mean)] for year, mean in points],
        separators=(",", ":"),
    )


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
        "new_link_distance_by_year",
        "new_link_baseline_distance_distribution",
        *_comparison_fieldnames("new_link"),
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
        "all_link_distance_by_year",
        "all_link_baseline_distance_distribution",
        *_comparison_fieldnames("all_link"),
    ]
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _comparison_fieldnames(prefix: str) -> list[str]:
    return [
        f"{prefix}_{suffix}"
        for suffix in (
            "disconnection_probability",
            "baseline_disconnection_probability",
            "disconnection_risk_difference",
            "disconnection_ci_low",
            "disconnection_ci_high",
            "disconnection_p_value",
            "disconnection_q_value",
            "mean_distance_shift",
            "mean_distance_shift_ci_low",
            "mean_distance_shift_ci_high",
            "wasserstein_distance",
            "wasserstein_ci_low",
            "wasserstein_ci_high",
            "wasserstein_p_value",
            "wasserstein_q_value",
            "repeat_probability",
            "baseline_repeat_probability",
            "repeat_risk_difference",
            "repeat_risk_difference_ci_low",
            "repeat_risk_difference_ci_high",
        )
    ]


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
