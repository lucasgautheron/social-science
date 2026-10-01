"""Binomial logistic curves with an unknown shift year.

Each cluster-year count is a binomial proportion of that year's papers. Four
curves compete, and the reported curve is the family with the highest Laplace
evidence. An exact tie prefers the earlier family: trend, then step, then
shock, then bump. The reported class is trend increasing, trend decreasing,
step, shock, bump increasing, or bump decreasing. Trend and bump take the
majority sign of their posterior draws. Step and shock are not split by sign.

    trend: logit p = intercept + beta * u
    step:  logit p = intercept + c * 1{year >= tau}
    shock: logit p = intercept + c * 1{year >= tau} * 2^(-(year - tau) / H)
    bump:  logit p = intercept + A * exp(-(year - tau)^2 / (2 w^2))

Trend, step, and shock try every interior year and keep the year with the
highest evidence. u is standardized time since that year and is zero through
it, so the trend is the intercept until the shift and a slope afterward. The
step is a permanent level shift. The shock jumps by c at tau and then loses
half of that jump every H calendar years, returning to the intercept.

The bump center is a continuous parameter. Its prior is normal, centered on
the midpoint of the observed years, with standard deviation equal to their
span, so the center may sit outside the window. A positive amplitude is bump
increasing and a negative amplitude is bump decreasing. The deviation returns
to the intercept on both sides. H and w are in calendar years. The reported
bump year is the posterior mean of the center.

Parallel sampling uses one process per worker. Each process compiles each
curve once, swaps the counts, and samples the retained shift. Chains stay
inside the process, and BLAS threads are pinned to one.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import logging
import multiprocessing as mp
import os
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

INCREASING = "increasing slope"
DECREASING = "decreasing slope"
STEP_UP = "upward level shift"
STEP_DOWN = "downward level shift"
SHOCK_UP = "upward shock"
SHOCK_DOWN = "downward shock"
INVERTED_U = "inverted U shape"
U_SHAPE = "U shape"
TREND_INCREASING = "trend increasing"
TREND_DECREASING = "trend decreasing"
STEP_LABEL = "step"
SHOCK_LABEL = "shock"
BUMP_INCREASING = "bump increasing"
BUMP_DECREASING = "bump decreasing"
CLUSTER_TYPES = (
    TREND_INCREASING,
    TREND_DECREASING,
    STEP_LABEL,
    SHOCK_LABEL,
    BUMP_INCREASING,
    BUMP_DECREASING,
)
MODEL_NAMES = ("trend", "step", "shock", "bump")
TREND_SHAPES = (INCREASING, DECREASING)
STEP_SHAPES = (STEP_UP, STEP_DOWN)
SHOCK_SHAPES = (SHOCK_UP, SHOCK_DOWN)
BUMP_SHAPES = (INVERTED_U, U_SHAPE)

DEFAULT_DRAWS = 1000
DEFAULT_TUNE = 1000
DEFAULT_CHAINS = 2
DEFAULT_LEVEL = 0
DEFAULT_SEED = 0
INTERCEPT_SIGMA = 5.0
SLOPE_SIGMA = 1.0
JUMP_SIGMA = 1.0
AMPLITUDE_SIGMA = 1.0
HALF_LIFE_SIGMA = 10.0
WIDTH_SIGMA = 10.0
BUMP_TAU_SPAN = 1.0
NUTS_SAMPLERS = ("pymc", "nutpie", "blackjax", "numpyro")
TRENDS_CSV = "cluster_trends.csv"
_REQUIRED_COLUMNS = ("level", "group", "year", "papers", "share")
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

FIELDNAMES = (
    "level",
    "group",
    "year_min",
    "year_max",
    "n_years",
    "successes_total",
    "trials_total",
    "preferred_model",
    "compatible",
    "trend_break_year",
    "step_break_year",
    "shock_break_year",
    "bump_break_year",
    "trend_log_evidence",
    "step_log_evidence",
    "shock_log_evidence",
    "bump_log_evidence",
    "log_bayes_factor",
    "trend_posterior",
    "step_posterior",
    "shock_posterior",
    "bump_posterior",
    "p_increasing",
    "p_decreasing",
    "p_upward_level_shift",
    "p_downward_level_shift",
    "p_upward_shock",
    "p_downward_shock",
    "p_inverted_u",
    "p_u",
    "trend_intercept_mean",
    "trend_intercept_sd",
    "trend_beta_mean",
    "trend_beta_sd",
    "trend_u_max",
    "step_intercept_mean",
    "step_intercept_sd",
    "step_c_mean",
    "step_c_sd",
    "shock_intercept_mean",
    "shock_intercept_sd",
    "shock_c_mean",
    "shock_c_sd",
    "shock_half_life_mean",
    "shock_half_life_sd",
    "bump_intercept_mean",
    "bump_intercept_sd",
    "bump_amplitude_mean",
    "bump_amplitude_sd",
    "bump_width_mean",
    "bump_width_sd",
    "time_mean",
    "time_scale",
    "trend_divergences",
    "step_divergences",
    "shock_divergences",
    "bump_divergences",
    "trend_rhat_max",
    "step_rhat_max",
    "shock_rhat_max",
    "bump_rhat_max",
)


@dataclass(frozen=True)
class ClusterSeries:
    """Binomial counts for one cluster on the shared year grid."""

    level: int
    group: int
    years: np.ndarray
    successes: np.ndarray
    trials: np.ndarray

    def __post_init__(self) -> None:
        if self.years.shape != self.successes.shape or self.years.shape != self.trials.shape:
            raise ValueError("Years, successes, and trials must have the same length")


@dataclass(frozen=True)
class BreakFit:
    """Posterior of the retained break for one curve family."""

    log_evidence: float
    break_year: float
    u_max: float
    parameters: Mapping[str, np.ndarray]
    divergences: int
    rhat_max: float


@dataclass(frozen=True)
class _WorkerTask:
    level: int
    years: np.ndarray
    standardized_time: np.ndarray
    time_mean: float
    time_scale: float
    groups: tuple[tuple[int, np.ndarray, np.ndarray], ...]
    draws: int
    tune: int
    chains: int
    seed: int
    nuts_sampler: str


def load_cluster_series(
    clusters_dir: str | Path,
    *,
    level: int,
    events_dir: str | Path | None = None,
    groups: Collection[int] | None = None,
) -> tuple[ClusterSeries, ...]:
    """Read cluster-year counts and align every cluster to one year grid.

    Years with no row for a cluster contribute zero successes. The trial total
    is the corpus size that year, taken from the events manifest when it is
    available and recovered from ``papers / share`` otherwise.
    """
    if level < 0:
        raise ValueError("--level must be >= 0")
    root = Path(clusters_dir).expanduser().resolve()
    path = root / "cluster_by_year.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} was not found. Run cluster-events before cluster-trends."
        )
    manifest_trials = _manifest_papers_by_year(
        None if events_dir is None else Path(events_dir).expanduser().resolve()
    )
    papers, inferred = _read_cluster_rows(path, level)
    if not papers and not manifest_trials:
        raise ValueError(f"No cluster-year rows at level {level}")

    trials_by_year: dict[int, int] = {}
    for year in sorted(set(inferred) | set(manifest_trials)):
        trials = _consensus_trials(inferred.get(year, []), manifest_trials.get(year), year)
        if trials > 0:
            trials_by_year[year] = trials
    if len(trials_by_year) < 3:
        raise ValueError(
            "At least three years with papers are required"
        )

    present = sorted({group for group, _year in papers})
    selected = _selected_groups(present, groups)
    year_grid = np.array(sorted(trials_by_year), dtype=np.int32)
    trial_grid = np.array([trials_by_year[int(year)] for year in year_grid], dtype=np.int64)
    series = []
    for group in selected:
        successes = np.array(
            [papers.get((group, int(year)), 0) for year in year_grid],
            dtype=np.int64,
        )
        if np.any(successes > trial_grid):
            raise ValueError(f"Group {group} has more cluster papers than corpus papers in a year")
        series.append(
            ClusterSeries(
                level=level,
                group=group,
                years=year_grid,
                successes=successes,
                trials=trial_grid,
            )
        )
    return tuple(series)


def default_events_dir(clusters_dir: str | Path) -> Path | None:
    """Return a sibling events directory when its manifest is present."""
    sibling = Path(clusters_dir).expanduser().resolve().parent / "events"
    if (sibling / "manifest.json").is_file():
        return sibling
    return None


def standardize_time(years: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Center years and scale them to unit standard deviation."""
    values = np.asarray(years, dtype=np.float64)
    if values.size < 3:
        raise ValueError(
            "At least three years are required"
        )
    center = float(values.mean())
    scale = float(values.std(ddof=0))
    if scale == 0.0:
        raise ValueError("Years must span more than one calendar year")
    return (values - center) / scale, center, scale


def break_indices(n_years: int) -> np.ndarray:
    """Return break positions with one year before them and one year after them."""
    if n_years < 4:
        raise ValueError(
            "At least four years are required so a break can sit after the first year and before the last"
        )
    return np.arange(1, n_years - 1)


@dataclass(frozen=True)
class BreakDesign:
    """Covariates for one candidate shift year."""

    after: np.ndarray
    since: np.ndarray
    elapsed_years: np.ndarray
    centered_years: np.ndarray
    u_max: float


def break_design(standardized_time: np.ndarray, years: np.ndarray, break_index: int) -> BreakDesign:
    """Build the trend, step, and shock covariates for one shift year.

    Standardized time since the shift is zero through that year. Calendar time
    since the shift is zero before it as well. ``centered_years`` is calendar
    time relative to the shift year.
    """
    time = np.asarray(standardized_time, dtype=np.float64)
    calendar = np.asarray(years, dtype=np.float64)
    if time.shape != calendar.shape:
        raise ValueError("Standardized time and calendar years must have the same length")
    if break_index <= 0 or break_index >= time.size - 1:
        raise ValueError("The break must leave one year before it and one year after it")
    after = np.zeros(time.shape, dtype=np.float64)
    after[break_index:] = 1.0
    since = (time - time[break_index]) * after
    centered = calendar - calendar[break_index]
    elapsed = np.maximum(centered, 0.0)
    return BreakDesign(
        after=after,
        since=since,
        elapsed_years=elapsed,
        centered_years=centered,
        u_max=float(since[-1]),
    )


def shock_multiplier(elapsed_years: np.ndarray, half_life: float) -> np.ndarray:
    """Remaining fraction of a jump after ``elapsed_years`` calendar years."""
    elapsed = np.asarray(elapsed_years, dtype=np.float64)
    if half_life <= 0:
        raise ValueError("Half-life must be positive")
    return np.exp(-np.log(2.0) * elapsed / float(half_life))


def bump_tau_prior(years: np.ndarray) -> tuple[float, float]:
    """Return the normal prior mean and sd of the bump center.

    The mean is the midpoint of the observed years. The sd is their span times
    ``BUMP_TAU_SPAN``, so a center one window-width outside either end is 1.5
    standard deviations from the middle when the scale is one span.
    """
    calendar = np.asarray(years, dtype=np.float64)
    if calendar.size == 0:
        raise ValueError("Years are required to place the bump center prior")
    low = float(calendar.min())
    high = float(calendar.max())
    span = high - low
    if span <= 0.0:
        raise ValueError("Years must span more than one calendar year")
    return 0.5 * (low + high), span * BUMP_TAU_SPAN


def bump_multiplier(centered_years: np.ndarray, width: float) -> np.ndarray:
    """Gaussian weight centered on the shift year, with width in calendar years."""
    centered = np.asarray(centered_years, dtype=np.float64)
    if width <= 0:
        raise ValueError("Bump width must be positive")
    return np.exp(-(centered**2) / (2.0 * float(width) ** 2))


def signed_labels(values: np.ndarray, positive: str, negative: str) -> np.ndarray:
    """Label each draw by the sign of one coefficient. Zero counts as positive."""
    draws = np.asarray(values, dtype=np.float64).ravel()
    if draws.size == 0:
        raise ValueError("Posterior draws are empty")
    labels = np.empty(draws.shape, dtype=object)
    labels[draws >= 0.0] = positive
    labels[draws < 0.0] = negative
    return labels


def laplace_log_evidence(log_joint: float, information: np.ndarray) -> float:
    """Laplace approximation of a marginal likelihood.

    ``information`` is the negative Hessian of the log joint at its mode.
    The dimension term is ``(d / 2) log(2π) - (1 / 2) log det(information)``.
    """
    matrix = np.atleast_2d(np.asarray(information, dtype=np.float64))
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Laplace information matrix must be square")
    sign, logdet = np.linalg.slogdet(matrix)
    if sign <= 0 or not np.isfinite(logdet) or not np.isfinite(log_joint):
        raise RuntimeError("Laplace approximation requires a positive definite information matrix")
    dimension = matrix.shape[0]
    return float(log_joint + 0.5 * dimension * np.log(2.0 * np.pi) - 0.5 * logdet)


def posterior_model_probabilities(log_evidences: Sequence[float]) -> np.ndarray:
    """Posterior model probabilities under equal prior odds."""
    evidences = np.asarray(list(log_evidences), dtype=np.float64)
    if evidences.ndim != 1 or evidences.size < 2:
        raise ValueError("At least two model evidences are required")
    if not np.all(np.isfinite(evidences)):
        raise RuntimeError("Model evidence is not finite")
    shifted = np.exp(evidences - np.max(evidences))
    return shifted / float(shifted.sum())


def preferred_model(log_evidences: Mapping[str, float]) -> str:
    """Return the family with the highest evidence. Ties follow ``MODEL_NAMES``."""
    missing = [name for name in MODEL_NAMES if name not in log_evidences]
    if missing:
        raise ValueError(f"Missing model evidence for {', '.join(missing)}")
    return max(MODEL_NAMES, key=lambda name: (float(log_evidences[name]), -MODEL_NAMES.index(name)))


def _class_share(labels: np.ndarray, order: Sequence[str]) -> dict[str, float]:
    return {name: float(np.mean(labels == name)) for name in order}


def _majority(share: Mapping[str, float], order: Sequence[str]) -> str:
    return max(order, key=lambda name: (share[name], -order.index(name)))


def _reported_type(winner: str, shares: Mapping[str, Mapping[str, float]]) -> str:
    """Collapse the winning curve to one of the six cluster classes."""
    if winner == "step":
        return STEP_LABEL
    if winner == "shock":
        return SHOCK_LABEL
    if winner == "trend":
        shape = _majority(shares["trend"], TREND_SHAPES)
        return TREND_INCREASING if shape == INCREASING else TREND_DECREASING
    shape = _majority(shares["bump"], BUMP_SHAPES)
    return BUMP_INCREASING if shape == INVERTED_U else BUMP_DECREASING


def trend_record(
    series: ClusterSeries,
    *,
    time_mean: float,
    time_scale: float,
    trend: BreakFit,
    step: BreakFit,
    shock: BreakFit,
    bump: BreakFit,
) -> dict[str, object]:
    """Summarize the retained shift year of each curve and the winning family."""
    fits = {"trend": trend, "step": step, "shock": shock, "bump": bump}
    evidences = {name: float(fit.log_evidence) for name, fit in fits.items()}
    probabilities = posterior_model_probabilities([evidences[name] for name in MODEL_NAMES])
    probability = dict(zip(MODEL_NAMES, probabilities, strict=True))
    winner = preferred_model(evidences)
    shares = {
        "trend": _class_share(signed_labels(trend.parameters["beta"], INCREASING, DECREASING), TREND_SHAPES),
        "step": _class_share(signed_labels(step.parameters["c"], STEP_UP, STEP_DOWN), STEP_SHAPES),
        "shock": _class_share(signed_labels(shock.parameters["c"], SHOCK_UP, SHOCK_DOWN), SHOCK_SHAPES),
        "bump": _class_share(signed_labels(bump.parameters["amplitude"], INVERTED_U, U_SHAPE), BUMP_SHAPES),
    }
    runner_up = max(evidences[name] for name in MODEL_NAMES if name != winner)
    trend_intercept = _mean_sd(np.asarray(trend.parameters["intercept"]))
    trend_slope = _mean_sd(np.asarray(trend.parameters["beta"]))
    step_intercept = _mean_sd(np.asarray(step.parameters["intercept"]))
    step_jump = _mean_sd(np.asarray(step.parameters["c"]))
    shock_intercept = _mean_sd(np.asarray(shock.parameters["intercept"]))
    shock_jump = _mean_sd(np.asarray(shock.parameters["c"]))
    shock_half_life = _mean_sd(np.asarray(shock.parameters["half_life"]))
    bump_intercept = _mean_sd(np.asarray(bump.parameters["intercept"]))
    bump_amplitude = _mean_sd(np.asarray(bump.parameters["amplitude"]))
    bump_width = _mean_sd(np.asarray(bump.parameters["width"]))
    return {
        "level": int(series.level),
        "group": int(series.group),
        "year_min": int(series.years.min()),
        "year_max": int(series.years.max()),
        "n_years": int(series.years.size),
        "successes_total": int(series.successes.sum()),
        "trials_total": int(series.trials.sum()),
        "preferred_model": winner,
        "compatible": _reported_type(winner, shares),
        "trend_break_year": int(trend.break_year),
        "step_break_year": int(step.break_year),
        "shock_break_year": int(shock.break_year),
        "bump_break_year": float(bump.break_year),
        "trend_log_evidence": evidences["trend"],
        "step_log_evidence": evidences["step"],
        "shock_log_evidence": evidences["shock"],
        "bump_log_evidence": evidences["bump"],
        "log_bayes_factor": evidences[winner] - runner_up,
        "trend_posterior": float(probability["trend"]),
        "step_posterior": float(probability["step"]),
        "shock_posterior": float(probability["shock"]),
        "bump_posterior": float(probability["bump"]),
        "p_increasing": shares["trend"][INCREASING],
        "p_decreasing": shares["trend"][DECREASING],
        "p_upward_level_shift": shares["step"][STEP_UP],
        "p_downward_level_shift": shares["step"][STEP_DOWN],
        "p_upward_shock": shares["shock"][SHOCK_UP],
        "p_downward_shock": shares["shock"][SHOCK_DOWN],
        "p_inverted_u": shares["bump"][INVERTED_U],
        "p_u": shares["bump"][U_SHAPE],
        "trend_intercept_mean": trend_intercept[0],
        "trend_intercept_sd": trend_intercept[1],
        "trend_beta_mean": trend_slope[0],
        "trend_beta_sd": trend_slope[1],
        "trend_u_max": float(trend.u_max),
        "step_intercept_mean": step_intercept[0],
        "step_intercept_sd": step_intercept[1],
        "step_c_mean": step_jump[0],
        "step_c_sd": step_jump[1],
        "shock_intercept_mean": shock_intercept[0],
        "shock_intercept_sd": shock_intercept[1],
        "shock_c_mean": shock_jump[0],
        "shock_c_sd": shock_jump[1],
        "shock_half_life_mean": shock_half_life[0],
        "shock_half_life_sd": shock_half_life[1],
        "bump_intercept_mean": bump_intercept[0],
        "bump_intercept_sd": bump_intercept[1],
        "bump_amplitude_mean": bump_amplitude[0],
        "bump_amplitude_sd": bump_amplitude[1],
        "bump_width_mean": bump_width[0],
        "bump_width_sd": bump_width[1],
        "time_mean": float(time_mean),
        "time_scale": float(time_scale),
        "trend_divergences": int(trend.divergences),
        "step_divergences": int(step.divergences),
        "shock_divergences": int(shock.divergences),
        "bump_divergences": int(bump.divergences),
        "trend_rhat_max": float(trend.rhat_max),
        "step_rhat_max": float(step.rhat_max),
        "shock_rhat_max": float(shock.rhat_max),
        "bump_rhat_max": float(bump.rhat_max),
    }


def write_trend_csv(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Atomically write one trend row per cluster."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDNAMES))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in FIELDNAMES})
    os.replace(temporary, destination)


def assign_chunks(items: Sequence[object], workers: int) -> list[list[object]]:
    """Interleave items across workers so one chunk cannot hold every later item."""
    if workers < 1:
        raise ValueError("--workers must be >= 1")
    count = min(workers, len(items))
    chunks: list[list[object]] = [[] for _ in range(count)]
    for index, item in enumerate(items):
        chunks[index % count].append(item)
    return chunks


def fit_cluster_trends(
    clusters_dir: str | Path,
    output: str | Path | None = None,
    *,
    level: int = DEFAULT_LEVEL,
    events_dir: str | Path | None = None,
    groups: Collection[int] | None = None,
    draws: int = DEFAULT_DRAWS,
    tune: int = DEFAULT_TUNE,
    chains: int = DEFAULT_CHAINS,
    workers: int | None = None,
    seed: int = DEFAULT_SEED,
    nuts_sampler: str = "pymc",
) -> dict[str, object]:
    """Score every interior break, sample the retained curves, and write the CSV.

    More than one worker uses the spawn start method, so a script that calls
    this function needs a ``if __name__ == "__main__"`` guard. The
    ``cluster-trends`` command already has one.
    """
    _validate_sampler_settings(
        draws=draws,
        tune=tune,
        chains=chains,
        workers=workers,
        nuts_sampler=nuts_sampler,
    )
    _limit_blas_threads()
    _import_pymc()

    root = Path(clusters_dir).expanduser().resolve()
    resolved_events = (
        Path(events_dir).expanduser().resolve()
        if events_dir is not None
        else default_events_dir(root)
    )
    if resolved_events is None:
        logger.info("Yearly corpus sizes are recovered from cluster shares")
    else:
        logger.info("Using yearly corpus sizes from %s", resolved_events)
    series = load_cluster_series(
        root,
        level=level,
        events_dir=resolved_events,
        groups=groups,
    )
    _shared_year_grid(series)
    break_indices(int(series[0].years.size))
    standardized, time_mean, time_scale = standardize_time(series[0].years)
    worker_count = workers if workers is not None else max(1, (os.cpu_count() or 1) - 1)
    worker_count = max(1, min(worker_count, len(series)))
    tasks = _worker_tasks(
        series,
        standardized_time=standardized,
        time_mean=time_mean,
        time_scale=time_scale,
        draws=draws,
        tune=tune,
        chains=chains,
        seed=seed,
        nuts_sampler=nuts_sampler,
        workers=worker_count,
    )
    logger.info(
        "Fitting %s clusters at level %s with %s workers",
        len(series),
        level,
        len(tasks),
    )
    if len(tasks) == 1:
        fitted = _fit_worker(tasks[0])
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(tasks), mp_context=context) as executor:
            fitted = [row for chunk in executor.map(_fit_worker, tasks) for row in chunk]
    fitted.sort(key=lambda row: (int(row["level"]), int(row["group"])))

    destination = Path(output).expanduser().resolve() if output is not None else root / TRENDS_CSV
    write_trend_csv(destination, fitted)
    summary = {
        "output": str(destination),
        "clusters": len(fitted),
        "level": int(level),
        "workers": len(tasks),
        "years": int(series[0].years.size),
        "preferred_model": dict(sorted(Counter(row["preferred_model"] for row in fitted).items())),
        "compatible": dict(sorted(Counter(row["compatible"] for row in fitted).items())),
    }
    logger.info("Wrote cluster trends for %s clusters to %s", len(fitted), destination)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a flat-then-linear trend, a level shift, a decaying shock, and a "
            "Gaussian bump. Trend, step, and shock keep the interior year with the "
            "highest Laplace evidence. The bump center is continuous and may fall "
            "outside the observed years. The preferred curve is the highest evidence."
        )
    )
    parser.add_argument("--clusters-dir", type=Path, default=Path("output/event_clusters"))
    parser.add_argument(
        "--events-dir",
        type=Path,
        default=None,
        help=(
            "Event directory whose manifest supplies papers_by_year. "
            "Defaults to a sibling events directory when manifest.json exists."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"CSV path. Defaults to cluster_trends.csv inside --clusters-dir ({TRENDS_CSV}).",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=DEFAULT_LEVEL,
        help=f"Hierarchy level to model (default {DEFAULT_LEVEL}).",
    )
    parser.add_argument(
        "--groups",
        type=_groups_argument,
        default=None,
        help="Comma-separated cluster ids. Defaults to every cluster at --level.",
    )
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS, help=f"Draws per chain (default {DEFAULT_DRAWS}).")
    parser.add_argument("--tune", type=int, default=DEFAULT_TUNE, help=f"Tuning steps per chain (default {DEFAULT_TUNE}).")
    parser.add_argument(
        "--chains",
        type=int,
        default=DEFAULT_CHAINS,
        help=f"Chains per model, run sequentially inside each worker (default {DEFAULT_CHAINS}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Process workers. Default is one less than the CPU count, capped by the cluster count.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--nuts-sampler", choices=NUTS_SAMPLERS, default="pymc")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args(argv)
    summary = fit_cluster_trends(
        args.clusters_dir,
        args.output,
        level=args.level,
        events_dir=args.events_dir,
        groups=args.groups,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        workers=args.workers,
        seed=args.seed,
        nuts_sampler=args.nuts_sampler,
    )
    print(json.dumps(summary, indent=2))
    return 0


_CURVE_DATA = {
    "trend": ("since",),
    "step": ("after",),
    "shock": ("after", "elapsed_years"),
}
_PROFILED_MODELS = ("trend", "step", "shock")
_CURVE_PARAMETERS = {
    "trend": ("intercept", "beta"),
    "step": ("intercept", "c"),
    "shock": ("intercept", "c", "half_life"),
    "bump": ("intercept", "amplitude", "width", "tau"),
}


def _fit_worker(task: _WorkerTask) -> list[dict[str, object]]:
    """Compile each curve once, score its shift, and sample the retained fit."""
    _configure_worker_logging()
    pm, az = _import_pymc()
    if not task.groups:
        return []
    design = break_design(task.standardized_time, task.years, int(break_indices(int(task.years.size))[0]))
    first_successes = task.groups[0][1]
    first_trials = task.groups[0][2]
    models = {
        "trend": _trend_model(pm, design.since, first_trials, first_successes),
        "step": _step_model(pm, design.after, first_trials, first_successes),
        "shock": _shock_model(pm, design.after, design.elapsed_years, first_trials, first_successes),
        "bump": _bump_model(pm, task.years, first_trials, first_successes),
    }
    evidence = {name: _LaplaceEvidence(model) for name, model in models.items()}
    rows = []
    for group, successes, trials in task.groups:
        cluster_seed = _cluster_seed(task.seed, task.level, group)
        logger.info("Scoring shift years for level %s group %s", task.level, group)
        retained = {
            name: _best_break(
                pm,
                models[name],
                evidence[name],
                successes,
                trials,
                task.standardized_time,
                task.years,
                data_names=_CURVE_DATA[name],
            )
            for name in _PROFILED_MODELS
        }
        _set_counts(pm, models["bump"], successes, trials)
        try:
            bump_log_evidence = float(evidence["bump"].log_evidence())
        except RuntimeError as exc:
            raise RuntimeError(
                f"Bump posterior mode is unidentified for level {task.level} group {group}"
            ) from exc
        fits = {}
        for offset, name in enumerate(MODEL_NAMES):
            if name == "bump":
                inference = _sample_model(
                    pm,
                    models[name],
                    draws=task.draws,
                    tune=task.tune,
                    chains=task.chains,
                    seed=cluster_seed + offset,
                    nuts_sampler=task.nuts_sampler,
                )
                center = np.asarray(inference.posterior["tau"].values, dtype=np.float64).ravel()
                center_year = float(np.mean(center))
                if not np.isfinite(center_year):
                    raise RuntimeError(
                        f"Bump center is unidentified for level {task.level} group {group}"
                    )
                fits[name] = _break_fit(
                    az,
                    inference,
                    names=_CURVE_PARAMETERS[name],
                    log_evidence=bump_log_evidence,
                    break_year=center_year,
                    u_max=0.0,
                )
                continue
            _set_curve_data(pm, models[name], successes, trials, retained[name].design, _CURVE_DATA[name])
            inference = _sample_model(
                pm,
                models[name],
                draws=task.draws,
                tune=task.tune,
                chains=task.chains,
                seed=cluster_seed + offset,
                nuts_sampler=task.nuts_sampler,
            )
            fits[name] = _break_fit(
                az,
                inference,
                names=_CURVE_PARAMETERS[name],
                log_evidence=retained[name].log_evidence,
                break_year=retained[name].break_year,
                u_max=retained[name].design.u_max,
            )
        series = ClusterSeries(
            level=task.level,
            group=group,
            years=task.years,
            successes=successes,
            trials=trials,
        )
        row = trend_record(
            series,
            time_mean=task.time_mean,
            time_scale=task.time_scale,
            trend=fits["trend"],
            step=fits["step"],
            shock=fits["shock"],
            bump=fits["bump"],
        )
        logger.info(
            "Level %s group %s: preferred %s, compatible %s, break %s, log Bayes factor %.2f",
            task.level,
            group,
            row["preferred_model"],
            row["compatible"],
            row[f"{row['preferred_model']}_break_year"],
            row["log_bayes_factor"],
        )
        worst_rhat = max(float(row[f"{name}_rhat_max"]) for name in MODEL_NAMES)
        if worst_rhat > 1.01:
            logger.warning(
                "Level %s group %s has split-R-hat above 1.01 (trend %.3f, step %.3f, shock %.3f, bump %.3f)",
                task.level,
                group,
                row["trend_rhat_max"],
                row["step_rhat_max"],
                row["shock_rhat_max"],
                row["bump_rhat_max"],
            )
        rows.append(row)
    return rows


@dataclass(frozen=True)
class _ScoredBreak:
    log_evidence: float
    break_year: int
    design: BreakDesign


def _best_break(pm, model, evidence, successes, trials, standardized_time, years, *, data_names: Sequence[str]) -> _ScoredBreak:
    chosen: _ScoredBreak | None = None
    for index in break_indices(len(years)):
        design = break_design(standardized_time, years, int(index))
        _set_curve_data(pm, model, successes, trials, design, data_names)
        try:
            value = float(evidence.log_evidence())
        except RuntimeError:
            logger.warning("Skipping shift year %s because the posterior mode is unidentified", int(years[index]))
            value = -np.inf
        if chosen is None or value > chosen.log_evidence:
            chosen = _ScoredBreak(log_evidence=value, break_year=int(years[index]), design=design)
    if chosen is None or not np.isfinite(chosen.log_evidence):
        raise RuntimeError("No shift year has a finite Laplace evidence")
    return chosen


def _binomial_data(pm, trials: np.ndarray, successes: np.ndarray):
    observed = pm.Data("successes", np.asarray(successes, dtype=np.int64))
    trial_totals = pm.Data("trials", np.asarray(trials, dtype=np.int64))
    return observed, trial_totals


def _trend_model(pm, since: np.ndarray, trials: np.ndarray, successes: np.ndarray):
    with pm.Model() as model:
        observed, trial_totals = _binomial_data(pm, trials, successes)
        elapsed = pm.Data("since", np.asarray(since, dtype=np.float64))
        intercept = pm.Normal("intercept", mu=0.0, sigma=INTERCEPT_SIGMA)
        slope = pm.Normal("beta", mu=0.0, sigma=SLOPE_SIGMA)
        pm.Binomial("obs", n=trial_totals, logit_p=intercept + slope * elapsed, observed=observed)
    return model


def _step_model(pm, after: np.ndarray, trials: np.ndarray, successes: np.ndarray):
    with pm.Model() as model:
        observed, trial_totals = _binomial_data(pm, trials, successes)
        active = pm.Data("after", np.asarray(after, dtype=np.float64))
        intercept = pm.Normal("intercept", mu=0.0, sigma=INTERCEPT_SIGMA)
        jump = pm.Normal("c", mu=0.0, sigma=JUMP_SIGMA)
        pm.Binomial("obs", n=trial_totals, logit_p=intercept + jump * active, observed=observed)
    return model


def _shock_model(pm, after: np.ndarray, elapsed_years: np.ndarray, trials: np.ndarray, successes: np.ndarray):
    with pm.Model() as model:
        observed, trial_totals = _binomial_data(pm, trials, successes)
        active = pm.Data("after", np.asarray(after, dtype=np.float64))
        elapsed = pm.Data("elapsed_years", np.asarray(elapsed_years, dtype=np.float64))
        intercept = pm.Normal("intercept", mu=0.0, sigma=INTERCEPT_SIGMA)
        jump = pm.Normal("c", mu=0.0, sigma=JUMP_SIGMA)
        half_life = pm.HalfNormal("half_life", sigma=HALF_LIFE_SIGMA)
        decay = pm.math.exp(-np.log(2.0) * elapsed / half_life)
        pm.Binomial("obs", n=trial_totals, logit_p=intercept + jump * active * decay, observed=observed)
    return model


def _bump_model(pm, years: np.ndarray, trials: np.ndarray, successes: np.ndarray):
    """Gaussian bump whose center is continuous and may leave the observed window."""
    calendar = np.asarray(years, dtype=np.float64)
    midpoint, scale = bump_tau_prior(calendar)
    with pm.Model() as model:
        observed, trial_totals = _binomial_data(pm, trials, successes)
        observed_years = pm.Data("years", calendar)
        intercept = pm.Normal("intercept", mu=0.0, sigma=INTERCEPT_SIGMA)
        amplitude = pm.Normal("amplitude", mu=0.0, sigma=AMPLITUDE_SIGMA)
        width = pm.HalfNormal("width", sigma=WIDTH_SIGMA)
        tau = pm.Normal("tau", mu=midpoint, sigma=scale)
        weight = pm.math.exp(-pm.math.sqr(observed_years - tau) / (2.0 * pm.math.sqr(width)))
        pm.Binomial("obs", n=trial_totals, logit_p=intercept + amplitude * weight, observed=observed)
    return model


def _set_counts(pm, model, successes, trials) -> None:
    with model:
        pm.set_data(
            {
                "successes": np.asarray(successes, dtype=np.int64),
                "trials": np.asarray(trials, dtype=np.int64),
            }
        )


def _set_curve_data(pm, model, successes, trials, design: BreakDesign, data_names: Sequence[str]) -> None:
    _set_counts(pm, model, successes, trials)
    payload = {}
    values = {
        "since": design.since,
        "after": design.after,
        "elapsed_years": design.elapsed_years,
        "centered_years": design.centered_years,
    }
    for name in data_names:
        payload[name] = np.asarray(values[name], dtype=np.float64)
    with model:
        pm.set_data(payload)


def _sample_model(pm, model, *, draws: int, tune: int, chains: int, seed: int, nuts_sampler: str):
    with model:
        kwargs = {
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "cores": 1,
            "random_seed": [seed + chain for chain in range(chains)],
            "progressbar": False,
            "nuts_sampler": nuts_sampler,
            "compute_convergence_checks": False,
        }
        parameters = inspect.signature(pm.sample).parameters
        if "blas_cores" in parameters:
            kwargs["blas_cores"] = 1
        if nuts_sampler == "pymc":
            kwargs["nuts"] = {"target_accept": 0.9}
        return pm.sample(**kwargs)


class _LaplaceEvidence:
    """Compiled Laplace marginal likelihood for one reused PyMC model.

    The log density and Hessian are compiled once. Later fits only change the
    binomial counts and the break design, then this reoptimizes the posterior mode.
    Optimization starts at the model initial point, which is the prior mean.
    """

    def __init__(self, model) -> None:
        from pymc.blocking import DictToArrayBijection, RaveledVars

        self._bijection = DictToArrayBijection
        self._raveled = RaveledVars
        self._logp = model.compile_logp(jacobian=True)
        self._dlogp = model.compile_dlogp(jacobian=True)
        self._hessian = model.compile_d2logp(jacobian=True, negate_output=False)
        start = model.initial_point()
        free = [var.name for var in model.continuous_value_vars]
        mapped = DictToArrayBijection.map({name: np.asarray(start[name]) for name in free})
        self._start = start
        self._info = mapped.point_map_info
        self._origin = np.asarray(mapped.data, dtype=np.float64).copy()

    def log_evidence(self) -> float:
        from scipy.optimize import minimize

        def objective(position: np.ndarray) -> tuple[float, np.ndarray]:
            point = self._point(position)
            logp = float(np.asarray(self._logp(point)))
            gradient = _ravel_gradient(self._dlogp(point))
            return -logp, -gradient

        fitted = minimize(
            objective,
            self._origin,
            method="L-BFGS-B",
            jac=True,
        )
        if not fitted.success:
            raise RuntimeError(f"Posterior mode search failed: {fitted.message}")
        mode = self._point(fitted.x)
        log_joint = float(np.asarray(self._logp(mode)))
        information = -np.asarray(self._hessian(mode), dtype=np.float64)
        return laplace_log_evidence(log_joint, information)

    def _point(self, position: np.ndarray):
        raveled = self._raveled(np.asarray(position, dtype=np.float64), self._info)
        return self._bijection.rmap(raveled, self._start)


def _break_fit(az, inference, *, names: Sequence[str], log_evidence: float, break_year: float, u_max: float) -> BreakFit:
    posterior = inference.posterior
    summary_kwargs = {"var_names": list(names), "kind": "all"}
    if "round_to" in inspect.signature(az.summary).parameters:
        summary_kwargs["round_to"] = "none"
    summary = az.summary(inference, **summary_kwargs)
    parameters = {
        name: np.asarray(posterior[name].values, dtype=np.float64).ravel()
        for name in names
    }
    return BreakFit(
        log_evidence=float(log_evidence),
        break_year=float(break_year),
        u_max=float(u_max),
        parameters=parameters,
        divergences=int(np.asarray(inference.sample_stats["diverging"]).sum()),
        rhat_max=_rhat_max(summary),
    )


def _ravel_gradient(value: object) -> np.ndarray:
    if isinstance(value, list | tuple):
        parts = [np.asarray(part, dtype=np.float64).ravel() for part in value]
        return np.concatenate(parts) if parts else np.array([], dtype=np.float64)
    return np.asarray(value, dtype=np.float64).ravel()


def _rhat_max(summary) -> float:
    for column in ("r_hat", "rhat"):
        if column in summary.columns:
            return float(summary[column].max())
    raise RuntimeError("Posterior summary did not include split-R-hat")


def _worker_tasks(
    series: Sequence[ClusterSeries],
    *,
    standardized_time: np.ndarray,
    time_mean: float,
    time_scale: float,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
    nuts_sampler: str,
    workers: int,
) -> list[_WorkerTask]:
    grouped = assign_chunks(
        [
            (
                int(item.group),
                np.asarray(item.successes, dtype=np.int64),
                np.asarray(item.trials, dtype=np.int64),
            )
            for item in series
        ],
        workers,
    )
    return [
        _WorkerTask(
            level=int(series[0].level),
            years=np.asarray(series[0].years, dtype=np.int32),
            standardized_time=np.asarray(standardized_time, dtype=np.float64),
            time_mean=float(time_mean),
            time_scale=float(time_scale),
            groups=tuple(chunk),
            draws=draws,
            tune=tune,
            chains=chains,
            seed=seed,
            nuts_sampler=nuts_sampler,
        )
        for chunk in grouped
        if chunk
    ]


def _read_cluster_rows(
    path: Path,
    level: int,
) -> tuple[dict[tuple[int, int], int], dict[int, list[int]]]:
    papers: dict[tuple[int, int], int] = {}
    inferred: dict[int, list[int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(_REQUIRED_COLUMNS) <= set(reader.fieldnames):
            raise ValueError(
                f"{path} must contain columns {', '.join(_REQUIRED_COLUMNS)}"
            )
        for row in reader:
            if _cell_int(row["level"], field="level") != level:
                continue
            group = _cell_int(row["group"], field="group")
            year = _cell_int(row["year"], field="year")
            count = _cell_int(row["papers"], field="papers")
            if count < 0:
                raise ValueError(f"Negative paper count for group {group} in {year}")
            key = (group, year)
            if key in papers:
                raise ValueError(f"Duplicate cluster-year row: level {level} group {group} year {year}")
            papers[key] = count
            if count > 0:
                share = float(row["share"])
                inferred.setdefault(year, []).append(_trials_from_share(count, share, year))
    return papers, inferred


def _trials_from_share(papers: int, share: float, year: int) -> int:
    if not np.isfinite(share) or share <= 0:
        raise ValueError(f"Year {year} has a positive paper count and a non-positive share")
    trials = int(round(papers / share))
    return max(trials, papers)


def _consensus_trials(candidates: Sequence[int], manifest_count: int | None, year: int) -> int:
    inferred = None
    if candidates:
        low = min(candidates)
        high = max(candidates)
        if high - low > 1:
            raise ValueError(
                f"Year {year} cluster shares imply inconsistent corpus sizes {sorted(set(candidates))}"
            )
        inferred = Counter(candidates).most_common(1)[0][0]
    if manifest_count is None:
        if inferred is None:
            raise ValueError(f"Year {year} has no corpus size")
        return int(inferred)
    if inferred is not None and abs(int(inferred) - int(manifest_count)) > 1:
        raise ValueError(
            f"Year {year} cluster shares imply {inferred} papers but the events manifest records {manifest_count}"
        )
    return int(manifest_count)


def _manifest_papers_by_year(events_dir: Path | None) -> dict[int, int]:
    if events_dir is None:
        return {}
    path = events_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} was not found. Re-run event extraction or omit --events-dir."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("papers_by_year")
    if not isinstance(raw, dict):
        raise ValueError(f"{path} does not contain papers_by_year")
    return {int(year): int(count) for year, count in raw.items() if int(count) > 0}


def _selected_groups(present: Sequence[int], groups: Collection[int] | None) -> list[int]:
    if groups is None:
        if not present:
            raise ValueError("No clusters have papers at this level")
        return list(present)
    selected = [int(group) for group in groups]
    if any(group < 0 for group in selected):
        raise ValueError("Cluster ids must be nonnegative")
    missing = [group for group in selected if group not in set(present)]
    if missing:
        raise ValueError(f"Cluster ids not present at this level: {missing}")
    return selected


def _cell_int(value: str, *, field: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Column {field} must be numeric, got {value!r}") from error
    if not np.isfinite(number) or not float(number).is_integer():
        raise ValueError(f"Column {field} must be an integer, got {value!r}")
    return int(number)


def _mean_sd(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("Posterior draws are empty")
    return float(np.mean(values)), float(np.std(values))


def _groups_argument(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip() != ""]
    if not parts:
        raise argparse.ArgumentTypeError("--groups must list at least one cluster id")
    try:
        groups = tuple(int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--groups must be comma-separated integers") from error
    if any(group < 0 for group in groups):
        raise argparse.ArgumentTypeError("Cluster ids must be nonnegative")
    return groups


def _shared_year_grid(series: Sequence[ClusterSeries]) -> None:
    years = series[0].years
    trials = series[0].trials
    for item in series[1:]:
        if not np.array_equal(item.years, years):
            raise ValueError("Cluster series must share one year grid")
        if not np.array_equal(item.trials, trials):
            raise ValueError("Cluster series must share one trial total per year")


def _validate_sampler_settings(*, draws: int, tune: int, chains: int, workers: int | None, nuts_sampler: str) -> None:
    if draws < 1:
        raise ValueError("--draws must be >= 1")
    if tune < 1:
        raise ValueError("--tune must be >= 1")
    if chains < 2:
        raise ValueError("--chains must be >= 2 so the split-R-hat is defined")
    if workers is not None and workers < 1:
        raise ValueError("--workers must be >= 1")
    if nuts_sampler not in NUTS_SAMPLERS:
        raise ValueError(f"--nuts-sampler must be one of {', '.join(NUTS_SAMPLERS)}")


def _cluster_seed(seed: int, level: int, group: int) -> int:
    value = (int(seed) + 0x9E3779B9) ^ (int(level) * 0x85EBCA6B) ^ (int(group) * 0xC2B2AE35)
    return int(value & 0x7FFFFFFF)


def _limit_blas_threads() -> None:
    for name in _THREAD_VARIABLES:
        os.environ[name] = "1"


def _configure_worker_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    for name in ("pymc", "pytensor", "arviz"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _import_pymc():
    try:
        import arviz
        import pymc
    except ImportError as error:
        raise RuntimeError(
            "PyMC and ArviZ are required for cluster-trends. "
            "Install the trends extra: python -m pip install -e '.[trends]'"
        ) from error
    return pymc, arviz


if __name__ == "__main__":
    raise SystemExit(main())
