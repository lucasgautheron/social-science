"""Shape classes for cluster publication series.

Each series is a yearly paper count divided by that year's corpus size. The
count is logged with a pseudocount of 5% of the series peak, then a
topic-specific year effect is removed. Flat, linear, and quadratic trends
compete with a permanent step and a positive decaying shock. Every candidate
includes an offset for 2023–2025 when both sides of that break are observed.
The abrupt shape is kept only when it beats the best smooth trend by more than
6 BIC points and explains at least 15% of the variance beyond that trend.

Smooth models are charged only for the columns they had before the break
offset is appended. A step is charged 4 parameters and a shock 5 when the
break is present, matching the discrete lambda on the shock grid. Without the
break those charges drop by one.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

TRENDS_CSV = "cluster_trends.csv"
MIN_YEARS = 8
MIN_DBIC = 6.0
MIN_BURST = 0.15
PSEUDO_FRACTION = 0.05
BREAK_YEAR = 2023
ONSET_FIRST = 2011
ONSET_LAST = 2022
TOPIC_REF_MIN = 15
LAMBDAS = (0.15, 0.25, 0.35, 0.5, 0.7, 1.0, 1.5, 2.5)
ARM_RATIO = 0.35

SHAPE_CLASSES = (
    "stable",
    "steady rise",
    "accelerating rise",
    "saturating rise",
    "rise then fall",
    "fall then rise",
    "steady decline",
    "accelerating decline",
    "flattening decline",
    "shock: transient",
    "shock: lasting",
    "abrupt drop",
)
RISE_CLASSES = frozenset({"steady rise", "accelerating rise", "saturating rise"})
DECLINE_CLASSES = frozenset(
    {"steady decline", "accelerating decline", "flattening decline"}
)
LINK_DISTANCE_CLASSES = (
    "stable",
    "rise",
    "decline",
    "rise then fall",
    "fall then rise",
    "shock: transient",
    "shock: lasting",
    "abrupt drop",
)
ABRUPT_CLASSES = frozenset({"shock: transient", "shock: lasting", "abrupt drop"})
FIELDNAMES = (
    "level",
    "group",
    "keywords",
    "papers",
    "family",
    "cls",
    "sub",
    "trend",
    "curvature",
    "burstiness",
    "shock_size",
    "onset",
    "half_life",
    "dbic_jump",
    "noise",
    "range",
    "peak_year",
    "activity",
)


def link_distance_class(subclass: str) -> str:
    """Collapse the three rise subclasses and the three decline subclasses."""
    if subclass in RISE_CLASSES:
        return "rise"
    if subclass in DECLINE_CLASSES:
        return "decline"
    return subclass


def log_activity(count: np.ndarray, share: np.ndarray) -> np.ndarray:
    """Log share with a pseudocount of 5% of each series' peak count.

    Years with a zero share take the median corpus size of the positive
    shares, so a missing year is not a divide-by-zero.
    """
    count = np.asarray(count, dtype=np.float64)
    share = np.asarray(share, dtype=np.float64)
    if count.shape != share.shape or count.ndim != 2 or count.shape[1] < 1:
        raise ValueError("Counts and shares must be matching series matrices")
    total = count / np.maximum(share, 1e-12)
    positive = total[share > 0]
    if positive.size == 0:
        raise ValueError("At least one positive share is required")
    total = np.array(total, copy=True)
    total[share == 0] = np.median(positive)
    total = np.maximum(total, 1e-12)
    pseudo = PSEUDO_FRACTION * count.max(axis=1, keepdims=True)
    numerator = np.maximum(count + pseudo, 1e-12)
    return np.log(numerator / total)


def topic_adjust(
    log_series: np.ndarray,
    distributions: Sequence[Sequence[Mapping[str, Any]] | None],
) -> np.ndarray:
    """Remove the topic-weighted median trajectory from each log series.

    A topic reference is the median demeaned trajectory of series whose top
    topic is that topic, and it is used only when at least 15 series share it.
    Missing topics fall back to the median of every series. Leftover topic
    weight, including a series with no topics, uses that global median.
    """
    activity = np.asarray(log_series, dtype=np.float64)
    if activity.ndim != 2:
        raise ValueError("Log activity must be a series matrix")
    if len(distributions) != activity.shape[0]:
        raise ValueError("Each series needs one topic distribution")
    centered = activity - activity.mean(axis=1, keepdims=True)
    global_reference = np.median(centered, axis=0)
    dominant = np.array([_dominant_topic(dist) for dist in distributions], dtype=int)
    references: dict[int, np.ndarray] = {}
    for topic in set(dominant.tolist()):
        if topic == -1:
            continue
        mask = dominant == topic
        if int(mask.sum()) >= TOPIC_REF_MIN:
            references[topic] = np.median(centered[mask], axis=0)
    adjusted = np.empty_like(activity)
    for index, distribution in enumerate(distributions):
        adjusted[index] = activity[index] - _topic_adjustment(
            distribution,
            references,
            global_reference,
        )
    return adjusted


def classify(activity: np.ndarray, years: np.ndarray) -> dict[str, Any]:
    """Choose a shape class for one topic-adjusted log-activity series."""
    year_grid = np.asarray(years, dtype=int)
    observed = np.asarray(activity, dtype=np.float64)
    if observed.shape != year_grid.shape or observed.ndim != 1:
        raise ValueError("Activity and years must have the same length")
    if observed.size < MIN_YEARS:
        raise ValueError(f"At least {MIN_YEARS} years are required")
    length = int(observed.size)
    centered = observed - observed.mean()
    total_variance = float(np.sum(centered**2))
    index = np.arange(length, dtype=np.float64)
    rescaled = (index - index.mean()) / index.mean()
    intercept = np.ones(length)
    break_column = (year_grid >= BREAK_YEAR).astype(np.float64)
    use_break = bool(np.any(break_column > 0) and np.any(break_column == 0))

    def fit(columns: np.ndarray) -> tuple[np.ndarray, float]:
        design = np.column_stack([columns, break_column]) if use_break else columns
        beta = np.linalg.lstsq(design, centered, rcond=None)[0]
        residual = centered - design @ beta
        return beta, float(np.sum(residual**2))

    def bic(rss: float, parameters: int) -> float:
        return length * np.log(rss / length + 1e-9) + parameters * np.log(length)

    rss: dict[str, float] = {}
    parameters: dict[str, int] = {}
    smooth_fit: dict[str, np.ndarray] = {}
    for name, columns in (
        ("flat", intercept[:, None]),
        ("linear", np.column_stack([intercept, rescaled])),
        ("quad", np.column_stack([intercept, rescaled, rescaled**2])),
    ):
        beta, rss[name] = fit(columns)
        parameters[name] = int(columns.shape[1])
        smooth_fit[name] = columns @ beta[: columns.shape[1]]

    onset_indices = [
        position
        for position, year in enumerate(year_grid.tolist())
        if position >= 1 and ONSET_FIRST <= int(year) <= ONSET_LAST
    ]
    best_shock: tuple[float, int, float, float] | None = None
    best_step: tuple[float, int, float] | None = None
    for onset in onset_indices:
        after = (index >= onset).astype(np.float64)
        beta, residual = fit(np.column_stack([intercept, rescaled, after]))
        if best_step is None or residual < best_step[0]:
            best_step = (residual, onset, float(beta[2]))
        for decay in LAMBDAS:
            pulse = np.where(index >= onset, np.exp(-decay * (index - onset)), 0.0)
            beta, residual = fit(np.column_stack([intercept, rescaled, pulse]))
            amplitude = float(beta[2])
            if amplitude > 0 and (best_shock is None or residual < best_shock[0]):
                best_shock = (residual, onset, amplitude, float(decay))
    if best_step is not None:
        rss["step"] = best_step[0]
        parameters["step"] = 4 if use_break else 3
    if best_shock is not None:
        rss["shock"] = best_shock[0]
        parameters["shock"] = 5 if use_break else 4

    scores = {name: bic(rss[name], parameters[name]) for name in rss}
    smooth = min(("flat", "linear", "quad"), key=scores.get)
    jump = None
    if "step" in scores:
        jump = "step"
        if scores.get("shock", np.inf) < scores["step"]:
            jump = "shock"
    if jump is None:
        burstiness = 0.0
        bayes_gap = -np.inf
        chosen = smooth
    else:
        burstiness = (
            max(0.0, (rss[smooth] - rss[jump]) / total_variance)
            if total_variance > 0
            else 0.0
        )
        bayes_gap = scores[smooth] - scores[jump]
        chosen = jump if bayes_gap > MIN_DBIC and burstiness >= MIN_BURST else smooth

    trend_name = min(("linear", "quad"), key=scores.get)
    trend_fit = smooth_fit[trend_name]
    midpoint = length // 2
    trend = float(trend_fit[-1] - trend_fit[0])
    curvature = float(
        (trend_fit[-1] - trend_fit[midpoint]) - (trend_fit[midpoint - 1] - trend_fit[0])
    )
    if jump == "shock" and best_shock is not None:
        _, onset_index, amplitude, decay = best_shock
        half_life = float(np.log(2) / decay)
    elif best_step is not None:
        _, onset_index, amplitude = best_step
        half_life = float("inf")
    else:
        onset_index = 0
        amplitude = 0.0
        half_life = float("inf")

    fitted = smooth_fit[smooth] if chosen in smooth_fit else None
    subclass = _subclass(
        chosen,
        fitted if fitted is not None else trend_fit,
        rescaled,
        trend,
        half_life,
        amplitude,
    )
    return {
        "family": "abrupt" if chosen in ("shock", "step") else "smooth",
        "cls": chosen,
        "sub": subclass,
        "trend": trend,
        "curvature": curvature,
        "burstiness": float(burstiness),
        "shock_size": abs(float(amplitude)),
        "onset": int(year_grid[onset_index]),
        "half_life": half_life,
        "dbic_jump": float(min(bayes_gap, 99.0)) if np.isfinite(bayes_gap) else 0.0,
        "noise": float(np.sqrt(rss[chosen] / length)),
        "range": float(centered.max() - centered.min()),
        "peak_year": int(year_grid[int(np.argmax(centered))]),
    }


def classify_cluster_list(
    clusters: Sequence[Mapping[str, Any]],
    *,
    papers_by_year: Mapping[int, int] | None = None,
) -> dict[str, Any] | None:
    """Classify one website cluster list.

    The year grid is the corpus years when they are supplied, otherwise the
    years that appear on the clusters. Fewer than eight years stays
    unclassified. Missing cluster-years contribute zero papers.
    """
    if not clusters:
        return None
    years = _year_grid(clusters, papers_by_year)
    if years.size < MIN_YEARS:
        return None
    count, share = _align_cluster_counts(clusters, years)
    try:
        activity = topic_adjust(
            log_activity(count, share),
            [cluster.get("topic_distribution") or [] for cluster in clusters],
        )
    except ValueError:
        return None
    series = []
    for cluster, row in zip(clusters, activity, strict=True):
        classified = classify(row, years)
        keywords = [str(keyword) for keyword in cluster.get("keywords") or []]
        series.append(
            {
                "group": int(cluster["group"]),
                "label": keywords[0] if keywords else str(cluster["group"]),
                "keywords": keywords,
                "sub": classified["sub"],
                "trend": round(float(classified["trend"]), 4),
                "burstiness": round(float(classified["burstiness"]), 4),
                "onset": int(classified["onset"]),
                "activity": [round(float(value), 5) for value in row],
            }
        )
    return {"years": [int(year) for year in years], "series": series}


def fit_cluster_trends(
    clusters_dir: str | Path,
    output: str | Path | None = None,
    *,
    level: int = 0,
    events_dir: str | Path | None = None,
    topics_dir: str | Path | None = None,
    groups: Collection[int] | None = None,
) -> dict[str, object]:
    """Classify one hierarchy level and write ``cluster_trends.csv``."""
    if level < 0:
        raise ValueError("--level must be >= 0")
    root = Path(clusters_dir).expanduser().resolve()
    path = root / "cluster_by_year.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} was not found. Run cluster-events before cluster-trends."
        )
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
    observed = _read_cluster_year(path, level)
    if not observed:
        raise ValueError(f"No cluster-year rows at level {level}")
    years = _combined_years(observed, papers_by_year)
    if years.size < MIN_YEARS:
        raise ValueError(f"At least {MIN_YEARS} years are required to classify shapes")
    present = sorted({group for group, _year in observed})
    selected = present if groups is None else [group for group in present if group in set(groups)]
    if groups is not None and not selected:
        raise ValueError("None of the requested groups are present at this level")
    keywords = _read_keywords(root / "clusters.csv", level)
    distributions = (
        {}
        if topics_dir is None
        else _read_topic_distributions(Path(topics_dir).expanduser().resolve())
    )
    clusters = [
        {
            "group": group,
            "keywords": keywords.get(group, []),
            "yearly": [
                {
                    "year": int(year),
                    "papers": int(observed.get((group, int(year)), (0, 0.0))[0]),
                    "share": float(observed.get((group, int(year)), (0, 0.0))[1]),
                }
                for year in years
            ],
            "topic_distribution": distributions.get(group, []),
        }
        for group in present
    ]
    classified = classify_cluster_list(clusters, papers_by_year={int(year): 1 for year in years})
    if classified is None:
        raise ValueError(f"At least {MIN_YEARS} years are required to classify shapes")
    by_group = {int(row["group"]): row for row in classified["series"]}
    records = []
    for cluster in clusters:
        group = int(cluster["group"])
        if group not in set(selected):
            continue
        shape = classify(by_group[group]["activity"], years)
        papers = int(sum(item["papers"] for item in cluster["yearly"]))
        records.append(
            _csv_row(
                level=level,
                group=group,
                keywords=cluster["keywords"],
                papers=papers,
                shape=shape,
                activity=by_group[group]["activity"],
            )
        )
    records.sort(key=lambda row: int(row["group"]))
    destination = Path(output).expanduser().resolve() if output is not None else root / TRENDS_CSV
    write_trend_csv(destination, records)
    summary = {
        "output": str(destination),
        "clusters": len(records),
        "level": int(level),
        "years": int(years.size),
        "sub": dict(sorted(Counter(row["sub"] for row in records).items())),
    }
    logger.info("Wrote shape classes for %s clusters to %s", len(records), destination)
    return summary


def write_trend_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write the shape-class table."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDNAMES))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in FIELDNAMES})
    temporary.replace(destination)


def default_events_dir(clusters_dir: str | Path) -> Path | None:
    """Return a sibling events directory when its manifest is present."""
    sibling = Path(clusters_dir).expanduser().resolve().parent / "events"
    if (sibling / "manifest.json").is_file():
        return sibling
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Classify each cluster's yearly publication share as a smooth trend "
            "or an abrupt shock. Topic-specific year effects are removed before "
            "the least-squares BIC comparison."
        )
    )
    parser.add_argument("--clusters-dir", type=Path, default=Path("output/event_clusters"))
    parser.add_argument(
        "--events-dir",
        type=Path,
        default=None,
        help=(
            "Event directory whose manifest supplies the year grid. "
            "Defaults to a sibling events directory when manifest.json exists."
        ),
    )
    parser.add_argument(
        "--cluster-topic-visualizations-dir",
        type=Path,
        default=None,
        help=(
            "visualize-cluster-topics output. Topic mixes in "
            "cluster_topic_summary.csv remove topic-specific year effects."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"CSV path. Defaults to {TRENDS_CSV} inside --clusters-dir.",
    )
    parser.add_argument("--level", type=int, default=0, help="Hierarchy level (default 0).")
    parser.add_argument(
        "--groups",
        type=_groups_argument,
        default=None,
        help="Comma-separated cluster ids. Defaults to every cluster at --level.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args(argv)
    summary = fit_cluster_trends(
        args.clusters_dir,
        args.output,
        level=args.level,
        events_dir=args.events_dir,
        topics_dir=args.cluster_topic_visualizations_dir,
        groups=args.groups,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _dominant_topic(distribution: Sequence[Mapping[str, Any]] | None) -> int:
    if not distribution:
        return -1
    top = min(
        distribution,
        key=lambda item: (-float(item.get("share") or 0.0), int(item["topic_id"])),
    )
    return int(top["topic_id"])


def _topic_adjustment(
    distribution: Sequence[Mapping[str, Any]] | None,
    references: Mapping[int, np.ndarray],
    global_reference: np.ndarray,
) -> np.ndarray:
    items = list(distribution or [])
    if not items:
        return global_reference
    weight = 0.0
    adjustment = np.zeros_like(global_reference)
    for item in items:
        share = float(item.get("share") or 0.0)
        weight += share
        reference = references.get(int(item["topic_id"]), global_reference)
        adjustment = adjustment + share * reference
    return adjustment + (1.0 - weight) * global_reference


def _subclass(
    chosen: str,
    fitted: np.ndarray,
    rescaled: np.ndarray,
    trend: float,
    half_life: float,
    amplitude: float,
) -> str:
    if chosen == "flat":
        return "stable"
    if chosen == "linear":
        return "steady rise" if fitted[-1] > fitted[0] else "steady decline"
    if chosen == "quad":
        quadratic, slope, offset = np.polyfit(rescaled, fitted, 2)
        if quadratic != 0:
            vertex = float(np.clip(-slope / (2 * quadratic), -1, 1))
        else:
            vertex = 1.0
        turning = float(np.polyval([quadratic, slope, offset], vertex))
        left = abs(turning - fitted[0])
        right = abs(fitted[-1] - turning)
        if min(left, right) > ARM_RATIO * max(left, right):
            return "rise then fall" if quadratic < 0 else "fall then rise"
        if trend > 0:
            return "accelerating rise" if quadratic > 0 else "saturating rise"
        return "accelerating decline" if quadratic < 0 else "flattening decline"
    if chosen == "shock":
        return "shock: transient" if half_life < 2 else "shock: lasting"
    return "shock: lasting" if amplitude > 0 else "abrupt drop"


def _year_grid(
    clusters: Sequence[Mapping[str, Any]],
    papers_by_year: Mapping[int, int] | None,
) -> np.ndarray:
    years = set()
    if papers_by_year:
        years.update(int(year) for year in papers_by_year)
    for cluster in clusters:
        for item in cluster.get("yearly") or []:
            years.add(int(item["year"]))
    if not years:
        return np.array([], dtype=int)
    return np.array(sorted(years), dtype=int)


def _align_cluster_counts(
    clusters: Sequence[Mapping[str, Any]],
    years: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    count = np.zeros((len(clusters), years.size), dtype=np.float64)
    share = np.zeros((len(clusters), years.size), dtype=np.float64)
    positions = {int(year): index for index, year in enumerate(years.tolist())}
    for row_index, cluster in enumerate(clusters):
        for item in cluster.get("yearly") or []:
            position = positions.get(int(item["year"]))
            if position is None:
                continue
            count[row_index, position] = float(item.get("papers") or 0)
            share[row_index, position] = float(item.get("share") or 0)
    return count, share


def _combined_years(
    observed: Mapping[tuple[int, int], tuple[int, float]],
    papers_by_year: Mapping[int, int],
) -> np.ndarray:
    years = {year for _group, year in observed}
    years.update(int(year) for year in papers_by_year)
    return np.array(sorted(years), dtype=int)


def _read_cluster_year(
    path: Path,
    level: int,
) -> dict[tuple[int, int], tuple[int, float]]:
    observed: dict[tuple[int, int], tuple[int, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"level", "group", "year", "papers", "share"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(
                f"{path} must contain columns level, group, year, papers, and share"
            )
        for row in reader:
            if int(row["level"]) != level:
                continue
            observed[(int(row["group"]), int(row["year"]))] = (
                int(row["papers"]),
                float(row["share"]),
            )
    return observed


def _read_keywords(path: Path, level: int) -> dict[int, list[str]]:
    if not path.is_file():
        return {}
    keywords: dict[int, list[str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"level", "group", "keywords"} <= set(
            reader.fieldnames
        ):
            raise ValueError(f"{path} must contain columns level, group, and keywords")
        for row in reader:
            if int(row["level"]) != level:
                continue
            members = [part.strip() for part in row["keywords"].split("|")]
            keywords[int(row["group"])] = [part for part in members if part]
    return keywords


def _read_topic_distributions(root: Path) -> dict[int, list[dict[str, Any]]]:
    path = root / "cluster_topic_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} was not found. Run visualize-cluster-topics first."
        )
    distributions: dict[int, list[dict[str, Any]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"cluster_id", "topic_distribution"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(
                f"{path} must contain columns cluster_id and topic_distribution"
            )
        for row in reader:
            parsed = json.loads(row["topic_distribution"] or "[]")
            distributions[int(row["cluster_id"])] = [
                {
                    "topic_id": int(item["topic_id"]),
                    "share": float(item.get("share") or 0.0),
                    "papers": int(item.get("papers") or 0),
                }
                for item in parsed
            ]
    return distributions


def _papers_by_year(path: Path) -> dict[int, int]:
    if not path.is_file():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    raw = manifest.get("papers_by_year") or {}
    return {int(year): int(count) for year, count in raw.items()}


def _csv_row(
    *,
    level: int,
    group: int,
    keywords: Sequence[str],
    papers: int,
    shape: Mapping[str, Any],
    activity: Sequence[float],
) -> dict[str, Any]:
    half_life = shape["half_life"]
    return {
        "level": level,
        "group": group,
        "keywords": "; ".join(keywords),
        "papers": papers,
        "family": shape["family"],
        "cls": shape["cls"],
        "sub": shape["sub"],
        "trend": round(float(shape["trend"]), 3),
        "curvature": round(float(shape["curvature"]), 3),
        "burstiness": round(float(shape["burstiness"]), 3),
        "shock_size": round(float(shape["shock_size"]), 3),
        "onset": int(shape["onset"]),
        "half_life": "" if not np.isfinite(half_life) else round(float(half_life), 3),
        "dbic_jump": round(float(shape["dbic_jump"]), 3),
        "noise": round(float(shape["noise"]), 3),
        "range": round(float(shape["range"]), 3),
        "peak_year": int(shape["peak_year"]),
        "activity": json.dumps([round(float(value), 5) for value in activity]),
    }


def _groups_argument(value: str) -> set[int]:
    groups = {int(part) for part in value.split(",") if part.strip()}
    if not groups:
        raise argparse.ArgumentTypeError("At least one group id is required")
    return groups


if __name__ == "__main__":
    raise SystemExit(main())
