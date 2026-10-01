import csv
import json

import numpy as np
import pytest

from openalex.analysis.cluster_trends import (
    BUMP_DECREASING,
    BUMP_INCREASING,
    DECREASING,
    FIELDNAMES,
    INCREASING,
    SHOCK_LABEL,
    STEP_LABEL,
    TREND_DECREASING,
    TREND_INCREASING,
    BreakFit,
    ClusterSeries,
    assign_chunks,
    break_design,
    break_indices,
    build_parser,
    bump_multiplier,
    default_events_dir,
    laplace_log_evidence,
    load_cluster_series,
    posterior_model_probabilities,
    shock_multiplier,
    signed_labels,
    standardize_time,
    trend_record,
    write_trend_csv,
)
from openalex.cli import COMMANDS


def write_yearly(path, rows, manifest=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["level", "group", "year", "papers", "share"],
        )
        writer.writeheader()
        writer.writerows(rows)
    if manifest is not None:
        events = path.parent.parent / "events"
        events.mkdir(parents=True, exist_ok=True)
        (events / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def fit(parameters, *, log_evidence, break_year, u_max):
    return BreakFit(
        log_evidence=log_evidence,
        break_year=break_year,
        u_max=u_max,
        parameters={name: np.asarray(values, dtype=float) for name, values in parameters.items()},
        divergences=0,
        rhat_max=1.0,
    )


def test_command_is_registered():
    assert COMMANDS["cluster-trends"] == "openalex.analysis.cluster_trends"


def test_load_recovers_trials_and_fills_missing_years(tmp_path):
    clusters = tmp_path / "event_clusters"
    write_yearly(
        clusters / "cluster_by_year.csv",
        [
            {"level": 0, "group": 0, "year": 2020, "papers": 20, "share": 0.2},
            {"level": 0, "group": 0, "year": 2021, "papers": 30, "share": 0.3},
            {"level": 0, "group": 1, "year": 2020, "papers": 10, "share": 0.1},
            {"level": 0, "group": 1, "year": 2022, "papers": 40, "share": 0.4},
            {"level": 1, "group": 0, "year": 2020, "papers": 5, "share": 0.05},
        ],
    )
    series = load_cluster_series(clusters, level=0)
    assert [item.group for item in series] == [0, 1]
    assert series[0].years.tolist() == [2020, 2021, 2022]
    assert series[0].trials.tolist() == [100, 100, 100]
    assert series[0].successes.tolist() == [20, 30, 0]
    assert series[1].successes.tolist() == [10, 0, 40]


def test_manifest_fills_a_year_with_no_cluster_hits(tmp_path):
    clusters = tmp_path / "event_clusters"
    write_yearly(
        clusters / "cluster_by_year.csv",
        [
            {"level": 0, "group": 3, "year": 2020, "papers": 10, "share": 0.1},
            {"level": 0, "group": 3, "year": 2021, "papers": 10, "share": 0.1},
        ],
        manifest={"papers_by_year": {"2019": 80, "2020": 100, "2021": 100}},
    )
    assert default_events_dir(clusters) == tmp_path / "events"
    series = load_cluster_series(clusters, level=0, events_dir=tmp_path / "events")
    assert series[0].years.tolist() == [2019, 2020, 2021]
    assert series[0].successes.tolist() == [0, 10, 10]
    assert series[0].trials.tolist() == [80, 100, 100]


def test_load_rejects_a_short_span_and_inconsistent_denominators(tmp_path):
    clusters = tmp_path / "event_clusters"
    write_yearly(
        clusters / "cluster_by_year.csv",
        [
            {"level": 0, "group": 0, "year": 2020, "papers": 10, "share": 0.1},
            {"level": 0, "group": 0, "year": 2021, "papers": 10, "share": 0.1},
        ],
    )
    with pytest.raises(ValueError, match="three years"):
        load_cluster_series(clusters, level=0)

    write_yearly(
        clusters / "cluster_by_year.csv",
        [
            {"level": 0, "group": 0, "year": 2020, "papers": 10, "share": 0.1},
            {"level": 0, "group": 1, "year": 2020, "papers": 40, "share": 0.2},
            {"level": 0, "group": 0, "year": 2021, "papers": 10, "share": 0.1},
            {"level": 0, "group": 0, "year": 2022, "papers": 10, "share": 0.1},
        ],
    )
    with pytest.raises(ValueError, match="inconsistent corpus sizes"):
        load_cluster_series(clusters, level=0)


def test_standardized_time_has_mean_zero_and_unit_scale():
    standardized, center, scale = standardize_time(np.array([2010, 2012, 2014]))
    assert center == pytest.approx(2012)
    assert standardized.mean() == pytest.approx(0)
    assert standardized.std() == pytest.approx(1)
    assert scale == pytest.approx(np.std([2010, 2012, 2014]))


def curves(**evidences):
    return {
        "trend": fit(
            {"intercept": [1.0, 3.0], "beta": [1.0, 1.0]},
            log_evidence=evidences["trend"],
            break_year=2011,
            u_max=1.5,
        ),
        "step": fit(
            {"intercept": [0.0], "c": [-1.0]},
            log_evidence=evidences["step"],
            break_year=2012,
            u_max=1.0,
        ),
        "shock": fit(
            {"intercept": [0.0, 0.0], "c": [2.0, 2.0], "half_life": [4.0, 4.0]},
            log_evidence=evidences["shock"],
            break_year=2012,
            u_max=1.0,
        ),
        "bump": fit(
            {"intercept": [0.0], "amplitude": [-0.5], "width": [3.0]},
            log_evidence=evidences["bump"],
            break_year=2011,
            u_max=1.0,
        ),
    }


def test_break_design_is_flat_through_the_shift_year():
    time = np.array([-1.5, -0.5, 0.5, 1.5])
    years = np.array([2010, 2011, 2012, 2013])
    design = break_design(time, years, 1)
    assert design.after.tolist() == [0.0, 1.0, 1.0, 1.0]
    assert design.since[0] == 0
    assert design.since[1] == 0
    assert design.since[2] == pytest.approx(1.0)
    assert design.u_max == pytest.approx(design.since[-1])
    assert design.elapsed_years.tolist() == [0.0, 0.0, 1.0, 2.0]
    assert design.centered_years.tolist() == [-1.0, 0.0, 1.0, 2.0]
    assert break_indices(4).tolist() == [1, 2]
    with pytest.raises(ValueError, match="four years"):
        break_indices(3)


def test_shock_halves_each_half_life_and_the_bump_returns_to_the_baseline():
    assert shock_multiplier(0.0, 4.0) == pytest.approx(1.0)
    assert shock_multiplier(4.0, 4.0) == pytest.approx(0.5)
    assert shock_multiplier(8.0, 4.0) == pytest.approx(0.25)
    assert bump_multiplier(0.0, 3.0) == pytest.approx(1.0)
    assert bump_multiplier(3.0, 3.0) == pytest.approx(np.exp(-0.5))
    assert signed_labels(np.array([0.2, -0.1, 0.0]), INCREASING, DECREASING).tolist() == [
        INCREASING,
        DECREASING,
        INCREASING,
    ]


def test_laplace_evidence_and_equal_prior_model_probabilities():
    evidence = laplace_log_evidence(-1.0, np.array([[4.0]]))
    assert evidence == pytest.approx(-1.0 + 0.5 * np.log(2.0 * np.pi) - np.log(2.0))
    first, second = posterior_model_probabilities([0.0, np.log(1.0 / 3.0)])
    assert first == pytest.approx(0.75)
    assert second == pytest.approx(0.25)
    tied = posterior_model_probabilities([-5.0, -5.0, -5.0, -5.0])
    assert tied.tolist() == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_trend_record_keeps_the_trend_on_a_tie_and_saves_every_family():
    years = np.array([2010, 2011, 2012, 2013])
    _standardized, time_mean, time_scale = standardize_time(years)
    series = ClusterSeries(
        level=0,
        group=4,
        years=years,
        successes=np.array([1, 2, 3, 4]),
        trials=np.array([10, 10, 10, 10]),
    )
    row = trend_record(
        series,
        time_mean=time_mean,
        time_scale=time_scale,
        **curves(trend=0.0, step=-10.0, shock=-10.0, bump=-10.0),
    )
    assert row["preferred_model"] == "trend"
    assert row["compatible"] == TREND_INCREASING
    assert row["trend_posterior"] == pytest.approx(1.0 / (1.0 + 3.0 * np.exp(-10.0)))
    assert row["log_bayes_factor"] == pytest.approx(10.0)
    assert row["trend_intercept_mean"] == pytest.approx(2)
    assert row["trend_beta_mean"] == pytest.approx(1)
    assert row["trend_break_year"] == 2011
    assert row["p_upward_shock"] == pytest.approx(1)
    assert row["p_u"] == pytest.approx(1)
    assert set(row) == set(FIELDNAMES)

    tied = trend_record(
        series,
        time_mean=time_mean,
        time_scale=time_scale,
        trend=fit(
            {"intercept": [0.0], "beta": [-0.2]},
            log_evidence=-5.0,
            break_year=2011,
            u_max=1.0,
        ),
        step=fit({"intercept": [0.0], "c": [1.0]}, log_evidence=-5.0, break_year=2012, u_max=1.0),
        shock=fit(
            {"intercept": [0.0], "c": [1.0], "half_life": [4.0]},
            log_evidence=-5.0,
            break_year=2012,
            u_max=1.0,
        ),
        bump=fit(
            {"intercept": [0.0], "amplitude": [1.0], "width": [2.0]},
            log_evidence=-5.0,
            break_year=2011,
            u_max=1.0,
        ),
    )
    assert tied["preferred_model"] == "trend"
    assert tied["compatible"] == TREND_DECREASING
    step = trend_record(
        series,
        time_mean=time_mean,
        time_scale=time_scale,
        **curves(trend=-4.0, step=0.0, shock=-4.0, bump=-4.0),
    )
    assert step["compatible"] == STEP_LABEL
    bump = trend_record(
        series,
        time_mean=time_mean,
        time_scale=time_scale,
        **curves(trend=-4.0, step=-4.0, shock=-4.0, bump=0.0),
    )
    assert bump["compatible"] == BUMP_DECREASING
    rising = fit(
        {"intercept": [0.0], "amplitude": [1.0], "width": [3.0]},
        log_evidence=0.0,
        break_year=2011,
        u_max=1.0,
    )
    assert (
        trend_record(
            series,
            time_mean=time_mean,
            time_scale=time_scale,
            trend=curves(trend=-4.0, step=-4.0, shock=-4.0, bump=-4.0)["trend"],
            step=curves(trend=-4.0, step=-4.0, shock=-4.0, bump=-4.0)["step"],
            shock=curves(trend=-4.0, step=-4.0, shock=-4.0, bump=-4.0)["shock"],
            bump=rising,
        )["compatible"]
        == BUMP_INCREASING
    )
    assert tied["log_bayes_factor"] == pytest.approx(0.0)


def test_write_trend_csv_round_trips_the_header(tmp_path):
    years = np.array([2010, 2011, 2012, 2013])
    _standardized, time_mean, time_scale = standardize_time(years)
    series = ClusterSeries(
        level=0,
        group=1,
        years=years,
        successes=np.array([1, 2, 3, 4]),
        trials=np.array([10, 10, 10, 10]),
    )
    row = trend_record(
        series,
        time_mean=time_mean,
        time_scale=time_scale,
        **curves(trend=-8.0, step=-6.0, shock=-1.0, bump=-4.0),
    )
    assert row["preferred_model"] == "shock"
    assert row["compatible"] == SHOCK_LABEL
    destination = tmp_path / "cluster_trends.csv"
    write_trend_csv(destination, [row])
    with destination.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(FIELDNAMES)
        stored = next(reader)
    assert stored["preferred_model"] == "shock"
    assert stored["compatible"] == SHOCK_LABEL
    assert float(stored["shock_c_mean"]) == pytest.approx(2)
    assert float(stored["shock_half_life_mean"]) == pytest.approx(4)
    assert int(stored["shock_break_year"]) == 2012
    assert not (tmp_path / ".cluster_trends.csv.tmp").exists()


def test_chunks_cover_every_cluster_once():
    chunks = assign_chunks([0, 1, 2, 3, 4], 2)
    assert chunks == [[0, 2, 4], [1, 3]]
    assert build_parser().parse_args(["--groups", "1, 2"]).groups == (1, 2)
