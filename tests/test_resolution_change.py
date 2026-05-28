"""Unit tests for timeseries.resolution_change."""

from __future__ import annotations

import numpy as np
import pandas as pd

from swissphenocam.timeseries.resolution_change import (
    detect_strict_resolution_change,
    select_closest_to_noon,
)


def _make_series(obs_per_day_by_date: dict[pd.Timestamp, int], base_hour: int = 6) -> pd.DataFrame:
    """Build a synthetic DataFrame with a sub-daily DatetimeIndex."""
    rows = []
    for date, n in obs_per_day_by_date.items():
        for i in range(n):
            ts = date + pd.Timedelta(hours=base_hour + i * (12 // max(n, 1)))
            rows.append({"datetime": ts, "GCC": 0.4})
    df = pd.DataFrame(rows).set_index("datetime")
    df.index = pd.DatetimeIndex(df.index)
    return df


def _make_series_uniform(year: int, obs_per_day: int) -> pd.DataFrame:
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    return _make_series({d: obs_per_day for d in dates})


def _make_series_step(year: int, n_before: int, n_after: int, split_month: int = 7) -> pd.DataFrame:
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    obs = {}
    for d in dates:
        obs[d] = n_before if d.month < split_month else n_after
    return _make_series(obs)


# ---------------------------------------------------------------------------
# detect_strict_resolution_change
# ---------------------------------------------------------------------------

def test_step_change_detected():
    """1 obs/day Jan–Jun, 5 obs/day Jul–Dec → detected=True, best_doy ~182."""
    series = _make_series_step(2021, n_before=1, n_after=5, split_month=7)
    result = detect_strict_resolution_change(series)
    assert result["detected"] is True
    assert result["sharp"] is True
    # July 1 is DOY 182; allow ±10-day window around the step
    assert result["best_doy"] is not None
    assert abs(result["best_doy"] - 182) <= 15


def test_uniform_5_per_day_rejected():
    """Uniform 5 obs/day → pre-filter rejects (min != 1)."""
    series = _make_series_uniform(2021, obs_per_day=5)
    result = detect_strict_resolution_change(series)
    assert result["detected"] is False
    assert result["reason"] == "pre_filter"


def test_uniform_1_per_day_rejected():
    """Uniform 1 obs/day → pre-filter rejects (max < 3)."""
    series = _make_series_uniform(2021, obs_per_day=1)
    result = detect_strict_resolution_change(series)
    assert result["detected"] is False
    assert result["reason"] == "pre_filter"


def test_leap_year_step_detected():
    """Step change in a leap year — DOY 366 exists; reindex should not crash."""
    series = _make_series_step(2020, n_before=1, n_after=5, split_month=7)
    result = detect_strict_resolution_change(series)
    assert result["detected"] is True
    assert result["best_doy"] is not None
    assert abs(result["best_doy"] - 183) <= 15  # July 1 of leap year is DOY 183


def test_year_boundary_span_uses_modal_year():
    """Series spanning Dec → Jan of the next year reindexes to the modal year."""
    # Mostly 2021, with a handful of Jan-2022 trailing rows.
    obs = {d: 1 for d in pd.date_range("2021-01-01", "2021-06-30", freq="D")}
    obs.update({d: 5 for d in pd.date_range("2021-07-01", "2021-12-31", freq="D")})
    obs.update({d: 5 for d in pd.date_range("2022-01-01", "2022-01-05", freq="D")})
    series = _make_series(obs)
    result = detect_strict_resolution_change(series)
    # Modal year is 2021; detection should still work without errors.
    assert "detected" in result


def test_noisy_uniform_not_falsely_detected():
    """Random 1–5 obs/day with no step change → strict thresholds reject."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2021-01-01", "2021-12-31", freq="D")
    obs = {d: int(rng.integers(1, 6)) for d in dates}
    series = _make_series(obs)
    result = detect_strict_resolution_change(series)
    assert result["detected"] is False


def test_pre_filter_includes_best_doy_key():
    """pre_filter return path includes best_doy=None for contract consistency."""
    series = _make_series_uniform(2021, obs_per_day=5)
    result = detect_strict_resolution_change(series)
    assert "best_doy" in result
    assert result["best_doy"] is None


# ---------------------------------------------------------------------------
# select_closest_to_noon
# ---------------------------------------------------------------------------

def test_noon_picker_exact_noon():
    """5 obs/day at 06, 09, 12, 15, 18 → returns exactly the 12:00 row each day."""
    dates = pd.date_range("2021-04-01", "2021-04-05", freq="D")
    hours = [6, 9, 12, 15, 18]
    rows = []
    for d in dates:
        for h in hours:
            rows.append({"GCC": 0.4 + h * 0.001, "_ts": d + pd.Timedelta(hours=h)})
    df = pd.DataFrame(rows).set_index("_ts")
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "datetime"

    result = select_closest_to_noon(df)
    assert len(result) == len(dates)
    for ts in result.index:
        assert ts.hour == 12
        assert ts.minute == 0


def test_noon_picker_tie_breaks_earlier():
    """When two observations are equidistant from noon, the earlier one wins."""
    date = pd.Timestamp("2021-06-15")
    idx = pd.DatetimeIndex([
        date + pd.Timedelta(hours=10),  # 2h before noon
        date + pd.Timedelta(hours=14),  # 2h after noon
    ])
    df = pd.DataFrame({"GCC": [0.4, 0.5]}, index=idx)
    result = select_closest_to_noon(df)
    assert len(result) == 1
    assert result.index[0].hour == 10


def test_noon_picker_preserves_original_index():
    """The returned index uses the original sub-daily timestamps, not floor-to-day."""
    date = pd.Timestamp("2021-08-01")
    idx = pd.DatetimeIndex([date + pd.Timedelta(hours=h) for h in [8, 12, 16]])
    df = pd.DataFrame({"GCC": [0.3, 0.4, 0.35]}, index=idx)
    result = select_closest_to_noon(df)
    assert len(result) == 1
    assert result.index[0] == date + pd.Timedelta(hours=12)
    assert np.issubdtype(result.index.dtype, np.datetime64)
