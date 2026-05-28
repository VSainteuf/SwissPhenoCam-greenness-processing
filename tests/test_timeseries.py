import numpy as np
import pandas as pd
import pytest

from swissphenocam.timeseries.aggregation import (
    aggregate_series,
    compute_signal_to_noise,
    extrema_amplitudes,
    get_n_cycles,
)
from swissphenocam.timeseries.filters import apply_brightness_filter, remove_snowy_days
from swissphenocam.timeseries.io import compute_temporal_sampling_flags
from swissphenocam.transitions.smoothing import compute_rolling_window


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gcc_series():
    """Daily GCC series with a clear seasonal bump."""
    np.random.seed(0)
    dates = pd.date_range("2020-01-01", periods=365, freq="D")
    doy = np.arange(1, 366)
    gcc = 0.3 + 0.1 * np.exp(-((doy - 180) ** 2) / (2 * 40**2))
    return pd.Series(gcc, index=dates, name="GCC_s33")


@pytest.fixture
def raw_df():
    """Sub-daily DataFrame with brightness and GCC columns (4 obs/day, 60 days).

    Timestamps start at 06:00 (not midnight) to match real webcam data.
    """
    np.random.seed(1)
    dates = pd.date_range("2020-04-01 06:00", periods=240, freq="6h")
    brightness = np.random.uniform(30, 700, 240)
    gcc = 0.35 + np.random.normal(0, 0.01, 240)
    return pd.DataFrame({"brightness": brightness, "GCC_s33": gcc}, index=dates)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def test_brightness_filter_removes_out_of_range(raw_df):
    filtered = apply_brightness_filter(raw_df, min_brightness=50, max_brightness=650)
    assert (filtered.brightness >= 50).all()
    assert (filtered.brightness <= 650).all()
    assert len(filtered) < len(raw_df)


def test_brightness_filter_keeps_in_range(raw_df):
    in_range = raw_df[raw_df.brightness.between(50, 650)]
    result = apply_brightness_filter(in_range)
    assert len(result) == len(in_range)


def test_remove_snowy_days_sets_nan(raw_df):
    snow_dates = pd.date_range("2020-04-01", periods=3, freq="D")
    snow_flag = pd.DataFrame({"snow": [1, 1, 0]}, index=snow_dates)
    result = remove_snowy_days(raw_df, snow_flag)
    assert result.loc["2020-04-01":"2020-04-02"].isna().all().all()
    # Day 3 has no snow — none of its rows should be all-NaN
    assert not result.loc["2020-04-03"].isna().all().all()


def test_remove_snowy_days_no_snow(raw_df):
    snow_dates = pd.date_range("2020-04-01", periods=3, freq="D")
    snow_flag = pd.DataFrame({"snow": [0, 0, 0]}, index=snow_dates)
    result = remove_snowy_days(raw_df, snow_flag)
    assert result.isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["avg", "p50", "p75", "p90"])
def test_aggregate_series_daily_all_methods(gcc_series, method):
    result = aggregate_series(gcc_series, freq="1D", method=method)
    assert len(result) == len(gcc_series)
    assert result.notna().all()


def test_aggregate_series_3d_doy_aligned_bins(gcc_series):
    result = aggregate_series(gcc_series, freq="3D", method="p90")
    assert 120 <= len(result) <= 122
    diffs = result.index.to_series().diff().dt.days.dropna()
    assert (diffs == 3).all()


def test_aggregate_series_3d_starts_at_doy2(gcc_series):
    result = aggregate_series(gcc_series, freq="3D", method="p90")
    assert result.index[0].dayofyear == 2


def test_aggregate_series_3d_excludes_doy1():
    dates = pd.date_range("2020-01-01", periods=365, freq="D")
    values = np.zeros(365)
    values[0] = 99.0
    series = pd.Series(values, index=dates)
    result = aggregate_series(series, freq="3D", method="p90")
    assert 99.0 not in result.values


def test_aggregate_series_3d_last_bin(gcc_series):
    result = aggregate_series(gcc_series, freq="3D", method="p90")
    assert result.index[-1].dayofyear == 365


def test_aggregate_series_3d_leap_year():
    dates = pd.date_range("2020-01-01", periods=366, freq="D")
    gcc = 0.35 + np.random.default_rng(0).normal(0, 0.01, 366)
    series = pd.Series(gcc, index=dates)
    result = aggregate_series(series, freq="3D", method="avg")
    last_doy = result.index[-1].dayofyear
    assert last_doy == 365
    assert result.iloc[-1] == result.iloc[-1]  # not NaN


def test_aggregate_series_invalid_method(gcc_series):
    with pytest.raises(ValueError, match="method must be"):
        aggregate_series(gcc_series, freq="1D", method="median")


def test_compute_signal_to_noise_positive(gcc_series):
    smoothed = gcc_series.rolling(11, center=True, min_periods=1).mean()
    snr = compute_signal_to_noise(gcc_series, smoothed)
    assert snr > 0


def test_extrema_amplitudes_columns(gcc_series):
    df = extrema_amplitudes(gcc_series)
    assert set(df.columns) >= {"extrema_index", "extrema_value", "next_extrema_index", "next_extrema_value", "amplitude"}


def test_get_n_cycles_single_peak(gcc_series):
    # gcc_series is a single smooth seasonal bump — should count as 1 cycle.
    gua = gcc_series.max() - gcc_series.min()
    assert get_n_cycles(gcc_series, gua, ratio=0.5) == 1


# ---------------------------------------------------------------------------
# IO — temporal sampling flags
# ---------------------------------------------------------------------------


def test_temporal_sampling_flags_keys(gcc_series):
    flags = compute_temporal_sampling_flags(gcc_series.to_frame())
    for key in ("start_date", "end_date", "avg_obs_per_day", "n_gaps",
                "spring_coverage_ratio", "summer_coverage_ratio", "autumn_coverage_ratio"):
        assert key in flags


def test_temporal_sampling_flags_no_gaps(gcc_series):
    flags = compute_temporal_sampling_flags(gcc_series.to_frame())
    assert flags["n_gaps"] == 0


def test_temporal_sampling_flags_detects_gap():
    dates = pd.date_range("2020-01-01", periods=30).tolist()
    # Remove Jan 16-18 → consecutive observations jump from Jan 15 to Jan 19 (4-day interval).
    gapped = dates[:15] + dates[18:]
    df = pd.DataFrame({"gcc": 0.35}, index=pd.DatetimeIndex(gapped))
    flags = compute_temporal_sampling_flags(df)
    assert flags["n_gaps"] == 1
    assert flags["gap_lengths_day"][0] == 4  # interval = 19-15 = 4


def test_temporal_sampling_flags_max_consecutive_missing_keys(gcc_series):
    flags = compute_temporal_sampling_flags(gcc_series.to_frame())
    for season in ("spring", "summer", "autumn"):
        assert f"{season}_max_consecutive_missing_days" in flags


def test_temporal_sampling_flags_max_consecutive_missing_correct():
    # Spring window DOY 60-160; create a 7-day gap at DOY 90-96 (missing).
    year = 2020
    doys = list(range(60, 90)) + list(range(97, 161))  # skip DOY 90-96
    dates = [pd.Timestamp(year, 1, 1) + pd.Timedelta(days=d - 1) for d in doys]
    df = pd.DataFrame({"gcc": 0.35}, index=pd.DatetimeIndex(dates))
    flags = compute_temporal_sampling_flags(df)
    assert flags["spring_max_consecutive_missing_days"] == 7


# ---------------------------------------------------------------------------
# Smoothing — NaN preservation
# ---------------------------------------------------------------------------


def test_smoothing_preserves_nan_positions(gcc_series):
    series = gcc_series.copy()
    series.iloc[90:95] = np.nan
    smoothed = compute_rolling_window(series, window=5)
    assert smoothed.iloc[90:95].isna().all()
    assert smoothed.iloc[80:85].notna().all()
