import numpy as np
import pandas as pd
import pytest

from swissphenocam.transitions.smoothing import (
    compute_rolling_window,
    cv_moving_average,
    cv_moving_average_python,
)
from swissphenocam.transitions.thresholds import PhenologyTransitionEstimator
from swissphenocam.transitions.uncertainty import compute_mc_uncertainties

# ---------------------------------------------------------------------------
# Try to import _native; skip Rust tests gracefully if unavailable.
# ---------------------------------------------------------------------------

try:
    from swissphenocam import _native

    _RUST_AVAILABLE = hasattr(_native, "rolling_mean_centered_rs")
except ImportError:
    _native = None
    _RUST_AVAILABLE = False

rust = pytest.mark.skipif(not _RUST_AVAILABLE, reason="_native Rust extension not built")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def seasonal_series():
    """Daily GCC series: gaussian bump peaking at DOY 180."""
    np.random.seed(42)
    doy = np.arange(1, 366, dtype=np.int32)
    gcc = 0.3 + 0.1 * np.exp(-((doy - 180) ** 2) / (2 * 40**2))
    dates = pd.date_range("2020-01-01", periods=365, freq="D")
    return pd.Series(gcc.astype(np.float64), index=dates)


@pytest.fixture
def smoothed_series(seasonal_series):
    best = cv_moving_average_python(seasonal_series.values, [3, 5, 7, 9, 11], num_splits=3)
    return compute_rolling_window(seasonal_series, best)


# ---------------------------------------------------------------------------
# Smoothing — Python
# ---------------------------------------------------------------------------


def test_cv_moving_average_python_returns_odd_window(seasonal_series):
    best = cv_moving_average_python(seasonal_series.values, [3, 5, 7, 9, 11], num_splits=3)
    assert best in [3, 5, 7, 9, 11]


def test_cv_moving_average_python_short_signal():
    assert cv_moving_average_python(np.array([1.0, 2.0]), [3, 5], num_splits=5) is None


def test_compute_rolling_window_shape(seasonal_series):
    result = compute_rolling_window(seasonal_series, 7)
    assert result.shape == seasonal_series.shape
    assert result.notna().all()


def test_compute_rolling_window_smooths(seasonal_series):
    smoothed = compute_rolling_window(seasonal_series, 11)
    # Smoothed std should be lower than raw std
    assert smoothed.std() < seasonal_series.std()


# ---------------------------------------------------------------------------
# Smoothing — Rust
# ---------------------------------------------------------------------------


@rust
def test_cv_moving_average_rust_matches_python(seasonal_series):
    py_best = cv_moving_average_python(seasonal_series.values, [3, 5, 7, 9, 11], num_splits=5)
    rs_best = cv_moving_average(seasonal_series.values, [3, 5, 7, 9, 11], num_splits=5, use_rust=True)
    assert rs_best == py_best


@rust
def test_rolling_mean_rust_close_to_pandas(seasonal_series):
    py_result = compute_rolling_window(seasonal_series, 7, use_rust=False)
    rs_result = compute_rolling_window(seasonal_series, 7, use_rust=True)
    np.testing.assert_allclose(rs_result.values, py_result.values, atol=1e-10)


# ---------------------------------------------------------------------------
# Thresholds — Python
# ---------------------------------------------------------------------------


def test_calculate_transitions_returns_expected_keys(smoothed_series):
    est = PhenologyTransitionEstimator(thresholds=[0.25, 0.50, 0.75], season_bounds=(60, 320))
    result = est.calculate_transitions(smoothed_series.index, smoothed_series.values)
    assert result is not None
    assert "PEAK" in result["dates"]
    assert "GU_50" in result["dates"]
    assert "GD_50" in result["dates"]
    assert result["amplitude"] > 0


def test_calculate_transitions_peak_in_season(smoothed_series):
    est = PhenologyTransitionEstimator(thresholds=[0.50], season_bounds=(60, 320))
    result = est.calculate_transitions(smoothed_series.index, smoothed_series.values)
    assert 60 <= result["dates"]["PEAK"] <= 320


def test_calculate_transitions_gu_before_eos(smoothed_series):
    est = PhenologyTransitionEstimator(thresholds=[0.50], season_bounds=(60, 320))
    result = est.calculate_transitions(smoothed_series.index, smoothed_series.values)
    assert result["dates"]["GU_50"] < result["dates"]["GD_50"]


def test_leap_year_doy_60_is_feb_29():
    """Guard the DOY convention documented in paths.py.

    On a leap year, Feb 29 must map to DOY 60 (the default ``doy_min``).
    The season window then includes Feb 29. On a non-leap year Feb 29 does
    not exist, so DOY 60 falls on Mar 1.
    """
    leap_idx = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    feb29_leap = pd.Timestamp("2020-02-29")
    assert leap_idx.dayofyear[leap_idx == feb29_leap][0] == 60

    nonleap_idx = pd.date_range("2021-01-01", "2021-12-31", freq="D")
    assert nonleap_idx.dayofyear[nonleap_idx == pd.Timestamp("2021-03-01")][0] == 60

    # Transition estimator must include Feb 29 when season starts at DOY 60.
    gcc = np.zeros(len(leap_idx), dtype=float)
    doy = leap_idx.dayofyear.to_numpy()
    gcc[(doy >= 150) & (doy <= 200)] = 0.5  # mid-summer bump
    series = pd.Series(gcc, index=leap_idx)
    est = PhenologyTransitionEstimator(thresholds=[0.50], season_bounds=(60, 320))
    result = est.calculate_transitions(series.index, series.values)
    assert result is not None
    assert result["dates"]["PEAK"] >= 60 and result["dates"]["PEAK"] <= 320


def test_calculate_transitions_empty_season():
    # Series only covers summer; season_bounds targets winter → no data in window.
    dates = pd.date_range("2020-06-01", periods=90, freq="D")  # DOY 153-242
    gcc = np.full(90, 0.35)
    series = pd.Series(gcc, index=dates)
    est = PhenologyTransitionEstimator(thresholds=[0.50], season_bounds=(1, 50))
    result = est.calculate_transitions(series.index, series.values)
    assert result is None


# ---------------------------------------------------------------------------
# Thresholds — Rust
# ---------------------------------------------------------------------------


@rust
def test_calculate_transitions_rust_returns_peak(smoothed_series):
    est = PhenologyTransitionEstimator(thresholds=[0.25, 0.50, 0.75], season_bounds=(60, 320))
    result = est.calculate_transitions(
        smoothed_series.index, smoothed_series.values, use_rust=True
    )
    assert result is not None
    assert "PEAK" in result["dates"]


@rust
def test_calculate_transitions_rust_matches_python(smoothed_series):
    est = PhenologyTransitionEstimator(
        thresholds=[0.10, 0.25, 0.50, 0.75, 0.90], season_bounds=(60, 320)
    )
    py = est.calculate_transitions(smoothed_series.index, smoothed_series.values, use_rust=False)
    rs = est.calculate_transitions(smoothed_series.index, smoothed_series.values, use_rust=True)

    assert py["dates"]["PEAK"] == rs["dates"]["PEAK"]
    for key in py["dates"]:
        if key == "PEAK":
            continue
        py_val = py["dates"][key]
        rs_val = rs["dates"][key]
        if py_val is None:
            assert rs_val is None, f"{key}: Python=None, Rust={rs_val}"
        else:
            assert rs_val is not None, f"{key}: Python={py_val}, Rust=None"
            assert abs(py_val - rs_val) < 1e-6, (
                f"{key}: Python={py_val}, Rust={rs_val}"
            )


@rust
def test_calculate_transitions_rust_matches_python_noisy_rising(smoothed_series):
    """Noisy rising edge creates multiple up-crossings — exercises last-crossing logic.

    A clean gaussian bump has a single up-crossing per threshold, so the first vs
    last choice is invisible. Inject noise around the threshold region on the
    rising edge to force multiple crossings and check Rust still agrees with the
    Python reference (which takes the *last* up-crossing).
    """
    rng = np.random.default_rng(123)
    noisy = smoothed_series.copy()
    rising = (smoothed_series.index.dayofyear >= 100) & (smoothed_series.index.dayofyear <= 160)
    noisy.loc[rising] = noisy.loc[rising] + rng.normal(0, 0.02, size=rising.sum())

    est = PhenologyTransitionEstimator(thresholds=[0.25, 0.50], season_bounds=(60, 320))
    py = est.calculate_transitions(noisy.index, noisy.values, use_rust=False)
    rs = est.calculate_transitions(noisy.index, noisy.values, use_rust=True)

    for key in ("GU_25", "GU_50", "GD_25", "GD_50"):
        py_val = py["dates"][key]
        rs_val = rs["dates"][key]
        assert py_val is not None and rs_val is not None, f"{key} missing"
        assert abs(py_val - rs_val) < 1e-6, (
            f"{key}: Python={py_val}, Rust={rs_val}"
        )


# ---------------------------------------------------------------------------
# Uncertainty — Python
# ---------------------------------------------------------------------------


def test_compute_mc_uncertainties_has_std_keys(smoothed_series):
    est = PhenologyTransitionEstimator(thresholds=[0.50], season_bounds=(60, 320))
    result = compute_mc_uncertainties(
        est,
        smoothed_series.index,
        smoothed_series.values,
        noise_std=0.005,
        n_iterations=20,
        window_candidates=[3, 5, 7],
        num_splits=3,
    )
    assert result is not None
    assert "GU_50_std" in result["uncertainties"]
    assert "GD_50_std" in result["uncertainties"]


def test_compute_mc_uncertainties_reproducible(smoothed_series):
    """Same seed -> identical results. Guards the ESSD reproducibility claim."""
    est = PhenologyTransitionEstimator(
        thresholds=[0.25, 0.50, 0.75], season_bounds=(60, 320)
    )
    kwargs = dict(
        noise_std=0.005,
        n_iterations=40,
        window_candidates=[3, 5, 7],
        num_splits=3,
    )
    r1 = compute_mc_uncertainties(
        est, smoothed_series.index, smoothed_series.values,
        rng=np.random.default_rng(42), **kwargs,
    )
    r2 = compute_mc_uncertainties(
        est, smoothed_series.index, smoothed_series.values,
        rng=np.random.default_rng(42), **kwargs,
    )
    assert r1 is not None and r2 is not None
    for key, v1 in r1["uncertainties"].items():
        v2 = r2["uncertainties"][key]
        if np.isnan(v1) and np.isnan(v2):
            continue
        assert v1 == v2, f"uncertainty mismatch at {key}: {v1} vs {v2}"
    for key, v1 in r1["dates"].items():
        v2 = r2["dates"][key]
        if v1 is None and v2 is None:
            continue
        assert v1 == v2, f"date mismatch at {key}: {v1} vs {v2}"


def test_compute_mc_uncertainties_std_positive(smoothed_series):
    est = PhenologyTransitionEstimator(thresholds=[0.50], season_bounds=(60, 320))
    result = compute_mc_uncertainties(
        est,
        smoothed_series.index,
        smoothed_series.values,
        noise_std=0.005,
        n_iterations=30,
        window_candidates=[3, 5, 7],
        num_splits=3,
    )
    assert result["uncertainties"]["GU_50_std"] > 0
