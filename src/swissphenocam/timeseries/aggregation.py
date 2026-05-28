"""Temporal aggregation of filtered per-observation greenness.

Produces 1-day and 3-day products, each with four aggregation strategies:
mean, 50th percentile, 75th percentile, 90th percentile.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _aggregate_doy_bins(
    series: pd.Series,
    bin_days: int,
    method: str,
    bin_start_doy: int = 2,
) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=series.dtype)

    doy = series.index.dayofyear
    mask = doy >= bin_start_doy
    series = series[mask]
    if series.empty:
        return pd.Series(dtype=series.dtype)

    doy = series.index.dayofyear
    bin_label = bin_start_doy + ((doy - bin_start_doy) // bin_days) * bin_days

    if method == "avg":
        result = series.groupby(bin_label).mean()
    else:
        q = int(method[1:]) / 100.0
        result = series.groupby(bin_label).quantile(q)

    year = series.index[0].year
    jan1 = pd.Timestamp(year, 1, 1)
    dates = jan1 + pd.to_timedelta(result.index.values - 1, unit="D")
    return pd.Series(result.values, index=dates)


def aggregate_series(
    series: pd.Series,
    freq: str = "1D",
    method: str = "p90",
) -> pd.Series:
    """Resample *series* to *freq* using *method*.

    For ``freq="1D"`` the standard ``resample`` path is used.
    For multi-day frequencies (e.g. ``"3D"``) observations are binned into
    DOY-aligned bins of width *freq* starting at DOY 2; the result has one
    row per bin (not interpolated back to daily).

    Parameters
    ----------
    series:
        Sub-daily or daily greenness series with a DatetimeIndex.
    freq:
        Resampling frequency (e.g. ``"1D"``, ``"3D"``).
    method:
        One of ``"avg"``, ``"p50"``, ``"p75"``, ``"p90"``.
    """
    if method not in ("avg", "p50", "p75", "p90"):
        raise ValueError(f"method must be one of avg/p50/p75/p90; got {method!r}")

    if freq == "1D":
        if method == "avg":
            return series.resample("1D").mean()
        else:
            q = int(method[1:]) / 100.0
            return series.resample("1D").quantile(q)

    return _aggregate_doy_bins(series, bin_days=int(freq.rstrip("D")), method=method)


def compute_signal_to_noise(series: pd.Series, signal: pd.Series) -> float:
    """Signal-to-noise ratio (dB) between smoothed *signal* and raw *series*."""
    residuals = series - signal
    noise_var = np.nanvar(residuals)
    if noise_var == 0:
        return float("inf")  # perfect reconstruction
    return float(10 * np.log10(np.nanvar(signal) / noise_var))


def extrema_amplitudes(s: pd.Series) -> pd.DataFrame:
    """Detect extrema from derivative sign changes and compute amplitudes between them."""
    d1 = s.diff()
    sign = np.sign(d1)
    sign_change = sign.diff()
    extrema_mask = sign_change != 0
    extrema = s[extrema_mask].dropna()
    # Drop the first extremum: it is a spurious endpoint artifact from the finite-difference derivative.
    extrema = extrema.iloc[1:]

    _empty_cols = ["extrema_index", "extrema_value", "next_extrema_index", "next_extrema_value", "amplitude"]
    if len(extrema) < 2:
        return pd.DataFrame(columns=_empty_cols)

    results = []
    for i in range(len(extrema) - 1):
        idx1 = extrema.index[i]
        idx2 = extrema.index[i + 1]
        val1 = extrema.iloc[i]
        val2 = extrema.iloc[i + 1]
        results.append({
            "extrema_index": idx1,
            "extrema_value": val1,
            "next_extrema_index": idx2,
            "next_extrema_value": val2,
            "amplitude": abs(val2 - val1),
        })
    return pd.DataFrame(results)


def get_n_cycles(series: pd.Series, gua: float, ratio: float = 0.5) -> int:
    """Count seasonal peaks whose upward amplitude exceeds *ratio* * *gua*.

    Parameters
    ----------
    series:
        Daily greenness series.
    gua:
        Global (annual) upward amplitude used as the significance threshold.
    ratio:
        Fraction of *gua* an upward amplitude must exceed to be counted.
        The default 0.5 is a domain convention for phenological cycle counting.
    """
    ext = extrema_amplitudes(series)
    ext = ext[(ext["extrema_value"] - ext["next_extrema_value"]) < 0]
    return int(ext[ext.amplitude > ratio * gua].shape[0])
