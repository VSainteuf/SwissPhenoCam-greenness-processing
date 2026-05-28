"""Detect mid-year sampling-rate jumps and select noon images as a uniform fallback.

Some tree-years switch from ~1 image/day to 3+ images/day mid-season (firmware
change or schedule edit).  Running p90 on such a series produces visible kinks
because the upper quantile drifts upward from the denser sampling alone.  The
fix is to keep only the closest-to-noon image per day for the whole year when
the detector fires.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _doy_mask(idx: pd.DatetimeIndex, lo: int = 60, hi: int = 320) -> np.ndarray:
    doy = idx.dayofyear.to_numpy()
    return (doy >= lo) & (doy <= hi)


def _pettitt_best_t(x: np.ndarray) -> tuple[int, float]:
    n = len(x)
    if n < 10:
        return -1, 0.0
    csum = np.cumsum(x)
    total = csum[-1]
    ks = np.arange(1, n)
    mean_before = csum[ks - 1] / ks
    mean_after = (total - csum[ks - 1]) / (n - ks)
    stat = np.abs(mean_after - mean_before) * np.sqrt(ks * (n - ks) / n)
    j = int(np.argmax(stat))
    return int(ks[j]), float(stat[j])


def detect_strict_resolution_change(series: pd.DataFrame) -> dict:
    """Detect a mid-year step-change from 1 obs/day to 3+ obs/day.

    Parameters
    ----------
    series:
        Per-tree-year DataFrame with a DatetimeIndex (sub-daily), as produced
        by stage-2 after brightness + snow filters.

    Returns
    -------
    dict with keys: detected, best_doy, pettitt_stat, med_before, med_after,
    frac_before_1, frac_after_ge2, sharp, reason.
    """
    year_vals = series.index.year
    modal_year = int(pd.Series(year_vals).mode().iloc[0])
    full_calendar_year = pd.date_range(f"{modal_year}-01-01", f"{modal_year}-12-31", freq="D")
    counts = (
        series.index.normalize()
        .value_counts()
        .reindex(full_calendar_year, fill_value=0)
        .sort_index()
    )

    x = counts.to_numpy()
    nz = x[x > 0]
    min_obs = int(nz.min()) if nz.size > 0 else 0
    max_obs = int(x.max())
    avg_obs = float(x.mean())

    # Thresholds characterise the firmware-change pattern: series starts at 1 image/day then jumps to 3+/day.
    if min_obs != 1 or max_obs < 3 or avg_obs < 0.3:
        return {"detected": False, "best_doy": None, "reason": "pre_filter"}

    mask = _doy_mask(counts.index)
    season_idx = np.where(mask)[0]
    if season_idx.size < 30:
        return {"detected": False, "best_doy": None, "reason": "too_short"}

    x_season = x[season_idx]
    k_rel, stat = _pettitt_best_t(x_season)
    if k_rel <= 0:
        return {"detected": False, "best_doy": None, "reason": "no_split"}

    k = season_idx[k_rel]
    before = x[:k]
    after = x[k:]
    before_nz = before[before > 0]
    after_nz = after[after > 0]
    if before_nz.size < 10 or after_nz.size < 10:
        return {"detected": False, "best_doy": None, "reason": "too_few_obs"}

    med_before = float(np.median(before_nz))
    med_after = float(np.median(after_nz))
    frac_before_1 = float((before_nz == 1).mean())
    frac_after_ge2 = float((after_nz >= 2).mean())

    w = 5  # empirically tuned half-window for the sharpness check around the detected step
    lo = max(k - w, 0)
    hi = min(k + w, len(x))
    window = x[lo:hi]
    sharp = bool(window.size > 0 and window[-1] >= 2 and window[0] <= 1)

    # Empirically derived detection criteria: median jump from 1→3+, ≥80% purity on both sides, and a sharp step.
    detected = bool(
        med_before == 1
        and med_after >= 3
        and frac_before_1 >= 0.8
        and frac_after_ge2 >= 0.8
        and sharp
    )
    return {
        "detected": detected,
        "best_doy": int(counts.index[k].dayofyear),
        "pettitt_stat": stat,
        "med_before": med_before,
        "med_after": med_after,
        "frac_before_1": frac_before_1,
        "frac_after_ge2": frac_after_ge2,
        "sharp": sharp,
        "reason": None,
    }


def select_closest_to_noon(series: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per calendar date — the row closest to 12:00 of that date.

    Ties are broken by the earlier timestamp.  The original sub-daily
    DatetimeIndex is preserved so downstream aggregation behaves as today.

    Parameters
    ----------
    series:
        Per-tree-year DataFrame with a sub-daily DatetimeIndex.

    Returns
    -------
    DataFrame with one row per calendar date, same columns as input.
    """
    dates = series.index.normalize()
    noon = dates + pd.Timedelta(hours=12)
    # abs distance in seconds; Series so groupby idxmin works correctly
    delta_s = pd.Series(
        np.abs((series.index - noon).total_seconds()), index=series.index
    )
    date_s = pd.Series(dates, index=series.index)
    # idxmin picks first occurrence on ties (earlier timestamp wins)
    chosen_idx = delta_s.groupby(date_s).idxmin()
    return series.loc[chosen_idx]
