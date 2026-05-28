"""Observation-level filters applied before aggregation.

- snow filter: drops entire days based on a per-location daily snow flag.
- brightness filter: drops individual observations whose polygon-mean
  brightness falls outside an accepted range.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_brightness_filter(
    series: pd.DataFrame,
    min_brightness: float = 50,
    max_brightness: float = 650,
    brightness_col: str = "brightness",
) -> pd.DataFrame:
    """Drop observations whose polygon-mean brightness is outside [min, max].

    The default bounds (50, 650) are the legacy SwissPhenoCam values; the
    stage-2 driver passes explicit values from :class:`ProcessingConfig`.
    """
    return series[
        series[brightness_col].between(min_brightness, max_brightness, inclusive="both")
    ]


def remove_snowy_days(series: pd.DataFrame, snow_flag: pd.DataFrame) -> pd.DataFrame:
    """Set rows to NaN for days flagged as snowy (snow == 1).

    snow_flag must have a daily DatetimeIndex and a ``"snow"`` column.
    The flag is forward-filled to match the sub-daily index of *series*.
    """
    snow_dates = snow_flag.index[snow_flag["snow"] == 1]
    clean = series.copy(deep=True)
    mask = series.index.normalize().isin(snow_dates)
    if mask.any():
        clean.loc[mask] = np.nan
    return clean
