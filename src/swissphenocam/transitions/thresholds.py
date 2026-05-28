"""Amplitude thresholding to extract phenological transition dates.

On the smoothed series restricted to DOY 60-320, identify (gpeak, tpeak) and
baseline gmin, then compute green-up and green-down dates at the 10, 25, 50,
75, and 90% amplitude levels.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from swissphenocam import _native
except ImportError:
    _native = None


class PhenologyTransitionEstimator:
    """Extract transition dates from a smoothed greenness time series.

    Parameters
    ----------
    thresholds:
        Amplitude fractions at which to compute transition dates.
    season_bounds:
        (start_doy, end_doy) window restricting the analysis.
    prefix:
        Single uppercase letter prepended to U_/D_ date keys (e.g. ``G``
        for GCC, ``R`` for RCC, ``B`` for BCC).
    """

    def __init__(
        self,
        thresholds: list[float] = (0.10, 0.25, 0.50, 0.75, 0.90),
        season_bounds: tuple[int, int] = (60, 320),
        prefix: str = "G",
    ) -> None:
        self.thresholds = sorted(thresholds)
        self.season_start, self.season_end = season_bounds
        self.prefix = prefix

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _interpolate_nans(self, series: np.ndarray) -> np.ndarray:
        s = pd.Series(series)
        return s.interpolate(method="linear", limit_direction="both").to_numpy()

    def _find_crossing_date(
        self,
        dates: np.ndarray,
        values: np.ndarray,
        threshold_value: float,
        direction: str = "up",
    ) -> float | None:
        diffs = values - threshold_value
        signs = np.sign(diffs)
        diff_sign = np.diff(signs)

        if len(diff_sign) == 0:
            return None

        if direction == "up":
            crossing_indices = np.where(diff_sign > 0)[0]
            # Last crossing = latest confirmed green-up start
            idx = crossing_indices[-1] if len(crossing_indices) else None
        else:
            crossing_indices = np.where(diff_sign < 0)[0]
            # First crossing = earliest green-down start
            idx = crossing_indices[0] if len(crossing_indices) else None

        if idx is None:
            return None

        x0, x1 = dates[idx], dates[idx + 1]
        y0, y1 = values[idx], values[idx + 1]
        if y1 - y0 == 0:
            # Horizontal segment: threshold is already met at x0.
            return float(x0)
        fraction = (threshold_value - y0) / (y1 - y0)
        try:
            return float(x0 + (x1 - x0) * fraction)
        except TypeError:
            return float(x0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_transitions(
        self,
        dates: np.ndarray | pd.DatetimeIndex,
        gcc_values: np.ndarray,
        use_rust: bool = False,
    ) -> dict | None:
        """Extract phenological transition dates from a smoothed series.

        Parameters
        ----------
        dates:
            DOY integers or DatetimeIndex aligned with *gcc_values*.
        gcc_values:
            Smoothed greenness time series.
        use_rust:
            Delegate to ``_native`` Rust extension when available.

        Returns
        -------
        dict with keys ``dates``, ``amplitude``, ``baseline``,
        ``peak_value``, or ``None`` if the season window is empty.
        """
        if isinstance(dates, pd.DatetimeIndex):
            doy_dates = dates.dayofyear.to_numpy()
        else:
            doy_dates = np.asarray(dates)

        gcc_values = np.asarray(gcc_values, dtype=float)

        if use_rust:
            if _native is None or not hasattr(_native, "calculate_transitions_rs"):
                raise ImportError("_native.calculate_transitions_rs not available")
            return _native.calculate_transitions_rs(
                doy_dates.astype(np.int32),  # i32 cast required by Rust signature
                gcc_values.astype(np.float64),
                self.thresholds,
                self.season_start,
                self.season_end,
                self.prefix,
            )

        values = self._interpolate_nans(gcc_values)

        mask = (doy_dates >= self.season_start) & (doy_dates <= self.season_end)
        values = values[mask]
        doy_dates = doy_dates[mask]

        if len(values) == 0:
            logger.warning("No data within season bounds (doy_min=%d, doy_max=%d)", self.season_start, self.season_end)
            return None

        if np.all(np.isnan(values)):
            logger.warning("All values NaN within season bounds")
            return None

        peak_idx = int(np.argmax(values))
        peak_value = float(values[peak_idx])
        peak_date = doy_dates[peak_idx]
        min_value = float(np.nanmin(values))
        amplitude = peak_value - min_value

        rising_dates = doy_dates[: peak_idx + 1]
        rising_values = values[: peak_idx + 1]
        falling_dates = doy_dates[peak_idx:]
        falling_values = values[peak_idx:]

        results: dict = {
            "dates": {"PEAK": int(peak_date)},
            "amplitude": amplitude,
            "baseline": min_value,
            "peak_value": peak_value,
        }

        for thresh_pct in self.thresholds:
            thresh_val = min_value + thresh_pct * amplitude
            pct_str = int(thresh_pct * 100)

            sos = self._find_crossing_date(
                rising_dates, rising_values, thresh_val, direction="up"
            )
            eos = self._find_crossing_date(
                falling_dates, falling_values, thresh_val, direction="down"
            )
            results["dates"][f"{self.prefix}U_{pct_str}"] = sos
            results["dates"][f"{self.prefix}D_{pct_str}"] = eos

        return results
