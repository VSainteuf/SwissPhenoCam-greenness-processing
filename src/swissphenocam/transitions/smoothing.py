"""Centered-rolling-window smoothing of the 3D-p90 greenness series.

The window size is selected per time series by 5-fold cross-validation over
odd candidate sizes in [3, 21] days, minimising mean MSE on held-out points.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

try:
    from swissphenocam import _native
except ImportError:
    _native = None


def cv_moving_average(
    signal: np.ndarray | pd.Series,
    window_candidates: list[int],
    num_splits: int = 5,
    verbose: bool = False,
    use_rust: bool = False,
    random_state: int = 42,
) -> int | None:
    """Select the best centered moving-average window via K-fold CV.

    Parameters
    ----------
    signal:
        1-D array of measurements.
    window_candidates:
        Odd integer window sizes to evaluate.
    num_splits:
        Number of KFold splits.
    use_rust:
        Delegate to the ``_native`` Rust extension when available.
    random_state:
        Seed threaded into the Python KFold shuffler. The Rust path ignores
        this parameter and uses its own internal seed; pass ``use_rust=False``
        if cross-fold reproducibility is required.
    """
    if use_rust:
        if random_state != 42:
            logger.warning(
                "cv_moving_average: random_state=%d ignored; "
                "Rust path uses fixed seed 42.",
                random_state,
            )
        if _native is None or not hasattr(_native, "cv_moving_average_rs"):
            raise ImportError("_native.cv_moving_average_rs not available")
        best = _native.cv_moving_average_rs(
            np.asarray(signal, dtype=float), window_candidates, num_splits  # float64 coercion required by PyO3 Rust binding
        )
        return int(best) if best is not None else None

    return cv_moving_average_python(
        signal=signal,
        window_candidates=window_candidates,
        num_splits=num_splits,
        verbose=verbose,
        random_state=random_state,
    )


def cv_moving_average_python(
    signal: np.ndarray | pd.Series,
    window_candidates: list[int],
    num_splits: int = 5,
    verbose: bool = False,
    random_state: int = 42,
) -> int | None:
    signal = np.asarray(signal)
    if signal.shape[0] < num_splits:
        logger.warning("Signal length (%d) less than number of splits (%d)", len(signal), num_splits)
        return None

    n = len(signal)
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    best_score = np.inf
    best_window: int | None = None

    for w in window_candidates:
        half = w // 2
        fold_errors = []
        for train_idx, test_idx in kf.split(signal):
            temp = np.full(n, np.nan)
            temp[train_idx] = signal[train_idx]

            smoothed = np.zeros_like(signal)
            for i in range(n):
                left = max(0, i - half)
                right = min(n, i + half + 1)
                window_vals = temp[left:right]
                if np.sum(~np.isnan(window_vals)) > 0:
                    smoothed[i] = np.nanmean(window_vals)
                else:
                    smoothed[i] = np.nan

            y_true = signal[test_idx]
            y_pred = smoothed[test_idx]
            mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
            if mask.sum() > 0:
                fold_errors.append(mean_squared_error(y_true[mask], y_pred[mask]))

        if not fold_errors:
            continue
        mean_err = float(np.mean(fold_errors))
        if verbose:
            logger.debug("Window=%d, CV MSE=%.5f", w, mean_err)
        if mean_err < best_score:
            best_score = mean_err
            best_window = w

    if best_window is None:
        logger.warning("No valid window size found among candidates %s", window_candidates)
    return best_window


def compute_rolling_window(
    series: pd.Series,
    window: int,
    use_rust: bool = False,
) -> pd.Series:
    """Apply a centered rolling mean of size *window* to *series*."""
    nan_mask = series.isna()
    if use_rust:
        if _native is None or not hasattr(_native, "rolling_mean_centered_rs"):
            raise ImportError("_native.rolling_mean_centered_rs not available")
        arr = _native.rolling_mean_centered_rs(series.to_numpy(dtype=np.float64), window=window)  # float64 required by PyO3 Rust binding
        result = pd.Series(index=series.index, data=arr)
    else:
        result = series.rolling(window=window, center=True, min_periods=1).mean()
    result[nan_mask] = np.nan
    return result
