"""Monte Carlo uncertainty on transition dates.

Gaussian noise (scale = RMSE between smoothed and raw series) is added for
N realizations (default 1000). For each realization the smoothing window is
re-selected via CV and transition dates recomputed. Uncertainty is the
std-dev of transition dates across realizations.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from swissphenocam.transitions.smoothing import compute_rolling_window, cv_moving_average
from swissphenocam.transitions.thresholds import PhenologyTransitionEstimator

logger = logging.getLogger(__name__)


def compute_mc_uncertainties(
    estimator: PhenologyTransitionEstimator,
    dates: np.ndarray | pd.DatetimeIndex,
    gcc_values: np.ndarray,
    noise_std: float,
    n_iterations: int = 1000,
    window_candidates: list[int] = (3, 5, 7, 9, 11, 13, 15, 17, 19, 21),
    num_splits: int = 5,
    use_rust: bool = False,
    rng: np.random.Generator | None = None,
) -> dict | None:
    """Estimate uncertainty in transition dates via Monte Carlo noise injection.

    For each iteration, Gaussian noise of standard deviation *noise_std* is
    added to *gcc_values*, the smoothing window is re-selected by CV, and
    transition dates are recomputed.  The std-dev across iterations is
    appended to the main estimate under the ``"uncertainties"`` key.

    Parameters
    ----------
    estimator:
        Configured :class:`PhenologyTransitionEstimator`.
    dates:
        DOY integers or DatetimeIndex aligned with *gcc_values*.
    gcc_values:
        Smoothed greenness time series (the model column, not the raw).
    noise_std:
        Standard deviation for additive Gaussian noise (typically RMSE
        between the smoothed model and the raw aggregated series).
    n_iterations:
        Number of Monte Carlo realizations.
    window_candidates:
        Window sizes passed to :func:`cv_moving_average` each iteration.
    num_splits:
        KFold splits for CV inside each realization.
    use_rust:
        Delegate inner CV window selection and rolling-mean to the ``_native``
        Rust extension when available. The outer MC loop always runs in Python.
    rng:
        Optional :class:`numpy.random.Generator` for reproducibility.

    Returns
    -------
    dict with ``dates``, ``amplitude``, ``baseline``, ``peak_value``, and
    ``uncertainties`` keys, or ``None`` if the main estimate failed.
    """
    if isinstance(dates, pd.DatetimeIndex):
        doy_dates = dates.dayofyear.to_numpy()
    else:
        doy_dates = np.asarray(dates)

    gcc_values = np.asarray(gcc_values, dtype=float)

    main_results = estimator.calculate_transitions(
        doy_dates, gcc_values, use_rust=use_rust
    )
    if main_results is None:
        return None

    if rng is None:
        logger.warning(
            "compute_mc_uncertainties: rng is None, falling back to global "
            "numpy random state. Results will not be reproducible."
        )

    # MC loop runs in Python; inner kernels (CV, rolling mean, transitions) delegate to Rust when use_rust=True
    all_dates: list[dict] = []
    for _ in range(n_iterations):
        if rng is None:
            noisy = gcc_values + np.random.normal(0, noise_std, size=len(gcc_values))
        else:
            # Noise added to smoothed model, not raw observations (methodological choice)
            noisy = gcc_values + rng.normal(0, noise_std, size=len(gcc_values))

        # Re-select CV window per iteration to avoid conditioning uncertainty on a fixed window
        best_window = cv_moving_average(
            signal=noisy,
            window_candidates=list(window_candidates),
            num_splits=num_splits,
            use_rust=use_rust,
        )
        if best_window is None:
            continue

        smoothed_noisy = compute_rolling_window(
            pd.Series(index=dates, data=noisy),
            best_window,
            use_rust=use_rust,
        )
        result = estimator.calculate_transitions(
            doy_dates, smoothed_noisy.to_numpy(), use_rust=use_rust
        )
        if result is not None:
            all_dates.append(result["dates"])

    if not all_dates:
        logger.warning("compute_mc_uncertainties: all %d realizations failed.", n_iterations)
        main_results["uncertainties"] = {}
        return main_results

    results_df = pd.DataFrame(all_dates)
    main_results["uncertainties"] = {
        f"{k}_std": v for k, v in results_df.std(skipna=True).to_dict().items()
    }
    return main_results
