#!/usr/bin/env python
"""Stage 3 driver: estimate phenological transition dates + MC uncertainty.

Reads a JSON config (see configs/transitions.example.json), discovers the
aggregated products from stage-2 under ``input_root``, selects the rolling
window via k-fold CV, extracts amplitude-threshold transition dates, and
quantifies uncertainty via a seeded Monte Carlo noise injection.

All randomness flows from a single master ``random_seed`` in the config.
"""

# Stage 3

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from swissphenocam.config import ModellingConfig, load_modelling_config
from swissphenocam.timeseries.aggregation import compute_signal_to_noise, get_n_cycles
from swissphenocam.timeseries.io import read_aggregated_product
from swissphenocam.transitions.smoothing import (
    compute_rolling_window,
    cv_moving_average,
)
from swissphenocam.transitions.thresholds import PhenologyTransitionEstimator
from swissphenocam.transitions.uncertainty import compute_mc_uncertainties

logger = logging.getLogger(__name__)


def _discover_products(config: ModellingConfig) -> list[Path]:
    """Return the data directories (parent of one product file per polygon-year)."""
    pattern = f"*-{config.series_index}-{config.series_window}-product.csv"
    return sorted({p.parent for p in config.input_root.rglob(pattern)})


def _process_one(
    data_path: Path,
    config: ModellingConfig,
    seed: np.random.SeedSequence,
    force: bool = False,
) -> str:
    rng = np.random.default_rng(seed)
    rel = data_path.relative_to(config.input_root)

    suffix = f"-{config.series_index}-{config.series_window}-product.csv"
    product_files = list(data_path.glob(f"*{suffix}"))
    if not product_files:
        return f"no-product {rel}"
    name = product_files[0].name.removesuffix(suffix)

    fname = f"{name}-{config.series_index}-{config.series_window}-{config.series_strategy}-transitions.json"
    out_path = config.output_root / rel / fname
    if not force and out_path.exists():
        return f"skip {rel}"

    product = read_aggregated_product(
        data_path, channel=config.series_index, freq=config.series_window, name=name
    )
    if product is None:
        return "missing-product"
    if config.series_strategy not in product.columns:
        return f"missing-strategy {rel}"

    raw_full = product[config.series_strategy]
    raw_valid = raw_full.dropna()
    if len(raw_valid) < config.cv_kfolds:
        return f"too-short {rel}"

    bin_days = int(config.series_window.rstrip("D"))
    max_w = config.cv_window_max if bin_days == 1 else min(config.cv_window_max, 9)
    candidates = list(range(config.cv_window_min, max_w + 1, 2))
    if not candidates:
        logger.warning(
            "No valid CV window candidates for %s: cv_window_min=%d exceeds "
            "max_w=%d for %s products. Reduce cv_window_min or use 1D products.",
            rel, config.cv_window_min, max_w, config.series_window,
        )
        return f"cv-no-candidates {rel}"
    best = cv_moving_average(
        raw_valid.values,
        window_candidates=candidates,
        num_splits=config.cv_kfolds,
        random_state=config.cv_random_state,
        use_rust=config.use_rust,
    )
    if best is None:
        return f"cv-failed {rel}"

    smoothed = compute_rolling_window(raw_full, best, use_rust=config.use_rust)
    noise_std = float(np.sqrt(np.nanmean((raw_full.values - smoothed.values) ** 2)))

    estimator = PhenologyTransitionEstimator(
        thresholds=config.amplitude_thresholds,
        season_bounds=(config.doy_min, config.doy_max),
        prefix=config.series_index[0],
    )

    result = compute_mc_uncertainties(
        estimator,
        smoothed.index,
        smoothed.values,
        noise_std=noise_std,
        n_iterations=config.mc_iterations,
        window_candidates=candidates,
        num_splits=config.cv_kfolds,
        rng=rng,
        use_rust=config.use_rust,
    )
    if result is None:
        return f"no-season-data {rel}"

    snr = compute_signal_to_noise(raw_full, smoothed)
    n_peak = get_n_cycles(smoothed, gua=result["amplitude"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_serialisable = {
        "dates": result["dates"],
        "amplitude": result["amplitude"],
        "baseline": result["baseline"],
        "peak_value": result["peak_value"],
        "uncertainties": result["uncertainties"],
        "window_selected": int(best),
        "noise_std": noise_std,
        "snr": snr,
        "n_peak": n_peak,
    }
    out_path.write_text(json.dumps(result_serialisable, indent=2, default=str))
    return f"done {rel}"


def orchestrator(args: argparse.Namespace) -> None:
    config = load_modelling_config(args.config)

    config.output_root.mkdir(parents=True, exist_ok=True)
    log_path = config.output_root / "transitions.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(file_handler)
    logger.info("Logging to %s", log_path)

    force = args.force
    for combo_config in config.iter_combinations():
        logger.info(
            "Processing %s / %s under %s …",
            combo_config.series_index,
            combo_config.series_window,
            combo_config.input_root,
        )
        data_paths = _discover_products(combo_config)
        logger.info("Found %d polygon-years to process.", len(data_paths))

        ss = np.random.SeedSequence(combo_config.random_seed)
        child_seeds = ss.spawn(len(data_paths))

        desc = f"stage-3 {combo_config.series_index}-{combo_config.series_window}"
        if combo_config.num_workers == 1:
            for data_path, seed in tqdm(
                zip(data_paths, child_seeds), total=len(data_paths), desc=desc
            ):
                result = _process_one(data_path, combo_config, seed, force=force)
                logger.debug("%s", result)
        else:
            with ProcessPoolExecutor(max_workers=combo_config.num_workers) as pool:
                futures = {
                    pool.submit(_process_one, dp, combo_config, seed, force): dp
                    for dp, seed in zip(data_paths, child_seeds)
                }
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc=desc
                ):
                    logger.debug("%s", future.result())

    logger.info("Stage-3 processing complete.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the transition-date JSON config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process units even if the output JSON already exists.",
    )
    args = parser.parse_args()
    orchestrator(args)


if __name__ == "__main__":
    main()
