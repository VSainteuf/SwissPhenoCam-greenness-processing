#!/usr/bin/env python
"""Stage 2 driver: filter + aggregate raw greenness into 1D/3D products.

Reads a JSON config (see configs/timeseries.example.json), discovers stage-1
per-polygon CSVs under ``input_root``, applies brightness + snow filters, and
writes aggregated GCC/RCC/BCC products (one file per (channel, frequency) pair)
to ``output_root``.
"""

# Stage 2

from __future__ import annotations

import argparse
import collections
import json
import logging
import multiprocessing
from functools import partial
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from swissphenocam.config import ProcessingConfig, load_processing_config
from swissphenocam.timeseries.aggregation import aggregate_series
from swissphenocam.timeseries.resolution_change import (
    detect_strict_resolution_change,
    select_closest_to_noon,
)
from swissphenocam.timeseries.filters import (
    apply_brightness_filter,
    remove_snowy_days,
)
from swissphenocam.timeseries.io import (
    compute_temporal_sampling_flags,
    read_raw_greenness,
    read_snow_flag,
    write_aggregated_product,
    write_sampling_flags,
    write_smoothed_product,
)
from swissphenocam.transitions.smoothing import compute_rolling_window, cv_moving_average

logger = logging.getLogger(__name__)

_BASE_CHANNELS = ("GCC", "RCC", "BCC")


def _resolve_columns(
    suffix: str | None,
    available: pd.Index,
) -> dict[str, str]:
    """Map each base channel to its concrete column name. Hard selection: no fallback."""
    selected: dict[str, str] = {}
    for ch in _BASE_CHANNELS:
        col = f"{ch}_s{suffix}" if suffix else ch
        if col in available:
            selected[ch] = col
    return selected


def _discover_units(input_root: Path) -> list[tuple[str, str, str, str]]:
    """Walk *input_root* and return (location, sublocation, year, full_name) tuples.

    Stage-1 writes ``<location>_<sublocation>/<full_name>/<year>/<full_name>_raw-cc-data.csv``.
    """
    pattern = "*_raw-cc-data.csv"
    units: list[tuple[str, str, str, str]] = []
    for csv_path in sorted(input_root.rglob(pattern)):
        year_dir = csv_path.parent    # year/
        uid_dir = year_dir.parent     # uid/
        site_dir = uid_dir.parent     # site_sub/
        identifier = uid_dir.name
        year = year_dir.name
        site = site_dir.name
        if "_" not in site:
            continue
        location, sublocation = site.split("_", 1)
        units.append((location, sublocation, year, identifier))
    return units


def _output_folder(config: ProcessingConfig, location: str, sublocation: str, year: str, full_name: str) -> Path:
    site = f"{location}_{sublocation}"
    return config.output_root / site / full_name / year


def _all_outputs_exist(
    out_folder: Path,
    full_name: str,
    windows: list[str],
    noon_only_gate: bool,
    shrink_suffix: str | None,
) -> bool:
    flags_path = out_folder / f"{full_name}-sampling-flags.json"
    if not flags_path.exists():
        return False
    # Auto-force when the noon-only gate differs from the prior run, otherwise
    # toggling the config silently reuses stale aggregated products.
    try:
        prior_flags = json.loads(flags_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if prior_flags.get("noon_only_gate_config") != noon_only_gate:
        return False
    if prior_flags.get("shrink_suffix") != shrink_suffix:
        return False
    for channel in ("GCC", "RCC", "BCC"):
        for freq in windows:
            if not (out_folder / f"{full_name}-{channel}-{freq}-product.csv").exists():
                return False
            if not (out_folder / f"{full_name}-{channel}-{freq}-smoothed.csv").exists():
                return False
    return True


def _process_one(
    unit: tuple[str, str, str, str],
    config_dict: dict,
    force: bool = False,
) -> str:
    """Worker entry point. Config is passed as a dict for pickling simplicity."""
    try:
        config = ProcessingConfig(**config_dict)

        location, sublocation, year, full_name = unit
        out_folder = _output_folder(config, location, sublocation, year, full_name)
        file_prefix = f"{full_name}_{year}"

        if not force and _all_outputs_exist(
            out_folder,
            file_prefix,
            config.aggregation_windows,
            config.noon_only_on_resolution_change,
            config.shrink_suffix,
        ):
            return f"skip {location}/{sublocation}/{year}/{full_name}"

        raw = read_raw_greenness(
            config.input_root, location, sublocation, year, full_name
        )

        brightness_col = f"brightness_s{config.shrink_suffix}" if config.shrink_suffix else "brightness"
        if brightness_col not in raw.columns:
            logger.warning(
                "%s/%s/%s/%s: expected brightness column %r not found; available: %s",
                location, sublocation, year, full_name,
                brightness_col, list(raw.columns),
            )
            return f"no-brightness-col {full_name}"

        filtered = apply_brightness_filter(
            raw, min_brightness=config.brightness_min, max_brightness=config.brightness_max,
            brightness_col=brightness_col,
        )
        if len(filtered) == 0:
            logger.warning(
                "%s/%s/%s/%s: all %d rows removed by brightness filter [%.0f, %.0f]",
                location, sublocation, year, full_name,
                len(raw), config.brightness_min, config.brightness_max,
            )
            return f"empty-after-brightness {full_name}"

        if config.snow_flags_root is not None:
            snow = read_snow_flag(config.snow_flags_root, location, year)
            if snow is not None:
                filtered = remove_snowy_days(filtered, snow)

        det = detect_strict_resolution_change(filtered)
        if config.noon_only_on_resolution_change and det["detected"]:
            filtered = select_closest_to_noon(filtered)
            noon_only_applied = True
        else:
            noon_only_applied = False

        column_selected = _resolve_columns(config.shrink_suffix, filtered.columns)
        if not column_selected:
            logger.warning(
                "%s/%s/%s/%s: no matching channels for shrink_suffix=%r; available columns: %s",
                location, sublocation, year, full_name,
                config.shrink_suffix, list(filtered.columns),
            )
            return f"no-matching-channels {full_name}"

        flags = compute_temporal_sampling_flags(filtered)
        flags.update({
            "resolution_change_detected": det["detected"],
            "noon_only_applied": noon_only_applied,
            "resolution_change_doy": det.get("best_doy"),
            "noon_only_gate_config": config.noon_only_on_resolution_change,
            "shrink_suffix": config.shrink_suffix,
            "column_selected": column_selected,
        })
        write_sampling_flags(out_folder, file_prefix, flags)

        for channel, col in column_selected.items():
            series = filtered[col].dropna()
            if series.empty:
                logger.debug("Channel %s empty after filtering for %s; skipping", channel, full_name)
                continue
            for freq in config.aggregation_windows:
                bin_days = int(freq.rstrip("D"))
                max_window = 21 if bin_days == 1 else 9
                valid_windows = [w for w in config.smoothing_window_candidates if w <= max_window]
                agg_cols = {
                    method: aggregate_series(series, freq=freq, method=method)
                    for method in config.aggregation_strategies
                }
                write_aggregated_product(out_folder, file_prefix, channel, freq, pd.DataFrame(agg_cols))

                smooth_cols = {}
                for method, agg in agg_cols.items():
                    best_window = cv_moving_average(
                        agg.dropna(),
                        window_candidates=valid_windows,
                        num_splits=config.smoothing_cv_kfolds,
                        use_rust=config.smoothing_use_rust,
                    )
                    if best_window is not None:
                        smooth_cols[method] = compute_rolling_window(
                            agg, best_window, use_rust=config.smoothing_use_rust
                        )
                if smooth_cols:
                    write_smoothed_product(out_folder, file_prefix, channel, freq, pd.DataFrame(smooth_cols))

        return f"done {full_name}"
    except Exception:
        import traceback
        return f"error {unit}: {traceback.format_exc()}"


def _config_to_kwargs(config: ProcessingConfig) -> dict:
    return {
        "input_root": config.input_root,
        "output_root": config.output_root,
        "num_workers": config.num_workers,
        "snow_flags_root": config.snow_flags_root,
        "brightness_min": config.brightness_min,
        "brightness_max": config.brightness_max,
        "aggregation_windows": list(config.aggregation_windows),
        "aggregation_strategies": list(config.aggregation_strategies),
        "smoothing_window_candidates": list(config.smoothing_window_candidates),
        "smoothing_cv_kfolds": config.smoothing_cv_kfolds,
        "smoothing_use_rust": config.smoothing_use_rust,
        "noon_only_on_resolution_change": config.noon_only_on_resolution_change,
        "shrink_suffix": config.shrink_suffix,
    }


def orchestrator(args: argparse.Namespace) -> None:
    config = load_processing_config(args.config)

    config.output_root.mkdir(parents=True, exist_ok=True)
    log_path = config.output_root / "process.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(file_handler)
    logger.info("Logging to %s", log_path)

    logger.info("Discovering stage-1 outputs under %s …", config.input_root)
    units = _discover_units(config.input_root)
    logger.info("Found %d polygon-years to process.", len(units))

    worker = partial(_process_one, config_dict=_config_to_kwargs(config), force=args.force)

    counts: dict[str, int] = collections.Counter()
    if config.num_workers <= 1:
        for unit in tqdm(units, desc="stage-2"):
            result = worker(unit)
            tag = result.split()[0] if result else "unknown"
            counts[tag] += 1
            if tag in ("done", "skip"):
                logger.debug("%s", result)
            else:
                logger.warning("%s", result)
    else:
        with multiprocessing.Pool(processes=config.num_workers) as pool:
            for result in tqdm(
                pool.imap_unordered(worker, units), total=len(units), desc="stage-2"
            ):
                tag = result.split()[0] if result else "unknown"
                counts[tag] += 1
                if tag in ("done", "skip"):
                    logger.debug("%s", result)
                else:
                    logger.warning("%s", result)

    logger.info("Stage-2 processing complete. Summary: %s", dict(counts))


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
        help="Path to the time-series processing JSON config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process units even if all output files already exist.",
    )
    args = parser.parse_args()
    orchestrator(args)


if __name__ == "__main__":
    main()
