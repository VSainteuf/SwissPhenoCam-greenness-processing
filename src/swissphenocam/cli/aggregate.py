"""Post-pipeline aggregation: collect per-tree-year records into one CSV.

Walks the greenness dataset root, reads metadata, sampling flags, and all
discovered transition-date JSON files for each tree-year, applies quality
thresholds, and writes a wide CSV with one row per accepted tree-year plus a
plain-text filtering report.

Column naming convention:
  - ``_`` within a group (product descriptor, compound metric name)
  - ``-`` between high-level groups (product | metric | std-qualifier)

Examples:
  GCC_3D_p90-PEAK         transition date for PEAK
  GCC_3D_p90-GU_10        green-up date at 10 % amplitude
  GCC_3D_p90-GU_10-std    MC uncertainty (std) for GU_10
  GCC_3D_p90-snr          signal-to-noise ratio of that product

Threshold keys follow the same naming convention, so thresholds can target
either bare sampling-flag fields (e.g. ``spring_coverage_ratio``) or specific
output columns (e.g. ``GCC_3D_p90-snr``, ``GCC_3D_p90-GU_50-std``).  A bare
key that is not an exact column match is treated as a wildcard suffix: it
applies to every column whose name ends with ``-<key>``.  If a threshold key
matches no column for a given tree-year the tree-year is excluded (missing
required field).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from swissphenocam.config import AggregateConfig, load_aggregate_config

logger = logging.getLogger(__name__)

# Fields from a transition JSON that are scalars we want in the output.
_TRANSITION_SCALAR_FIELDS = (
    "amplitude",
    "baseline",
    "peak_value",
    "window_selected",
    "noise_std",
    "snr",
    "n_peak",
)

# Fields in sampling-flags.json that are lists — skip them in the output row.
_LIST_FLAGS = frozenset({"gap_start_dates", "gap_lengths_day"})

# Defaults for sampling-flags keys introduced after some tree-years were already
# processed. Ensures the aggregated table has uniform columns across mixed runs.
_FLAG_DEFAULTS: dict = {
    "resolution_change_detected": False,
    "noon_only_applied": False,
    "resolution_change_doy": None,
    "noon_only_gate_config": False,
}


def _parse_product_prefix(full_name: str, stem: str) -> str | None:
    """Extract ``INDEX_WINDOW_STRATEGY`` prefix from a transition file stem.

    Stem example: ``Tree0012-GCC-3D-p90-transitions``
    Returns: ``GCC_3D_p90``
    """
    suffix = "-transitions"
    if not stem.endswith(suffix):
        return None
    inner = stem[len(full_name) + 1 : -len(suffix)]  # e.g. GCC-3D-p90
    return inner.replace("-", "_")


def _build_row(meta: dict, flags: dict, transition_data: dict[str, dict]) -> dict:
    row: dict = {
        "location": meta.get("location"),
        "sublocation": meta.get("sub_location"),
        "year": meta.get("year"),
        "full_name": meta.get("full_name"),
        "annotation_id": meta.get("annotation_id"),
        "name": meta.get("name"),
        "category": meta.get("category"),
        "genus": meta.get("genus"),
        "species": meta.get("species"),
        "area": meta.get("area"),
        "area_s33": meta.get("area_s33"),
    }

    for k, v in flags.items():
        if k not in _LIST_FLAGS:
            row[k] = v
    for k, default in _FLAG_DEFAULTS.items():
        row.setdefault(k, default)

    for prefix, data in sorted(transition_data.items()):
        for metric, val in data.get("dates", {}).items():
            row[f"{prefix}-{metric}"] = val
        for metric, val in data.get("uncertainties", {}).items():
            # uncertainty keys from JSON are like "GU_10_std" — reformat to "GU_10-std"
            if metric.endswith("_std"):
                base = metric[:-4]  # strip "_std"
                row[f"{prefix}-{base}-std"] = val
            else:
                row[f"{prefix}-{metric}"] = val
        for field in _TRANSITION_SCALAR_FIELDS:
            if field in data:
                row[f"{prefix}-{field}"] = data[field]

    gu = row.get("GCC_3D_p90-GU_50")
    gd = row.get("GCC_3D_p90-GD_25")
    if isinstance(gu, (int, float)) and isinstance(gd, (int, float)):
        row["GCC_3D_p90-season_length"] = gd - gu
    else:
        row["GCC_3D_p90-season_length"] = None

    return row


def _resolve_values(key: str, row: dict) -> list[float | None]:
    """Return the list of numeric values that *key* covers in *row*.

    If *key* is an exact column name, returns a single-element list.
    Otherwise treats *key* as a bare metric name and returns values for every
    column ending with ``-<key>`` (wildcard suffix match).
    Returns an empty list when no column matches (missing field).
    """
    if key in row:
        v = row[key]
        return [float(v) if isinstance(v, (int, float)) else None]
    suffix = f"-{key}"
    matched = [row[k] for k in row if isinstance(k, str) and k.endswith(suffix)]
    return [float(v) if isinstance(v, (int, float)) else None for v in matched]


def _row_fails(
    key: str, bound: float, is_min: bool, row: dict, *, nan_ok: bool = False
) -> bool:
    """Return True when *row* violates the threshold (or the field is missing).

    When *nan_ok* is True, missing or None values are silently skipped instead
    of treated as failures.  This is used for derived metrics like season_length
    where the absence of a value should not exclude the tree-year.
    """
    vals = _resolve_values(key, row)
    if not vals:
        return not nan_ok
    for v in vals:
        if v is None:
            if not nan_ok:
                return True
            continue
        if v < bound if is_min else v > bound:
            return True
    return False


def _classify_threshold(key: str) -> str:
    if key.startswith("spring_") or re.search(r'-[A-Z]U_', key):
        return "spring"
    if key.startswith("autumn_") or re.search(r'-[A-Z]D_', key):
        return "autumn"
    return "whole_year"


def _nan_season_columns(row: dict, season_tag: str) -> None:
    marker = f"-{season_tag}_"
    for key in row:
        if isinstance(key, str) and marker in key:
            row[key] = float("nan")


def _evaluate_filters(
    row: dict,
    min_thresholds: dict,
    max_thresholds: dict,
    nan_ok_keys: frozenset[str] = frozenset(),
) -> tuple[bool, bool, bool]:
    whole_year_ok = True
    spring_ok = True
    autumn_ok = True
    for field, bound in min_thresholds.items():
        if _row_fails(field, bound, is_min=True, row=row, nan_ok=field in nan_ok_keys):
            season = _classify_threshold(field)
            if season == "spring":
                spring_ok = False
            elif season == "autumn":
                autumn_ok = False
            else:
                whole_year_ok = False
    for field, bound in max_thresholds.items():
        if _row_fails(field, bound, is_min=False, row=row, nan_ok=field in nan_ok_keys):
            season = _classify_threshold(field)
            if season == "spring":
                spring_ok = False
            elif season == "autumn":
                autumn_ok = False
            else:
                whole_year_ok = False
    return whole_year_ok, spring_ok, autumn_ok


def _build_report(
    total: int,
    excluded_combined: int,
    excluded_by_csv: int,
    partial_spring_removed: int,
    partial_autumn_removed: int,
    threshold_stats: list[tuple[str, int]],
) -> str:
    fully_accepted = total - excluded_combined - excluded_by_csv - partial_spring_removed - partial_autumn_removed
    partially_accepted = partial_spring_removed + partial_autumn_removed

    def pct(n: int) -> str:
        return f"{n / total * 100:.1f}" if total else "N/A"

    lines = [
        "Filtering Report",
        "================",
        f"Total tree-years found : {total}",
        f"Fully accepted         : {fully_accepted} ({pct(fully_accepted)} %)",
        f"Partially accepted     : {partially_accepted:>3} ({pct(partially_accepted)} %)",
        f"  - Spring removed (GU NaN) : {partial_spring_removed:>3} ({pct(partial_spring_removed)} %)",
        f"  - Autumn removed (GD NaN) : {partial_autumn_removed:>3} ({pct(partial_autumn_removed)} %)",
        f"Excluded (thresholds)  : {excluded_combined:>3} ({pct(excluded_combined)} %)",
        f"Excluded (external CSV): {excluded_by_csv:>3} ({pct(excluded_by_csv)} %)",
        "",
        "Per-filter exclusions (independent counts — a tree-year may appear in multiple):",
        "-" * 78,
        f"  {'Filter':<55}  {'Excluded':>8}  {'%':>6}",
        "-" * 78,
    ]
    for label, count in threshold_stats:
        lines.append(f"  {label:<55}  {count:>8d}  {pct(count):>6} %")
    lines.append("-" * 78)
    return "\n".join(lines) + "\n"


def _load_exclusion_set(path: Path) -> set[tuple[str, int]]:
    """Load a CSV with ``uid`` and ``year`` columns into a set of (uid, year) pairs."""
    df = pd.read_csv(path, usecols=["uid", "year"])
    return set(zip(df["uid"], df["year"].astype(int)))


def _process_dataset(
    cfg: AggregateConfig, verbose: bool
) -> tuple[pd.DataFrame, str]:
    # Load optional exclusion list
    exclude_set: set[tuple[str, int]] | None = None
    if cfg.exclude_csv_path is not None:
        exclude_set = _load_exclusion_set(cfg.exclude_csv_path)
        logger.info("Loaded %d exclusions from %s", len(exclude_set), cfg.exclude_csv_path)

    # Phase 1: collect all rows regardless of thresholds
    all_rows: list[dict] = []
    skipped_no_meta = 0

    for flags_path in tqdm(sorted(cfg.input_root.glob("*/*/*/*-sampling-flags.json"))):
        full_name_dir = flags_path.parent
        full_name = flags_path.name.split("-sampling-flags.json")[0]

        meta_path = full_name_dir / f"{full_name}_metadata.json"
        if not meta_path.exists():
            skipped_no_meta += 1
            if verbose:
                logger.info("SKIP (no metadata): %s", flags_path)
            continue

        meta = json.loads(meta_path.read_text())
        flags = json.loads(flags_path.read_text())

        transition_data: dict[str, dict] = {}
        for tf in sorted(full_name_dir.glob("*-transitions.json")):
            prefix = _parse_product_prefix(full_name, tf.stem)
            if prefix is None:
                continue
            transition_data[prefix] = json.loads(tf.read_text())

        all_rows.append(_build_row(meta, flags, transition_data))

    total = len(all_rows)
    logger.info("%d tree-years found in dataset root.", total)
    if total == 0:
        logger.warning("No tree-years found in %s — check input_root", cfg.input_root)

    # Phase 2: per-threshold exclusion counts (independent)
    nan_ok = frozenset(cfg.nan_ok_keys)
    threshold_stats: list[tuple[str, int]] = []
    for field, bound in cfg.min_thresholds.items():
        season = _classify_threshold(field)
        tag = f" [{season}]" if season != "whole_year" else ""
        count = sum(1 for row in all_rows if _row_fails(field, bound, True, row, nan_ok=field in nan_ok))
        threshold_stats.append((f"min  {field} >= {bound}{tag}", count))
    for field, bound in cfg.max_thresholds.items():
        season = _classify_threshold(field)
        tag = f" [{season}]" if season != "whole_year" else ""
        count = sum(1 for row in all_rows if _row_fails(field, bound, False, row, nan_ok=field in nan_ok))
        threshold_stats.append((f"max  {field} <= {bound}{tag}", count))

    # Phase 3: combined filtering
    accepted: list[dict] = []
    excluded_combined = 0
    excluded_by_csv = 0
    partial_spring_removed = 0
    partial_autumn_removed = 0

    for row in all_rows:
        # Check external exclusion list first
        if exclude_set is not None:
            full_name = row.get("full_name", "")
            uid = full_name.split("_")[0] if full_name else ""
            yr = int(row["year"]) if row.get("year") is not None else 0
            if (uid, yr) in exclude_set:
                excluded_by_csv += 1
                if verbose:
                    logger.info("EXCLUDED (csv): %s / year=%s", full_name, row.get("year"))
                continue

        whole_ok, spring_ok, autumn_ok = _evaluate_filters(
            row, cfg.min_thresholds, cfg.max_thresholds, nan_ok
        )

        if not whole_ok or (not spring_ok and not autumn_ok):
            excluded_combined += 1
            if verbose:
                logger.info("EXCLUDED: %s / year=%s", row.get("full_name"), row.get("year"))
            continue

        if not spring_ok:
            _nan_season_columns(row, "GU")
            partial_spring_removed += 1
            if verbose:
                logger.info("PARTIAL (spring removed): %s / year=%s", row.get("full_name"), row.get("year"))

        if not autumn_ok:
            _nan_season_columns(row, "GD")
            partial_autumn_removed += 1
            if verbose:
                logger.info("PARTIAL (autumn removed): %s / year=%s", row.get("full_name"), row.get("year"))

        accepted.append(row)
        if verbose and spring_ok and autumn_ok:
            logger.info("ACCEPTED: %s / year=%s", row.get("full_name"), row.get("year"))

    if skipped_no_meta:
        logger.warning("Skipped %d tree-years with missing metadata.", skipped_no_meta)
    if excluded_by_csv:
        logger.info("Excluded %d tree-years via external CSV.", excluded_by_csv)
    logger.info("Done: %d accepted (%d partial), %d excluded by thresholds, %d excluded by CSV.",
                len(accepted), partial_spring_removed + partial_autumn_removed,
                excluded_combined, excluded_by_csv)

    report = _build_report(total, excluded_combined, excluded_by_csv,
                           partial_spring_removed, partial_autumn_removed, threshold_stats)
    return pd.DataFrame(accepted), report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-tree-year phenology records into one CSV."
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to JSON config.")
    parser.add_argument("--exclude-csv", type=Path, default=None,
                        help="CSV with uid,year columns of tree-years to exclude.")
    parser.add_argument("--verbose", action="store_true", help="Log one line per tree-year.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Starting aggregation with config: %s", args.config)

    cfg = load_aggregate_config(args.config)
    if args.exclude_csv is not None:
        cfg.exclude_csv_path = args.exclude_csv
    df, report = _process_dataset(cfg, verbose=args.verbose)

    out_dir = cfg.output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(cfg.output_path, index=False)
    logger.info("Written %d rows to %s", len(df), cfg.output_path)

    report_path = out_dir / (cfg.output_path.stem + "-filtering-report.txt")
    report_path.write_text(report)
    logger.info("Filtering report written to %s", report_path)
    logger.info("%s", report)


if __name__ == "__main__":
    main()
