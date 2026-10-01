# Configs ⚙️

JSON configs for the four pipeline stages. Each `*.example.json` is a template — copy, edit the paths, and pass with `--config`.

| File | Consumed by | Dataclass |
| --- | --- | --- |
| `stage1-extraction.example.json` | `swissphenocam-extract` / `scripts/extract.py` | `ExtractionConfig` |
| `stage2-timeseries.example.json` | `swissphenocam-process` / `scripts/process_timeseries.py` | `ProcessingConfig` |
| `stage3-transitions.example.json` | `swissphenocam-transitions` / `scripts/estimate_transitions.py` | `ModellingConfig` |
| `stage4-aggregate-full.example.json`, `stage4-aggregate-curated.example.json` | `swissphenocam-aggregate` / `scripts/aggregate_records.py` | `AggregateConfig` |

All dataclasses + validators live in `src/swissphenocam/config.py`. Invalid configs raise `ValueError` on load with a message pointing at the offending field.

## Stage 1 — extraction 💚

```json
{
  "image_dataset_root":  "/path/to/webcam-archive",
  "output_root":         "/path/to/greenness-dataset",
  "polygon_files_root":  "/path/to/polygon-coords",
  "polygon_config_path": "/path/to/annotation_config_v5.json",
  "uid_mapping_path":    "/path/to/uid-mapping.json",

  "categories_kept":     ["individual tree"],
  "shrink_ratios":       [33],
  "plot_dir":            null,
  "camera_hashes_path":  null
}
```

- `shrink_ratios` — one or more percent-area shrinks to apply to each polygon. The paper uses `[33]`, producing `GCC_s33` / `RCC_s33` columns. Each ratio must be in `(0, 100)` exclusive.
- `categories_kept` — polygon categories to process. Default keeps only individual trees.
- `uid_mapping_path` — JSON mapping annotation IDs to tree UIDs (`iXXXXX`) (see `uid-mapping.json` at the repository root).
- `camera_hashes_path` — optional JSON mapping readable names → MD5 hashes. Provide when the image archive still uses hash-named location folders.
- `plot_dir` — if set, stage 1 will write one polygon-overlay sanity image per site-year here.
- `output_root` does not need to exist; it will be created.

## Stage 2 — time-series processing 📊

```json
{
  "input_root":              "/path/to/greenness-dataset",
  "output_root":             "/path/to/timeseries-output",
  "snow_flags_root":         "/path/to/snow-flags-dir",
  "brightness_min":          50.0,
  "brightness_max":          650.0,
  "aggregation_windows":     ["1D", "3D"],
  "aggregation_strategies":  ["avg", "p50", "p75", "p90"],
  "num_workers":             1,
  "exclude_locations":       []
}
```

- `snow_flags_root` — folder with one daily snow-flag CSV per site-year, `<location>/<location>-<year>-daily_snow_flag.csv` (the layout of `auxiliary-data/snow-index/` in the published dataset). Days with `snow == 1` are removed. A missing file or day counts as snow-free. `null` turns the snow filter off.
- `exclude_locations` — location IDs (the folder name before `_<sublocation>`) to skip entirely. Default: none.

## Stage 3 — transitions 📈

```json
{
  "input_root":           "/path/to/timeseries-output",
  "output_root":          "/path/to/transitions-output",
  "random_seed":          0,
  "series_index":         "GCC",
  "series_window":        "3D",
  "series_strategy":      "p90",
  "cv_window_min":        3,
  "cv_window_max":        21,
  "cv_kfolds":            5,
  "amplitude_thresholds": [0.10, 0.25, 0.50, 0.75, 0.90],
  "doy_min":              60,
  "doy_max":              320,
  "mc_iterations":        1000,
  "num_workers":          1
}
```

- `series_index` and `series_window` accept a single string or a list (e.g. `["GCC", "RCC"]`); the stage runs the cartesian product of all combinations.
- `cv_window_min` / `cv_window_max` must both be odd integers ≥ 3; the CV grid covers all odd values between them.
- `amplitude_thresholds` — fractions in `(0, 1)` of the seasonal amplitude at which green-up / green-down crossings are reported.

## Stage 4 — aggregation 📋

Two example configs reproduce the published v1.1 record files: `stage4-aggregate-full.example.json` and `stage4-aggregate-curated.example.json`. The curated one:

```json
{
  "input_root":       "/path/to/greenness-dataset",
  "output_path":      "/path/to/greenness-dataset/phenology_records_curated.csv",
  "exclude_csv_path": "/path/to/optional/visual_check_exclusions_curated.csv",
  "min_thresholds": {
    "GCC_3D_p90-snr": 6.0,
    "GCC_3D_p90-season_length": 145
  },
  "max_thresholds": {
    "GCC_3D_p90-n_peak": 1,
    "spring_max_consecutive_missing_days": 14.0,
    "autumn_max_consecutive_missing_days": 14.0,
    "GCC_3D_p90-GU_50-std": 7.0,
    "GCC_3D_p90-GD_25-std": 7.0
  },
  "nan_ok_keys":       ["GCC_3D_p90-season_length"],
  "exclude_locations": [],
  "allow_partial":     false,
  "required_keys":     ["GCC_3D_p90-GU_50", "GCC_3D_p90-GD_25"]
}
```

The full set uses SNR ≥ 0, season length ≥ 100 d, at most 21 consecutive missing days per season, GU₅₀ / GD₂₅ std ≤ 14 d, and the defaults for `allow_partial` and `required_keys`.

- `input_root` — greenness dataset root containing stage-1/2/3 outputs.
- `output_path` — destination CSV (one row per accepted tree-year). A plain-text filtering report is written alongside it as `<output name>-filtering-report.txt`.
- `min_thresholds` / `max_thresholds` — dicts mapping field names to numeric bounds. Bare suffix keys (e.g. `"snr"`) match all columns ending in that suffix across every product found for the tree-year. A missing value fails the threshold; a NaN value passes.
- `nan_ok_keys` — threshold keys for which a missing value is not a failure.
- `exclude_csv_path` — optional CSV of `(uid, year)` pairs to exclude (the published sets use it for tree-years that failed visual checks).
- `exclude_locations` — location IDs whose tree-years are dropped before any counting. Default: none.
- `allow_partial` — how per-season thresholds act (keys starting with `spring_` / `autumn_`, or on green-up / green-down dates such as `-GU_50` / `-GD_25`). `true` (default): a failing season's transition columns are set to NaN and the tree-year is kept if the other season passes. `false`: any failing season excludes the tree-year.
- `required_keys` — columns that must have a value; a tree-year with a missing or NaN value is excluded. Default: none.
