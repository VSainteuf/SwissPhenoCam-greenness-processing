# Configs ⚙️

JSON configs for the four pipeline stages. Each `*.example.json` is a template — copy, edit the paths, and pass with `--config`.

| File | Consumed by | Dataclass |
| --- | --- | --- |
| `extraction.example.json` | `swissphenocam-extract` / `scripts/extract.py` | `ExtractionConfig` |
| `timeseries.example.json` | `swissphenocam-process` / `scripts/process_timeseries.py` | `ProcessingConfig` |
| `transitions.example.json` | `swissphenocam-transitions` / `scripts/estimate_transitions.py` | `ModellingConfig` |
| `aggregate.example.json` | `swissphenocam-aggregate` / `scripts/aggregate_records.py` | `AggregateConfig` |

All dataclasses + validators live in `src/swissphenocam/config.py`. Invalid configs raise `ValueError` on load with a message pointing at the offending field.

## Stage 1 — extraction 💚

```json
{
  "image_dataset_root":  "/path/to/webcam-archive",
  "output_root":         "/path/to/greenness-dataset",
  "polygon_files_root":  "/path/to/polygon-coords",
  "polygon_config_path": "/path/to/annotation_config_v5.json",

  "categories_kept":     ["individual tree"],
  "shrink_ratios":       [33],
  "plot_dir":            null,
  "camera_hashes_path":  null
}
```

- `shrink_ratios` — one or more percent-area shrinks to apply to each polygon. The paper uses `[33]`, producing `GCC_s33` / `RCC_s33` columns. Each ratio must be in `(0, 100)` exclusive.
- `categories_kept` — polygon categories to process. Default keeps only individual trees.
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
  "num_workers":             1
}
```

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

```json
{
  "input_root":  "/path/to/greenness-dataset",
  "output_path": "/path/to/phenology_records_filtered.csv",
  "min_thresholds": {
    "GCC_3D_p90-snr": 15.0
  },
  "max_thresholds": {
    "GCC_3D_p90-n_peak": 1,
    "spring_max_consecutive_missing_days": 14.0,
    "autumn_max_consecutive_missing_days": 21.0,
    "GCC_3D_p90-GU_50-std": 7.0,
    "GCC_3D_p90-GD_25-std": 7.0
  }
}
```

- `input_root` — greenness dataset root containing stage-1/2/3 outputs.
- `output_path` — destination CSV (one row per accepted tree-year).
- `min_thresholds` / `max_thresholds` — dicts mapping field names to numeric bounds. Bare suffix keys (e.g. `"snr"`) match all columns ending in that suffix across every product found for the tree-year. Tree-years missing a required field are excluded. A plain-text filtering report is written alongside the CSV.
