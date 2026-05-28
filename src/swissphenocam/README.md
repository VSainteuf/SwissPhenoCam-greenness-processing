# `swissphenocam` — Python package 📦

Library code for the four pipeline stages. The scripts under `scripts/` are thin drivers over this package.

## Layout 📁

```
swissphenocam/
├── config.py          # JSON config dataclasses + validators for all stages
├── paths.py           # dataset directory-structure conventions (canonical reference)
├── _native.*.so       # Rust extension (built by `maturin develop`)
├── cli/               # console-script entry points (swissphenocam-extract/process/transitions/aggregate)
│   ├── extract.py     # Stage 1 driver
│   ├── process.py     # Stage 2 driver
│   ├── transitions.py # Stage 3 driver
│   └── aggregate.py   # Stage 4 driver
├── extraction/        # stage 1
│   ├── dataset.py     # site-year and image discovery on disk
│   ├── polygons.py    # polygon loading, category filter, shrink, area/bbox
│   ├── indices.py     # per-image GCC/RCC extraction, worker init, mask packing
│   ├── io.py          # per-polygon CSV / JSON writers, full-name convention
│   └── plotting.py    # polygon-overlay QC images
├── timeseries/        # stage 2
│   ├── filters.py          # snow + brightness filters
│   ├── aggregation.py      # 1D / 3D × avg / p50 / p75 / p90
│   ├── resolution_change.py # Pettitt-test-based sampling-rate jump detector
│   └── io.py               # stage-2 CSV readers / writers
└── transitions/       # stage 3
    ├── smoothing.py   # centered rolling window + 5-fold CV window selection
    ├── thresholds.py  # amplitude thresholding, peak / baseline extraction
    └── uncertainty.py # Monte Carlo noise + std-dev aggregation
```

`paths.py` is the single source of truth for on-disk structure — consult it before adding files or renaming outputs.

### UID-mapped output layout (optional)

By default, Stage 1 writes per-tree files under `<site>/<year>/<full_name>/`. The `full_name` (e.g. `Tree0012`) is derived from the polygon annotation name, which can change across years when a site is re-annotated. To get a stable cross-year layout, supply a `uid_mapping_path` in the stage config:

```json
{ "uid_mapping_path": "/path/to/unique-id-mapping.json" }
```

The mapping file is a flat JSON object `{ annotation_id: unique_id, ... }`. When provided, all three stages switch to:

```
<site>/
  <unique_id>/
    <year>/
      <unique_id>_raw-cc-data.csv
      <unique_id>_metadata_v0.json
      <unique_id>-GCC-3D-product.csv
      transitions-v0.json
```

Polygons whose `annotation_id` is absent from the mapping are skipped, and their ids are recorded in `<output_root>/missing_uids.txt` (one `site/year/annotation_id` line each).

## Module boundaries ✨

- **`extraction/`** — image → per-observation GCC/RCC for each polygon.
- **`timeseries/`** — raw per-observation → filtered and aggregated daily / 3-day series.
- **`transitions/`** — aggregated series → smoothed series + transition dates + MC uncertainty.

Do not add new top-level modules without reason; the stage boundaries match the paper and the four driver scripts.

## Interop with Rust 🦀

Compute-heavy inner loops live in `rust/src/` and are exposed through `swissphenocam._native`. The Python side imports them directly, e.g. from within `extraction/indices.py`. Keep the Python layer thin: I/O, config, orchestration, and anything involving pathlib stays in Python.

## Testing 🧪

```bash
uv run pytest                  # all
uv run pytest tests/test_extraction.py::test_placeholder  # one
```

Tests live in `tests/` at the repo root, mirroring the package layout by stage.
