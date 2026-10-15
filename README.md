# SwissPhenoCam Greenness Processing Pipeline

> Tree-level phenocam greenness extraction, time-series processing, phenological transition-date estimation, and record aggregation for the **SwissPhenoCam** dataset.

Code companion to the ESSD paper
*SwissPhenoCam: A country-scale dataset of tree-level phenocam greenness captures species-specific phenological variation along elevation gradients in Switzerland.*

This repository turns a country-scale archive of raw webcam images plus hand-drawn tree-crown polygons into analysis-ready greenness time series, phenological transition dates with uncertainty, and a quality-filtered aggregated dataset.

---

## Pipeline at a glance 🌱

```
 📷 webcam images + ROI polygons
            │
            ▼
  ┌───────────────────────┐
  │ 1. Greenness          │   scripts/extract.py            (swissphenocam-extract)
  │    extraction 💚      │   → per-polygon raw CC CSVs (one row per image)
  └───────────────────────┘
            │
            ▼
  ┌───────────────────────┐
  │ 2. Time-series        │   scripts/process_timeseries.py (swissphenocam-process)
  │    processing 📊      │   → snow + brightness filter, 1D/3D products × mean/p50/p75/p90
  └───────────────────────┘
            │
            ▼
  ┌───────────────────────┐
  │ 3. Transition-date    │   scripts/estimate_transitions.py (swissphenocam-transitions)
  │    estimation 📈      │   → smoothed series, amplitude thresholds, MC uncertainty
  └───────────────────────┘
            │
            ▼
  ┌───────────────────────┐
  │ 4. Record             │   scripts/aggregate_records.py    (swissphenocam-aggregate)
  │    aggregation 📋     │   → quality-filtered wide CSV + filtering report
  └───────────────────────┘
            │
            ▼
  🎯 analysis-ready phenology dataset
```

Each stage writes into the same *greenness dataset* directory tree (see [Data layout](#data-layout)), so re-running a stage only touches files relevant to that stage.

---

## Quick start 🚀

### Requirements

- 🐍 **Python 3.11** (pinned in `.python-version`).
- 📦 **[uv](https://docs.astral.sh/uv/)** for environment and dependency management.
- 🦀 **Rust toolchain** (stable) for building the PyO3 extension via `maturin`.


### Installing prerequisites

```bash
# uv (Python environment manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# C compiler / linker (required by maturin / PyO3)
# Linux (Debian/Ubuntu):
sudo apt install build-essential
# macOS:
xcode-select --install

# Rust toolchain (stable)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Run code

```bash
# 1. Environment (creates .venv/, installs deps)
uv sync

# 2. Build the Rust extension in editable mode
uv run maturin develop

# 3. Sanity-check
uv run pytest

# 4. Run the stages
swissphenocam-extract     --config configs/stage1-extraction.example.json --workers-per-site-year 4 --site-years-in-parallel 4
swissphenocam-process     --config configs/stage2-timeseries.example.json
swissphenocam-transitions --config configs/stage3-transitions.example.json
swissphenocam-aggregate   --config configs/stage4-aggregate-curated.example.json
```

Or invoke via `uv run python scripts/<script>.py --config configs/<stage config>.example.json` if you prefer to run from source directly.

> Re-run `uv run maturin develop` after any change under `rust/`.


---

## Stage details ✨

### 1. Greenness extraction 💚

▶️ **In:** webcam images + polygon JSONs.

▶️ **Out:** one CSV per polygon-year with one row per image observation, plus a JSON of polygon metadata and QC stats.

- **Shrunk ROI.** Manual polygons are rough and often leak into sky or neighboring crowns. To stay within the target crown, greenness is extracted on a **33%-area-shrunk version** of each polygon (boundary-offset algorithm that preserves shape). Canonical columns are `GCC_s33` / `RCC_s33` / `BCC_s33`. The original polygon is kept only for metadata (area, width, height). The shrink ratio is configurable; do **not** remove the shrinking.
- **Category filter.** By default only `category == "individual tree"` is processed (`categories_kept`).
- **Modes** (`--mode`):
  - `normal` — process site-years whose output folder is absent.
  - `check` — re-run only polygons whose per-polygon CSV is missing (resume-friendly).
  - `plot` — render polygon overlays on a sample image and exit (QC sanity-check).
- Parallelism: `--workers-per-site-year N` fans out per-image work inside each site-year via `multiprocessing`; `--site-years-in-parallel M` processes M site-years concurrently via a thread pool. Total processes ≈ N×M.

### 2. Time-series processing 📈

▶️ **In:** stage-1 per-observation CSVs + daily snow-flag CSVs (one per site-year).

▶️ **Out:** six product CSVs per polygon-year — `(GCC|RCC|BCC) × (1D|3D)` — each containing all four aggregation strategies side-by-side.

- Snow filter: drops observations on days flagged as snowy (`snow == 1`) in `<snow_flags_root>/<location>/<location>-<year>-daily_snow_flag.csv`, the layout of `auxiliary-data/snow-index/` in the published dataset. A missing file or a missing day counts as snow-free, so the snow index must cover every observed day.
- Brightness filter: drops observations whose ROI brightness falls outside `[min, max]`.
- Aggregation: 1-day and 3-day windows × `{avg, p50, p75, p90}`.
- Location exclusion: polygon-years of the location IDs in `exclude_locations` are skipped.

### 3. Transition-date estimation 🗓️

▶️ **In:** one aggregated series per tree-year (configurable; paper default = `GCC, 3D, p90`).

▶️ **Out:** `<name>-<index>-<window>-<strategy>-transitions.json` with central dates and Monte Carlo uncertainties.

- **Smoothing.** Centered rolling-window mean; window size selected by 5-fold CV over odd window sizes `3–21 d` (MSE on held-out 20% per fold).
- **Amplitude thresholding.** Restrict to DOY `60–320`, extract peak `(t_peak, g_peak)` and baseline `g_min`, then report the first/last crossings of `{10, 25, 50, 75, 90}%` of the amplitude (green-up and green-down).
- **Uncertainty.** Monte Carlo with Gaussian noise (σ estimated from the residual RMSE between smoothed and raw series); **1000 realizations** by default; CV re-run per realization. Reported uncertainty is the per-transition standard deviation across realizations.

### 4. Record aggregation 💿

▶️ **In:** stage-3 `*-transitions.json` files + metadata JSONs + sampling-flags JSONs across the entire dataset.

▶️ **Out:** a single wide CSV (one row per accepted tree-year) + a plain-text filtering report.

- **Collection.** Walks the dataset root, discovers all `*-sampling-flags.json` files, and joins each tree-year's metadata, sampling flags, and transition metrics into a flat row.
- **Quality thresholds.** Configurable min/max thresholds applied to any column (e.g. `GCC_3D_p90-snr >= 6`, `GCC_3D_p90-n_peak <= 1`, `spring_max_consecutive_missing_days <= 14`). Wildcard keys (e.g. bare `snr`) match all columns ending in that suffix. A tree-year whose value is missing fails the threshold, unless the key is listed in `nan_ok_keys`. A NaN value passes.
- **Per-season filters.** Thresholds on spring keys (`spring_*`, green-up dates such as `-GU_50`) and autumn keys (`autumn_*`, green-down dates such as `-GD_25`) apply to one season. With `allow_partial: true` (default), a failing season's transition columns are set to NaN and the tree-year is kept if the other season passes. With `allow_partial: false`, any failing season excludes the tree-year.
- **Required values.** Columns listed in `required_keys` must have a value; a tree-year with a missing or NaN value is excluded.
- **Exclusions.** `exclude_locations` drops all tree-years of the listed location IDs before counting. `exclude_csv_path` drops listed `(uid, year)` pairs (the published sets use it for tree-years that failed visual checks).
- **Full and curated sets.** The published dataset has two record files built from the same stage-3 outputs: `phenology_records_full.csv` and `phenology_records_curated.csv`. Their settings are in `configs/stage4-aggregate-full.example.json` and `configs/stage4-aggregate-curated.example.json`.
- **Filtering report.** Per-threshold exclusion counts (independent impact of each filter) plus the combined pass/fail count, written next to the CSV as `<output name>-filtering-report.txt`.

---

## Paper figures 📊

`notebooks/analysis.ipynb` makes all figures and tables of the paper from the published SwissPhenoCam dataset. It does not need the pipeline or the Rust extension, only the downloaded dataset.

```bash
uv sync                                    # installs the dev group (jupyter, matplotlib, ...)
uv run jupyter lab notebooks/analysis.ipynb
```

Set `DATA_ROOT` in the first code cell to the folder of the downloaded dataset (the one holding `phenology_records_curated.csv`). A full run writes the figures (PDF) and a species colour map (`species_colormap.json`) to `notebooks/figures/`, and prints the LaTeX tables. Figures for the 6 most frequent species are for the main text; the files with the `_supp` suffix show the next 8 species.

---

## Configuration ⚙️

All four stages take a single `--config path/to/config.json`. Start from `configs/*.example.json` and edit paths. Config dataclasses and validators are in `src/swissphenocam/config.py` — invalid configs fail fast with a clear `ValueError`.

Minimal stage-1 config (see `configs/stage1-extraction.example.json` for the full set):

```json
{
  "image_dataset_root": "/path/to/webcam-archive",
  "output_root":        "/path/to/greenness-dataset",
  "polygon_files_root": "/path/to/polygon-coords",
  "polygon_config_path": "/path/to/annotation_config_v5.json",
  "uid_mapping_path":   "/path/to/uid-mapping.json",
  "categories_kept":    ["individual tree"],
  "shrink_ratios":      [33],
  "plot_dir":           null
}
```



---

## Data layout 🗂️

**Output greenness dataset** (written by stages 1–4)

```
<output_root>/
  <location>_<sublocation>/                            # site directory (e.g. 1150_2)
    <uid>/                                             # tree directory (e.g. i01339)
      <year>/                                          # year directory (e.g. 2022)
        <uid>_<year>_raw-cc-data.csv                   # stage 1, per-polygon observations
        <uid>_<year>_metadata.json                     # stage 1, polygon metadata + QC stats
        <uid>_<year>-GCC-1D-product.csv                # stage 2 ─┐
        <uid>_<year>-GCC-3D-product.csv                #           │ six product CSVs
        <uid>_<year>-RCC-1D-product.csv                #           │ (GCC|RCC|BCC) × (1D|3D)
        <uid>_<year>-RCC-3D-product.csv                #           │
        <uid>_<year>-BCC-1D-product.csv                #           │
        <uid>_<year>-BCC-3D-product.csv                #          ─┘
        <uid>_<year>-GCC-1D-smoothed.csv               # stage 3 ─┐ smoothed series
        <uid>_<year>-GCC-3D-smoothed.csv               #           │ (one per product)
        <uid>_<year>-RCC-1D-smoothed.csv               #           │
        <uid>_<year>-RCC-3D-smoothed.csv               #           │
        <uid>_<year>-BCC-1D-smoothed.csv               #           │
        <uid>_<year>-BCC-3D-smoothed.csv               #          ─┘
        <uid>_<year>-GCC-3D-p90-transitions.json       # stage 3, transition dates + MC uncertainty
        <uid>_<year>-GCC-1D-p90-transitions.json       #   one JSON per (index, agg, strategy)
        <uid>_<year>-RCC-3D-p90-transitions.json       #   ...
        <uid>_<year>-sampling-flags.json               # stage 3, per-tree-year QC flags
  phenology_records_full.csv                           # stage 4, full set
  phenology_records_full-filtering-report.txt          # stage 4, per-threshold exclusion counts
  phenology_records_curated.csv                        # stage 4, curated set
  phenology_records_curated-filtering-report.txt       #   (one CSV + report per stage-4 config)
  extract.log                                          # stage 1 log
  process.log                                          # stage 2 log
  transitions.log                                      # stage 3 log
```

**UID convention.** Each tree receives a stable identifier `iXXXXX` (e.g. `i01339`) derived from the annotation database. This UID drives directory names and file prefixes throughout the dataset.


---

## Repository layout 📁

```
.
├── scripts/                       # thin stage-driver stubs (delegate to swissphenocam.cli.*)
│   ├── extract.py
│   ├── process_timeseries.py
│   ├── estimate_transitions.py
│   └── aggregate_records.py
├── src/swissphenocam/             # Python package
│   ├── cli/                       # console-script entry points (installed by pip/uv)
│   │   ├── extract.py             # → swissphenocam-extract
│   │   ├── process.py             # → swissphenocam-process
│   │   ├── transitions.py         # → swissphenocam-transitions
│   │   └── aggregate.py           # → swissphenocam-aggregate
│   ├── config.py                  # JSON config dataclasses + validators
│   ├── paths.py                   # dataset directory-structure conventions
│   ├── extraction/                # stage 1: dataset, polygons, indices, io, plotting
│   ├── timeseries/                # stage 2: filters, aggregation, io
│   └── transitions/               # stage 3: smoothing, thresholds, uncertainty
├── rust/                          # PyO3 extension (swissphenocam._native)
│   └── src/{lib.rs, extraction.rs, core.rs}
├── configs/*.example.json         # template configs — copy and edit
├── notebooks/analysis.ipynb       # figures and tables of the paper, from the published dataset
├── tests/                         # pytest suite
└── pyproject.toml                 # maturin build + project metadata
```

See per-folder READMEs under `src/swissphenocam/`, `scripts/`, `configs/`, and `rust/` for stage-specific details. 

---

## Citation 📚

If you use this code or the SwissPhenoCam dataset, please cite the ESSD paper:

> Garnot V.S.F., Lever J. J., de Boer M., Spafford L., Vitasse Y., Sigg C., Pietragalla B., Zweifel R., Gessler A., Wegner J. D., *SwissPhenoCam: A country-scale dataset of tree-level phenocam greenness captures species-specific phenological variation along elevation gradients in Switzerland.* , preprint, 2026.

--- 
## Aknowledgements 

This work was carried out as part of the SwissPhenoCam project funded by MeteoSwiss/GCOS ­CH.

---

## License 📄

MIT — see `pyproject.toml`.
