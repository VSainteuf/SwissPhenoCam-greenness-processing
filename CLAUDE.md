# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Code companion to the ESSD paper *SwissPhenoCam: A country-scale dataset of tree-level phenocam greenness captures species-specific phenological variation along elevation gradients in Switzerland*.

The codebase is being built up incrementally by migrating existing research code into this scaffold. Most modules currently contain only docstrings + `# TODO: migrate from research code` markers.

## Pipeline (four stages, one script each)

1. **Extraction** (`src/swissphenocam/extraction/`, `scripts/extract.py`) — webcam image sequences + polygon ROI JSONs → per-observation GCC/RCC CSVs.
2. **Time-series processing** (`src/swissphenocam/timeseries/`, `scripts/process_timeseries.py`) — snow filter + brightness filter + 1D/3D aggregation with four strategies (mean, p50, p75, p90).
3. **Transitions** (`src/swissphenocam/transitions/`, `scripts/estimate_transitions.py`) — rolling-window smoothing (5-fold CV over odd windows 3–21 d) + amplitude thresholding (10/25/50/75/90% between DOY 60–320) + Monte Carlo uncertainty (1000 realizations).
4. **Aggregation** (`src/swissphenocam/cli/aggregate.py`, `scripts/aggregate_records.py`) — collects per-tree-year transition records into one CSV, applies configurable quality thresholds (SNR, peak count, etc.), and writes a filtering report.

Each stage script takes `--config path/to/config.json`. Example configs live in `configs/*.example.json`.

## Stack

- **Python 3.11** (pinned in `.python-version`).
- **uv** for environment and dependency management.
- **maturin + PyO3** to build the Rust extension `swissphenocam._native` from the `rust/` crate. Hot per-pixel and per-realization loops live there; `rust/src/lib.rs` exports `rolling_mean_centered_rs`, `cv_moving_average_rs`, `calculate_transitions_rs`, `mc_uncertainties_from_noises_rs`, and `compute_chromcoord_rs`.
- **pytest** for tests.

## Commands

```bash
uv sync                         # create venv, install Python deps
uv run maturin develop          # build the Rust extension in-place (editable)
uv run pytest                   # run test suite
uv run pytest tests/test_extraction.py::test_placeholder   # single test
uv run python scripts/extract.py --config configs/extraction.example.json
uv run ruff check src tests scripts
```

`uv run maturin develop` must be re-run after changes to `rust/`.

## Layout conventions

- **`src/` layout** for the Python package; `[tool.maturin] python-source = "src"` in `pyproject.toml` keeps Python in `src/swissphenocam/` and Rust in `rust/` so the two don't collide on `src/`.
- Dataset directory structure (webcam input and greenness output) is documented in `src/swissphenocam/paths.py`. Input layout: `location/sublocation/year/month/day/image`. The greenness dataset mirrors this per tree-year.
- Real data lives under paths gitignored by `.gitignore` (`data/`, `sample_data/`, `bench_runs/`, `transition-dates.jsonl`). `example-files/` contains a small committed sample (a polygon JSON).
- Config files are **JSON** (not YAML/TOML). Schemas will be defined in `src/swissphenocam/config.py`.

## Annotation quality checks and fixes (2026-05)

A full annotation quality audit was run on `phenocam_records.csv` (7,760 rows, 2,503 unique tree UIDs).

**Key concept**: the true per-tree UID is the `iXXXX` prefix in `full_name` (e.g. `i00240_2010` → `i00240`). `annotation_id` identifies one annotation session and can change when a tree is re-digitised in a later year.

**Notebook**: `debug/annotation_checks.ipynb` — runs 9 checks (null fields, tree-UID identity consistency, annotation session consistency, area CV, species/genus coherence, full_name uniqueness, polygon-name mapping, DB cross-reference). Re-run after any data update to verify labels.

**Fix script**: `debug/fix_annotations.py` — reads `debug/fix_plan.csv` and applies corrections to three data sources: `phenocam_records.csv`, metadata JSONs in `full-greenness-rerun-2.7/`, and source polygon files in `phenocam-data/polygon-coords/`. Supports `--dry-run`. Backs up `phenocam_records.csv` before writing.

**Fixes applied**:

| Trees | Field | Old → New | Reason |
|---|---|---|---|
| i00001, i00011 (2018) | species | `Juglans major` → `Juglans regia` | Wrong label in 2018 annotation; DB-confirmed |
| i00156 (2022) | species | `Picea pungens` → `Picea abies` | Error in 2022 re-annotation; DB-confirmed |
| i02150 (2020–2024) | genus | `Gingko` → `Ginkgo` | Typo; species already spelled correctly |
| i02148 (2019–2024) | genus | `Acer` → `Robinia` | Species is `Robinia pseudoacacia`; genus was wrong |

**Known open issues** (not fixed automatically):
- i00509: genus+species changes between annotation sessions (Picea abies 2016–2017 → Quercus robur 2020–2021) — may represent a genuinely reassigned polygon slot; needs manual visual check.
- 345 polygon names (loc+subloc+name) map to >1 tree_uid — pattern of 2020+ re-annotation batches creating new iXXXX IDs without inheriting species labels from the prior session.
- 739 annotation_id+year pairs absent from PhenoCamDatabase — same 2020+ re-annotation batch not yet pushed to DB.
- 651 rows have genus but no species; 1,883 rows have neither — expected for unannotated cameras.

## When adding code from the research codebase

- Place code in the matching stage module (`extraction` / `timeseries` / `transitions`) — don't create new top-level modules without reason.
- The module docstrings describe each file's intended responsibility; keep pasted code aligned with them, or update the docstring if the boundary shifts.
- Push hot per-pixel / per-realization loops into `rust/src/lib.rs` and expose them through `swissphenocam._native`; call them from the thin Python layer (e.g. `extraction/indices.py`).
