# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Code companion to the ESSD paper *SwissPhenoCam: A country-scale dataset of tree-level phenocam greenness captures species-specific phenological variation along elevation gradients in Switzerland*.

## Pipeline (four stages, one script each)

1. **Extraction** (`src/swissphenocam/extraction/`, `scripts/extract.py`) — webcam image sequences + polygon ROI JSONs → per-observation GCC/RCC CSVs.
2. **Time-series processing** (`src/swissphenocam/timeseries/`, `scripts/process_timeseries.py`) — snow filter + brightness filter + 1D/3D aggregation with four strategies (mean, p50, p75, p90).
3. **Transitions** (`src/swissphenocam/transitions/`, `scripts/estimate_transitions.py`) — rolling-window smoothing (5-fold CV over odd windows 3–21 d) + amplitude thresholding (10/25/50/75/90% between DOY 60–320) + Monte Carlo uncertainty (1000 realizations).
4. **Aggregation** (`src/swissphenocam/cli/aggregate.py`, `scripts/aggregate_records.py`) — collects per-tree-year transition records into one CSV, applies configurable quality thresholds (SNR, peak count, etc.), and writes a filtering report.

Each stage script takes `--config path/to/config.json`. Example configs live in `configs/*.example.json`; the stage-4 ones reproduce the full and curated sets of SwissPhenoCam v1.1.

## Stack

- **Python 3.11** (pinned in `.python-version`).
- **uv** for environment and dependency management.
- **maturin + PyO3** to build the Rust extension `swissphenocam._native` from the `rust/` crate. Hot per-pixel and per-realization loops live there; `rust/src/lib.rs` exports `rolling_mean_centered_rs`, `cv_moving_average_rs`, `calculate_transitions_rs`, and `compute_chromcoord_rs`.
- **pytest** for tests.

## Commands

```bash
uv sync                         # create venv, install Python deps
uv run maturin develop          # build the Rust extension in-place (editable)
uv run pytest                   # run test suite
uv run pytest tests/test_aggregate.py   # one test file
uv run python scripts/extract.py --config configs/stage1-extraction.example.json
uv run ruff check src tests scripts
```

`uv run maturin develop` must be re-run after changes to `rust/`.

## Layout conventions

- **`src/` layout** for the Python package; `[tool.maturin] python-source = "src"` in `pyproject.toml` keeps Python in `src/swissphenocam/` and Rust in `rust/` so the two don't collide on `src/`.
- Dataset directory structure (webcam input and greenness output) is documented in `src/swissphenocam/paths.py`. Input layout: `location/sublocation/year/month/day/image`. The greenness dataset mirrors this per tree-year.
- Real data lives under paths gitignored by `.gitignore` (`data/`, `sample_data/`, `bench_runs/`, `transition-dates.jsonl`).
- Config files are **JSON** (not YAML/TOML). Schemas will be defined in `src/swissphenocam/config.py`.

## When adding code

- Place code in the matching stage module (`extraction` / `timeseries` / `transitions`) — don't create new top-level modules without reason.
- The module docstrings describe each file's intended responsibility; keep pasted code aligned with them, or update the docstring if the boundary shifts.
- Push hot per-pixel / per-realization loops into `rust/src/lib.rs` and expose them through `swissphenocam._native`; call them from the thin Python layer (e.g. `extraction/indices.py`).
