# Pipeline stage drivers 🎬

One script per stage. Each takes `--config path/to/config.json` and is the canonical entry point for its stage.

The driver logic lives in `src/swissphenocam/cli/` (importable, installed as console scripts). The files here are thin stubs that call into those modules.

| Script | Console script | Stage | Reads | Writes |
| --- | --- | --- | --- | --- |
| `extract.py` | `swissphenocam-extract` | 1 — Extraction 💚 | webcam images + polygon JSONs | per-polygon GCC/RCC CSVs |
| `process_timeseries.py` | `swissphenocam-process` | 2 — Time-series 📊 | stage-1 CSVs + snow flags | aggregated GCC/RCC products |
| `estimate_transitions.py` | `swissphenocam-transitions` | 3 — Transitions 📈 | one aggregated series per tree-year | `transitions-*.json` with MC uncertainties |
| `aggregate_records.py` | `swissphenocam-aggregate` | 4 — Aggregation 📋 | stage-3 transitions + metadata + sampling flags | filtered wide CSV + filtering report |
| `generate_dashboard.py` | — | utility | pipeline overview CSV | self-contained HTML dashboard with Plotly heatmaps and client-side filter controls |
| `cleanup_old_stage2_names.py` | — | utility | stage-2 output root | removes files written with the pre-fix `<uid>-` naming convention (dry-run by default) |
| `cleanup_old_stage3_names.py` | — | utility | stage-3 output root | removes files written with the pre-fix `transitions-<version>.json` naming convention (dry-run by default) |

## Usage 🚀

After `uv sync && uv run maturin develop` the console scripts are available directly:

```bash
swissphenocam-extract     --config configs/extraction.example.json --workers-per-site-year 4 --site-years-in-parallel 4
swissphenocam-process     --config configs/timeseries.example.json
swissphenocam-transitions --config configs/transitions.example.json
```

Or run the scripts directly with `uv run python`:

```bash
uv run python scripts/extract.py            --config configs/extraction.example.json --workers-per-site-year 4 --site-years-in-parallel 4
uv run python scripts/process_timeseries.py --config configs/timeseries.example.json
uv run python scripts/estimate_transitions.py --config configs/transitions.example.json
```

## Stage-1 modes (`--mode`) 🎯

- `normal` — process site-years whose output folder is absent.
- `check` — re-run only polygons whose per-polygon CSV is missing. Use to resume after a crash or to fill partial outputs.
- `plot` — render polygon overlays on a sample image per site-year and exit. Useful for visually validating polygon / image alignment before a full run.

## Parallelism ⚡

Stage 1 parallelizes per-image decoding and indexing inside each site-year via a `multiprocessing.Pool` initialized with shared, row-pointer-packed polygon masks. It also processes several site-years concurrently via a thread pool, each thread driving its own image-level `Pool`. Set `--workers-per-site-year × --site-years-in-parallel ≲ physical cores`. On HDD-backed archives keep `--site-years-in-parallel 1` to avoid seek thrashing; on NVMe, raising it usually improves end-to-end throughput. Stages 2 and 3 are cheap enough to run single-process.
