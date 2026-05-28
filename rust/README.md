# `swissphenocam._native` — Rust extension 🦀

Hot numerical kernels for the pipeline, exposed to Python via [PyO3](https://pyo3.rs/) and built with [maturin](https://www.maturin.rs/).

## Layout 📁

```
rust/
├── Cargo.toml
└── src/
    ├── lib.rs         # #[pymodule] swissphenocam._native — registers Python-visible fns
    ├── core.rs        # shared numeric primitives (masking, reductions)
    └── extraction.rs  # stage-1 per-pixel GCC/RCC kernels
```

Python sees one module: `swissphenocam._native`. Everything callable from Python is registered in `lib.rs`.

## Build 🔨

Editable install (rebuilds the `.so` in-place under `src/swissphenocam/`):

```bash
uv run maturin develop
```

Release build for benchmarks / production:

```bash
uv run maturin develop --release
```

Re-run after any change under `rust/`. The Python side does not auto-rebuild.

## When to add Rust ⚡

Push a kernel to Rust when:

- it runs per-pixel or per-Monte-Carlo-realization across a whole site-year;
- the Python version is measurably a bottleneck (profile first — `cProfile` or a quick `time.perf_counter` loop);
- the interface is narrow (numpy arrays in, numpy arrays out).

Keep orchestration, I/O, config validation, and anything that touches the filesystem in Python. The thin Python wrapper (e.g. `extraction/indices.py`) is where Rust kernels are called from.

## Convention 📋

- One Rust module per pipeline stage (`extraction.rs`, later `timeseries.rs`, `transitions.rs`).
- Shared helpers go in `core.rs`.
- PyO3 bindings stay in `lib.rs`; do not register Python-visible functions from other files.
