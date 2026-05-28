"""JSON config loading and validation for the three pipeline stages.

Each stage reads a JSON config (see ``configs/*.example.json``). This module
hosts dataclass schemas and load/validate helpers.
"""

from __future__ import annotations

import copy
import itertools
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field, fields
from pathlib import Path

logger = logging.getLogger(__name__)


def load_uid_mapping(path: Path) -> dict[str, str]:
    """Load annotation_id → unique_id mapping from a JSON file."""
    with path.open() as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Stage 1 — Greenness extraction
# ---------------------------------------------------------------------------

@dataclass
class ExtractionConfig:
    """Configuration for stage-1 greenness extraction."""

    image_dataset_root: Path
    output_root: Path
    polygon_files_root: Path
    polygon_config_path: Path  # sublocation-year -> polygon-file mapping JSON

    uid_mapping_path: Path  # annotation_id → unique_id mapping JSON

    categories_kept: list[str] = field(default_factory=lambda: ["individual tree"])
    shrink_ratios: list[int] = field(default_factory=lambda: [33])
    plot_dir: Path | None = None
    # Maps MD5 hash folder names → readable names used in the polygon config.
    # Required when the image archive still uses hash-named location folders.
    camera_hashes_path: Path | None = None


def _validate_extraction_config(cfg: ExtractionConfig) -> None:
    """Raise ValueError if *cfg* contains invalid or inconsistent values."""

    # Required source roots must exist
    for attr in ("image_dataset_root", "polygon_files_root"):
        p: Path = getattr(cfg, attr)
        if not p.exists():
            raise ValueError(
                f"ExtractionConfig.{attr} does not exist: {p!r}"
            )

    # polygon_config_path must be a readable JSON file
    if not cfg.polygon_config_path.exists():
        raise ValueError(
            f"ExtractionConfig.polygon_config_path does not exist: "
            f"{cfg.polygon_config_path!r}"
        )
    if not cfg.polygon_config_path.is_file():
        raise ValueError(
            f"ExtractionConfig.polygon_config_path is not a file: "
            f"{cfg.polygon_config_path!r}"
        )
    try:
        with cfg.polygon_config_path.open("r") as fh:
            json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"ExtractionConfig.polygon_config_path is not a readable JSON file "
            f"({cfg.polygon_config_path!r}): {exc}"
        ) from exc

    # output_root does not need to exist — it will be created by the pipeline.

    # camera_hashes_path, if given, must be a readable JSON file
    if cfg.camera_hashes_path is not None:
        p = cfg.camera_hashes_path
        if not p.exists() or not p.is_file():
            raise ValueError(
                f"ExtractionConfig.camera_hashes_path does not exist or is not a "
                f"file: {p!r}"
            )
        try:
            with p.open("r") as fh:
                json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"ExtractionConfig.camera_hashes_path is not a readable JSON "
                f"file ({p!r}): {exc}"
            ) from exc

    # uid_mapping_path must be a readable JSON file
    p = cfg.uid_mapping_path
    if not p.exists() or not p.is_file():
        raise ValueError(
            f"ExtractionConfig.uid_mapping_path does not exist or is not a "
            f"file: {p!r}"
        )
    try:
        with p.open("r") as fh:
            json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"ExtractionConfig.uid_mapping_path is not a readable JSON "
            f"file ({p!r}): {exc}"
        ) from exc

    # categories_kept must be a non-empty list of strings
    if not cfg.categories_kept:
        raise ValueError("ExtractionConfig.categories_kept must not be empty.")
    if not all(isinstance(c, str) for c in cfg.categories_kept):
        raise ValueError(
            "ExtractionConfig.categories_kept must be a list[str]; "
            f"got: {cfg.categories_kept!r}"
        )

    # each shrink ratio must be in the open interval (0, 100)
    for ratio in cfg.shrink_ratios:
        if not (0 < ratio < 100):
            raise ValueError(
                f"Each shrink ratio must be in (0, 100) exclusive; got {ratio!r}."
            )


def load_extraction_config(path: Path) -> ExtractionConfig:
    """Load and validate an :class:`ExtractionConfig` from a JSON file.

    Path-typed fields are coerced from strings.  All validation rules are
    applied before the config is returned; a :exc:`ValueError` is raised on
    the first problem found.

    Parameters
    ----------
    path:
        Filesystem path to the JSON config file.

    Returns
    -------
    ExtractionConfig
        A fully validated extraction configuration.
    """
    path = Path(path)
    try:
        with path.open("r") as fh:
            raw: dict = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read config file {path!r}: {exc}") from exc

    # Coerce path-typed fields from strings to pathlib.Path
    path_fields = (
        "image_dataset_root",
        "output_root",
        "polygon_files_root",
        "polygon_config_path",
        "plot_dir",
        "camera_hashes_path",
        "uid_mapping_path",
    )
    for key in path_fields:
        if key in raw and raw[key] is not None:
            raw[key] = Path(raw[key])

    known = {f.name for f in fields(ExtractionConfig)}
    unknown = set(raw.keys()) - known
    if unknown:
        logger.warning("Ignoring unknown config key(s): %s", ", ".join(sorted(unknown)))
    raw = {k: v for k, v in raw.items() if k in known}

    cfg = ExtractionConfig(**raw)
    _validate_extraction_config(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Stage 2 — Timeseries processing (filters + aggregation)
# ---------------------------------------------------------------------------

_VALID_AGGREGATION_STRATEGIES = frozenset({"avg", "p50", "p75", "p90"})


@dataclass
class ProcessingConfig:
    """Configuration for stage-2 timeseries processing."""

    input_root: Path  # stage-1 output root (per-observation GCC/RCC CSVs)
    output_root: Path

    num_workers: int = 1

    # Snow-filter inputs. None => skip snow filtering.
    snow_flags_root: Path | None = None

    # Brightness filter bounds (sum of R+G+B on a uint8 scale).
    brightness_min: float = 50.0
    brightness_max: float = 650.0

    # Aggregation grid: cartesian product of windows × strategies.
    aggregation_windows: list[str] = field(default_factory=lambda: ["1D", "3D"])
    aggregation_strategies: list[str] = field(
        default_factory=lambda: ["avg", "p50", "p75", "p90"]
    )

    # Smoothing (CV moving-average) applied to each aggregated series.
    smoothing_window_candidates: list[int] = field(
        default_factory=lambda: [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    )
    smoothing_cv_kfolds: int = 5
    smoothing_use_rust: bool = False

    # When True, tree-years with a detected mid-year sampling-rate jump are
    # re-processed against the closest-to-noon image per day for the whole year.
    noon_only_on_resolution_change: bool = True
    shrink_suffix: str | None = "33"


def _validate_processing_config(cfg: ProcessingConfig) -> None:
    """Raise ValueError if *cfg* contains invalid or inconsistent values."""

    if not cfg.input_root.exists():
        raise ValueError(
            f"ProcessingConfig.input_root does not exist: {cfg.input_root!r}"
        )

    if cfg.snow_flags_root is not None and not cfg.snow_flags_root.exists():
        raise ValueError(
            f"ProcessingConfig.snow_flags_root does not exist: "
            f"{cfg.snow_flags_root!r}"
        )

    if cfg.num_workers < 1:
        raise ValueError(
            f"ProcessingConfig.num_workers must be >= 1; got {cfg.num_workers!r}"
        )

    if cfg.brightness_min < 0 or cfg.brightness_max < 0:
        raise ValueError(
            "ProcessingConfig.brightness_{min,max} must be non-negative; "
            f"got {cfg.brightness_min!r}, {cfg.brightness_max!r}"
        )
    if cfg.brightness_min >= cfg.brightness_max:
        raise ValueError(
            "ProcessingConfig.brightness_min must be < brightness_max; "
            f"got {cfg.brightness_min!r} >= {cfg.brightness_max!r}"
        )

    if not cfg.aggregation_windows:
        raise ValueError("ProcessingConfig.aggregation_windows must not be empty.")
    if not all(isinstance(w, str) for w in cfg.aggregation_windows):
        raise ValueError(
            "ProcessingConfig.aggregation_windows must be a list[str]; "
            f"got {cfg.aggregation_windows!r}"
        )

    if not cfg.aggregation_strategies:
        raise ValueError("ProcessingConfig.aggregation_strategies must not be empty.")
    unknown = set(cfg.aggregation_strategies) - _VALID_AGGREGATION_STRATEGIES
    if unknown:
        raise ValueError(
            "ProcessingConfig.aggregation_strategies contains unknown values "
            f"{sorted(unknown)!r}; must be a subset of "
            f"{sorted(_VALID_AGGREGATION_STRATEGIES)!r}"
        )


def load_processing_config(path: Path) -> ProcessingConfig:
    """Load and validate a :class:`ProcessingConfig` from a JSON file."""
    path = Path(path)
    try:
        with path.open("r") as fh:
            raw: dict = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read config file {path!r}: {exc}") from exc

    path_fields = ("input_root", "output_root", "snow_flags_root")
    for key in path_fields:
        if key in raw and raw[key] is not None:
            raw[key] = Path(raw[key])

    known = {f.name for f in fields(ProcessingConfig)}
    unknown = set(raw.keys()) - known
    if unknown:
        logger.warning("Ignoring unknown config key(s): %s", ", ".join(sorted(unknown)))
    raw = {k: v for k, v in raw.items() if k in known}

    cfg = ProcessingConfig(**raw)
    _validate_processing_config(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Stage 3 — Transition-date estimation (smoothing + thresholding + MC)
# ---------------------------------------------------------------------------

@dataclass
class ModellingConfig:
    """Configuration for stage-3 transition-date estimation."""

    input_root: Path  # stage-2 output (aggregated greenness products)
    output_root: Path
    random_seed: int  # master seed — threaded into numpy default_rng for MC

    num_workers: int = 1

    # Which aggregated series to load (matches stage-2 outputs).
    # Accept a single string or a list of strings; the orchestrator
    # iterates over the cartesian product via iter_combinations().
    series_index: str | list[str] = "GCC"
    series_window: str | list[str] = "3D"
    series_strategy: str = "p90"

    def iter_combinations(self) -> Iterator[ModellingConfig]:
        """Yield a config copy for each (series_index, series_window) pair."""
        indices = [self.series_index] if isinstance(self.series_index, str) else self.series_index
        windows = [self.series_window] if isinstance(self.series_window, str) else self.series_window
        for idx, win in itertools.product(indices, windows):
            cfg = copy.copy(self)
            cfg.series_index = idx
            cfg.series_window = win
            yield cfg

    # Rolling-window smoothing + CV window selection.
    cv_window_min: int = 3
    cv_window_max: int = 21
    cv_kfolds: int = 5
    cv_random_state: int = 42

    # Amplitude thresholds in (0, 1) and DOY season bounds.
    amplitude_thresholds: list[float] = field(
        default_factory=lambda: [0.10, 0.25, 0.50, 0.75, 0.90]
    )
    doy_min: int = 60
    doy_max: int = 320

    # Monte Carlo uncertainty.
    mc_iterations: int = 1000

    # Use Rust extension for smoothing and threshold computation.
    use_rust: bool = False


def _validate_modelling_config(cfg: ModellingConfig) -> None:
    """Raise ValueError if *cfg* contains invalid or inconsistent values."""

    if not cfg.input_root.exists():
        raise ValueError(
            f"ModellingConfig.input_root does not exist: {cfg.input_root!r}"
        )

    valid_indices = {"GCC", "RCC", "BCC"}
    valid_windows = {"1D", "3D"}
    indices = [cfg.series_index] if isinstance(cfg.series_index, str) else cfg.series_index
    windows = [cfg.series_window] if isinstance(cfg.series_window, str) else cfg.series_window
    if not indices:
        raise ValueError("ModellingConfig.series_index must not be empty.")
    if not windows:
        raise ValueError("ModellingConfig.series_window must not be empty.")
    for v in indices:
        if v not in valid_indices:
            raise ValueError(
                f"ModellingConfig.series_index: {v!r} not in {valid_indices}"
            )
    for v in windows:
        if v not in valid_windows:
            raise ValueError(
                f"ModellingConfig.series_window: {v!r} not in {valid_windows}"
            )

    if cfg.series_strategy not in _VALID_AGGREGATION_STRATEGIES:
        raise ValueError(
            f"ModellingConfig.series_strategy: {cfg.series_strategy!r} not in "
            f"{sorted(_VALID_AGGREGATION_STRATEGIES)!r}"
        )

    if cfg.num_workers < 1:
        raise ValueError(
            f"ModellingConfig.num_workers must be >= 1; got {cfg.num_workers!r}"
        )

    if cfg.cv_window_min < 3 or cfg.cv_window_max < 3:
        raise ValueError(
            "ModellingConfig.cv_window_{min,max} must be >= 3; "
            f"got {cfg.cv_window_min!r}, {cfg.cv_window_max!r}"
        )
    if cfg.cv_window_min > cfg.cv_window_max:
        raise ValueError(
            "ModellingConfig.cv_window_min must be <= cv_window_max; "
            f"got {cfg.cv_window_min!r} > {cfg.cv_window_max!r}"
        )
    if cfg.cv_window_min % 2 == 0 or cfg.cv_window_max % 2 == 0:
        raise ValueError(
            "ModellingConfig.cv_window_{min,max} must both be odd; "
            f"got {cfg.cv_window_min!r}, {cfg.cv_window_max!r}"
        )

    if cfg.cv_kfolds < 2:
        raise ValueError(
            f"ModellingConfig.cv_kfolds must be >= 2; got {cfg.cv_kfolds!r}"
        )

    if not cfg.amplitude_thresholds:
        raise ValueError("ModellingConfig.amplitude_thresholds must not be empty.")
    for t in cfg.amplitude_thresholds:
        if not (0.0 < t < 1.0):
            raise ValueError(
                "ModellingConfig.amplitude_thresholds entries must be in (0, 1); "
                f"got {t!r}"
            )

    if not (1 <= cfg.doy_min < cfg.doy_max <= 366):
        raise ValueError(
            "ModellingConfig requires 1 <= doy_min < doy_max <= 366; "
            f"got doy_min={cfg.doy_min!r}, doy_max={cfg.doy_max!r}"
        )

    if cfg.mc_iterations < 1:
        raise ValueError(
            f"ModellingConfig.mc_iterations must be >= 1; got {cfg.mc_iterations!r}"
        )



# ---------------------------------------------------------------------------
# Aggregation — post-pipeline CSV assembly
# ---------------------------------------------------------------------------

@dataclass
class AggregateConfig:
    """Configuration for the post-pipeline record aggregation script."""

    input_root: Path   # greenness dataset root (stage 1/2/3 outputs)
    output_path: Path  # destination CSV path

    # Threshold dicts: bare field name (no product prefix) → bound.
    # Fields from sampling-flags.json are checked once; fields from transition
    # JSONs (snr, n_peak, noise_std, …) are checked across all products found
    # for the tree-year — any product failing excludes the row.
    min_thresholds: dict = field(default_factory=dict)
    max_thresholds: dict = field(default_factory=dict)

    # Keys listed here treat NaN/missing values as passing rather than failing.
    # Used for derived metrics (e.g. season_length) where absence should not
    # exclude the tree-year.
    nan_ok_keys: list = field(default_factory=list)

    # Optional CSV of (uid, year) pairs to exclude from the final output.
    exclude_csv_path: Path | None = None


def _validate_aggregate_config(cfg: AggregateConfig) -> None:
    if not cfg.input_root.exists():
        raise ValueError(
            f"AggregateConfig.input_root does not exist: {cfg.input_root!r}"
        )
    if not isinstance(cfg.min_thresholds, dict):
        raise ValueError("AggregateConfig.min_thresholds must be a dict.")
    if not isinstance(cfg.max_thresholds, dict):
        raise ValueError("AggregateConfig.max_thresholds must be a dict.")


def load_aggregate_config(path: Path) -> AggregateConfig:
    """Load and validate an :class:`AggregateConfig` from a JSON file."""
    path = Path(path)
    try:
        with path.open("r") as fh:
            raw: dict = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read config file {path!r}: {exc}") from exc

    for key in ("input_root", "output_path", "exclude_csv_path"):
        if key in raw and raw[key] is not None:
            raw[key] = Path(raw[key])

    known = {f.name for f in fields(AggregateConfig)}
    unknown = set(raw.keys()) - known
    if unknown:
        logger.warning("Ignoring unknown config key(s): %s", ", ".join(sorted(unknown)))
    raw = {k: v for k, v in raw.items() if k in known}

    cfg = AggregateConfig(**raw)
    _validate_aggregate_config(cfg)
    return cfg


def load_modelling_config(path: Path) -> ModellingConfig:
    """Load and validate a :class:`ModellingConfig` from a JSON file."""
    path = Path(path)
    try:
        with path.open("r") as fh:
            raw: dict = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read config file {path!r}: {exc}") from exc

    path_fields = ("input_root", "output_root")
    for key in path_fields:
        if key in raw and raw[key] is not None:
            raw[key] = Path(raw[key])

    known = {f.name for f in fields(ModellingConfig)}
    unknown = set(raw.keys()) - known
    if unknown:
        logger.warning("Ignoring unknown config key(s): %s", ", ".join(sorted(unknown)))
    raw = {k: v for k, v in raw.items() if k in known}

    cfg = ModellingConfig(**raw)
    _validate_modelling_config(cfg)
    return cfg
