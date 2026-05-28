"""CSV I/O and metadata helpers for the stage-2 greenness products.

Reads stage-1 per-polygon CSVs and metadata JSONs; reads daily snow-flag
CSVs; writes the aggregated 1D/3D greenness products.  Also provides
``compute_temporal_sampling_flags`` which characterises raw series coverage
and is stored in the per-polygon metadata JSON.
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pandas as pd

_SEASON_BOUNDS: dict[str, tuple[int, int]] = {
    "spring": (60, 160),
    "summer": (161, 230),
    "autumn": (231, 320),
}


def compute_temporal_sampling_flags(series: pd.DataFrame) -> dict:
    """Return coverage and gap statistics for the raw observation series.

    The returned dict is merged into the per-polygon metadata JSON by the
    stage-2 driver.
    """
    out: dict = {}

    interval_to_next = abs(
        series.index.day_of_year[:-1] - series.index.day_of_year[1:]
    )
    out.update(
        dict(
            start_date=series.index.min().strftime("%Y-%m-%d"),
            end_date=series.index.max().strftime("%Y-%m-%d"),
            avg_obs_per_day=len(series)
            / ((series.index.max() - series.index.min()).days + 1),
            max_obs_per_day=int(series.index.day_of_year.value_counts().max()),
            min_obs_per_day=int(series.index.day_of_year.value_counts().min()),
            n_gaps=int((interval_to_next > 1).sum()),
            gap_start_dates=[
                (d + timedelta(days=1)).strftime("%Y-%m-%d")
                for d in series.iloc[:-1].loc[interval_to_next > 1].index
            ],
            gap_lengths_day=interval_to_next[interval_to_next > 1]
            .astype(int)
            .tolist(),
        )
    )

    for season, (start_day, end_day) in _SEASON_BOUNDS.items():
        doy = series.index.day_of_year
        season_series = series[(doy >= start_day) & (doy <= end_day)]
        if len(season_series) < 2:
            out.update(
                {
                    f"{season}_coverage_ratio": 0.0,
                    f"{season}_avg_obs_per_day": 0,
                    f"{season}_n_gaps": 1,
                    f"{season}_max_gap_days": end_day - start_day + 1,
                    f"{season}_max_consecutive_missing_days": end_day - start_day + 1,
                }
            )
            continue

        sdoy = season_series.index.day_of_year
        interval_season = abs(sdoy[:-1] - sdoy[1:])
        out.update(
            {
                f"{season}_coverage_ratio": float(sdoy.unique().size)
                / (end_day - start_day + 1),
                f"{season}_avg_obs_per_day": len(season_series)
                / (
                    (season_series.index.max() - season_series.index.min()).days
                    + 1
                ),
                f"{season}_n_gaps": int((interval_season > 1).sum()),
                f"{season}_max_gap_days": int(interval_season.max()),
            }
        )

        all_doys = set(range(start_day, end_day + 1))
        obs_doys = set(doy[(doy >= start_day) & (doy <= end_day)].unique())
        missing = sorted(all_doys - obs_doys)
        if not missing:
            max_consec = 0
        else:
            max_consec = current = 1
            for i in range(1, len(missing)):
                if missing[i] == missing[i - 1] + 1:
                    current += 1
                    max_consec = max(max_consec, current)
                else:
                    current = 1
        out[f"{season}_max_consecutive_missing_days"] = max_consec

    return out


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def read_raw_greenness(
    root: Path,
    location: str,
    sublocation: str,
    year: str,
    full_name: str,
) -> pd.DataFrame:
    site = f"{location}_{sublocation}"
    path = root / site / full_name / year / f"{full_name}_{year}_raw-cc-data.csv"
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d_%H%M")
    return df


def read_greenness_metadata(
    root: Path,
    location: str,
    sublocation: str,
    year: str,
    full_name: str,
) -> dict:
    site = f"{location}_{sublocation}"
    path = root / site / full_name / year / f"{full_name}_{year}_metadata.json"
    return json.loads(path.read_text())


def read_snow_flag(snow_dir: Path, location: str, year: str) -> pd.DataFrame | None:
    path = snow_dir / location / f"{location}-{year}-daily_snow_flag.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
    return df


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_sampling_flags(out_folder: Path, full_name: str, flags: dict) -> None:
    """Write temporal sampling flags dict to *out_folder* as JSON."""
    out_folder.mkdir(parents=True, exist_ok=True)
    path = out_folder / f"{full_name}-sampling-flags.json"
    path.write_text(json.dumps(flags, indent=2))


def write_aggregated_product(
    out_folder: Path,
    full_name: str,
    channel: str,
    freq: str,
    df: pd.DataFrame,
) -> None:
    """Write aggregated product CSV to *out_folder*.

    The file is named ``<full_name>-<channel>-<freq>-product.csv``.
    """
    out_folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_folder / f"{full_name}-{channel}-{freq}-product.csv")


def write_smoothed_product(
    out_folder: Path,
    full_name: str,
    channel: str,
    freq: str,
    df: pd.DataFrame,
) -> None:
    """Write CV-smoothed product CSV to *out_folder*.

    The file is named ``<full_name>-<channel>-<freq>-smoothed.csv``.
    """
    out_folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_folder / f"{full_name}-{channel}-{freq}-smoothed.csv")


def read_aggregated_product(
    data_path: Path,
    channel: str = "GCC",
    freq: str = "1D",
    name: str | None = None,
) -> pd.DataFrame | None:
    """Read a stage-2 aggregated product CSV from *data_path*.

    Returns ``None`` if the file does not exist.
    """
    identifier = name if name is not None else data_path.name
    file_path = data_path / f"{identifier}-{channel}-{freq}-product.csv"
    if not file_path.exists():
        return None
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
    return df
