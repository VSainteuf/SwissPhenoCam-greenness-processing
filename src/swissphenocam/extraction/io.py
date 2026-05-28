"""CSV + JSON output writing for per-site-year greenness extraction results (stage 1)."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


NAMING_CONVENTION: dict[str, str] = dict(
    I="Tree",
    Gp="Group",
    Gs="Grassland",
    C="Cropland",
    S="Shrubs",
    O="Other",
    X="DeadTree",
)


def _add_full_name(row: pd.Series) -> str:
    """Row-wise helper for building ``full_name`` from a ``name`` column."""
    name, number = row["name"].split(".")
    return f"{NAMING_CONVENTION[name]}{number.zfill(4)}"


def format_obs_df(values: dict) -> pd.DataFrame:
    """Convert a ``{datetime: {metric: value, ...}}`` dict to a tidy DataFrame.

    The index is a **string** index (not DatetimeIndex) with the format
    ``YYYY-MM-DD_HHMM``, named ``datetime``.  Rows are sorted chronologically.
    Callers that require a DatetimeIndex must convert after the fact via
    ``pd.to_datetime(df.index, format="%Y-%m-%d_%H%M")``.
    """
    df = pd.DataFrame.from_dict(values, orient="index")
    df.index = pd.to_datetime(df.index).strftime("%Y-%m-%d_%H%M")
    df.index.name = "datetime"
    df.sort_index(inplace=True)
    return df


def write_site_year(
    *,
    output_dir: Path,
    polygons: dict,
    per_polygon_results: dict[str, pd.DataFrame],
    data_dir: Path,
    uid_mapping: dict[str, str],
) -> list[str]:
    """Write all CSV and JSON outputs for one processed site-year.

    Creates ``output_dir`` if it does not exist, then writes:

    * ``<uid>_<year>_raw-cc-data.csv`` — per-observation time
      series for each annotation.
    * ``<uid>_<year>_metadata.json`` — per-annotation metadata
      dict including location, temporal-QC stats, and processing provenance.

    Parameters
    ----------
    output_dir:
        Base destination directory; actual outputs are written under
        ``<output_dir.parent>/<uid>/<year>``.
    polygons:
        Polygon dict as produced by ``extraction.polygons`` — keyed by
        annotation ID, values contain at least ``name``, ``category``,
        ``coords``, and any shrunk/area entries added by ``add_areas``.
    per_polygon_results:
        Mapping from annotation ID to an observations DataFrame as returned
        by ``format_obs_df``.
    data_dir:
        Path to the site-year image folder
        (``<archive>/<location>/<sub_location>/<year>``).  Used to derive
        ``location``, ``sub_location``, and ``year`` metadata fields.
    uid_mapping:
        Mapping from annotation_id to unique_id. Outputs are written under
        ``<output_dir.parent>/<uid>/<year>`` and metadata includes the unique_id.

    Returns
    -------
    list[str]
        List of annotation_ids that were missing from uid_mapping.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    location = data_dir.parent.parent.name
    sub_location = data_dir.parent.name
    year = data_dir.name

    if not polygons:
        logger.warning("write_site_year called with empty polygons for %s — skipping.", data_dir)
        return []

    # --- Build site-year metadata dataframe ---
    meta = pd.DataFrame.from_dict(polygons, orient="index")
    area_keys = [k for k in meta.columns if k.startswith("area")]
    try:
        meta = meta[["name", "category", "mix", "type", "genus", "species"] + area_keys]
    except KeyError:
        meta = meta[["name", "category"] + area_keys]
    meta["annotation_id"] = meta.index
    meta["full_name"] = meta.apply(_add_full_name, axis=1)
    # --- Write per-annotation CSV + JSON ---
    missing: list[str] = []
    for annotation_id, df in per_polygon_results.items():
        uid = uid_mapping.get(annotation_id)
        if uid is None:
            missing.append(annotation_id)
            continue
        identifier = f"{uid}_{year}"
        tree_dir = output_dir.parent / uid / year

        # Start with a copy of the source polygon info so we don't mutate the
        # caller's dict, then layer in provenance and temporal-QC fields.
        time_series_meta: dict = dict(polygons[annotation_id])
        time_series_meta.update(
            dict(
                location=location,
                sub_location=sub_location,
                year=year,
                annotation_id=annotation_id,
                full_name=identifier,
                date_processed=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        )

        time_series_meta["unique_id"] = uid

        try:
            dt_index = pd.to_datetime(df.index, format="%Y-%m-%d_%H%M")
            doy = dt_index.day_of_year.to_numpy()
            interval_to_next = abs(doy[:-1] - doy[1:])
            gap_mask = interval_to_next > 1
            time_series_meta.update(
                dict(
                    start_date=dt_index.min().strftime("%Y-%m-%d"),
                    end_date=dt_index.max().strftime("%Y-%m-%d"),
                    avg_obs_per_day=len(df)
                    / ((dt_index.max() - dt_index.min()).days + 1),
                    max_obs_per_day=int(dt_index.day_of_year.value_counts().max()),
                    min_obs_per_day=int(dt_index.day_of_year.value_counts().min()),
                    n_gaps=int(gap_mask.sum()),
                    gap_start_dates=[
                        (d + timedelta(days=1)).strftime("%Y-%m-%d")
                        for d in dt_index[:-1][gap_mask]
                    ],
                    gap_lengths_day=interval_to_next[gap_mask].astype(int).tolist(),
                )
            )
        except (AttributeError, ValueError, TypeError):
            logger.warning(
                "Could not compute temporal-QC stats for annotation %s.",
                annotation_id,
                exc_info=True,
            )

        tree_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(tree_dir / f"{identifier}_raw-cc-data.csv")
        with open(
            tree_dir / f"{identifier}_metadata.json", "w"
        ) as fh:
            fh.write(json.dumps(time_series_meta, indent=4))

    return missing
