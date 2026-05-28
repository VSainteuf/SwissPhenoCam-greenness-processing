"""Load and validate polygon ROI JSON files.

A polygon file contains multiple annotations keyed by annotation_id. Only
entries whose ``category`` is in ``categories_kept`` are consumed downstream.

Schema per entry (see example-files/):
    {
        "name":     str,        # e.g. "I.12" (not a uid)
        "coords":   [[x, y], ...],  # pixel coordinates
        "category": str,        # e.g. "individual tree"
        "genus":    str | None,
        "species":  str | None,
    }

After processing the polygon dict has the shape:
    {
        annotation_id: {
            "name":           str,
            "category":       str,
            "coords":         list,
            "coords_s{ratio}": list,   # one entry per shrink ratio
            "area":           float,
            "area_s{ratio}":  float,   # one entry per shrink ratio
            "polygon_width":  float,
            "polygon_height": float,
            "genus":          str | None,
            "species":        str | None,
        }
    }
"""

import json
import warnings
from pathlib import Path

from shapely.geometry import MultiPolygon, Polygon


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_polygon_config(
    polygon_config_path: Path,
    polygon_dir: Path,
    name_to_hash: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, dict]]:
    """Load the annotation config JSON and return site-years with polygon dicts.

    The config JSON maps ``{location_key: {year: polygon_filename}}``.
    ``location_key`` already encodes ``{location}_{sublocation}`` and must
    match image-archive folder names (no cam-hash rewriting).

    Parameters
    ----------
    polygon_config_path:
        Path to the annotation config JSON
        (e.g. ``annotation_config_v5.json``).
    polygon_dir:
        Directory that contains the individual polygon JSON files.
    name_to_hash:
        Optional mapping from human-readable name to camera hash for 
        the non-meteoswiss sites. This is required to properly match
        the polygon files with the image data. 

    Returns
    -------
    site_years:
        List of ``"{location}_{sublocation}-{year}"`` strings for every
        site-year that has a valid polygon file on disk.
    site_year_to_polygons:
        Mapping from the same site-year strings to their raw polygon dicts
        (annotation_id → entry).  No filtering or shrinking is applied here.
    """
    polygon_config_path = Path(polygon_config_path)
    polygon_dir = Path(polygon_dir)

    with open(polygon_config_path, "r") as f:
        config = json.load(f)

    site_years: list[str] = []
    site_year_to_polygons: dict[str, dict] = {}

    for location, location_config in config.items():
        for year, year_polygon_file in location_config.items():
            polygon_file = polygon_dir / year_polygon_file
            if not polygon_file.exists():
                warnings.warn(
                    f"Polygon file {year_polygon_file!r} does not exist for "
                    f"{location!r} in {year}. Skipping this site-year.",
                    stacklevel=2,
                )
                continue
            site_year = f"{location}-{year}"
            if name_to_hash is not None:
                if not is_meteoswiss(location) and location in name_to_hash.keys():
                    site_year = f"{name_to_hash[location]}_1-{year}"

            site_years.append(site_year)
            with open(polygon_file) as fh:
                site_year_to_polygons[site_year] = json.load(fh)

    return site_years, site_year_to_polygons

def is_meteoswiss(location: str) -> bool:
    return len(location) == 4 and location.isnumeric()

def filter_categories(
    polygons: dict,
    categories_kept: list[str],
) -> dict:
    """Return a new polygon dict containing only entries in *categories_kept*.

    Parameters
    ----------
    polygons:
        Raw polygon dict as loaded by :func:`load_polygon_config`.
    categories_kept:
        List of category strings to retain.  The comparison is exact
        (case-sensitive) and uses the full strings stored in the JSON
        (e.g. ``"individual tree"``, ``"group of trees"``).

    Returns
    -------
    Filtered copy of *polygons* (shallow-copies the retained entries).
    """
    return {
        ann_id: entry
        for ann_id, entry in polygons.items()
        if entry.get("category") in categories_kept
    }


def add_shrunk_segments(polygons: dict, shrink_ratios: list[int]) -> dict:
    """Add shrunk coordinate lists to each polygon entry in-place.

    For each ratio *r* in *shrink_ratios* a ``"coords_s{r}"`` key is added
    whose value is the exterior coords of the polygon shrunk to ``r/100`` of
    its original area (via :func:`scale_polygon_inw_offset`).  Entries for
    which the shrinking produces an invalid or empty geometry are silently
    skipped (the key is not added).

    Parameters
    ----------
    polygons:
        Polygon dict (typically already filtered by category).
    shrink_ratios:
        List of integer target-area percentages, e.g. ``[33]``.

    Returns
    -------
    The same *polygons* dict, mutated in-place and returned for chaining.
    """
    # ratio 33 means shrink TO 33% of original area, not BY 33% (i.e. target area = 0.33 * A0).
    ratios = [int(r) / 100.0 for r in shrink_ratios]
    for annotation_id in polygons:
        original_polygon = Polygon(polygons[annotation_id]["coords"])
        for ratio in ratios:
            shrunk_polygon = scale_polygon_inw_offset(original_polygon, ratio)
            if (
                not shrunk_polygon.is_empty
                and shrunk_polygon.is_valid
                and type(shrunk_polygon) is Polygon
            ):
                polygons[annotation_id][f"coords_s{int(ratio * 100)}"] = list(
                    shrunk_polygon.exterior.coords
                )
    return polygons


def add_areas(polygons: dict) -> dict:
    """Compute areas and bounding-box dimensions for each polygon entry.

    For every ``coords``-prefixed key (``coords``, ``coords_s33``, …) an
    ``area`` / ``area_s{ratio}`` entry is added.  For ``coords`` only,
    ``polygon_width`` and ``polygon_height`` (bounding-box extents) are also
    added.

    Parameters
    ----------
    polygons:
        Polygon dict after :func:`add_shrunk_segments` has been applied.

    Returns
    -------
    The same *polygons* dict, mutated in-place and returned for chaining.
    """
    for annotation_id in polygons:
        new_entries = {}
        for k in polygons[annotation_id].keys():
            if k.startswith("coords"):
                polygon = Polygon(polygons[annotation_id][k])
                out_k = "area" if k == "coords" else f"area_s{k.split('_s')[-1]}"
                new_entries[out_k] = polygon.area
            if k == "coords":
                polygon = Polygon(polygons[annotation_id][k])
                minx, miny, maxx, maxy = polygon.bounds
                new_entries["polygon_width"] = maxx - minx
                new_entries["polygon_height"] = maxy - miny

        if new_entries:
            polygons[annotation_id].update(new_entries)
    return polygons


# ---------------------------------------------------------------------------
# Core shrink algorithm — keep verbatim
# ---------------------------------------------------------------------------

def scale_polygon_inw_offset(polygon, factor, tol=1e-9, max_iters=100):
    """
    Scales a polygon by offsetting its boundary until the area
    matches a given fraction of the original area.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Input polygon to shrink.
    factor : float
        Target area fraction (e.g., 0.5 means half the original area).
    tol : float, optional
        Tolerance for negligible area changes.
    max_iters : int, optional
        Maximum iterations for bracketing.

    Returns
    -------
    scaled_polygon : shapely.geometry.Polygon
        Shrunk polygon.  Returns an empty ``Polygon()`` if the input is empty,
        invalid after buffer-cleaning, or if the shrink produces a degenerate
        geometry.
    """

    # --- Step 1: Build and clean polygon ---
    p_in = polygon
    if p_in.is_empty:
        return Polygon()

    # Try to clean up possible self-intersections
    p_in = p_in.buffer(0)
    if p_in.is_empty or not p_in.is_valid:
        return Polygon()

    # --- Step 2: Compute target area ---
    A0 = p_in.area
    if A0 <= 0:
        return Polygon()
    A_target = factor * A0

    # --- Step 3: Quick exit if negligible change ---
    if abs(A_target - A0) < tol:
        return Polygon(p_in.exterior.coords)

    # --- Step 4: Determine direction (+grow / -shrink) ---
    sgn = 1 if A_target > A0 else -1

    def area_diff(d):
        """Compute difference between buffered area and target area."""
        p_buf = p_in.buffer(sgn * d)
        # Handle possible MultiPolygon or empty
        if p_buf.is_empty:
            return -A_target
        if isinstance(p_buf, MultiPolygon):
            p_buf = max(p_buf.geoms, key=lambda g: g.area)
        return p_buf.area - A_target

    # --- Step 5: Bracket the root (find range where sign changes) ---
    d2 = 1.0
    iters = 0
    while iters < max_iters:
        A = p_in.buffer(sgn * d2).area
        if (sgn < 0 and A <= A_target) or (sgn > 0 and A >= A_target):
            break
        d2 *= 1.5
        iters += 1

    # --- Step 6: Binary search for offset distance ---
    # Uses Shapely buffer-based inward offset to achieve area-proportional erosion; up to 50 bisection iterations.
    d1 = 0.0
    for _ in range(50):
        mid = 0.5 * (d1 + d2)
        f_mid = area_diff(mid)
        f_d1 = area_diff(d1)
        if f_mid * f_d1 <= 0:
            d2 = mid
        else:
            d1 = mid
        if abs(f_mid) < tol:
            break

    d_final = sgn * 0.5 * (d1 + d2)

    # --- Step 7: Apply final offset ---
    p_out = p_in.buffer(d_final)

    # If the offset splits the polygon, keep the largest piece
    if isinstance(p_out, MultiPolygon):
        p_out = max(p_out.geoms, key=lambda g: g.area)

    # Validate the final geometry
    if p_out.is_empty or not p_out.is_valid or p_out.area < tol:
        return Polygon()

    # Ensure containment when shrinking: intersection clips any buffer overshoot back inside the original boundary.
    if factor < 1:
        p_out = p_out.intersection(p_in)

    return Polygon(list(p_out.exterior.coords))
