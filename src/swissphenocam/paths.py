"""Dataset folder-structure conventions shared across the pipeline.

Input webcam dataset layout:
    <root>/<location>/<sublocation>/<year>/<month>/<day>/<image_file>

Supported image extensions: .jpeg, .jpg, .jp2, jp2k (no leading dot — legacy archive
convention), .png

Greenness dataset layout (mirrors the above, per tree-year):
    <root>/<location>/<sublocation>/<year>/
        site-year-metadata.csv
        <full_name>/
            <full_name>_raw-cc-data.csv                   # per-observation values (stage 1)
            <full_name>_metadata.json                     # metadata + QC stats (stage 1)
            <full_name>-GCC-1D-product.csv                # 1-day aggregated GCC (stage 2)
            <full_name>-GCC-3D-product.csv                # 3-day aggregated GCC (stage 2)
            <full_name>-RCC-1D-product.csv
            <full_name>-RCC-3D-product.csv
            <full_name>-BCC-1D-product.csv                # 1-day aggregated BCC (stage 2)
            <full_name>-BCC-3D-product.csv                # 3-day aggregated BCC (stage 2)
            <full_name>-<INDEX>-<WINDOW>-<STRATEGY>-transitions.json  # phenological transition dates (stage 3)

Full name convention (stage 1 output filenames): the short code in the
polygon's ``name`` field (e.g. ``I.12``, ``Gp.3``, ``Gs.1``) is expanded via
NAMING_CONVENTION to ``Tree0012``, ``Group0003``, ``Grassland0001`` etc.

Region of interest: by default greenness is extracted on the 33%-area-shrunk
polygon (columns ``GCC_s33`` / ``RCC_s33`` / ``BCC_s33``). The full polygon is retained for
metadata only (area, width, height).

Time convention
---------------
Image filenames encode **naive local time at the camera site** (no timezone
offset). All downstream timestamps — per-observation CSV indexes, daily snow
flags, aggregated 1D/3D products, transition DOYs — are naive and interpreted
as the camera's local wall-clock. No UTC conversion is applied anywhere in
the pipeline. Consumers comparing across sites in different timezones must
apply an offset themselves.

DOY convention
--------------
DOY values are 1-indexed via pandas ``DatetimeIndex.dayofyear`` — Jan 1 is
DOY 1, Dec 31 is DOY 365 (non-leap) or 366 (leap). On leap years, Feb 29 is
DOY 60, so the default season window ``[doy_min=60, doy_max=320]`` spans a
slightly different calendar range than on non-leap years. This is the
intended behaviour: season edges are expressed in photoperiod/DOY, not
calendar date.
"""
