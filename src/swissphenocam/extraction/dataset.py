"""Walk a webcam image archive and parse image metadata from file paths.

Responsibilities:
- Enumerate sub-directories and image files within the archive tree
  (layout: location / sublocation / year / month / day / image).
- Build the list of site-year keys used throughout the pipeline
  (format: ``"{location}_{sublocation}-{year}"``).
- Parse observation datetimes from image filenames, supporting both
  the legacy ``YYYY_MMDD_HHMMSS.jpg`` pattern and the current
  ``YYYY-MM-DD_HHMM.jpg`` pattern.
"""

import os
import re
from datetime import datetime
from pathlib import Path


# Image extensions recognised by the archive scanner.
# Note: 'jp2k' intentionally lacks a leading dot — this matches observed
# filenames in the archive and must not be "fixed".
IMAGE_EXTENSIONS: tuple[str, ...] = ('.jpg', '.jpeg', '.jp2', 'jp2k')


def list_subdirs(parent: Path) -> list[str]:
    """Return the names of all immediate sub-directories of *parent*.

    Parameters
    ----------
    parent:
        Directory to inspect.

    Returns
    -------
    list[str]
        Sorted list of sub-directory names (not full paths).
    """
    return [
        f for f in os.listdir(parent)
        if os.path.isdir(os.path.join(parent, f))
    ]


def list_images(site_year_folder: Path) -> list[Path]:
    """Return a sorted list of all image paths under *site_year_folder*.

    Expects the layout ``<site_year_folder>/month/day/<image>``.

    Parameters
    ----------
    site_year_folder:
        The year-level directory (e.g. ``.../location/sublocation/2023``).

    Returns
    -------
    list[Path]
        All files whose names end with one of :data:`IMAGE_EXTENSIONS`,
        collected across all month/day sub-directories and sorted
        lexicographically.
    """
    all_files: list[Path] = []
    for month in list_subdirs(site_year_folder):
        for day in list_subdirs(site_year_folder / month):
            day_folder = site_year_folder / month / day
            day_files = [
                f for f in os.listdir(day_folder)
                if f.lower().endswith(IMAGE_EXTENSIONS)
            ]
            all_files.extend([day_folder / f for f in day_files])
    return sorted(all_files)


def get_list_of_site_years(
    input_dir: Path,
    hash_to_name: dict[str, str] | None = None,
) -> list[str]:
    """Walk *input_dir* three levels deep and return site-year keys.

    The archive layout is expected to be::

        <input_dir> / <location> / <sublocation> / <year> / ...

    Each discovered ``(location, sublocation, year)`` triple is encoded
    as the string ``"{location}_{sublocation}-{year}"``.

    Some older cameras use an MD5 hash as the ``location`` folder name
    instead of a human-readable name.  When *hash_to_name* is provided,
    any hash-named location folder is translated to its readable equivalent
    before building the site-year key, so that the result matches the keys
    used in the polygon annotation config.

    Parameters
    ----------
    input_dir:
        Root of the webcam image archive.
    hash_to_name:
        Optional mapping ``{md5_hash: readable_name}`` (the inverse of
        ``camera_hashes.json``).  Pass ``None`` (default) to skip translation.

    Returns
    -------
    list[str]
        All site-year strings found, in traversal order.
    """
    site_years: list[str] = []
    for location in list_subdirs(input_dir):
        for subloc in list_subdirs(input_dir / location):
            for year in list_subdirs(input_dir / location / subloc):
                site_years.append(f"{location}_{subloc}-{year}")
    return site_years


# Compiled regex patterns for the two supported filename date formats.
_NEW_PATTERN = re.compile(r'(\d{4}-\d{2}-\d{2})_(\d{4})\.(jpg|png|jpeg|jp2)')
_OLD_PATTERN = re.compile(r'(\d{4}_\d{4})_(\d{6})\.(jpg|png|jpeg|jp2)')


def parse_datetime_from_filepath(p: Path) -> datetime | None:
    """Parse the observation datetime encoded in an image filename.

    Two filename conventions are recognised:

    * **New pattern** — ``<prefix>_2024-08-15_1200.jpg``
      → parsed with format ``%Y-%m-%d_%H%M``.
    * **Old pattern** — ``2021_0604_120000.jpg``
      → parsed with format ``%Y_%m%d_%H%M%S``.

    The new pattern is tried first; the old pattern is the fallback.

    The returned datetime is **naive local time** at the camera site — see
    the time-convention note in :mod:`swissphenocam.paths`.

    Parameters
    ----------
    p:
        Path to an image file.  Only :attr:`~pathlib.Path.name` is examined.

    Returns
    -------
    datetime | None
        Parsed datetime, or ``None`` if neither pattern matches.
    """
    match = _NEW_PATTERN.search(p.name)
    if match:
        return datetime.strptime(
            f"{match.group(1)}_{match.group(2)}", "%Y-%m-%d_%H%M"
        )

    match = _OLD_PATTERN.search(p.name)
    if match:
        return datetime.strptime(
            f"{match.group(1)}_{match.group(2)}", "%Y_%m%d_%H%M%S"
        )

    return None
