"""GCC, RCC, and BCC index computation for polygons in an image.

Mask-matrix preparation and per-image GCC/RCC/BCC computation.  The hot loop is
implemented in Rust (``swissphenocam._native.compute_chromcoord_rs``) and
operates directly on the uint8 BGR image bytes and a CSR-style flat mask
representation.  Module-level globals (``MASK_ROW_PTR``, ``MASK_PIXELS``,
``MASK_COUNTS``, ``MASK_KEYS``) are populated once per worker via
``worker_init`` so that multiprocessing pools share the masks without
re-serialising them on every task.  The globals are set explicitly via
``worker_init`` because multiprocessing workers cannot inherit them through
forking when arrays are large (avoids re-serialisation overhead per task).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import csr_matrix

from swissphenocam import _native
from swissphenocam.extraction.dataset import parse_datetime_from_filepath

logger = logging.getLogger(__name__)

# Prevent multiprocessing workers from oversubscribing CPUs.
# These must run at import time (not inside any function).
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

# === GLOBALS SHARED BY WORKERS ===
MASK_ROW_PTR: np.ndarray | None = None
MASK_PIXELS: np.ndarray | None = None
MASK_COUNTS: np.ndarray | None = None
MASK_KEYS: list[tuple[str, str]] | None = None
MASK_SHAPE: tuple[int, int] | None = None
_SHAPE_WARNED: set[tuple[int, int]] = set()


def prepare_masks_vectorized(
    polygons: dict,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """Rasterise polygons into a CSR-style flat mask representation.

    Each polygon may contribute multiple rows — one per ``coords*`` key it
    contains (``coords``, ``coords_s33``, etc.).

    Parameters
    ----------
    polygons:
        Annotation dict ``{annotation_id: {"coords": [...], "coords_s33":
        [...], ...}}``.
    shape:
        ``(H, W)`` of the target image.

    Returns
    -------
    mask_row_ptr:
        uint32 CSR row pointer of length ``n_masks + 1``. Pixels of mask
        ``m`` are ``mask_pixels[mask_row_ptr[m]:mask_row_ptr[m+1]]``.
    mask_pixels:
        uint32 flat pixel indices (``row*W + col``), concatenated per mask.
    mask_counts:
        uint32 per-mask pixel counts, length ``n_masks``.
    mask_keys:
        List of ``(annotation_id, key)`` identifying each mask row.
    """
    H, W = shape
    n_pixels = H * W

    data: list[int] = []
    rows: list[int] = []
    cols: list[int] = []
    mask_keys: list[tuple[str, str]] = []
    mask_idx = 0

    for ann_id, submasks in polygons.items():
        for key, mask_coords in submasks.items():
            if "coords" not in key:
                continue
            coords = np.array(mask_coords).astype(np.int32)
            mask = np.zeros((H, W), dtype=np.uint8)
            pts = coords.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
            if mask.sum() == 0:
                continue

            nz = np.flatnonzero(mask)
            rows.extend([mask_idx] * len(nz))
            cols.extend(nz)
            data.extend([1] * len(nz))
            mask_keys.append((ann_id, key))
            mask_idx += 1

    # Build via scipy just to get sorted CSR structure, then extract flat arrays.
    mask_matrix = csr_matrix(
        (data, (rows, cols)), shape=(mask_idx, n_pixels), dtype=np.uint8
    )
    mask_matrix.sort_indices()
    mask_matrix.eliminate_zeros()

    mask_row_ptr = mask_matrix.indptr.astype(np.uint32, copy=False)
    mask_pixels = mask_matrix.indices.astype(np.uint32, copy=False)
    mask_counts = np.diff(mask_row_ptr).astype(np.uint32, copy=False)

    return mask_row_ptr, mask_pixels, mask_counts, mask_keys


def worker_init(mask_row_ptr, mask_pixels, mask_counts, mask_keys, mask_shape) -> None:
    """Multiprocessing pool initialiser — called once per worker process.

    Stores the shared mask arrays in module-level globals so each worker
    can access pre-built masks without re-serialisation on every task.
    """
    global MASK_ROW_PTR, MASK_PIXELS, MASK_COUNTS, MASK_KEYS, MASK_SHAPE
    MASK_ROW_PTR = mask_row_ptr
    MASK_PIXELS = mask_pixels
    MASK_COUNTS = mask_counts
    MASK_KEYS = mask_keys
    MASK_SHAPE = mask_shape


def _assemble_per_annotation(
    mask_keys: list[tuple[str, str]],
    gcc: np.ndarray,
    rcc: np.ndarray,
    bcc: np.ndarray,
    mean_brightness: np.ndarray,
) -> dict:
    """Pack per-mask outputs into the per-annotation dict shape.

    Mirrors the column naming convention:
      - ``coords`` key  →  ``GCC`` / ``RCC`` / ``BCC``
      - ``coords_s{N}`` key  →  ``GCC_s{N}`` / ``RCC_s{N}`` / ``BCC_s{N}``

    ``brightness`` is the mean of (R+G+B) over the last mask processed for
    each annotation (matches the previous behaviour).
    """
    per_annotation: dict = {}
    for (ann_id, key), mean_g, mean_r, mean_b, mean_br in zip(
        mask_keys, gcc, rcc, bcc, mean_brightness
    ):
        if ann_id not in per_annotation:
            per_annotation[ann_id] = {}

        if key == "coords":
            per_annotation[ann_id]["GCC"] = float(mean_g)
            per_annotation[ann_id]["RCC"] = float(mean_r)
            per_annotation[ann_id]["BCC"] = float(mean_b)
            per_annotation[ann_id]["brightness"] = float(mean_br)
        else:
            suffix = key.split("_s")[-1]
            per_annotation[ann_id][f"GCC_s{suffix}"] = float(mean_g)
            per_annotation[ann_id][f"RCC_s{suffix}"] = float(mean_r)
            per_annotation[ann_id][f"BCC_s{suffix}"] = float(mean_b)
            per_annotation[ann_id][f"brightness_s{suffix}"] = float(mean_br)
    return per_annotation


def compute_chromcoord_python(
    img: np.ndarray,
    mask_row_ptr: np.ndarray,
    mask_pixels: np.ndarray,
    mask_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference Python/scipy implementation of the per-image kernel.

    Kept for equivalence testing and as a fallback. Uses the same CSR
    structure as the Rust kernel but reconstructs a scipy ``csr_matrix`` to
    drive the existing SpMV path.
    """
    n_masks = len(mask_counts)
    n_pixels = img.shape[0] * img.shape[1]
    data = np.ones(len(mask_pixels), dtype=np.uint8)
    mask_matrix = csr_matrix(
        (data, mask_pixels, mask_row_ptr),
        shape=(n_masks, n_pixels),
        dtype=np.uint8,
    )
    img_float = img.astype(np.float32)
    b, g, r = cv2.split(img_float)
    brightness = r + g + b
    counts = mask_counts.astype(np.float64)

    mean_g = (mask_matrix @ g.ravel()) / counts
    mean_r = (mask_matrix @ r.ravel()) / counts
    mean_b_arr = (mask_matrix @ b.ravel()) / counts
    mean_br = (mask_matrix @ brightness.ravel()) / counts
    mean_br = mean_br.astype(np.float64, copy=True)
    mean_br[mean_br == 0] = 1e-6
    # "Ratio of means" formulation: mean(channel) / mean(R+G+B), not mean(channel / (R+G+B)) per pixel.
    gcc = mean_g.astype(np.float64) / mean_br
    rcc = mean_r.astype(np.float64) / mean_br
    bcc = mean_b_arr.astype(np.float64) / mean_br
    return gcc, rcc, bcc, mean_br


def process_image(
    image_path: Path | str,
) -> tuple[datetime, dict] | None:
    """Compute GCC and RCC for all masks in one image.

    Uses the module-level mask globals populated by ``worker_init`` and the
    Rust kernel ``_native.compute_chromcoord_rs``.

    Returns ``(datetime, per_annotation_dict)`` on success, or ``None`` on
    failure. The per-annotation dict shape is::

        {
            annotation_id: {
                "GCC": float,
                "RCC": float,
                "BCC": float,
                "GCC_s33": float,   # present when coords_s33 exists
                "RCC_s33": float,
                "BCC_s33": float,
                "brightness": float,
            }
        }
    """
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        if not img.flags["C_CONTIGUOUS"]:
            img = np.ascontiguousarray(img)

        if MASK_SHAPE is not None and img.shape[:2] != MASK_SHAPE:
            shape_key = img.shape[:2]
            if shape_key not in _SHAPE_WARNED:
                _SHAPE_WARNED.add(shape_key)
                H, W = MASK_SHAPE
                H2, W2 = shape_key
                logger.warning(
                    "Image resolution changes during the year (expected %dx%d, got"
                    " %dx%d at %s). Images with a different shape are ignored —"
                    " look into the input archive to harmonise the image shape if"
                    " you want to use all input observations.",
                    H, W, H2, W2, image_path,
                )
            return None

        gcc, rcc, bcc, mean_br = _native.compute_chromcoord_rs(
            img, MASK_ROW_PTR, MASK_PIXELS, MASK_COUNTS
        )

        per_annotation = _assemble_per_annotation(MASK_KEYS, gcc, rcc, bcc, mean_br)
        image_dt = parse_datetime_from_filepath(Path(image_path))
        return (image_dt, dict(per_annotation))

    except Exception:
        logger.warning("process_image failed for %s", image_path, exc_info=True)
        return None
