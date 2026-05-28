"""Equivalence + unit tests for the stage-1 greenness kernel.

Covers the Rust-vs-Python per-image kernel, the flat mask representation,
and the brightness-zero guard.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from swissphenocam import _native
from swissphenocam.extraction import indices


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _two_polygons_fixture(H: int = 40, W: int = 60) -> dict:
    """A tiny polygon dict with two annotations and one shrunk variant."""
    return {
        "annA": {
            "coords": [(5, 5), (25, 5), (25, 25), (5, 25)],
            "coords_s33": [(10, 10), (20, 10), (20, 20), (10, 20)],
        },
        "annB": {
            "coords": [(30, 15), (55, 15), (55, 35), (30, 35)],
        },
    }


# ---------------------------------------------------------------------------
# prepare_masks_vectorized
# ---------------------------------------------------------------------------

def test_prepare_masks_flat_shapes_and_counts() -> None:
    H, W = 40, 60
    polys = _two_polygons_fixture(H, W)
    row_ptr, pixels, counts, keys = indices.prepare_masks_vectorized(polys, (H, W))

    # 3 masks: A/coords, A/coords_s33, B/coords
    assert len(keys) == 3
    assert row_ptr.shape == (4,)
    assert row_ptr.dtype == np.uint32
    assert pixels.dtype == np.uint32
    assert counts.dtype == np.uint32
    assert row_ptr[0] == 0
    assert row_ptr[-1] == pixels.size
    assert (np.diff(row_ptr) == counts).all()
    # Pixel indices must be in-range.
    assert (pixels < H * W).all()


def test_prepare_masks_drops_out_of_image_polygons() -> None:
    H, W = 20, 30
    polys = {
        "good": {"coords": [(2, 2), (10, 2), (10, 10), (2, 10)]},
        "offscreen": {"coords": [(100, 100), (120, 100), (120, 120), (100, 120)]},
    }
    row_ptr, pixels, counts, keys = indices.prepare_masks_vectorized(polys, (H, W))
    assert [k[0] for k in keys] == ["good"]
    assert counts.size == 1


# ---------------------------------------------------------------------------
# Rust vs Python equivalence
# ---------------------------------------------------------------------------

def test_compute_chromcoord_matches_python() -> None:
    H, W = 40, 60
    polys = _two_polygons_fixture(H, W)
    row_ptr, pixels, counts, _keys = indices.prepare_masks_vectorized(polys, (H, W))

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)

    gcc_rs, rcc_rs, bcc_rs, mb_rs = _native.compute_chromcoord_rs(img, row_ptr, pixels, counts)
    gcc_py, rcc_py, bcc_py, mb_py = indices.compute_chromcoord_python(img, row_ptr, pixels, counts)

    # Python sums over float32, Rust sums over u64. Numerics differ slightly
    # but should agree to float32 precision.
    np.testing.assert_allclose(gcc_rs, gcc_py, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(rcc_rs, rcc_py, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(bcc_rs, bcc_py, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(mb_rs, mb_py, atol=1e-3, rtol=1e-6)


def test_compute_chromcoord_exact_on_constant_image() -> None:
    """With a constant-valued image, every mask gets the same exact ratio."""
    H, W = 20, 30
    polys = _two_polygons_fixture(H, W)
    row_ptr, pixels, counts, _keys = indices.prepare_masks_vectorized(polys, (H, W))

    img = np.empty((H, W, 3), dtype=np.uint8)
    img[..., 0] = 10  # B
    img[..., 1] = 20  # G
    img[..., 2] = 30  # R

    gcc, rcc, bcc, mb = _native.compute_chromcoord_rs(img, row_ptr, pixels, counts)
    np.testing.assert_allclose(gcc, np.full(counts.size, 20.0 / 60.0), atol=1e-12)
    np.testing.assert_allclose(rcc, np.full(counts.size, 30.0 / 60.0), atol=1e-12)
    np.testing.assert_allclose(bcc, np.full(counts.size, 10.0 / 60.0), atol=1e-12)
    np.testing.assert_allclose(mb, np.full(counts.size, 60.0), atol=1e-12)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_brightness_zero_guard() -> None:
    H, W = 20, 30
    polys = {"a": {"coords": [(2, 2), (10, 2), (10, 10), (2, 10)]}}
    row_ptr, pixels, counts, _ = indices.prepare_masks_vectorized(polys, (H, W))
    img = np.zeros((H, W, 3), dtype=np.uint8)

    gcc, rcc, bcc, mb = _native.compute_chromcoord_rs(img, row_ptr, pixels, counts)
    assert mb[0] == pytest.approx(1e-6)
    assert gcc[0] == 0.0
    assert rcc[0] == 0.0
    assert bcc[0] == 0.0

    gcc_py, rcc_py, bcc_py, mb_py = indices.compute_chromcoord_python(img, row_ptr, pixels, counts)
    assert mb_py[0] == pytest.approx(1e-6)
    assert gcc_py[0] == 0.0
    assert rcc_py[0] == 0.0
    assert bcc_py[0] == 0.0


def test_rust_rejects_out_of_range_pixel() -> None:
    H, W = 10, 10
    img = np.zeros((H, W, 3), dtype=np.uint8)
    row_ptr = np.array([0, 1], dtype=np.uint32)
    pixels = np.array([H * W + 5], dtype=np.uint32)
    counts = np.array([1], dtype=np.uint32)
    with pytest.raises(ValueError):
        _native.compute_chromcoord_rs(img, row_ptr, pixels, counts)


def test_rust_rejects_non_bgr_shape() -> None:
    img = np.zeros((10, 10), dtype=np.uint8)  # missing channel dim
    row_ptr = np.array([0, 0], dtype=np.uint32)
    pixels = np.array([], dtype=np.uint32)
    counts = np.array([0], dtype=np.uint32)
    with pytest.raises((ValueError, TypeError)):
        _native.compute_chromcoord_rs(img, row_ptr, pixels, counts)


# ---------------------------------------------------------------------------
# End-to-end: process_image via a synthetic JPEG
# ---------------------------------------------------------------------------

def test_process_image_end_to_end(tmp_path) -> None:
    H, W = 40, 60
    polys = _two_polygons_fixture(H, W)
    row_ptr, pixels, counts, keys = indices.prepare_masks_vectorized(polys, (H, W))

    img = np.empty((H, W, 3), dtype=np.uint8)
    img[..., 0] = 50
    img[..., 1] = 100
    img[..., 2] = 150

    # Filename must parse via parse_datetime_from_filepath (YYYY-MM-DD_HHMM).
    image_path = tmp_path / "cam_2022-10-27_1230.jpg"
    cv2.imwrite(str(image_path), img, [cv2.IMWRITE_JPEG_QUALITY, 100])

    indices.worker_init(row_ptr, pixels, counts, keys, (H, W))
    result = indices.process_image(image_path)
    assert result is not None
    _dt, per_annotation = result

    assert set(per_annotation.keys()) == {"annA", "annB"}
    assert set(per_annotation["annA"].keys()) == {
        "GCC", "RCC", "BCC", "GCC_s33", "RCC_s33", "BCC_s33",
        "brightness", "brightness_s33",
    }
    assert set(per_annotation["annB"].keys()) == {"GCC", "RCC", "BCC", "brightness"}

    for ann in ("annA", "annB"):
        assert per_annotation[ann]["GCC"] == pytest.approx(100.0 / 300.0, abs=5e-3)
        assert per_annotation[ann]["RCC"] == pytest.approx(150.0 / 300.0, abs=5e-3)
        assert per_annotation[ann]["BCC"] == pytest.approx(50.0 / 300.0, abs=5e-3)
    assert per_annotation["annA"]["GCC_s33"] == pytest.approx(100.0 / 300.0, abs=5e-3)
    assert per_annotation["annA"]["RCC_s33"] == pytest.approx(150.0 / 300.0, abs=5e-3)
    assert per_annotation["annA"]["BCC_s33"] == pytest.approx(50.0 / 300.0, abs=5e-3)
