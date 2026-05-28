"""Polygon-overlay plotting utility for visual QA (stage-1 'plot' mode)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon as shpPoly

from swissphenocam.extraction.polygons import scale_polygon_inw_offset


# ---------------------------------------------------------------------------
# Plotting convention — category → hex colour
# ---------------------------------------------------------------------------

_GRAPHIC_STYLE: dict[str, dict] = {
    "individual tree": {"color": "#FF34FF", "short": "I"},
    "group of trees": {"color": "#1CE6FF", "short": "Gp"},
    "grassland": {"color": "#008941", "short": "Gs"},
    "cropland": {"color": "#F7E24F", "short": "C"},
    "shrubs": {"color": "#72FF00", "short": "S"},
    "other": {"color": "#FF4A46", "short": "O"},
}


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    bgr = tuple(int(hex_color[i : i + 2], 16) for i in (4, 2, 0))
    return bgr  # type: ignore[return-value]


def _shade_polygon(
    image: np.ndarray,
    coords,
    color: tuple[int, int, int],
    alpha: float = 0.5,
) -> None:
    """Fill a polygon with a semi-transparent colour overlay (in-place)."""
    overlay = image.copy()
    pts = np.array(list(coords), dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_polygons_on_image(
    image_path: Path,
    polygons: Path | dict,
    out_path: Path | None = None,
    return_im: bool = False,
    uid_mapping: dict[str, str] | None = None,
) -> np.ndarray | None:
    """Draw polygon overlays on *image_path* using cv2.

    Parameters
    ----------
    image_path:
        Path to the source image.
    polygons:
        Either a ``Path`` to a polygon JSON file, or a pre-loaded ``dict``
        mapping annotation IDs to annotation dicts.  When a ``dict`` is
        supplied directly some metadata (filename annotations) is unavailable
        and the image-info text label is skipped (``no_info`` fallback).
    out_path:
        If provided, write the annotated image to this path (JPEG, quality 50).
        Parent directories are created automatically.
    return_im:
        If ``True``, return the annotated image array; otherwise return
        ``None``.

    Returns
    -------
    np.ndarray | None
    """
    # Font / drawing parameters — keep identical to source
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    thickness = 6

    # ------------------------------------------------------------------
    # Load polygon dict
    # ------------------------------------------------------------------
    no_info = False
    if isinstance(polygons, Path):
        with open(polygons) as fh:
            polygon_dict: dict = json.loads(fh.read())
    else:
        polygon_dict = polygons
        no_info = True

    # ------------------------------------------------------------------
    # Parse annotation lists
    # ------------------------------------------------------------------
    ids: list[str] = []
    classes: list[str] = []
    coords: list[list] = []
    names: list[str] = []

    for labelboxid, annotation in polygon_dict.items():
        coords.append(annotation["coords"])
        classes.append(annotation["category"])
        ids.append(labelboxid)
        if uid_mapping is not None:
            names.append(uid_mapping.get(labelboxid))
        else:
            names.append(annotation["name"])

    # ------------------------------------------------------------------
    # Load image
    # ------------------------------------------------------------------
    image = cv2.imread(str(image_path))
    if image is None:
        print("Corrupt image !")
        print(image_path)
        return None

    h, w, _ = image.shape

    # Optional image-info text (requires polygon Path for filename)
    if not no_info:
        img_info = (
            f"Image {image_path.name.split('_')[2]} / "
            f"Annotation {polygons.name.split('_')[2]}"  # type: ignore[union-attr]
        )
        cv2.putText(
            image,
            img_info,
            (w // 2, int(0.99 * h)),
            font,
            2 * font_scale,
            (255, 255, 255),
            3 * thickness,
            lineType=cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    # Draw polygons
    # ------------------------------------------------------------------
    for polygon_class in np.unique(classes):
        indices = np.where(np.array(classes) == polygon_class)[0]
        for idx in indices:
            label = classes[idx]
            poly = coords[idx]
            shortname = names[idx]
            if label not in _GRAPHIC_STYLE:
                print(f"Passing object of class {label}")
                continue

            color = _hex_to_bgr(_GRAPHIC_STYLE[label]["color"])
            polygon_points = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                image,
                [polygon_points],
                isClosed=True,
                color=color,
                thickness=thickness,
            )

            shrunk = scale_polygon_inw_offset(shpPoly(poly), factor=0.333333)
            if not shrunk.is_empty and shrunk.is_valid:
                _shade_polygon(image, shrunk.exterior.coords, color, alpha=0.5)

            center_x, center_y = shpPoly(poly).representative_point().coords[0]
            center_x, center_y = int(center_x), int(center_y)
            if shortname is not None:
                cv2.putText(
                    image,
                    shortname,
                    (center_x, center_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    lineType=cv2.LINE_AA,
                )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    if out_path is not None:
        out_path = Path(out_path)
        os.makedirs(out_path.parent, exist_ok=True)
        cv2.imwrite(str(out_path), image, [cv2.IMWRITE_JPEG_QUALITY, 50])
        print(out_path)

    if return_im:
        return image
    return None
