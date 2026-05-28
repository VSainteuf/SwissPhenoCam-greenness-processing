#!/usr/bin/env python
"""Stage 1 driver: extract raw per-observation GCC/RCC from webcam images.

Reads a JSON config (see configs/extraction.example.json), walks the image
dataset, loads the corresponding polygon ROI files, and writes raw greenness
CSVs into the output dataset root.
"""

# Stage 1

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
from tqdm import tqdm

from swissphenocam.config import load_extraction_config, load_uid_mapping
from swissphenocam.extraction import dataset, indices, io, plotting
from swissphenocam.extraction.polygons import (
    add_areas,
    add_shrunk_segments,
    filter_categories,
    load_polygon_config,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-site-year processing helper
# ---------------------------------------------------------------------------

def process_site_year(
    data_dir: Path,
    polygons_dict: dict,
    output_dir_path: Path,
    num_workers: int,
    uid_mapping: dict[str, str],
) -> list[str]:
    """Process all images for one site-year and write greenness CSVs.

    Parameters
    ----------
    data_dir:
        Year-level image directory
        (``<archive>/<location>/<sublocation>/<year>``).
    polygons_dict:
        Polygon dict after filtering, shrinking, and area annotation.
    output_dir_path:
        Destination directory for CSV + JSON outputs.
    num_workers:
        Number of worker processes in the multiprocessing pool.
    uid_mapping:
        Mapping from annotation_id to unique_id.

    Returns
    -------
    list[str]
        List of annotation_ids missing from uid_mapping.
    """
    data_dir = Path(data_dir)
    image_paths = dataset.list_images(data_dir)
    if not image_paths:
        logger.warning("No images found in %s. Skipping.", data_dir)
        return []

    # Determine canonical image shape as the most common (mode) resolution.
    # Reads a sample of images spread across the year to avoid picking an
    # outlier resolution that only appears at the start.
    max_sample = 25
    step = max(1, len(image_paths) // max_sample)
    shape_counts: Counter[tuple[int, int]] = Counter()
    for p in image_paths[::step]:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None and img.size != 0:
            shape_counts[img.shape[:2]] += 1
    if not shape_counts:
        logger.warning("No readable images found in %s. Skipping.", data_dir)
        return []
    H, W = max(shape_counts, key=lambda s: (shape_counts[s], s[0] * s[1]))

    mask_row_ptr, mask_pixels, mask_counts, mask_keys = indices.prepare_masks_vectorized(
        polygons_dict, (H, W)
    )

    with multiprocessing.Pool(
        processes=num_workers,
        initializer=indices.worker_init,
        initargs=(mask_row_ptr, mask_pixels, mask_counts, mask_keys, (H, W)),
    ) as pool:
        results = list(pool.imap(indices.process_image, image_paths))

    # Accumulate (datetime -> metric dict) per annotation_id.
    per_annotation_obs: dict[str, dict] = {}
    for result in results:
        if result is None:
            continue
        obs_dt, per_annotation = result
        for ann_id, values in per_annotation.items():
            if ann_id not in per_annotation_obs:
                per_annotation_obs[ann_id] = {}
            per_annotation_obs[ann_id][obs_dt] = values

    per_polygon_results = {
        ann_id: io.format_obs_df(obs)
        for ann_id, obs in per_annotation_obs.items()
    }

    missing = io.write_site_year(
        output_dir=output_dir_path,
        polygons=polygons_dict,
        per_polygon_results=per_polygon_results,
        data_dir=data_dir,
        uid_mapping=uid_mapping,
    )

    return missing


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def orchestrator(args: argparse.Namespace) -> None:
    """Top-level driver: build tasks from config then dispatch to workers."""
    config = load_extraction_config(args.config)

    config.output_root.mkdir(parents=True, exist_ok=True)
    log_path = config.output_root / "extract.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(file_handler)
    logger.info("Logging to %s", log_path)

    uid_mapping = load_uid_mapping(config.uid_mapping_path)

    # Allow CLI override of plot_dir.
    plot_dir = Path(args.plot_dir) if args.plot_dir is not None else config.plot_dir

    # ---- Build hash→name translation map (optional) ----------------------
    hash_to_name: dict[str, str] | None = None
    if config.camera_hashes_path is not None:
        print(f"Loading camera hash mapping from {config.camera_hashes_path} …")
        with config.camera_hashes_path.open("r") as fh:
            name_to_hash: dict[str, str] = json.load(fh)
        hash_to_name = {v: k for k, v in name_to_hash.items()}

    # ---- Discover site-years from image archive and polygon config --------
    list_of_site_years_data = dataset.get_list_of_site_years(
        config.image_dataset_root, hash_to_name=hash_to_name
    )
    list_of_site_years_polygons, site_year_segments = load_polygon_config(
        config.polygon_config_path,
        config.polygon_files_root,
        name_to_hash=name_to_hash,
    )


    common_site_years = sorted(
        set(list_of_site_years_data).intersection(set(list_of_site_years_polygons))
    )
    logger.info("Site-years in image archive:      %d", len(list_of_site_years_data))
    logger.info("Site-years with polygon config:   %d", len(list_of_site_years_polygons))
    logger.info("Common site-years to process:     %d", len(common_site_years))

    # ---- Polygon pipeline: filter → shrink → areas -----------------------
    # (applied per-site-year inside each mode so we only process what's needed)

    # ---- Mode: plot -------------------------------------------------------
    if args.mode == "plot":
        assert plot_dir is not None, "--plot-dir must be specified in plot mode"
        logger.info("Rendering polygon overlays for %d site-years …", len(common_site_years))
        for site_year in tqdm(common_site_years, desc="plot"):
            segments = site_year_segments[site_year]
            segments = filter_categories(segments, config.categories_kept)
            segments = add_areas(segments)

            # site_year format: "{location}_{sublocation}-{year}"
            site, year = site_year.split("-", 1)
            location, sublocation = site.split("_", 1)
            data_dir = config.image_dataset_root / location / sublocation / year

            images = dataset.list_images(data_dir)
            if not images:
                logger.warning("No images in %s, skipping plot.", data_dir)
                continue
            plot_image = images[int(len(images) * 0.5)]
            out_path = plot_dir / f"{site}-{year}-polygon_plot.jpg"
            plotting.plot_polygons_on_image(plot_image, segments, out_path=out_path, uid_mapping=uid_mapping)
        return

    # ---- Modes: normal and check -----------------------------------------
    tasks: list[tuple[Path, dict, Path, dict[str, str]]] = []

    if args.mode == "normal":
        logger.info("Preparing tasks (normal mode) …")
        for site_year in tqdm(common_site_years, desc="build tasks"):
            segments = site_year_segments[site_year]
            segments = filter_categories(segments, config.categories_kept)
            if not segments:
                continue
            segments = add_shrunk_segments(segments, config.shrink_ratios)
            segments = add_areas(segments)

            site, year = site_year.split("-", 1)
            location, sublocation = site.split("_", 1)
            data_dir = config.image_dataset_root / location / sublocation / year
            output_dir_path = config.output_root / site / year

            if output_dir_path.exists():
                logger.debug("Output already exists, skipping: %s", output_dir_path)
                continue

            tasks.append((data_dir, segments, output_dir_path, uid_mapping))

            if plot_dir is not None:
                images = dataset.list_images(data_dir)
                if images:
                    plot_image = images[int(len(images) * 0.5)]
                    out_path = plot_dir / f"{site}-{year}-polygon_plot.jpg"
                    plotting.plot_polygons_on_image(
                        plot_image, segments, out_path=out_path
                    )

    elif args.mode == "check":
        logger.info("Scanning for missing per-polygon CSVs (check mode) …")
        missing_total = 0
        for site_year in tqdm(common_site_years, desc="scan"):
            site, year = site_year.split("-", 1)
            location, sublocation = site.split("_", 1)
            data_dir = config.image_dataset_root / location / sublocation / year
            output_dir_path = config.output_root / site / year

            segments_to_do: dict = {}
            for polygon_id, poly_info in site_year_segments[site_year].items():
                if poly_info.get("category") not in config.categories_kept:
                    continue
                uid = uid_mapping.get(polygon_id)
                if uid is None:
                    continue
                greenness_file = (
                    config.output_root / site / uid / year
                    / f"{uid}_{year}_raw-cc-data.csv"
                )
                if not greenness_file.exists():
                    segments_to_do[polygon_id] = poly_info
                    missing_total += 1

            if segments_to_do:
                segments_to_do = filter_categories(segments_to_do, config.categories_kept)
                segments_to_do = add_shrunk_segments(segments_to_do, config.shrink_ratios)
                segments_to_do = add_areas(segments_to_do)
                tasks.append(
                    (data_dir, segments_to_do, output_dir_path, uid_mapping)
                )

                if plot_dir is not None:
                    images = dataset.list_images(data_dir)
                    if images:
                        plot_image = images[int(len(images) * 0.6)]
                        out_path = plot_dir / f"{site}-{year}-polygon_plot.jpg"
                        plotting.plot_polygons_on_image(
                            plot_image, segments_to_do, out_path=out_path
                        )

        logger.info(
            "Scan complete. Missing polygon-years: %d across %d site-years.",
            missing_total,
            len(tasks),
        )

    # ---- Run extraction tasks --------------------------------------------
    logger.info(
        "Processing %d site-years with %d site-years in parallel × %d worker(s) per site-year …",
        len(tasks),
        args.site_years_in_parallel,
        args.workers_per_site_year,
    )

    def _run_one(task: tuple[Path, dict, Path, dict[str, str]]) -> tuple[Path, BaseException | None, list[str]]:
        data_dir, segments, output_dir_path, uid_map = task
        try:
            missing = process_site_year(
                data_dir,
                segments,
                output_dir_path,
                args.workers_per_site_year,
                uid_mapping=uid_map,
            )
            return data_dir, None, missing
        except BaseException as exc:  # noqa: BLE001 — isolate failures, log below
            return data_dir, exc, []

    all_missing: dict[str, list[str]] = {}

    with ThreadPoolExecutor(max_workers=args.site_years_in_parallel) as executor:
        futures = [executor.submit(_run_one, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="extract"):
            data_dir, exc, missing = future.result()
            if exc is not None:
                logger.exception("Site-year failed: %s", data_dir, exc_info=exc)
            else:
                # Collect missing UIDs per site-year
                if missing:
                    location = data_dir.parent.parent.name
                    sub_location = data_dir.parent.name
                    year = data_dir.name
                    site = f"{location}_{sub_location}"
                    site_year_key = f"{site}/{year}"
                    all_missing[site_year_key] = missing

    # Write missing UIDs to file if any were found
    if all_missing:
        missing_uids_file = config.output_root / "missing_uids.txt"
        with missing_uids_file.open("w") as fh:
            for site_year_key in sorted(all_missing.keys()):
                for annotation_id in all_missing[site_year_key]:
                    fh.write(f"{site_year_key}/{annotation_id}\n")
        logger.info("Wrote missing UIDs to %s", missing_uids_file)

    logger.info("Processing complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the extraction JSON config (see configs/extraction.example.json).",
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "check", "plot"],
        default="normal",
        help=(
            "normal: process all site-years whose output folder is absent; "
            "check: re-run only site-years with missing per-polygon CSVs; "
            "plot: render polygon overlays on a sample image and exit."
        ),
    )
    parser.add_argument(
        "--workers-per-site-year",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Worker processes within one site-year's image batch "
            "(default: 1)."
        ),
    )
    parser.add_argument(
        "--site-years-in-parallel",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of site-years to process concurrently. Total compute "
            "processes ≈ workers-per-site-year × site-years-in-parallel "
            "(default: 1)."
        ),
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Directory for polygon overlay images; overrides config.plot_dir.",
    )

    args = parser.parse_args()
    if args.workers_per_site_year < 1:
        parser.error("--workers-per-site-year must be >= 1")
    if args.site_years_in_parallel < 1:
        parser.error("--site-years-in-parallel must be >= 1")
    orchestrator(args)


if __name__ == "__main__":
    main()
