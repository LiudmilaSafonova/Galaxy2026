#!/usr/bin/env python3
"""Highlight likely water in a grayscale 12U GeoTIFF and retain its georeferencing.

This is an image-based heuristic for the 112-5-2025 Saint Petersburg frame.
Google map content is not used to classify pixels or derive coordinates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from osgeo import gdal
from scipy import ndimage as ndi
from skimage.morphology import convex_hull_image

gdal.UseExceptions()


def classify(gray: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find broad smooth dark regions and connected urban waterways."""
    footprint = convex_hull_image((gray > 3) & (alpha > 0)) & (alpha > 0)
    smooth = ndi.median_filter(gray, size=5)
    values = gray.astype(np.float32)
    mean = ndi.uniform_filter(values, size=7)
    local_std = np.sqrt(np.maximum(0, ndi.uniform_filter(values * values, size=7) - mean * mean))
    candidates = footprint & (gray > 3) & (
        ((smooth < 85) & (local_std < 18))
        | ((smooth < 42) & (local_std < 35))
    )

    labels, count = ndi.label(candidates)
    if count == 0:
        raise ValueError("no water candidates were found")
    sizes = np.bincount(labels.ravel())
    boxes = ndi.find_objects(labels)
    # Exclude the large dark park at the top of this particular frame. Keep
    # the bay, the Neva branches and other sizeable water regions below it.
    min_bottom = int(0.34 * gray.shape[0])
    accepted = np.zeros(count + 1, dtype=bool)
    for label in range(1, count + 1):
        box = boxes[label - 1]
        if box is not None and sizes[label] >= 2000 and box[0].stop > min_bottom:
            accepted[label] = True
    water = accepted[labels]
    # Remove isolated single-pixel artifacts while preserving narrow channels.
    water &= footprint
    return water, footprint


def create_outputs(source: Path, output_dir: Path) -> dict[str, object]:
    dataset = gdal.Open(str(source), gdal.GA_ReadOnly)
    if dataset is None or dataset.RasterCount < 2 or not dataset.GetProjectionRef():
        raise ValueError("input must be a georeferenced grayscale TIFF with alpha")
    gray = dataset.GetRasterBand(1).ReadAsArray()
    alpha = dataset.GetRasterBand(2).ReadAsArray()
    water, footprint = classify(gray, alpha)

    base = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)
    tint = np.array([0.0, 125.0, 255.0], dtype=np.float32)
    base[water] = base[water] * 0.35 + tint * 0.65
    edge = water & ~ndi.binary_erosion(water, iterations=2)
    base[edge] = [0, 220, 255]
    rgba = np.empty((*gray.shape, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.clip(base, 0, 255).astype(np.uint8)
    rgba[:, :, 3] = np.where(footprint, 255, 0).astype(np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    geo_tiff = output_dir / "water_highlight_georeferenced.tif"
    driver = gdal.GetDriverByName("GTiff")
    output = driver.Create(
        str(geo_tiff), dataset.RasterXSize, dataset.RasterYSize, 4, gdal.GDT_Byte,
        options=["TILED=YES", "COMPRESS=DEFLATE"],
    )
    if output is None:
        raise RuntimeError("could not create the output GeoTIFF")
    output.SetGeoTransform(dataset.GetGeoTransform())
    output.SetProjection(dataset.GetProjectionRef())
    interpretations = (gdal.GCI_RedBand, gdal.GCI_GreenBand, gdal.GCI_BlueBand, gdal.GCI_AlphaBand)
    for index, interpretation in enumerate(interpretations, start=1):
        band = output.GetRasterBand(index)
        band.WriteArray(rgba[:, :, index - 1])
        band.SetColorInterpretation(interpretation)
    output.FlushCache()
    output = None

    png = output_dir / "water_highlight.png"
    translated = gdal.Translate(str(png), str(geo_tiff), format="PNG")
    if translated is None:
        raise RuntimeError("could not create the PNG")
    translated = None
    summary: dict[str, object] = {
        "source": str(source),
        "png": png.name,
        "geotiff": geo_tiff.name,
        "water_pixels": int(water.sum()),
        "footprint_pixels": int(footprint.sum()),
        "water_fraction_of_footprint": float(water.sum() / footprint.sum()),
        "georeferencing": "copied from the input GeoTIFF; not verified against Google Maps",
        "method": "grayscale intensity and local texture, minimum region 2000 pixels",
    }
    (output_dir / "water_metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dataset = None
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="source 12U grayscale GeoTIFF")
    parser.add_argument("--output-dir", type=Path, default=Path("results/12u_water_112-5-2025"))
    args = parser.parse_args()
    try:
        summary = create_outputs(args.source, args.output_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
