#!/usr/bin/env python3
"""Extract textured urban fabric from the 112-5-2025 grayscale 12U frame.

The outputs are image-based estimates. They inherit the TIFF's unverified
georeferencing and do not use Google imagery or map tiles as training data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from osgeo import gdal
from scipy import ndimage as ndi

from highlight_water_12u import classify as classify_water

gdal.UseExceptions()


def classify_urban(gray: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    water, footprint = classify_water(gray, alpha)
    values = gray.astype(np.float32)
    mean = ndi.uniform_filter(values, size=11)
    local_std = np.sqrt(np.maximum(0, ndi.uniform_filter(values * values, size=11) - mean * mean))
    textured = (local_std > 22) & (gray > 8) & footprint & ~water
    density = ndi.uniform_filter(textured.astype(np.float32), size=31)
    candidates = (density > 0.30) & footprint & ~water

    labels, count = ndi.label(candidates)
    if count == 0:
        raise ValueError("no urban candidates were found")
    sizes = np.bincount(labels.ravel())
    accepted = np.zeros(count + 1, dtype=bool)
    accepted[1:] = sizes[1:] >= 1000
    return accepted[labels], footprint


def save_geotiff(source: gdal.Dataset, image: np.ndarray, path: Path) -> None:
    dataset = gdal.GetDriverByName("GTiff").Create(
        str(path), source.RasterXSize, source.RasterYSize, 4, gdal.GDT_Byte,
        options=["TILED=YES", "COMPRESS=DEFLATE"],
    )
    if dataset is None:
        raise RuntimeError(f"cannot create {path}")
    dataset.SetGeoTransform(source.GetGeoTransform())
    dataset.SetProjection(source.GetProjectionRef())
    interpretations = (gdal.GCI_RedBand, gdal.GCI_GreenBand, gdal.GCI_BlueBand, gdal.GCI_AlphaBand)
    for index, interpretation in enumerate(interpretations, start=1):
        band = dataset.GetRasterBand(index)
        band.WriteArray(image[:, :, index - 1])
        band.SetColorInterpretation(interpretation)
    dataset.FlushCache()
    dataset = None


def save_png(tiff: Path, png: Path) -> None:
    translated = gdal.Translate(str(png), str(tiff), format="PNG")
    if translated is None:
        raise RuntimeError(f"cannot create {png}")
    translated = None


def export(source: Path, output_dir: Path) -> dict[str, object]:
    dataset = gdal.Open(str(source), gdal.GA_ReadOnly)
    if dataset is None or dataset.RasterCount < 2 or not dataset.GetProjectionRef():
        raise ValueError("input must be a georeferenced grayscale TIFF with alpha")
    gray = dataset.GetRasterBand(1).ReadAsArray()
    alpha = dataset.GetRasterBand(2).ReadAsArray()
    urban, footprint = classify_urban(gray, alpha)
    output_dir.mkdir(parents=True, exist_ok=True)

    original = np.repeat(gray[:, :, None], 3, axis=2)
    highlighted = original.copy()
    orange = np.array([255, 112, 0], dtype=np.float32)
    highlighted[urban] = np.clip(
        original[urban].astype(np.float32) * 0.4 + orange * 0.6, 0, 255
    ).astype(np.uint8)
    edge = urban & ~ndi.binary_erosion(urban, iterations=2)
    highlighted[edge] = [255, 220, 0]

    overlay = np.empty((*gray.shape, 4), dtype=np.uint8)
    overlay[:, :, :3] = highlighted
    overlay[:, :, 3] = np.where(footprint, 255, 0).astype(np.uint8)
    overlay_tiff = output_dir / "urban_highlight_georeferenced.tif"
    overlay_png = output_dir / "urban_highlight.png"
    save_geotiff(dataset, overlay, overlay_tiff)
    save_png(overlay_tiff, overlay_png)

    city_only = np.empty((*gray.shape, 4), dtype=np.uint8)
    city_only[:, :, :3] = original
    city_only[:, :, 3] = np.where(urban, 255, 0).astype(np.uint8)
    only_tiff = output_dir / "urban_only_georeferenced.tif"
    only_png = output_dir / "urban_only.png"
    save_geotiff(dataset, city_only, only_tiff)
    save_png(only_tiff, only_png)

    report: dict[str, object] = {
        "source": str(source),
        "urban_pixels": int(urban.sum()),
        "footprint_pixels": int(footprint.sum()),
        "urban_fraction_of_footprint": float(urban.sum() / footprint.sum()),
        "products": [overlay_png.name, only_png.name, overlay_tiff.name, only_tiff.name],
        "method": "local grayscale texture, region density, exclusion of segmented water",
        "limitations": "Built-up areas are approximate; forests, shadows and industrial terrain can affect the mask.",
    }
    (output_dir / "assessment.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dataset = None
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/12u_urban_112-5-2025"))
    args = parser.parse_args()
    try:
        result = export(args.source, args.output_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
