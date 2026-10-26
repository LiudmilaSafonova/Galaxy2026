#!/usr/bin/env python3
"""Estimate local spatial resolution in the centre and at image edges.

Two calculation modes are available:

``camera``
    For a source JPG/PNG.  Neighbouring pixel rays are projected onto a
    WGS-84 ellipsoid using the satellite telemetry and a calibrated horizontal
    field of view (HFOV).  This estimates the ground sampling distance (GSD)
    before orthorectification.

``geotiff``
    For a georeferenced raster.  Pixel centres are transformed through its
    affine geotransform.  This reports the sampling interval stored in the
    product, which is normally constant in an orthorectified image.

The result is a sampling distance in metres per pixel, not an optical MTF
measurement.  Camera mode assumes the same camera/attitude convention as
``orthorectify.py`` and a constant ground height.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Callable, Iterable, Sequence

try:
    from osgeo import gdal, osr
except ImportError as exc:  # pragma: no cover - depends on local GDAL install
    raise SystemExit(
        "GDAL Python bindings are required. Verify the installation with "
        '``python -c "from osgeo import gdal"``.'
    ) from exc

from orthorectify import (  # reuse exactly the orthorectification camera model
    _attitude_ned,
    _geodetic_to_ecef,
    _matmul,
    _matvec,
    _ned_to_ecef,
    _ray_ellipsoid,
    read_telemetry,
)

gdal.UseExceptions()

Point3D = tuple[float, float, float]
GroundMapper = Callable[[float, float], Point3D]

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
REGION_ORDER = ("centre", "top", "right", "bottom", "left", "all_edges")
REGION_LABELS = {
    "centre": "центр",
    "top": "верхний край",
    "right": "правый край",
    "bottom": "нижний край",
    "left": "левый край",
    "all_edges": "все края",
}


@dataclass(frozen=True)
class Sample:
    """Local pixel dimensions at one image position."""

    x_m_per_px: float
    y_m_per_px: float
    effective_m_per_px: float
    axis_angle_deg: float


@dataclass(frozen=True)
class RegionStats:
    """Summary for one image region."""

    samples: int
    x_mean_m_per_px: float
    y_mean_m_per_px: float
    effective_mean_m_per_px: float
    effective_min_m_per_px: float
    effective_max_m_per_px: float
    resolution_class: str


def resolution_class(gsd: float) -> str:
    """Return the practical class boundaries used in the referenced article."""
    if gsd < 1.0:
        return "сверхвысокое"
    if gsd <= 5.0:
        return "высокое"
    if gsd <= 30.0:
        return "среднее"
    return "низкое"


def _subtract(a: Point3D, b: Point3D) -> Point3D:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _norm(vector: Point3D) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _dot(a: Point3D, b: Point3D) -> float:
    return sum(x * y for x, y in zip(a, b))


def local_resolution(map_ground: GroundMapper, u: float, v: float,
                     width: int, height: int) -> Sample:
    """Calculate a local two-dimensional GSD using centred differences."""
    u0, u1 = max(0.0, u - 0.5), min(width - 1.0, u + 0.5)
    v0, v1 = max(0.0, v - 0.5), min(height - 1.0, v + 0.5)
    if u1 <= u0 or v1 <= v0:
        raise ValueError("The raster must be at least 2 x 2 pixels")

    x_vector = _subtract(map_ground(u1, v), map_ground(u0, v))
    y_vector = _subtract(map_ground(u, v1), map_ground(u, v0))
    x_gsd = _norm(x_vector) / (u1 - u0)
    y_gsd = _norm(y_vector) / (v1 - v0)

    cosine = max(-1.0, min(1.0, _dot(x_vector, y_vector) /
                            (_norm(x_vector) * _norm(y_vector))))
    angle = math.degrees(math.acos(cosine))
    # The sine term accounts for non-orthogonal pixel axes.  sqrt(area) is a
    # useful scalar equivalent when horizontal and vertical GSD differ.
    area = x_gsd * y_gsd * abs(math.sin(math.radians(angle)))
    return Sample(x_gsd, y_gsd, math.sqrt(area), angle)


def camera_mapper(width: int, height: int, telemetry: dict[str, float],
                  hfov_deg: float, ground_height: float,
                  camera_axis: str = "+z") -> GroundMapper:
    """Build a source-pixel to WGS-84 ECEF ground mapper."""
    if not 0.1 < hfov_deg < 120.0:
        raise ValueError("--hfov-deg must be between 0.1 and 120 degrees")
    focal_px = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    body_to_ecef = _matmul(
        _ned_to_ecef(telemetry["latitude"], telemetry["longitude"]),
        _attitude_ned(telemetry["yaw"], telemetry["pitch"], telemetry["roll"]),
    )
    origin = _geodetic_to_ecef(
        telemetry["latitude"], telemetry["longitude"], telemetry["height"]
    )

    def map_ground(u: float, v: float) -> Point3D:
        image_top = (height / 2.0 - v) / focal_px
        image_right = (u - width / 2.0) / focal_px
        if camera_axis == "+z":
            ray_body = (image_top, image_right, 1.0)
        elif camera_axis == "-x":
            ray_body = (-1.0, image_right, image_top)
        else:
            raise ValueError(f"Unsupported camera axis: {camera_axis}")
        length = _norm(ray_body)
        ray_ecef = _matvec(body_to_ecef, (value / length for value in ray_body))
        return _ray_ellipsoid(origin, ray_ecef, ground_height)

    return map_ground


def geotiff_mapper(dataset: gdal.Dataset) -> GroundMapper:
    """Build an affine pixel to WGS-84 ECEF mapper for a georaster."""
    transform = dataset.GetGeoTransform(can_return_null=True)
    projection = dataset.GetProjectionRef()
    if transform is None or not projection:
        raise ValueError("the raster has no affine georeferencing and CRS")

    source_srs = osr.SpatialReference()
    source_srs.ImportFromWkt(projection)
    source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ecef_srs = osr.SpatialReference()
    ecef_srs.ImportFromEPSG(4978)
    ecef_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    to_ecef = osr.CoordinateTransformation(source_srs, ecef_srs)

    def map_ground(u: float, v: float) -> Point3D:
        x = transform[0] + u * transform[1] + v * transform[2]
        y = transform[3] + u * transform[4] + v * transform[5]
        return tuple(to_ecef.TransformPoint(x, y, 0.0))  # type: ignore[return-value]

    return map_ground


def sample_regions(map_ground: GroundMapper, width: int, height: int,
                   grid_size: int, centre_fraction: float,
                   edge_fraction: float) -> dict[str, RegionStats]:
    """Sample a regular grid and summarise the central and border strips."""
    if grid_size < 5:
        raise ValueError("--grid-size must be at least 5")
    if not 0.0 < centre_fraction <= 1.0:
        raise ValueError("--centre-fraction must be in (0, 1]")
    if not 0.0 < edge_fraction < 0.5:
        raise ValueError("--edge-fraction must be in (0, 0.5)")

    values: dict[str, list[Sample]] = {name: [] for name in REGION_ORDER}
    centre_min = 0.5 - centre_fraction / 2.0
    centre_max = 0.5 + centre_fraction / 2.0

    for row in range(grid_size):
        ny = row / (grid_size - 1)
        v = ny * (height - 1)
        for col in range(grid_size):
            nx = col / (grid_size - 1)
            u = nx * (width - 1)
            regions: list[str] = []
            if centre_min <= nx <= centre_max and centre_min <= ny <= centre_max:
                regions.append("centre")
            if ny <= edge_fraction:
                regions.append("top")
            if nx >= 1.0 - edge_fraction:
                regions.append("right")
            if ny >= 1.0 - edge_fraction:
                regions.append("bottom")
            if nx <= edge_fraction:
                regions.append("left")
            if regions and any(name != "centre" for name in regions):
                regions.append("all_edges")
            if not regions:
                continue
            sample = local_resolution(map_ground, u, v, width, height)
            for name in regions:
                values[name].append(sample)

    result: dict[str, RegionStats] = {}
    for name, samples in values.items():
        if not samples:
            raise ValueError(
                f"Sampling grid is too sparse for region {name!r}; increase --grid-size"
            )
        effective = [sample.effective_m_per_px for sample in samples]
        result[name] = RegionStats(
            samples=len(samples),
            x_mean_m_per_px=fmean(sample.x_m_per_px for sample in samples),
            y_mean_m_per_px=fmean(sample.y_m_per_px for sample in samples),
            effective_mean_m_per_px=fmean(effective),
            effective_min_m_per_px=min(effective),
            effective_max_m_per_px=max(effective),
            resolution_class=resolution_class(fmean(effective)),
        )
    return result


def find_images(paths: Iterable[Path]) -> list[Path]:
    """Expand input files and directories, preserving a deterministic order."""
    images: list[Path] = []
    for path in paths:
        if path.is_dir():
            images.extend(
                candidate for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        elif path.is_file():
            images.append(path)
        else:
            raise FileNotFoundError(f"Input path not found: {path}")
    return sorted(set(images), key=lambda item: str(item).casefold())


def find_paired_telemetry(image: Path) -> Path:
    """Find telemetry beside a 12U or 3U image with the same base name."""
    candidates = (
        image.with_suffix(".telemetry.txt"),
        image.with_suffix(".telemetry.Can.0.txt"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    expected = " or ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Paired telemetry file not found: {expected}")


def analyse_image(image: Path, args: argparse.Namespace) -> dict[str, object]:
    image = Path(os.path.abspath(image))
    dataset = gdal.Open(str(image), gdal.GA_ReadOnly)
    if dataset is None:
        raise ValueError("GDAL could not open the image")
    width, height = dataset.RasterXSize, dataset.RasterYSize

    if args.mode == "camera":
        telemetry_path = Path(os.path.abspath(
            args.telemetry if args.telemetry is not None else find_paired_telemetry(image)
        ))
        telemetry = read_telemetry(telemetry_path)
        mapper = camera_mapper(width, height, telemetry, args.hfov_deg,
                               args.ground_height_m, args.camera_axis)
        extra: dict[str, object] = {
            "telemetry": str(telemetry_path),
            "hfov_deg": args.hfov_deg,
            "camera_axis": args.camera_axis,
            "ground_height_m": args.ground_height_m,
            "satellite_height_m": telemetry["height"],
        }
    else:
        mapper = geotiff_mapper(dataset)
        extra = {}

    stats = sample_regions(
        mapper, width, height, args.grid_size,
        args.centre_fraction, args.edge_fraction,
    )
    edge_change = (
        stats["all_edges"].effective_mean_m_per_px /
        stats["centre"].effective_mean_m_per_px - 1.0
    ) * 100.0
    return {
        "image": str(image),
        "mode": args.mode,
        "width_px": width,
        "height_px": height,
        **extra,
        "edge_to_centre_change_percent": edge_change,
        "regions": {name: asdict(stats[name]) for name in REGION_ORDER},
    }


def print_result(result: dict[str, object]) -> None:
    print(f"\n{result['image']}")
    print(f"Режим: {result['mode']}; размер: {result['width_px']} x {result['height_px']} px")
    print(f"{'Область':<18} {'GSD X':>10} {'GSD Y':>10} {'GSD эфф.':>10} {'min..max':>19} {'N':>5}")
    regions = result["regions"]
    assert isinstance(regions, dict)
    for name in REGION_ORDER:
        stat = regions[name]
        assert isinstance(stat, dict)
        interval = (f"{stat['effective_min_m_per_px']:.2f}.."
                    f"{stat['effective_max_m_per_px']:.2f}")
        print(
            f"{REGION_LABELS[name]:<18} "
            f"{stat['x_mean_m_per_px']:>10.2f} "
            f"{stat['y_mean_m_per_px']:>10.2f} "
            f"{stat['effective_mean_m_per_px']:>10.2f} "
            f"{interval:>19} "
            f"{stat['samples']:>5}"
        )
    print("Единицы: метры на пиксель; меньшее значение означает более высокую детализацию.")
    change = result["edge_to_centre_change_percent"]
    assert isinstance(change, float)
    direction = "хуже" if change > 0 else "лучше"
    print(f"Среднее GSD краёв на {abs(change):.2f}% {direction}, чем в центре.")


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("images", type=Path, nargs="+",
                        help="image file(s) or directories (searched recursively)")
    common.add_argument("--grid-size", type=int, default=21,
                        help="sampling grid dimension (default: 21)")
    common.add_argument("--centre-fraction", type=float, default=0.20,
                        help="central square side as image fraction (default: 0.20)")
    common.add_argument("--edge-fraction", type=float, default=0.10,
                        help="border strip width as image fraction (default: 0.10)")
    common.add_argument("--json-output", type=Path,
                        help="also save complete results as UTF-8 JSON")

    root = argparse.ArgumentParser(description=__doc__)
    modes = root.add_subparsers(dest="mode", required=True)

    camera = modes.add_parser(
        "camera", parents=[common],
        help="estimate source-image GSD from camera geometry and telemetry",
    )
    camera.add_argument("--hfov-deg", type=float, required=True,
                        help="calibrated horizontal field of view, degrees")
    camera.add_argument("--telemetry", type=Path,
                        help="telemetry for one image (default: paired *.telemetry.txt)")
    camera.add_argument("--ground-height-m", type=float, default=0.0,
                        help="constant WGS-84 ellipsoidal ground height (default: 0)")
    camera.add_argument("--camera-axis", choices=("+z", "-x"), default="+z",
                        help="body optical axis: +z for 12U, -x for 3U (default: +z)")

    modes.add_parser(
        "geotiff", parents=[common],
        help="measure the stored GSD of affine-georeferenced rasters",
    )
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        images = find_images(args.images)
        if not images:
            raise ValueError("No supported images found")
        if args.mode == "camera" and args.telemetry is not None and len(images) != 1:
            raise ValueError("--telemetry can only be used with one image")
        results = [analyse_image(image, args) for image in images]
        for result in results:
            print_result(result)
        if args.json_output is not None:
            output = Path(os.path.abspath(args.json_output))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"\nJSON: {output}")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
