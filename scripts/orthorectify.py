#!/usr/bin/env python3
"""Approximate orthorectification of a 12U image from its telemetry.

The image is modelled as a central projection.  A regular grid of image
points is back-projected through a pinhole camera and intersected with a
WGS-84 ellipsoid (optionally raised by a constant ground height).  The
resulting ground control points are then resampled by GDAL into a GeoTIFF.

The telemetry does not contain camera calibration or a terrain model, so a
horizontal field of view must be supplied.  The output is not survey-grade
without calibrated intrinsics and a DEM.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Iterable

try:
    from osgeo import gdal, osr
except ImportError as exc:  # pragma: no cover - depends on local GDAL install
    raise SystemExit(
        "GDAL Python bindings are required: install a GDAL build matching "
        "your Python interpreter, then verify with `python -c \"from osgeo "
        "import gdal\"`."
    ) from exc

gdal.UseExceptions()

WGS84_A = 6_378_137.0
WGS84_F = 1.0 / 298.257_223_563


def _number(text: str, label: str) -> float:
    match = re.search(rf"(?im)^\s*{re.escape(label)}:\s*([-+]?\d+(?:[.,]\d+)?)", text)
    if not match:
        raise ValueError(f"Telemetry field {label!r} was not found")
    return float(match.group(1).replace(",", "."))


def read_telemetry(path: Path) -> dict[str, float]:
    """Read position and attitude from a 12U ``*.telemetry.txt`` file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "latitude": _number(text, "Latitude"),
        "longitude": _number(text, "Longitude"),
        "height": _number(text, "Height"),
        "yaw": _number(text, "Yaw"),
        "pitch": _number(text, "Pitch"),
        "roll": _number(text, "Roll"),
    }


def paired_telemetry(image: Path) -> Path:
    candidate = image.with_suffix(".telemetry.txt")
    if not candidate.exists():
        raise FileNotFoundError(f"Paired telemetry file not found: {candidate}")
    return candidate


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _matvec(a: list[list[float]], v: Iterable[float]) -> tuple[float, float, float]:
    values = tuple(v)
    return tuple(sum(row[k] * values[k] for k in range(3)) for row in a)  # type: ignore[return-value]


def _attitude_ned(yaw: float, pitch: float, roll: float) -> list[list[float]]:
    """Body-to-NED rotation, aerospace Z-Y-X convention, angles in degrees."""
    y, p, r = map(math.radians, (yaw, pitch, roll))
    rz = [[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]]
    ry = [[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]]
    rx = [[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]]
    return _matmul(_matmul(rz, ry), rx)


def _geodetic_to_ecef(lat_deg: float, lon_deg: float, height: float) -> tuple[float, float, float]:
    lat, lon = map(math.radians, (lat_deg, lon_deg))
    e2 = WGS84_F * (2.0 - WGS84_F)
    n = WGS84_A / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
    return ((n + height) * math.cos(lat) * math.cos(lon),
            (n + height) * math.cos(lat) * math.sin(lon),
            (n * (1.0 - e2) + height) * math.sin(lat))


def _ned_to_ecef(lat_deg: float, lon_deg: float) -> list[list[float]]:
    lat, lon = map(math.radians, (lat_deg, lon_deg))
    # Columns are local north, east and down unit vectors in ECEF.
    return [[-math.sin(lat) * math.cos(lon), -math.sin(lon), -math.cos(lat) * math.cos(lon)],
            [-math.sin(lat) * math.sin(lon), math.cos(lon), -math.cos(lat) * math.sin(lon)],
            [math.cos(lat), 0.0, -math.sin(lat)]]


def _ecef_to_geodetic(x: float, y: float, z: float) -> tuple[float, float, float]:
    b = WGS84_A * (1.0 - WGS84_F)
    e2 = WGS84_F * (2.0 - WGS84_F)
    ep2 = (WGS84_A**2 - b**2) / b**2
    p = math.hypot(x, y)
    theta = math.atan2(z * WGS84_A, p * b)
    lon = math.atan2(y, x)
    lat = math.atan2(z + ep2 * b * math.sin(theta) ** 3,
                     p - e2 * WGS84_A * math.cos(theta) ** 3)
    n = WGS84_A / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
    h = p / math.cos(lat) - n
    return math.degrees(lat), math.degrees(lon), h


def _ray_ellipsoid(origin: tuple[float, float, float], direction: tuple[float, float, float],
                   ground_height: float) -> tuple[float, float, float]:
    a = WGS84_A + ground_height
    b = WGS84_A * (1.0 - WGS84_F) + ground_height
    ox, oy, oz = origin
    dx, dy, dz = direction
    qa = (dx * dx + dy * dy) / (a * a) + dz * dz / (b * b)
    qb = 2.0 * ((ox * dx + oy * dy) / (a * a) + oz * dz / (b * b))
    qc = (ox * ox + oy * oy) / (a * a) + oz * oz / (b * b) - 1.0
    disc = qb * qb - 4.0 * qa * qc
    if disc < 0:
        raise ValueError("A camera ray does not intersect Earth; check attitude/FOV conventions")
    roots = [t for t in ((-qb - math.sqrt(disc)) / (2.0 * qa),
                         (-qb + math.sqrt(disc)) / (2.0 * qa)) if t > 0]
    if not roots:
        raise ValueError("Earth intersection is behind the camera")
    t = min(roots)
    return ox + t * dx, oy + t * dy, oz + t * dz


def make_gcps(width: int, height: int, telemetry: dict[str, float], hfov_deg: float,
              grid_size: int, ground_height: float) -> list[gdal.GCP]:
    if not 0.1 < hfov_deg < 120.0:
        raise ValueError("--hfov-deg must be between 0.1 and 120 degrees")
    if grid_size < 2:
        raise ValueError("--grid-size must be at least 2")
    focal_px = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    body_to_ned = _attitude_ned(telemetry["yaw"], telemetry["pitch"], telemetry["roll"])
    body_to_ecef = _matmul(_ned_to_ecef(telemetry["latitude"], telemetry["longitude"]), body_to_ned)
    origin = _geodetic_to_ecef(telemetry["latitude"], telemetry["longitude"], telemetry["height"])
    gcps: list[gdal.GCP] = []
    for row in range(grid_size):
        v = row * (height - 1) / (grid_size - 1)
        for col in range(grid_size):
            u = col * (width - 1) / (grid_size - 1)
            # Camera/body axes: X forward (image top), Y right, Z down (optical axis).
            ray_body = ((height / 2.0 - v) / focal_px, (u - width / 2.0) / focal_px, 1.0)
            norm = math.sqrt(sum(value * value for value in ray_body))
            ray_ecef = _matvec(body_to_ecef, (value / norm for value in ray_body))
            x, y, z = _ray_ellipsoid(origin, ray_ecef, ground_height)
            lat, lon, h = _ecef_to_geodetic(x, y, z)
            gcps.append(gdal.GCP(lon, lat, h, u, v))
    return gcps


def _utm_srs(latitude: float, longitude: float) -> osr.SpatialReference:
    zone = max(1, min(60, int((longitude + 180.0) / 6.0) + 1))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG((32600 if latitude >= 0 else 32700) + zone)
    return srs


def orthorectify(args: argparse.Namespace) -> None:
    # Do not use Path.resolve() here: on Windows it expands 8.3 aliases back to
    # Unicode names, which some third-party GDAL builds cannot encode.
    image = Path(os.path.abspath(args.image))
    telemetry_path = Path(os.path.abspath(args.telemetry or paired_telemetry(image)))
    metadata = read_telemetry(telemetry_path)
    source = gdal.Open(str(image), gdal.GA_ReadOnly)
    if source is None:
        raise RuntimeError(f"GDAL could not open {image}")
    gcps = make_gcps(source.RasterXSize, source.RasterYSize, metadata,
                     args.hfov_deg, args.grid_size, args.ground_height_m)

    wgs84 = osr.SpatialReference()
    wgs84.SetWellKnownGeogCS("WGS84")
    output_srs = _utm_srs(metadata["latitude"], metadata["longitude"])
    output = Path(os.path.abspath(args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    vrt_path = f"/vsimem/{image.stem}-gcps.vrt"
    try:
        vrt = gdal.Translate(vrt_path, source, format="VRT", GCPs=gcps,
                             outputSRS=wgs84.ExportToWkt())
        vrt = None
        options = gdal.WarpOptions(
            format="GTiff", dstSRS=output_srs.ExportToWkt(), tps=True,
            xRes=args.resolution_m, yRes=args.resolution_m,
            resampleAlg=args.resampling, dstAlpha=True,
            creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
            multithread=True,
        )
        result = gdal.Warp(str(output), vrt_path, options=options)
        if result is None:
            raise RuntimeError("gdal.Warp failed")
        print(f"Created: {output}")
        print(f"Size: {result.RasterXSize} x {result.RasterYSize}; CRS: {output_srs.GetName()}")
        result = None
    finally:
        gdal.Unlink(vrt_path)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", type=Path, help="source JPG/PNG image")
    p.add_argument("output", type=Path, help="output orthorectified GeoTIFF")
    p.add_argument("--telemetry", type=Path, help="telemetry TXT (default: paired file)")
    p.add_argument("--hfov-deg", type=float, required=True,
                   help="calibrated horizontal camera field of view, degrees")
    p.add_argument("--ground-height-m", type=float, default=0.0,
                   help="constant WGS-84 ellipsoidal ground height (default: 0)")
    p.add_argument("--grid-size", type=int, default=9, help="GCP grid dimension (default: 9)")
    p.add_argument("--resolution-m", type=float, default=30.0,
                   help="output pixel size in metres (default: 30)")
    p.add_argument("--resampling", choices=("near", "bilinear", "cubic", "cubicspline", "lanczos"),
                   default="cubic", help="GDAL resampling method (default: cubic)")
    return p


def main() -> int:
    args = parser().parse_args()
    try:
        orthorectify(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
