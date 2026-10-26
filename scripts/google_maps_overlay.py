#!/usr/bin/env python3
"""Build a local Google Maps viewer for a georeferenced TIFF.

The viewer is deliberately a visual quality-control tool: it overlays the
GeoTIFF on Google Maps at zoom levels 12 and 13 without downloading, caching,
or analysing Google imagery.  This keeps Google attribution visible and avoids
using Maps content to derive a new geospatial dataset.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path

try:
    from osgeo import gdal, osr
except ImportError as exc:  # pragma: no cover - depends on local GDAL install
    raise SystemExit(
        "GDAL Python bindings are required. Verify with "
        "`python -c \"from osgeo import gdal\"`."
    ) from exc

gdal.UseExceptions()


HTML_TEMPLATE = r"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>12U — проверка геопривязки</title>
  <style>
    html, body, #map { height: 100%; margin: 0; }
    body { font: 14px/1.4 system-ui, sans-serif; color: #15202b; }
    .panel {
      position: absolute; z-index: 5; top: 12px; left: 12px; width: min(390px, calc(100% - 48px));
      padding: 12px 14px; border-radius: 10px; background: rgba(255,255,255,.96);
      box-shadow: 0 2px 12px rgba(0,0,0,.25);
    }
    .row { display: flex; align-items: center; gap: 8px; margin-top: 8px; }
    input[type=password] { flex: 1; min-width: 80px; padding: 7px; }
    input[type=range] { flex: 1; }
    button { padding: 7px 10px; cursor: pointer; }
    .muted { color: #52606d; font-size: 12px; }
    .error { color: #b42318; margin-top: 8px; }
    #tools { display: none; }
    code { font-size: 12px; }
  </style>
</head>
<body>
  <div id="map"></div>
  <section class="panel">
    <strong>12U: визуальная проверка привязки</strong>
    <div class="muted">Google Maps используется только как интерактивная подложка. Ключ не сохраняется в файлах.</div>
    <div id="login" class="row">
      <input id="apiKey" type="password" autocomplete="off" placeholder="Google Maps JavaScript API key">
      <button id="loadMap">Открыть</button>
    </div>
    <div id="tools">
      <div class="row">
        <label for="opacity">Прозрачность TIFF</label>
        <input id="opacity" type="range" min="0" max="1" step="0.05" value="0.55">
        <output id="opacityValue">55%</output>
      </div>
      <div class="row">
        <button data-zoom="12">Zoom 12</button>
        <button data-zoom="13">Zoom 13</button>
        <button id="toggleOverlay">Скрыть TIFF</button>
      </div>
      <div class="muted" style="margin-top:8px">
        Режим карты переключается стандартной кнопкой Google. Для дорог удобен Hybrid или Map.
        Границы: <code>__BOUNDS_TEXT__</code>
      </div>
    </div>
    <div id="error" class="error"></div>
  </section>
  <script>
    const rasterBounds = __BOUNDS_JSON__;
    const center = {
      lat: (rasterBounds.north + rasterBounds.south) / 2,
      lng: (rasterBounds.east + rasterBounds.west) / 2
    };
    let overlay;
    let overlayVisible = true;

    function initMap() {
      const map = new google.maps.Map(document.getElementById('map'), {
        center, zoom: 12, minZoom: 12, maxZoom: 13,
        mapTypeId: 'hybrid',
        mapTypeControlOptions: {
          mapTypeIds: ['roadmap', 'satellite', 'hybrid']
        },
        streetViewControl: false
      });
      overlay = new google.maps.GroundOverlay('__OVERLAY_URL__', rasterBounds, {
        clickable: false, opacity: 0.55
      });
      overlay.setMap(map);
      document.getElementById('login').style.display = 'none';
      document.getElementById('tools').style.display = 'block';

      document.querySelectorAll('[data-zoom]').forEach(button => {
        button.addEventListener('click', () => map.setZoom(Number(button.dataset.zoom)));
      });
      const opacity = document.getElementById('opacity');
      opacity.addEventListener('input', () => {
        overlay.setOpacity(Number(opacity.value));
        document.getElementById('opacityValue').value = `${Math.round(opacity.value * 100)}%`;
      });
      document.getElementById('toggleOverlay').addEventListener('click', event => {
        overlayVisible = !overlayVisible;
        overlay.setMap(overlayVisible ? map : null);
        event.currentTarget.textContent = overlayVisible ? 'Скрыть TIFF' : 'Показать TIFF';
      });
    }

    function showError(message) {
      document.getElementById('error').textContent = message;
    }

    document.getElementById('loadMap').addEventListener('click', () => {
      const key = document.getElementById('apiKey').value.trim();
      if (!key) { showError('Введите ключ Maps JavaScript API.'); return; }
      showError('');
      window.gm_authFailure = () => showError('Google отклонил ключ: проверьте API, billing и ограничения referrer.');
      const script = document.createElement('script');
      script.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(key)}&callback=initMap&v=weekly&loading=async`;
      script.async = true;
      script.onerror = () => showError('Не удалось загрузить Google Maps JavaScript API.');
      document.head.appendChild(script);
    });
  </script>
</body>
</html>
"""

OSM_TEMPLATE = r"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>12U — проверка на OpenStreetMap</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
          integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
  <style>
    html, body, #map { height: 100%; margin: 0; }
    body { font: 14px/1.4 system-ui, sans-serif; }
    .panel {
      position: absolute; z-index: 1000; top: 12px; left: 55px;
      width: min(330px, calc(100% - 90px)); padding: 12px 14px;
      border-radius: 10px; background: rgba(255,255,255,.96);
      box-shadow: 0 2px 12px rgba(0,0,0,.25);
    }
    .row { display: flex; align-items: center; gap: 8px; margin-top: 8px; }
    input[type=range] { flex: 1; }
    button { padding: 7px 10px; cursor: pointer; }
    .muted { color: #52606d; font-size: 12px; }
  </style>
</head>
<body>
  <div id="map"></div>
  <section class="panel">
    <strong>12U на карте дорог OpenStreetMap</strong>
    <div class="muted">Ключ API не нужен. Наложение показывает текущую привязку TIFF.</div>
    <div class="row">
      <label for="opacity">Прозрачность TIFF</label>
      <input id="opacity" type="range" min="0" max="1" step="0.05" value="0.55">
      <output id="opacityValue">55%</output>
    </div>
    <div class="row">
      <button data-zoom="12">Zoom 12</button>
      <button data-zoom="13">Zoom 13</button>
      <button id="toggleOverlay">Скрыть TIFF</button>
    </div>
    <div class="muted" style="margin-top:8px">Границы: __BOUNDS_TEXT__</div>
  </section>
  <script>
    const rasterBounds = __BOUNDS_JSON__;
    const bounds = [[rasterBounds.south, rasterBounds.west],
                    [rasterBounds.north, rasterBounds.east]];
    const map = L.map('map', {minZoom: 12, maxZoom: 13});
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      minZoom: 12, maxZoom: 13,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>'
    }).addTo(map);
    const overlay = L.imageOverlay('__OVERLAY_URL__', bounds, {
      opacity: 0.55, alt: 'Геопривязанный снимок 12U'
    }).addTo(map);
    map.setView([(rasterBounds.north + rasterBounds.south) / 2,
                 (rasterBounds.east + rasterBounds.west) / 2], 12);
    document.querySelectorAll('[data-zoom]').forEach(button => {
      button.addEventListener('click', () => map.setZoom(Number(button.dataset.zoom)));
    });
    const opacity = document.getElementById('opacity');
    opacity.addEventListener('input', () => {
      overlay.setOpacity(Number(opacity.value));
      document.getElementById('opacityValue').value = `${Math.round(opacity.value * 100)}%`;
    });
    document.getElementById('toggleOverlay').addEventListener('click', event => {
      const visible = map.hasLayer(overlay);
      if (visible) map.removeLayer(overlay);
      else overlay.addTo(map);
      event.currentTarget.textContent = visible ? 'Показать TIFF' : 'Скрыть TIFF';
    });
  </script>
</body>
</html>
"""


def _absolute(path: Path) -> Path:
    # resolve() can expose Unicode-path issues in a few Windows GDAL builds.
    return Path(os.path.abspath(path))


def _warp_to_png(source: Path, output: Path) -> dict[str, float]:
    dataset = gdal.Open(str(source), gdal.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"GDAL could not open {source}")
    if not dataset.GetProjectionRef():
        raise ValueError("the TIFF has no coordinate reference system")

    has_alpha = any(
        dataset.GetRasterBand(index).GetColorInterpretation() == gdal.GCI_AlphaBand
        for index in range(1, dataset.RasterCount + 1)
    )
    options = gdal.WarpOptions(
        format="MEM",
        dstSRS="EPSG:3857",
        resampleAlg="bilinear",
        srcAlpha=has_alpha,
        dstAlpha=True,
        srcNodata=None if has_alpha else 0,
        multithread=True,
    )
    warped = gdal.Warp("", dataset, options=options)
    if warped is None:
        raise RuntimeError("GDAL could not reproject the TIFF to EPSG:3857")

    transform = warped.GetGeoTransform()
    west_m = transform[0]
    north_m = transform[3]
    east_m = west_m + transform[1] * warped.RasterXSize
    south_m = north_m + transform[5] * warped.RasterYSize
    mercator = osr.SpatialReference()
    mercator.ImportFromEPSG(3857)
    geographic = osr.SpatialReference()
    geographic.ImportFromEPSG(4326)
    geographic.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    to_geographic = osr.CoordinateTransformation(mercator, geographic)
    west, south, _ = to_geographic.TransformPoint(west_m, south_m)
    east, north, _ = to_geographic.TransformPoint(east_m, north_m)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = gdal.Translate(
        str(output), warped, format="PNG", creationOptions=["ZLEVEL=6"]
    )
    if result is None:
        raise RuntimeError(f"GDAL could not create {output}")
    result = None
    warped = None
    dataset = None
    return {"north": north, "south": south, "east": east, "west": west}


def build_viewer(source: Path, output_dir: Path) -> tuple[Path, Path]:
    source = _absolute(source)
    output_dir = _absolute(output_dir)
    if not source.is_file():
        raise FileNotFoundError(source)
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay_path = output_dir / "12u-overlay.png"
    bounds = _warp_to_png(source, overlay_path)
    bounds_text = (
        f"N {bounds['north']:.6f}, S {bounds['south']:.6f}, "
        f"E {bounds['east']:.6f}, W {bounds['west']:.6f}"
    )
    def render(template: str) -> str:
        return (
            template.replace("__BOUNDS_JSON__", json.dumps(bounds))
            .replace("__BOUNDS_TEXT__", html.escape(bounds_text))
            .replace("__OVERLAY_URL__", overlay_path.name)
        )

    index_path = output_dir / "index.html"
    index_path.write_text(render(HTML_TEMPLATE), encoding="utf-8")
    (output_dir / "osm.html").write_text(render(OSM_TEMPLATE), encoding="utf-8")
    metadata = {
        "source": str(source),
        "overlay": overlay_path.name,
        "bounds_wgs84": bounds,
        "allowed_zoom": [12, 13],
        "purpose": "visual georeferencing quality control only",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return index_path, overlay_path


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("tiff", type=Path, help="input georeferenced TIFF")
    argument_parser.add_argument(
        "--output-dir", type=Path, default=Path("results/12u_google_overlay"),
        help="viewer output directory (default: results/12u_google_overlay)",
    )
    return argument_parser


def main() -> int:
    args = parser().parse_args()
    try:
        page, overlay = build_viewer(args.tiff, args.output_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Created: {page}")
    print(f"Overlay: {overlay}")
    print(f"Serve with: python -m http.server 8000 --directory \"{page.parent}\"")
    print("Open without a key: http://localhost:8000/osm.html")
    print("Open with a Google key: http://localhost:8000/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
