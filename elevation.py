"""
Fetch elevation data for a given local-projected bounding area.

Primary:  Swisstopo SwissALTI3D via STAC API (Switzerland only, 2 m resolution).
          Used automatically when the current geometry projection is LV95
          (i.e. the track's centroid lies inside Switzerland).
Fallback: Copernicus DEM GLO-30 from AWS S3 (global, 30 m, free, cached locally).
"""

import math
from pathlib import Path

import numpy as np
import requests
import rasterio
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.warp import reproject as warp_reproject, Resampling
from pyproj import Transformer

from geometry import is_swiss_area, local_to_wgs84, local_crs

MAX_PIXELS = 2000

STAC_ITEMS_URL = (
    "https://data.geo.admin.ch/api/stac/v0.9/collections/"
    "ch.swisstopo.swissalti3d/items"
)
SWISSALTI3D_CACHE_DIR = Path.home() / ".cache" / "gps-terrain-stl" / "swissalti3d"
MAX_SWISSALTI3D_TILES = 200   # fall back to Copernicus for very large areas

COPERNICUS_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
COPERNICUS_CACHE_DIR = Path.home() / ".cache" / "gps-terrain-stl" / "copernicus"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_elevation(
    center_lv95: tuple[float, float],
    radius_m: float,
    resolution: int,
) -> tuple[np.ndarray, dict]:
    """
    Fetch a square elevation raster centred on center_lv95.

    Tries SwissALTI3D (STAC) first; falls back to Copernicus DEM GLO-30.

    Returns:
        elevation: (H, W) float32, north-up (row 0 = north), NaN = no data
        grid_info: {"x_coords": (W,), "y_coords": (H,)} in LV95 metres
    """
    resolution = min(resolution, MAX_PIXELS)
    # SwissALTI3D only covers Switzerland and requires LV95 coordinates.
    # For tracks anywhere else, go straight to the global Copernicus dataset.
    if is_swiss_area():
        try:
            return _fetch_swissalti3d_stac(center_lv95, radius_m, resolution)
        except Exception as exc:
            print(f"  SwissALTI3D unavailable ({type(exc).__name__}: {exc}), "
                  f"falling back to Copernicus DEM GLO-30…")
    return _fetch_copernicus(center_lv95, radius_m, resolution)


# ---------------------------------------------------------------------------
# SwissALTI3D via STAC API (download + cache individual 1km×1km tiles)
# ---------------------------------------------------------------------------

def _fetch_swissalti3d_stac(center_lv95, radius_m, resolution):
    to_wgs84 = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    ce, cn = center_lv95

    # Bounding box in WGS84 for STAC query
    lons, lats = to_wgs84.transform(
        [ce - radius_m, ce + radius_m],
        [cn - radius_m, cn + radius_m],
    )
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)
    bbox_wgs84 = f"{lon_min},{lat_min},{lon_max},{lat_max}"

    # Collect tile URLs via paginated STAC query
    tile_urls: list[tuple[str, str]] = []   # (url, cache_filename)
    next_url = STAC_ITEMS_URL
    params: dict = {"bbox": bbox_wgs84, "limit": 100}

    while next_url:
        resp = requests.get(next_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for feature in data.get("features", []):
            assets = feature.get("assets", {})
            # Pick 2 m resolution tile (key contains "_2_2056")
            asset_url = None
            for key, asset in assets.items():
                href = asset.get("href", "")
                if "_2_2056" in href and href.endswith(".tif"):
                    asset_url = href
                    break
            if asset_url is None:
                continue
            fname = asset_url.split("/")[-1]
            tile_urls.append((asset_url, fname))

        # Follow pagination
        next_url = None
        params = {}
        for link in data.get("links", []):
            if link.get("rel") == "next":
                next_url = link["href"]
                break

    if not tile_urls:
        raise RuntimeError("No SwissALTI3D tiles found for this area (outside Switzerland?)")

    if len(tile_urls) > MAX_SWISSALTI3D_TILES:
        raise RuntimeError(
            f"Area requires {len(tile_urls)} tiles (limit {MAX_SWISSALTI3D_TILES}); "
            "falling back to Copernicus for large areas"
        )

    # Download and cache tiles
    SWISSALTI3D_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tile_paths = []
    for url, fname in tile_urls:
        cache_path = SWISSALTI3D_CACHE_DIR / fname
        if not cache_path.exists():
            print(f"  Downloading SwissALTI3D tile {fname} …")
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            cache_path.write_bytes(r.content)
            mb = len(r.content) / 1e6
            print(f"  Saved to cache ({mb:.1f} MB)")
        else:
            print(f"  Using cached SwissALTI3D tile {fname}")
        tile_paths.append(cache_path)

    # Merge tiles and resample to target resolution, cropped to LV95 bbox
    bbox_lv95 = (ce - radius_m, cn - radius_m, ce + radius_m, cn + radius_m)
    dst_crs = CRS.from_epsg(2056)
    dst_transform = from_bounds(*bbox_lv95, resolution, resolution)

    datasets = [rasterio.open(p) for p in tile_paths]
    try:
        if len(datasets) == 1:
            raw = datasets[0].read(1).astype(np.float32)
            raw_transform = datasets[0].transform
            src_crs = datasets[0].crs
            raw_nodata = datasets[0].nodata
        else:
            arr, raw_transform = merge(datasets)
            raw = arr[0].astype(np.float32)
            src_crs = datasets[0].crs
            raw_nodata = datasets[0].nodata
    finally:
        for ds in datasets:
            ds.close()

    if raw_nodata is not None:
        raw[raw == raw_nodata] = np.nan
    raw[raw < -500] = np.nan

    out = np.full((resolution, resolution), np.nan, dtype=np.float32)
    warp_reproject(
        source=raw,
        destination=out,
        src_transform=raw_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    x_coords = dst_transform.c + (np.arange(resolution) + 0.5) * dst_transform.a
    y_coords = dst_transform.f + (np.arange(resolution) + 0.5) * dst_transform.e

    return out, {"x_coords": x_coords, "y_coords": y_coords}


# ---------------------------------------------------------------------------
# Copernicus DEM GLO-30 (AWS public dataset)
# ---------------------------------------------------------------------------

def _tile_url(lat: int, lon: int) -> tuple[str, str]:
    """Return (URL, tile_name) for the Copernicus DEM tile whose SW corner is (lat, lon)."""
    lat_s = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
    lon_s = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
    name = f"Copernicus_DSM_COG_10_{lat_s}_00_{lon_s}_00_DEM"
    return f"{COPERNICUS_BASE}/{name}/{name}.tif", name


def _fetch_copernicus(center_lv95, radius_m, resolution):
    ce, cn = center_lv95

    # Local centre + radius → WGS84 bounding box
    lons, lats = local_to_wgs84(
        [ce - radius_m, ce + radius_m],
        [cn - radius_m, cn + radius_m],
    )
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)

    # Download (and cache) needed 1°×1° tiles
    COPERNICUS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tile_paths = []
    for tlat in range(math.floor(lat_min), math.ceil(lat_max)):
        for tlon in range(math.floor(lon_min), math.ceil(lon_max)):
            url, name = _tile_url(tlat, tlon)
            cache_path = COPERNICUS_CACHE_DIR / f"{name}.tif"

            if not cache_path.exists():
                print(f"  Downloading {name} …")
                resp = requests.get(url, timeout=180)
                resp.raise_for_status()
                cache_path.write_bytes(resp.content)
                mb = len(resp.content) / 1e6
                print(f"  Saved to cache ({mb:.1f} MB)")
            else:
                print(f"  Using cached {name}")

            tile_paths.append(cache_path)

    if not tile_paths:
        raise RuntimeError("No DEM tiles found for this area")

    # Open, merge, reproject to LV95
    datasets = [rasterio.open(p) for p in tile_paths]
    try:
        if len(datasets) == 1:
            raw = datasets[0].read(1).astype(np.float32)
            raw_transform = datasets[0].transform
            src_crs = datasets[0].crs
            raw_nodata = datasets[0].nodata
        else:
            arr, raw_transform = merge(datasets)
            raw = arr[0].astype(np.float32)
            src_crs = datasets[0].crs
            raw_nodata = datasets[0].nodata
    finally:
        for ds in datasets:
            ds.close()

    # Mask nodata / voids
    if raw_nodata is not None:
        raw[raw == raw_nodata] = np.nan
    raw[raw < -500] = np.nan    # catch any remaining void flags

    # Reproject raw WGS84 raster → local CRS, cropped to our bbox.
    # rasterio.warp wants a rasterio CRS; pyproj CRS works via WKT.
    dst_crs = CRS.from_wkt(local_crs().to_wkt())
    bbox_lv95 = (ce - radius_m, cn - radius_m, ce + radius_m, cn + radius_m)
    dst_transform = from_bounds(*bbox_lv95, resolution, resolution)

    out = np.full((resolution, resolution), np.nan, dtype=np.float32)
    warp_reproject(
        source=raw,
        destination=out,
        src_transform=raw_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    # Pixel-centre LV95 coordinates matching the dst_transform
    x_coords = dst_transform.c + (np.arange(resolution) + 0.5) * dst_transform.a
    y_coords = dst_transform.f + (np.arange(resolution) + 0.5) * dst_transform.e

    return out, {"x_coords": x_coords, "y_coords": y_coords}
