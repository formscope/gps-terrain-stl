"""
Fetch elevation data for a given LV95 bounding area.

Primary:  Swisstopo SwissALTI3D via WCS (Switzerland only, very high res).
Fallback: Copernicus DEM GLO-30 from AWS S3 (global, 30 m, free, cached locally).
"""

import math
from pathlib import Path

import numpy as np
import requests
import rasterio
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.warp import reproject as warp_reproject, Resampling
from pyproj import Transformer

WCS_URL = "https://wcs.geo.admin.ch/"
WCS_COVERAGE = "ch.swisstopo.swissalti3d"
MAX_PIXELS = 2000

COPERNICUS_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
CACHE_DIR = Path.home() / ".cache" / "gps-terrain-stl" / "copernicus"


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

    Tries SwissALTI3D WCS first; falls back to Copernicus DEM GLO-30.

    Returns:
        elevation: (H, W) float32, north-up (row 0 = north), NaN = no data
        grid_info: {"x_coords": (W,), "y_coords": (H,)} in LV95 metres
    """
    resolution = min(resolution, MAX_PIXELS)
    try:
        return _fetch_wcs(center_lv95, radius_m, resolution)
    except Exception as exc:
        print(f"  WCS unavailable ({type(exc).__name__}), "
              f"falling back to Copernicus DEM GLO-30…")
        return _fetch_copernicus(center_lv95, radius_m, resolution)


# ---------------------------------------------------------------------------
# SwissALTI3D via WCS
# ---------------------------------------------------------------------------

def _fetch_wcs(center_lv95, radius_m, resolution):
    ce, cn = center_lv95
    bbox = (ce - radius_m, cn - radius_m, ce + radius_m, cn + radius_m)

    params = {
        "SERVICE": "WCS",
        "VERSION": "1.0.0",
        "REQUEST": "GetCoverage",
        "COVERAGE": WCS_COVERAGE,
        "CRS": "EPSG:2056",
        "RESPONSE_CRS": "EPSG:2056",
        "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "WIDTH": resolution,
        "HEIGHT": resolution,
        "FORMAT": "GTiff",
    }

    resp = requests.get(WCS_URL, params=params, timeout=60)
    resp.raise_for_status()
    if resp.content[:5] in (b"<?xml", b"<Serv", b"<exce"):
        raise RuntimeError(f"WCS service error: {resp.text[:200]}")

    with MemoryFile(resp.content) as mf:
        with mf.open() as ds:
            elevation = ds.read(1).astype(np.float32)
            t = ds.transform
            h, w = elevation.shape
            if ds.nodata is not None:
                elevation[elevation == ds.nodata] = np.nan

    x_coords = t.c + (np.arange(w) + 0.5) * t.a   # E, ascending
    y_coords = t.f + (np.arange(h) + 0.5) * t.e   # N, descending
    return elevation, {"x_coords": x_coords, "y_coords": y_coords}


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
    to_wgs84 = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    ce, cn = center_lv95

    # Bounding box in WGS84
    lons, lats = to_wgs84.transform(
        [ce - radius_m, ce + radius_m],
        [cn - radius_m, cn + radius_m],
    )
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)

    # Download (and cache) needed 1°×1° tiles
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tile_paths = []
    for tlat in range(math.floor(lat_min), math.ceil(lat_max)):
        for tlon in range(math.floor(lon_min), math.ceil(lon_max)):
            url, name = _tile_url(tlat, tlon)
            cache_path = CACHE_DIR / f"{name}.tif"

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

    # Reproject raw WGS84 raster → LV95, cropped to our bbox
    dst_crs = CRS.from_epsg(2056)
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
