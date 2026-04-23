"""Coordinate transforms and track geometry calculations."""

import numpy as np
import shapely
from shapely.geometry import MultiPoint
from pyproj import Transformer

_to_lv95 = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
_to_wgs84 = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)


def wgs84_to_lv95(lats, lons) -> tuple[np.ndarray, np.ndarray]:
    """Convert WGS84 (lat, lon) arrays to LV95 (E, N) in meters."""
    return _to_lv95.transform(np.asarray(lons), np.asarray(lats))


def compute_geometry(
    points: list[tuple[float, float]], padding: float = 0.20
) -> tuple[tuple[float, float], float]:
    """
    Compute track centroid and padded bounding radius in LV95.

    Args:
        points:  list of (lat, lon) in WGS84
        padding: fractional radial padding, e.g. 0.20 = 20%

    Returns:
        center_lv95: (E, N) centroid in meters (EPSG:2056)
        radius_m:    padded radius in meters
    """
    lats = np.array([p[0] for p in points])
    lons = np.array([p[1] for p in points])

    east, north = wgs84_to_lv95(lats, lons)

    # Minimum bounding circle of the track: smallest circle enclosing all
    # points.  Using this instead of the centroid + max-distance gives an
    # optimal center for asymmetric tracks and maximises the usable area
    # inside the disc.
    mp = MultiPoint(list(zip(east.tolist(), north.tolist())))
    try:
        circ = shapely.minimum_bounding_circle(mp)
        ce = float(circ.centroid.x)
        cn = float(circ.centroid.y)
        max_dist = float(shapely.minimum_bounding_radius(mp))
    except Exception:
        # Fallback: centroid + max distance (older shapely)
        ce, cn = float(east.mean()), float(north.mean())
        max_dist = float(np.hypot(east - ce, north - cn).max())

    radius_m = max_dist * (1.0 + padding)

    return (float(ce), float(cn)), float(radius_m)
