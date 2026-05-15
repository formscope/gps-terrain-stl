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


def compute_rect_geometry(
    points: list[tuple[float, float]],
    rect_width_mm: float,
    rect_height_mm: float,
    edge_margin_mm: float = 20.0,
) -> tuple[tuple[float, float], float, float]:
    """
    Compute an oriented bounding box for the track and the parameters needed
    to render it inside a rectangular plate with the *long* track axis
    aligned to the *long* rectangle side.

    Returns:
        center_lv95:  (E, N) — centre of the track's oriented bounding box
        radius_m:     effective LV95 "radius" so that scale_xy = radius_mm /
                      radius_m in mesh.py gives the desired uniform scale
                      (where radius_mm = max(rect_w, rect_h) / 2).
        rotation_rad: angle by which LV95 must be rotated to align the
                      track's long axis with the model x-axis (when the
                      rectangle is wider than tall) or y-axis (when taller).
    """
    lats = np.array([p[0] for p in points])
    lons = np.array([p[1] for p in points])
    east, north = wgs84_to_lv95(lats, lons)

    # Compute principal axes via PCA on (E, N).
    east_c = east - east.mean()
    north_c = north - north.mean()
    pts = np.column_stack([east_c, north_c])
    cov = np.cov(pts.T)
    evals, evecs = np.linalg.eigh(cov)
    # Major axis = eigenvector with the largest eigenvalue.
    major = evecs[:, int(np.argmax(evals))]
    theta_major = float(np.arctan2(major[1], major[0]))

    # Project track onto the rotated (u = major, v = minor) basis.
    cos_t, sin_t = np.cos(theta_major), np.sin(theta_major)
    u = east_c * cos_t + north_c * sin_t
    v = -east_c * sin_t + north_c * cos_t
    u_mid = (u.max() + u.min()) / 2.0
    v_mid = (v.max() + v.min()) / 2.0
    u_range = float(u.max() - u.min())
    v_range = float(v.max() - v.min())

    # OBB centre in LV95.
    ce = float(east.mean() + u_mid * cos_t - v_mid * sin_t)
    cn = float(north.mean() + u_mid * sin_t + v_mid * cos_t)

    # We want the track's long axis (= u) parallel to the rectangle's long side.
    if rect_width_mm >= rect_height_mm:
        long_side, short_side = rect_width_mm, rect_height_mm
        # Long side of rectangle is the model x-axis; rotate LV95 by -theta_major.
        rotation_rad = theta_major
    else:
        long_side, short_side = rect_height_mm, rect_width_mm
        # Long side is the model y-axis; the major axis must align with y.
        rotation_rad = theta_major - np.pi / 2.0

    # Scale (mm per metre) chosen so the track fits with margin on the
    # binding side. Both sides have at least `edge_margin_mm` of clearance.
    scale_long = (long_side - 2 * edge_margin_mm) / max(u_range, 1e-6)
    scale_short = (short_side - 2 * edge_margin_mm) / max(v_range, 1e-6)
    scale = float(min(scale_long, scale_short))
    if scale <= 0:
        scale = 1e-6

    radius_mm_grid = max(rect_width_mm, rect_height_mm) / 2.0
    radius_m = radius_mm_grid / scale

    return (ce, cn), float(radius_m), float(rotation_rad)
