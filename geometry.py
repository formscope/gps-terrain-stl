"""Coordinate transforms and track geometry calculations.

The "local" projection used by every other module is configured by
set_projection().  Tracks inside Switzerland use EPSG:2056 (LV95) so we
can keep using SwissALTI3D; tracks anywhere else use a track-centred
Transverse Mercator (metres, no scale distortion at the centre), which
keeps the rest of the pipeline unchanged.
"""

import numpy as np
import shapely
from shapely.geometry import MultiPoint
from pyproj import CRS, Transformer

# Approximate WGS84 bounding box of Switzerland (slightly padded). Tracks
# whose centroid falls inside this box use LV95; everywhere else falls back
# to a track-centred Transverse Mercator.
_CH_BOUNDS = (5.9, 45.7, 10.6, 47.9)   # (lon_min, lat_min, lon_max, lat_max)


class _ProjectorState:
    """Holds the currently active local projection.  Configured per request."""
    def __init__(self):
        self.to_local: Transformer | None = None
        self.to_wgs84: Transformer | None = None
        self.crs: CRS | None = None
        self.is_swiss: bool = False
        self.centre_lon: float = 8.0
        self.centre_lat: float = 46.8


_state = _ProjectorState()


def set_projection(centre_lon: float, centre_lat: float) -> None:
    """Configure the local projection used by every other module.

    Must be called before compute_geometry / fetch_elevation / etc. The
    chosen CRS is EPSG:2056 (LV95) for tracks inside Switzerland and a
    track-centred Transverse Mercator otherwise.
    """
    lon_min, lat_min, lon_max, lat_max = _CH_BOUNDS
    is_swiss = (lon_min < centre_lon < lon_max) and (lat_min < centre_lat < lat_max)
    if is_swiss:
        crs = CRS.from_epsg(2056)
    else:
        crs = CRS.from_proj4(
            f"+proj=tmerc +lat_0={centre_lat} +lon_0={centre_lon} "
            "+k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
        )
    _state.to_local = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    _state.to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    _state.crs = crs
    _state.is_swiss = is_swiss
    _state.centre_lon = centre_lon
    _state.centre_lat = centre_lat


def _ensure_default_projection():
    if _state.to_local is None:
        set_projection(8.0, 46.8)


def wgs84_to_local(lats, lons) -> tuple[np.ndarray, np.ndarray]:
    """Convert WGS84 (lat, lon) arrays to local projected (E, N) in metres."""
    _ensure_default_projection()
    return _state.to_local.transform(np.asarray(lons), np.asarray(lats))


def local_to_wgs84(east, north) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of wgs84_to_local — returns (lon, lat)."""
    _ensure_default_projection()
    return _state.to_wgs84.transform(np.asarray(east), np.asarray(north))


def is_swiss_area() -> bool:
    """True iff the currently configured projection is LV95 (Switzerland)."""
    _ensure_default_projection()
    return _state.is_swiss


def local_crs() -> CRS:
    """Return the currently active local CRS (a pyproj CRS object)."""
    _ensure_default_projection()
    return _state.crs


# Backward-compat alias — the rest of the codebase will be migrated.
def wgs84_to_lv95(lats, lons) -> tuple[np.ndarray, np.ndarray]:
    """Deprecated alias for wgs84_to_local()."""
    return wgs84_to_local(lats, lons)


def init_projection_from_points(points) -> tuple[float, float]:
    """Configure the local projection from the centroid of the given track.

    Returns the centroid (lon, lat) for callers that want to log it.
    """
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    centre_lon = float(np.mean(lons))
    centre_lat = float(np.mean(lats))
    set_projection(centre_lon, centre_lat)
    return centre_lon, centre_lat


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

    east, north = wgs84_to_local(lats, lons)

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
    east, north = wgs84_to_local(lats, lons)

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
    # Robustness: a degenerate track (one point, or all points colinear)
    # makes u_range or v_range zero, which would blow up the scale.  Floor
    # both ranges at 1 m so we still produce a sensible plate.
    u_range_safe = max(u_range, 1.0)
    v_range_safe = max(v_range, 1.0)
    scale_long = (long_side - 2 * edge_margin_mm) / u_range_safe
    scale_short = (short_side - 2 * edge_margin_mm) / v_range_safe
    scale = float(min(scale_long, scale_short))
    if scale <= 0:
        # Margin was larger than the rectangle → fall back to filling the
        # whole plate.  No physical track will produce this on purpose.
        scale = float(max(long_side, short_side) / 2.0 / max(u_range_safe, v_range_safe))

    radius_mm_grid = max(rect_width_mm, rect_height_mm) / 2.0
    radius_m = radius_mm_grid / scale

    # Sanity cap: a transverse-Mercator projection stays well-behaved up to
    # roughly ±500 km from its centre.  If the plate would need elevation
    # data over a larger area, something is wrong with the input (single
    # point, garbage coordinates, …).
    if radius_m > 500_000:
        raise ValueError(
            f"Plate would need elevation data over a {2*radius_m/1000:.0f} km area, "
            "which is far beyond the supported projection range. "
            "Check that the GPX file has reasonable coordinates and more than one point."
        )

    return (ce, cn), float(radius_m), float(rotation_rad)


def compute_rect_tiles(
    center_full_lv95: tuple[float, float],
    radius_m_full: float,
    rotation_rad: float,
    rect_width_mm: float,
    rect_height_mm: float,
    max_tile_long_mm: float = 240.0,
) -> list[dict]:
    """
    Split a rectangular plate whose long side exceeds max_tile_long_mm into
    equal-sized tiles that each fit within the limit.  Each tile is described
    by its own LV95 centre, effective LV95 'radius', rectangle dimensions
    (model space), and the set of edge identifiers that are flush cuts
    between adjacent tiles.

    Returns a list of dicts (one per tile, in order along the long axis):
        {
          "index": int (1-based),
          "total": int,
          "center_lv95": (E, N),
          "radius_m": float,
          "rect_width_mm": float,
          "rect_height_mm": float,
          "cut_edges": set[str],   # subset of {'-x','+x','-y','+y'}
        }
    """
    long_side = max(rect_width_mm, rect_height_mm)
    short_side = min(rect_width_mm, rect_height_mm)
    n_tiles = max(1, int(np.ceil(long_side / max_tile_long_mm)))
    tile_long = long_side / n_tiles
    long_is_x = (rect_width_mm >= rect_height_mm)

    # Same model-space scale across tiles; preserves the track size.
    radius_mm_full = max(rect_width_mm, rect_height_mm) / 2.0
    scale_xy_full = radius_mm_full / radius_m_full

    cos_t = float(np.cos(rotation_rad))
    sin_t = float(np.sin(rotation_rad))

    tiles = []
    for i in range(n_tiles):
        offset_long_mm = (i - (n_tiles - 1) / 2.0) * tile_long

        # Tile centre in LV95: shift along the long axis (in model space)
        # back through rotation+scale.
        if long_is_x:
            de = (offset_long_mm / scale_xy_full) * cos_t
            dn = (offset_long_mm / scale_xy_full) * sin_t
            tile_w = tile_long
            tile_h = short_side
        else:
            de = (offset_long_mm / scale_xy_full) * (-sin_t)
            dn = (offset_long_mm / scale_xy_full) * cos_t
            tile_w = short_side
            tile_h = tile_long

        tile_center = (center_full_lv95[0] + de, center_full_lv95[1] + dn)
        tile_radius_mm = max(tile_w, tile_h) / 2.0
        tile_radius_m = tile_radius_mm / scale_xy_full

        cut_edges: set[str] = set()
        if long_is_x:
            if i > 0:
                cut_edges.add("-x")
            if i < n_tiles - 1:
                cut_edges.add("+x")
        else:
            if i > 0:
                cut_edges.add("-y")
            if i < n_tiles - 1:
                cut_edges.add("+y")

        tiles.append({
            "index": i + 1,
            "total": n_tiles,
            "center_lv95": tile_center,
            "radius_m": tile_radius_m,
            "rect_width_mm": tile_w,
            "rect_height_mm": tile_h,
            "cut_edges": cut_edges,
        })
    return tiles
