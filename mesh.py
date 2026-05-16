"""Build a watertight 3D terrain mesh and export to STL."""

import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from stl import mesh as stl_mesh

VALID_SHAPES = ("circle", "square", "hexagon", "rectangle")


def _generate_shape_ring(shape: str, radius_mm: float, N: int,
                          width_mm: float | None = None,
                          height_mm: float | None = None):
    """
    Return (ring_x, ring_y, inside_mask_func) for the given shape.

    ring_x, ring_y: 1-D arrays of boundary vertices (closed smooth ring).
    inside_func(Xm, Ym, shrink) -> bool mask: True where grid point is inside shape.

    width_mm / height_mm are only consulted for shape='rectangle'; the other
    shapes use the symmetric `radius_mm` (half-diameter).
    """
    if shape == "circle":
        N_ring = max(720, N * 2)
        angles = np.linspace(0, 2 * np.pi, N_ring, endpoint=False)
        ring_x = radius_mm * np.cos(angles)
        ring_y = radius_mm * np.sin(angles)

        def inside_func(Xm, Ym, shrink):
            return np.hypot(Xm, Ym) <= (radius_mm - shrink)

    elif shape == "square":
        # Square with side = diameter, corners at +-radius_mm
        # Smooth edges: many points per side for consistent stitching
        pts_per_side = max(180, N // 2)
        ring_x_list, ring_y_list = [], []
        r = radius_mm
        # bottom edge: left to right
        ring_x_list.extend(np.linspace(-r, r, pts_per_side, endpoint=False))
        ring_y_list.extend(np.full(pts_per_side, -r))
        # right edge: bottom to top
        ring_x_list.extend(np.full(pts_per_side, r))
        ring_y_list.extend(np.linspace(-r, r, pts_per_side, endpoint=False))
        # top edge: right to left
        ring_x_list.extend(np.linspace(r, -r, pts_per_side, endpoint=False))
        ring_y_list.extend(np.full(pts_per_side, r))
        # left edge: top to bottom
        ring_x_list.extend(np.full(pts_per_side, -r))
        ring_y_list.extend(np.linspace(r, -r, pts_per_side, endpoint=False))
        ring_x = np.array(ring_x_list)
        ring_y = np.array(ring_y_list)

        def inside_func(Xm, Ym, shrink):
            return (np.abs(Xm) <= (r - shrink)) & (np.abs(Ym) <= (r - shrink))

    elif shape == "hexagon":
        # Regular hexagon (flat-top): 6 corners, many interpolated edge points
        pts_per_edge = max(120, N // 3)
        hex_angles = np.linspace(0, 2 * np.pi, 7)  # 6 corners + wrap
        corners_x = radius_mm * np.cos(hex_angles)
        corners_y = radius_mm * np.sin(hex_angles)
        ring_x_list, ring_y_list = [], []
        for i in range(6):
            xs = np.linspace(corners_x[i], corners_x[i + 1], pts_per_edge, endpoint=False)
            ys = np.linspace(corners_y[i], corners_y[i + 1], pts_per_edge, endpoint=False)
            ring_x_list.extend(xs)
            ring_y_list.extend(ys)
        ring_x = np.array(ring_x_list)
        ring_y = np.array(ring_y_list)

        def inside_func(Xm, Ym, shrink):
            # A point is inside a regular hexagon if the max of three
            # projected distances is <= radius.
            # For a flat-top hexagon at origin with circumradius R:
            #   |x| <= R, |x/2 + y*sqrt(3)/2| <= R, |x/2 - y*sqrt(3)/2| <= R
            R = radius_mm - shrink
            d1 = np.abs(Xm)
            d2 = np.abs(Xm * 0.5 + Ym * (np.sqrt(3) / 2))
            d3 = np.abs(Xm * 0.5 - Ym * (np.sqrt(3) / 2))
            return (d1 <= R) & (d2 <= R) & (d3 <= R)

    elif shape == "rectangle":
        # Axis-aligned rectangle: width × height (mm), centred at origin.
        if width_mm is None or height_mm is None:
            raise ValueError("rectangle shape requires width_mm and height_mm")
        hw = width_mm / 2.0
        hh = height_mm / 2.0
        pts_per_side = max(180, N // 2)
        ring_x_list, ring_y_list = [], []
        # bottom edge: left to right
        ring_x_list.extend(np.linspace(-hw, hw, pts_per_side, endpoint=False))
        ring_y_list.extend(np.full(pts_per_side, -hh))
        # right edge: bottom to top
        ring_x_list.extend(np.full(pts_per_side, hw))
        ring_y_list.extend(np.linspace(-hh, hh, pts_per_side, endpoint=False))
        # top edge: right to left
        ring_x_list.extend(np.linspace(hw, -hw, pts_per_side, endpoint=False))
        ring_y_list.extend(np.full(pts_per_side, hh))
        # left edge: top to bottom
        ring_x_list.extend(np.full(pts_per_side, -hw))
        ring_y_list.extend(np.linspace(hh, -hh, pts_per_side, endpoint=False))
        ring_x = np.array(ring_x_list)
        ring_y = np.array(ring_y_list)

        def inside_func(Xm, Ym, shrink):
            return (np.abs(Xm) <= (hw - shrink)) & (np.abs(Ym) <= (hh - shrink))

    else:
        raise ValueError(f"Unknown shape '{shape}'. Use one of {VALID_SHAPES}")

    return ring_x, ring_y, inside_func


def build_and_export(
    elevation: np.ndarray,
    grid_info: dict,
    center_lv95: tuple[float, float],
    radius_m: float,
    track_lv95: list[tuple[float, float]],
    output_path: str,
    track_alts: list[float] | None = None,
    diameter_mm: float = 100.0,
    base_height_mm: float = 3.0,
    exaggeration: float = 1.0,
    track_width_mm: float = 1.0,
    track_raise_mm: float = 1.0,
    track_intrude_mm: float = 2.0,
    track_tolerance_mm: float = 0.2,
    water_polys_lv95: list | None = None,
    rivers_lv95: list | None = None,
    river_width_mm: float = 0.9,
    shape: str = "circle",
    rect_width_mm: float | None = None,
    rect_height_mm: float | None = None,
    rotation_rad: float = 0.0,
    cut_edges: set | None = None,
) -> None:
    """
    Build a solid terrain STL plus a track-tube body.

    The terrain is centred at (0,0), radius = diameter_mm/2.
    Shape can be 'circle', 'square', 'hexagon', or 'rectangle'.
    For 'rectangle', rect_width_mm and rect_height_mm specify the side
    lengths; diameter_mm is ignored.
    A groove is carved into the terrain along the track path with width
    (track_width + 2*tolerance) down to basis_level, so the track tube
    fits without intersection.
    """
    ce, cn = center_lv95
    if shape == "rectangle":
        if rect_width_mm is None or rect_height_mm is None:
            raise ValueError("rectangle shape requires rect_width_mm and rect_height_mm")
        # Grid is square, sized to the larger side; ring & inside-mask use the
        # rectangular extents.  radius_mm is the half-extent of the *grid*.
        radius_mm = max(rect_width_mm, rect_height_mm) / 2.0
    else:
        radius_mm = diameter_mm / 2.0
    scale_xy = radius_mm / radius_m

    # Rotation matrix (LV95 ↔ model). For all shapes except rectangle, rotation
    # is 0.  For rectangle, app.py / main.py choose an angle that aligns the
    # track's long axis with the rectangle's long side.
    import math as _math
    rot_cos = _math.cos(rotation_rad)
    rot_sin = _math.sin(rotation_rad)

    # ------------------------------------------------------------------
    # 1. Elevation interpolator (LV95 coords → metres above sea level)
    # ------------------------------------------------------------------
    x_coords = grid_info["x_coords"]   # E, ascending
    y_coords = grid_info["y_coords"]   # N, descending

    interp = RegularGridInterpolator(
        (y_coords[::-1], x_coords),
        elevation[::-1, :],
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    # ------------------------------------------------------------------
    # 2. NxN model-space grid, shape-masked
    # ------------------------------------------------------------------
    N = len(x_coords)
    lin = np.linspace(-radius_mm, radius_mm, N)
    Xm, Ym = np.meshgrid(lin, lin[::-1])   # row 0 = north
    pixel_size_mm = 2.0 * radius_mm / N

    # Inverse transform: unscale, un-rotate, un-centre.
    _u = Xm / scale_xy
    _v = Ym / scale_xy
    E_grid = ce + _u * rot_cos - _v * rot_sin
    N_grid = cn + _u * rot_sin + _v * rot_cos

    ring_x, ring_y, inside_func = _generate_shape_ring(
        shape, radius_mm, N,
        width_mm=rect_width_mm, height_mm=rect_height_mm,
    )
    N_ring = len(ring_x)
    # Shrink mask by one pixel so the grid sits fully inside the boundary ring
    inside = inside_func(Xm, Ym, pixel_size_mm)

    query_pts = np.stack([N_grid[inside], E_grid[inside]], axis=1)
    elev_vals = interp(query_pts)

    # Sea detection.  Copernicus DEM uses two conventions:
    #   - NODATA / NaN over open ocean (e.g. Mediterranean tiles)
    #   - exactly 0 m over the ocean (e.g. the Atlantic near Boston)
    # We treat *both* as sea candidates and then filter the resulting mask by
    # connected-component size so that tiny coincidentally low-elevation
    # patches (small ponds, voids) are not misclassified.
    SEA_ELEV_MAX_M = 0.5
    sea_nan_1d = np.isnan(elev_vals)
    sea_low_1d = (~sea_nan_1d) & (elev_vals <= SEA_ELEV_MAX_M)
    sea_candidate_1d = sea_nan_1d | sea_low_1d
    if sea_nan_1d.any():
        elev_vals[sea_nan_1d] = 0.0

    Z_elev = np.full((N, N), np.nan)
    Z_elev[inside] = elev_vals
    Z_sea_candidate = np.zeros((N, N), dtype=bool)
    Z_sea_candidate[inside] = sea_candidate_1d

    Z_sea = np.zeros((N, N), dtype=bool)
    if Z_sea_candidate.any():
        try:
            from scipy.ndimage import label as _label
            labels_arr, n_labels = _label(Z_sea_candidate)
            # Keep only components that are >= 0.05 % of the total grid (with
            # an absolute floor of 50 pixels) — large enough to plausibly be
            # an ocean / large bay, not an isolated low patch.
            min_pixels = max(50, (N * N) // 2000)
            for lid in range(1, n_labels + 1):
                comp = (labels_arr == lid)
                if comp.sum() >= min_pixels:
                    Z_sea |= comp
        except Exception:
            # Without scipy, fall back to using all candidates.
            Z_sea = Z_sea_candidate
    elev_min = np.nanmin(Z_elev)
    scale_z = scale_xy * exaggeration

    Zm = np.where(inside,
                  (Z_elev - elev_min) * scale_z + base_height_mm,
                  np.nan)

    # ------------------------------------------------------------------
    # 2b. Pre-compute track in model space and carve groove into Zm
    #     (must happen before terrain triangles are built)
    # ------------------------------------------------------------------
    tx = ty = track_zm = tube_raise = tube_intrude = basis_level = None

    if track_lv95:
        track_e = np.array([p[0] for p in track_lv95])
        track_n = np.array([p[1] for p in track_lv95])
        # Forward transform: centre, rotate, scale.
        _dx = track_e - ce
        _dy = track_n - cn
        tx = (_dx * rot_cos + _dy * rot_sin) * scale_xy
        ty = (-_dx * rot_sin + _dy * rot_cos) * scale_xy

        if track_alts is not None:
            # IGC: use real GPS altitude mapped through the same scale as terrain
            track_zm = (np.array(track_alts, dtype=np.float64) - elev_min) * scale_z + base_height_mm
            # Symmetric tube centred on the flight altitude
            tube_raise = track_width_mm / 2.0
            tube_intrude = track_width_mm / 2.0
            print("  Using GPS altitude for track height")
        else:
            # GPX: snap to terrain surface + raise/intrude
            track_elev = interp(np.stack([track_n, track_e], axis=1))
            track_elev = np.where(np.isnan(track_elev), elev_min, track_elev)
            track_zm = (track_elev - elev_min) * scale_z + base_height_mm
            tube_raise = track_raise_mm
            tube_intrude = track_intrude_mm

        basis_level = float(np.min(track_zm) - tube_intrude)

        # Build a smooth groove polygon (LineString.buffer) instead of
        # rasterizing into the grid.  Section 4 uses this polygon to clip
        # terrain triangles, so the groove edge follows the polygon exactly
        # — no pixel stairsteps — preserving fit precision with the track
        # solid (which uses the same buffer recipe minus track_tolerance_mm).
        pixel_size_mm = 2.0 * radius_mm / N
        tx_dense, ty_dense = _densify_track(tx, ty, pixel_size_mm * 0.5)
        groove_half_width = track_width_mm / 2.0 + track_tolerance_mm
        try:
            from shapely.geometry import LineString as _SLS
            line_ms = _SLS(list(zip(tx_dense.tolist(), ty_dense.tolist())))
            groove_poly_ms = line_ms.buffer(
                groove_half_width, cap_style=1, join_style=1, resolution=16
            )
            if not groove_poly_ms.is_valid:
                groove_poly_ms = groove_poly_ms.buffer(0)
        except ImportError:
            # Shapely missing → fall back to per-pixel carving.
            groove_poly_ms = None
            min_dist_sq = np.full((N, N), np.inf)
            for k in range(len(tx_dense) - 1):
                ax, ay = float(tx_dense[k]), float(ty_dense[k])
                bx, by = float(tx_dense[k + 1]), float(ty_dense[k + 1])
                dx, dy = bx - ax, by - ay
                seg_len2 = dx * dx + dy * dy
                if seg_len2 < 1e-12:
                    d2 = (Xm - ax) ** 2 + (Ym - ay) ** 2
                else:
                    t = np.clip(((Xm - ax) * dx + (Ym - ay) * dy) / seg_len2, 0.0, 1.0)
                    cx = ax + t * dx
                    cy = ay + t * dy
                    d2 = (Xm - cx) ** 2 + (Ym - cy) ** 2
                np.minimum(min_dist_sq, d2, out=min_dist_sq)
            groove_mask = (~np.isnan(Zm)) & (min_dist_sq < groove_half_width ** 2)
            Zm = np.where(groove_mask & (Zm > basis_level), basis_level, Zm)
    else:
        groove_poly_ms = None

    # ------------------------------------------------------------------
    # 2c. Water polygon conversion + terrain carving
    #     Must happen before terrain triangles are built.
    # ------------------------------------------------------------------
    water_parts_ms = []   # (shapely Polygon in model space, z_top, z_bot)

    # ------------------------------------------------------------------
    # Sea: pixels with NaN in the source DEM (= ocean in Copernicus).
    # We turn them into shapely polygons in model space and feed them
    # through the same water-plate pipeline as lakes.
    # ------------------------------------------------------------------
    sea_polys_ms: list = []
    if Z_sea.any():
        try:
            import rasterio.features as _rfeat
            from rasterio.transform import Affine as _Affine
            from shapely.geometry import shape as _sh_shape, Polygon as _SPoly
            pixel_size_mm = 2.0 * radius_mm / N
            # Pixel (col, row) → model (x, y). Row 0 is the top of the
            # grid (y = +radius_mm). Apply the same orientation as Xm/Ym.
            sea_transform = (
                _Affine.translation(-radius_mm, radius_mm)
                * _Affine.scale(pixel_size_mm, -pixel_size_mm)
            )
            for geom_dict, val in _rfeat.shapes(
                Z_sea.astype("uint8"), mask=Z_sea, transform=sea_transform,
            ):
                if val != 1:
                    continue
                try:
                    p = _sh_shape(geom_dict)
                    if not p.is_valid:
                        p = p.buffer(0)
                    if p.is_empty or not isinstance(p, _SPoly):
                        continue
                    # Smooth pixel stairsteps; drop slivers below 25 mm².
                    p = p.buffer(pixel_size_mm * 0.6).buffer(-pixel_size_mm * 0.6)
                    if p.is_valid and not p.is_empty and p.area >= 25.0:
                        sea_polys_ms.append(p)
                except Exception:
                    continue
        except Exception:
            sea_polys_ms = []

    if water_polys_lv95 or rivers_lv95 or sea_polys_ms:
        try:
            import shapely as shp
            from shapely.geometry import Point as SPoint, Polygon as SPolygon, \
                MultiPolygon as SMultiPolygon, LineString as SLineString
            from shapely.ops import unary_union

            # Build disc shape matching the selected form
            disc_ring_coords = list(zip(ring_x.tolist(), ring_y.tolist()))
            disc_ring_coords.append(disc_ring_coords[0])  # close the ring
            disc_shape = SPolygon(disc_ring_coords)
            if not disc_shape.is_valid:
                disc_shape = disc_shape.buffer(0)
            plate_thickness = 1.0

            # Shrink disc by 1 mm for water clipping so water polygons leave a
            # clean terrain border at the disc edge instead of running right
            # to the boundary (required for 3D printing).
            # For a tiled rectangle, the inset is 0 on edges that are cut
            # boundaries between tiles, so water continues seamlessly across
            # adjacent tiles.
            if shape == "rectangle" and cut_edges:
                hw = rect_width_mm / 2.0
                hh = rect_height_mm / 2.0
                m_left   = 0.0 if "-x" in cut_edges else 1.0
                m_right  = 0.0 if "+x" in cut_edges else 1.0
                m_bottom = 0.0 if "-y" in cut_edges else 1.0
                m_top    = 0.0 if "+y" in cut_edges else 1.0
                water_disc_shape = SPolygon([
                    (-hw + m_left,  -hh + m_bottom),
                    ( hw - m_right, -hh + m_bottom),
                    ( hw - m_right,  hh - m_top),
                    (-hw + m_left,   hh - m_top),
                ])
            else:
                water_disc_shape = disc_shape.buffer(-1.0)
                if water_disc_shape.is_empty or not water_disc_shape.is_valid:
                    water_disc_shape = disc_shape

            # Build track buffer in model space for subtracting from water plates.
            # Uses the same half-width as the terrain groove so the cutout matches.
            track_buf_ms = None
            if tx is not None and len(tx) >= 2:
                groove_hw = track_width_mm / 2.0 + track_tolerance_mm
                track_buf_ms = SLineString(zip(tx.tolist(), ty.tolist())).buffer(groove_hw)

            # Pre-process: merge lake polygons that are contained within other lake
            # polygons.  In OSM, islands within lakes are sometimes returned as
            # separate closed-way polygons (distinct from the multipolygon relation
            # that already carries them as interior rings).  Unioning all polygons
            # and then taking the union resolves duplicates; remaining interior rings
            # represent true islands.
            if len(water_polys_lv95) > 1:
                try:
                    merged = unary_union(water_polys_lv95)
                    if not merged.is_empty:
                        if hasattr(merged, 'geoms'):
                            water_polys_lv95 = list(merged.geoms)
                        else:
                            water_polys_lv95 = [merged]
                except Exception:
                    pass   # keep original list on failure

            for poly_lv95 in water_polys_lv95:
                ext = list(poly_lv95.exterior.coords)
                def _lv95_to_model(_e, _n):
                    _dx = _e - ce
                    _dy = _n - cn
                    return (
                        (_dx * rot_cos + _dy * rot_sin) * scale_xy,
                        (-_dx * rot_sin + _dy * rot_cos) * scale_xy,
                    )
                ext_ms = [_lv95_to_model(e, n) for e, n in ext]
                holes_ms = [
                    [_lv95_to_model(e, n) for e, n in ring.coords]
                    for ring in poly_lv95.interiors
                ]
                try:
                    poly_ms = SPolygon(ext_ms, holes_ms)
                    if not poly_ms.is_valid:
                        poly_ms = poly_ms.buffer(0)
                except Exception:
                    continue

                clipped = poly_ms.intersection(water_disc_shape)
                if clipped.is_empty:
                    continue

                # Subtract track groove from water polygon so the two parts
                # don't intersect and can be printed as separate bodies.
                if track_buf_ms is not None:
                    clipped = clipped.difference(track_buf_ms)
                    if clipped.is_empty:
                        continue

                # 1 mm² minimum: filters slivers produced by disc-clip or
                # track subtraction, even when the parent lake passed the OSM filter.
                parts = [g for g in (clipped.geoms if hasattr(clipped, 'geoms') else [clipped])
                         if isinstance(g, SPolygon) and not g.is_empty and g.area >= 1.0]

                for part in parts:
                    # Sample terrain elevation at centroid for plate height
                    cent = part.centroid
                    _u_c = cent.x / scale_xy
                    _v_c = cent.y / scale_xy
                    wlv95_e = ce + _u_c * rot_cos - _v_c * rot_sin
                    wlv95_n = cn + _u_c * rot_sin + _v_c * rot_cos
                    w_elev = interp(np.array([[wlv95_n, wlv95_e]]))[0]
                    if np.isnan(w_elev):
                        w_elev = elev_min
                    z_top = (w_elev - elev_min) * scale_z + base_height_mm
                    z_bot = z_top - plate_thickness

                    # Carve terrain: rasterize polygon (buffered by tolerance)
                    carved = part.buffer(track_tolerance_mm)
                    water_mask = shp.contains_xy(
                        carved, Xm.ravel(), Ym.ravel()
                    ).reshape(N, N)
                    water_mask &= ~np.isnan(Zm)
                    Zm = np.where(water_mask & (Zm > z_bot), z_bot, Zm)

                    water_parts_ms.append((part, z_top, z_bot))

            # ----------------------------------------------------------
            # Rivers: buffer LV95 LineStrings in model space so they
            # render as a constant printable-width ribbon (river_width_mm).
            # ----------------------------------------------------------
            if rivers_lv95:
                for river_line in rivers_lv95:
                    try:
                        coords_ms = []
                        for (e, n) in river_line.coords:
                            _dx = e - ce
                            _dy = n - cn
                            coords_ms.append((
                                (_dx * rot_cos + _dy * rot_sin) * scale_xy,
                                (-_dx * rot_sin + _dy * rot_cos) * scale_xy,
                            ))
                        if len(coords_ms) < 2:
                            continue
                        river_ms = SLineString(coords_ms).buffer(
                            river_width_mm / 2.0,
                            cap_style=2, join_style=1, resolution=8,
                        )
                        if not river_ms.is_valid:
                            river_ms = river_ms.buffer(0)
                    except Exception:
                        continue

                    clipped = river_ms.intersection(water_disc_shape)
                    if clipped.is_empty:
                        continue
                    if track_buf_ms is not None:
                        clipped = clipped.difference(track_buf_ms)
                        if clipped.is_empty:
                            continue

                    parts = [g for g in (clipped.geoms if hasattr(clipped, 'geoms') else [clipped])
                             if isinstance(g, SPolygon) and not g.is_empty
                             and g.area >= 0.05]

                    for part in parts:
                        cent = part.centroid
                        _u_c = cent.x / scale_xy
                        _v_c = cent.y / scale_xy
                        wlv95_e = ce + _u_c * rot_cos - _v_c * rot_sin
                        wlv95_n = cn + _u_c * rot_sin + _v_c * rot_cos
                        w_elev = interp(np.array([[wlv95_n, wlv95_e]]))[0]
                        if np.isnan(w_elev):
                            w_elev = elev_min
                        z_top = (w_elev - elev_min) * scale_z + base_height_mm
                        z_bot = z_top - plate_thickness

                        carved = part.buffer(track_tolerance_mm)
                        water_mask = shp.contains_xy(
                            carved, Xm.ravel(), Ym.ravel()
                        ).reshape(N, N)
                        water_mask &= ~np.isnan(Zm)
                        Zm = np.where(water_mask & (Zm > z_bot), z_bot, Zm)
                        water_parts_ms.append((part, z_top, z_bot))

            # ----------------------------------------------------------
            # Sea (already in model space from the NaN-pixel detector).
            # Sea level is z = base_height_mm (the actual 0 m elevation
            # maps there once elev_min has been subtracted), so the sea
            # plate sits flush with the lowest point of the terrain.
            # ----------------------------------------------------------
            if sea_polys_ms:
                z_top = (0.0 - elev_min) * scale_z + base_height_mm
                z_bot = z_top - plate_thickness
                for sea_poly in sea_polys_ms:
                    clipped = sea_poly.intersection(water_disc_shape)
                    if clipped.is_empty:
                        continue
                    if track_buf_ms is not None:
                        clipped = clipped.difference(track_buf_ms)
                        if clipped.is_empty:
                            continue
                    parts = [g for g in (clipped.geoms if hasattr(clipped, "geoms") else [clipped])
                             if isinstance(g, SPolygon) and not g.is_empty
                             and g.area >= 1.0]
                    for part in parts:
                        carved = part.buffer(track_tolerance_mm)
                        water_mask = shp.contains_xy(
                            carved, Xm.ravel(), Ym.ravel()
                        ).reshape(N, N)
                        water_mask &= ~np.isnan(Zm)
                        Zm = np.where(water_mask & (Zm > z_bot), z_bot, Zm)
                        water_parts_ms.append((part, z_top, z_bot))

        except ImportError:
            pass

    # ------------------------------------------------------------------
    # 3. Smooth boundary ring (shape-dependent, generated in section 2)
    # ------------------------------------------------------------------
    # ring_x, ring_y, N_ring already set by _generate_shape_ring()

    # Interpolate elevation at ring vertices
    _u_ring = ring_x / scale_xy
    _v_ring = ring_y / scale_xy
    ring_E = ce + _u_ring * rot_cos - _v_ring * rot_sin
    ring_N = cn + _u_ring * rot_sin + _v_ring * rot_cos
    ring_elev = interp(np.stack([ring_N, ring_E], axis=1))
    ring_elev = np.where(np.isnan(ring_elev), elev_min, ring_elev)
    ring_z = (ring_elev - elev_min) * scale_z + base_height_mm

    # Apply groove carving to ring vertices near the track
    if tx is not None and basis_level is not None:
        groove_hw2 = (track_width_mm / 2.0 + track_tolerance_mm) ** 2
        tx_d, ty_d = _densify_track(tx, ty, pixel_size_mm * 0.5)
        for ri in range(N_ring):
            rx, ry = float(ring_x[ri]), float(ring_y[ri])
            min_d2 = np.inf
            for k in range(len(tx_d) - 1):
                ax, ay = float(tx_d[k]), float(ty_d[k])
                bx, by = float(tx_d[k + 1]), float(ty_d[k + 1])
                dx, dy = bx - ax, by - ay
                seg2 = dx * dx + dy * dy
                if seg2 < 1e-12:
                    d2 = (rx - ax) ** 2 + (ry - ay) ** 2
                else:
                    t = max(0.0, min(1.0, ((rx - ax) * dx + (ry - ay) * dy) / seg2))
                    d2 = (rx - (ax + t * dx)) ** 2 + (ry - (ay + t * dy)) ** 2
                if d2 < min_d2:
                    min_d2 = d2
            if min_d2 < groove_hw2 and ring_z[ri] > basis_level:
                ring_z[ri] = basis_level

    # Also carve ring for water polygons
    if water_parts_ms:
        try:
            import shapely as shp
            for part, _z_top, z_bot in water_parts_ms:
                carved = part.buffer(track_tolerance_mm)
                ring_inside_water = shp.contains_xy(carved, ring_x, ring_y)
                ring_z = np.where(ring_inside_water & (ring_z > z_bot), z_bot, ring_z)
        except Exception:
            pass

    ring_verts_top = list(zip(ring_x.tolist(), ring_y.tolist(), ring_z.tolist()))

    # ------------------------------------------------------------------
    # 4. Terrain top-surface triangles (interior grid + groove clipping)
    # ------------------------------------------------------------------
    triangles = []

    def v(r, c):
        return (float(Xm[r, c]), float(Ym[r, c]), float(Zm[r, c]))

    # Helper: terrain z (model space) at any (x_mm, y_mm) via the LV95 interp.
    def _terrain_z(x_mm, y_mm):
        _u = x_mm / scale_xy
        _v = y_mm / scale_xy
        E = ce + _u * rot_cos - _v * rot_sin
        Nn = cn + _u * rot_sin + _v * rot_cos
        elev = float(interp(np.array([[Nn, E]]))[0])
        if np.isnan(elev):
            elev = elev_min
        return (elev - elev_min) * scale_z + base_height_mm

    # Per-grid-corner groove mask (True = corner inside groove polygon).
    # Used to short-circuit pixels that sit fully inside or fully outside.
    if groove_poly_ms is not None:
        try:
            import shapely as shp
            from shapely.geometry import Polygon as _SPoly
            from shapely.geometry.polygon import orient as _orient
            corner_in_groove = shp.contains_xy(
                groove_poly_ms, Xm.ravel(), Ym.ravel()
            ).reshape(N, N)
        except Exception:
            corner_in_groove = None
            groove_poly_ms = None
    else:
        corner_in_groove = None

    for row in range(N - 1):
        for col in range(N - 1):
            tl, tr = v(row, col), v(row, col + 1)
            bl, br = v(row + 1, col), v(row + 1, col + 1)
            if any(np.isnan(vert[2]) for vert in (tl, tr, bl, br)):
                continue

            if corner_in_groove is None:
                triangles.append((tl, bl, br))
                triangles.append((tl, br, tr))
                continue

            n_in = (int(corner_in_groove[row, col])
                    + int(corner_in_groove[row, col + 1])
                    + int(corner_in_groove[row + 1, col])
                    + int(corner_in_groove[row + 1, col + 1]))

            if n_in == 0:
                # Fully outside groove → standard terrain triangulation.
                triangles.append((tl, bl, br))
                triangles.append((tl, br, tr))
            elif n_in == 4:
                # Fully inside groove → groove floor triangulation handles
                # this area smoothly (skip here to avoid double-coverage).
                continue
            else:
                # Quad straddles groove boundary → clip the OUTSIDE part
                # against the groove polygon and triangulate that.
                # Quad as CCW polygon: tl → bl → br → tr.
                quad_poly = _SPoly([
                    (tl[0], tl[1]),
                    (bl[0], bl[1]),
                    (br[0], br[1]),
                    (tr[0], tr[1]),
                ])
                outside = quad_poly.difference(groove_poly_ms)
                if outside.is_empty:
                    continue
                geoms = (list(outside.geoms) if hasattr(outside, "geoms")
                         else [outside])
                for geom in geoms:
                    if not isinstance(geom, _SPoly) or geom.is_empty:
                        continue
                    try:
                        if hasattr(shp, "constrained_delaunay_triangles"):
                            tri_coll = shp.constrained_delaunay_triangles(geom)
                            sub_tris = list(tri_coll.geoms)
                        else:
                            tri_coll = shp.delaunay_triangles(geom, only_edges=False)
                            sub_tris = [g for g in tri_coll.geoms
                                        if geom.contains(g.representative_point())]
                    except Exception:
                        sub_tris = []
                    for tri in sub_tris:
                        coords = list(tri.exterior.coords)[:3]
                        verts = [(x, y, _terrain_z(x, y)) for (x, y) in coords]
                        a, b, c = verts
                        # CCW winding so the normal points up.
                        if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0:
                            b, c = c, b
                        triangles.append((a, b, c))

    # ------------------------------------------------------------------
    # 4b. Groove floor + walls (smooth, polygon-based)
    # ------------------------------------------------------------------
    if groove_poly_ms is not None and corner_in_groove is not None:
        try:
            disc_ring_coords = list(zip(ring_x.tolist(), ring_y.tolist()))
            disc_ring_coords.append(disc_ring_coords[0])
            disc_shape_groove = _SPoly(disc_ring_coords)
            if not disc_shape_groove.is_valid:
                disc_shape_groove = disc_shape_groove.buffer(0)
            groove_in_disc = groove_poly_ms.intersection(disc_shape_groove)
        except Exception:
            groove_in_disc = groove_poly_ms

        groove_geoms = (list(groove_in_disc.geoms)
                        if hasattr(groove_in_disc, "geoms")
                        else [groove_in_disc])

        for poly in groove_geoms:
            if not isinstance(poly, _SPoly) or poly.is_empty:
                continue
            poly = _orient(poly, sign=1.0)

            # --- Floor: flat plate at basis_level (normal up, into cavity) ---
            try:
                if hasattr(shp, "constrained_delaunay_triangles"):
                    tri_coll = shp.constrained_delaunay_triangles(poly)
                    floor_tris = list(tri_coll.geoms)
                else:
                    tri_coll = shp.delaunay_triangles(poly, only_edges=False)
                    floor_tris = [g for g in tri_coll.geoms
                                  if poly.contains(g.representative_point())]
            except Exception:
                floor_tris = []
            for tri in floor_tris:
                coords = list(tri.exterior.coords)[:3]
                a = (coords[0][0], coords[0][1], basis_level)
                b = (coords[1][0], coords[1][1], basis_level)
                c = (coords[2][0], coords[2][1], basis_level)
                # CCW in plan → normal points up (into the empty cavity above).
                if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0:
                    b, c = c, b
                triangles.append((a, b, c))

            # --- Walls: cavity is on the LEFT for both exterior (CCW) and
            #     hole (CW) rings after orient(poly, 1.0).  A single winding
            #     puts the wall normal into the cavity for all rings. ---
            for ring in [poly.exterior] + list(poly.interiors):
                coords = list(ring.coords)
                for i in range(len(coords) - 1):
                    x0, y0 = coords[i]
                    x1, y1 = coords[i + 1]
                    zt0 = _terrain_z(x0, y0)
                    zt1 = _terrain_z(x1, y1)
                    a_bot = (x0, y0, basis_level)
                    b_bot = (x1, y1, basis_level)
                    a_top = (x0, y0, zt0)
                    b_top = (x1, y1, zt1)
                    triangles.append((a_bot, b_top, b_bot))
                    triangles.append((a_bot, a_top, b_top))

    # ------------------------------------------------------------------
    # 5. Stitch smooth ring to inner grid boundary
    # ------------------------------------------------------------------
    # Collect grid boundary vertices (outermost valid vertices)
    def quad_valid(r, c):
        if r < 0 or c < 0 or r >= N - 1 or c >= N - 1:
            return False
        return (not np.isnan(Zm[r, c]) and not np.isnan(Zm[r, c + 1])
                and not np.isnan(Zm[r + 1, c]) and not np.isnan(Zm[r + 1, c + 1]))

    boundary_verts_set: set[tuple[int, int]] = set()
    for row in range(N - 1):
        for col in range(N - 1):
            if not quad_valid(row, col):
                continue
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if not quad_valid(row + dr, col + dc):
                    # This quad is on the boundary
                    for r2 in (row, row + 1):
                        for c2 in (col, col + 1):
                            if not np.isnan(Zm[r2, c2]):
                                boundary_verts_set.add((r2, c2))
                    break

    bv_list = list(boundary_verts_set)
    bv_xy = np.array([(float(Xm[r, c]), float(Ym[r, c])) for r, c in bv_list])
    bv_z = np.array([float(Zm[r, c]) for r, c in bv_list])
    bv_angles = np.arctan2(bv_xy[:, 1], bv_xy[:, 0])

    # Stitch: for each pair of consecutive ring vertices, find the nearest
    # grid boundary vertex and triangulate the gap.
    # Strategy: merge ring + boundary verts, Delaunay triangulate the annulus.
    from scipy.spatial import Delaunay

    all_xy = np.vstack([
        np.column_stack([ring_x, ring_y]),
        bv_xy,
    ])
    all_z = np.concatenate([ring_z, bv_z])
    n_ring_pts = N_ring
    n_bv = len(bv_list)

    # Mark: ring verts are indices [0, N_ring), boundary verts are [N_ring, N_ring+n_bv)
    is_ring = np.zeros(len(all_xy), dtype=bool)
    is_ring[:n_ring_pts] = True

    tri_dt = Delaunay(all_xy)

    # Keep only triangles that span the annular gap (have both ring and grid verts)
    # and lie inside the shape boundary
    for simplex in tri_dt.simplices:
        has_ring = any(is_ring[i] for i in simplex)
        has_grid = any(not is_ring[i] for i in simplex)
        if not (has_ring and has_grid):
            continue
        # Check centroid is inside the shape (with generous margin)
        cx = all_xy[simplex, 0].mean()
        cy = all_xy[simplex, 1].mean()
        if not inside_func(np.array([cx]), np.array([cy]), -pixel_size_mm * 2)[0]:
            continue
        i0, i1, i2 = simplex
        v0 = (float(all_xy[i0, 0]), float(all_xy[i0, 1]), float(all_z[i0]))
        v1 = (float(all_xy[i1, 0]), float(all_xy[i1, 1]), float(all_z[i1]))
        v2 = (float(all_xy[i2, 0]), float(all_xy[i2, 1]), float(all_z[i2]))
        # Ensure outward (upward) normal via CCW winding
        cross = (v1[0]-v0[0])*(v2[1]-v0[1]) - (v1[1]-v0[1])*(v2[0]-v0[0])
        if cross < 0:
            v1, v2 = v2, v1
        triangles.append((v0, v1, v2))

    # ------------------------------------------------------------------
    # 6. Smooth side walls (ring top → z=0)
    # ------------------------------------------------------------------
    for i in range(N_ring):
        j = (i + 1) % N_ring
        a_top = ring_verts_top[i]
        b_top = ring_verts_top[j]
        a_bot = (a_top[0], a_top[1], 0.0)
        b_bot = (b_top[0], b_top[1], 0.0)
        triangles.append((a_top, b_bot, b_top))
        triangles.append((a_top, a_bot, b_bot))

    # ------------------------------------------------------------------
    # 7. Smooth bottom face (fan from centre)
    # ------------------------------------------------------------------
    cbot = (0.0, 0.0, 0.0)
    for i in range(N_ring):
        j = (i + 1) % N_ring
        a = (ring_verts_top[i][0], ring_verts_top[i][1], 0.0)
        b = (ring_verts_top[j][0], ring_verts_top[j][1], 0.0)
        triangles.append((cbot, b, a))

    terrain_tris = _remove_small_components(triangles, min_area_mm2=1.0)
    print(f"  Terrain: {len(terrain_tris)} triangles")

    # ------------------------------------------------------------------
    # 8. Track tube
    # ------------------------------------------------------------------
    track_tris = []
    if track_lv95 and tx is not None:
        # Top of the track follows terrain at every (x,y) → +raise overhang.
        # Bottom is the constant basis_level plane (flat underside).
        def _terrain_top_z(xs_mm, ys_mm):
            """Vectorised terrain z (model space) for an array of (x,y) in mm."""
            _u = np.asarray(xs_mm) / scale_xy
            _v = np.asarray(ys_mm) / scale_xy
            E = ce + _u * rot_cos - _v * rot_sin
            N = cn + _u * rot_sin + _v * rot_cos
            elev = interp(np.stack([N, E], axis=-1))
            elev = np.where(np.isnan(elev), elev_min, elev)
            return (elev - elev_min) * scale_z + base_height_mm

        # Clip the track footprint to the plate's outline so tiled plates
        # produce track segments that end flush at the cut edges.
        try:
            from shapely.geometry import Polygon as _SP
            disc_clip = _SP(list(zip(ring_x.tolist(), ring_y.tolist())))
            if not disc_clip.is_valid:
                disc_clip = disc_clip.buffer(0)
        except Exception:
            disc_clip = None
        track_tris = _build_track_solid(
            tx, ty, track_width_mm, tube_raise, basis_level, _terrain_top_z,
            clip_polygon=disc_clip,
        )
        print(f"  Track solid: {len(track_tris)} triangles")

    # ------------------------------------------------------------------
    # 9. Water plates
    # ------------------------------------------------------------------
    water_tris = []
    if water_parts_ms:
        for part, z_top, z_bot in water_parts_ms:
            plate_tris = _build_water_plate(part, z_bot, z_top)
            water_tris.extend(plate_tris)
        print(f"  Water plates: {len(water_tris)} triangles")

    # ------------------------------------------------------------------
    # 10. Export separate STL files
    # ------------------------------------------------------------------
    base, ext = os.path.splitext(output_path)

    def _save_stl(tris, path):
        if not tris:
            return
        solid = stl_mesh.Mesh(np.zeros(len(tris), dtype=stl_mesh.Mesh.dtype))
        for i, (v0, v1, v2) in enumerate(tris):
            solid.vectors[i] = [v0, v1, v2]
        solid.update_normals()
        solid.save(path)

    terrain_path = f"{base}_terrain{ext}"
    track_path = f"{base}_track{ext}"
    water_path = f"{base}_water{ext}"

    _save_stl(terrain_tris, terrain_path)
    print(f"  Saved: {terrain_path}")

    if track_tris:
        _save_stl(track_tris, track_path)
        print(f"  Saved: {track_path}")

    if water_tris:
        _save_stl(water_tris, water_path)
        print(f"  Saved: {water_path}")

    # Also save combined STL for backward compatibility
    all_tris = terrain_tris + track_tris + water_tris
    _save_stl(all_tris, output_path)

    total = len(all_tris)
    print(f"  Total: {total} triangles "
          f"({len(terrain_tris)} terrain + {len(track_tris)} track + {len(water_tris)} water)")


# ------------------------------------------------------------------
# Artifact filter: remove small disconnected components
# ------------------------------------------------------------------

def _remove_small_components(triangles: list, min_area_mm2: float = 1.0) -> list:
    """
    Drop any group of triangles that is disconnected from the rest of the
    mesh AND whose total surface area is below min_area_mm2.

    Works by union-find on rounded vertex positions: two triangles belong to
    the same component if they share at least one vertex (after rounding to
    3 decimal places = 1 µm).  Components whose summed triangle area is below
    the threshold are discarded.
    """
    if not triangles:
        return triangles

    n = len(triangles)
    tri_arr = np.array(triangles, dtype=np.float64)   # (N, 3, 3)

    # Map each unique rounded vertex to an integer id
    vert_id: dict[tuple, int] = {}
    tri_vids = np.empty((n, 3), dtype=np.int32)
    for i, (v0, v1, v2) in enumerate(triangles):
        for j, v in enumerate((v0, v1, v2)):
            key = (round(v[0], 3), round(v[1], 3), round(v[2], 3))
            if key not in vert_id:
                vert_id[key] = len(vert_id)
            tri_vids[i, j] = vert_id[key]

    # Union-Find (path-compressed)
    parent = list(range(len(vert_id)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for a, b, c in tri_vids:
        union(a, b)
        union(b, c)

    # Group triangle indices by component root
    from collections import defaultdict
    comp: dict[int, list[int]] = defaultdict(list)
    for i, (a, _, _) in enumerate(tri_vids):
        comp[find(a)].append(i)

    # Per-triangle area (half cross-product magnitude)
    e1 = tri_arr[:, 1] - tri_arr[:, 0]
    e2 = tri_arr[:, 2] - tri_arr[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)

    result: list = []
    n_removed = 0
    for idxs in comp.values():
        if areas[idxs].sum() >= min_area_mm2:
            result.extend(triangles[i] for i in idxs)
        else:
            n_removed += len(idxs)

    if n_removed:
        print(f"  Removed {n_removed} artifact triangles ({len(comp)} → "
              f"{len(comp) - sum(1 for idxs in comp.values() if areas[idxs].sum() < min_area_mm2)} components)")

    return result


# ------------------------------------------------------------------
# Water plate builder
# ------------------------------------------------------------------

def _build_water_plate(poly, z_bot, z_top):
    """
    Build a watertight 1mm-thick flat plate from a shapely Polygon.

    Triangulates top and bottom faces using scipy Delaunay on all ring
    vertices, then filters to triangles whose centroid lies inside the
    polygon.  shapely.Polygon.contains() respects interior rings (islands),
    so island areas are automatically excluded — no degenerate fan needed.

    Side walls are built directly from exterior and interior ring coordinates.
    Interior ring side walls use reversed winding so normals face into the
    island (away from the solid plate material).
    """
    import shapely as shp
    from scipy.spatial import Delaunay

    tris = []

    # --- Triangulate top and bottom faces via Delaunay + containment filter ---
    all_pts = list(poly.exterior.coords[:-1])
    for ring in poly.interiors:
        all_pts.extend(ring.coords[:-1])
    pts_arr = np.array(all_pts, dtype=np.float64)

    if len(pts_arr) >= 3:
        try:
            dt = Delaunay(pts_arr)
            simplices = dt.simplices
            # Compute centroids in bulk
            v = pts_arr[simplices]          # (M, 3, 2)
            cx = v[:, :, 0].mean(axis=1)   # (M,)
            cy = v[:, :, 1].mean(axis=1)   # (M,)
            inside = shp.contains_xy(poly, cx, cy)
            for idx, keep in enumerate(inside):
                if not keep:
                    continue
                v0, v1, v2 = pts_arr[simplices[idx]]
                # Ensure CCW winding for top face (normal up)
                cross = (v1[0]-v0[0])*(v2[1]-v0[1]) - (v1[1]-v0[1])*(v2[0]-v0[0])
                if cross < 0:
                    v1, v2 = v2, v1
                a = (float(v0[0]), float(v0[1]))
                b = (float(v1[0]), float(v1[1]))
                c = (float(v2[0]), float(v2[1]))
                tris.append((a + (z_top,), b + (z_top,), c + (z_top,)))   # top, normal up
                tris.append((a + (z_bot,), c + (z_bot,), b + (z_bot,)))   # bottom, normal down
        except Exception:
            pass

    # --- Side walls: exterior ring (normals point outward from lake) ---
    ext = [(x, y) for x, y in poly.exterior.coords[:-1]]
    n = len(ext)
    for i in range(n):
        ax, ay = ext[i]
        bx, by = ext[(i + 1) % n]
        tris.append(((ax, ay, z_bot), (ax, ay, z_top), (bx, by, z_top)))
        tris.append(((ax, ay, z_bot), (bx, by, z_top), (bx, by, z_bot)))

    # --- Side walls: interior rings (normals point into the island) ---
    for hole in poly.interiors:
        hpts = [(x, y) for x, y in hole.coords[:-1]]
        m = len(hpts)
        for i in range(m):
            ax, ay = hpts[i]
            bx, by = hpts[(i + 1) % m]
            tris.append(((ax, ay, z_bot), (bx, by, z_top), (ax, ay, z_top)))
            tris.append(((ax, ay, z_bot), (bx, by, z_bot), (bx, by, z_top)))

    return tris


# ------------------------------------------------------------------
# Track helpers
# ------------------------------------------------------------------

def _densify_track(tx, ty, max_step_mm):
    """
    Linearly interpolate along the track so no segment exceeds max_step_mm.
    Returns densified (tx, ty) arrays suitable for sub-pixel groove carving.
    """
    xs, ys = [tx[0]], [ty[0]]
    for k in range(len(tx) - 1):
        dx, dy = tx[k + 1] - tx[k], ty[k + 1] - ty[k]
        seg_len = np.hypot(dx, dy)
        n = max(1, int(np.ceil(seg_len / max_step_mm)))
        for i in range(1, n + 1):
            t = i / n
            xs.append(tx[k] + t * dx)
            ys.append(ty[k] + t * dy)
    return np.array(xs), np.array(ys)


# ------------------------------------------------------------------
# Track tube builder
# ------------------------------------------------------------------

def _build_track_solid(tx, ty, width_mm, raise_mm, basis_level, terrain_top_z,
                       clip_polygon=None):
    """
    Build a single watertight track solid using a 2-D Shapely buffer of the
    polyline.  Self-intersections (loops, lap repeats, out-and-back sections)
    merge into one polygon, so the resulting STL is always a single body.

    Underside  : constant z = basis_level  (flat plane shared with the groove).
    Top surface: follows the terrain elevation + raise_mm at every (x, y),
                 sampled via the supplied terrain_top_z(xs, ys) callable.
    """
    n = len(tx)
    if n < 2:
        return []

    try:
        from shapely.geometry import LineString, MultiPolygon, Polygon
        from shapely.geometry.polygon import orient
    except ImportError:
        # Shapely missing — fall back to a constant-height swept tube.
        return _build_track_tube(tx, ty, np.full(n, basis_level + raise_mm),
                                 width_mm, raise_mm, basis_level)

    coords = list(zip(tx.tolist(), ty.tolist())) if hasattr(tx, "tolist") else list(zip(tx, ty))
    line = LineString(coords)
    # round/round caps so two passes meeting at any angle merge cleanly
    footprint = line.buffer(width_mm / 2.0, cap_style=1, join_style=1, resolution=8)

    if clip_polygon is not None and not footprint.is_empty:
        footprint = footprint.intersection(clip_polygon)

    if footprint.is_empty:
        return []
    if isinstance(footprint, MultiPolygon):
        polys = list(footprint.geoms)
    elif isinstance(footprint, Polygon):
        polys = [footprint]
    else:
        polys = []

    z_bot = float(basis_level)

    # Cache top z for any (x,y) we encounter so duplicate vertices share the
    # same height (avoids tiny seams between top triangles and side walls).
    z_top_cache = {}

    def top_z(x, y):
        key = (round(x, 6), round(y, 6))
        z = z_top_cache.get(key)
        if z is None:
            z = float(terrain_top_z(np.array([x]), np.array([y]))[0]) + raise_mm
            z_top_cache[key] = z
        return z

    tris = []
    for poly in polys:
        poly = orient(poly, sign=1.0)   # exterior CCW, holes CW

        # ---- top + bottom faces via constrained Delaunay -------------
        # Constrained Delaunay respects the polygon boundary, so no triangles
        # ever span across concavities (which is what produced visible
        # "Ausbrüche" on long, curvy track polygons).
        try:
            import shapely as shp
            if hasattr(shp, "constrained_delaunay_triangles"):
                tri_collection = shp.constrained_delaunay_triangles(poly)
                inside_tris = list(tri_collection.geoms)
            else:
                tri_collection = shp.delaunay_triangles(poly, only_edges=False)
                inside_tris = [g for g in tri_collection.geoms
                               if poly.contains(g.representative_point())]
        except Exception:
            inside_tris = []

        for tri in inside_tris:
            (x0, y0), (x1, y1), (x2, y2), _ = list(tri.exterior.coords)
            zt0, zt1, zt2 = top_z(x0, y0), top_z(x1, y1), top_z(x2, y2)
            tris.append(((x0, y0, zt0), (x1, y1, zt1), (x2, y2, zt2)))
            # bottom: flat at z_bot, flip winding so normal points down
            tris.append(((x0, y0, z_bot), (x2, y2, z_bot), (x1, y1, z_bot)))

        # ---- side walls (CCW exterior → outward, CW holes → inward) --
        rings = [list(poly.exterior.coords)] + [list(r.coords) for r in poly.interiors]
        for r_idx, ring in enumerate(rings):
            ccw = (r_idx == 0)
            for i in range(len(ring) - 1):
                ax, ay = ring[i]
                bx, by = ring[i + 1]
                za = top_z(ax, ay)
                zb = top_z(bx, by)
                a_top = (ax, ay, za)
                b_top = (bx, by, zb)
                a_bot = (ax, ay, z_bot)
                b_bot = (bx, by, z_bot)
                if ccw:
                    tris.append((a_bot, b_bot, b_top))
                    tris.append((a_bot, b_top, a_top))
                else:
                    tris.append((a_bot, b_top, b_bot))
                    tris.append((a_bot, a_top, b_top))

    return tris


def _build_track_tube(tx, ty, track_zm, width_mm, raise_mm, basis_level):
    """
    Continuous watertight rectangular tube swept along the track polyline.

    A cross-section is placed at every track point using the averaged travel
    direction (miter joints), so consecutive sections share edges with no gaps.
    Only the very first and last cross-sections receive end caps.

    basis_level is the constant bottom Z for all cross-sections — the lowest
    point of the tube including intrusion, shared across the whole track.
    """
    n = len(tx)
    if n < 2:
        return []

    hw = width_mm / 2.0

    # Perpendicular (left-hand) direction at each point
    perps = []
    for k in range(n):
        if k == 0:
            dx, dy = tx[1] - tx[0], ty[1] - ty[0]
        elif k == n - 1:
            dx, dy = tx[k] - tx[k - 1], ty[k] - ty[k - 1]
        else:
            dx, dy = tx[k + 1] - tx[k - 1], ty[k + 1] - ty[k - 1]
        length = np.hypot(dx, dy)
        if length < 1e-6:
            perps.append(perps[-1] if perps else (1.0, 0.0))
        else:
            perps.append((-dy / length, dx / length))

    # Cross-section at each point: (left-top, right-top, left-bot, right-bot)
    # Bottom is a flat plane at basis_level shared across all sections.
    sections = []
    for k in range(n):
        nx, ny = perps[k]
        zt = float(track_zm[k]) + raise_mm
        zb = basis_level
        sections.append((
            (tx[k] + nx * hw, ty[k] + ny * hw, zt),   # left-top
            (tx[k] - nx * hw, ty[k] - ny * hw, zt),   # right-top
            (tx[k] + nx * hw, ty[k] + ny * hw, zb),   # left-bot
            (tx[k] - nx * hw, ty[k] - ny * hw, zb),   # right-bot
        ))

    tris = []

    # Connect consecutive cross-sections (4 faces × 2 triangles = 8 per gap)
    for k in range(n - 1):
        s_lt, s_rt, s_lb, s_rb = sections[k]
        e_lt, e_rt, e_lb, e_rb = sections[k + 1]
        tris += [(s_lt, s_rt, e_rt), (s_lt, e_rt, e_lt)]   # top
        tris += [(s_rb, s_lb, e_lb), (s_rb, e_lb, e_rb)]   # bottom
        tris += [(s_lb, s_lt, e_lt), (s_lb, e_lt, e_lb)]   # left
        tris += [(s_rt, s_rb, e_rb), (s_rt, e_rb, e_rt)]   # right

    # Start cap
    lt, rt, lb, rb = sections[0]
    tris += [(lt, lb, rb), (lt, rb, rt)]
    # End cap
    lt, rt, lb, rb = sections[-1]
    tris += [(rt, rb, lb), (rt, lb, lt)]

    return tris
