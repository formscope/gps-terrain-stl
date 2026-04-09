"""Build a watertight 3D terrain mesh and export to STL."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from stl import mesh as stl_mesh


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
) -> None:
    """
    Build a solid circular terrain STL plus a track-tube body.

    The terrain disc is centred at (0,0), radius = diameter_mm/2.
    A groove is carved into the terrain along the track path with width
    (track_width + 2*tolerance) down to basis_level, so the track tube
    fits without intersection.
    """
    ce, cn = center_lv95
    radius_mm = diameter_mm / 2.0
    scale_xy = radius_mm / radius_m

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
    # 2. NxN model-space grid, circle-masked
    # ------------------------------------------------------------------
    N = len(x_coords)
    lin = np.linspace(-radius_mm, radius_mm, N)
    Xm, Ym = np.meshgrid(lin, lin[::-1])   # row 0 = north

    E_grid = ce + Xm / scale_xy
    N_grid = cn + Ym / scale_xy
    inside = np.hypot(Xm, Ym) <= radius_mm

    query_pts = np.stack([N_grid[inside], E_grid[inside]], axis=1)
    elev_vals = interp(query_pts)

    if np.any(np.isnan(elev_vals)):
        valid = ~np.isnan(elev_vals)
        if valid.any():
            from scipy.spatial import cKDTree
            tree = cKDTree(query_pts[valid])
            bad = np.where(~valid)[0]
            _, nn = tree.query(query_pts[bad])
            elev_vals[bad] = elev_vals[valid][nn]
        else:
            elev_vals[:] = 0.0

    Z_elev = np.full((N, N), np.nan)
    Z_elev[inside] = elev_vals
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
        tx = (track_e - ce) * scale_xy
        ty = (track_n - cn) * scale_xy

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

        # Carve groove: lower terrain to basis_level within (width/2 + tolerance)
        # of the track path. Densify the track to sub-pixel spacing first so the
        # groove boundary follows curves faithfully at the grid resolution.
        pixel_size_mm = 2.0 * radius_mm / N
        tx_dense, ty_dense = _densify_track(tx, ty, pixel_size_mm * 0.5)

        groove_half_width = track_width_mm / 2.0 + track_tolerance_mm
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

    # ------------------------------------------------------------------
    # 2c. Water polygon conversion + terrain carving
    #     Must happen before terrain triangles are built.
    # ------------------------------------------------------------------
    water_parts_ms = []   # (shapely Polygon in model space, z_top, z_bot)

    if water_polys_lv95:
        try:
            import shapely as shp
            from shapely.geometry import Point as SPoint, Polygon as SPolygon, \
                MultiPolygon as SMultiPolygon, LineString as SLineString

            disc_shape = SPoint(0.0, 0.0).buffer(radius_mm)
            plate_thickness = 1.0

            # Build track buffer in model space for subtracting from water plates.
            # Uses the same half-width as the terrain groove so the cutout matches.
            track_buf_ms = None
            if tx is not None and len(tx) >= 2:
                groove_hw = track_width_mm / 2.0 + track_tolerance_mm
                track_buf_ms = SLineString(zip(tx.tolist(), ty.tolist())).buffer(groove_hw)

            for poly_lv95 in water_polys_lv95:
                ext = list(poly_lv95.exterior.coords)
                ext_ms = [((e - ce) * scale_xy, (n - cn) * scale_xy) for e, n in ext]
                holes_ms = [
                    [((e - ce) * scale_xy, (n - cn) * scale_xy) for e, n in ring.coords]
                    for ring in poly_lv95.interiors
                ]
                try:
                    poly_ms = SPolygon(ext_ms, holes_ms)
                    if not poly_ms.is_valid:
                        poly_ms = poly_ms.buffer(0)
                except Exception:
                    continue

                clipped = poly_ms.intersection(disc_shape)
                if clipped.is_empty:
                    continue

                # Subtract track groove from water polygon so the two parts
                # don't intersect and can be printed as separate bodies.
                if track_buf_ms is not None:
                    clipped = clipped.difference(track_buf_ms)
                    if clipped.is_empty:
                        continue

                parts = [g for g in (clipped.geoms if hasattr(clipped, 'geoms') else [clipped])
                         if isinstance(g, SPolygon) and not g.is_empty]

                for part in parts:
                    # Sample terrain elevation at centroid for plate height
                    cent = part.centroid
                    wlv95_e = ce + cent.x / scale_xy
                    wlv95_n = cn + cent.y / scale_xy
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

        except ImportError:
            pass

    # ------------------------------------------------------------------
    # 3. Terrain top-surface triangles
    #    Quad: TL=(row,col)  TR=(row,col+1)
    #          BL=(row+1,col) BR=(row+1,col+1)
    #    Upward normal: (TL,BL,BR) and (TL,BR,TR)
    # ------------------------------------------------------------------
    triangles = []

    def v(r, c):
        return (float(Xm[r, c]), float(Ym[r, c]), float(Zm[r, c]))

    for row in range(N - 1):
        for col in range(N - 1):
            tl, tr = v(row, col), v(row, col + 1)
            bl, br = v(row + 1, col), v(row + 1, col + 1)
            if any(np.isnan(vert[2]) for vert in (tl, tr, bl, br)):
                continue
            triangles.append((tl, bl, br))
            triangles.append((tl, br, tr))

    # ------------------------------------------------------------------
    # 4. Side walls
    # ------------------------------------------------------------------
    def quad_valid(r, c):
        if r < 0 or c < 0 or r >= N - 1 or c >= N - 1:
            return False
        return (not np.isnan(Zm[r, c]) and not np.isnan(Zm[r, c + 1])
                and not np.isnan(Zm[r + 1, c]) and not np.isnan(Zm[r + 1, c + 1]))

    boundary_edges = []
    for row in range(N - 1):
        for col in range(N - 1):
            if not quad_valid(row, col):
                continue
            if not quad_valid(row - 1, col):
                boundary_edges.append((v(row, col), v(row, col + 1)))
            if not quad_valid(row + 1, col):
                boundary_edges.append((v(row + 1, col + 1), v(row + 1, col)))
            if not quad_valid(row, col - 1):
                boundary_edges.append((v(row + 1, col), v(row, col)))
            if not quad_valid(row, col + 1):
                boundary_edges.append((v(row, col + 1), v(row + 1, col + 1)))

    for (ax, ay, az), (bx, by, bz) in boundary_edges:
        a_bot = (ax, ay, 0.0)
        b_bot = (bx, by, 0.0)
        triangles.append(((ax, ay, az), (bx, by, bz), b_bot))
        triangles.append(((ax, ay, az), b_bot, a_bot))

    # ------------------------------------------------------------------
    # 5. Bottom face
    # ------------------------------------------------------------------
    seen: set[tuple] = set()
    bottom_ring: list[tuple] = []
    for (ax, ay, _), (bx, by, _) in boundary_edges:
        for px, py in ((ax, ay), (bx, by)):
            key = (round(px, 3), round(py, 3))
            if key not in seen:
                seen.add(key)
                bottom_ring.append((px, py, 0.0))

    bottom_ring.sort(key=lambda p: np.arctan2(p[1], p[0]))
    cbot = (0.0, 0.0, 0.0)
    for i in range(len(bottom_ring)):
        a = bottom_ring[i]
        b = bottom_ring[(i + 1) % len(bottom_ring)]
        triangles.append((cbot, b, a))

    terrain_count = len(triangles)

    # ------------------------------------------------------------------
    # 6. Track tube
    # ------------------------------------------------------------------
    if track_lv95 and tx is not None:
        track_tris = _build_track_tube(
            tx, ty, track_zm, track_width_mm, tube_raise, basis_level,
        )
        triangles.extend(track_tris)
        print(f"  Track tube: {len(track_tris)} triangles")

    # ------------------------------------------------------------------
    # 7. Water plates
    # ------------------------------------------------------------------
    if water_parts_ms:
        water_tri_count = 0
        for part, z_top, z_bot in water_parts_ms:
            plate_tris = _build_water_plate(part, z_bot, z_top)
            triangles.extend(plate_tris)
            water_tri_count += len(plate_tris)
        print(f"  Water plates: {water_tri_count} triangles")

    # ------------------------------------------------------------------
    # 8. Export
    # ------------------------------------------------------------------
    solid = stl_mesh.Mesh(np.zeros(len(triangles), dtype=stl_mesh.Mesh.dtype))
    for i, (v0, v1, v2) in enumerate(triangles):
        solid.vectors[i] = [v0, v1, v2]
    solid.update_normals()
    solid.save(output_path)
    print(f"  Total: {len(triangles)} triangles "
          f"({terrain_count} terrain + {len(triangles) - terrain_count} track)")


# ------------------------------------------------------------------
# Water plate builder
# ------------------------------------------------------------------

def _build_water_plate(poly, z_bot, z_top):
    """
    Build a watertight 1mm-thick flat plate from a shapely Polygon.

    Uses filtered Delaunay for the top/bottom faces (handles concave shapes).
    Side walls are derived from the boundary edges of the face triangulation
    itself (directed-edge cancellation), not from the ring coordinates.
    This guarantees every edge is shared by exactly 2 triangles regardless
    of how the Delaunay chose to connect the vertices.
    """
    from shapely.ops import triangulate as shp_triangulate

    tris = []

    # Filtered Delaunay: keep only triangles whose centroid lies inside the
    # polygon. Use a tiny buffer so centroids on the exact boundary are kept.
    poly_buf = poly.buffer(1e-6)
    face_tris = [t for t in shp_triangulate(poly) if poly_buf.contains(t.centroid)]
    if not face_tris:
        return tris

    # Directed-edge tracking: interior edges appear twice (opposite directions)
    # and cancel; what remains are the boundary edges with correct outward winding.
    boundary_edges: dict[tuple, bool] = {}

    for tri in face_tris:
        # Round to avoid float noise in edge matching
        coords = [(round(x, 9), round(y, 9)) for x, y in tri.exterior.coords[:3]]
        a, b, c = coords[0], coords[1], coords[2]

        # Top face: CCW winding → normal up
        tris.append(((a[0], a[1], z_top), (b[0], b[1], z_top), (c[0], c[1], z_top)))
        # Bottom face: CW winding → normal down
        tris.append(((a[0], a[1], z_bot), (c[0], c[1], z_bot), (b[0], b[1], z_bot)))

        # Register directed edges; cancel reverse duplicates (interior edges)
        for v1, v2 in ((a, b), (b, c), (c, a)):
            if (v2, v1) in boundary_edges:
                del boundary_edges[(v2, v1)]
            else:
                boundary_edges[(v1, v2)] = True

    # Side walls for boundary edges only — guaranteed manifold
    for (v1, v2) in boundary_edges:
        tris.append(((v1[0], v1[1], z_bot), (v1[0], v1[1], z_top), (v2[0], v2[1], z_top)))
        tris.append(((v1[0], v1[1], z_bot), (v2[0], v2[1], z_top), (v2[0], v2[1], z_bot)))

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
