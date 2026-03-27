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
) -> None:
    """
    Build a solid circular terrain STL plus a separate track-tube body.

    The terrain disc is centred at (0,0), radius = diameter_mm/2.
    The track is a continuous swept rectangular tube, disconnected from
    the terrain, that intrudes track_intrude_mm into the surface and
    rises track_raise_mm above it.
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
    # 6. Track tube — separate body
    # ------------------------------------------------------------------
    if track_lv95:
        track_e = np.array([p[0] for p in track_lv95])
        track_n = np.array([p[1] for p in track_lv95])
        tx = (track_e - ce) * scale_xy
        ty = (track_n - cn) * scale_xy

        if track_alts is not None:
            # IGC: use real GPS altitude mapped through the same scale as terrain
            track_zm = (np.array(track_alts, dtype=np.float64) - elev_min) * scale_z + base_height_mm
            # Symmetric tube centred on the flight altitude — no terrain intrusion
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

        track_tris = _build_track_tube(
            tx, ty, track_zm, track_width_mm, tube_raise, tube_intrude,
        )
        triangles.extend(track_tris)
        print(f"  Track tube: {len(track_tris)} triangles")

    # ------------------------------------------------------------------
    # 7. Export
    # ------------------------------------------------------------------
    solid = stl_mesh.Mesh(np.zeros(len(triangles), dtype=stl_mesh.Mesh.dtype))
    for i, (v0, v1, v2) in enumerate(triangles):
        solid.vectors[i] = [v0, v1, v2]
    solid.update_normals()
    solid.save(output_path)
    print(f"  Total: {len(triangles)} triangles "
          f"({terrain_count} terrain + {len(triangles) - terrain_count} track)")


# ------------------------------------------------------------------
# Track tube builder
# ------------------------------------------------------------------

def _build_track_tube(tx, ty, track_zm, width_mm, raise_mm, intrude_mm):
    """
    Continuous watertight rectangular tube swept along the track polyline.

    A cross-section is placed at every track point using the averaged travel
    direction (miter joints), so consecutive sections share edges with no gaps.
    Only the very first and last cross-sections receive end caps.
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
    sections = []
    for k in range(n):
        nx, ny = perps[k]
        zt = float(track_zm[k]) + raise_mm
        zb = float(track_zm[k]) - intrude_mm
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
