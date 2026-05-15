#!/usr/bin/env python3
"""
gps-terrain-stl  —  Convert a GPS track (GPX/IGC) to a 3D terrain STL.

Elevation data: Swisstopo SwissALTI3D (Switzerland only, EPSG:2056).
Output: a round, solid STL disc centred on the track, 100 mm diameter by default.
"""

import argparse
import os
import sys

import numpy as np

from parse import load_track
from geometry import (
    compute_geometry,
    compute_rect_geometry,
    compute_rect_tiles,
    wgs84_to_lv95,
)
from elevation import fetch_elevation
from mesh import build_and_export
from water import fetch_water_bodies


def main():
    parser = argparse.ArgumentParser(
        description="Convert GPX/IGC track to a 3D terrain STL (SwissALTI3D).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Input GPS file (.gpx or .igc)")
    parser.add_argument("-o", "--output", default=None, help="Output STL file (default: input filename with .stl extension)")
    parser.add_argument("--diameter", type=float, default=100.0,
                        help="Output disc diameter in mm")
    parser.add_argument("--padding", type=float, default=None,
                        help="Radial padding around outermost track point (0.20 = 20%%). "
                             "If omitted, computed from --edge-margin.")
    parser.add_argument("--edge-margin", type=float, default=10.0,
                        help="Minimum distance in mm between track and disc edge "
                             "(used when --padding is not given)")
    parser.add_argument("--exaggeration", type=float, default=1.0,
                        help="Vertical exaggeration factor")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Raster/grid resolution (pixels per side, max 2000)")
    parser.add_argument("--base-height", type=float, default=3.0,
                        help="Base/plinth thickness in mm")
    parser.add_argument("--track-width", type=float, default=0.6,
                        help="Track tube width in mm")
    parser.add_argument("--track-raise", type=float, default=0.6,
                        help="How far the track rises above terrain in mm")
    parser.add_argument("--track-intrude", type=float, default=2.0,
                        help="How far the track intrudes into terrain in mm")
    parser.add_argument("--track-tolerance", type=float, default=0.2,
                        help="Clearance gap carved into terrain on each side of track in mm")
    parser.add_argument("--min-water-area", type=float, default=8_000_000,
                        help="Minimum water body area in m² to include (default: 8,000,000 = 8 km²)")
    parser.add_argument("--rivers", action="store_true",
                        help="Include the main Swiss rivers as 0.9 mm wide ribbons (disabled by default)")
    parser.add_argument("--river-width", type=float, default=0.9,
                        help="Render width of main rivers in mm (printable on a 3D printer)")
    parser.add_argument("--no-water", action="store_true",
                        help="Skip water body detection and plates")
    parser.add_argument("--shape", choices=["circle", "square", "hexagon", "rectangle"],
                        default="circle",
                        help="Shape of the terrain plate")
    parser.add_argument("--rect-width", type=float, default=100.0,
                        help="Rectangle width in mm (only when --shape=rectangle)")
    parser.add_argument("--rect-height", type=float, default=60.0,
                        help="Rectangle height in mm (only when --shape=rectangle)")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + ".stl"

    # Derive padding from edge margin if not explicitly provided.
    # Shape-aware: hexagon's inscribed circle is smaller than its circumradius,
    # and a rectangle is bounded by its shorter side.
    if args.padding is None:
        if args.shape == "rectangle":
            disc_radius_mm = max(args.rect_width, args.rect_height) / 2.0
            inscribed_radius_mm = min(args.rect_width, args.rect_height) / 2.0
        elif args.shape == "hexagon":
            disc_radius_mm = args.diameter / 2.0
            inscribed_radius_mm = disc_radius_mm * (3 ** 0.5) / 2.0
        else:
            disc_radius_mm = args.diameter / 2.0
            inscribed_radius_mm = disc_radius_mm
        track_radius_mm = max(inscribed_radius_mm - args.edge_margin, 1.0)
        args.padding = (disc_radius_mm - track_radius_mm) / track_radius_mm

    # --- Load track ---
    print(f"Loading track: {args.input}")
    points = load_track(args.input)
    print(f"  {len(points)} track points")

    # --- Geometry ---
    print("Computing geometry…")
    if args.shape == "rectangle":
        # 20 mm gap is measured to the track edge → margin accounts for the
        # track tube half-width too.
        rect_edge_margin_mm = 20.0 + args.track_width / 2.0
        center_lv95, radius_m, rotation_rad = compute_rect_geometry(
            points, args.rect_width, args.rect_height,
            edge_margin_mm=rect_edge_margin_mm,
        )
        print(f"  Rectangle orientation: {np.degrees(rotation_rad):.1f}°")
    else:
        center_lv95, radius_m = compute_geometry(points, padding=args.padding)
        rotation_rad = 0.0
    ce, cn = center_lv95
    print(f"  Centre (LV95):  E={ce:.0f} m,  N={cn:.0f} m")
    print(f"  Radius (padded): {radius_m:.0f} m  ({radius_m/1000:.2f} km)")

    # --- Elevation ---
    print("Fetching SwissALTI3D elevation data…")
    elevation, grid_info = fetch_elevation(center_lv95, radius_m, args.resolution)
    elev_min = float(np.nanmin(elevation))
    elev_max = float(np.nanmax(elevation))
    print(f"  Grid: {elevation.shape[1]}×{elevation.shape[0]} px,  "
          f"elevation {elev_min:.0f}–{elev_max:.0f} m")

    # --- Track in LV95 ---
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    east, north = wgs84_to_lv95(lats, lons)
    track_lv95 = list(zip(east.tolist(), north.tolist()))
    # Altitude is present for IGC files (3-tuple), absent for GPX (2-tuple)
    track_alts = [p[2] for p in points] if len(points[0]) > 2 else None

    # --- Water bodies ---
    if args.no_water:
        water_polys, river_lines = [], []
    else:
        print("Fetching water bodies from OpenStreetMap…")
        water_polys, river_lines = fetch_water_bodies(
            center_lv95, radius_m,
            min_area_m2=args.min_water_area,
            include_rivers=args.rivers,
        )
        print(f"  {len(water_polys)} water polygon(s), "
              f"{len(river_lines)} main river segment(s) found")

    # --- Build mesh & export (split long rectangles into ≤240 mm tiles) ---
    if args.shape == "rectangle" and max(args.rect_width, args.rect_height) > 240.0:
        tiles = compute_rect_tiles(
            center_lv95, radius_m, rotation_rad,
            args.rect_width, args.rect_height, max_tile_long_mm=240.0,
        )
        print(f"Splitting into {len(tiles)} tile(s) "
              f"of <=240 mm along the long side")
    else:
        tiles = [{
            "index": 1, "total": 1,
            "center_lv95": center_lv95, "radius_m": radius_m,
            "rect_width_mm": args.rect_width, "rect_height_mm": args.rect_height,
            "cut_edges": set(),
        }]

    base, ext = os.path.splitext(args.output)
    for tile in tiles:
        if tile["total"] > 1:
            tile_label = f"_tile{tile['index']}of{tile['total']}"
        else:
            tile_label = ""
        print(f"Building mesh{(' for ' + tile_label) if tile_label else ''}…")
        build_and_export(
            elevation=elevation,
            grid_info=grid_info,
            center_lv95=tile["center_lv95"],
            radius_m=tile["radius_m"],
            track_lv95=track_lv95,
            track_alts=track_alts,
            output_path=f"{base}{tile_label}{ext}",
            diameter_mm=args.diameter,
            base_height_mm=args.base_height,
            exaggeration=args.exaggeration,
            track_width_mm=args.track_width,
            track_raise_mm=args.track_raise,
            track_intrude_mm=args.track_intrude,
            track_tolerance_mm=args.track_tolerance,
            water_polys_lv95=water_polys,
            rivers_lv95=river_lines,
            river_width_mm=args.river_width,
            shape=args.shape,
            rect_width_mm=tile["rect_width_mm"],
            rect_height_mm=tile["rect_height_mm"],
            rotation_rad=rotation_rad,
            cut_edges=tile["cut_edges"] if tile["cut_edges"] else None,
        )
    print("Done.")


if __name__ == "__main__":
    main()
