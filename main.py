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
from geometry import compute_geometry, wgs84_to_lv95
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
    parser.add_argument("--padding", type=float, default=0.20,
                        help="Radial padding around outermost track point (0.20 = 20%%)")
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
    parser.add_argument("--min-water-area", type=float, default=500_000,
                        help="Minimum water body area in m² to include (default: 500,000 = 50 ha)")
    parser.add_argument("--rivers", action="store_true",
                        help="Include rivers and riverbanks as water plates (disabled by default)")
    parser.add_argument("--no-water", action="store_true",
                        help="Skip water body detection and plates")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + ".stl"

    # --- Load track ---
    print(f"Loading track: {args.input}")
    points = load_track(args.input)
    print(f"  {len(points)} track points")

    # --- Geometry ---
    print("Computing geometry…")
    center_lv95, radius_m = compute_geometry(points, padding=args.padding)
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
        water_polys = []
    else:
        print("Fetching water bodies from OpenStreetMap…")
        water_polys = fetch_water_bodies(center_lv95, radius_m, min_area_m2=args.min_water_area,
                                         include_rivers=args.rivers)
        print(f"  {len(water_polys)} water polygon(s) found")

    # --- Build mesh & export ---
    print("Building mesh…")
    build_and_export(
        elevation=elevation,
        grid_info=grid_info,
        center_lv95=center_lv95,
        radius_m=radius_m,
        track_lv95=track_lv95,
        track_alts=track_alts,
        output_path=args.output,
        diameter_mm=args.diameter,
        base_height_mm=args.base_height,
        exaggeration=args.exaggeration,
        track_width_mm=args.track_width,
        track_raise_mm=args.track_raise,
        track_intrude_mm=args.track_intrude,
        track_tolerance_mm=args.track_tolerance,
        water_polys_lv95=water_polys,
    )
    print("Done.")


if __name__ == "__main__":
    main()
