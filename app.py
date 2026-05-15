"""Flask web interface for gps-terrain-stl."""

import glob
import io
import os
import sys
import tempfile
import zipfile

from flask import Flask, render_template, request, send_file, jsonify

import numpy as np

from parse import load_track
from geometry import compute_geometry, wgs84_to_lv95
from elevation import fetch_elevation
from mesh import build_and_export
from water import fetch_water_bodies

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    f = request.files.get("gpx_file")
    if not f or f.filename == "":
        return jsonify({"error": "No file uploaded"}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in (".gpx", ".igc"):
        return jsonify({"error": "Only .gpx and .igc files are supported"}), 400

    # Read parameters from form
    diameter = float(request.form.get("diameter", 100.0))
    shape = request.form.get("shape", "circle")
    if shape not in ("circle", "square", "hexagon", "rectangle"):
        shape = "circle"
    # Rectangle dimensions (only used when shape == "rectangle").
    rect_width = float(request.form.get("rect_width", 100.0))
    rect_height = float(request.form.get("rect_height", 60.0))
    # Edge margin in mm: minimum distance between outermost track point and disc edge.
    # Shape-aware: the inscribed-circle radius is smaller than the half-extent for
    # a hexagon, and for a rectangle is set by the shorter side.
    edge_margin_mm = float(request.form.get("edge_margin", 10.0))
    if shape == "rectangle":
        disc_radius_mm = max(rect_width, rect_height) / 2.0
        inscribed_radius_mm = min(rect_width, rect_height) / 2.0
    elif shape == "hexagon":
        disc_radius_mm = diameter / 2.0
        inscribed_radius_mm = disc_radius_mm * (3 ** 0.5) / 2.0
    else:
        disc_radius_mm = diameter / 2.0
        inscribed_radius_mm = disc_radius_mm
    track_radius_mm = max(inscribed_radius_mm - edge_margin_mm, 1.0)
    padding = (disc_radius_mm - track_radius_mm) / track_radius_mm
    exaggeration = float(request.form.get("exaggeration", 1.0))
    resolution = int(request.form.get("resolution", 512))
    base_height = float(request.form.get("base_height", 3.0))
    track_width = float(request.form.get("track_width", 0.6))
    track_raise = float(request.form.get("track_raise", 0.6))
    track_intrude = float(request.form.get("track_intrude", 2.0))
    track_tolerance = float(request.form.get("track_tolerance", 0.2))
    min_water_area = float(request.form.get("min_water_area", 500000))
    rivers = request.form.get("rivers") == "on"
    no_water = request.form.get("no_water") == "on"

    # Save uploaded file to temp dir
    tmp_dir = tempfile.mkdtemp()
    input_path = os.path.join(tmp_dir, f.filename)
    f.save(input_path)

    output_path = os.path.splitext(input_path)[0] + ".stl"
    base_name = os.path.splitext(f.filename)[0]

    try:
        # Load track
        points = load_track(input_path)

        # Geometry
        center_lv95, radius_m = compute_geometry(points, padding=padding)

        # Elevation
        elevation, grid_info = fetch_elevation(center_lv95, radius_m, resolution)

        # Track in LV95
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        east, north = wgs84_to_lv95(lats, lons)
        track_lv95 = list(zip(east.tolist(), north.tolist()))
        track_alts = [p[2] for p in points] if len(points[0]) > 2 else None

        # Water bodies
        if no_water:
            water_polys = []
        else:
            water_polys = fetch_water_bodies(
                center_lv95, radius_m,
                min_area_m2=min_water_area,
                include_rivers=rivers,
            )

        # Build mesh & export (creates separate STL files)
        build_and_export(
            elevation=elevation,
            grid_info=grid_info,
            center_lv95=center_lv95,
            radius_m=radius_m,
            track_lv95=track_lv95,
            track_alts=track_alts,
            output_path=output_path,
            diameter_mm=diameter,
            base_height_mm=base_height,
            exaggeration=exaggeration,
            track_width_mm=track_width,
            track_raise_mm=track_raise,
            track_intrude_mm=track_intrude,
            track_tolerance_mm=track_tolerance,
            water_polys_lv95=water_polys,
            shape=shape,
            rect_width_mm=rect_width,
            rect_height_mm=rect_height,
        )

        # Collect the separate STL part files into a ZIP (only STL files)
        stl_base = os.path.splitext(input_path)[0]
        part_suffixes = ["_terrain", "_track", "_water"]
        stl_files = [f"{stl_base}{s}.stl" for s in part_suffixes
                      if os.path.exists(f"{stl_base}{s}.stl")]

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for stl_path in stl_files:
                arcname = base_name + os.path.basename(stl_path)[len(os.path.basename(stl_base)):]
                zf.write(stl_path, arcname)
        buf.seek(0)

        return send_file(
            buf,
            as_attachment=True,
            download_name=f"{base_name}.zip",
            mimetype="application/zip",
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
