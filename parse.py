"""Parse GPX and IGC GPS track files. Returns list of (lat, lon) in WGS84."""

from pathlib import Path
import gpxpy


def load_track(filepath: str) -> list[tuple[float, float]]:
    """Load a GPS track from a .gpx or .igc file."""
    path = Path(filepath)
    suffix = path.suffix.lower()
    if suffix == ".gpx":
        return _parse_gpx(path)
    elif suffix == ".igc":
        return _parse_igc(path)
    else:
        raise ValueError(f"Unsupported format: {suffix!r}. Use .gpx or .igc")


def _parse_gpx(path: Path) -> list[tuple[float, float]]:
    with open(path) as f:
        gpx = gpxpy.parse(f)

    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                points.append((pt.latitude, pt.longitude))

    if not points:
        for route in gpx.routes:
            for pt in route.points:
                points.append((pt.latitude, pt.longitude))

    if not points:
        raise ValueError("No track points found in GPX file")
    return points


def _parse_igc(path: Path) -> list[tuple[float, float]]:
    """Parse IGC B-record fix lines.

    B-record layout (0-indexed):
      [0]     'B'
      [1:7]   HHMMSS
      [7:9]   lat degrees
      [9:11]  lat minutes
      [11:14] lat decimal minutes (thousandths)
      [14]    N/S
      [15:18] lon degrees
      [18:20] lon minutes
      [20:23] lon decimal minutes (thousandths)
      [23]    E/W
      [24]    validity (A=3D fix, V=2D/invalid)
      [25:30] pressure altitude (m)
      [30:35] GPS altitude (m)
    """
    points = []
    with open(path, encoding="ascii", errors="replace") as f:
        for line in f:
            if not line.startswith("B") or len(line) < 35:
                continue
            try:
                # Only accept valid 3D fixes
                if line[24] != "A":
                    continue

                lat = int(line[7:9]) + (int(line[9:11]) + int(line[11:14]) / 1000.0) / 60.0
                if line[14] == "S":
                    lat = -lat

                lon = int(line[15:18]) + (int(line[18:20]) + int(line[20:23]) / 1000.0) / 60.0
                if line[23] == "W":
                    lon = -lon

                gps_alt = int(line[30:35])
                points.append((lat, lon, gps_alt))
            except (ValueError, IndexError):
                continue

    if not points:
        raise ValueError("No valid A-fix B-records found in IGC file")

    # Thin the track: skip points closer than ~20m to the previous kept point.
    # IGC records every second; at typical flight speeds this is far more
    # resolution than a 100mm print needs.
    thinned = [points[0]]
    for pt in points[1:]:
        lat, lon = pt[0], pt[1]
        plat, plon = thinned[-1][0], thinned[-1][1]
        # Rough metre distance (good enough for thinning)
        dlat = (lat - plat) * 111320
        dlon = (lon - plon) * 111320 * abs(lat * 3.14159 / 180)
        if (dlat**2 + dlon**2) ** 0.5 >= 20:
            thinned.append(pt)

    print(f"  IGC: {len(points)} fixes → {len(thinned)} after thinning (A-only, ≥20 m)")
    return thinned
