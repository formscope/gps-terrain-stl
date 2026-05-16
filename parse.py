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

    cleaned = _clean_track(points)
    if len(cleaned) < len(points):
        print(f"  GPX: {len(points)} raw points -> {len(cleaned)} after dedup / spike removal")
    return cleaned


def _clean_track(points: list[tuple[float, float]],
                 dup_tol_deg: float = 1.5e-5) -> list[tuple[float, float]]:
    """Remove duplicate and spike artefacts from a GPS track.

    Some route planners (e.g. MySchweizMobil) insert tiny "marker" detours
    that look like:  A → B → B → A → next, with B less than ~1 m from A.
    When the track is later buffered to a tube these spurs become visible
    micro-spikes ("ausbrechende Formen") on the printed track.

    Two-pass cleanup:
      1. collapse consecutive duplicates within `dup_tol_deg` of each other;
      2. drop any middle point whose neighbours coincide within tolerance —
         i.e. a back-and-forth spike around a single location.
    Final dedup pass catches duplicates produced by step 2.

    `dup_tol_deg` ≈ 1.5e-5° ≈ 1.7 m in latitude.  Real out-and-back
    sections in a route are at least an order of magnitude longer than
    this and are therefore preserved.
    """
    if len(points) < 3:
        return points

    def close(a, b):
        return abs(a[0] - b[0]) < dup_tol_deg and abs(a[1] - b[1]) < dup_tol_deg

    # Pass 1: consecutive dedup
    deduped = [points[0]]
    for pt in points[1:]:
        if not close(pt, deduped[-1]):
            deduped.append(pt)

    # Pass 2: drop "spike" middles where the neighbours collapse onto each other
    pruned = [deduped[0]]
    for i in range(1, len(deduped) - 1):
        before = deduped[i - 1]
        after = deduped[i + 1]
        if close(before, after):
            continue   # spike — middle point is an artefact
        pruned.append(deduped[i])
    pruned.append(deduped[-1])

    # Pass 3: dedup again (pass 2 may have left two identical neighbours)
    final = [pruned[0]]
    for pt in pruned[1:]:
        if not close(pt, final[-1]):
            final.append(pt)
    return final


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
