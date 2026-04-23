"""Fetch water body polygons from OpenStreetMap via the Overpass API."""

import hashlib
import json
import os
import requests
from pyproj import Transformer

try:
    from shapely.geometry import Polygon
    SHAPELY_OK = True
except ImportError:
    SHAPELY_OK = False

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]

# Overpass enforces a usage policy that requires a descriptive User-Agent
# and an Accept header; without these, some mirrors return 406/403.
REQUEST_HEADERS = {
    "User-Agent": "gps-terrain-stl/1.0 (https://github.com/ChristophSiegenthaler/gps-terrain-stl)",
    "Accept": "application/json",
}

CACHE_DIR = os.path.expanduser("~/.cache/gps-terrain-stl/water")


def fetch_water_bodies(
    center_lv95: tuple,
    radius_m: float,
    min_area_m2: float = 100_000,
    include_rivers: bool = False,
) -> list:
    """
    Query Overpass for natural=water / landuse=reservoir polygons in the area.

    When include_rivers is True also fetches natural=water + water=river and
    waterway=riverbank polygons (disabled by default because they produce thin
    elongated plates that can look odd on small discs).

    Results are cached locally so repeated runs skip the network request.
    Returns a list of shapely Polygon objects in LV95 coordinates,
    or an empty list if shapely is unavailable or the request fails.
    """
    if not SHAPELY_OK:
        print("  shapely not installed — skipping water bodies.")
        return []

    # LV95 centre + radius → WGS84 bounding box
    to_wgs84 = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    ce, cn = center_lv95
    lons, lats = to_wgs84.transform(
        [ce - radius_m, ce + radius_m],
        [cn - radius_m, cn + radius_m],
    )
    south, north = min(lats), max(lats)
    west,  east  = min(lons), max(lons)

    bb = f"({south},{west},{north},{east})"

    if include_rivers:
        # Include rivers; still exclude canals and streams.
        exclude = '["water"!="canal"]["water"!="stream"]'
        river_lines = (
            f'  way["waterway"="riverbank"]{bb};\n'
            f'  relation["waterway"="riverbank"]["type"="multipolygon"]{bb};\n'
        )
    else:
        # Exclude rivers, canals and streams — they appear as thin elongated
        # polygons that look wrong on a terrain disc.
        exclude = '["water"!="river"]["water"!="canal"]["water"!="stream"]'
        river_lines = ""

    query = (
        f"[out:json][timeout:60];\n"
        f"(\n"
        f'  way["natural"="water"]{exclude}{bb};\n'
        f'  relation["natural"="water"]["type"="multipolygon"]{exclude}{bb};\n'
        f'  way["landuse"="reservoir"]{bb};\n'
        f'  relation["landuse"="reservoir"]["type"="multipolygon"]{bb};\n'
        f"{river_lines}"
        f");\n"
        f"out geom;\n"
    )

    elements = _fetch_with_cache(query)
    if elements is None:
        print("  Warning: all Overpass mirrors failed — skipping water bodies.")
        return []

    to_lv95 = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)

    def nodes_to_lv95(nodes):
        lons_ = [n["lon"] for n in nodes]
        lats_ = [n["lat"] for n in nodes]
        e, n = to_lv95.transform(lons_, lats_)
        return list(zip(e, n))

    all_polys = []
    for elem in elements:
        if elem["type"] == "way":
            geom = elem.get("geometry", [])
            if len(geom) < 3:
                continue
            try:
                p = Polygon(nodes_to_lv95(geom))
                if p.is_valid and not p.is_empty:
                    all_polys.append(p)
            except Exception:
                pass

        elif elem["type"] == "relation":
            # Collect the raw node sequences for each role, then assemble
            # into closed rings. Large lakes (e.g. Zürichsee) split their
            # boundary across many OSM ways that must be stitched together.
            outer_ways, inner_ways = [], []
            for member in elem.get("members", []):
                geom = member.get("geometry", [])
                if len(geom) < 2:
                    continue
                coords = [(n["lon"], n["lat"]) for n in geom]
                if member.get("role") == "inner":
                    inner_ways.append(coords)
                else:
                    outer_ways.append(coords)

            outer_rings = _assemble_rings(outer_ways)
            inner_rings = _assemble_rings(inner_ways)

            for outer in outer_rings:
                try:
                    outer_lv95 = nodes_to_lv95(
                        [{"lon": c[0], "lat": c[1]} for c in outer]
                    )
                    inner_lv95_list = [
                        nodes_to_lv95([{"lon": c[0], "lat": c[1]} for c in ir])
                        for ir in inner_rings
                    ]
                    p = Polygon(outer_lv95, inner_lv95_list)
                    if not p.is_valid:
                        p = p.buffer(0)
                    if p.is_valid and not p.is_empty:
                        all_polys.append(p)
                except Exception:
                    pass

    if all_polys:
        areas = [p.area for p in all_polys]
        print(f"  Found {len(all_polys)} raw polygon(s), "
              f"area range: {min(areas):.0f} – {max(areas):.0f} m²")
        polys = [p for p in all_polys if p.area >= min_area_m2]
        print(f"  {len(polys)} polygon(s) kept after area filter (>= {min_area_m2:,.0f} m²)")
    else:
        polys = []

    return polys


def _fetch_with_cache(query: str) -> list | None:
    """
    Return Overpass elements for the query, using a local cache to avoid
    repeated network calls for the same query.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_key = hashlib.sha256(query.encode()).hexdigest()[:16]
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")

    if os.path.exists(cache_path):
        print(f"  Using cached water data ({cache_path})")
        with open(cache_path) as f:
            return json.load(f)

    for mirror in OVERPASS_MIRRORS:
        # Try POST first, then GET (some mirrors handle one better)
        for method, kwargs in [
            ("POST", {"data": {"data": query}}),
            ("GET",  {"params": {"data": query}}),
        ]:
            try:
                print(f"  Trying {mirror} ({method}) …")
                resp = requests.request(
                    method, mirror, timeout=(10, 60),
                    headers=REQUEST_HEADERS, **kwargs
                )
                resp.raise_for_status()
                elements = resp.json().get("elements", [])
                with open(cache_path, "w") as f:
                    json.dump(elements, f)
                print(f"  Cached to {cache_path}")
                return elements
            except Exception as exc:
                print(f"  Failed: {exc}")

    return None


def _assemble_rings(ways: list) -> list:
    """
    Stitch a list of open way coordinate sequences into closed rings.

    Each way is a list of (lon, lat) tuples. Ways are connected end-to-end
    (reversing if necessary) until the ring closes or no further extension
    is possible. Returns a list of closed rings (each a list of tuples).
    """
    remaining = [list(w) for w in ways]
    rings = []

    while remaining:
        ring = remaining.pop(0)

        while True:
            if len(ring) >= 2 and _coords_match(ring[0], ring[-1]):
                break  # closed

            extended = False
            for i, way in enumerate(remaining):
                if _coords_match(ring[-1], way[0]):
                    ring.extend(way[1:])
                    remaining.pop(i)
                    extended = True
                    break
                if _coords_match(ring[-1], way[-1]):
                    ring.extend(reversed(way[:-1]))
                    remaining.pop(i)
                    extended = True
                    break
            if not extended:
                break  # open ring — keep as-is

        if len(ring) >= 3:
            rings.append(ring)

    return rings


def _coords_match(a, b, tol: float = 1e-7) -> bool:
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol
