"""Fetch water body polygons from OpenStreetMap via the Overpass API."""

import hashlib
import json
import os
import requests

from geometry import wgs84_to_local, local_to_wgs84, is_swiss_area

try:
    from shapely.geometry import LineString, Polygon
    SHAPELY_OK = True
except ImportError:
    SHAPELY_OK = False

# Whitelist of "main rivers" rendered as 0.9 mm wide ribbons on the plate.
# Names are case-insensitive and matched after stripping whitespace; multilingual
# variants are listed because OpenStreetMap stores the locally dominant form.
MAIN_RIVER_NAMES = {
    # Largest Swiss rivers
    "rhein", "rhin", "rhine",
    "aare", "aar",
    "reuss", "reuß",
    "rhône", "rhone", "rotten",
    "ticino", "tessin",
    "inn", "en",
    # Major tributaries / well-known rivers
    "limmat",
    "sihl",
    "thur",
    "linth",
    "töss", "toess",
    "birs",
    "birsig",
    "saane", "sarine",
    "doubs",
    "glatt",
    "verzasca",
    "maggia",
    "emme",
    "kleine emme",
    "murg",
    "necker",
    "suhre",
    "wigger",
    "wiese",
    "lorze",
    "engelberger aa",
    "sarner aa",
    "muota",
    "broye",
    "orbe",
    "areuse",
    "versoix",
    "arve",
    "venoge",
    "goldach",
}

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
    min_area_m2: float = 8_000_000,
    include_rivers: bool = False,
) -> tuple[list, list]:
    """
    Query Overpass for natural=water / landuse=reservoir polygons in the area.

    Returns: (polygons, rivers)
      polygons - list of shapely Polygon (lakes, reservoirs) with area
                 >= min_area_m2.  Default 8 km².
      rivers   - list of shapely LineString in LV95.  Only main Swiss rivers
                 (see MAIN_RIVER_NAMES) are returned, and only when
                 include_rivers is True.

    Both lists are empty if shapely is unavailable or the request fails.
    """
    if not SHAPELY_OK:
        print("  shapely not installed — skipping water bodies.")
        return [], []

    # Local centre + radius → WGS84 bounding box
    ce, cn = center_lv95
    lons, lats = local_to_wgs84(
        [ce - radius_m, ce + radius_m],
        [cn - radius_m, cn + radius_m],
    )
    south, north = min(lats), max(lats)
    west,  east  = min(lons), max(lons)

    # For very large bboxes, `out geom;` silently drops the largest multipolygon
    # relations (we observed Lago di Garda disappearing while smaller lakes in
    # the same area came through).  Split the bbox into tiles of <=1.5° per
    # side and union the per-tile responses.
    elements = _fetch_in_tiles(south, west, north, east, include_rivers,
                                max_tile_deg=1.5)
    if elements is None:
        print("  Warning: all Overpass mirrors failed — skipping water bodies.")
        return [], []

    def nodes_to_lv95(nodes):
        lons_ = [n["lon"] for n in nodes]
        lats_ = [n["lat"] for n in nodes]
        e, n = wgs84_to_local(lats_, lons_)
        return list(zip(e, n))

    all_polys = []
    river_lines_lv95: list = []
    for elem in elements:
        tags = elem.get("tags", {}) or {}
        is_river_way = (
            elem["type"] == "way"
            and tags.get("waterway") == "river"
        )
        if is_river_way:
            # Keep main rivers (by name) as LineStrings — rendered as a
            # fixed-width ribbon by mesh.py so they survive on small prints.
            # In Switzerland we apply the curated whitelist; outside CH we
            # accept every named river (OSM tags rivers worldwide).
            name = (tags.get("name") or "").strip().lower()
            keep = name and (
                name in MAIN_RIVER_NAMES if is_swiss_area() else True
            )
            if keep:
                geom = elem.get("geometry", [])
                if len(geom) >= 2:
                    try:
                        line = LineString(nodes_to_lv95(geom))
                        if line.is_valid and not line.is_empty:
                            river_lines_lv95.append(line)
                    except Exception:
                        pass
            continue

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

    if river_lines_lv95:
        # Merge segments that belong to the same river into longer LineStrings
        # before returning, so the rendered ribbon doesn't have gaps.
        try:
            from shapely.ops import linemerge
            from shapely.geometry import MultiLineString
            merged = linemerge(MultiLineString(river_lines_lv95))
            if isinstance(merged, LineString):
                river_lines_lv95 = [merged]
            else:
                river_lines_lv95 = [g for g in merged.geoms
                                     if isinstance(g, LineString) and not g.is_empty]
        except Exception:
            pass
        print(f"  {len(river_lines_lv95)} main river segment(s) kept")

    return polys, river_lines_lv95


def _build_query(south: float, west: float, north: float, east: float,
                  include_rivers: bool) -> str:
    bb = f"({south},{west},{north},{east})"
    exclude = '["water"!="river"]["water"!="canal"]["water"!="stream"]'
    river_lines = (
        f'  way["waterway"="river"]["name"]{bb};\n'
        if include_rivers else ""
    )
    return (
        f"[out:json][timeout:180];\n"
        f"(\n"
        f'  way["natural"="water"]{exclude}{bb};\n'
        f'  relation["natural"="water"]["type"="multipolygon"]{exclude}{bb};\n'
        f'  way["landuse"="reservoir"]{bb};\n'
        f'  relation["landuse"="reservoir"]["type"="multipolygon"]{bb};\n'
        f"{river_lines}"
        f");\n"
        f"out geom;\n"
    )


def _fetch_in_tiles(south: float, west: float, north: float, east: float,
                     include_rivers: bool, max_tile_deg: float = 1.5):
    """Run the Overpass query in tiles of up to max_tile_deg per side and
    merge the results.  Overpass quietly truncates the response on very
    large `out geom;` queries (large multipolygon relations like Lago di
    Garda go missing), so tiling keeps every response small enough to come
    back complete."""
    import math
    lat_span = north - south
    lon_span = east - west

    # Single-query fast path for small bboxes.
    if lat_span <= max_tile_deg and lon_span <= max_tile_deg:
        return _fetch_with_cache(_build_query(south, west, north, east,
                                              include_rivers))

    n_lat = max(1, math.ceil(lat_span / max_tile_deg))
    n_lon = max(1, math.ceil(lon_span / max_tile_deg))
    print(f"  Splitting Overpass request into {n_lat}x{n_lon} tiles")

    merged_by_key: dict = {}   # (type, id) -> element  (dedup across tiles)
    for i in range(n_lat):
        s = south + lat_span * i / n_lat
        n = south + lat_span * (i + 1) / n_lat
        for j in range(n_lon):
            w = west + lon_span * j / n_lon
            e = west + lon_span * (j + 1) / n_lon
            tile_elems = _fetch_with_cache(_build_query(s, w, n, e,
                                                         include_rivers))
            if not tile_elems:
                continue
            for el in tile_elems:
                key = (el.get("type"), el.get("id"))
                if key not in merged_by_key:
                    merged_by_key[key] = el
    return list(merged_by_key.values())


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
