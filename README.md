# gps-terrain-stl

Convert a GPS track (GPX or IGC) into a 3D-printable terrain model — a circular disc with the track path and lakes rendered as separate, interlocking parts.

## Example output

A 100 mm disc centred on a half-marathon route or paragliding flight, with realistic terrain relief, the track path as a raised/inset tube, and lakes as flat inset plates.

## Requirements

- Python 3.10+
- Dependencies:

```bash
pip install -r requirements.txt
```

Elevation data is fetched automatically:
- **SwissALTI3D** (Swisstopo STAC API) — 2 m resolution, Switzerland only; tiles cached in `~/.cache/gps-terrain-stl/swissalti3d/`
- **Copernicus DEM GLO-30** (AWS S3, public) — 30 m global coverage, used as fallback; tiles cached in `~/.cache/gps-terrain-stl/copernicus/`

Water body data is fetched from **OpenStreetMap** via the Overpass API and cached locally in `~/.cache/gps-terrain-stl/water/`.

## Usage

```bash
python3 main.py <input.gpx|input.igc> [options]
```

The output filename defaults to the input filename with a `.stl` extension. Both `.stl` and `.3mf` are always written.

### Examples

```bash
# GPX track — defaults
python3 main.py Halbmarathon.gpx

# IGC flight — larger disc and vertical exaggeration
python3 main.py flight.igc --diameter 120 --exaggeration 2.0

# Skip water detection
python3 main.py Halbmarathon.gpx --no-water
```

## Options

| Option | Default | Description |
|---|---|---|
| `-o / --output` | input filename + `.stl` | Output file path |
| `--diameter` | `100` | Disc diameter in mm |
| `--padding` | `0.20` | Radial padding around track (0.20 = 20 %) |
| `--exaggeration` | `1.0` | Vertical scale multiplier |
| `--resolution` | `512` | Elevation raster grid size in pixels (max 2000) |
| `--base-height` | `3.0` | Bottom plinth thickness in mm |
| `--track-width` | `0.6` | Track tube width in mm |
| `--track-raise` | `0.6` | How far the track rises above terrain in mm (GPX) |
| `--track-intrude` | `2.0` | How deep the track cuts into terrain in mm (GPX) |
| `--track-tolerance` | `0.2` | Clearance gap between track and terrain groove in mm |
| `--min-water-area` | `500000` | Minimum lake area in m² to include (default = 50 ha) |
| `--rivers` | off | Include rivers and riverbanks as water plates |
| `--no-water` | off | Skip water body detection entirely |

## Output parts

The STL contains two or three separate bodies intended to be printed in different colours and assembled:

1. **Terrain disc** — solid circular disc with elevation relief, a groove cut along the track path, and grooves cut for any lake outlines.
2. **Track tube** — swept rectangular tube following the GPS path. Sits in the terrain groove with clearance defined by `--track-tolerance`. The tube has a flat bottom at the deepest point of its groove so it lies flush.
3. **Lake plates** *(optional)* — 1 mm thick flat plates, one per detected lake, inset into the terrain surface. Where the track crosses a lake the track groove is subtracted from the plate so the two parts don't intersect.

## Track rendering

- **GPX** (running, hiking, cycling): track snaps to the terrain surface and is raised above it by `--track-raise` and inset below it by `--track-intrude`.
- **IGC** (paragliding, gliding): uses recorded GPS/barometric altitude directly; tube is symmetric around the recorded altitude.

## Architecture

```
parse.py      — reads GPX/IGC into (lat, lon, alt) point arrays
geometry.py   — WGS84 → LV95 (EPSG:2056) reprojection, centroid, bounding radius
elevation.py  — fetches elevation raster (SwissALTI3D or Copernicus DEM)
water.py      — fetches lake polygons from OpenStreetMap Overpass API, with local cache
mesh.py       — builds terrain disc, track tube, and lake plates; exports STL + 3MF
main.py       — CLI entry point, orchestrates the pipeline
```
