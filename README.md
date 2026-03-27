# gps-terrain-stl

Convert a GPS track (GPX or IGC) into a 3D printable terrain model — a circular disc with the track raised on the surface.

## Example output

A 100 mm disc centred on a half-marathon route or paragliding flight, with realistic terrain relief and the track path rendered as a tube.

## Requirements

- Python 3.10+
- Dependencies (install via pip):

```bash
pip install -r requirements.txt
```

Elevation data is fetched automatically:
- **SwissALTI3D** (Swisstopo WCS) — high resolution, Switzerland only
- **Copernicus DEM GLO-30** (AWS S3, public) — 30 m global coverage, used as fallback

## Usage

```bash
python3 main.py <input.gpx|input.igc> [options]
```

### Examples

```bash
# GPX track with default settings
python3 main.py Halbmarathon.gpx -o halbmarathon_terrain.stl

# IGC flight with larger disc and vertical exaggeration
python3 main.py flight.igc --diameter 120 --exaggeration 2.0 -o flight_terrain.stl
```

Output is always written as both `.stl` (binary) and `.3mf` alongside the specified path.

## Options

| Option | Default | Description |
|---|---|---|
| `--diameter` | `100` | Disc diameter in mm |
| `--padding` | `0.20` | Radial padding around track (0.20 = 20%) |
| `--exaggeration` | `1.0` | Vertical scale multiplier |
| `--resolution` | `512` | Elevation raster grid size in pixels (max 2000) |
| `--base-height` | `3.0` | Bottom plinth thickness in mm |
| `--track-width` | `0.6` | Track tube width in mm |
| `--track-raise` | `0.6` | How far the track rises above the terrain in mm (GPX) |
| `--track-intrude` | `2.0` | How deep the track cuts into the terrain in mm (GPX) |
| `-o / --output` | `terrain.stl` | Output file path |

## Track rendering

- **GPX** (hiking, running, cycling): track snaps to terrain surface and is raised/cut in via `--track-raise` / `--track-intrude`.
- **IGC** (paragliding, gliding): uses recorded GPS/barometric altitude directly; tube is symmetric around the recorded altitude.

## Architecture

```
parse.py      — reads GPX/IGC into (lat, lon, alt) point arrays
geometry.py   — WGS84 → LV95 (EPSG:2056) reprojection, centroid, bounding radius
elevation.py  — fetches and caches elevation raster (SwissALTI3D or Copernicus)
mesh.py       — builds watertight circular disc STL and exports STL + 3MF
main.py       — CLI entry point, orchestrates the pipeline
```
