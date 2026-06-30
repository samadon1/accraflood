"""Central configuration: bounding boxes, date windows, GEE assets, thresholds, paths.

Everything tunable lives here so the pipeline modules stay declarative. Bounding boxes are
(min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
"""

from __future__ import annotations

import os
from pathlib import Path

# ----------------------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
REPO_DIR = PKG_DIR.parents[1]          # the accraflood/ repo root
DATA_DIR = Path(os.environ.get("ACCRAFLOOD_DATA", REPO_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_dotenv() -> None:
    """Read KEY=VALUE lines from a gitignored .env at the repo root (no dependency). Real
    environment variables take precedence, so this is only a local convenience."""
    envf = REPO_DIR / ".env"
    if not envf.exists():
        return
    for line in envf.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())


_load_dotenv()

# ----------------------------------------------------------------------------------------
# Google Earth Engine
# ----------------------------------------------------------------------------------------
# A Cloud project with the Earth Engine API enabled (the modern ee.Initialize(project=...)
# flow). Set EARTHENGINE_PROJECT in your environment or in a local .env file (see .env.example).
EE_PROJECT = os.environ.get("EARTHENGINE_PROJECT")

# Terrain source.
# Default is Copernicus GLO-30: official, permissively licensed, reliable. It is a SURFACE
# model, so it includes building rooftops and tree canopy. In theory that routes water over
# roofs, which is wrong for a city. The principled alternative is FABDEM (the same 30 m DEM
# with buildings and forests removed, i.e. bare earth) plus re-imposing building footprints
# as walls so flow is forced down streets. On the OSM check that variant came out within
# noise of plain Copernicus at 30 m, so we default to the simpler, license-clean asset and
# keep the variant one flag away. To enable it, set USE_FABDEM=True and BURN_BUILDINGS=True.
# Note: FABDEM is CC BY-NC and a third-party (sat-io) asset, so the variant is research-only.
DEM_ASSET = "COPERNICUS/DEM/GLO30"                       # surface model, band "DEM" (default)
FABDEM_ASSET = "projects/sat-io/open-datasets/FABDEM"    # bare-earth, opt-in
USE_FABDEM = False
BURN_BUILDINGS = False                                   # only meaningful with USE_FABDEM
BUILDING_BARRIER_M = 10.0

S1_ASSET = "COPERNICUS/S1_GRD"                           # Sentinel-1, ground-range-detected
OPEN_BUILDINGS = "GOOGLE/Research/open-buildings/v3/polygons"
OPEN_BUILDINGS_TEMPORAL = "GOOGLE/Research/open-buildings-temporal/v1"

# ----------------------------------------------------------------------------------------
# Bounding boxes  (min_lon, min_lat, max_lon, max_lat)
# ----------------------------------------------------------------------------------------
BBOXES: dict[str, tuple[float, float, float, float]] = {
    # Whole Greater Accra metro. Use once the small-bbox pipeline is proven (heavy run).
    "accra": (-0.32, 5.48, 0.02, 5.78),
    # Small test area over the Odaw channel, Alajo, and Kwame Nkrumah Circle corridor: the
    # heart of the recurrent flooding. Start here (about 4.5 x 4.5 km, fast EE exports).
    "odaw": (-0.235, 5.550, -0.190, 5.590),
}
DEFAULT_BBOX = "odaw"

# ----------------------------------------------------------------------------------------
# Date windows for Sentinel-1 validation (ISO yyyy-mm-dd, end-exclusive on EE filters)
# ----------------------------------------------------------------------------------------
# Dry-season reference (normal backscatter, channels mostly dry).
BASELINE_WINDOW = ("2026-01-10", "2026-03-01")
# Wet comparison window. Note: the June 29 2026 flash flood has no post-event Sentinel-1
# scene yet (next Accra pass is around July 2-3, not ingested as of June 30 2026). So this
# defaults to the peak-wet pre-flood period, i.e. observed seasonal inundation. Repoint it to
# the post-flood window and re-run `accraflood validate --method sar` once that scene lands.
EVENT_WINDOW = ("2026-06-01", "2026-06-28")
# Use one orbit direction so the baseline and event imaging geometry match (147 dominates).
S1_ORBIT_PASS = "ASCENDING"

# ----------------------------------------------------------------------------------------
# Hydrology, overlay, and SAR tuning
# ----------------------------------------------------------------------------------------
DEM_SCALE_M = 30                # Copernicus GLO-30 native resolution
# Project to a metric CRS so cells are true 30 m squares. Then flow-accumulation thresholds
# map to real drainage area and the metre-based channel buffers are exact. Accra is UTM 30N.
PROJECTED_CRS = "EPSG:32630"

# Stream extraction: a cell counts as channel if its upstream contributing area exceeds this
# many cells. With 30 m cells (900 m2 each), 200 cells is about 0.18 km2 minimum drainage.
STREAM_THRESHOLD_CELLS = 200

# Channel keep-out buffer (metres), sized by upstream flow accumulation: bigger upstream area
# means a larger channel and a wider buffer. width = BASE + SCALE * log10(accum_cells).
BUFFER_BASE_M = 15.0
BUFFER_SCALE_M = 12.0
BUFFER_MAX_M = 120.0

# Open Buildings confidence floor (drop low-confidence footprints).
BUILDING_CONFIDENCE_MIN = 0.70

# Sentinel-1 water detection: backscatter (dB) below this is candidate open water. The Otsu
# threshold on the event scene refines it per image.
S1_WATER_DB_MAX = -17.0
# Predicted ponding: a cell counts as ponding if its filled-depression depth (m) exceeds this.
SINK_PONDING_M = 0.3
# When aggregating the 10 m SAR water mask onto the 30 m DEM grid, a cell is "wet" if at least
# this fraction of its sub-pixels are water.
SAR_WET_FRACTION = 0.3

# OSM-waterway validation: match tolerances (m) between derived streams and mapped channels.
# 45 m is about 1.5 DEM cells, the positional slack expected from 30 m terrain.
OSM_MATCH_BUFFERS_M = [30, 45, 60, 90]
OSM_PRIMARY_BUFFER_M = 45
