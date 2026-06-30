"""Derive the natural drainage network from the DEM.

Backend: richdem (pure C++/GDAL, no fragile subprocess). We tried WhiteboxTools first but its
v2.4.0 binary panics on output-write on this platform. Pipeline:

    fill depressions ─► D8 flow accumulation ─► threshold to stream cells ─► vectorise (D8 links)

The "terrain remembers the channel" idea lives here: flow accumulation finds the watercourse
from elevation alone, even where it has been built over.

Outputs cached to ``data/``:
    filled_<bbox>.tif    depression-filled DEM
    accum_<bbox>.tif     flow accumulation (upstream cell count)
    sinks_<bbox>.tif     filled-minus-original depth → ponding candidates (used in validation)
    streams_<bbox>.parquet   vector stream network (LineStrings, with accum attributes)
"""

from __future__ import annotations

import contextlib
import math
import os
from pathlib import Path

from . import config

# D8 neighbour offsets (row, col) and their cell-centre distances (1 ortho, √2 diagonal).
_SQRT2 = math.sqrt(2)
_NB = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
       (-1, -1, _SQRT2), (-1, 1, _SQRT2), (1, -1, _SQRT2), (1, 1, _SQRT2)]


def paths(bbox_name: str) -> dict[str, Path]:
    d = config.DATA_DIR
    return {
        "dem": d / f"dem_{bbox_name}.tif",
        "filled": d / f"filled_{bbox_name}.tif",
        "accum": d / f"accum_{bbox_name}.tif",
        "sinks": d / f"sinks_{bbox_name}.tif",
        "streams_vec": d / f"streams_{bbox_name}.parquet",
    }


@contextlib.contextmanager
def _silence_c_stderr():
    """Mute richdem's C-level progress bars (they write straight to fd 2)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def _d8_downstream(filled):
    """Return, per cell, the flat index of its steepest-descent D8 neighbour (or -1 if none).

    Computed on the filled DEM so every non-outlet cell has a defined downslope path.
    """
    import numpy as np

    z = np.asarray(filled, dtype="float64")
    rows, cols = z.shape
    best_slope = np.full(z.shape, 0.0)
    down = np.full(z.shape, -1, dtype="int64")
    for dr, dc, dist in _NB:
        # Shift neighbour elevations into alignment; out-of-bounds → +inf (never chosen).
        shifted = np.full(z.shape, np.inf)
        r0, r1 = max(0, -dr), rows - max(0, dr)
        c0, c1 = max(0, -dc), cols - max(0, dc)
        shifted[r0:r1, c0:c1] = z[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        slope = (z - shifted) / dist
        better = slope > best_slope
        best_slope = np.where(better, slope, best_slope)
        # flat index of the neighbour cell
        rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        nidx = (rr + dr) * cols + (cc + dc)
        down = np.where(better, nidx, down)
    return down


def _burn_buildings(bbox_name, dem, transform, crs):
    """Add a barrier height to cells covered by a building footprint (street-routing)."""
    import numpy as np
    from rasterio.features import rasterize

    from . import fetch

    bpath = fetch.buildings_path(bbox_name)
    if not bpath.exists():
        return dem   # buildings not cached yet, so skip silently
    import geopandas as gpd

    buildings = gpd.read_parquet(bpath).to_crs(crs)
    mask = rasterize(
        ((g, 1) for g in buildings.geometry), out_shape=dem.shape,
        transform=transform, fill=0, dtype="uint8",
    ).astype(bool)
    out = dem.copy()
    out[mask] += config.BUILDING_BARRIER_M
    return out


def _write_raster(ref_path: Path, array, out_path: Path, nodata=None) -> None:
    import rasterio

    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
    profile.update(dtype="float32", count=1, compress="deflate")
    if nodata is not None:
        profile.update(nodata=nodata)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array.astype("float32"), 1)


def derive_drainage(bbox_name: str = config.DEFAULT_BBOX, force: bool = False):
    """Run the full hydrology chain. Returns the stream network as a GeoDataFrame."""
    import geopandas as gpd
    import numpy as np
    import rasterio
    import richdem as rd
    from shapely.geometry import LineString

    p = paths(bbox_name)
    if not p["dem"].exists():
        raise FileNotFoundError(
            f"DEM not found: {p['dem']}. Run `accraflood fetch --bbox-name {bbox_name}` first."
        )
    if p["streams_vec"].exists() and not force:
        return load_streams(bbox_name)

    with rasterio.open(p["dem"]) as src:
        dem = src.read(1).astype("float64")
        transform = src.transform
        crs = src.crs
        nd = src.nodata if src.nodata is not None else -9999.0

    # 0. Urban correction: raise building-footprint cells into walls so flow routes via streets.
    if config.BURN_BUILDINGS:
        dem = _burn_buildings(bbox_name, dem, transform, crs)

    # 1. Fill depressions (epsilon → tiny gradient across flats so flow is always defined).
    rd_dem = rd.rdarray(dem.copy(), no_data=nd)
    with _silence_c_stderr():
        rd.FillDepressions(rd_dem, epsilon=True, in_place=True)
        accum = np.asarray(rd.FlowAccumulation(rd_dem, method="D8"), dtype="float64")
    filled = np.asarray(rd_dem, dtype="float64")

    # 2. Persist rasters: filled DEM, accumulation, and sink depth (ponding candidates).
    _write_raster(p["dem"], filled, p["filled"])
    _write_raster(p["dem"], accum, p["accum"])
    _write_raster(p["dem"], np.clip(filled - dem, 0, None), p["sinks"], nodata=0)

    # 3. Threshold → stream cells; vectorise via D8 links (cell centre → downstream centre).
    rows, cols = accum.shape
    down = _d8_downstream(filled)
    cell_area_km2 = (config.DEM_SCALE_M ** 2) / 1e6
    stream_idx = np.flatnonzero(accum.ravel() >= config.STREAM_THRESHOLD_CELLS)

    def centre(flat_i):
        r, c = divmod(int(flat_i), cols)
        x, y = transform * (c + 0.5, r + 0.5)
        return x, y

    geoms, acc_cells = [], []
    for i in stream_idx:
        j = down.ravel()[i]
        if j < 0:
            continue                      # outlet cell, no downstream link
        geoms.append(LineString([centre(i), centre(j)]))
        acc_cells.append(float(accum.ravel()[i]))

    gdf = gpd.GeoDataFrame(
        {"accum_cells": acc_cells, "accum_km2": [a * cell_area_km2 for a in acc_cells]},
        geometry=geoms, crs=crs,
    )
    gdf.to_parquet(p["streams_vec"])
    return gdf


def load_streams(bbox_name: str = config.DEFAULT_BBOX):
    """Read the cached vector stream network."""
    import geopandas as gpd

    return gpd.read_parquet(paths(bbox_name)["streams_vec"])
