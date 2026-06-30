"""Flag and rank buildings encroaching on the derived drainage.

Each stream segment is buffered by a width that grows with its upstream flow accumulation
(bigger channel → wider keep-out). Buildings intersecting a buffer are flagged; each is scored
by the largest channel (by drainage area) it sits on, then ranked. The buffer is a geometric
proxy for "in the watercourse": it ranks candidates, it does not adjudicate.

Outputs:
    encroachers_<bbox>.parquet   flagged buildings (metric CRS) with score + rank
    encroachers_<bbox>.geojson   same, EPSG:4326, for mapping/portability
"""

from __future__ import annotations

from pathlib import Path

from . import config


def paths(bbox_name: str) -> dict[str, Path]:
    d = config.DATA_DIR
    return {
        "parquet": d / f"encroachers_{bbox_name}.parquet",
        "geojson": d / f"encroachers_{bbox_name}.geojson",
        "buffers": d / f"channel_buffers_{bbox_name}.parquet",
    }


def _buffer_width(accum_cells):
    """Channel keep-out radius (m): BASE + SCALE·log10(upstream cells), capped at MAX."""
    import numpy as np

    w = config.BUFFER_BASE_M + config.BUFFER_SCALE_M * np.log10(np.maximum(accum_cells, 1.0))
    return np.minimum(w, config.BUFFER_MAX_M)


def flag_encroachers(bbox_name: str = config.DEFAULT_BBOX, force: bool = False):
    """Spatial-join buildings against channel buffers; return ranked encroachers."""
    import geopandas as gpd

    from . import fetch, hydrology

    p = paths(bbox_name)
    if p["parquet"].exists() and not force:
        return gpd.read_parquet(p["parquet"])

    streams = hydrology.load_streams(bbox_name)            # metric CRS, has accum_cells
    bpath = fetch.buildings_path(bbox_name)
    if not bpath.exists():
        raise FileNotFoundError(
            f"Buildings not cached: {bpath}. Run `accraflood fetch --bbox-name {bbox_name}`."
        )
    buildings = gpd.read_parquet(bpath)

    # Per-segment buffer sized by upstream accumulation.
    buffers = streams.copy()
    buffers["width_m"] = _buffer_width(streams["accum_cells"].to_numpy())
    buffers["geometry"] = streams.geometry.buffer(buffers["width_m"].to_numpy())
    buffers.to_parquet(p["buffers"])

    # A building encroaches if it intersects any buffer; keep the worst (largest) channel.
    hits = gpd.sjoin(
        buildings, buffers[["geometry", "accum_cells", "accum_km2", "width_m"]],
        predicate="intersects", how="inner",
    )
    agg = hits.groupby(level=0).agg(
        channel_accum_cells=("accum_cells", "max"),
        channel_accum_km2=("accum_km2", "max"),
        channel_width_m=("width_m", "max"),
    )

    flagged = buildings.loc[agg.index].copy()
    for col in agg.columns:
        flagged[col] = agg[col].to_numpy()
    flagged["score"] = flagged["channel_accum_km2"]        # drainage area of its watercourse
    flagged = flagged.sort_values("score", ascending=False).reset_index(drop=True)
    flagged.insert(0, "rank", range(1, len(flagged) + 1))

    flagged.to_parquet(p["parquet"])
    flagged.to_crs(4326).to_file(p["geojson"], driver="GeoJSON")
    return flagged
