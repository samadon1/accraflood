"""Render pipeline outputs as interactive HTML maps (folium, via GeoDataFrame.explore).

Layers, added as each slice lands: derived drainage network, flagged buildings (coloured by
score), SAR observed flood extent, temporal encroachment (coloured by first-appearance year).
``run`` composes them into one combined map.
"""

from __future__ import annotations

from pathlib import Path

from . import config

# Cap the stream segments embedded in a map. A metro run derives ~300k segments, and folium
# embeds each as GeoJSON, so an uncapped map is hundreds of MB and freezes the browser. The
# largest-drainage channels are the meaningful ones to show anyway.
STREAM_RENDER_TOP = 4000


def _streams_for_render(bbox_name: str):
    """Load the derived streams, keeping only the top channels by accumulation, in EPSG:4326."""
    from . import hydrology

    s = hydrology.load_streams(bbox_name)
    if len(s) > STREAM_RENDER_TOP:
        s = s.nlargest(STREAM_RENDER_TOP, "accum_cells")
    return s.to_crs(4326)


def render_drainage(bbox_name: str = config.DEFAULT_BBOX) -> Path:
    """Plot the derived drainage network over a basemap and save an HTML map."""
    gdf = _streams_for_render(bbox_name)
    m = gdf.explore(
        color="#1d6fb8",
        style_kwds={"weight": 2},
        tiles="CartoDB positron",
        name="natural drainage",
        tooltip=False,
    )
    out = config.DATA_DIR / f"drainage_{bbox_name}.html"
    m.save(str(out))
    return out


def render_overlay(bbox_name: str = config.DEFAULT_BBOX, top: int = 3000) -> Path:
    """Drainage + flagged buildings coloured by score (top-N by score for map performance)."""
    import geopandas as gpd

    from . import overlay

    streams = _streams_for_render(bbox_name)
    flagged = gpd.read_parquet(overlay.paths(bbox_name)["parquet"]).to_crs(4326)
    shown = flagged.head(top)

    m = streams.explore(
        color="#1d6fb8", style_kwds={"weight": 1.2},
        tiles="CartoDB positron", name="natural drainage", tooltip=False,
    )
    shown.explore(
        m=m, column="score", cmap="Reds", legend=True,
        name=f"encroachers (top {len(shown)})",
        style_kwds={"weight": 0, "fillOpacity": 0.75},
        tooltip=["rank", "score", "channel_accum_km2"],
    )
    out = config.DATA_DIR / f"overlay_{bbox_name}.html"
    m.save(str(out))
    return out


def render_validation(bbox_name: str = config.DEFAULT_BBOX) -> Path:
    """PNG comparing SAR-observed new-water vs predicted ponding (with agreement in green)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio

    from . import validate

    p = validate.paths(bbox_name)
    with rasterio.open(p["obs_water"]) as s:
        obs = s.read(1).astype(bool)
        ext = [s.bounds.left, s.bounds.right, s.bounds.bottom, s.bounds.top]
    with rasterio.open(p["pred_pond"]) as s:
        pred = s.read(1).astype(bool)

    # RGB: predicted-only=orange, observed-only=blue, agreement=green.
    rgb = np.ones((*obs.shape, 3))
    rgb[pred & ~obs] = [0.95, 0.55, 0.15]
    rgb[obs & ~pred] = [0.20, 0.45, 0.85]
    rgb[obs & pred] = [0.15, 0.65, 0.25]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb, extent=ext)
    ax.set_title(f"{bbox_name}: SAR water vs predicted ponding\n"
                 "green=agreement  orange=predicted-only  blue=observed-only")
    ax.set_xticks([]); ax.set_yticks([])
    out = config.DATA_DIR / f"validation_{bbox_name}.png"
    fig.tight_layout(); fig.savefig(str(out), dpi=120)
    plt.close(fig)
    return out


def render_osm_validation(bbox_name: str = config.DEFAULT_BBOX) -> Path:
    """Map of OSM-mapped waterways (blue) vs terrain-derived drainage (red)."""
    import geopandas as gpd

    from . import fetch

    osm = gpd.read_parquet(fetch.osm_waterways_path(bbox_name)).to_crs(4326)
    streams = _streams_for_render(bbox_name)

    m = osm.explore(
        color="#1f78b4", style_kwds={"weight": 2.5}, tiles="CartoDB positron",
        name="OSM waterways (reference)", tooltip=["waterway", "name"],
    )
    streams.explore(
        m=m, color="#e3120b", style_kwds={"weight": 1.0},
        name="terrain-derived drainage", tooltip=False,
    )
    out = config.DATA_DIR / f"validation_osm_{bbox_name}.html"
    m.save(str(out))
    return out


def render_temporal(bbox_name: str = config.DEFAULT_BBOX, top: int = 3000) -> Path:
    """Drainage + encroachers coloured by first-appearance year (uncertain ones greyed)."""
    import geopandas as gpd

    from . import temporal

    enc = gpd.read_parquet(temporal.paths(bbox_name)["parquet"]).to_crs(4326)
    streams = _streams_for_render(bbox_name)

    m = streams.explore(
        color="#1d6fb8", style_kwds={"weight": 1.0},
        tiles="CartoDB positron", name="natural drainage", tooltip=False,
    )
    uncertain = enc[enc["first_year"].isna()].head(top)
    if len(uncertain):
        uncertain.explore(m=m, color="#cccccc", style_kwds={"weight": 0, "fillOpacity": 0.5},
                          name="first year uncertain", tooltip=False)
    known = enc[enc["first_year"].notna()].head(top)
    if len(known):
        known.explore(
            m=m, column="first_year", cmap="viridis", legend=True,
            name="encroachers by first year",
            style_kwds={"weight": 0, "fillOpacity": 0.85},
            tooltip=["rank", "first_year_label", "channel_accum_km2"],
        )
    out = config.DATA_DIR / f"temporal_{bbox_name}.html"
    m.save(str(out))
    return out


def render_combined(bbox_name: str = config.DEFAULT_BBOX, top: int = 3000) -> Path:
    """One HTML map with every layer, toggleable: OSM reference, derived drainage,
    encroachers by score, encroachers by first-appearance year."""
    import folium
    import geopandas as gpd
    from folium.plugins import HeatMap

    from . import fetch, temporal

    streams = _streams_for_render(bbox_name)
    osm = gpd.read_parquet(fetch.osm_waterways_path(bbox_name)).to_crs(4326)
    dated = gpd.read_parquet(temporal.paths(bbox_name)["parquet"]).to_crs(4326)

    b = streams.total_bounds
    m = folium.Map(tiles="CartoDB positron")
    m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])      # show the whole area, not a fixed zoom

    osm.explore(m=m, color="#1f78b4", style_kwds={"weight": 2},
                name="OSM waterways (reference)", tooltip=["waterway", "name"], show=False)
    streams.explore(m=m, color="#1d6fb8", style_kwds={"weight": 1},
                    name="natural drainage", tooltip=False)

    # Encroacher-density heatmap over all encroachers. Individual footprints are sub-pixel at
    # city zoom, so this is the layer that shows where encroachment concentrates metro-wide.
    cen = dated.geometry.centroid
    pts = [[y, x] for x, y in zip(cen.x, cen.y)]
    if len(pts) > 40000:
        pts = pts[::len(pts) // 40000]              # thin for browser performance
    # Default (blue) palette, original radius/blur. A point heatmap is a zoomed-out overview;
    # zoom in to the footprint layer for building-level detail.
    hm = folium.FeatureGroup(name=f"encroacher density ({len(dated)} total)")
    HeatMap(pts, radius=9, blur=12, min_opacity=0.3).add_to(hm)
    hm.add_to(m)

    # Highest-risk encroachers as their real Open Buildings footprints, coloured by score.
    # At full-metro zoom these are sub-pixel (the heatmap is the overview); they render as crisp
    # building shapes once you zoom into a neighbourhood, the same view as a small-bbox map.
    top_enc = dated.head(top)
    top_enc.explore(
        m=m, column="score", cmap="Reds", legend=True,
        name=f"highest-risk encroachers (top {top})",
        style_kwds={"weight": 0, "fillOpacity": 0.85},
        tooltip=["rank", "score", "channel_accum_km2", "first_year_label"],
    )
    known = top_enc[top_enc["first_year"].notna()]
    if len(known):
        known.explore(
            m=m, column="first_year", cmap="viridis", legend=True,
            name="highest-risk by first year", show=False,
            style_kwds={"weight": 0, "fillOpacity": 0.85},
            tooltip=["rank", "first_year_label"],
        )
    folium.LayerControl(collapsed=False).add_to(m)
    out = config.DATA_DIR / f"combined_{bbox_name}.html"
    m.save(str(out))
    return out
