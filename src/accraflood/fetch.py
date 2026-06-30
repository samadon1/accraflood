"""Earth Engine access: authenticate, then clip/export the source layers to ``data/``.

Hydrology runs on a *local* DEM raster, so the DEM is exported to GeoTIFF. Sentinel-1 and the
building footprints are pulled in their own slices. Every fetch caches to ``data/`` so the
downstream steps run independently and offline.
"""

from __future__ import annotations

from pathlib import Path

from . import config


def init_ee():
    """Initialise Earth Engine, prompting for auth on first use.

    Uses the modern project-scoped flow: ``ee.Initialize(project=...)``. The project must be a
    Google Cloud project with the Earth Engine API enabled (set ``EARTHENGINE_PROJECT``).
    """
    import ee

    if not config.EE_PROJECT:
        raise RuntimeError(
            "EARTHENGINE_PROJECT is not set. Run `export EARTHENGINE_PROJECT=<gcp-project-id>` "
            "(a Cloud project with the Earth Engine API enabled), then retry."
        )
    try:
        ee.Initialize(project=config.EE_PROJECT)
    except Exception:
        # Not authenticated yet, so launch the one-time browser flow, then initialise.
        ee.Authenticate()
        ee.Initialize(project=config.EE_PROJECT)
    return ee


def bbox_geometry(bbox: tuple[float, float, float, float]):
    """Return an ``ee.Geometry.Rectangle`` for a (min_lon, min_lat, max_lon, max_lat) box."""
    import ee

    return ee.Geometry.Rectangle(list(bbox))


def dem_path(bbox_name: str) -> Path:
    return config.DATA_DIR / f"dem_{bbox_name}.tif"


def export_dem(bbox_name: str = config.DEFAULT_BBOX, force: bool = False) -> Path:
    """Mosaic Copernicus GLO-30, clip to the named bbox, and export a local GeoTIFF.

    Returns the path to the cached raster (skips the export if it already exists).
    """
    import geemap

    out = dem_path(bbox_name)
    if out.exists() and not force:
        return out

    ee = init_ee()
    bbox = config.BBOXES[bbox_name]
    region = bbox_geometry(bbox)
    if config.USE_FABDEM:
        dem = ee.ImageCollection(config.FABDEM_ASSET).mosaic().rename("DEM").clip(region)
    else:
        dem = ee.ImageCollection(config.DEM_ASSET).select("DEM").mosaic().clip(region)

    geemap.ee_export_image(
        dem,
        filename=str(out),
        scale=config.DEM_SCALE_M,
        region=region,
        crs=config.PROJECTED_CRS,   # metric grid → true 30 m cells for hydrology
        file_per_band=False,
    )
    if not out.exists():
        raise RuntimeError(f"DEM export failed, no file at {out}")
    return out


# --------------------------------------------------------------------------------------
# Open Buildings footprints
# --------------------------------------------------------------------------------------
# Earth Engine getInfo() aborts once a FeatureCollection query passes ~5000 elements. We use
# that abort as the "this tile is too dense, split it" signal, which avoids the very slow
# size() count (size() scales with feature count: ~80s for ~475k features). Tiles are fetched
# in parallel because the metro needs a few hundred of them.
_GETINFO_PROBE = 5001        # ask for one over the cap so a dense tile aborts instead of truncating
_GRID_DEG = 0.01             # top-level tile size in degrees (about 1.1 km; dense tiles subdivide)
_FETCH_WORKERS = 24          # concurrent getInfo requests


def buildings_path(bbox_name: str) -> Path:
    return config.DATA_DIR / f"buildings_{bbox_name}.parquet"


def _is_overflow(err: Exception) -> bool:
    msg = str(err).lower()
    return "accumulating over" in msg or "5000 element" in msg


def _fetch_tile(fc, ee, x0, y0, x1, y1, depth: int = 0, max_depth: int = 5) -> list[dict]:
    """Fetch all features in a tile, splitting into quadrants if it overflows the getInfo cap."""
    sub = fc.filterBounds(ee.Geometry.Rectangle([x0, y0, x1, y1]))
    try:
        return sub.limit(_GETINFO_PROBE).getInfo()["features"]   # complete if <= 5000
    except Exception as err:
        if not _is_overflow(err) or depth >= max_depth:
            # not a density overflow (re-raise), or too deep to split further (take first 5000)
            if not _is_overflow(err):
                raise
            return sub.limit(5000).getInfo()["features"]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        out: list[dict] = []
        for qx0, qx1 in ((x0, mx), (mx, x1)):
            for qy0, qy1 in ((y0, my), (my, y1)):
                out += _fetch_tile(fc, ee, qx0, qy0, qx1, qy1, depth + 1, max_depth)
        return out


def _collect_features(fc, ee, bbox, grid_deg: float = _GRID_DEG,
                      workers: int = _FETCH_WORKERS) -> list[dict]:
    """Tile the bbox into a grid and fetch every tile in parallel (dense tiles self-subdivide)."""
    from concurrent.futures import ThreadPoolExecutor

    min_lon, min_lat, max_lon, max_lat = bbox
    tiles, x = [], min_lon
    while x < max_lon:
        y = min_lat
        while y < max_lat:
            tiles.append((x, y, min(x + grid_deg, max_lon), min(y + grid_deg, max_lat)))
            y += grid_deg
        x += grid_deg

    feats: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for chunk in pool.map(lambda t: _fetch_tile(fc, ee, *t), tiles):
            feats += chunk
    return feats


def fetch_buildings(bbox_name: str = config.DEFAULT_BBOX, force: bool = False) -> Path:
    """Cache Open Buildings polygons (confidence-filtered) for a bbox as GeoParquet."""
    import geopandas as gpd
    from shapely.geometry import shape

    out = buildings_path(bbox_name)
    if out.exists() and not force:
        return out

    ee = init_ee()
    region = bbox_geometry(config.BBOXES[bbox_name])
    fc = (ee.FeatureCollection(config.OPEN_BUILDINGS)
          .filterBounds(region)
          .filter(ee.Filter.gte("confidence", config.BUILDING_CONFIDENCE_MIN)))

    feats = _collect_features(fc, ee, config.BBOXES[bbox_name])
    rows = {}
    for f in feats:
        pid = f["properties"].get("full_plus_code") or f["id"]   # dedup key across tile seams
        if pid in rows:
            continue
        rows[pid] = {
            "plus_code": pid,
            "confidence": f["properties"].get("confidence"),
            "area_m2": f["properties"].get("area_in_meters"),
            "geometry": shape(f["geometry"]),
        }
    gdf = gpd.GeoDataFrame(list(rows.values()), geometry="geometry", crs="EPSG:4326")
    gdf = gdf.to_crs(config.PROJECTED_CRS)   # metric, to match the drainage layer
    gdf.to_parquet(out)
    return out


# --------------------------------------------------------------------------------------
# Sentinel-1 SAR (for validation)
# --------------------------------------------------------------------------------------
def s1_path(bbox_name: str, tag: str) -> Path:
    return config.DATA_DIR / f"s1_{tag}_{bbox_name}.tif"


def export_s1(bbox_name: str, window: tuple[str, str], tag: str, force: bool = False) -> Path:
    """Export a median VV composite (dB, 10 m, projected) for a date window.

    A median over the window suppresses speckle and transient noise; one orbit direction
    keeps imaging geometry consistent between baseline and event.
    """
    import geemap

    out = s1_path(bbox_name, tag)
    if out.exists() and not force:
        return out

    ee = init_ee()
    region = bbox_geometry(config.BBOXES[bbox_name])
    col = (ee.ImageCollection(config.S1_ASSET)
           .filterBounds(region)
           .filterDate(window[0], window[1])
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.eq("orbitProperties_pass", config.S1_ORBIT_PASS))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .select("VV"))
    if col.size().getInfo() == 0:
        raise RuntimeError(f"No Sentinel-1 {config.S1_ORBIT_PASS} VV scenes in {window} over "
                           f"'{bbox_name}'. (The June-29 flood's post-event scene may not be "
                           f"ingested yet; see config.EVENT_WINDOW.)")
    vv = col.median().clip(region)
    geemap.ee_export_image(vv, filename=str(out), scale=10, region=region,
                           crs=config.PROJECTED_CRS, file_per_band=False)
    if not out.exists():
        raise RuntimeError(f"S1 export failed, no file at {out}")
    return out


# --------------------------------------------------------------------------------------
# OpenStreetMap waterways (independent reference for validation)
# --------------------------------------------------------------------------------------
_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


def osm_waterways_path(bbox_name: str) -> Path:
    return config.DATA_DIR / f"osm_waterways_{bbox_name}.parquet"


def fetch_osm_waterways(bbox_name: str = config.DEFAULT_BBOX, force: bool = False) -> Path:
    """Download OSM ``waterway=*`` lines for a bbox as GeoParquet (projected CRS)."""
    import geopandas as gpd
    import requests
    from shapely.geometry import LineString

    out = osm_waterways_path(bbox_name)
    if out.exists() and not force:
        return out

    min_lon, min_lat, max_lon, max_lat = config.BBOXES[bbox_name]
    query = (f"[out:json][timeout:90];"
             f'(way["waterway"]({min_lat},{min_lon},{max_lat},{max_lon}););'
             f"out tags geom;")
    headers = {"User-Agent": "accraflood/0.1 (research; flood drainage mapping)"}

    elements = None
    for url in _OVERPASS_ENDPOINTS:
        try:
            r = requests.post(url, data={"data": query}, headers=headers, timeout=120)
            if r.status_code == 200:
                elements = r.json()["elements"]
                break
        except Exception:
            continue
    if elements is None:
        raise RuntimeError("Overpass request failed on all endpoints.")

    rows = []
    for w in elements:
        geom = w.get("geometry")
        if not geom or len(geom) < 2:
            continue
        rows.append({
            "osm_id": w["id"],
            "waterway": w.get("tags", {}).get("waterway"),
            "name": w.get("tags", {}).get("name"),
            "geometry": LineString([(p["lon"], p["lat"]) for p in geom]),
        })
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    gdf = gdf.to_crs(config.PROJECTED_CRS)   # metric, to match the derived streams
    gdf.to_parquet(out)
    return out
