"""Dated encroachment records from Open Buildings Temporal.

For each flagged encroacher, sample the annual ``building_presence`` (2016–2023) at its centroid
and find the first year presence crosses a threshold → a dated, geolocated record of when a
structure appeared on the watercourse. This is the accountability layer: it turns "people build
in waterways" into measured, time-stamped evidence.

Caveats: presence is noisy year-to-year; buildings already present in 2016 are recorded as
"≤2016" (predate the series); the dataset ends ~2023, so very recent construction won't show.

Output: ``dated_encroachers_<bbox>.parquet`` / ``.geojson`` (encroachers + first_year).
"""

from __future__ import annotations

from pathlib import Path

from . import config

PRESENCE_THRESHOLD = 0.5      # building_presence at or above this counts as "built"
SAMPLE_BATCH = 4000           # stay under the getInfo feature cap per call
SAMPLE_WORKERS = 12           # concurrent sampleRegions requests


def paths(bbox_name: str) -> dict[str, Path]:
    d = config.DATA_DIR
    return {
        "parquet": d / f"dated_encroachers_{bbox_name}.parquet",
        "geojson": d / f"dated_encroachers_{bbox_name}.geojson",
    }


def _presence_stack(ee, region):
    """Multi-band image: one ``p<year>`` band of building_presence per available year."""
    col = ee.ImageCollection(config.OPEN_BUILDINGS_TEMPORAL).filterBounds(region)

    def with_year(img):
        y = ee.Date(ee.Number(img.get("imagery_start_time_epoch_s")).multiply(1000)).get("year")
        return img.set("year", y)

    col = col.map(with_year)
    years = sorted(col.aggregate_array("year").distinct().getInfo())
    bands = [col.filter(ee.Filter.eq("year", y)).select("building_presence").mosaic().rename(f"p{y}")
             for y in years]
    return ee.Image.cat(bands), years


def _first_year(presence_by_year: dict[int, float], years: list[int]):
    """First year presence ≥ threshold. Returns (first_year, label)."""
    ordered = [(y, presence_by_year.get(y, 0.0) or 0.0) for y in years]
    if ordered and ordered[0][1] >= PRESENCE_THRESHOLD:
        return years[0], f"≤{years[0]}"          # already there at series start
    for y, p in ordered:
        if p >= PRESENCE_THRESHOLD:
            return y, str(y)
    return None, "uncertain"                       # never crosses threshold


def date_encroachers(bbox_name: str = config.DEFAULT_BBOX, force: bool = False):
    import geopandas as gpd

    from . import fetch, overlay

    p = paths(bbox_name)
    if p["parquet"].exists() and not force:
        return gpd.read_parquet(p["parquet"])

    enc = gpd.read_parquet(overlay.paths(bbox_name)["parquet"])   # projected CRS, ranked
    if not len(enc):
        raise RuntimeError("No encroachers to date. Run `accraflood overlay` first.")

    ee = fetch.init_ee()
    region = fetch.bbox_geometry(config.BBOXES[bbox_name])
    stack, years = _presence_stack(ee, region)

    # Centroids to WGS84 points, sampled in batches (fetched in parallel; at metro scale this
    # is hundreds of thousands of points, so a sequential loop would take many minutes).
    from concurrent.futures import ThreadPoolExecutor

    cen = enc.geometry.centroid.to_crs(4326)
    coords = {int(i): (cen.loc[i].x, cen.loc[i].y) for i in enc.index}
    idx = list(enc.index)
    chunks = [idx[s:s + SAMPLE_BATCH] for s in range(0, len(idx), SAMPLE_BATCH)]

    def _sample(chunk):
        feats = [ee.Feature(ee.Geometry.Point(list(coords[int(i)])), {"idx": int(i)})
                 for i in chunk]
        return stack.sampleRegions(collection=ee.FeatureCollection(feats), scale=4,
                                   geometries=False).getInfo()["features"]

    presence: dict[int, dict[int, float]] = {}
    with ThreadPoolExecutor(max_workers=SAMPLE_WORKERS) as pool:
        for feats in pool.map(_sample, chunks):
            for f in feats:
                pr = f["properties"]
                presence[pr["idx"]] = {y: pr.get(f"p{y}") for y in years}

    first_years, labels = [], []
    for i in enc.index:
        fy, lab = _first_year(presence.get(i, {}), years)
        first_years.append(fy)
        labels.append(lab)
    enc = enc.copy()
    enc["first_year"] = first_years
    enc["first_year_label"] = labels
    enc["appeared_during_series"] = [fy is not None and fy > years[0] for fy in first_years]

    enc.to_parquet(p["parquet"])
    enc.to_crs(4326).to_file(p["geojson"], driver="GeoJSON")
    return enc
