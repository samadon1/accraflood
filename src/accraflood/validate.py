"""Validate the pipeline. Two methods.

validate_against_osm (primary, works today): compare the terrain-derived streams to the
waterways mapped in OpenStreetMap. This directly tests the core claim that terrain recovers
Accra's real watercourses.

validate_against_sar (deferred): compare predicted ponding to Sentinel-1 open water. The SAR
pipeline is: median VV composites (dB) for a dry baseline and a wet event window, speckle
filter, Otsu threshold to open-water masks, take "new water" (wet now, dry before), aggregate
the 10 m mask onto the 30 m DEM grid, and score the overlap with predicted ponding (filled
depressions plus channel buffers) using IoU and lift over chance.

Note: the June 29 2026 flash flood has no post-event Sentinel-1 scene yet, so EVENT_WINDOW
defaults to the peak-wet pre-flood period (observed seasonal inundation). Repoint EVENT_WINDOW
to the post-flood scene and re-run for the true-event check once it ingests.
"""

from __future__ import annotations

from pathlib import Path

from . import config


def paths(bbox_name: str) -> dict[str, Path]:
    d = config.DATA_DIR
    return {
        "sinks": d / f"sinks_{bbox_name}.tif",
        "buffers": d / f"channel_buffers_{bbox_name}.parquet",
        "obs_water": d / f"sar_newwater_{bbox_name}.tif",
        "pred_pond": d / f"pred_ponding_{bbox_name}.tif",
        "metrics": d / f"validation_{bbox_name}.json",
        "osm_metrics": d / f"validation_osm_{bbox_name}.json",
    }


def validate_against_osm(bbox_name: str = config.DEFAULT_BBOX, force: bool = False) -> dict:
    """Validate terrain-derived streams against OSM-mapped waterways (the primary check).

    Reports, at several match tolerances:
      recall:    share of mapped-channel length that has a derived stream nearby
      precision: share of derived-stream length that is near a mapped channel
      lift:      recall divided by chance, where chance is the fraction of area within the
                 tolerance of a derived stream. This is how much more the derived network
                 hugs real channels than a random network would.
    Plus the recall for major channels alone (river, canal, stream: the Odaw and trunks).
    """
    import json

    import geopandas as gpd
    import numpy as np
    import rasterio
    from shapely.geometry import Point

    from . import fetch, hydrology

    streams = hydrology.load_streams(bbox_name)
    osm = gpd.read_parquet(fetch.fetch_osm_waterways(bbox_name, force=force))

    with rasterio.open(hydrology.paths(bbox_name)["dem"]) as s:
        b = s.bounds

    osm_union = osm.geometry.union_all()
    osm_len = float(osm.geometry.length.sum())
    derived_len = float(streams.geometry.length.sum())

    # Spatial index over the (possibly hundreds of thousands of) derived segments. We never
    # buffer or intersect the whole network at once; we prefilter to the few segments near OSM.
    sidx = streams.sindex
    geoms = streams.geometry.values
    major = osm[osm["waterway"].isin(["river", "canal", "stream"])]

    # Monte-Carlo chance baseline: fraction of the bbox within tolerance of a derived stream
    # (what recall would be if the mapped channels were placed at random).
    rng = np.random.default_rng(0)
    k = 4000
    rand_pts = [Point(x, y) for x, y in zip(rng.uniform(b.left, b.right, k),
                                            rng.uniform(b.bottom, b.top, k))]

    def _chance(n):
        hits = 0
        for p in rand_pts:
            cand = sidx.query(p.buffer(n))
            if len(cand) and min(geoms[i].distance(p) for i in cand) <= n:
                hits += 1
        return hits / k

    by_buffer = {}
    cover_primary = None
    primary = config.OSM_PRIMARY_BUFFER_M
    for n in config.OSM_MATCH_BUFFERS_M:
        osm_buf = osm_union.buffer(n).simplify(2)
        cand = streams.iloc[list(sidx.query(osm_buf, predicate="intersects"))]   # streams near OSM
        precision = (float(cand.geometry.intersection(osm_buf).length.sum()) / derived_len
                     if len(cand) else 0.0)
        cover = cand.geometry.buffer(n).union_all() if len(cand) else None        # derived coverage
        recall = (float(osm.geometry.intersection(cover).length.sum()) / osm_len
                  if cover is not None else 0.0)
        chance = _chance(n)
        by_buffer[n] = {
            "recall": round(recall, 3),
            "precision": round(precision, 3),
            "chance": round(chance, 3),
            "lift": round(recall / chance, 2) if chance else 0.0,
        }
        if n == primary:
            cover_primary = cover

    major_recall = None
    if len(major) and cover_primary is not None:
        major_recall = round(
            float(major.geometry.intersection(cover_primary).length.sum())
            / float(major.geometry.length.sum()), 3
        )

    metrics = {
        "bbox": bbox_name,
        "osm_ways": int(len(osm)),
        "osm_total_km": round(osm_len / 1000, 2),
        "derived_total_km": round(derived_len / 1000, 2),
        "primary_buffer_m": primary,
        "primary": by_buffer[primary],
        "major_channel_recall": major_recall,
        "by_buffer": by_buffer,
    }
    paths(bbox_name)["osm_metrics"].write_text(json.dumps(metrics, indent=2))
    return metrics


def _otsu(values) -> float:
    """Otsu threshold on a 1-D array of finite dB values (water is the low-value mode)."""
    import numpy as np

    hist, edges = np.histogram(values, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    w = hist.cumsum().astype("float64")
    wb = w / w[-1]
    mu = (hist * centers).cumsum() / np.maximum(w, 1)
    mu_t = mu[-1]
    sigma_b = (mu_t * wb - mu) ** 2 / np.maximum(wb * (1 - wb), 1e-12)
    return float(centers[np.nanargmax(sigma_b)])


def _read(path: Path):
    import rasterio

    with rasterio.open(path) as s:
        return s.read(1).astype("float64"), s.profile


def _speckle(arr):
    from scipy.ndimage import median_filter

    return median_filter(arr, size=3)


def _reproject_to(src_arr, src_profile, ref_profile, resampling):
    """Resample src raster onto the reference grid; returns the aligned array."""
    import numpy as np
    import rasterio
    from rasterio.warp import reproject

    dst = np.zeros((ref_profile["height"], ref_profile["width"]), dtype="float64")
    reproject(
        source=src_arr, destination=dst,
        src_transform=src_profile["transform"], src_crs=src_profile["crs"],
        dst_transform=ref_profile["transform"], dst_crs=ref_profile["crs"],
        resampling=resampling,
    )
    return dst


def validate_against_sar(bbox_name: str = config.DEFAULT_BBOX, force: bool = False,
                         baseline_window=None, event_window=None) -> dict:
    import json

    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio.features import rasterize
    from rasterio.warp import Resampling

    from . import fetch

    baseline_window = tuple(baseline_window) if baseline_window else config.BASELINE_WINDOW
    event_window = tuple(event_window) if event_window else config.EVENT_WINDOW

    p = paths(bbox_name)
    if not p["sinks"].exists():
        raise FileNotFoundError(
            f"Run `accraflood drainage --bbox-name {bbox_name}` first (need sinks/streams)."
        )

    # 1–2. SAR composites → water masks.
    base_tif = fetch.export_s1(bbox_name, baseline_window, "baseline", force=force)
    evt_tif = fetch.export_s1(bbox_name, event_window, "event", force=force)
    base, base_prof = _read(base_tif)
    evt, evt_prof = _read(evt_tif)
    base, evt = _speckle(base), _speckle(evt)

    finite = np.isfinite(evt) & (evt != evt_prof.get("nodata"))
    thr = _otsu(evt[finite])
    water_evt = (evt < thr) & finite
    water_base = (base < thr) & np.isfinite(base)
    new_water = (water_evt & ~water_base).astype("float64")   # newly inundated since dry season

    with rasterio.open(evt_tif) as s:
        evt_profile = s.profile

    # 3. Aggregate 10 m new-water onto the 30 m DEM grid (mean → fraction wet → threshold).
    _, ref_profile = _read(p["sinks"])
    obs_frac = _reproject_to(new_water, evt_profile, ref_profile, Resampling.average)
    obs_mask = obs_frac >= config.SAR_WET_FRACTION

    # 4. Predicted ponding (30 m): filled-depression cells ∪ channel buffers.
    sinks, _ = _read(p["sinks"])
    pond_sink = sinks > config.SINK_PONDING_M
    buffers = gpd.read_parquet(p["buffers"])
    ref_transform = ref_profile["transform"]
    shape = (ref_profile["height"], ref_profile["width"])
    buf_mask = rasterize(
        ((geom, 1) for geom in buffers.geometry),
        out_shape=shape, transform=ref_transform, fill=0, dtype="uint8",
    ).astype(bool)
    pred_mask = pond_sink | buf_mask

    # 5. Metrics.
    inter = int((pred_mask & obs_mask).sum())
    union = int((pred_mask | obs_mask).sum())
    n = int(pred_mask.size)
    pred_n, obs_n = int(pred_mask.sum()), int(obs_mask.sum())
    iou = inter / union if union else 0.0
    detection = inter / obs_n if obs_n else 0.0          # share of observed water we predicted
    precision = inter / pred_n if pred_n else 0.0        # share of predicted that was wet
    chance = pred_n / n if n else 0.0                    # overlap expected if water were random
    lift = detection / chance if chance else 0.0

    metrics = {
        "bbox": bbox_name,
        "baseline_window": list(baseline_window),
        "event_window": list(event_window),
        "otsu_threshold_db": round(thr, 2),
        "observed_wet_cells": obs_n,
        "predicted_pond_cells": pred_n,
        "intersection_cells": inter,
        "iou": round(iou, 3),
        "detection_rate": round(detection, 3),
        "precision": round(precision, 3),
        "chance_overlap": round(chance, 3),
        "lift_over_chance": round(lift, 2),
    }

    # Persist masks + metrics for rendering / inspection.
    out_prof = ref_profile.copy()
    out_prof.update(dtype="uint8", count=1, nodata=0, compress="deflate")
    with rasterio.open(p["obs_water"], "w", **out_prof) as d:
        d.write(obs_mask.astype("uint8"), 1)
    with rasterio.open(p["pred_pond"], "w", **out_prof) as d:
        d.write(pred_mask.astype("uint8"), 1)
    p["metrics"].write_text(json.dumps(metrics, indent=2))
    return metrics
