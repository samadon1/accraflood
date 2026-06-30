"""Cross-check derived high-risk zones against Accra's documented flood hotspots (validation).

There is no free observed flood-extent map for Accra's urban (pluvial) flooding, because SAR
cannot see standing water in the dense built-up core (we and UNOSAT both hit this). So the
achievable ground-truth-style check is: do our model's high-risk zones coincide with the
neighbourhoods documented (news, NADMO, academic literature) as Accra's recurrent flood hotspots?

We compare the density of flagged encroachers near each hotspot to the density at random points.
The lift shows whether the model is selecting real flood-prone places, not lighting up everywhere.

Hotspot coordinates are approximate neighbourhood centroids (indicative, not surveyed).
"""

from __future__ import annotations

import json
from pathlib import Path

from . import config

# Approximate centroids (lat, lon) of recurrently-flooded Accra locations.
ACCRA_FLOOD_HOTSPOTS: dict[str, tuple[float, float]] = {
    "Kwame Nkrumah Circle": (5.5709, -0.2074),
    "Alajo": (5.5847, -0.2186),
    "Kaneshie": (5.5639, -0.2330),
    "Avenor": (5.5786, -0.2220),
    "Adabraka": (5.5610, -0.2140),
    "Agbogbloshie / Korle Lagoon": (5.5470, -0.2240),
    "Caprice (Odaw)": (5.5970, -0.2150),
    "Achimota": (5.6170, -0.2230),
    "Tesano": (5.6010, -0.2280),
    "Dzorwulu": (5.6060, -0.1960),
    "Weija": (5.5660, -0.3430),
    "Mallam": (5.5670, -0.2950),
    "Dansoman": (5.5350, -0.2650),
    "Spintex": (5.6280, -0.1280),
}

HOTSPOT_RADIUS_M = 600          # a hotspot is "hit" if flagged encroachers fall within this
N_RANDOM = 500                  # random control points for the chance baseline


def metrics_path(bbox_name: str) -> Path:
    return config.DATA_DIR / f"hotspots_{bbox_name}.json"


def _hits(points_xy, enc, radius):
    """For each (x, y), count flagged encroachers within radius; return list of (n, max_km2)."""
    from shapely.geometry import Point

    enc_sindex = enc.sindex
    out = []
    for x, y in points_xy:
        # bbox prefilter via spatial index, then precise distance
        cand = list(enc_sindex.intersection((x - radius, y - radius, x + radius, y + radius)))
        n, mx = 0, 0.0
        if cand:
            sub = enc.iloc[cand]
            d = sub.geometry.distance(Point(x, y))
            near = sub[d <= radius]
            n = len(near)
            if n:
                mx = float(near["channel_accum_km2"].max())
        out.append((n, mx))
    return out


def check_hotspots(bbox_name: str = config.DEFAULT_BBOX, radius_m: float = HOTSPOT_RADIUS_M) -> dict:
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import Point

    from . import overlay

    enc = gpd.read_parquet(overlay.paths(bbox_name)["parquet"])      # projected, has channel_accum_km2
    crs = enc.crs

    # Which hotspots fall inside this bbox? (others can't be assessed here.)
    min_lon, min_lat, max_lon, max_lat = config.BBOXES[bbox_name]
    in_bbox = {name: (lat, lon) for name, (lat, lon) in ACCRA_FLOOD_HOTSPOTS.items()
               if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat}

    hs = gpd.GeoSeries([Point(lon, lat) for lat, lon in in_bbox.values()],
                       crs="EPSG:4326").to_crs(crs)
    hs_xy = [(p.x, p.y) for p in hs]
    hs_hits = _hits(hs_xy, enc, radius_m)

    hotspots = []
    hit_count = 0
    for (name, (lat, lon)), (n, mx) in zip(in_bbox.items(), hs_hits):
        hit = n > 0
        hit_count += hit
        hotspots.append({"name": name, "lat": lat, "lon": lon,
                         "flagged_within_radius": n, "max_channel_km2": round(mx, 1), "hit": hit})

    # Random control points within the data extent give the chance baseline.
    # (A binary hit-rate is uninformative here: flagged buildings are so dense that almost any
    #  point "hits". Density and channel size are the signals that actually discriminate.)
    b = enc.total_bounds
    rng = np.random.default_rng(42)
    rx = rng.uniform(b[0], b[2], N_RANDOM)
    ry = rng.uniform(b[1], b[3], N_RANDOM)
    rand_hits = _hits(list(zip(rx, ry)), enc, radius_m)

    def _means(hits):
        counts = [n for n, _ in hits]
        kms = [mx for _, mx in hits]
        return (float(np.mean(counts)) if counts else 0.0,
                float(np.mean(kms)) if kms else 0.0)

    hs_count, hs_km = _means(hs_hits)
    rand_count, rand_km = _means(rand_hits)

    metrics = {
        "bbox": bbox_name,
        "radius_m": radius_m,
        "hotspots_in_bbox": len(in_bbox),
        "hotspots_hit": hit_count,
        # density: avg flagged encroachers within radius
        "hotspot_mean_flagged": round(hs_count, 1),
        "random_mean_flagged": round(rand_count, 1),
        "density_lift": round(hs_count / rand_count, 2) if rand_count else None,
        # severity: avg largest channel (drainage km²) a location sits on
        "hotspot_mean_channel_km2": round(hs_km, 2),
        "random_mean_channel_km2": round(rand_km, 2),
        "channel_lift": round(hs_km / rand_km, 2) if rand_km else None,
        "detail": sorted(hotspots, key=lambda h: -h["max_channel_km2"]),
    }
    metrics_path(bbox_name).write_text(json.dumps(metrics, indent=2))
    return metrics
