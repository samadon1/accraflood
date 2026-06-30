"""accraflood command-line interface.

    accraflood auth                       Check Earth Engine auth with a tiny round-trip
    accraflood fetch    --bbox-name odaw  Cache the DEM and Open Buildings for a bbox
    accraflood drainage                   Derive and render the natural drainage network
    accraflood overlay                    Flag and rank buildings encroaching on the drainage
    accraflood validate                   Validate derived streams against OSM (or SAR)
    accraflood temporal                   Date each encroacher's first appearance
    accraflood hotspots                   Cross-check vs documented Accra flood hotspots
    accraflood run      --bbox-name odaw  Run the whole pipeline into one combined map
"""

from __future__ import annotations

import typer
from rich import print as rprint

from . import config

app = typer.Typer(add_completion=False,
                  help="Terrain-derived drainage and encroachment detection for Accra.")

BBoxOpt = typer.Option(config.DEFAULT_BBOX, "--bbox-name", help=f"One of: {', '.join(config.BBOXES)}")


@app.command()
def auth():
    """Check that Earth Engine is authenticated by running a tiny compute round-trip."""
    from . import fetch

    if not config.EE_PROJECT:
        rprint("[red]EARTHENGINE_PROJECT not set.[/] "
               "Run [bold]export EARTHENGINE_PROJECT=<gcp-project-id>[/] and retry.")
        raise typer.Exit(1)

    rprint(f"Initialising Earth Engine (project [bold]{config.EE_PROJECT}[/])...")
    ee = fetch.init_ee()
    region = fetch.bbox_geometry(config.BBOXES[config.DEFAULT_BBOX])
    dem = ee.ImageCollection(config.DEM_ASSET).select("DEM").mosaic()
    mean = dem.reduceRegion(ee.Reducer.mean(), region, scale=config.DEM_SCALE_M).get("DEM")
    rprint(f"[green]OK[/]. Earth Engine round-trip works. Mean elevation over "
           f"'{config.DEFAULT_BBOX}': [bold]{ee.Number(mean).getInfo():.1f} m[/]")


@app.command()
def fetch(bbox_name: str = BBoxOpt, force: bool = typer.Option(False, help="Re-fetch even if cached")):
    """Cache the source layers (DEM and Open Buildings) for a bbox."""
    from . import fetch as _fetch

    rprint(f"Exporting DEM for '[bold]{bbox_name}[/]' into data/ ...")
    dem = _fetch.export_dem(bbox_name, force=force)
    rprint(f"[green]OK[/]. DEM cached at [bold]{dem}[/]")

    rprint(f"Fetching Open Buildings for '[bold]{bbox_name}[/]' (adaptive tiling)...")
    bld = _fetch.fetch_buildings(bbox_name, force=force)
    import geopandas as gpd
    n = len(gpd.read_parquet(bld))
    rprint(f"[green]OK[/]. {n} buildings cached at [bold]{bld}[/]")


@app.command()
def drainage(bbox_name: str = BBoxOpt,
             force: bool = typer.Option(False, help="Recompute even if cached")):
    """Derive and render the natural drainage network."""
    from . import hydrology, render

    rprint(f"Deriving drainage for '[bold]{bbox_name}[/]' "
           "(fill, flow direction, accumulation, streams)...")
    gdf = hydrology.derive_drainage(bbox_name, force=force)
    rprint(f"[green]OK[/]. {len(gdf)} stream segments extracted.")
    out = render.render_drainage(bbox_name)
    rprint(f"[green]OK[/]. Drainage map: [bold]{out}[/]")


@app.command()
def overlay(bbox_name: str = BBoxOpt,
            force: bool = typer.Option(False, help="Recompute even if cached")):
    """Flag and rank buildings encroaching on the drainage."""
    from . import overlay as _overlay
    from . import render

    rprint(f"Flagging encroachers for '[bold]{bbox_name}[/]' "
           "(buildings intersecting channel buffers)...")
    flagged = _overlay.flag_encroachers(bbox_name, force=force)
    rprint(f"[green]OK[/]. {len(flagged)} buildings flagged on the drainage.")
    if len(flagged):
        worst = flagged.iloc[0]
        rprint(f"  Worst: rank 1 sits on a channel draining "
               f"[bold]{worst['channel_accum_km2']:.1f} km2[/] (score {worst['score']:.1f})")
    out = render.render_overlay(bbox_name)
    rprint(f"[green]OK[/]. Overlay map: [bold]{out}[/]")


@app.command()
def validate(bbox_name: str = BBoxOpt,
             method: str = typer.Option("osm", help="osm (independent drainage map) or sar"),
             event_start: str = typer.Option(None, help="SAR: override event window start (YYYY-MM-DD)"),
             event_end: str = typer.Option(None, help="SAR: override event window end (YYYY-MM-DD)"),
             force: bool = typer.Option(False, help="Re-fetch and recompute")):
    """Validate the pipeline.

    osm  Compare terrain-derived streams to OSM-mapped waterways. This is the primary check
         and works today.
    sar  Compare predicted ponding to Sentinel-1 water. Needs a post-event scene (see
         config.EVENT_WINDOW); pass --event-start / --event-end once the July pass ingests.
    """
    from . import render
    from . import validate as _validate

    if method == "osm":
        rprint(f"Validating '[bold]{bbox_name}[/]' against OSM-mapped waterways...")
        m = _validate.validate_against_osm(bbox_name, force=force)
        pr = m["primary"]
        rprint(f"  {m['osm_ways']} OSM waterways ({m['osm_total_km']} km) vs "
               f"{m['derived_total_km']} km derived")
        rprint(f"  At {m['primary_buffer_m']} m tolerance: recall [bold]{pr['recall']}[/], "
               f"precision [bold]{pr['precision']}[/]")
        lift = pr["lift"]
        colour = "green" if lift >= 2 else "yellow" if lift >= 1.2 else "red"
        rprint(f"  [bold]Lift over chance: [{colour}]{lift}x[/][/]. Derived streams hug real "
               f"channels {lift}x more than random.")
        if m["major_channel_recall"] is not None:
            rprint(f"  Major-channel (Odaw/river/canal) recall: "
                   f"[bold]{m['major_channel_recall']}[/]")
        out = render.render_osm_validation(bbox_name)
        rprint(f"[green]OK[/]. Validation map: [bold]{out}[/]")
        return

    rprint(f"Validating '[bold]{bbox_name}[/]' against Sentinel-1 (SAR water vs predicted)...")
    event_window = (event_start, event_end) if event_start and event_end else None
    m = _validate.validate_against_sar(bbox_name, force=force, event_window=event_window)
    rprint(f"  Baseline {m['baseline_window']} vs event {m['event_window']}")
    rprint(f"  Observed wet cells: [bold]{m['observed_wet_cells']}[/], "
           f"predicted ponding cells: [bold]{m['predicted_pond_cells']}[/]")
    rprint(f"  IoU [bold]{m['iou']}[/], detection [bold]{m['detection_rate']}[/], "
           f"precision [bold]{m['precision']}[/]")
    lift = m["lift_over_chance"]
    colour = "green" if lift >= 1.5 else "yellow" if lift >= 1.0 else "red"
    rprint(f"  [bold]Lift over chance: [{colour}]{lift}x[/][/] "
           f"(SAR water is {lift}x more likely to fall in predicted zones than at random)")
    out = render.render_validation(bbox_name)
    rprint(f"[green]OK[/]. Validation map: [bold]{out}[/]")


@app.command()
def temporal(bbox_name: str = BBoxOpt,
             force: bool = typer.Option(False, help="Recompute even if cached")):
    """Produce dated encroachment records from the temporal building footprints."""
    from . import render
    from . import temporal as _temporal

    rprint(f"Dating encroachers for '[bold]{bbox_name}[/]' via Open Buildings Temporal...")
    enc = _temporal.date_encroachers(bbox_name, force=force)
    appeared = int(enc["appeared_during_series"].sum())
    rprint(f"[green]OK[/]. Dated {len(enc)} encroachers; "
           f"[bold]{appeared}[/] appeared on the watercourse during 2016-2023.")
    counts = enc["first_year_label"].value_counts()
    for lab in sorted(counts.index, key=lambda s: (s == "uncertain", s)):
        rprint(f"   {lab}: {counts[lab]}")
    out = render.render_temporal(bbox_name)
    rprint(f"[green]OK[/]. Temporal map: [bold]{out}[/]")


@app.command()
def hotspots(bbox_name: str = BBoxOpt,
             radius: float = typer.Option(600.0, help="Match radius (m) around each hotspot")):
    """Cross-check high-risk zones against Accra's documented flood hotspots."""
    from . import hotspots as _hotspots

    rprint(f"Cross-checking '[bold]{bbox_name}[/]' against documented Accra flood hotspots...")
    m = _hotspots.check_hotspots(bbox_name, radius_m=radius)
    rprint(f"  {m['hotspots_in_bbox']} documented hotspots in bbox; "
           f"{m['hotspots_hit']} with flagged encroachers within {int(radius)} m")
    dl = m["density_lift"]
    colour = "green" if dl and dl >= 1.3 else "yellow" if dl and dl >= 1.0 else "red"
    rprint(f"  [bold]Density lift [{colour}]{dl}x[/][/]. Hotspots have {dl}x more flagged "
           f"encroachers nearby than random locations.")
    rprint(f"  Channel-size lift {m['channel_lift']}x. Hotspots sit on larger-drainage channels.")
    for h in m["detail"][:8]:
        rprint(f"   {h['name']:30s} flagged within {int(radius)}m: {h['flagged_within_radius']:4d}, "
               f"max channel {h['max_channel_km2']} km2")


@app.command()
def run(bbox_name: str = BBoxOpt,
        force: bool = typer.Option(False, help="Recompute every stage")):
    """Run the full pipeline end-to-end and emit one combined HTML map."""
    from . import fetch as _fetch
    from . import hydrology, render
    from . import overlay as _overlay
    from . import temporal as _temporal
    from . import validate as _validate

    rprint(f"[bold]accraflood run[/]: '{bbox_name}'")
    rprint("1/6 fetch DEM and buildings...")
    _fetch.export_dem(bbox_name, force=force)
    _fetch.fetch_buildings(bbox_name, force=force)
    rprint("2/6 derive drainage...")
    streams = hydrology.derive_drainage(bbox_name, force=force)
    rprint("3/6 flag encroachers...")
    flagged = _overlay.flag_encroachers(bbox_name, force=force)
    rprint("4/6 validate against OSM waterways...")
    v = _validate.validate_against_osm(bbox_name, force=force)
    rprint("5/6 date encroachers...")
    dated = _temporal.date_encroachers(bbox_name, force=force)
    rprint("6/6 render combined map...")
    out = render.render_combined(bbox_name)

    appeared = int(dated["appeared_during_series"].sum())
    rprint("\n[bold green]Done.[/]")
    rprint(f"  drainage: {len(streams)} stream segments")
    rprint(f"  encroachers: {len(flagged)} buildings on the drainage "
           f"(worst on a {flagged.iloc[0]['channel_accum_km2']:.1f} km2 channel)")
    rprint(f"  validation: OSM lift {v['primary']['lift']}x, major-channel recall "
           f"{v['major_channel_recall']}")
    rprint(f"  temporal: {appeared} appeared on the watercourse 2016-2023")
    rprint(f"  [bold]combined map:[/] {out}")


if __name__ == "__main__":
    app()
