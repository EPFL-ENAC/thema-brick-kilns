# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pystac-client", "planetary-computer", "odc-stac", "xarray", "dask",
#   "numpy", "scipy", "rasterio", "geopandas", "scikit-learn",
# ]
# ///
"""Step 1: screen Anand district for clay-excavation depressions using open data.

Data comes from Microsoft Planetary Computer: Sentinel-1 RTC, Sentinel-2 L2A,
ESA WorldCover and JRC Global Surface Water. No account is needed.
Python port of step1_screening.js.

    uv run step1_screening.py                                   # whole district (approx. bbox)
    uv run step1_screening.py --aoi anand.gpkg                  # clip to a boundary file
    uv run step1_screening.py --bbox 73.0 22.44 73.06 22.5 --first-year 2023 --last-year 2024

Outputs go to --out: layers.tif (one named band per layer), hotspots.gpkg, kilns.gpkg.
Work is done in tiles cached in --out/tiles/, so rerunning the same command resumes an interrupted run.
Delete that folder after changing the area, resolution, years or any S1/S2 threshold.
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.stac import configure_rio, load
from rasterio import features
from scipy import ndimage
from shapely.geometry import box, shape

# Calibration knobs. Tune them on a few known dug and undug fields near Adas.
# The S1 dB thresholds are about 1 dB above the Earth Engine script's, because PC's RTC product is
# gamma0 (terrain-flattened), which reads brighter than GEE's sigma0 at these incidence angles.
CFG = dict(
    water_db=-15,        # S1 VV below this means open water
    rice_flood_db=-21,   # S1 VH minimum during transplant flooding (Jun 15 to Aug 31)
    rice_rise_db=5,      # VH rise from flooding to canopy peak (Sep to Oct)
    bare_ndvi=0.2,       # rabi bare soil: NDVI below this and BSI > 0
    neighbour_m=500,     # half-width of the "neighbours" window for relative anomalies
    water_anom_min=0.2,  # post-monsoon water frequency above the neighbour mean
    water_dry_min=0.3,   # dry-season (Feb to May) water frequency, for deep pits
    bare_anom_min=0.3,   # rabi bare frequency above the neighbour mean
    kiln_ndvi=0.15, kiln_red_blue=1.5, kiln_vv_db=-7,
    kiln_min_ha=0.2,
    kiln_buffer_m=3000,  # "near a kiln"
    min_score=2,         # flags needed (out of 4) to call a pixel a hotspot
    min_hotspot_ha=1,
    river_min_ha=50,     # connected water bodies larger than this are rivers/reservoirs, not dug fields
    river_buffer_m=100,  # also drop their banks and sandbars
)
CRS = "EPSG:32643"  # UTM 43N
ANAND_BBOX = (72.45, 22.05, 73.25, 22.80)  # approx. district extent (lon/lat)
CATALOG = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")


TILE = 512  # px per tile side. Bounds memory to a few GB per tile; the whole district at once needs >100 GB.
S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "SCL"]
S2_KW = dict(resampling={"SCL": "nearest", "*": "average"})


def search(collection, bbox, dates, query=None):
    items = CATALOG.search(collections=[collection], bbox=bbox, datetime=dates, query=query).item_collection()
    print(f"{collection} {dates}: {len(items)} items")
    assert len(items), f"no {collection} items for {dates}"
    return items


def odc_load(items, bands, gbox, **kw):
    return load(items, bands=bands, geobox=gbox, chunks={"x": TILE, "y": TILE},
                groupby="solar_day", patch_url=pc.sign, fail_on_error=False, **kw)


def tile_stats(tb, items, years, kiln_year):
    """Per-pixel time-series statistics for one tile (GeoBox `tb`), as numpy arrays."""
    s1 = odc_load(items["s1"], ["vv", "vh"], tb, resampling="average")
    # ponytail: 3x3 boxcar speckle filter in linear power; use a refined Lee filter if field edges look ragged
    s1 = s1.where(s1 > 0).rolling(x=3, y=3, center=True, min_periods=1).mean()
    vv, vh = 10 * np.log10(s1.vv), 10 * np.log10(s1.vh)

    def water_freq(months):
        v = vv.sel(time=vv.time.dt.month.isin(months))
        return (v < CFG["water_db"]).sum("time") / v.notnull().sum("time")

    # Lowered fields stay ponded after the monsoon (Oct to Dec) and deep pits hold water in the dry season
    lazy = {"waterPM": water_freq([10, 11, 12]), "waterDry": water_freq([2, 3, 4, 5])}
    # Rice by year from S1 VH. Monsoon clouds make S2 unreliable in Jul to Sep.
    # Rice shows as a flooded transplant (low VH) followed by a canopy rise.
    for y in years:
        flood = vh.sel(time=slice(f"{y}-06-15", f"{y}-08-31")).min("time")
        peak = vh.sel(time=slice(f"{y}-09-01", f"{y}-10-31")).max("time")
        lazy[f"rice{y}"] = (flood < CFG["rice_flood_db"]) & (peak - flood > CFG["rice_rise_db"])
        # Bare soil in rabi (Dec to Feb) while the neighbours are cropped suggests active digging
        ix = s2_indices(odc_load(items[f"rabi{y}"], S2_BANDS, tb, **S2_KW))
        lazy[f"bare{y}"] = ((ix.NDVI < CFG["bare_ndvi"]) & (ix.BSI > 0)).sum("time") / ix.NDVI.notnull().sum("time")
    # Kiln features: dry-season (Mar to May) medians
    lazy["k_VV"] = vv.sel(time=slice(f"{kiln_year}-03-01", f"{kiln_year}-05-31")).chunk(time=-1).median("time")
    kf = s2_indices(odc_load(items["kiln"], S2_BANDS, tb, **S2_KW)).chunk(time=-1).median("time")
    lazy.update({f"k_{b}": kf[b] for b in kf.data_vars})
    ds = xr.Dataset(lazy).compute()
    return {k: ds[k].values for k in ds.data_vars}


def neighbour_anom(a, res):
    """Value minus the NaN-aware mean of a ~1 km box around it, i.e. a field compared with its neighbours."""
    size = int(2 * CFG["neighbour_m"] / res) + 1
    v = np.isfinite(a)
    num = ndimage.uniform_filter(np.where(v, a, 0).astype("float32"), size)
    den = ndimage.uniform_filter(v.astype("float32"), size)
    return a - num / np.where(den > 0, den, np.nan)


def sieve(mask, min_px):
    lab, _ = ndimage.label(mask, structure=np.ones((3, 3)))
    keep = np.bincount(lab.ravel()) >= min_px
    keep[0] = False
    return keep[lab]


def to_gpkg(path, mask, gbox):
    geoms = [shape(g) for g, _ in features.shapes(mask.astype("uint8"), mask=mask,
                                                  transform=gbox.affine, connectivity=8)]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=CRS)
    gdf["area_ha"] = gdf.area / 1e4
    gdf.to_file(path)
    print(f"{path}: {len(gdf)} polygons")


def s2_indices(ds):
    ok = ds.SCL.isin([4, 5, 6, 7]) & (ds.B02 > 0)  # vegetation, bare, water, unclassified
    refl = ds[["B02", "B03", "B04", "B08", "B11"]].astype("float32")
    # Processing baseline 04.00 (from 2022-01-25) adds a +1000 offset to L2A reflectance
    refl = (refl - xr.where(ds.time >= np.datetime64("2022-01-25"), 1000, 0)) / 10000
    refl = refl.where(ok)
    refl["NDVI"] = (refl.B08 - refl.B04) / (refl.B08 + refl.B04)
    refl["BSI"] = ((refl.B11 + refl.B04) - (refl.B08 + refl.B02)) / ((refl.B11 + refl.B04) + (refl.B08 + refl.B02))
    return refl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aoi", help="district boundary (any vector file); default: approx. Anand bbox")
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"), help="lon/lat bbox")
    ap.add_argument("--first-year", type=int, default=2017)
    ap.add_argument("--last-year", type=int, default=2025)
    ap.add_argument("--kiln-year", type=int, default=2026, help="dry season (Mar to May) used for kiln mapping")
    ap.add_argument("--res", type=float, default=20, help="pixel size in metres")
    ap.add_argument("--kiln-points", help="points with a 'class' column (1 = kiln, 0 = other) to train a random forest")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    configure_rio(cloud_defaults=True)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    aoi = gpd.read_file(args.aoi).to_crs(4326) if args.aoi else None
    bbox = list(args.bbox or (aoi.total_bounds if aoi is not None else ANAND_BBOX))
    utm = gpd.GeoSeries([box(*bbox)], crs=4326).to_crs(CRS).total_bounds
    gbox = GeoBox.from_bbox(tuple(utm), crs=CRS, resolution=args.res)
    res, years = args.res, list(range(args.first_year, args.last_year + 1))
    print(f"grid {gbox.shape} at {res} m, years {years[0]}-{years[-1]}")

    # ---------------------------------------------------------------- masks
    # Exclude trees, built-up areas and permanent water (rivers, old village ponds).
    # WorldCover water is kept on purpose, since ponded excavations are what we are looking for.
    wc = odc_load(search("esa-worldcover", bbox, "2021"), ["map"], gbox, resampling="nearest")["map"].max("time").values
    gsw = odc_load(search("jrc-gsw", bbox, None), ["occurrence"], gbox, resampling="nearest").occurrence.max("time").values
    inside = np.ones(tuple(gbox.shape), bool)
    if aoi is not None:
        inside = features.geometry_mask(aoi.to_crs(CRS).geometry, out_shape=tuple(gbox.shape),
                                       transform=gbox.affine, invert=True)
    # The Mahi is seasonal (JRC occurrence ~30-60%), so it also needs a size rule: any large connected
    # body of "ever water" (WorldCover water or JRC occurrence >= 10%) plus a bank buffer is removed.
    ever_water = (wc == 80) | ((gsw >= 10) & (gsw <= 100))  # >100 is GSW nodata
    river = ndimage.binary_dilation(sieve(ever_water, CFG["river_min_ha"] * 1e4 / res**2),
                                    iterations=int(CFG["river_buffer_m"] / res))
    land = inside & (wc != 10) & (wc != 50) & ~((gsw > 50) & (gsw <= 100)) & ~river

    # ---------------------------------------------------------------- per-tile time-series stats
    cloud = {"eo:cloud_cover": {"lt": 60}}
    items = {"s1": search("sentinel-1-rtc", bbox, f"{years[0]}-01-01/{args.kiln_year}-05-31"),
             "kiln": search("sentinel-2-l2a", bbox, f"{args.kiln_year}-03-01/{args.kiln_year}-05-31", cloud)}
    for y in years:
        items[f"rabi{y}"] = search("sentinel-2-l2a", bbox, f"{y}-12-01/{y + 1}-02-28", cloud)

    # Tiles are cached in out/tiles so an interrupted run resumes. Delete that folder after
    # changing --bbox/--aoi/--res/years or the S1/S2 thresholds.
    (out / "tiles").mkdir(exist_ok=True)
    ny, nx = gbox.shape
    tiles = [(y0, x0) for y0 in range(0, ny, TILE) for x0 in range(0, nx, TILE)]
    full = {}
    for n, (y0, x0) in enumerate(tiles, 1):
        sl = np.s_[y0:min(y0 + TILE, ny), x0:min(x0 + TILE, nx)]
        if not inside[sl].any():
            continue
        cache = out / "tiles" / f"{y0}_{x0}.npz"
        if cache.exists():
            st = dict(np.load(cache))
        else:
            st = tile_stats(gbox[sl], items, years, args.kiln_year)
            np.savez_compressed(cache, **st)
        for k, a in st.items():
            full.setdefault(k, np.full((ny, nx), np.nan, "float32"))[sl] = a
        print(f"tile {n}/{len(tiles)} done", flush=True)

    water_pm, water_dry = full["waterPM"], full["waterDry"]
    water_anom = neighbour_anom(water_pm, res)
    rice = np.stack([full[f"rice{y}"] == 1 for y in years])
    rice_freq = rice.mean(0)
    first_rice = np.where(rice.any(0), np.array(years)[rice.argmax(0)], np.nan)
    # Switched to rice: none in the first 2 years, rice in at least 2 of the last 3
    rice_switch = ~rice[:2].any(0) & (rice[-3:].sum(0) >= 2)

    # Keeping the year with the strongest bare-soil anomaly gives a rough excavation date
    bare_anoms = np.stack([neighbour_anom(full[f"bare{y}"], res) for y in years])
    filled = np.where(np.isfinite(bare_anoms), bare_anoms, -np.inf)
    bare_anom = filled.max(0)
    bare_year = np.where(np.isfinite(bare_anom), np.array(years)[filled.argmax(0)] + 1, np.nan)
    bare_anom[~np.isfinite(bare_anom)] = np.nan

    # ---------------------------------------------------------------- Kilns
    kf = {k[2:]: a for k, a in full.items() if k.startswith("k_")}
    if args.kiln_points:
        from sklearn.ensemble import RandomForestClassifier
        feats = np.stack(list(kf.values()))  # (band, y, x)
        pts = gpd.read_file(args.kiln_points).to_crs(CRS)
        cols, rows = ~gbox.affine * (pts.geometry.x.values, pts.geometry.y.values)
        X, yl = feats[:, rows.astype(int), cols.astype(int)].T, pts["class"].values
        ok = np.isfinite(X).all(1)
        rf = RandomForestClassifier(100, n_jobs=-1).fit(X[ok], yl[ok])
        flat = feats.reshape(len(feats), -1).T
        valid = np.isfinite(flat).all(1)
        pred = np.zeros(len(flat), bool)
        pred[valid] = rf.predict(flat[valid]) == 1
        kiln = pred.reshape(tuple(gbox.shape))
    else:
        # ponytail: spectral heuristic (bare + reddish fired brick + bright SAR from stacks and chimney);
        # it will also hit red bare soil. Pass --kiln-points (~50 per class) to use the random forest instead.
        with np.errstate(invalid="ignore", divide="ignore"):
            kiln = ((kf["NDVI"] < CFG["kiln_ndvi"]) & (kf["B04"] / kf["B02"] > CFG["kiln_red_blue"])
                    & (kf["VV"] > CFG["kiln_vv_db"]))
    kiln = sieve(kiln, CFG["kiln_min_ha"] * 1e4 / res**2)
    # The uncalibrated heuristic flags kilns almost everywhere, so it only scores once trained with --kiln-points
    near_kiln = (ndimage.distance_transform_edt(~kiln) * res <= CFG["kiln_buffer_m"]) \
        if args.kiln_points and kiln.any() else np.zeros_like(kiln)

    # ---------------------------------------------------------------- Score + hotspots
    with np.errstate(invalid="ignore"):
        flags = {
            "f_water": (water_anom > CFG["water_anom_min"]) | (water_dry > CFG["water_dry_min"]),
            "f_rice": rice_switch,
            "f_bare": bare_anom > CFG["bare_anom_min"],
            "f_kiln": near_kiln,
        }
    score = np.where(land, sum(f.astype("float32") for f in flags.values()), np.nan)
    hot = sieve(np.nan_to_num(score) >= CFG["min_score"], CFG["min_hotspot_ha"] * 1e4 / res**2)

    layers = dict(waterPM=water_pm, waterDry=water_dry, waterAnom=water_anom, riceFreq=rice_freq,
                  firstRiceYear=first_rice, riceSwitch=rice_switch, bareAnom=bare_anom, bareYear=bare_year,
                  kiln=kiln, **flags, score=score, landMask=land)
    h, w = gbox.shape
    with rasterio.open(out / "layers.tif", "w", driver="GTiff", height=h, width=w, count=len(layers),
                       dtype="float32", crs=CRS, transform=gbox.affine, nodata=np.nan,
                       compress="deflate", tiled=True, BIGTIFF="IF_SAFER") as dst:
        for i, (name, a) in enumerate(layers.items(), 1):
            dst.write(np.where(inside, a, np.nan).astype("float32"), i)
            dst.set_band_description(i, name)
    print(f"{out / 'layers.tif'}: {list(layers)}")
    to_gpkg(out / "hotspots.gpkg", hot, gbox)
    to_gpkg(out / "kilns.gpkg", kiln, gbox)


if __name__ == "__main__":
    main()
