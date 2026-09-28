# Clay-excavation depressions around brick kilns — Anand district, Gujarat

## Context

Around brick kilns in Anand district (Gujarat, India), topsoil and clay are dug out of
agricultural fields, lowering them by ~1–2 m. The excavated areas are field-sized (tens to hundreds
of metres wide), often have sharp edges against untouched neighbouring fields, and are frequently
converted to rice because they hold water. The goal is to map these lowered fields.

Study area: Anand district (~3,000 km²), a very flat alluvial plain (Charotar), with a focus around
Adas village (Anand Rural taluka, ~11 km south of Anand, near Vasad and the Mahi river).

No free LiDAR exists here and free 30 m DEMs are too coarse, so the work is tiered:

1. **Screen the whole district with free satellite data** (this repository, step 1): where does water
   persist longer than in neighbouring fields, which fields switched to rice, which fields are bare
   at unusual times, and where are the kilns.
2. **Elevation over hotspot clusters only**: buy dry-season stereo imagery (Pléiades tri-stereo or
   Cartosat-3) and flag parcels ≥ ~1 m below their surroundings.
3. **Validate** with drone photogrammetry near Adas and ICESat-2 tracks.

All detection uses **relative** signals — a field compared with its ~1 km neighbourhood — because
rice and bunded fields are common everywhere, so an absolute signal proves little.

## Scripts

| File | Runs on | Notes |
|---|---|---|
| `step1_screening.py` | your machine, data from Microsoft Planetary Computer | Main version, documented below. No account needed. |

## Running `step1_screening.py`

Requires [uv](https://docs.astral.sh/uv/). It reads the dependencies from the script header and
installs them itself, so there's nothing else to set up.

```bash
# Quick test: 12×12 km around Adas, 2 years (~1 minute)
uv run step1_screening.py --bbox 72.96 22.42 73.08 22.53 --first-year 2023 --last-year 2024 --kiln-year 2025 --out out_test

# Whole district, 2017–2025 (several hours)
uv run step1_screening.py --aoi anand.gpkg

# With trained kiln detection
uv run step1_screening.py --aoi anand.gpkg --kiln-points kiln_points.gpkg
```

| Option | Default | Meaning |
|---|---|---|
| `--aoi FILE` | none | District boundary (any vector file). Outputs are clipped to it. |
| `--bbox W S E N` | approx. Anand extent | Area as a lon/lat box, used when there is no `--aoi`. |
| `--first-year`, `--last-year` | 2017, 2025 | Years of the time series. The rice switch needs at least 5 years. |
| `--kiln-year` | 2026 | Year whose March–May dry season is used for kiln mapping. |
| `--res` | 20 | Pixel size in metres. 10 m is 4× slower and heavier. |
| `--kiln-points FILE` | none | Points with a `class` column (1 = kiln, 0 = not kiln), ~50 per class, to train a random forest. |
| `--out DIR` | `out` | Output folder. |

Without `--aoi`, the script covers a rough rectangle around the district. For the real outline, use a
district boundary file, for example India ADM2 from [geoBoundaries](https://www.geoboundaries.org/).

### Data used

All from [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/), read on demand:

- **Sentinel-1 RTC** (radar, VV and VH): water frequency, rice detection, kiln brightness.
  Since Sentinel-1B was lost (Dec 2021), the area has one descending pass every 12 days.
- **Sentinel-2 L2A** (optical, scenes < 60% cloud, masked with the scene classification band):
  bare soil and kiln colour.
- **ESA WorldCover 2021** and **JRC Global Surface Water**: masks for trees, built-up areas,
  permanent water and the river corridor.

### Tiles, memory and resuming

The area is processed in 512×512-pixel tiles (~10×10 km at 20 m), which keeps memory to a few GB.
Each finished tile is saved to `DIR/tiles/`, and rerunning the same command resumes from there.
Progress is printed as `tile n/N done`.

Delete `DIR/tiles/` when you change the area, `--res`, the years, `--kiln-year`, or the tile-level
thresholds (`water_db`, `rice_flood_db`, `rice_rise_db`, `bare_ndvi`). All other thresholds are
applied after the tiles, so changing them only requires a rerun, which is quick with the cache.

### Outputs

| File | Content |
|---|---|
| `layers.tif` | One band per layer (see below), EPSG:32643 (UTM 43N), float32. |
| `hotspots.gpkg` | Hotspot polygons with `area_ha`. |
| `kilns.gpkg` | Kiln polygons with `area_ha`. |

### Calibration

Every threshold is in the `CFG` block at the top of the script. They are typical published starting
values, **not calibrated for this area**. Tune them on a few fields near Adas that you know are dug,
undug, rice or not rice.

| Setting | Default | Effect |
|---|---|---|
| `water_db` | −15 dB | VV below this counts as open water. |
| `rice_flood_db`, `rice_rise_db` | −21 dB, 5 dB | Rice test (see `riceFreq`). If known paddies show no rice, try −19. |
| `bare_ndvi` | 0.2 | NDVI below this (and BSI > 0) counts as bare soil. |
| `neighbour_m` | 500 m | Half-width of the neighbourhood window used by the anomaly layers. |
| `water_anom_min`, `water_dry_min`, `bare_anom_min` | 0.2, 0.3, 0.3 | Cut-offs for the flags. |
| `kiln_*` | | Untrained kiln rule, and the 3 km "near a kiln" distance. |
| `min_score`, `min_hotspot_ha` | 2, 1 ha | How many flags, and how large an area, make a hotspot. |
| `river_min_ha`, `river_buffer_m` | 50 ha, 100 m | Size above which a water body is treated as river or reservoir, and the bank strip removed with it. |

The Sentinel-1 thresholds are about 1 dB higher than in the Earth Engine script: Planetary
Computer's RTC product (gamma0) reads brighter than Earth Engine's sigma0.

## Layers in `layers.tif`

### Water (Sentinel-1 VV, open water = VV < `water_db`)

| Band | Values | Meaning |
|---|---|---|
| `waterPM` | 0–1 | Share of October–December passes when the pixel was water. Lowered fields keep monsoon water longer. |
| `waterDry` | 0–1 | Same for February–May. High values point to deep pits that hold water all year (or ponds the masks missed). |
| `waterAnom` | about −1 to +1 | `waterPM` minus its ~1 km neighbourhood mean. **Positive = ponds more than its neighbours**, the main water signal. |

### Rice (Sentinel-1 VH, one test per year)

A pixel is rice in a year when it is flooded at planting (lowest VH between 15 June and 31 August
below `rice_flood_db`) **and** the crop then grows (highest VH in September–October at least
`rice_rise_db` above that low point). A field that floods but grows nothing fails the test and shows
up in the water layers instead.

| Band | Values | Meaning |
|---|---|---|
| `riceFreq` | 0–1 | Share of years the pixel was rice. |
| `firstRiceYear` | year, empty if never | First year rice was detected. Helps date a conversion. |
| `riceSwitch` | 0/1 | No rice in the first 2 years, rice in at least 2 of the last 3: a field that switched to rice. Always 0 when the period is shorter than 5 years. |

### Bare soil (Sentinel-2, December–February)

| Band | Values | Meaning |
|---|---|---|
| `bareAnom` | about −1 to +1 | Share of rabi-season scenes where the pixel was bare, minus its neighbourhood mean, taking the strongest year. Bare while neighbours are cropped suggests digging. |
| `bareYear` | year | The year (Jan–Feb) of that strongest anomaly: a rough excavation date. |

### Kilns

| Band | Values | Meaning |
|---|---|---|
| `kiln` | 0/1 | Kiln pixels, groups ≥ 0.2 ha. Without `--kiln-points` this is a loose spectral rule (bare, reddish, radar-bright) that also catches red bare soil. |

### Flags, score and mask

| Band | Values | Meaning |
|---|---|---|
| `f_water` | 0/1 | `waterAnom` > 0.2 **or** `waterDry` > 0.3 |
| `f_rice` | 0/1 | `riceSwitch` |
| `f_bare` | 0/1 | `bareAnom` > 0.3 |
| `f_kiln` | 0/1 | Within 3 km of a kiln. **Only set with `--kiln-points`**, because the untrained rule flags almost everywhere. |
| `score` | 0–4 | Sum of the flags (at most 3 without `--kiln-points`). Empty where `landMask` = 0. |
| `landMask` | 0/1 | 1 = analysed. 0 = excluded as trees, built-up, permanent water, or the river corridor (large water bodies plus a bank buffer). |

`hotspots.gpkg` holds the areas with `score` ≥ `min_score` covering at least `min_hotspot_ha`.

## Viewing in QGIS

### Loading

Drag `layers.tif`, `hotspots.gpkg` and `kilns.gpkg` into QGIS, or use **Layer ▸ Add Layer ▸
Add Raster Layer / Add Vector Layer**. For context, add a satellite basemap: in the Browser panel,
right-click **XYZ Tiles ▸ New Connection** and use, for example,
`https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}` (Google Satellite) or Esri World Imagery.

### Showing one band

QGIS draws any raster with 3 or more bands as a colour image of bands 1–3 (`waterPM`, `waterDry`,
`waterAnom`), so the other bands seem missing. To show one band:

1. Right-click `layers.tif` ▸ **Properties** ▸ **Symbology**.
2. Set **Render type** to **Singleband pseudocolor**.
3. Pick the band in **Band** (e.g. `Band 4: riceFreq`), set **Min/Max**, choose a ramp, click
   **Classify**, then **Apply**.

To see several bands side by side, right-click the layer ▸ **Duplicate Layer** once per band and
rename each copy after its band. To read all 15 values at one spot, use the **Identify Features**
tool (the "i" cursor) and click a pixel.

### Suggested styles

| Band | Render type | Min–max | Ramp |
|---|---|---|---|
| `waterPM`, `waterDry` | Singleband pseudocolor | 0–0.6 | white → blue |
| `waterAnom` | Singleband pseudocolor | −0.3–0.3 | brown → white → blue (e.g. `BrBG`) |
| `riceFreq` | Singleband pseudocolor | 0–1 | white → green (`Greens`) |
| `firstRiceYear`, `bareYear` | Singleband pseudocolor | first–last year | yellow → red (`YlOrRd`) |
| `bareAnom` | Singleband pseudocolor | −0.3–0.3 | green → white → brown |
| `riceSwitch`, `kiln`, `f_*`, `landMask` | Paletted/Unique values | 0, 1 | 0 transparent, 1 in a strong colour |
| `score` | Paletted/Unique values | 0–4 | white, yellow, orange, red, dark red |

To make 0 transparent: **Properties ▸ Transparency ▸ Additional no data value** = `0`.

For `hotspots.gpkg` and `kilns.gpkg`: **Symbology ▸ Simple fill**, **Fill style** = *No brush*, a
bright outline (cyan for hotspots, magenta for kilns), 0.6 mm wide, so the imagery stays visible.
Label hotspots with `area_ha` under **Labels ▸ Single labels**.

### Checking hotspots

For each hotspot, look at the basemap and at historical imagery (Google Earth Pro ▸ View ▸
Historical Imagery) for a sharp-edged field below its neighbours, standing water, or digging scars.
Compare with `firstRiceYear` and `bareYear` to date the excavation. Rivers, ponds and village tanks
that slip through the masks are the usual false positives.

## Known limitations

- Thresholds are uncalibrated (see [Calibration](#calibration)).
- The untrained kiln rule over-detects heavily. Digitise ~50 kiln and ~50 non-kiln points in QGIS
  (a point layer with an integer `class` field) and pass them with `--kiln-points`.
- WorldCover classes many field boundaries with trees as tree cover, so a large share can be masked
  out (~44% in the Adas test area). If real fields fall inside `landMask` = 0, remove the tree
  exclusion (`wc != 10`) in the script.
- One Sentinel-1 pass every 12 days since 2022 can miss short rice-flooding periods.
- Speckle is reduced with a simple 3×3 filter, so field edges in the radar layers can look ragged.
