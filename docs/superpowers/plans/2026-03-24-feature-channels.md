# Multi-Channel Feature Download Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the download pipeline to produce a complete multi-channel training dataset — fire labels + weather/terrain/vegetation input features — in a single `wildfire download` command.

**Architecture:** Three new GEE feature modules (weather, terrain, features) produce `ee.Image` stacks at different temporal cadences (hourly, daily, static). A new `download_features()` orchestrator calls these alongside the existing `download_fire_stack()`, aligns temporal resolutions, and saves everything into a single multi-channel `.npz` output. The existing fire label pipeline is untouched — features are separate arrays that get stacked during training.

**Tech Stack:** Python, Google Earth Engine API, NumPy, existing `wildfire_pipeline` package infrastructure (config, retry, logging, I/O)

---

## File Structure

```
src/wildfire_pipeline/
  gee/
    weather.py      — CREATE: RTMA + GRIDMET + ERA5 weather feature extraction
    terrain.py      — CREATE: static terrain/fuel/vegetation/WUI features
    features.py     — CREATE: slow-varying features (NDVI, drought, LST)
    download.py     — MODIFY: add download_features() orchestrator
  config.py         — MODIFY: add FeatureConfig to PipelineConfig
  cli.py            — MODIFY: add --features flag to download command
tests/
  gee/
    test_weather.py  — CREATE: weather module unit tests
    test_terrain.py  — CREATE: terrain module unit tests
    test_features.py — CREATE: features module unit tests
config/
  fires.json        — MODIFY: add feature_config section
```

Each GEE module returns an `ee.Image` (multi-band) that gets sampled to NumPy in the orchestrator. This keeps GEE logic separate from sampling/stacking logic.

---

### Task 1: Weather Module — RTMA + GRIDMET Hourly/Daily Weather

**Files:**
- Create: `src/wildfire_pipeline/gee/weather.py`
- Test: `tests/gee/test_weather.py`

This module extracts weather features from two sources: RTMA (2.5km, hourly — wind, temp, dewpoint) and GRIDMET (4km, daily — ERC, BI, fuel moisture, VPD).

- [ ] **Step 1: Write test for RTMA dataset ID selection**

```python
# tests/gee/test_weather.py
"""Tests for weather feature extraction logic (no GEE auth needed)."""
from __future__ import annotations


class TestRtmaConfig:
    def test_rtma_dataset_id(self) -> None:
        from wildfire_pipeline.gee.weather import RTMA_DATASET
        assert RTMA_DATASET == "NOAA/NWS/RTMA"

    def test_rtma_bands_defined(self) -> None:
        from wildfire_pipeline.gee.weather import RTMA_BANDS
        assert "UGRD" in RTMA_BANDS  # u-wind
        assert "VGRD" in RTMA_BANDS  # v-wind
        assert "TMP" in RTMA_BANDS   # temperature
        assert "GUST" in RTMA_BANDS  # wind gust


class TestGridmetConfig:
    def test_gridmet_fire_weather_bands(self) -> None:
        from wildfire_pipeline.gee.weather import GRIDMET_FIRE_BANDS
        expected = {"erc", "bi", "fm100", "fm1000", "vpd", "rmin", "rmax", "vs", "th"}
        assert set(GRIDMET_FIRE_BANDS) == expected

    def test_gridmet_dataset_id(self) -> None:
        from wildfire_pipeline.gee.weather import GRIDMET_DATASET
        assert GRIDMET_DATASET == "IDAHO_EPSCOR/GRIDMET"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/gee/test_weather.py -v`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Write weather.py module**

```python
# src/wildfire_pipeline/gee/weather.py
"""Weather feature extraction from Google Earth Engine.

Sources:
- RTMA (NOAA/NWS/RTMA): 2.5km hourly — wind u/v, gust, temp, dewpoint
- GRIDMET (IDAHO_EPSCOR/GRIDMET): 4km daily — ERC, BI, fuel moisture, VPD, wind, humidity
- ERA5-Land (ECMWF/ERA5_LAND/HOURLY): 11km hourly — soil moisture
"""
from __future__ import annotations

import ee

RTMA_DATASET = "NOAA/NWS/RTMA"
RTMA_BANDS = ["UGRD", "VGRD", "GUST", "TMP", "DPT"]

GRIDMET_DATASET = "IDAHO_EPSCOR/GRIDMET"
GRIDMET_FIRE_BANDS = ["erc", "bi", "fm100", "fm1000", "vpd", "rmin", "rmax", "vs", "th"]

ERA5_LAND_DATASET = "ECMWF/ERA5_LAND/HOURLY"
GRIDMET_DROUGHT_DATASET = "GRIDMET/DROUGHT"


def get_hourly_rtma(aoi: ee.Geometry, hour_start: ee.Date, hour_end: ee.Date) -> ee.Image:
    """Get RTMA weather for one hour: wind u/v, gust, temp, dewpoint."""
    rtma = (
        ee.ImageCollection(RTMA_DATASET)
        .filterDate(hour_start, hour_end)
        .filterBounds(aoi)
        .select(RTMA_BANDS)
    )
    # Take mean if multiple images in the hour window
    return rtma.mean().unmask(0).toFloat()


def get_daily_gridmet(aoi: ee.Geometry, day_start: ee.Date, day_end: ee.Date) -> ee.Image:
    """Get GRIDMET daily fire weather: ERC, BI, fuel moisture, VPD, wind, humidity."""
    gridmet = (
        ee.ImageCollection(GRIDMET_DATASET)
        .filterDate(day_start, day_end)
        .filterBounds(aoi)
        .select(GRIDMET_FIRE_BANDS)
    )
    return gridmet.mean().unmask(0).toFloat()


def get_hourly_soil_moisture(aoi: ee.Geometry, hour_start: ee.Date, hour_end: ee.Date) -> ee.Image:
    """Get ERA5-Land soil moisture for one hour."""
    era5 = (
        ee.ImageCollection(ERA5_LAND_DATASET)
        .filterDate(hour_start, hour_end)
        .filterBounds(aoi)
        .select("volumetric_soil_water_layer_1")
    )
    return era5.mean().unmask(0).rename("soil_moisture").toFloat()


def get_drought_indices(aoi: ee.Geometry, fire_start: ee.Date) -> ee.Image:
    """Get most recent drought indices before/during fire."""
    drought = (
        ee.ImageCollection(GRIDMET_DROUGHT_DATASET)
        .filterDate(fire_start.advance(-30, "day"), fire_start)
        .filterBounds(aoi)
        .sort("system:time_start", False)
        .first()
    )
    pdsi = drought.select("pdsi").rename("pdsi")
    eddi14 = drought.select("eddi14d").rename("eddi_14d")
    eddi30 = drought.select("eddi30d").rename("eddi_30d")
    return pdsi.addBands(eddi14).addBands(eddi30).toFloat()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/gee/test_weather.py -v`
Expected: PASS

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check src/wildfire_pipeline/gee/weather.py && uv run ruff format src/wildfire_pipeline/gee/weather.py`

---

### Task 2: Terrain Module — Static Terrain/Fuel/WUI Features

**Files:**
- Create: `src/wildfire_pipeline/gee/terrain.py`
- Test: `tests/gee/test_terrain.py`

Static features computed once per fire AOI: terrain (6 bands), fuel/land cover (3 bands), vegetation (3 bands), fire history (2 bands), roads (1 band), population/WUI (2 bands). Total: 17 static bands.

- [ ] **Step 1: Write test for terrain band names**

```python
# tests/gee/test_terrain.py
"""Tests for terrain/fuel feature extraction logic."""
from __future__ import annotations


class TestTerrainBands:
    def test_terrain_band_names(self) -> None:
        from wildfire_pipeline.gee.terrain import TERRAIN_BANDS
        expected = {"slope_deg", "aspect_sin", "aspect_cos", "tpi", "terrain_ruggedness", "elevation"}
        assert set(TERRAIN_BANDS) == expected

    def test_fuel_band_names(self) -> None:
        from wildfire_pipeline.gee.terrain import FUEL_BANDS
        assert "fuel_load" in FUEL_BANDS
        assert "is_firebreak" in FUEL_BANDS
        assert "canopy_cover_pct" in FUEL_BANDS

    def test_static_feature_count(self) -> None:
        from wildfire_pipeline.gee.terrain import ALL_STATIC_BAND_COUNT
        assert ALL_STATIC_BAND_COUNT == 17

    def test_wui_bands_included(self) -> None:
        from wildfire_pipeline.gee.terrain import WUI_BANDS
        assert "population" in WUI_BANDS
        assert "built_up" in WUI_BANDS
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Write terrain.py module**

Port the logic from `gee/terrain_fuel.js` plus new GHSL population/WUI layers. Each function returns an `ee.Image`.

```python
# src/wildfire_pipeline/gee/terrain.py
"""Static terrain, fuel, and WUI feature extraction from Google Earth Engine.

Features computed once per fire AOI (17 bands total):
- Terrain (6): slope, aspect sin/cos, TPI, ruggedness, elevation
- Fuel (3): fuel load, firebreak indicator, canopy cover
- Vegetation (3): type, cover, height (LANDFIRE)
- Fire history (2): years since burn, burn count (MTBS)
- Roads (1): road presence
- WUI (2): population density, built-up surface
"""
from __future__ import annotations

import math

import ee

TERRAIN_BANDS = ["slope_deg", "aspect_sin", "aspect_cos", "tpi", "terrain_ruggedness", "elevation"]
FUEL_BANDS = ["fuel_load", "is_firebreak", "canopy_cover_pct"]
VEGETATION_BANDS = ["vegetation_type", "vegetation_cover", "vegetation_height"]
HISTORY_BANDS = ["years_since_burn", "burn_count"]
ROAD_BANDS = ["has_road"]
WUI_BANDS = ["population", "built_up"]
ALL_STATIC_BAND_COUNT = 17  # sum of all above


def get_terrain(aoi: ee.Geometry) -> ee.Image:
    """Compute terrain features from 3DEP 10m DEM."""
    dem = ee.Image("USGS/3DEP/10m").select("elevation")
    slope = ee.Terrain.slope(dem).rename("slope_deg")
    aspect = ee.Terrain.aspect(dem)
    aspect_rad = aspect.multiply(math.pi / 180)
    aspect_sin = aspect_rad.sin().rename("aspect_sin")
    aspect_cos = aspect_rad.cos().rename("aspect_cos")
    mean_elev = dem.reduceNeighborhood(
        reducer=ee.Reducer.mean(), kernel=ee.Kernel.circle(500, "meters")
    )
    tpi = dem.subtract(mean_elev).rename("tpi")
    ruggedness = dem.reduceNeighborhood(
        reducer=ee.Reducer.stdDev(), kernel=ee.Kernel.circle(500, "meters")
    ).rename("terrain_ruggedness")
    elevation = dem.rename("elevation")
    return (
        slope.addBands(aspect_sin).addBands(aspect_cos)
        .addBands(tpi).addBands(ruggedness).addBands(elevation)
    ).toFloat()


def get_fuel(aoi: ee.Geometry) -> ee.Image:
    """Compute fuel/land cover features from NLCD."""
    nlcd = ee.Image("USGS/NLCD_RELEASES/2021_REL/NLCD").select("landcover")
    fuel_load = nlcd.remap(
        [11,12,21,22,23,24,31,41,42,43,51,52,71,72,73,74,81,82,90,95],
        [0,0,0.1,0.05,0.02,0.01,0.05,0.7,0.9,0.8,0.6,0.8,0.5,0.4,0.3,0.3,0.3,0.2,0.6,0.4],
    ).rename("fuel_load")
    firebreak = nlcd.remap(
        [11,12,21,22,23,24,31,41,42,43,51,52,71,72,73,74,81,82,90,95],
        [1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ).rename("is_firebreak")
    canopy = (
        ee.Image("USGS/NLCD_RELEASES/2021_REL/NLCD")
        .select("percent_tree_canopy")
        .rename("canopy_cover_pct")
    )
    return fuel_load.addBands(firebreak).addBands(canopy).toFloat()


def get_vegetation() -> ee.Image:
    """Get LANDFIRE vegetation type, cover, height."""
    evt = ee.Image("LANDFIRE/Vegetation/EVT/v1_4_0").select("EVT").rename("vegetation_type")
    evc = ee.Image("LANDFIRE/Vegetation/EVC/v1_4_0").select("EVC").rename("vegetation_cover")
    evh = ee.Image("LANDFIRE/Vegetation/EVH/v1_4_0").select("EVH").rename("vegetation_height")
    return evt.addBands(evc).addBands(evh).toFloat()


def get_fire_history(aoi: ee.Geometry, fire_year: int) -> ee.Image:
    """Get MTBS burn history features."""
    mtbs = (
        ee.ImageCollection("USFS/GTAC/MTBS/annual_burn_severity_mosaics/v1")
        .filterDate("2000-01-01", f"{fire_year}-01-01")
        .filterBounds(aoi)
    )
    last_burn_year = mtbs.map(
        lambda img: img.select("Severity").gte(2).And(img.select("Severity").lte(4))
        .multiply(ee.Date(img.get("system:time_start")).get("year"))
        .selfMask().rename("burn_year")
    ).max()
    years_since = ee.Image(fire_year).subtract(last_burn_year).unmask(99).rename("years_since_burn")
    burn_count = mtbs.map(
        lambda img: img.select("Severity").gte(2).And(img.select("Severity").lte(4))
        .selfMask().rename("burned")
    ).count().unmask(0).rename("burn_count")
    return years_since.addBands(burn_count).toFloat()


def get_roads(aoi: ee.Geometry) -> ee.Image:
    """Get road presence as firebreak indicator."""
    roads = ee.FeatureCollection("TIGER/2016/Roads").filterBounds(aoi)
    return (
        roads.reduceToImage(properties=["linearid"], reducer=ee.Reducer.count())
        .gt(0).unmask(0).rename("has_road").toFloat()
    )


def get_wui(aoi: ee.Geometry) -> ee.Image:
    """Get population density and built-up surface for WUI context."""
    pop = ee.Image("JRC/GHSL/P2023A/GHS_POP").select("population_count").unmask(0).rename("population")
    built = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_S").select("built_surface").unmask(0).rename("built_up")
    return pop.addBands(built).toFloat()


def get_all_static(aoi: ee.Geometry, fire_year: int) -> ee.Image:
    """Get all 17 static feature bands combined."""
    result: ee.Image = (
        get_terrain(aoi)
        .addBands(get_fuel(aoi))
        .addBands(get_vegetation())
        .addBands(get_fire_history(aoi, fire_year))
        .addBands(get_roads(aoi))
        .addBands(get_wui(aoi))
    )
    return result
```

- [ ] **Step 4: Run tests, lint, commit**

---

### Task 3: Slow-Varying Features Module — NDVI, LST

**Files:**
- Create: `src/wildfire_pipeline/gee/features.py`
- Test: `tests/gee/test_features.py`

Features that change over days/weeks: NDVI/EVI (16-day), LST (daily).

- [ ] **Step 1: Write tests**

```python
# tests/gee/test_features.py
"""Tests for slow-varying feature extraction logic."""
from __future__ import annotations


class TestNdviConfig:
    def test_ndvi_dataset_id(self) -> None:
        from wildfire_pipeline.gee.features import NDVI_DATASET
        assert NDVI_DATASET == "MODIS/061/MOD13Q1"

    def test_ndvi_bands(self) -> None:
        from wildfire_pipeline.gee.features import NDVI_BANDS
        assert "NDVI" in NDVI_BANDS
        assert "EVI" in NDVI_BANDS


class TestLstConfig:
    def test_lst_dataset_id(self) -> None:
        from wildfire_pipeline.gee.features import LST_DATASET
        assert LST_DATASET == "MODIS/061/MOD11A1"
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Write features.py**

```python
# src/wildfire_pipeline/gee/features.py
"""Slow-varying feature extraction: NDVI/EVI and Land Surface Temperature.

Sources:
- MODIS MOD13Q1: 250m, 16-day NDVI and EVI composites
- MODIS MOD11A1: 1km, daily Land Surface Temperature (day + night)
"""
from __future__ import annotations

import ee

NDVI_DATASET = "MODIS/061/MOD13Q1"
NDVI_BANDS = ["NDVI", "EVI"]

LST_DATASET = "MODIS/061/MOD11A1"


def get_pre_fire_ndvi(aoi: ee.Geometry, fire_start: ee.Date) -> ee.Image:
    """Get most recent NDVI/EVI composite before fire start."""
    ndvi = (
        ee.ImageCollection(NDVI_DATASET)
        .filterDate(fire_start.advance(-32, "day"), fire_start)
        .filterBounds(aoi)
        .select(NDVI_BANDS)
        .sort("system:time_start", False)
        .first()
    )
    # MODIS NDVI has scale factor 0.0001
    return ndvi.multiply(0.0001).unmask(0).toFloat()


def get_daily_lst(aoi: ee.Geometry, day_start: ee.Date, day_end: ee.Date) -> ee.Image:
    """Get MODIS daily land surface temperature (day and night)."""
    lst = (
        ee.ImageCollection(LST_DATASET)
        .filterDate(day_start, day_end)
        .filterBounds(aoi)
        .first()
    )
    lst_day = lst.select("LST_Day_1km").multiply(0.02).unmask(0).rename("lst_day_k")
    lst_night = lst.select("LST_Night_1km").multiply(0.02).unmask(0).rename("lst_night_k")
    return lst_day.addBands(lst_night).toFloat()
```

- [ ] **Step 4: Run tests, lint, commit**

---

### Task 4: Config — Add Feature Download Settings

**Files:**
- Modify: `src/wildfire_pipeline/config.py`
- Modify: `config/fires.json`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write test for new config field**

```python
# Add to tests/test_config.py TestPipelineConfig class:
def test_download_features_default_true(self):
    pc = PipelineConfig()
    assert pc.download_features is True

def test_rtma_wind_default_true(self):
    pc = PipelineConfig()
    assert pc.rtma_wind is True
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Add fields to PipelineConfig**

```python
# In config.py PipelineConfig class, add:
    download_features: bool = Field(
        default=True,
        description="Download weather/terrain/vegetation features alongside fire labels",
    )
    rtma_wind: bool = Field(
        default=True,
        description="Use RTMA for hourly wind (2.5km). Falls back to GRIDMET daily if False.",
    )
```

- [ ] **Step 4: Run tests, lint, commit**

---

### Task 5: Feature Download Orchestrator

**Files:**
- Modify: `src/wildfire_pipeline/gee/download.py`
- Test: `tests/gee/test_feature_download.py`

This is the core integration: a `download_features()` function that calls the weather/terrain/features modules, samples everything to NumPy, and saves alongside the fire data.

- [ ] **Step 1: Write test for feature channel count**

```python
# tests/gee/test_feature_download.py
"""Tests for the feature download orchestrator (pure logic, no GEE)."""
from __future__ import annotations


class TestFeatureChannelInventory:
    """Verify the expected channel counts for each feature category."""

    def test_hourly_weather_channels(self) -> None:
        from wildfire_pipeline.gee.weather import RTMA_BANDS
        # RTMA: UGRD, VGRD, GUST, TMP, DPT = 5 bands
        # + soil_moisture = 1 band
        assert len(RTMA_BANDS) + 1 == 6

    def test_daily_weather_channels(self) -> None:
        from wildfire_pipeline.gee.weather import GRIDMET_FIRE_BANDS
        # erc, bi, fm100, fm1000, vpd, rmin, rmax, vs, th = 9 bands
        assert len(GRIDMET_FIRE_BANDS) == 9

    def test_static_channels(self) -> None:
        from wildfire_pipeline.gee.terrain import ALL_STATIC_BAND_COUNT
        assert ALL_STATIC_BAND_COUNT == 17

    def test_slow_varying_channels(self) -> None:
        from wildfire_pipeline.gee.features import NDVI_BANDS
        # NDVI + EVI = 2 bands, LST day + night = 2 bands, drought = 3 bands
        assert len(NDVI_BANDS) == 2
```

- [ ] **Step 2: Write download_features() in download.py**

Add to `src/wildfire_pipeline/gee/download.py`:

```python
def download_features(
    fire_name: str,
    config: FiresConfig | dict,
    output_dir: Path,
    fmt: str = "npz",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Download weather/terrain/vegetation features for one fire event.

    Downloads three categories at different temporal cadences, all resampled
    to the GOES 2km grid:
    - Hourly: RTMA wind/temp/dewpoint + ERA5 soil moisture (T, H, W, 6)
    - Daily: GRIDMET fire weather + MODIS LST (repeated 24x per day)
    - Static: terrain/fuel/WUI features (broadcast to all timesteps)

    Returns:
        Tuple of (feature arrays dict, metadata dict).
    """
    cfg = _ensure_fires_config(config)
    fire = cfg.fires[fire_name]
    pipeline = cfg.pipeline_config
    scale = pipeline.export_scale_m

    aoi_coords = list(fire.aoi)
    aoi = ee.Geometry.Rectangle(aoi_coords)
    start = ee.Date(fire.start_utc.isoformat())
    n_hours = fire.n_hours

    n_rows, n_cols = compute_grid_shape(aoi_coords, scale)

    logger.info("downloading_features", fire_name=fire_name, n_hours=n_hours)

    fire_dir = output_dir / fire_name
    fire_dir.mkdir(parents=True, exist_ok=True)

    # --- Static features (computed once) ---
    from wildfire_pipeline.gee.terrain import get_all_static
    static_img = get_all_static(aoi, fire.year)
    static_sample = safe_sample_rectangle(static_img, aoi)
    static_bands = {}
    for band in static_img.bandNames().getInfo():
        static_bands[band] = np.array(
            safe_get_info(static_sample.get(band)), dtype=np.float32
        )

    # --- Slow-varying features (pre-fire NDVI, drought) ---
    from wildfire_pipeline.gee.features import get_pre_fire_ndvi
    from wildfire_pipeline.gee.weather import get_drought_indices
    ndvi_img = get_pre_fire_ndvi(aoi, start)
    drought_img = get_drought_indices(aoi, start)
    slow_img = ndvi_img.addBands(drought_img)
    slow_sample = safe_sample_rectangle(slow_img, aoi)
    slow_bands = {}
    for band in slow_img.bandNames().getInfo():
        slow_bands[band] = np.array(
            safe_get_info(slow_sample.get(band)), dtype=np.float32
        )

    # --- Hourly features (RTMA + soil moisture) ---
    from wildfire_pipeline.gee.weather import get_hourly_rtma, get_hourly_soil_moisture

    hourly_stacks: dict[str, list[np.ndarray]] = {}
    for h in range(n_hours):
        hour_start = start.advance(h, "hour")
        hour_end = start.advance(h + 1, "hour")

        try:
            if pipeline.rtma_wind:
                rtma = get_hourly_rtma(aoi, hour_start, hour_end)
                rtma_sample = safe_sample_rectangle(rtma, aoi)
                for band in ["UGRD", "VGRD", "GUST", "TMP", "DPT"]:
                    arr = np.array(safe_get_info(rtma_sample.get(band)), dtype=np.float32)
                    hourly_stacks.setdefault(band.lower(), []).append(arr)

            soil = get_hourly_soil_moisture(aoi, hour_start, hour_end)
            soil_sample = safe_sample_rectangle(soil, aoi)
            arr = np.array(safe_get_info(soil_sample.get("soil_moisture")), dtype=np.float32)
            hourly_stacks.setdefault("soil_moisture", []).append(arr)

        except Exception as e:
            logger.warning("feature_hour_failed", hour=h, error=str(e))
            shape = (n_rows, n_cols)
            for key in ["ugrd", "vgrd", "gust", "tmp", "dpt", "soil_moisture"]:
                hourly_stacks.setdefault(key, []).append(np.zeros(shape, dtype=np.float32))

        step = max(1, min(24, n_hours // 10))
        if (h + 1) % step == 0 or h + 1 == n_hours:
            pct = 100 * (h + 1) / n_hours
            logger.info("feature_progress", hour=h + 1, total=n_hours, pct=f"{pct:.0f}%")

    # --- Daily features (GRIDMET + LST, repeated for 24 hours) ---
    from wildfire_pipeline.gee.features import get_daily_lst
    from wildfire_pipeline.gee.weather import get_daily_gridmet

    n_days = (n_hours + 23) // 24
    daily_stacks: dict[str, list[np.ndarray]] = {}
    for d in range(n_days):
        day_start = start.advance(d, "day")
        day_end = start.advance(d + 1, "day")

        try:
            gridmet = get_daily_gridmet(aoi, day_start, day_end)
            gridmet_sample = safe_sample_rectangle(gridmet, aoi)
            for band in ["erc", "bi", "fm100", "fm1000", "vpd", "rmin", "rmax", "vs", "th"]:
                arr = np.array(safe_get_info(gridmet_sample.get(band)), dtype=np.float32)
                daily_stacks.setdefault(band, []).append(arr)

            lst = get_daily_lst(aoi, day_start, day_end)
            lst_sample = safe_sample_rectangle(lst, aoi)
            for band in ["lst_day_k", "lst_night_k"]:
                arr = np.array(safe_get_info(lst_sample.get(band)), dtype=np.float32)
                daily_stacks.setdefault(band, []).append(arr)

        except Exception as e:
            logger.warning("feature_day_failed", day=d, error=str(e))
            shape = (n_rows, n_cols)
            for key in ["erc", "bi", "fm100", "fm1000", "vpd", "rmin", "rmax",
                         "vs", "th", "lst_day_k", "lst_night_k"]:
                daily_stacks.setdefault(key, []).append(np.zeros(shape, dtype=np.float32))

    # --- Assemble output arrays ---
    arrays: dict[str, np.ndarray] = {}

    # Static: (H, W) per band
    for name, arr in static_bands.items():
        arrays[f"static_{name}"] = arr

    # Slow-varying: (H, W) per band
    for name, arr in slow_bands.items():
        arrays[f"slow_{name}"] = arr

    # Hourly: stack to (T, H, W)
    for name, frames in hourly_stacks.items():
        arrays[f"hourly_{name}"] = np.stack(frames, axis=0)

    # Daily: repeat each day 24x to align with hourly, then trim to n_hours
    for name, day_frames in daily_stacks.items():
        repeated = np.repeat(np.stack(day_frames, axis=0), 24, axis=0)[:n_hours]
        arrays[f"daily_{name}"] = repeated

    metadata: dict[str, Any] = {
        "fire_name": fire_name,
        "year": fire.year,
        "n_hours": n_hours,
        "grid_shape": [n_rows, n_cols],
        "aoi": aoi_coords,
        "feature_type": "multi_channel",
        "hourly_bands": sorted(hourly_stacks.keys()),
        "daily_bands": sorted(daily_stacks.keys()),
        "static_bands": sorted(static_bands.keys()),
        "slow_bands": sorted(slow_bands.keys()),
    }

    out_path = fire_dir / f"{fire_name}_{fire.year}_Features"
    actual_path = save_fire_data(out_path, arrays, metadata, fmt=fmt)
    logger.info("features_saved", path=str(actual_path))

    return arrays, metadata
```

- [ ] **Step 3: Run tests, lint, commit**

---

### Task 6: CLI Integration — Add --features Flag

**Files:**
- Modify: `src/wildfire_pipeline/cli.py`

- [ ] **Step 1: Add --features flag to download command**

```python
@app.command()
def download(
    fire: Annotated[str, typer.Argument(help="Fire name from config")],
    config: Annotated[Path, typer.Option(help="Config file path")] = Path("config/fires.json"),
    output: Annotated[Path, typer.Option(help="Output directory")] = Path("data"),
    fmt: Annotated[str, typer.Option("--format", help="Output format")] = "npz",
    features: Annotated[bool, typer.Option("--features/--no-features", help="Download input features")] = True,
) -> None:
    """Download fire detection data from Google Earth Engine."""
    # ... existing fire download code ...

    # After fire download, also download features if requested
    if features:
        from wildfire_pipeline.gee.download import download_features
        try:
            download_features(fire, cfg, output_dir, fmt=fmt)
        except Exception as e:
            logger.error("feature_download_failed", error=str(e))
            # Features are optional — don't exit on failure
            logger.warning("continuing_without_features")
```

- [ ] **Step 2: Run CLI help to verify flag appears**

Run: `uv run wildfire download --help`
Expected: Shows `--features / --no-features` option

- [ ] **Step 3: Lint, commit**

---

### Task 7: Update DVC Pipeline and Config

**Files:**
- Modify: `config/fires.json`
- Modify: `dvc.yaml`

- [ ] **Step 1: Add new config fields to fires.json**

```json
{
  "pipeline_config": {
    "export_scale_m": 2004,
    "export_crs": "EPSG:3857",
    "goes_confidence_threshold": 0.30,
    "download_features": true,
    "rtma_wind": true,
    ...
  }
}
```

- [ ] **Step 2: Update DVC stages**

The download stages now produce both fire data and feature data in the same output directory.

- [ ] **Step 3: Run full test suite**

Run: `uv run python -m pytest tests/ -v --tb=short`
Expected: All existing tests pass + new tests pass

- [ ] **Step 4: Final lint + mypy + format check**

Run: `uv run ruff check . && uv run mypy src/ && uv run ruff format --check .`
Expected: All clean

- [ ] **Step 5: Commit all changes**
