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

TERRAIN_BANDS = [
    "slope_deg",
    "aspect_sin",
    "aspect_cos",
    "tpi",
    "terrain_ruggedness",
    "elevation",
]
FUEL_BANDS = ["fuel_load", "is_firebreak", "canopy_cover_pct"]
VEGETATION_BANDS = ["vegetation_type", "vegetation_cover", "vegetation_height"]
HISTORY_BANDS = ["years_since_burn", "burn_count"]
ROAD_BANDS = ["has_road"]
WUI_BANDS = ["population", "built_up"]
ALL_STATIC_BAND_COUNT = 17  # sum of all above


def get_terrain(aoi: ee.Geometry) -> ee.Image:
    """Compute terrain features from 3DEP 10m DEM."""
    dem = ee.ImageCollection("USGS/3DEP/10m_collection").mosaic().select("elevation")
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
    result: ee.Image = (
        slope.addBands(aspect_sin)
        .addBands(aspect_cos)
        .addBands(tpi)
        .addBands(ruggedness)
        .addBands(elevation)
    ).toFloat()
    return result


def get_fuel(aoi: ee.Geometry) -> ee.Image:
    """Compute fuel/land cover features from NLCD."""
    nlcd_img = ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD").first()
    nlcd = nlcd_img.select("landcover")
    fuel_load = nlcd.remap(
        [11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82, 90, 95],
        [
            0,
            0,
            0.1,
            0.05,
            0.02,
            0.01,
            0.05,
            0.7,
            0.9,
            0.8,
            0.6,
            0.8,
            0.5,
            0.4,
            0.3,
            0.3,
            0.3,
            0.2,
            0.6,
            0.4,
        ],
    ).rename("fuel_load")
    firebreak = nlcd.remap(
        [11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82, 90, 95],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ).rename("is_firebreak")
    canopy = (
        ee.ImageCollection("MODIS/061/MOD44B")
        .sort("system:time_start", False)
        .first()
        .select("Percent_Tree_Cover")
        .unmask(0)
        .rename("canopy_cover_pct")
    )
    result: ee.Image = fuel_load.addBands(firebreak).addBands(canopy).toFloat()
    return result


def get_vegetation() -> ee.Image:
    """Get LANDFIRE vegetation type, cover, height."""
    # LANDFIRE ImageCollections contain images with mixed integer types
    # (Short vs Integer<0,65535>). Cast each image to float BEFORE mosaic
    # to avoid sampleRectangle "incompatible band" errors.
    evt = (
        ee.ImageCollection("LANDFIRE/Vegetation/EVT/v1_4_0")
        .map(lambda img: img.select("EVT").toFloat())
        .mosaic()
        .rename("vegetation_type")
    )
    evc = (
        ee.ImageCollection("LANDFIRE/Vegetation/EVC/v1_4_0")
        .map(lambda img: img.select("EVC").toFloat())
        .mosaic()
        .rename("vegetation_cover")
    )
    evh = (
        ee.ImageCollection("LANDFIRE/Vegetation/EVH/v1_4_0")
        .map(lambda img: img.select("EVH").toFloat())
        .mosaic()
        .rename("vegetation_height")
    )
    result: ee.Image = evt.addBands(evc).addBands(evh)
    return result


def get_fire_history(aoi: ee.Geometry, fire_year: int) -> ee.Image:
    """Get MTBS burn history features."""
    mtbs = (
        ee.ImageCollection("USFS/GTAC/MTBS/annual_burn_severity_mosaics/v1")
        .filterDate("2000-01-01", f"{fire_year}-01-01")
        .filterBounds(aoi)
    )
    # Cast to float everywhere — MTBS has mixed integer subtypes that cause
    # sampleRectangle "incompatible band" errors when combined with other bands.
    last_burn_year = mtbs.map(
        lambda img: (
            img.select("Severity")
            .gte(2)
            .And(img.select("Severity").lte(4))
            .toFloat()
            .multiply(ee.Number(ee.Date(img.get("system:time_start")).get("year")).toFloat())
            .selfMask()
            .rename("burn_year")
        )
    ).max()
    years_since = (
        ee.Image.constant(fire_year)
        .toFloat()
        .subtract(last_burn_year.toFloat())
        .unmask(99)
        .rename("years_since_burn")
    )
    burn_count = (
        mtbs.map(
            lambda img: (
                img.select("Severity")
                .gte(2)
                .And(img.select("Severity").lte(4))
                .toFloat()
                .selfMask()
                .rename("burned")
            )
        )
        .count()
        .toFloat()
        .unmask(0)
        .rename("burn_count")
    )
    return years_since.toFloat().addBands(burn_count.toFloat())


def get_roads(aoi: ee.Geometry) -> ee.Image:
    """Get road presence as firebreak indicator."""
    roads = ee.FeatureCollection("TIGER/2016/Roads").filterBounds(aoi)
    result: ee.Image = ee.Image(0).paint(roads, 1).gt(0).unmask(0).rename("has_road").toFloat()
    return result


def get_wui(aoi: ee.Geometry) -> ee.Image:
    """Get population density and built-up surface for WUI context."""
    pop = (
        ee.ImageCollection("JRC/GHSL/P2023A/GHS_POP")
        .sort("system:time_start", False)
        .first()
        .select("population_count")
        .unmask(0)
        .rename("population")
    )
    built = (
        ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_S")
        .sort("system:time_start", False)
        .first()
        .select("built_surface")
        .unmask(0)
        .rename("built_up")
    )
    result: ee.Image = pop.addBands(built).toFloat()
    return result


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
