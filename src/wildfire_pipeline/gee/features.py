"""Slow-varying feature extraction: NDVI/EVI, Land Surface Temperature, and Smoke.

Sources:
- MODIS MOD13Q1: 250m, 16-day NDVI and EVI composites
- MODIS MOD11A1: 1km, daily Land Surface Temperature (day + night)
- TROPOMI S5P (COPERNICUS/S5P/OFFL/L3_AER_AI): UV Aerosol Index (smoke through clouds)
"""

from __future__ import annotations

import ee

NDVI_DATASET = "MODIS/061/MOD13Q1"
NDVI_BANDS = ["NDVI", "EVI"]

LST_DATASET = "MODIS/061/MOD11A1"

TROPOMI_AAI_DATASET = "COPERNICUS/S5P/OFFL/L3_AER_AI"


def get_pre_fire_ndvi(aoi: ee.Geometry, fire_start: ee.Date) -> ee.Image:
    """Get most recent NDVI/EVI composite before fire start."""
    collection = (
        ee.ImageCollection(NDVI_DATASET)
        .filterDate(fire_start.advance(-64, "day"), fire_start)
        .filterBounds(aoi)
        .select(NDVI_BANDS)
        .sort("system:time_start", False)
    )
    # Use mosaic of sorted collection — gives most recent valid pixel per location.
    # If empty, .mosaic() returns an image with all masked pixels, and unmask(0) fills.
    # MODIS NDVI has scale factor 0.0001
    result: ee.Image = collection.mosaic().multiply(0.0001).unmask(0).toFloat()
    return result


def get_smoke_aerosol_index(aoi: ee.Geometry, fire_start: ee.Date, fire_end: ee.Date) -> ee.Image:
    """Get TROPOMI UV Aerosol Index (smoke detection above clouds).

    Positive values indicate UV-absorbing aerosols (smoke, dust).
    Values > 1.0 suggest significant smoke. Works through clouds.
    """
    collection = (
        ee.ImageCollection(TROPOMI_AAI_DATASET)
        .filterDate(fire_start, fire_end)
        .filterBounds(aoi)
        .select("absorbing_aerosol_index")
    )
    # Use max AAI across the fire period — smoke persists
    result: ee.Image = collection.max().unmask(0).rename("smoke_aerosol_index").toFloat()
    return result


def get_daily_lst(aoi: ee.Geometry, day_start: ee.Date, day_end: ee.Date) -> ee.Image:
    """Get MODIS daily land surface temperature (day and night)."""
    collection = ee.ImageCollection(LST_DATASET).filterDate(day_start, day_end).filterBounds(aoi)
    lst = ee.Image(
        ee.Algorithms.If(
            collection.size().gt(0),
            collection.first(),
            ee.Image([0, 0]).rename(["LST_Day_1km", "LST_Night_1km"]),
        )
    )
    lst_day = lst.select("LST_Day_1km").multiply(0.02).unmask(0).rename("lst_day_k")
    lst_night = lst.select("LST_Night_1km").multiply(0.02).unmask(0).rename("lst_night_k")
    result: ee.Image = lst_day.addBands(lst_night).toFloat()
    return result
