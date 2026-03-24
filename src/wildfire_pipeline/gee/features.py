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
    result: ee.Image = ndvi.multiply(0.0001).unmask(0).toFloat()
    return result


def get_daily_lst(aoi: ee.Geometry, day_start: ee.Date, day_end: ee.Date) -> ee.Image:
    """Get MODIS daily land surface temperature (day and night)."""
    lst = ee.ImageCollection(LST_DATASET).filterDate(day_start, day_end).filterBounds(aoi).first()
    lst_day = lst.select("LST_Day_1km").multiply(0.02).unmask(0).rename("lst_day_k")
    lst_night = lst.select("LST_Night_1km").multiply(0.02).unmask(0).rename("lst_night_k")
    result: ee.Image = lst_day.addBands(lst_night).toFloat()
    return result
