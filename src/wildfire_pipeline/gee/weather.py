"""Weather feature extraction from Google Earth Engine.

Sources:
- RTMA (NOAA/NWS/RTMA): 2.5km hourly — wind u/v, gust, temp, dewpoint
- GRIDMET (IDAHO_EPSCOR/GRIDMET): 4km daily — ERC, BI, fuel moisture, VPD, wind, humidity
- ERA5-Land (ECMWF/ERA5_LAND/HOURLY): 11km hourly — soil moisture, precipitation
- GPM IMERG (NASA/GPM_L3/IMERG_V07): 10km, 30-min — precipitation rate
"""

from __future__ import annotations

import ee

RTMA_DATASET = "NOAA/NWS/RTMA"
RTMA_BANDS = ["UGRD", "VGRD", "GUST", "TMP", "DPT"]

GRIDMET_DATASET = "IDAHO_EPSCOR/GRIDMET"
GRIDMET_FIRE_BANDS = ["erc", "bi", "fm100", "fm1000", "vpd", "rmin", "rmax", "vs", "th"]

ERA5_LAND_DATASET = "ECMWF/ERA5_LAND/HOURLY"
GRIDMET_DROUGHT_DATASET = "GRIDMET/DROUGHT"
GPM_IMERG_DATASET = "NASA/GPM_L3/IMERG_V07"


def get_hourly_rtma(aoi: ee.Geometry, hour_start: ee.Date, hour_end: ee.Date) -> ee.Image:
    """Get RTMA weather for one hour: wind u/v, gust, temp, dewpoint."""
    rtma = (
        ee.ImageCollection(RTMA_DATASET)
        .filterDate(hour_start, hour_end)
        .filterBounds(aoi)
        .select(RTMA_BANDS)
    )
    result: ee.Image = rtma.mean().unmask(0).toFloat()
    return result


def get_daily_gridmet(aoi: ee.Geometry, day_start: ee.Date, day_end: ee.Date) -> ee.Image:
    """Get GRIDMET daily fire weather: ERC, BI, fuel moisture, VPD, wind, humidity."""
    gridmet = (
        ee.ImageCollection(GRIDMET_DATASET)
        .filterDate(day_start, day_end)
        .filterBounds(aoi)
        .select(GRIDMET_FIRE_BANDS)
    )
    result: ee.Image = gridmet.mean().unmask(0).toFloat()
    return result


def get_hourly_soil_moisture(aoi: ee.Geometry, hour_start: ee.Date, hour_end: ee.Date) -> ee.Image:
    """Get ERA5-Land soil moisture for one hour."""
    era5 = (
        ee.ImageCollection(ERA5_LAND_DATASET)
        .filterDate(hour_start, hour_end)
        .filterBounds(aoi)
        .select("volumetric_soil_water_layer_1")
    )
    result: ee.Image = era5.mean().unmask(0).rename("soil_moisture").toFloat()
    return result


def get_hourly_precipitation(aoi: ee.Geometry, hour_start: ee.Date, hour_end: ee.Date) -> ee.Image:
    """Get ERA5-Land hourly total precipitation."""
    era5 = (
        ee.ImageCollection(ERA5_LAND_DATASET)
        .filterDate(hour_start, hour_end)
        .filterBounds(aoi)
        .select("total_precipitation_hourly")
    )
    result: ee.Image = era5.sum().unmask(0).rename("precipitation_m").toFloat()
    return result


def get_hourly_gpm_precipitation(
    aoi: ee.Geometry, hour_start: ee.Date, hour_end: ee.Date
) -> ee.Image:
    """Get GPM IMERG precipitation rate (mm/hr), 30-min cadence averaged to hourly."""
    gpm = (
        ee.ImageCollection(GPM_IMERG_DATASET)
        .filterDate(hour_start, hour_end)
        .filterBounds(aoi)
        .select("precipitation")
    )
    result: ee.Image = gpm.mean().unmask(0).rename("gpm_precipitation_mmhr").toFloat()
    return result


def get_drought_indices(aoi: ee.Geometry, fire_start: ee.Date) -> ee.Image:
    """Get most recent drought indices before/during fire."""
    collection = (
        ee.ImageCollection(GRIDMET_DROUGHT_DATASET)
        .filterDate(fire_start.advance(-60, "day"), fire_start)
        .filterBounds(aoi)
        .sort("system:time_start", False)
    )
    # Use mosaic of sorted collection — gives most recent valid pixel per location.
    # Select specific bands and rename to avoid type conflicts with string-type bands.
    drought = collection.select(["pdsi", "eddi14d", "eddi30d"]).mosaic().unmask(0)
    pdsi = drought.select("pdsi").rename("pdsi")
    eddi14 = drought.select("eddi14d").rename("eddi_14d")
    eddi30 = drought.select("eddi30d").rename("eddi_30d")
    result: ee.Image = pdsi.addBands(eddi14).addBands(eddi30).toFloat()
    return result
