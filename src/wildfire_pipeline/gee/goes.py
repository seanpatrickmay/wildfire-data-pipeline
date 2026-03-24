"""GOES fire detection confidence mapping and hourly fusion.

Extracted from scripts/download_fire_data.py into a reusable module.
Converts GOES FDCC/FDCF Mask bands to calibrated fire confidence values
with DQF-based cloud masking.
"""

from __future__ import annotations

import ee


def goes_fire_confidence(image: ee.Image) -> ee.Image:
    """Convert GOES FDCC/FDCF Mask to fire confidence with DQF cloud masking.

    GOES Mask band encodes fire detection category:
        10/30 = Processed fire       -> 1.0 (highest confidence)
        11/31 = Saturated fire       -> 1.0 (sensor saturated, very hot)
        12/32 = Cloud contaminated   -> 0.8 (fire seen through partial cloud)
        13/33 = High probability     -> 0.5
        14/34 = Medium probability   -> 0.3
        15/35 = Low probability      -> 0.1 (highest false alarm rate)

    Codes 30-35 are temporally filtered (2+ detections in 12h window).
    The .mod(10) strips the temporal filter offset.

    DQF quality flags:
        0 = good fire pixel
        1 = good fire-free land
        2 = CLOUD (marked as unknown, not negative)
        3 = invalid (sunglint, bad surface, off-earth, missing)
        4 = bad input data
        5 = algorithm failure
    """
    mask = image.select("Mask")
    dqf = image.select("DQF")

    is_fire = mask.gte(10).And(mask.lte(35))
    category = mask.mod(10)
    confidence = (
        category.expression(
            "(c == 0) * 1.0 + (c == 1) * 1.0 + (c == 2) * 0.8 + "
            "(c == 3) * 0.5 + (c == 4) * 0.3 + (c == 5) * 0.1",
            {"c": category},
        )
        .updateMask(is_fire)
        .rename("fire_confidence")
    )

    cloud_flag = dqf.eq(2).rename("is_cloud")
    valid_flag = dqf.lte(1).rename("is_valid")

    frp = image.select("Power").updateMask(is_fire.And(dqf.eq(0))).rename("frp_mw")

    result: ee.Image = (
        confidence.addBands(cloud_flag)
        .addBands(valid_flag)
        .addBands(frp)
        .copyProperties(image, ["system:time_start"])
    )
    return result


def get_hourly_goes(
    aoi: ee.Geometry, hour_start: ee.Date, hour_end: ee.Date, fire_year: int = 2019
) -> ee.Image:
    """Get max GOES fire confidence for one hour with cloud masking.

    Combines GOES-East (16) and GOES-West (17 or 18 depending on year).
    Prefers CONUS (FDCC, 5-min cadence) with fallback to Full Disk (FDCF, 10-min).

    Returns a 4-band image: confidence, frp, obs_valid, is_cloud.
    """
    # GOES-17 (West) decommissioned Jan 2023, replaced by GOES-18
    goes_west_conus = "NOAA/GOES/17/FDCC" if fire_year < 2023 else "NOAA/GOES/18/FDCC"
    goes_west_full = "NOAA/GOES/17/FDCF" if fire_year < 2023 else "NOAA/GOES/18/FDCF"

    goes16 = (
        ee.ImageCollection("NOAA/GOES/16/FDCC").filterDate(hour_start, hour_end).filterBounds(aoi)
    )
    goes17 = ee.ImageCollection(goes_west_conus).filterDate(hour_start, hour_end).filterBounds(aoi)

    # Fallback to full disk if CONUS is empty
    goes16 = ee.ImageCollection(
        ee.Algorithms.If(
            goes16.size().gt(0),
            goes16,
            ee.ImageCollection("NOAA/GOES/16/FDCF")
            .filterDate(hour_start, hour_end)
            .filterBounds(aoi),
        )
    )
    goes17 = ee.ImageCollection(
        ee.Algorithms.If(
            goes17.size().gt(0),
            goes17,
            ee.ImageCollection(goes_west_full).filterDate(hour_start, hour_end).filterBounds(aoi),
        )
    )

    all_goes = goes16.merge(goes17).map(goes_fire_confidence)

    conf = all_goes.select("fire_confidence").max().unmask(0)
    frp = all_goes.select("frp_mw").max().unmask(0)
    any_cloud = all_goes.select("is_cloud").max().unmask(0)
    any_valid = all_goes.select("is_valid").max().unmask(0)

    # Cloud = cloudy AND no fire detected
    is_cloud_not_fire = any_cloud.And(conf.lte(0))
    obs_valid = any_valid.And(is_cloud_not_fire.Not())

    result: ee.Image = (
        conf.rename("confidence")
        .addBands(frp.rename("frp"))
        .addBands(obs_valid.rename("obs_valid"))
        .addBands(is_cloud_not_fire.rename("is_cloud"))
    )
    return result
