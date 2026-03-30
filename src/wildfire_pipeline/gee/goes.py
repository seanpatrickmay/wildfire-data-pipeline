"""GOES fire detection confidence mapping, hourly fusion, and smoke discrimination.

Converts GOES FDCC/FDCF Mask bands to calibrated fire confidence values
with DQF-based cloud masking and brightness-temperature-based smoke/cloud
discrimination using GOES ABI MCMIPC radiance data.

Key physics:
- BTD(3.9μm - 11.2μm) is strongly positive for fire (sub-pixel hotspots
  boost 3.9μm via Wien's law peak near fire temperatures), slightly positive
  for smoke (semi-transparent aerosol), and negative for cloud (opaque,
  cold tops absorb reflected solar at 3.9μm).
- BTD(12.3μm - 11.2μm) is strongly negative for thin cirrus, near-zero for
  smoke, and slightly negative for thick cloud (split-window effect).
"""

from __future__ import annotations

import ee

# Smoke/cloud discrimination thresholds (Kelvin)
# BTD(3.9μm - 11.2μm): fire >> 0, smoke ≈ 0 to +5, cloud < 0
SMOKE_BTD_FIRE_THRESHOLD_K = -2.0  # above this, likely smoke or fire (not cloud)
# BTD(12.3μm - 11.2μm): cirrus << 0, smoke ≈ 0, thick cloud < 0
SMOKE_BTD_SPLIT_THRESHOLD_K = -1.0  # above this, not cirrus


def goes_fire_confidence(image: ee.Image) -> ee.Image:
    """Convert GOES FDCC/FDCF Mask to fire confidence with DQF cloud masking.

    GOES Mask band encodes fire detection category:
        10 = Processed fire       -> 1.0 (highest confidence)
        11 = Saturated fire       -> 1.0 (sensor saturated, very hot)
        12 = Cloud contaminated   -> 0.8 (fire seen through partial cloud)
        13 = High probability     -> 0.5
        14 = Medium probability   -> 0.3
        15 = Low probability      -> 0.1 (highest false alarm rate)

    Codes 30-35 are temporally filtered (2+ detections in 12h window).
    These get BOOSTED confidence because temporal confirmation reduces
    false alarm probability. E.g., a "medium probability" detection that
    persists across multiple satellite passes (code 34) is more likely
    real than a single-pass detection (code 14).

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

    # Valid fire codes are 10-15 (single-pass) and 30-35 (temporally filtered).
    # Codes 16-29 are not assigned in GOES fire products — exclude them.
    is_fire = (mask.gte(10).And(mask.lte(15))).Or(mask.gte(30).And(mask.lte(35)))
    category = mask.mod(10)
    is_temporal = mask.gte(30)  # codes 30-35 = temporally filtered

    # Base confidence from single-pass detection
    base_conf = category.expression(
        "(c == 0) * 1.0 + (c == 1) * 1.0 + (c == 2) * 0.8 + "
        "(c == 3) * 0.5 + (c == 4) * 0.3 + (c == 5) * 0.1",
        {"c": category},
    )

    # Boosted confidence for temporally confirmed detections
    temporal_conf = category.expression(
        "(c == 0) * 1.0 + (c == 1) * 1.0 + (c == 2) * 0.9 + "
        "(c == 3) * 0.7 + (c == 4) * 0.5 + (c == 5) * 0.25",
        {"c": category},
    )

    # Start with base confidence, then upgrade temporally filtered pixels
    confidence = (
        base_conf.where(is_temporal, temporal_conf)
        .updateMask(is_fire)
        .rename("fire_confidence")
    )

    cloud_flag = dqf.eq(2).rename("is_cloud")
    valid_flag = dqf.lte(1).rename("is_valid")

    frp = image.select("Power").updateMask(is_fire.And(dqf.eq(0))).rename("frp_mw")

    # FDCC Area and Temp bands are stored as raw int16 in GEE.
    # Apply scale/offset from GEE catalog to convert to physical units.
    # Area: physical_m2 = raw * 60.98 + 4000, then convert to km2
    # Temp: physical_K = raw * 0.0549367 + 400
    raw_area = image.select("Area").updateMask(is_fire.And(dqf.eq(0)))
    fire_area = raw_area.multiply(60.98).add(4000).divide(1e6).rename("fire_area_km2")

    raw_temp = image.select("Temp").updateMask(is_fire.And(dqf.eq(0)))
    fire_temp = raw_temp.multiply(0.0549367).add(400).rename("fire_temp_k")

    result: ee.Image = (
        confidence.addBands(cloud_flag)
        .addBands(valid_flag)
        .addBands(frp)
        .addBands(fire_area)
        .addBands(fire_temp)
        .copyProperties(image, ["system:time_start"])
    )
    return result


def get_hourly_goes(
    aoi: ee.Geometry,
    hour_start: ee.Date,
    hour_end: ee.Date,
    fire_year: int = 2019,
    smoke_discrimination: bool = False,
) -> ee.Image:
    """Get max GOES fire confidence for one hour with cloud masking.

    Combines GOES-East (16) and GOES-West (17 or 18 depending on year).
    Prefers CONUS (FDCC, 5-min cadence) with fallback to Full Disk (FDCF, 10-min).

    When smoke_discrimination=True, also fetches GOES ABI brightness
    temperatures to classify DQF==2 pixels as cloud vs smoke using
    BTD(3.9μm - 11.2μm) and BTD(12.3μm - 11.2μm).

    Returns a 4-band image: confidence, frp, obs_valid, is_cloud.
    When smoke_discrimination=True, returns 6 bands:
        confidence, frp, obs_valid, is_cloud, is_smoke, btd_fire_smoke.
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

    # Guard: if no GOES images exist for this hour (coverage gap),
    # return a zero-filled image instead of crashing on .unmask()
    if smoke_discrimination:
        empty_img = (
            ee.Image(0)
            .rename("confidence")
            .addBands(ee.Image(0).rename("frp"))
            .addBands(ee.Image(0).rename("fire_area"))
            .addBands(ee.Image(0).rename("fire_temp"))
            .addBands(ee.Image(0).rename("obs_valid"))
            .addBands(ee.Image(0).rename("is_cloud"))
            .addBands(ee.Image(0).rename("is_smoke"))
            .addBands(ee.Image(0).rename("btd_fire_smoke"))
            .addBands(ee.Image(0).rename("blue_swir_smoke_ratio"))
            .toFloat()
        )
    else:
        empty_img = (
            ee.Image(0)
            .rename("confidence")
            .addBands(ee.Image(0).rename("frp"))
            .addBands(ee.Image(0).rename("fire_area"))
            .addBands(ee.Image(0).rename("fire_temp"))
            .addBands(ee.Image(0).rename("obs_valid"))
            .addBands(ee.Image(0).rename("is_cloud"))
            .toFloat()
        )

    fire_img = ee.Image(
        ee.Algorithms.If(
            all_goes.size().gt(0),
            _goes_reduce(all_goes),
            empty_img,
        )
    )

    if not smoke_discrimination:
        return fire_img

    # --- Smoke/cloud discrimination using ABI brightness temperatures ---
    smoke_img = _get_smoke_discrimination(aoi, hour_start, hour_end, fire_year, fire_img)
    return smoke_img


def _goes_reduce(all_goes: ee.ImageCollection) -> ee.Image:
    """Reduce a non-empty GOES collection to a single image."""
    conf = all_goes.select("fire_confidence").max().unmask(0)
    frp = all_goes.select("frp_mw").max().unmask(0)
    fire_area = all_goes.select("fire_area_km2").max().unmask(0)
    fire_temp = all_goes.select("fire_temp_k").max().unmask(0)
    any_cloud = all_goes.select("is_cloud").max().unmask(0)
    any_valid = all_goes.select("is_valid").max().unmask(0)

    is_cloud_not_fire = any_cloud.And(conf.lte(0))
    obs_valid = any_valid.And(is_cloud_not_fire.Not())

    result: ee.Image = (
        conf.rename("confidence")
        .addBands(frp.rename("frp"))
        .addBands(fire_area.rename("fire_area"))
        .addBands(fire_temp.rename("fire_temp"))
        .addBands(obs_valid.rename("obs_valid"))
        .addBands(is_cloud_not_fire.rename("is_cloud"))
    )
    return result


# ---------------------------------------------------------------------------
# Smoke/cloud discrimination via GOES ABI brightness temperatures
# ---------------------------------------------------------------------------


def _select_goes_abi_collection(fire_year: int) -> tuple[str, str, str, str]:
    """Return (east_conus, east_full, west_conus, west_full) ABI collection IDs."""
    west_conus = "NOAA/GOES/17/MCMIPC" if fire_year < 2023 else "NOAA/GOES/18/MCMIPC"
    west_full = "NOAA/GOES/17/MCMIPF" if fire_year < 2023 else "NOAA/GOES/18/MCMIPF"
    return "NOAA/GOES/16/MCMIPC", "NOAA/GOES/16/MCMIPF", west_conus, west_full


def _get_smoke_discrimination(
    aoi: ee.Geometry,
    hour_start: ee.Date,
    hour_end: ee.Date,
    fire_year: int,
    fire_img: ee.Image,
) -> ee.Image:
    """Classify DQF==2 cloud pixels as smoke vs true cloud using ABI BTDs.

    Fetches GOES ABI MCMIPC brightness temperatures for channels 7 (3.9μm),
    14 (11.2μm), and 15 (12.3μm), then applies threshold-based classification.

    Physics:
    - BTD(3.9 - 11.2) > -2K: likely smoke or fire (not opaque cloud).
      Smoke aerosols are semi-transparent in thermal IR; the warm surface
      contributes to 3.9μm radiance. Clouds are opaque and cold.
    - BTD(12.3 - 11.2) > -1K: not thin cirrus (which has strong split-window
      absorption at 12.3μm due to ice crystal size effects).

    Returns fire_img with 2 extra bands: is_smoke, btd_fire_smoke.
    """
    east_c, east_f, west_c, west_f = _select_goes_abi_collection(fire_year)
    abi_bands = ["CMI_C01", "CMI_C06", "CMI_C07", "CMI_C14", "CMI_C15"]

    # Try CONUS first, fallback to Full Disk (same pattern as FDCC/FDCF)
    east = ee.ImageCollection(east_c).filterDate(hour_start, hour_end).filterBounds(aoi)
    east = ee.ImageCollection(
        ee.Algorithms.If(
            east.size().gt(0),
            east,
            ee.ImageCollection(east_f).filterDate(hour_start, hour_end).filterBounds(aoi),
        )
    )
    west = ee.ImageCollection(west_c).filterDate(hour_start, hour_end).filterBounds(aoi)
    west = ee.ImageCollection(
        ee.Algorithms.If(
            west.size().gt(0),
            west,
            ee.ImageCollection(west_f).filterDate(hour_start, hour_end).filterBounds(aoi),
        )
    )

    all_abi = east.merge(west).select(abi_bands)

    # If no ABI data available, return fire_img with zero smoke bands
    zero_smoke = (
        fire_img.addBands(ee.Image(0).rename("is_smoke").toFloat())
        .addBands(ee.Image(0).rename("btd_fire_smoke").toFloat())
        .addBands(ee.Image(0).rename("blue_swir_smoke_ratio").toFloat())
    )

    result: ee.Image = ee.Image(
        ee.Algorithms.If(
            all_abi.size().gt(0),
            _apply_smoke_classification(all_abi, fire_img),
            zero_smoke,
        )
    )
    return result


def _apply_smoke_classification(
    abi_collection: ee.ImageCollection, fire_img: ee.Image
) -> ee.Image:
    """Apply BTD-based smoke classification to an ABI collection.

    Uses median composite of ABI brightness temperatures within the hour
    to reduce noise, then classifies cloud-flagged pixels.
    """
    abi_median = abi_collection.median()

    # CRITICAL: GEE stores MCMIPC CMI bands as raw int16 values.
    # Scale/offset must be applied to convert to physical units.
    # IR bands (7-16): physical_BT = raw * scale + offset (Kelvin)
    # VIS bands (1-6): physical_refl = raw * scale + offset (reflectance factor)
    _IR_SCALE = 0.039316241
    _IR_OFFSET = 173.15
    _C07_SCALE = 0.01384667  # Band 7 has a different scale
    _C07_OFFSET = 173.15
    _VIS_SCALE = 0.0002442
    _VIS_OFFSET = 0.0

    raw_c07 = abi_median.select("CMI_C07")
    raw_c14 = abi_median.select("CMI_C14")
    raw_c15 = abi_median.select("CMI_C15")

    bt_swir = raw_c07.multiply(_C07_SCALE).add(_C07_OFFSET)   # 3.9μm BT (K)
    bt_tir1 = raw_c14.multiply(_IR_SCALE).add(_IR_OFFSET)     # 11.2μm BT (K)
    bt_tir2 = raw_c15.multiply(_IR_SCALE).add(_IR_OFFSET)     # 12.3μm BT (K)

    # Track where ABI data is actually valid (not masked/missing).
    # Without this guard, unmask(0) would give BTD=0 for missing pixels,
    # which exceeds the -2K threshold and could falsely classify missing
    # data as smoke.
    abi_valid = raw_c07.mask().And(raw_c14.mask()).And(raw_c15.mask())

    # BTD(3.9 - 11.2): fire >> 0, smoke ≈ 0 to +5, cloud < -2
    btd_fire = bt_swir.subtract(bt_tir1).unmask(0).rename("btd_fire_smoke")

    # BTD(12.3 - 11.2): cirrus << -1, smoke ≈ 0, thick cloud slightly negative
    btd_split = bt_tir2.subtract(bt_tir1).unmask(0)

    # Classify: pixel is likely smoke (not true cloud) if:
    # 1. GOES fire product flagged it as cloud (is_cloud == 1)
    # 2. ABI brightness temperature data is valid (not missing)
    # 3. BTD(3.9-11.2) > -2K (not cold enough to be opaque cloud)
    # 4. BTD(12.3-11.2) > -1K (not cirrus ice)
    is_cloud = fire_img.select("is_cloud")
    is_smoke = (
        is_cloud.eq(1)
        .And(abi_valid)
        .And(btd_fire.gt(SMOKE_BTD_FIRE_THRESHOLD_K))
        .And(btd_split.gt(SMOKE_BTD_SPLIT_THRESHOLD_K))
        .rename("is_smoke")
        .toFloat()
    )

    # Blue/SWIR reflectance ratio for daytime smoke detection.
    # CMI_C01 (0.47μm) is scattered by fine smoke particles (Mie regime).
    # CMI_C06 (2.2μm) passes through smoke (particles too small for SWIR).
    # Smoke: ratio >> 1. Cloud: ratio ≈ 1. Only valid during daytime.
    # Apply VIS scale factor to convert from raw int16 to reflectance.
    blue = abi_median.select("CMI_C01").multiply(_VIS_SCALE).add(_VIS_OFFSET).unmask(0)
    swir_refl = abi_median.select("CMI_C06").multiply(_VIS_SCALE).add(_VIS_OFFSET).unmask(0)

    # Guard: only compute where blue reflectance is measurable (daytime).
    # At night, CMI_C01 ≈ 0 since there's no reflected sunlight.
    is_daytime = blue.gt(0.01)
    safe_swir = swir_refl.max(0.001)  # avoid division by zero
    blue_swir_ratio = (
        blue.divide(safe_swir)
        .min(20.0)  # clamp extreme values
        .multiply(is_daytime)  # zero at night
        .rename("blue_swir_smoke_ratio")
        .toFloat()
    )

    # Reclassify: pixels identified as smoke are no longer "cloud"
    # They should NOT be excluded from training — the fire may well be burning
    # underneath semi-transparent smoke.
    updated_cloud = is_cloud.And(is_smoke.Not()).rename("is_cloud").toFloat()

    # Update obs_valid: smoke pixels should be valid (fire visible through smoke)
    updated_valid = fire_img.select("obs_valid").Or(is_smoke).rename("obs_valid").toFloat()

    result: ee.Image = (
        fire_img.select("confidence")
        .addBands(fire_img.select("frp"))
        .addBands(fire_img.select("fire_area"))
        .addBands(fire_img.select("fire_temp"))
        .addBands(updated_valid)
        .addBands(updated_cloud)
        .addBands(is_smoke)
        .addBands(btd_fire.toFloat())
        .addBands(blue_swir_ratio)
    )
    return result
