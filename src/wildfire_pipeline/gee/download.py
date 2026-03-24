"""Download fire detection data from Google Earth Engine."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import ee
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

from wildfire_pipeline.config import FiresConfig
from wildfire_pipeline.gee.goes import get_hourly_goes
from wildfire_pipeline.gee.retry import safe_get_info, safe_sample_rectangle
from wildfire_pipeline.logging import get_logger
from wildfire_pipeline.processing.io import save_fire_data

logger = get_logger()


def _ensure_fires_config(config: FiresConfig | dict) -> FiresConfig:
    """Accept either a FiresConfig or a dict, returning a validated model."""
    if isinstance(config, FiresConfig):
        return config
    return FiresConfig.model_validate(config)


# sampleRectangle hard limit: 262,144 pixels (512x512)
SAMPLE_RECT_LIMIT = 262_144


def compute_grid_shape(aoi: list[float] | tuple[float, ...], scale: int) -> tuple[int, int]:
    """Compute the expected grid shape from an AOI bounding box and scale.

    Args:
        aoi: [xmin, ymin, xmax, ymax] in degrees
        scale: export scale in meters

    Returns:
        (n_rows, n_cols) grid dimensions

    Raises:
        ValueError: if the resulting grid exceeds sampleRectangle pixel limit
    """
    lat_mid = (aoi[1] + aoi[3]) / 2.0
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(lat_mid))
    n_rows = max(1, round(abs(aoi[3] - aoi[1]) * meters_per_deg_lat / scale))
    n_cols = max(1, round(abs(aoi[2] - aoi[0]) * meters_per_deg_lon / scale))

    total_pixels = n_rows * n_cols
    if total_pixels > SAMPLE_RECT_LIMIT:
        raise ValueError(
            f"AOI produces {total_pixels:,} pixels ({n_rows}x{n_cols}) which exceeds "
            f"sampleRectangle limit of {SAMPLE_RECT_LIMIT:,}. "
            f"Increase export_scale_m or reduce AOI."
        )

    return n_rows, n_cols


class TooManyFailuresError(Exception):
    """Raised when the hourly download failure rate exceeds the threshold."""

    def __init__(self, failed: int, total: int, rate: float, hours: list[int]) -> None:
        self.failed = failed
        self.total = total
        self.rate = rate
        self.hours = hours
        super().__init__(
            f"Too many failures: {failed}/{total} hours failed ({rate:.1%}). Failed hours: {hours}"
        )


def download_fire_stack(
    fire_name: str,
    config: FiresConfig | dict,
    output_dir: Path,
    fmt: str = "npz",
    failure_threshold: float = 0.10,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Download hourly fire data for one fire event.

    Args:
        fire_name: Name of the fire event (must exist in config.fires).
        config: FiresConfig model or raw dict (auto-validated).
        output_dir: Directory to save output files.
        fmt: Output format ("npz", "zarr", or "json").
        failure_threshold: Max fraction of failed hours before aborting.

    Returns:
        Tuple of (arrays dict, metadata dict).

    Raises:
        TooManyFailuresError: If the hourly failure rate exceeds failure_threshold.
        KeyError: If fire_name is not found in config.fires.
    """
    cfg = _ensure_fires_config(config)
    fire = cfg.fires[fire_name]
    pipeline = cfg.pipeline_config

    aoi_coords = list(fire.aoi)
    aoi = ee.Geometry.Rectangle(aoi_coords)
    start = ee.Date(fire.start_utc.isoformat())
    n_hours = fire.n_hours
    scale = pipeline.export_scale_m

    logger.info("downloading_fire", fire_name=fire_name, n_hours=n_hours)

    fire_dir = output_dir / fire_name
    fire_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute expected grid shape (also validates pixel limit)
    n_rows, n_cols = compute_grid_shape(aoi_coords, scale)
    expected_shape = (n_rows, n_cols)
    logger.debug("expected_grid_shape", rows=n_rows, cols=n_cols)

    # Download each hour
    all_conf: list[np.ndarray] = []
    all_valid: list[np.ndarray] = []
    all_cloud: list[np.ndarray] = []
    all_frp: list[np.ndarray] = []
    failed_hours: list[int] = []

    for h in range(n_hours):
        hour_start = start.advance(h, "hour")
        hour_end = start.advance(h + 1, "hour")

        hourly = get_hourly_goes(aoi, hour_start, hour_end, fire_year=fire.year)
        hourly = hourly.reproject(crs=pipeline.export_crs, scale=pipeline.export_scale_m)

        try:
            sample = safe_sample_rectangle(hourly, aoi)
            info = safe_get_info(sample)  # ONE call gets all bands
            props = info["properties"]
            conf_arr = np.array(props["confidence"], dtype=np.float32)
            valid_arr = np.array(props["obs_valid"], dtype=np.float32)
            cloud_arr = np.array(props["is_cloud"], dtype=np.float32)
            frp_arr = np.array(props["frp"], dtype=np.float32)

            all_conf.append(conf_arr)
            all_valid.append(valid_arr)
            all_cloud.append(cloud_arr)
            all_frp.append(frp_arr)

        except Exception as e:
            # Re-raise programming errors immediately; only handle GEE/network failures
            if isinstance(e, (TypeError, AttributeError, KeyError, ImportError)):
                raise
            logger.error("hour_download_failed", hour=h, error=str(e))
            failed_hours.append(h)
            shape = all_conf[-1].shape if all_conf else expected_shape
            all_conf.append(np.zeros(shape, dtype=np.float32))
            all_valid.append(np.zeros(shape, dtype=np.float32))
            all_cloud.append(np.zeros(shape, dtype=np.float32))
            all_frp.append(np.zeros(shape, dtype=np.float32))
            continue

        step = max(1, min(24, n_hours // 10))
        if (h + 1) % step == 0 or h + 1 == n_hours:
            pct = 100 * (h + 1) / n_hours
            logger.info("progress", hour=h + 1, total=n_hours, pct=f"{pct:.0f}%")

    failure_rate = len(failed_hours) / n_hours if n_hours > 0 else 0.0
    if failure_rate > failure_threshold:
        raise TooManyFailuresError(
            failed=len(failed_hours),
            total=n_hours,
            rate=failure_rate,
            hours=failed_hours,
        )
    elif failed_hours:
        logger.warning(
            "some_hours_failed",
            failed=len(failed_hours),
            total=n_hours,
            rate=failure_rate,
            hours=failed_hours,
        )

    conf_stack = np.stack(all_conf, axis=0)
    valid_stack = np.stack(all_valid, axis=0)
    cloud_stack = np.stack(all_cloud, axis=0)
    frp_stack = np.stack(all_frp, axis=0)

    arrays = {
        "data": conf_stack,
        "observation_valid": valid_stack,
        "cloud_mask": cloud_stack,
        "frp": frp_stack,
    }
    metadata: dict[str, Any] = {
        "fire_name": fire_name,
        "year": fire.year,
        "start_utc": fire.start_utc.isoformat(),
        "n_hours": n_hours,
        "grid_shape": list(conf_stack.shape[1:]),
        "aoi": aoi_coords,
        "pipeline": "wildfire-data-pipeline",
        "cloud_masking": True,
        "multi_source_fusion": True,
        "failed_hours": failed_hours,
        "failure_rate": failure_rate,
    }

    out_path = fire_dir / f"{fire_name}_{fire.year}_FusedConf"
    actual_path = save_fire_data(out_path, arrays, metadata, fmt=fmt)
    logger.info("saved", path=str(actual_path), shape=conf_stack.shape)

    return arrays, metadata


def download_features(
    fire_name: str,
    config: FiresConfig | dict,
    output_dir: Path,
    fmt: str = "npz",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Download weather/terrain/vegetation features for one fire event.

    Downloads three categories at different temporal cadences, all resampled
    to the GOES 2km grid:
    - Hourly: RTMA wind/temp/dewpoint + ERA5 soil moisture
    - Daily: GRIDMET fire weather + MODIS LST (repeated 24x per day)
    - Static: terrain/fuel/WUI features (broadcast to all timesteps)
    - Slow-varying: NDVI/EVI + drought indices

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
    static_img = static_img.reproject(crs=pipeline.export_crs, scale=pipeline.export_scale_m)
    static_sample = safe_sample_rectangle(static_img, aoi)
    static_info = safe_get_info(static_sample)  # ONE call gets all bands
    static_bands: dict[str, np.ndarray] = {}
    for band, values in static_info["properties"].items():
        static_bands[band] = np.array(values, dtype=np.float32)

    # --- Slow-varying features (pre-fire NDVI, drought) ---
    from wildfire_pipeline.gee.features import get_pre_fire_ndvi
    from wildfire_pipeline.gee.weather import get_drought_indices

    slow_bands: dict[str, np.ndarray] = {}
    # Sample NDVI and drought separately — they may have different CRS/projections
    # and ee.Algorithms.If fallbacks lose CRS info
    for label, img_fn in [
        ("ndvi", lambda: get_pre_fire_ndvi(aoi, start)),
        ("drought", lambda: get_drought_indices(aoi, start)),
    ]:
        try:
            img = img_fn().reproject(crs=pipeline.export_crs, scale=pipeline.export_scale_m)
            sample = safe_sample_rectangle(img, aoi)
            info = safe_get_info(sample)
            for band, values in info["properties"].items():
                slow_bands[band] = np.array(values, dtype=np.float32)
        except Exception as e:
            if isinstance(e, (TypeError, AttributeError, KeyError, ImportError)):
                raise
            logger.warning("slow_feature_failed", feature=label, error=str(e))
            # Fill with zeros at expected shape
            shape = (n_rows, n_cols)
            if label == "ndvi":
                slow_bands["NDVI"] = np.zeros(shape, dtype=np.float32)
                slow_bands["EVI"] = np.zeros(shape, dtype=np.float32)
            else:
                for k in ["pdsi", "eddi_14d", "eddi_30d"]:
                    slow_bands[k] = np.zeros(shape, dtype=np.float32)

    # --- Slow-varying: TROPOMI smoke aerosol index ---
    from wildfire_pipeline.gee.features import get_smoke_aerosol_index

    try:
        fire_end = start.advance(n_hours, "hour")
        smoke_img = get_smoke_aerosol_index(aoi, start, fire_end).reproject(
            crs=pipeline.export_crs, scale=pipeline.export_scale_m
        )
        smoke_sample = safe_sample_rectangle(smoke_img, aoi)
        smoke_info = safe_get_info(smoke_sample)
        for band, values in smoke_info["properties"].items():
            slow_bands[band] = np.array(values, dtype=np.float32)
    except Exception as e:
        if isinstance(e, (TypeError, AttributeError, KeyError, ImportError)):
            raise
        logger.warning("slow_feature_failed", feature="smoke_aerosol", error=str(e))
        slow_bands["smoke_aerosol_index"] = np.zeros((n_rows, n_cols), dtype=np.float32)

    # --- Hourly features (RTMA + soil moisture + precipitation) ---
    from wildfire_pipeline.gee.weather import (
        get_hourly_gpm_precipitation,
        get_hourly_precipitation,
        get_hourly_rtma,
        get_hourly_soil_moisture,
    )

    hourly_stacks: dict[str, list[np.ndarray]] = {}
    for h in range(n_hours):
        hour_start = start.advance(h, "hour")
        hour_end = start.advance(h + 1, "hour")

        try:
            soil = get_hourly_soil_moisture(aoi, hour_start, hour_end)
            precip_era5 = get_hourly_precipitation(aoi, hour_start, hour_end)
            precip_gpm = get_hourly_gpm_precipitation(aoi, hour_start, hour_end)
            if pipeline.rtma_wind:
                rtma = get_hourly_rtma(aoi, hour_start, hour_end)
                combined = (
                    rtma.addBands(soil)
                    .addBands(precip_era5)
                    .addBands(precip_gpm)
                    .reproject(crs=pipeline.export_crs, scale=pipeline.export_scale_m)
                )
            else:
                combined = (
                    soil.addBands(precip_era5)
                    .addBands(precip_gpm)
                    .reproject(crs=pipeline.export_crs, scale=pipeline.export_scale_m)
                )

            info = safe_get_info(safe_sample_rectangle(combined, aoi))  # ONE call
            props = info["properties"]
            for band_name, values in props.items():
                hourly_stacks.setdefault(band_name.lower(), []).append(
                    np.array(values, dtype=np.float32)
                )

        except Exception as e:
            # Re-raise programming errors immediately; only handle GEE/network failures
            if isinstance(e, (TypeError, AttributeError, KeyError, ImportError)):
                raise
            logger.warning("feature_hour_failed", hour=h, error=str(e))
            shape = (n_rows, n_cols)
            fallback_keys = (
                [
                    "ugrd",
                    "vgrd",
                    "gust",
                    "tmp",
                    "dpt",
                    "soil_moisture",
                    "precipitation_m",
                    "gpm_precipitation_mmhr",
                ]
                if pipeline.rtma_wind
                else ["soil_moisture", "precipitation_m", "gpm_precipitation_mmhr"]
            )
            for key in fallback_keys:
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
            lst = get_daily_lst(aoi, day_start, day_end)
            combined = gridmet.addBands(lst).reproject(
                crs=pipeline.export_crs, scale=pipeline.export_scale_m
            )
            info = safe_get_info(safe_sample_rectangle(combined, aoi))  # ONE call
            props = info["properties"]
            for band_name, values in props.items():
                daily_stacks.setdefault(band_name, []).append(np.array(values, dtype=np.float32))

        except Exception as e:
            # Re-raise programming errors immediately; only handle GEE/network failures
            if isinstance(e, (TypeError, AttributeError, KeyError, ImportError)):
                raise
            logger.warning("feature_day_failed", day=d, error=str(e))
            shape = (n_rows, n_cols)
            for key in [
                "erc",
                "bi",
                "fm100",
                "fm1000",
                "vpd",
                "rmin",
                "rmax",
                "vs",
                "th",
                "lst_day_k",
                "lst_night_k",
            ]:
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

    # --- Temporal encoding (computed locally, no GEE calls) ---
    from datetime import UTC

    start_dt = (
        fire.start_utc.replace(tzinfo=UTC) if fire.start_utc.tzinfo is None else fire.start_utc
    )

    # Hour of day (diurnal cycle) — sin/cos encoding
    hour_of_day = np.array([(start_dt.hour + h) % 24 for h in range(n_hours)], dtype=np.float32)
    arrays["temporal_hour_sin"] = np.broadcast_to(
        (np.sin(2 * np.pi * hour_of_day / 24.0))[:, np.newaxis, np.newaxis],
        (n_hours, n_rows, n_cols),
    ).copy()
    arrays["temporal_hour_cos"] = np.broadcast_to(
        (np.cos(2 * np.pi * hour_of_day / 24.0))[:, np.newaxis, np.newaxis],
        (n_hours, n_rows, n_cols),
    ).copy()

    # Day of year (seasonal cycle)
    day_of_year = np.array(
        [(start_dt.timetuple().tm_yday + h // 24) % 365 for h in range(n_hours)],
        dtype=np.float32,
    )
    arrays["temporal_doy_sin"] = np.broadcast_to(
        (np.sin(2 * np.pi * day_of_year / 365.25))[:, np.newaxis, np.newaxis],
        (n_hours, n_rows, n_cols),
    ).copy()
    arrays["temporal_doy_cos"] = np.broadcast_to(
        (np.cos(2 * np.pi * day_of_year / 365.25))[:, np.newaxis, np.newaxis],
        (n_hours, n_rows, n_cols),
    ).copy()

    # --- Normalization statistics for ML training ---
    from wildfire_pipeline.processing.quality import compute_normalization_stats

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
        "temporal_bands": [
            "temporal_hour_sin",
            "temporal_hour_cos",
            "temporal_doy_sin",
            "temporal_doy_cos",
        ],
        "normalization_stats": compute_normalization_stats(arrays),
    }

    out_path = fire_dir / f"{fire_name}_{fire.year}_Features"
    actual_path = save_fire_data(out_path, arrays, metadata, fmt=fmt)
    logger.info("features_saved", path=str(actual_path))

    return arrays, metadata
