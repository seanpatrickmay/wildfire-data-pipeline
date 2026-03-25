"""Feature channel registry — single source of truth for all pipeline output channels.

Documents each channel's units, range, source, temporal cadence, and normalization hint.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    """Specification for a single feature channel."""

    name: str
    unit: str
    source: str
    temporal: str  # "hourly", "daily", "static", "slow"
    range_min: float
    range_max: float
    dtype_hint: str  # "continuous", "binary", "categorical"
    normalization: str  # "zscore", "minmax", "none", "embedding"
    safe_as_input: bool = True  # False for targets, diagnostics, and loss weights
    requires_temporal_lag: bool = False  # True if must be used at t-1 to predict t


FEATURE_REGISTRY: dict[str, FeatureSpec] = {
    # --- Fire detection (hourly) ---
    "confidence": FeatureSpec(
        "confidence", "probability", "GOES FDCC/FDCF", "hourly", 0.0, 1.0, "continuous", "none"
    ),
    "frp": FeatureSpec("frp", "MW", "GOES ABI", "hourly", 0.0, 5000.0, "continuous", "zscore"),
    # --- RTMA weather (hourly) ---
    "ugrd": FeatureSpec(
        "ugrd", "m/s", "NOAA/NWS/RTMA", "hourly", -50.0, 50.0, "continuous", "zscore"
    ),
    "vgrd": FeatureSpec(
        "vgrd", "m/s", "NOAA/NWS/RTMA", "hourly", -50.0, 50.0, "continuous", "zscore"
    ),
    "gust": FeatureSpec(
        "gust", "m/s", "NOAA/NWS/RTMA", "hourly", 0.0, 80.0, "continuous", "zscore"
    ),
    "tmp": FeatureSpec("tmp", "K", "NOAA/NWS/RTMA", "hourly", 200.0, 330.0, "continuous", "zscore"),
    "dpt": FeatureSpec("dpt", "K", "NOAA/NWS/RTMA", "hourly", 200.0, 310.0, "continuous", "zscore"),
    "soil_moisture": FeatureSpec(
        "soil_moisture", "m3/m3", "ECMWF/ERA5_LAND", "hourly", 0.0, 0.6, "continuous", "zscore"
    ),
    "precipitation_m": FeatureSpec(
        "precipitation_m", "m", "ECMWF/ERA5_LAND", "hourly", 0.0, 0.1, "continuous", "zscore"
    ),
    "gpm_precipitation_mmhr": FeatureSpec(
        "gpm_precipitation_mmhr",
        "mm/hr",
        "NASA/GPM_L3/IMERG_V07",
        "hourly",
        0.0,
        100.0,
        "continuous",
        "zscore",
    ),
    # --- GRIDMET weather (daily) ---
    "erc": FeatureSpec(
        "erc", "BTU/ft2", "IDAHO_EPSCOR/GRIDMET", "daily", 0.0, 150.0, "continuous", "zscore"
    ),
    "bi": FeatureSpec(
        "bi", "index", "IDAHO_EPSCOR/GRIDMET", "daily", 0.0, 300.0, "continuous", "zscore"
    ),
    "fm100": FeatureSpec(
        "fm100", "%", "IDAHO_EPSCOR/GRIDMET", "daily", 0.0, 40.0, "continuous", "zscore"
    ),
    "fm1000": FeatureSpec(
        "fm1000", "%", "IDAHO_EPSCOR/GRIDMET", "daily", 0.0, 40.0, "continuous", "zscore"
    ),
    "vpd": FeatureSpec(
        "vpd", "kPa", "IDAHO_EPSCOR/GRIDMET", "daily", 0.0, 10.0, "continuous", "zscore"
    ),
    "rmin": FeatureSpec(
        "rmin", "%", "IDAHO_EPSCOR/GRIDMET", "daily", 0.0, 100.0, "continuous", "zscore"
    ),
    "rmax": FeatureSpec(
        "rmax", "%", "IDAHO_EPSCOR/GRIDMET", "daily", 0.0, 100.0, "continuous", "zscore"
    ),
    "vs": FeatureSpec(
        "vs", "m/s", "IDAHO_EPSCOR/GRIDMET", "daily", 0.0, 30.0, "continuous", "zscore"
    ),
    "th": FeatureSpec(
        "th", "degrees", "IDAHO_EPSCOR/GRIDMET", "daily", 0.0, 360.0, "continuous", "zscore"
    ),
    "lst_day_k": FeatureSpec(
        "lst_day_k", "K", "MODIS/061/MOD11A1", "daily", 250.0, 350.0, "continuous", "zscore"
    ),
    "lst_night_k": FeatureSpec(
        "lst_night_k", "K", "MODIS/061/MOD11A1", "daily", 230.0, 310.0, "continuous", "zscore"
    ),
    # --- Slow-varying ---
    "NDVI": FeatureSpec(
        "NDVI", "index", "MODIS/061/MOD13Q1", "slow", -0.2, 1.0, "continuous", "minmax"
    ),
    "EVI": FeatureSpec(
        "EVI", "index", "MODIS/061/MOD13Q1", "slow", -0.2, 1.0, "continuous", "minmax"
    ),
    "pdsi": FeatureSpec(
        "pdsi", "index", "GRIDMET/DROUGHT", "slow", -10.0, 10.0, "continuous", "zscore"
    ),
    "eddi_14d": FeatureSpec(
        "eddi_14d", "index", "GRIDMET/DROUGHT", "slow", -3.0, 3.0, "continuous", "zscore"
    ),
    "eddi_30d": FeatureSpec(
        "eddi_30d", "index", "GRIDMET/DROUGHT", "slow", -3.0, 3.0, "continuous", "zscore"
    ),
    "smoke_aerosol_index": FeatureSpec(
        "smoke_aerosol_index",
        "index",
        "COPERNICUS/S5P",
        "slow",
        -5.0,
        15.0,
        "continuous",
        "zscore",
    ),
    # --- Terrain (static) ---
    "slope_deg": FeatureSpec(
        "slope_deg", "degrees", "USGS/3DEP/10m", "static", 0.0, 90.0, "continuous", "minmax"
    ),
    "aspect_sin": FeatureSpec(
        "aspect_sin", "unitless", "USGS/3DEP/10m", "static", -1.0, 1.0, "continuous", "none"
    ),
    "aspect_cos": FeatureSpec(
        "aspect_cos", "unitless", "USGS/3DEP/10m", "static", -1.0, 1.0, "continuous", "none"
    ),
    "tpi": FeatureSpec(
        "tpi", "m", "USGS/3DEP/10m", "static", -500.0, 500.0, "continuous", "zscore"
    ),
    "terrain_ruggedness": FeatureSpec(
        "terrain_ruggedness", "m", "USGS/3DEP/10m", "static", 0.0, 500.0, "continuous", "zscore"
    ),
    "elevation": FeatureSpec(
        "elevation", "m", "USGS/3DEP/10m", "static", 0.0, 4500.0, "continuous", "zscore"
    ),
    # --- Fuel/land cover (static) ---
    "fuel_load": FeatureSpec(
        "fuel_load", "proxy 0-1", "USGS/NLCD", "static", 0.0, 1.0, "continuous", "none"
    ),
    "is_firebreak": FeatureSpec(
        "is_firebreak", "binary", "USGS/NLCD", "static", 0.0, 1.0, "binary", "none"
    ),
    "canopy_cover_pct": FeatureSpec(
        "canopy_cover_pct", "%", "USGS/NLCD", "static", 0.0, 100.0, "continuous", "minmax"
    ),
    # --- Vegetation (static, categorical) ---
    "vegetation_type": FeatureSpec(
        "vegetation_type", "EVT code", "LANDFIRE", "static", 0.0, 9999.0, "categorical", "embedding"
    ),
    "vegetation_cover": FeatureSpec(
        "vegetation_cover", "EVC code", "LANDFIRE", "static", 0.0, 999.0, "categorical", "embedding"
    ),
    "vegetation_height": FeatureSpec(
        "vegetation_height",
        "EVH code",
        "LANDFIRE",
        "static",
        0.0,
        999.0,
        "categorical",
        "embedding",
    ),
    # --- Fire history (static) ---
    "years_since_burn": FeatureSpec(
        "years_since_burn", "years", "USFS/GTAC/MTBS", "static", 0.0, 99.0, "continuous", "minmax"
    ),
    "burn_count": FeatureSpec(
        "burn_count", "count", "USFS/GTAC/MTBS", "static", 0.0, 10.0, "continuous", "minmax"
    ),
    # --- Roads/WUI (static) ---
    "has_road": FeatureSpec(
        "has_road", "binary", "TIGER/2016/Roads", "static", 0.0, 1.0, "binary", "none"
    ),
    "population": FeatureSpec(
        "population", "count", "JRC/GHSL", "static", 0.0, 50000.0, "continuous", "zscore"
    ),
    "built_up": FeatureSpec(
        "built_up", "m2", "JRC/GHSL", "static", 0.0, 10000.0, "continuous", "zscore"
    ),
    # --- Previous fire state (pre-shifted, safe as input at time t) ---
    "prev_fire_state": FeatureSpec(
        "prev_fire_state",
        "binary",
        "pipeline",
        "hourly",
        0.0,
        1.0,
        "binary",
        "none",
        safe_as_input=True,
        requires_temporal_lag=False,
    ),
    "prev_distance_to_fire": FeatureSpec(
        "prev_distance_to_fire",
        "pixels",
        "derived",
        "hourly",
        -1.0,
        100.0,
        "continuous",
        "zscore",
        safe_as_input=True,
        requires_temporal_lag=False,
    ),
    "prev_fire_neighborhood": FeatureSpec(
        "prev_fire_neighborhood",
        "fraction",
        "derived",
        "hourly",
        0.0,
        1.0,
        "continuous",
        "none",
        safe_as_input=True,
        requires_temporal_lag=False,
    ),
    # --- Targets (NOT safe as model input) ---
    "labels": FeatureSpec(
        "labels", "binary", "pipeline", "hourly", 0.0, 1.0, "binary", "none", safe_as_input=False
    ),
    "soft_labels": FeatureSpec(
        "soft_labels",
        "probability",
        "pipeline",
        "hourly",
        0.0,
        1.0,
        "continuous",
        "none",
        safe_as_input=False,
    ),
    "fire_change": FeatureSpec(
        "fire_change",
        "binary",
        "pipeline",
        "hourly",
        0.0,
        1.0,
        "binary",
        "none",
        safe_as_input=False,
    ),
    # --- Loss weights (NOT safe as model input) ---
    "loss_weights": FeatureSpec(
        "loss_weights",
        "weight",
        "pipeline",
        "hourly",
        0.0,
        1.0,
        "continuous",
        "none",
        safe_as_input=False,
    ),
    # --- Diagnostics (NOT safe as model input) ---
    "_diag_raw_confidence": FeatureSpec(
        "_diag_raw_confidence",
        "probability",
        "GOES",
        "hourly",
        0.0,
        1.0,
        "continuous",
        "none",
        safe_as_input=False,
    ),
    "_diag_capped_frp": FeatureSpec(
        "_diag_capped_frp",
        "MW",
        "GOES",
        "hourly",
        0.0,
        2000.0,
        "continuous",
        "none",
        safe_as_input=False,
    ),
    "_diag_frp_reliability": FeatureSpec(
        "_diag_frp_reliability",
        "score",
        "pipeline",
        "hourly",
        0.0,
        1.0,
        "continuous",
        "none",
        safe_as_input=False,
    ),
    "_diag_was_imputed": FeatureSpec(
        "_diag_was_imputed",
        "binary",
        "pipeline",
        "hourly",
        0.0,
        1.0,
        "binary",
        "none",
        safe_as_input=False,
    ),
}


def get_safe_input_features() -> list[str]:
    """Return names of features that are safe to use as model inputs.

    WARNING: Features with requires_temporal_lag=True must be used at
    time t-1 to predict time t. Using them at the same timestep as the
    target is label leakage.
    """
    return [name for name, spec in FEATURE_REGISTRY.items() if spec.safe_as_input]


def get_lagged_features() -> list[str]:
    """Return features that require a temporal lag (t-1) to avoid label leakage."""
    return [name for name, spec in FEATURE_REGISTRY.items() if spec.requires_temporal_lag]


def get_feature_spec(name: str) -> FeatureSpec | None:
    """Look up a feature specification by channel name (strips common prefixes)."""
    clean = name
    for prefix in ("static_", "slow_", "hourly_", "daily_"):
        if clean.startswith(prefix):
            clean = clean[len(prefix) :]
            break
    return FEATURE_REGISTRY.get(clean)
