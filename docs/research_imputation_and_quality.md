"""
Spatiotemporal Imputation & Data Quality Methods for Satellite Fire Time Series
================================================================================

Research compilation for the wildfire-data-pipeline project.
Contains concrete algorithms, implementations, and recommendations
for GOES satellite fire detection data used in ML training.

All code examples are runnable with numpy, scipy, and optionally
pykalman/filterpy. They operate on arrays shaped (T, H, W) matching
the pipeline's existing convention.

References are cited inline. Key papers:
  - Roberts & Wooster (2014), RSE 152:392-412  (Kalman filter for geostationary fire FRP)
  - GOFER: Systematically tracking hourly wildfire progression, ESSD 16:1395 (2024)
  - GOES-R ATBD Fire v2.6 (NOAA/NESDIS)
  - Google Earth Engine GOES-16 FDCC/FDCF catalog
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# =============================================================================
# 1. PHYSICAL BOUNDS FOR GOES FIRE PARAMETERS
# =============================================================================
#
# Source: Google Earth Engine GOES-16 FDCC catalog + GOES-R ATBD + literature
#
# Fire Radiative Power (FRP)
#   - GEE catalog range: 0-200,000 MW (stored as uint, units MW)
#   - Practical detection floor: ~30-35 MW (Li et al., 2021 - fires below this
#     are largely undetected by ABI)
#   - Typical range for detected wildfires: 30-2,000 MW per pixel
#   - Saturation: GOES-16 ABI saturates in <0.2% of detections (vs 6% for GOES-13)
#   - Extreme fires (e.g., August Complex 2020): up to ~2,000+ MW per pixel
#   - For ML: values above 500 MW are increasingly uncertain due to smoke/
#     pyro-cumulonimbus obscuration
#   - Recommended cap: 99th percentile or 2,000 MW, whichever is lower
#
# Fire Temperature
#   - GEE catalog: scale=0.0549367, offset=400 -> range ~400-1,792 K
#   - Wildland fire flame temperatures: 800-1,500 K (typical), up to ~1,800 K
#   - Sub-pixel retrieval is an average of fire+background, so retrieved values
#     are often 400-800 K (mixed signal)
#   - Physical bounds for capping: 400 K (background) to 2,000 K (upper physical limit)
#   - Values below 500 K likely indicate mostly background, not useful fire signal
#
# Fire Area
#   - GEE catalog: scale=60.98, offset=4,000 -> range ~4,000-1,023,814 m^2
#   - GOES pixel at nadir: 2 km x 2 km = 4,000,000 m^2 = 4 km^2
#   - Sub-pixel fire area is typically 0.001-1 km^2 within a 2km pixel
#   - Physical upper bound: pixel area (~4 km^2 at nadir, larger at limb)
#   - For ML: fire area > 4 km^2 indicates possible multi-pixel fire or error

PHYSICAL_BOUNDS = {
    "frp_mw": {"min": 0.0, "max": 200_000.0, "practical_max": 2_000.0, "detection_floor": 30.0},
    "fire_temp_k": {"min": 400.0, "max": 2_000.0, "typical_min": 600.0, "typical_max": 1_500.0},
    "fire_area_m2": {"min": 0.0, "max": 4_000_000.0, "typical_max": 1_000_000.0},
}


# =============================================================================
# 2. TEMPORAL IMPUTATION METHODS
# =============================================================================


def forward_fill_temporal(data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Forward-fill along the time axis for missing observations.

    For fire confidence time series, forward-fill assumes "if it was on fire
    an hour ago, it's still on fire now." This is physically reasonable for
    wildfire spread (fires don't spontaneously extinguish in 1 hour).

    WHEN TO USE: Short gaps (1-3 hours) caused by cloud/scan issues.
    WHEN NOT TO USE: Gaps > 6 hours (fire may have actually been suppressed
    or changed direction). Mark as missing instead.

    Args:
        data: (T, H, W) array of values (e.g., confidence or FRP)
        valid_mask: (T, H, W) boolean array, True where observation exists

    Returns:
        (T, H, W) array with gaps forward-filled. Pixels never observed
        remain as their original value.
    """
    T, _H, _W = data.shape
    result = data.copy()
    for t in range(1, T):
        missing = ~valid_mask[t]
        result[t] = np.where(missing, result[t - 1], result[t])
    return result


def backward_fill_temporal(data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Backward-fill along the time axis.

    CAUTION: Introduces future information. Only appropriate for offline
    (non-causal) analysis, never for real-time prediction targets. Use
    for creating "best estimate" training labels, not for model inputs.

    Args:
        data: (T, H, W) array
        valid_mask: (T, H, W) boolean array

    Returns:
        (T, H, W) backward-filled array
    """
    T, _H, _W = data.shape
    result = data.copy()
    for t in range(T - 2, -1, -1):
        missing = ~valid_mask[t]
        result[t] = np.where(missing, result[t + 1], result[t])
    return result


def linear_interpolation_temporal(
    data: np.ndarray,
    valid_mask: np.ndarray,
    max_gap: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation along the time axis per pixel.

    Interpolates FRP or confidence values between valid observations.
    Gaps longer than max_gap hours are NOT interpolated (marked missing).

    WHEN TO USE: Continuous-valued data (FRP, confidence) with short gaps.
    WHEN NOT TO USE: Binary fire/no-fire labels (use morphological closing
    or majority vote instead).

    Args:
        data: (T, H, W) array
        valid_mask: (T, H, W) boolean mask
        max_gap: Maximum gap length (hours) to interpolate across

    Returns:
        Tuple of (interpolated_data, was_interpolated_mask)
    """
    _T, H, W = data.shape
    result = data.copy()
    was_interpolated = np.zeros_like(data, dtype=bool)

    for i in range(H):
        for j in range(W):
            pixel_valid = valid_mask[:, i, j]
            valid_times = np.where(pixel_valid)[0]

            if len(valid_times) < 2:
                continue

            # Identify gaps and their lengths
            for k in range(len(valid_times) - 1):
                t_start = valid_times[k]
                t_end = valid_times[k + 1]
                gap_len = t_end - t_start - 1

                if gap_len == 0 or gap_len > max_gap:
                    continue

                # Linear interpolation between t_start and t_end
                t_gap = np.arange(t_start + 1, t_end)
                frac = (t_gap - t_start) / (t_end - t_start)
                result[t_gap, i, j] = data[t_start, i, j] * (1 - frac) + data[t_end, i, j] * frac
                was_interpolated[t_gap, i, j] = True

    return result, was_interpolated


def morphological_closing_temporal(binary: np.ndarray, gap_hours: int = 3) -> np.ndarray:
    """Fill short temporal gaps in binary fire detection using morphological closing.

    binary_closing = erosion(dilation(input)). The structuring element is
    a 1D temporal line of length (2*gap_hours+1), applied independently at
    each spatial pixel. This fills gaps of up to gap_hours consecutive
    missing-fire timesteps while preserving the spatial extent.

    WHEN TO USE: Binary fire labels with 1-3 hour gaps caused by satellite
    detection flickering (cloud, scan angle, sensor noise).
    WHEN NOT TO USE: Continuous-valued data. Use linear interpolation instead.

    This is equivalent to the pipeline's existing majority_vote_smooth but
    is more principled for gap-filling because it:
    1. Only fills gaps BETWEEN existing detections (doesn't extend fire forward)
    2. Has a well-defined morphological interpretation
    3. Is faster (single scipy call vs Python loop)

    Args:
        binary: (T, H, W) binary fire array (0/1)
        gap_hours: Maximum gap to close (structuring element half-width)

    Returns:
        (T, H, W) closed binary array
    """
    # Structuring element: 1D line along time axis only.
    # Shape: (2*gap_hours+1, 1, 1) so it acts only temporally.
    struct = np.zeros((2 * gap_hours + 1, 1, 1), dtype=bool)
    struct[:, 0, 0] = True

    closed = ndimage.binary_closing(binary.astype(bool), structure=struct, iterations=1)
    return closed.astype(np.float32)


def kalman_smooth_temporal(
    data: np.ndarray,
    valid_mask: np.ndarray,
    process_noise: float = 0.01,
    measurement_noise: float = 0.1,
) -> np.ndarray:
    """Kalman smoothing for continuous-valued fire time series (e.g., FRP).

    Based on Roberts & Wooster (2014) approach to geostationary fire
    detection. Uses a simple random-walk state-space model:

        State:  x[t] = x[t-1] + w,     w ~ N(0, Q)
        Obs:    z[t] = x[t] + v,        v ~ N(0, R)

    Missing observations (valid_mask=False) are handled natively by the
    Kalman filter -- the predict step runs but the update step is skipped,
    producing a smooth interpolation through gaps.

    WHEN TO USE: Continuous-valued time series (FRP) where you want smooth
    estimates that respect temporal dynamics.
    WHEN NOT TO USE: Binary data. Use morphological closing instead.

    IMPLEMENTATION OPTIONS:
    1. pykalman (recommended): Handles masked arrays natively, includes EM
       for parameter estimation. pip install pykalman
    2. filterpy: Lower-level, more control. pip install filterpy
    3. Manual numpy: Below for zero-dependency version.

    Args:
        data: (T, H, W) array of observations
        valid_mask: (T, H, W) boolean mask
        process_noise: Q (state transition noise variance)
        measurement_noise: R (observation noise variance)

    Returns:
        (T, H, W) Kalman-smoothed array
    """
    T, H, W = data.shape
    result = np.zeros_like(data)

    # Per-pixel 1D Kalman smoother (numpy-only implementation)
    for i in range(H):
        for j in range(W):
            obs = data[:, i, j]
            mask = valid_mask[:, i, j]

            # Forward pass (Kalman filter)
            x_filt = np.zeros(T)  # filtered state mean
            p_filt = np.zeros(T)  # filtered state variance
            x_pred = np.zeros(T)  # predicted state mean
            p_pred = np.zeros(T)  # predicted state variance

            Q = process_noise
            R = measurement_noise

            # Initialize from first valid observation
            first_valid = np.argmax(mask) if mask.any() else 0
            x_filt[0] = obs[first_valid] if mask.any() else 0.0
            p_filt[0] = R

            for t in range(1, T):
                # Predict
                x_pred[t] = x_filt[t - 1]
                p_pred[t] = p_filt[t - 1] + Q

                if mask[t]:
                    # Update
                    K = p_pred[t] / (p_pred[t] + R)
                    x_filt[t] = x_pred[t] + K * (obs[t] - x_pred[t])
                    p_filt[t] = (1 - K) * p_pred[t]
                else:
                    # No observation: prediction is the best estimate
                    x_filt[t] = x_pred[t]
                    p_filt[t] = p_pred[t]

            # Backward pass (RTS smoother)
            x_smooth = np.zeros(T)
            x_smooth[T - 1] = x_filt[T - 1]

            for t in range(T - 2, -1, -1):
                if p_pred[t + 1] > 0:
                    L = p_filt[t] / p_pred[t + 1]
                    x_smooth[t] = x_filt[t] + L * (x_smooth[t + 1] - x_pred[t + 1])
                else:
                    x_smooth[t] = x_filt[t]

            result[:, i, j] = x_smooth

    return result


def kalman_smooth_with_pykalman(
    data: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Kalman smoothing using pykalman library (recommended for production).

    pykalman handles missing observations natively via numpy masked arrays
    and can learn optimal Q, R parameters via EM algorithm.

    Requires: pip install pykalman

    Args:
        data: (T, H, W) array
        valid_mask: (T, H, W) boolean mask

    Returns:
        (T, H, W) smoothed array
    """
    from numpy import ma
    from pykalman import KalmanFilter

    _T, H, W = data.shape
    result = np.zeros_like(data)

    for i in range(H):
        for j in range(W):
            obs = data[:, i, j]
            mask = valid_mask[:, i, j]

            # Create masked array (pykalman treats masked values as missing)
            masked_obs = ma.array(obs, mask=~mask)

            kf = KalmanFilter(
                transition_matrices=np.array([[1.0]]),
                observation_matrices=np.array([[1.0]]),
                initial_state_mean=np.array([obs[mask].mean() if mask.any() else 0.0]),
                initial_state_covariance=np.array([[1.0]]),
                n_dim_state=1,
                n_dim_obs=1,
            )

            # EM learns optimal noise parameters from available data
            if mask.sum() > 10:
                kf = kf.em(masked_obs.reshape(-1, 1), n_iter=5)

            smoothed, _ = kf.smooth(masked_obs.reshape(-1, 1))
            result[:, i, j] = smoothed.ravel()

    return result


# Decision guide: when to impute vs when to mark as missing
#
# IMPUTE when:
#   - Gap is short (1-3 hours for fire labels, 1-6 hours for FRP)
#   - Data on both sides of gap exists (not at sequence boundaries)
#   - Physical process is smooth (FRP changes gradually)
#   - You're creating TRAINING LABELS (not model inputs)
#
# MARK AS MISSING when:
#   - Gap is long (>6 hours)
#   - Entire spatial region is missing (no spatial context)
#   - At sequence boundaries (can't interpolate, only extrapolate)
#   - Data is for model INPUT features (never hide missingness from the model)
#   - Fire boundary is involved (imputation would create false boundaries)
#   - The cause of missingness correlates with the target
#     (e.g., thick smoke blocks satellite -> active fire underneath)


# =============================================================================
# 3. SPATIAL IMPUTATION METHODS
# =============================================================================


def inverse_distance_weighting(
    data: np.ndarray,
    valid_mask: np.ndarray,
    power: float = 2.0,
    max_radius: int = 5,
) -> np.ndarray:
    """Inverse distance weighting for missing pixels in a 2D spatial field.

    Estimates missing pixel values as a weighted average of nearby valid
    pixels, where weights are 1/distance^power. Follows Tobler's first
    law of geography.

    WHEN TO USE: Continuous fields (FRP, temperature) with scattered
    missing pixels. Works well when the field is spatially smooth.

    WHEN TO USE WITH CAUTION:
    - Near fire boundaries (IDW will blur the boundary)
    - Across terrain discontinuities (mountains, water bodies)
    - When most neighbors are missing (result is unreliable)

    WHEN NOT TO USE:
    - Binary fire/no-fire labels (use morphological operations instead)
    - Large contiguous missing regions (>50% of local neighborhood)
    - Across known physical discontinuities

    Args:
        data: (H, W) 2D array with valid and missing values
        valid_mask: (H, W) boolean mask, True where data is valid
        power: Distance weighting exponent (2.0 is standard)
        max_radius: Maximum search radius in pixels

    Returns:
        (H, W) array with missing pixels filled
    """
    H, W = data.shape
    result = data.copy()
    missing = ~valid_mask

    if not missing.any():
        return result

    # Pre-compute coordinate grids for missing pixels
    missing_coords = np.argwhere(missing)

    for idx in range(len(missing_coords)):
        y, x = missing_coords[idx]

        # Extract local neighborhood
        y_min = max(0, y - max_radius)
        y_max = min(H, y + max_radius + 1)
        x_min = max(0, x - max_radius)
        x_max = min(W, x + max_radius + 1)

        local_valid = valid_mask[y_min:y_max, x_min:x_max]
        local_data = data[y_min:y_max, x_min:x_max]

        if not local_valid.any():
            continue

        # Compute distances from (y, x) to all valid pixels in window
        ys, xs = np.where(local_valid)
        ys_abs = ys + y_min
        xs_abs = xs + x_min
        distances = np.sqrt((ys_abs - y) ** 2 + (xs_abs - x) ** 2)

        # Avoid division by zero (exact co-location)
        distances = np.maximum(distances, 1e-10)

        weights = 1.0 / distances**power
        values = local_data[local_valid]
        result[y, x] = np.sum(weights * values) / np.sum(weights)

    return result


def nearest_neighbor_spatial(
    data: np.ndarray,
    valid_mask: np.ndarray,
    max_distance: int = 3,
) -> np.ndarray:
    """Nearest-neighbor interpolation for missing spatial pixels.

    Each missing pixel takes the value of its closest valid neighbor.
    Fast and preserves sharp boundaries (no averaging/blurring).

    WHEN TO USE: Binary labels, categorical data, or when you want to
    preserve sharp fire boundaries.
    WHEN NOT TO USE: Continuous data where smooth transitions are expected.

    Args:
        data: (H, W) 2D array
        valid_mask: (H, W) boolean mask
        max_distance: Maximum search distance in pixels

    Returns:
        (H, W) array with gaps filled by nearest valid value
    """
    from scipy.ndimage import distance_transform_edt

    result = data.copy()

    if valid_mask.all():
        return result

    # distance_transform_edt gives distance to nearest True pixel
    # and indices of that nearest pixel
    distances, indices = distance_transform_edt(
        ~valid_mask, return_distances=True, return_indices=True
    )

    # Only fill pixels within max_distance
    fill_mask = (~valid_mask) & (distances <= max_distance)

    # indices[0] and indices[1] give the row/col of the nearest valid pixel
    result[fill_mask] = data[indices[0][fill_mask], indices[1][fill_mask]]

    return result


# DANGERS OF SPATIAL IMPUTATION:
#
# 1. Fire Boundaries: IDW/interpolation will BLUR the boundary between
#    fire and no-fire. A pixel that's missing at the fire edge will get
#    an intermediate value (~0.5) rather than a clean 0 or 1. For binary
#    labels, use nearest-neighbor instead.
#
# 2. Terrain Discontinuities: Mountains, rivers, and roads create natural
#    firebreaks. Interpolating FRP or temperature across a ridge or river
#    where the fire hasn't crossed creates physically impossible values.
#    Solution: use a terrain mask to block interpolation across known barriers.
#
# 3. Sensor Geometry: GOES pixels grow larger toward the Earth's limb.
#    A "missing" pixel at high view angle covers much more area than one
#    at nadir. Spatial interpolation doesn't account for this.
#
# 4. Cloud Correlation: If a pixel is missing BECAUSE of cloud cover,
#    the surrounding valid pixels are biased toward clear-sky conditions.
#    The missing pixel (under cloud) may have very different conditions.
#    This is MNAR (Missing Not At Random) and all imputation methods
#    will be biased.


# =============================================================================
# 4. OUTLIER DETECTION AND CAPPING
# =============================================================================


def iqr_outlier_detection(
    data: np.ndarray,
    valid_mask: np.ndarray | None = None,
    factor: float = 1.5,
) -> tuple[np.ndarray, float, float]:
    """IQR-based outlier detection for geospatial arrays.

    The IQR method is non-parametric and makes no distribution assumptions,
    making it robust for heavy-tailed FRP distributions.

    Args:
        data: Array of values (any shape)
        valid_mask: Optional boolean mask for valid pixels
        factor: IQR multiplier (1.5 = standard, 3.0 = extreme only)

    Returns:
        Tuple of (outlier_mask, lower_bound, upper_bound)
    """
    if valid_mask is not None:
        values = data[valid_mask]
    else:
        values = data[~np.isnan(data)] if np.isnan(data).any() else data.ravel()

    if len(values) == 0:
        return np.zeros_like(data, dtype=bool), 0.0, 0.0

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    outlier_mask = (data < lower_bound) | (data > upper_bound)
    if valid_mask is not None:
        outlier_mask &= valid_mask

    return outlier_mask, lower_bound, upper_bound


def winsorize_array(
    data: np.ndarray,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
    valid_mask: np.ndarray | None = None,
    physical_bounds: tuple[float, float] | None = None,
) -> np.ndarray:
    """Winsorize (percentile-cap) array values.

    Clips extreme values to specified percentiles, preserving the data
    point but reducing outlier influence on ML training.

    For FRP data, combine with physical bounds:
      winsorize_array(frp, upper_pct=99, physical_bounds=(0, 2000))

    Args:
        data: Array to winsorize (any shape)
        lower_pct: Lower percentile to clip to (default 1st)
        upper_pct: Upper percentile to clip to (default 99th)
        valid_mask: Optional mask for percentile computation
        physical_bounds: Optional (min, max) physical bounds to also enforce

    Returns:
        Winsorized array (clipped copy)
    """
    values = data[valid_mask] if valid_mask is not None else data.ravel()

    lower_val = np.percentile(values, lower_pct)
    upper_val = np.percentile(values, upper_pct)

    if physical_bounds is not None:
        lower_val = max(lower_val, physical_bounds[0])
        upper_val = min(upper_val, physical_bounds[1])

    return np.clip(data, lower_val, upper_val)


def zscore_spatial_outliers(
    data: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float = 3.0,
    window_size: int = 5,
) -> np.ndarray:
    """Z-score based spatial anomaly detection.

    For each pixel, computes the local mean and std from its spatial
    neighborhood. Pixels more than `threshold` standard deviations from
    the local mean are flagged as outliers.

    Useful for detecting sensor artifacts (stuck pixels, hot spots from
    industrial sources, etc.) that appear as spatial anomalies.

    Args:
        data: (H, W) 2D array
        valid_mask: (H, W) boolean mask
        threshold: Z-score threshold for outlier detection
        window_size: Side length of local neighborhood window

    Returns:
        (H, W) boolean mask, True where outlier detected
    """
    # Compute local mean and variance using generic_filter
    # We need to handle masked values carefully
    masked_data = np.where(valid_mask, data, np.nan)

    def _nanmean(values):
        valid = values[~np.isnan(values)]
        return np.nanmean(valid) if len(valid) > 0 else np.nan

    def _nanstd(values):
        valid = values[~np.isnan(values)]
        return np.nanstd(valid) if len(valid) > 1 else np.nan

    local_mean = ndimage.generic_filter(
        masked_data, _nanmean, size=window_size, mode="constant", cval=np.nan
    )
    local_std = ndimage.generic_filter(
        masked_data, _nanstd, size=window_size, mode="constant", cval=np.nan
    )

    # Avoid division by zero
    local_std = np.where(local_std > 0, local_std, np.inf)

    z_scores = np.abs(data - local_mean) / local_std
    outlier_mask = (z_scores > threshold) & valid_mask

    return outlier_mask


def apply_physical_bounds_frp(frp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply GOES FRP physical bounds and flag anomalies.

    Based on GOES-R ATBD and GEE catalog specifications:
    - Stored range: 0-200,000 MW
    - Practical detection floor: ~30 MW
    - Practical ceiling for reliable values: ~2,000 MW
    - Values above ~500 MW have increasing uncertainty
      (smoke obscuration, saturation effects)

    Args:
        frp: (T, H, W) FRP array in MW

    Returns:
        Tuple of (capped_frp, reliability_score) where reliability_score
        is in [0, 1], highest for moderate FRP values.
    """
    capped = np.clip(frp, 0.0, 2_000.0)

    # Reliability decreases for very high values (saturation effects)
    # and very low values (near detection threshold)
    reliability = np.ones_like(frp)

    # Low FRP: linear ramp from 0 at 0 MW to 1 at 50 MW
    low_mask = frp < 50.0
    reliability = np.where(low_mask, np.clip(frp / 50.0, 0.0, 1.0), reliability)

    # High FRP: linear ramp from 1 at 500 MW to 0.5 at 2000 MW
    high_mask = frp > 500.0
    reliability = np.where(
        high_mask, np.clip(1.0 - 0.5 * (frp - 500.0) / 1500.0, 0.5, 1.0), reliability
    )

    # Beyond 2000 MW: floor at 0.3 reliability
    extreme_mask = frp > 2_000.0
    reliability = np.where(extreme_mask, 0.3, reliability)

    return capped, reliability


def apply_physical_bounds_temperature(temp_k: np.ndarray) -> np.ndarray:
    """Apply physical bounds to GOES fire temperature retrievals.

    Sub-pixel fire temperature from GOES is a mixed signal of fire and
    background. True flame temperatures for wildland fires: 800-1,500 K.
    Retrieved sub-pixel temperatures (mixed): often 400-800 K.
    Physical upper bound: ~2,000 K (even industrial fires rarely exceed this).

    Args:
        temp_k: Temperature array in Kelvin

    Returns:
        Clipped temperature array
    """
    return np.clip(temp_k, 400.0, 2_000.0)


# =============================================================================
# 5. DATA COMPLETENESS SCORING
# =============================================================================


def compute_pixel_quality_score(
    valid_mask: np.ndarray,
    cloud_mask: np.ndarray,
    confidence: np.ndarray,
    confidence_threshold: float = 0.3,
) -> np.ndarray:
    """Compute per-pixel data quality score for ML training weights.

    Quality score is in [0, 1] and reflects how trustworthy each
    pixel-timestep is for training. Components:

    1. Observation completeness: Was the satellite looking at this pixel?
    2. Cloud freedom: Was the pixel cloud-free?
    3. Detection confidence: How confident is the fire/no-fire label?
    4. Temporal context: Do neighboring timesteps agree?

    Args:
        valid_mask: (T, H, W) observation validity
        cloud_mask: (T, H, W) cloud presence
        confidence: (T, H, W) fire detection confidence
        confidence_threshold: Threshold used for binary labeling

    Returns:
        (T, H, W) quality score in [0, 1]
    """
    T, H, W = valid_mask.shape

    # Component 1: Observation exists and is cloud-free
    obs_quality = valid_mask.astype(np.float32) * (1.0 - cloud_mask.astype(np.float32))

    # Component 2: Confidence is decisive (far from threshold)
    # Pixels near the threshold are ambiguous; those far away are more certain
    distance_from_threshold = np.abs(confidence - confidence_threshold)
    # Normalize: 0 at threshold, 1 when confidence is at 0 or 1
    max_distance = max(confidence_threshold, 1.0 - confidence_threshold)
    confidence_certainty = np.clip(distance_from_threshold / max_distance, 0.0, 1.0)

    # Component 3: Temporal consistency
    # A pixel that agrees with its temporal neighbors is more trustworthy
    temporal_agreement = np.ones((T, H, W), dtype=np.float32)
    binary = (confidence >= confidence_threshold).astype(np.float32)
    for t in range(1, T - 1):
        agrees_prev = (binary[t] == binary[t - 1]).astype(np.float32)
        agrees_next = (binary[t] == binary[t + 1]).astype(np.float32)
        temporal_agreement[t] = 0.5 * (agrees_prev + agrees_next)

    # Combine with weights
    quality = 0.4 * obs_quality + 0.3 * confidence_certainty + 0.3 * temporal_agreement

    # Zero out quality where we have no observation at all
    quality *= valid_mask.astype(np.float32)

    return quality


def compute_timestep_completeness(
    valid_mask: np.ndarray,
    cloud_mask: np.ndarray,
) -> np.ndarray:
    """Compute per-timestep data completeness ratio.

    Args:
        valid_mask: (T, H, W) observation validity
        cloud_mask: (T, H, W) cloud presence

    Returns:
        (T,) array of completeness ratios in [0, 1]
    """
    H, W = valid_mask.shape[1], valid_mask.shape[2]
    total_pixels = H * W
    usable = valid_mask.astype(np.float32) * (1.0 - cloud_mask.astype(np.float32))
    return usable.sum(axis=(1, 2)) / total_pixels


def compute_spatial_completeness(
    valid_mask: np.ndarray,
    cloud_mask: np.ndarray,
) -> np.ndarray:
    """Compute per-pixel temporal completeness ratio.

    Args:
        valid_mask: (T, H, W) observation validity
        cloud_mask: (T, H, W) cloud presence

    Returns:
        (H, W) array of completeness ratios in [0, 1]
    """
    T = valid_mask.shape[0]
    usable = valid_mask.astype(np.float32) * (1.0 - cloud_mask.astype(np.float32))
    return usable.sum(axis=0) / T


# Minimum data completeness thresholds for ML training:
#
# These thresholds are informed by the literature and practical experience:
#
# PER-TIMESTEP:
#   - Minimum 20% valid pixels to include in training (below this, the
#     spatial context is too sparse for a CNN to learn from)
#   - Ideal: >60% valid pixels
#   - Below 10%: exclude entirely, don't even impute
#
# PER-PIXEL (temporal):
#   - Minimum 30% valid timesteps to include in training
#   - Below this, temporal patterns can't be reliably learned
#   - For Kalman smoothing: need at least 10 valid observations to
#     estimate noise parameters (via EM)
#
# OVERALL FIRE EVENT:
#   - If >50% of total pixel-timesteps are missing, the fire event
#     should be flagged for review or excluded from training
#   - The pipeline's current 10% failure threshold is for download failures
#     only; cloud masking can push actual data loss much higher

COMPLETENESS_THRESHOLDS = {
    "timestep_min": 0.20,  # Min fraction of valid pixels per hour
    "timestep_ideal": 0.60,  # Ideal fraction
    "timestep_exclude": 0.10,  # Below this, exclude hour entirely
    "pixel_temporal_min": 0.30,  # Min fraction of valid hours per pixel
    "event_max_missing": 0.50,  # Max missing fraction for entire event
    "kalman_min_obs": 10,  # Min valid observations for EM parameter learning
}


# =============================================================================
# 6. DATA QUALITY WEIGHT MASK FOR TRAINING LOSS
# =============================================================================


def create_training_weight_mask(
    quality_score: np.ndarray,
    valid_mask: np.ndarray,
    imputation_mask: np.ndarray,
    min_quality: float = 0.2,
    imputed_weight: float = 0.3,
) -> np.ndarray:
    """Create a weight mask for ML training loss computation.

    The weight mask encodes our confidence in each pixel-timestep:
    - Weight 1.0: High-quality observed data
    - Weight 0.0-1.0: Reduced weight for lower quality
    - Weight 0.0: Excluded from loss entirely (missing/invalid)
    - Imputed data: Reduced weight (we're less certain about these)

    The ML training loss should be:
        loss = sum(weight * pixel_loss) / sum(weight)

    This prevents the model from fitting to noise in low-quality data
    while still using it as weak supervision.

    BEST PRACTICES:
    1. ALWAYS flag imputed values separately from observed values
    2. Give imputed values lower weight in training loss
    3. Store the imputation_mask alongside the data for reproducibility
    4. Monitor model performance on high-quality vs low-quality subsets
       separately during validation

    Args:
        quality_score: (T, H, W) per-pixel quality from compute_pixel_quality_score
        valid_mask: (T, H, W) raw observation validity
        imputation_mask: (T, H, W) boolean, True where values were imputed
        min_quality: Minimum quality to include (below = weight 0)
        imputed_weight: Weight multiplier for imputed values

    Returns:
        (T, H, W) weight mask in [0, 1]
    """
    weights = quality_score.copy()

    # Zero out below minimum quality threshold
    weights[weights < min_quality] = 0.0

    # Reduce weight for imputed values
    weights[imputation_mask] *= imputed_weight

    # Zero out completely invalid observations
    weights[~valid_mask.astype(bool)] = 0.0

    return weights


# =============================================================================
# 7. METADATA FOR IMPUTED DATA
# =============================================================================
#
# When saving imputed data, the following metadata MUST accompany it:
#
# {
#     "imputation": {
#         "temporal_method": "morphological_closing",  # or "kalman", "linear", etc.
#         "temporal_max_gap_hours": 3,
#         "spatial_method": "none",  # or "idw", "nearest_neighbor"
#         "spatial_max_radius_pixels": 5,
#         "outlier_method": "winsorize_99",
#         "physical_bounds_applied": {
#             "frp_mw": [0, 2000],
#             "temp_k": [400, 2000],
#         },
#     },
#     "quality": {
#         "mean_completeness": 0.73,
#         "min_timestep_completeness": 0.22,
#         "max_timestep_completeness": 0.95,
#         "pct_imputed": 0.08,
#         "pct_excluded": 0.19,
#         "outliers_detected": 42,
#         "outliers_capped": 42,
#     },
#     "masks_included": [
#         "imputation_mask",      # bool: True where imputed
#         "quality_score",        # float [0,1]: per-pixel quality
#         "training_weights",     # float [0,1]: suggested loss weights
#         "outlier_mask",         # bool: True where outlier was detected
#     ],
# }


# =============================================================================
# 8. COMPLETE PIPELINE EXAMPLE: applying all methods in sequence
# =============================================================================


def impute_and_score_fire_data(
    confidence: np.ndarray,
    frp: np.ndarray,
    obs_valid: np.ndarray,
    cloud_mask: np.ndarray,
    confidence_threshold: float = 0.30,
    temporal_gap_hours: int = 3,
    frp_cap_percentile: float = 99.0,
    frp_physical_max: float = 2_000.0,
) -> dict[str, np.ndarray]:
    """Full imputation and quality scoring pipeline.

    Demonstrates the recommended sequence of operations:
    1. Apply physical bounds and outlier capping to FRP
    2. Temporal gap-filling for binary labels (morphological closing)
    3. Temporal interpolation for FRP (linear for short gaps)
    4. Compute quality scores
    5. Create training weight mask

    Args:
        confidence: (T, H, W) fire detection confidence [0, 1]
        frp: (T, H, W) fire radiative power in MW
        obs_valid: (T, H, W) observation validity mask
        cloud_mask: (T, H, W) cloud presence mask
        confidence_threshold: Binary label threshold
        temporal_gap_hours: Max gap to fill temporally
        frp_cap_percentile: Percentile for FRP winsorization
        frp_physical_max: Physical maximum FRP

    Returns:
        Dictionary of output arrays with metadata keys
    """
    _T, _H, _W = confidence.shape

    # Step 1: Compute usable mask (valid AND cloud-free)
    usable = obs_valid.astype(bool) & ~cloud_mask.astype(bool)

    # Step 2: Outlier detection and capping for FRP
    frp_capped = winsorize_array(
        frp,
        upper_pct=frp_cap_percentile,
        valid_mask=usable & (frp > 0),
        physical_bounds=(0.0, frp_physical_max),
    )
    frp_capped, frp_reliability = apply_physical_bounds_frp(frp_capped)

    # Step 3: Binary label creation with threshold
    binary = (confidence >= confidence_threshold).astype(np.float32)

    # Step 4: Temporal gap-filling for binary labels
    labels_closed = morphological_closing_temporal(binary, gap_hours=temporal_gap_hours)

    # Step 5: Temporal interpolation for FRP (short gaps only)
    frp_interp, frp_was_imputed = linear_interpolation_temporal(
        frp_capped, usable, max_gap=temporal_gap_hours * 2
    )

    # Step 6: Track all imputation
    label_was_imputed = (labels_closed != binary) & usable
    any_imputed = frp_was_imputed | label_was_imputed

    # Step 7: Compute quality scores
    quality = compute_pixel_quality_score(obs_valid, cloud_mask, confidence, confidence_threshold)

    # Step 8: Create training weights
    training_weights = create_training_weight_mask(quality, obs_valid, any_imputed)

    # Step 9: Compute completeness metrics
    timestep_completeness = compute_timestep_completeness(obs_valid, cloud_mask)
    spatial_completeness = compute_spatial_completeness(obs_valid, cloud_mask)

    return {
        # Primary outputs
        "labels": labels_closed,
        "frp": frp_interp,
        "validity": (obs_valid * (1.0 - cloud_mask)).astype(np.float32),
        # Quality and weight masks
        "quality_score": quality,
        "training_weights": training_weights,
        "frp_reliability": frp_reliability,
        # Imputation tracking
        "imputation_mask": any_imputed.astype(np.float32),
        "label_imputation_mask": label_was_imputed.astype(np.float32),
        "frp_imputation_mask": frp_was_imputed.astype(np.float32),
        # Completeness metrics
        "timestep_completeness": timestep_completeness,
        "spatial_completeness": spatial_completeness,
        # Raw data preserved for comparison
        "raw_confidence": confidence,
        "raw_frp": frp,
        "raw_binary": binary,
    }
