"""Data quality functions for wildfire detection arrays.

Provides imputation, outlier detection, quality scoring, and spatial/temporal
consistency checks grounded in GOES/VIIRS fire detection physics.

Physical bounds reference:
- FRP detection floor: ~35 MW (GOES ABI minimum detectable)
- FRP practical ceiling: ~2000 MW (extreme mega-fire)
- FRP sensor saturation: ~5000 MW (ABI 3.9um channel)
- Fire temperature: 400-2000 K (sub-pixel retrieval range)
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# ---------------------------------------------------------------------------
# Physical constants for GOES fire detection
# ---------------------------------------------------------------------------

FRP_DETECTION_FLOOR_MW = 35.0
FRP_PRACTICAL_CEILING_MW = 2000.0
FRP_SATURATION_MW = 5000.0
FIRE_TEMP_MIN_K = 400.0
FIRE_TEMP_MAX_K = 2000.0


# ---------------------------------------------------------------------------
# 1. Cloud-aware persistence
# ---------------------------------------------------------------------------


def cloud_aware_persistence(
    binary: np.ndarray,
    validity: np.ndarray,
    max_gap_hours: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-fill fire state through cloud/invalid gaps.

    If a pixel was fire at time t and invalid at t+1..t+k (where k <= max_gap_hours),
    propagate the fire label through the gap. This is physically motivated:
    fires rarely self-extinguish in 1-3 hours.

    Only fills gaps — does not create new fire detections. The filled pixels
    should be weighted lower in training loss (see compute_quality_weights).

    Args:
        binary: (T, H, W) binary fire array (after thresholding)
        validity: (T, H, W) validity mask (1=valid observation, 0=cloud/invalid)
        max_gap_hours: maximum gap length to fill (default 3)

    Returns:
        Tuple of (filled_binary, was_imputed) where was_imputed is a boolean mask
        indicating which pixels were forward-filled.
    """
    T, H, W = binary.shape
    filled = binary.copy()
    was_imputed = np.zeros_like(binary, dtype=np.float32)

    for t in range(1, T):
        # Pixels that are: currently invalid AND were fire in previous step
        gap_mask = (validity[t] == 0) & (filled[t - 1] == 1)
        filled[t] = np.where(gap_mask, 1.0, filled[t])
        was_imputed[t] = np.where(gap_mask, 1.0, was_imputed[t])

    # Enforce max gap: if a pixel has been imputed for more than max_gap_hours
    # consecutive steps, stop imputing
    if max_gap_hours < T:
        consecutive_imputed = np.zeros((H, W), dtype=np.int32)
        for t in range(T):
            is_imp = was_imputed[t] > 0
            consecutive_imputed = np.where(is_imp, consecutive_imputed + 1, 0)
            # Clear imputation where gap exceeds limit
            too_long = consecutive_imputed > max_gap_hours
            filled[t] = np.where(too_long, binary[t], filled[t])
            was_imputed[t] = np.where(too_long, 0.0, was_imputed[t])

    return filled, was_imputed


# ---------------------------------------------------------------------------
# 2. Isolated pixel filtering
# ---------------------------------------------------------------------------


def filter_isolated_pixels(
    binary: np.ndarray,
    min_spatial_neighbors: int = 1,
    require_temporal_support: bool = True,
) -> np.ndarray:
    """Remove spatially and temporally isolated fire detections.

    A fire pixel is considered isolated if it has fewer than
    min_spatial_neighbors fire neighbors in the same frame AND no fire
    at the same location in adjacent hours. Such detections are likely
    false positives from sun glint, hot surfaces, or sensor noise.

    Args:
        binary: (T, H, W) binary fire array
        min_spatial_neighbors: minimum 4-connected fire neighbors to keep (default 1)
        require_temporal_support: if True, also require fire at same pixel in t-1 (backward only)

    Returns:
        Filtered binary array with isolated pixels removed.
    """
    T, H, W = binary.shape
    filtered = binary.copy()

    # Skip filtering for grids too small for spatial neighbors to exist
    if H <= 1 or W <= 1:
        return filtered

    # 4-connected spatial structuring element (no diagonal)
    spatial_kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)

    for t in range(T):
        frame = binary[t]
        if frame.sum() == 0:
            continue

        # Count spatial neighbors per pixel
        neighbor_count = ndimage.convolve(frame, spatial_kernel, mode="constant", cval=0)

        # Spatially isolated: fire pixel with too few neighbors
        spatially_isolated = (frame == 1) & (neighbor_count < min_spatial_neighbors)

        if require_temporal_support and spatially_isolated.any():
            # Check temporal support: fire at same pixel in t-1 only.
            # Never use binary[t+1] — that would leak future info into labels.
            has_temporal = np.zeros((H, W), dtype=bool)
            if t > 0:
                has_temporal |= binary[t - 1].astype(bool)

            # Remove only if BOTH spatially AND temporally isolated
            remove = spatially_isolated & ~has_temporal
        else:
            remove = spatially_isolated

        filtered[t] = np.where(remove, 0.0, filtered[t])

    return filtered


# ---------------------------------------------------------------------------
# 3. FRP outlier detection and capping
# ---------------------------------------------------------------------------


def detect_frp_outliers(
    frp: np.ndarray,
    confidence: np.ndarray | None = None,
    cap_percentile: float = 99.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect and cap FRP outliers using physical bounds and percentile capping.

    Returns capped FRP values and a reliability mask:
    - 1.0: high reliability (FRP in normal range)
    - 0.5: moderate (above practical ceiling but below saturation)
    - 0.0: unreliable (above saturation threshold or negative)

    Args:
        frp: (T, H, W) fire radiative power in MW
        confidence: optional (T, H, W) confidence array to cross-check
        cap_percentile: percentile for winsorization (default 99.5)

    Returns:
        Tuple of (capped_frp, frp_reliability)
    """
    capped = frp.copy()
    reliability = np.ones_like(frp)

    # Physical floor: FRP should not be negative
    capped = np.maximum(capped, 0.0)
    reliability = np.where(frp < 0, 0.0, reliability)

    # Fixed physical cap — avoids temporal dependency from data-driven percentile
    cap_value = FRP_PRACTICAL_CEILING_MW  # 2000.0 MW
    capped = np.minimum(capped, cap_value)

    # Reliability scoring based on physical ranges
    # Above practical ceiling: reduced reliability
    reliability = np.where(frp > FRP_PRACTICAL_CEILING_MW, 0.5, reliability)
    # Above saturation: unreliable
    reliability = np.where(frp > FRP_SATURATION_MW, 0.0, reliability)

    # Cross-check: FRP > 0 but confidence == 0 is suspicious
    if confidence is not None:
        suspicious = (frp > 0) & (confidence == 0)
        reliability = np.where(suspicious, 0.0, reliability)

    return capped.astype(np.float32), reliability.astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Per-pixel quality scoring and training weights
# ---------------------------------------------------------------------------


def compute_quality_weights(
    validity: np.ndarray,
    was_imputed: np.ndarray | None = None,
    frp_reliability: np.ndarray | None = None,
    imputation_weight: float = 0.3,
) -> np.ndarray:
    """Compute per-pixel training loss weights based on data quality.

    Produces a weight array for use in masked training loss:
        loss = sum(weight * pixel_loss) / sum(weight)

    Weight levels:
    - 1.0: high-quality observed data (valid, not imputed)
    - imputation_weight: data was forward-filled through a cloud gap
    - 0.0: missing/invalid data (excluded from loss)

    Args:
        validity: (T, H, W) validity mask (1=valid, 0=invalid)
        was_imputed: optional (T, H, W) imputation mask from cloud_aware_persistence
        frp_reliability: optional (T, H, W) reliability from detect_frp_outliers
        imputation_weight: weight for imputed pixels (default 0.3)

    Returns:
        (T, H, W) weight array in [0, 1]
    """
    weights = validity.copy().astype(np.float32)

    # Imputed pixels should get imputation_weight even though validity=0
    # (they were forward-filled through cloud gaps — we have some confidence)
    if was_imputed is not None:
        weights = np.where(was_imputed > 0, imputation_weight, weights)

    # Further reduce weight where FRP is unreliable
    if frp_reliability is not None:
        weights = weights * np.maximum(frp_reliability, 0.3)

    return weights.astype(np.float32)


# ---------------------------------------------------------------------------
# 5. Consecutive gap tracking
# ---------------------------------------------------------------------------


def compute_gap_stats(validity: np.ndarray) -> dict[str, np.ndarray | float]:
    """Compute temporal gap statistics from validity mask.

    Returns metrics about cloud/invalid gaps useful for data quality assessment.

    Args:
        validity: (T, H, W) validity mask

    Returns:
        Dict with:
        - max_gap_per_pixel: (H, W) maximum consecutive invalid hours per pixel
        - mean_gap_length: scalar, mean gap length across all pixels
        - max_gap_overall: scalar, longest gap anywhere in the data
        - gap_fraction: scalar, fraction of (pixel, time) pairs that are invalid
    """
    T, H, W = validity.shape
    invalid = (validity < 0.5).astype(np.int32)

    # Compute max consecutive gap per pixel
    max_gap_per_pixel = np.zeros((H, W), dtype=np.int32)
    current_gap = np.zeros((H, W), dtype=np.int32)

    for t in range(T):
        is_invalid = invalid[t] > 0
        current_gap = np.where(is_invalid, current_gap + 1, 0)
        max_gap_per_pixel = np.maximum(max_gap_per_pixel, current_gap)

    # Vectorized mean gap computation using diff-based boundary detection.
    # Reshape to (T, N) where N = H*W, then detect gap start/end transitions.
    flat = invalid.reshape(T, -1)  # (T, N)
    N = flat.shape[1]

    # Pad with zeros at boundaries so gaps at start/end are captured
    padded = np.vstack([np.zeros((1, N), dtype=np.int32), flat, np.zeros((1, N), dtype=np.int32)])
    # Diff along time axis: +1 = gap starts, -1 = gap ends
    transitions = np.diff(padded, axis=0)  # (T+1, N)

    # For each pixel column, count gaps and sum their lengths
    gap_sum = 0
    gap_count = 0
    for col in range(N):
        starts = np.where(transitions[:, col] == 1)[0]
        ends = np.where(transitions[:, col] == -1)[0]
        if len(starts) > 0 and len(ends) > 0:
            lengths = ends[: len(starts)] - starts[: len(ends)]
            gap_sum += int(lengths.sum())
            gap_count += len(lengths)

    mean_gap = gap_sum / gap_count if gap_count > 0 else 0.0
    gap_fraction = float(invalid.sum()) / (T * H * W)

    return {
        "max_gap_per_pixel": max_gap_per_pixel,
        "mean_gap_length": mean_gap,
        "max_gap_overall": int(max_gap_per_pixel.max()),
        "gap_fraction": gap_fraction,
    }


# ---------------------------------------------------------------------------
# 6. Normalization statistics for ML training
# ---------------------------------------------------------------------------


def compute_normalization_stats(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Compute per-channel normalization statistics for ML training.

    Returns a dict mapping channel name to {mean, std, min, max, p1, p99}.
    """
    stats: dict[str, dict[str, float]] = {}
    for name, arr in arrays.items():
        flat = arr.flatten().astype(np.float64)
        if len(flat) == 0:
            continue
        stats[name] = {
            "mean": round(float(flat.mean()), 6),
            "std": round(float(flat.std()), 6),
            "min": round(float(flat.min()), 6),
            "max": round(float(flat.max()), 6),
            "p1": round(float(np.percentile(flat, 1)), 6),
            "p99": round(float(np.percentile(flat, 99)), 6),
        }
    return stats


# ---------------------------------------------------------------------------
# 7. Derived spatial features for fire spread prediction
# ---------------------------------------------------------------------------


def compute_distance_to_fire(labels: np.ndarray) -> np.ndarray:
    """Compute per-pixel Euclidean distance to nearest fire pixel per timestep.

    For fire pixels, distance is 0. For non-fire pixels, distance is in pixel units.
    Useful as a model input feature — fire spreads outward from its edge.

    Args:
        labels: (T, H, W) binary fire mask

    Returns:
        (T, H, W) float32 array of distances
    """
    from scipy.ndimage import distance_transform_edt

    T = labels.shape[0]
    distances = np.zeros_like(labels, dtype=np.float32)
    for t in range(T):
        if labels[t].sum() > 0:
            # Distance from non-fire to nearest fire pixel
            distances[t] = distance_transform_edt(labels[t] == 0).astype(np.float32)
        else:
            # No fire this timestep — set to a large sentinel
            distances[t] = -1.0  # -1 indicates no fire reference
    return distances


def compute_fire_neighborhood(labels: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Compute fraction of fire pixels in a local neighborhood per timestep.

    Args:
        labels: (T, H, W) binary fire mask
        kernel_size: size of the square averaging window (default 5)

    Returns:
        (T, H, W) float32 array with values in [0, 1]
    """
    from scipy.ndimage import uniform_filter

    result = np.zeros_like(labels, dtype=np.float32)
    for t in range(labels.shape[0]):
        result[t] = uniform_filter(labels[t].astype(np.float64), size=kernel_size).astype(
            np.float32
        )
    return result
