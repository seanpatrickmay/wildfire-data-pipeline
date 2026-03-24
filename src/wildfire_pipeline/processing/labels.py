"""Process raw fire detection into high-quality training labels.

Processing pipeline:
1. Confidence thresholding (filter low-probability detections)
2. Cloud-aware persistence (forward-fill fire through cloud gaps)
3. Isolated pixel filtering (remove spatially+temporally isolated false positives)
4. Temporal smoothing (majority vote to reduce flicker)
5. Cloud masking (exclude cloudy pixels from loss, don't label them negative)
6. FRP outlier detection and quality scoring
7. Quality metrics computation (flicker rate, oracle F1, gap stats)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

from wildfire_pipeline.config import PipelineConfig
from wildfire_pipeline.logging import get_logger
from wildfire_pipeline.processing.io import load_fire_data, save_fire_data
from wildfire_pipeline.processing.quality import (
    cloud_aware_persistence,
    compute_distance_to_fire,
    compute_fire_neighborhood,
    compute_gap_stats,
    compute_quality_weights,
    detect_frp_outliers,
    filter_isolated_pixels,
)

logger = get_logger()


def _ensure_pipeline_config(config: PipelineConfig | dict) -> PipelineConfig:
    """Accept either a PipelineConfig or a dict, returning a validated model."""
    if isinstance(config, PipelineConfig):
        return config
    return PipelineConfig.model_validate(config)


def majority_vote_smooth(binary: np.ndarray, window: int, min_votes: int) -> np.ndarray:
    """Temporal majority voting.

    A pixel is fire at time t if it was detected as fire in >= min_votes
    of the hours [t-window+1, ..., t]. This fills 1-3 hour gaps caused
    by satellite detection flickering (clouds, scan angle, sensor noise).

    Args:
        binary: (T, H, W) binary fire array
        window: lookback window in hours
        min_votes: minimum detections in window to count as fire

    Returns:
        (T, H, W) smoothed binary array
    """
    T = binary.shape[0]

    # Vectorized sliding window via cumulative sum along the time axis.
    # cumsum[t] = binary[0] + ... + binary[t]
    # votes[t] = cumsum[t] - cumsum[t - window]  (clamped to 0)
    cumsum = np.cumsum(binary, axis=0)  # (T, H, W)

    # For t < window, the lookback is [0..t], so votes = cumsum[t].
    # For t >= window, votes = cumsum[t] - cumsum[t - window].
    shifted = np.zeros_like(cumsum)
    if window < T:
        shifted[window:] = cumsum[: T - window]
    votes = cumsum - shifted

    result: np.ndarray = (votes >= min_votes).astype(np.float32)
    return result


def apply_cloud_masking(obs_valid: np.ndarray, cloud_mask: np.ndarray) -> np.ndarray:
    """Create combined validity mask.

    A pixel is valid for training if:
    - The satellite had a valid observation (DQF <= 1)
    - The pixel was NOT cloud-covered (or fire was detected through cloud)

    Invalid pixels should be EXCLUDED from loss computation during training,
    NOT labeled as "no fire". This is the key improvement over GOFER.

    Returns:
        (T, H, W) validity mask (1 = use for training, 0 = exclude)
    """
    # Valid = observation exists AND not obscured by cloud
    return (obs_valid * (1.0 - cloud_mask)).astype(np.float32)


def process_fire(
    fire_path: Path, config: PipelineConfig | dict, fmt: str = "npz"
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Process one fire's raw data into training labels.

    Args:
        fire_path: Path to raw fire data file (npz, zarr, or json).
        config: Pipeline config — either a PipelineConfig model or a dict
                (dicts are auto-validated via Pydantic).
        fmt: Output format for saved processed data.

    Returns:
        Tuple of (output arrays dict, output metadata dict).
    """
    cfg = _ensure_pipeline_config(config)

    arrays, input_metadata = load_fire_data(fire_path)

    conf = arrays["data"].astype(np.float32)
    obs_valid = arrays.get("observation_valid", np.ones_like(conf)).astype(np.float32)
    cloud = arrays.get("cloud_mask", np.zeros_like(conf)).astype(np.float32)
    frp = arrays.get("frp", np.zeros_like(conf)).astype(np.float32)

    T, H, W = conf.shape
    fire_name = input_metadata.get("fire_name", fire_path.stem)
    logger.info("processing_fire", fire_name=fire_name, hours=T, height=H, width=W)

    # Step 1: Apply confidence threshold
    threshold = cfg.goes_confidence_threshold
    binary = (conf >= threshold).astype(np.float32)
    n_fire_raw = int(binary.sum())
    logger.info(
        "raw_fire_pixels",
        count=n_fire_raw,
        threshold=threshold,
        pct=round(100 * n_fire_raw / (T * H * W), 3),
    )

    # Step 2: Cloud-aware persistence — forward-fill fire through cloud gaps
    if cfg.cloud_masking:
        validity_raw = apply_cloud_masking(obs_valid, cloud)
        binary, was_imputed = cloud_aware_persistence(
            binary, validity_raw, max_gap_hours=cfg.max_persistence_gap_hours
        )
        n_imputed = int(was_imputed.sum())
        if n_imputed > 0:
            logger.info(
                "cloud_persistence_fill",
                imputed_pixels=n_imputed,
                max_gap_hours=cfg.max_persistence_gap_hours,
            )
    else:
        validity_raw = np.ones((T, H, W), dtype=np.float32)
        was_imputed = np.zeros_like(binary)

    # Step 3: Filter isolated false positives
    n_before_filter = int(binary.sum())
    binary = filter_isolated_pixels(binary, min_spatial_neighbors=1, require_temporal_support=True)
    n_removed = n_before_filter - int(binary.sum())
    if n_removed > 0:
        logger.info("isolated_pixels_removed", count=n_removed)

    # Step 4: Apply temporal smoothing
    smooth = cfg.label_smoothing
    if smooth.method == "majority_vote":
        smoothed = majority_vote_smooth(binary, smooth.window_hours, smooth.min_votes)
    elif smooth.method == "rolling_max":
        smoothed = np.zeros_like(binary)
        for t in range(T):
            start = max(0, t - smooth.window_hours + 1)
            smoothed[t] = binary[start : t + 1].max(axis=0)
    else:
        smoothed = binary

    # Soft labels: running mean of confidence over the smoothing window
    # Provides continuous [0,1] targets as alternative to binary labels
    if T > 1:
        conf_cumsum = np.cumsum(conf, axis=0)
        shifted_conf = np.zeros_like(conf_cumsum)
        win = smooth.window_hours
        if win < T:
            shifted_conf[win:] = conf_cumsum[: T - win]
        window_sizes = np.minimum(np.arange(1, T + 1, dtype=np.float32), float(win))[
            :, np.newaxis, np.newaxis
        ]
        soft_labels: np.ndarray = np.clip(
            (conf_cumsum - shifted_conf) / window_sizes, 0.0, 1.0
        ).astype(np.float32)
    else:
        soft_labels = conf.copy()

    n_fire_smooth = int(smoothed.sum())
    logger.info(
        "smoothed_fire_pixels",
        count=n_fire_smooth,
        pct=round(100 * n_fire_smooth / (T * H * W), 3),
    )

    # Step 5: Apply cloud masking for final validity
    if cfg.cloud_masking:
        validity = validity_raw
        n_valid = int(validity.sum())
        n_total = T * H * W
        n_cloudy = n_total - n_valid
        logger.info(
            "cloud_masked_pixels",
            excluded=n_cloudy,
            total=n_total,
            pct_excluded=round(100 * n_cloudy / n_total, 1),
        )
    else:
        validity = np.ones((T, H, W), dtype=np.float32)

    # Step 6: FRP outlier detection and quality scoring
    capped_frp, frp_reliability = detect_frp_outliers(frp, confidence=conf)
    quality_weights = compute_quality_weights(
        validity=validity,
        was_imputed=was_imputed,
        frp_reliability=frp_reliability,
        imputation_weight=cfg.imputation_weight,
    )

    # Step 7: Compute quality metrics (vectorized — no Python loops)
    n_fire_total = int(smoothed.sum())
    if T > 1:
        # Flicker: fire at t, no fire at t+1
        flicker_count = int(((smoothed[:-1] == 1) & (smoothed[1:] == 0)).sum())

        # Oracle F1: how well does "copy t to predict t+1" work?
        valid_pairs = (validity[:-1] > 0) & (validity[1:] > 0)
        pred_fire = (smoothed[:-1] == 1) & valid_pairs
        true_fire = (smoothed[1:] == 1) & valid_pairs
        tp = int((pred_fire & true_fire).sum())
        fp = int((pred_fire & ~true_fire).sum())
        fn = int((~pred_fire & true_fire).sum())
    else:
        flicker_count = 0
        tp = fp = fn = 0
    flicker_rate = flicker_count / max(n_fire_total, 1)

    oracle_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    oracle_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    oracle_f1 = (
        2 * oracle_p * oracle_r / (oracle_p + oracle_r) if (oracle_p + oracle_r) > 0 else 0.0
    )

    gap_stats = compute_gap_stats(validity)

    fire_frp = capped_frp[smoothed > 0]
    frp_stats: dict[str, Any] = {}
    if len(fire_frp) > 0:
        frp_stats = {
            "frp_mean": round(float(fire_frp.mean()), 2),
            "frp_median": round(float(np.median(fire_frp)), 2),
            "frp_p95": round(float(np.percentile(fire_frp, 95)), 2),
            "frp_p99": round(float(np.percentile(fire_frp, 99)), 2),
            "frp_max": round(float(fire_frp.max()), 2),
            "frp_saturated_pixels": int((frp_reliability == 0).sum()),
        }

    logger.info(
        "quality_metrics",
        flicker_rate=round(flicker_rate, 4),
        oracle_f1=round(oracle_f1, 4),
        max_gap_hours=gap_stats["max_gap_overall"],
        isolated_removed=n_removed,
        imputed_pixels=int(was_imputed.sum()),
        **frp_stats,
    )

    # Derived spatial features (for model input, not labels)
    distance_to_fire = compute_distance_to_fire(smoothed)
    fire_neighborhood = compute_fire_neighborhood(smoothed, kernel_size=5)

    # Class balance for training loss weighting
    valid_mask = validity > 0
    n_valid_pixels = int(valid_mask.sum())
    n_fire_in_valid = int((smoothed[valid_mask] == 1).sum()) if n_valid_pixels > 0 else 0
    fire_fraction = n_fire_in_valid / max(n_valid_pixels, 1)

    out_arrays = {
        "labels": smoothed,
        "soft_labels": soft_labels,
        "validity": validity,
        "quality_weights": quality_weights,
        "raw_confidence": conf,
        "capped_frp": capped_frp,
        "frp_reliability": frp_reliability,
        "was_imputed": was_imputed,
        "distance_to_fire": distance_to_fire,
        "fire_neighborhood": fire_neighborhood,
    }
    out_metadata: dict[str, Any] = {
        **input_metadata,
        "processing": {
            "confidence_threshold": threshold,
            "smoothing_method": smooth.method,
            "smoothing_window": smooth.window_hours,
            "smoothing_min_votes": smooth.min_votes,
            "cloud_masking": cfg.cloud_masking,
            "cloud_persistence_max_gap": cfg.max_persistence_gap_hours,
            "isolated_pixel_filtering": True,
            "frp_outlier_detection": True,
            "imputation_weight": cfg.imputation_weight,
        },
        "quality": {
            "raw_fire_pixels": n_fire_raw,
            "smoothed_fire_pixels": n_fire_smooth,
            "isolated_pixels_removed": n_removed,
            "imputed_pixels": int(was_imputed.sum()),
            "flicker_rate": flicker_rate,
            "oracle_f1": oracle_f1,
            "cloud_excluded_fraction": float(1.0 - validity.mean()),
            "max_gap_hours": gap_stats["max_gap_overall"],
            "mean_gap_length": gap_stats["mean_gap_length"],
            "gap_fraction": gap_stats["gap_fraction"],
            "fire_fraction": fire_fraction,
            "pos_weight": round(1.0 / max(fire_fraction, 0.001), 2),
            **frp_stats,
        },
    }

    # Save output
    output_dir = fire_path.parent / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = fire_path.stem.removesuffix(".zarr")
    output_path = output_dir / f"{stem}_processed"
    actual_path = save_fire_data(output_path, out_arrays, out_metadata, fmt=fmt)
    logger.info("saved", path=str(actual_path))

    return out_arrays, out_metadata
