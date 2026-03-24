"""Data validation checks for the wildfire pipeline.

Validates data at each stage to catch corruption before it propagates
to ML training labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ValidationResult:
    """Result of running validation checks on pipeline data."""

    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False


def validate_download(
    confidence: np.ndarray,
    obs_valid: np.ndarray,
    cloud_mask: np.ndarray,
    frp: np.ndarray,
    failed_hours: list[int] | None = None,
) -> ValidationResult:
    """Validate raw download data before label processing.

    Checks:
    - Shape consistency across all arrays
    - No NaN or Inf values
    - Value ranges (confidence in [0,1], FRP >= 0)
    - Temporal completeness (not too many failed hours)
    - Spatial coverage (at least some valid observations)
    """
    result = ValidationResult()

    # Shape consistency
    shapes = [confidence.shape, obs_valid.shape, cloud_mask.shape, frp.shape]
    if len(set(shapes)) != 1:
        result.add_error(
            f"Shape mismatch: conf={confidence.shape}, valid={obs_valid.shape}, "
            f"cloud={cloud_mask.shape}, frp={frp.shape}"
        )
        return result  # Can't do further checks with mismatched shapes

    T, _H, _W = confidence.shape

    # NaN/Inf checks
    for name, arr in [
        ("confidence", confidence),
        ("obs_valid", obs_valid),
        ("cloud_mask", cloud_mask),
        ("frp", frp),
    ]:
        if np.isnan(arr).any():
            result.add_error(f"{name} contains NaN values")
        if np.isinf(arr).any():
            result.add_error(f"{name} contains Inf values")

    # Value range checks
    if confidence.min() < 0 or confidence.max() > 1:
        result.add_error(
            f"Confidence out of [0,1]: min={confidence.min():.4f}, max={confidence.max():.4f}"
        )
    if frp.min() < 0:
        result.add_error(f"FRP has negative values: min={frp.min():.4f}")
    if frp.max() > 10000:
        result.add_warning(f"FRP unusually high: max={frp.max():.1f} MW")

    # Temporal completeness
    if failed_hours:
        failure_rate = len(failed_hours) / T
        if failure_rate > 0.10:
            result.add_error(f"Too many failed hours: {len(failed_hours)}/{T} ({failure_rate:.1%})")
        elif failure_rate > 0.05:
            result.add_warning(f"Some hours failed: {len(failed_hours)}/{T} ({failure_rate:.1%})")

    # Spatial coverage
    valid_per_hour = obs_valid.sum(axis=(1, 2))
    fully_invalid_hours = int((valid_per_hour < 0.5).sum())
    if fully_invalid_hours > T * 0.5:
        result.add_error(f"Over 50% of hours have no valid observations: {fully_invalid_hours}/{T}")
    elif fully_invalid_hours > T * 0.2:
        result.add_warning(f"{fully_invalid_hours}/{T} hours have no valid observations")

    # Consecutive gap length — long gaps degrade temporal coherence
    if T > 1:
        invalid = (obs_valid < 0.5).astype(np.int32)
        max_gap = 0
        current_gap = np.zeros((_H, _W), dtype=np.int32)
        for t in range(T):
            current_gap = np.where(invalid[t] > 0, current_gap + 1, 0)
            frame_max = int(current_gap.max())
            if frame_max > max_gap:
                max_gap = frame_max
        if max_gap > 12:
            result.add_warning(f"Longest consecutive invalid gap: {max_gap} hours")
        elif max_gap > 6:
            result.add_warning(f"Notable consecutive gap: {max_gap} hours")

    # FRP distribution — detect anomalies (saturation, industrial sources)
    fire_frp = frp[confidence > 0]
    if len(fire_frp) > 10:
        p99 = float(np.percentile(fire_frp, 99))
        if p99 > 5000:
            result.add_warning(f"FRP p99={p99:.0f} MW exceeds sensor saturation threshold")
        n_saturated = int((frp > 5000).sum())
        if n_saturated > 0:
            result.add_warning(f"{n_saturated} pixels with FRP above saturation (5000 MW)")

    # Confidence-FRP cross-check
    suspicious = int(((frp > 0) & (confidence == 0)).sum())
    if suspicious > 0:
        result.add_warning(f"{suspicious} pixels have FRP>0 but confidence=0 (suspicious)")

    return result


def validate_labels(
    labels: np.ndarray,
    validity: np.ndarray,
    raw_confidence: np.ndarray,
) -> ValidationResult:
    """Validate processed labels before use in ML training.

    Checks:
    - Shape consistency
    - Labels are binary (0 or 1)
    - Validity is in [0, 1]
    - Fire fraction sanity (<50% of AOI per timestep)
    - Cloud exclusion fraction not too high
    - Temporal coherence (cumulative area roughly non-decreasing)
    """
    result = ValidationResult()

    # Shape consistency
    if labels.shape != validity.shape:
        result.add_error(f"Label/validity shape mismatch: {labels.shape} vs {validity.shape}")
        return result
    if labels.shape != raw_confidence.shape:
        result.add_error(
            f"Label/confidence shape mismatch: {labels.shape} vs {raw_confidence.shape}"
        )
        return result

    T, H, W = labels.shape

    # Binary check
    unique_labels = np.unique(labels)
    if not np.all(np.isin(unique_labels, [0.0, 1.0])):
        result.add_error(f"Labels not binary, found values: {unique_labels}")

    # Validity range
    if validity.min() < 0 or validity.max() > 1:
        result.add_error(
            f"Validity out of [0,1]: min={validity.min():.4f}, max={validity.max():.4f}"
        )

    # Fire fraction sanity per timestep
    fire_fraction_per_t = labels.sum(axis=(1, 2)) / (H * W)
    max_fire_fraction = fire_fraction_per_t.max()
    if max_fire_fraction > 0.5:
        result.add_warning(
            f"Fire fraction exceeds 50% at some timestep: max={max_fire_fraction:.1%}"
        )

    # At least some fire detected
    total_fire_pixels = labels.sum()
    if total_fire_pixels == 0:
        result.add_warning("No fire pixels detected in entire dataset")

    # Cloud exclusion fraction
    cloud_excluded = 1.0 - validity.mean()
    if cloud_excluded > 0.6:
        result.add_warning(f"Over 60% of pixels excluded (cloud/invalid): {cloud_excluded:.1%}")

    # Flicker rate
    if T > 1:
        flicker_count = int(((labels[:-1] == 1) & (labels[1:] == 0)).sum())
        total_fire = int(labels.sum())
        flicker_rate = flicker_count / max(total_fire, 1)
        if flicker_rate > 0.3:
            result.add_warning(f"High flicker rate after smoothing: {flicker_rate:.1%}")

    # Temporal monotonicity — fire area should generally not shrink dramatically.
    # A large single-timestep drop (>50% of peak) suggests label noise or data corruption.
    # Small drops are normal (cloud masking, edge effects), so we only flag dramatic ones.
    if T > 2 and total_fire_pixels > 0:
        fire_area_per_t = labels.sum(axis=(1, 2))
        # Only check timesteps with valid observations (ignore fully clouded hours)
        valid_per_t = validity.sum(axis=(1, 2))
        has_valid = valid_per_t > 0

        # Compute cumulative max fire area (only over valid hours)
        peak_area = 0.0
        max_drop_frac = 0.0
        drop_timestep = -1
        for t in range(T):
            if not has_valid[t]:
                continue
            area = float(fire_area_per_t[t])
            if area > peak_area:
                peak_area = area
            elif peak_area > 0:
                drop_frac = (peak_area - area) / peak_area
                if drop_frac > max_drop_frac:
                    max_drop_frac = drop_frac
                    drop_timestep = t

        if max_drop_frac > 0.5:
            result.add_warning(
                f"Fire area dropped {max_drop_frac:.0%} from peak at timestep {drop_timestep} "
                f"(possible label noise or data gap)"
            )

    return result
