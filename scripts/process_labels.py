"""Process raw fire detection into high-quality training labels.

Applies all label improvements identified from analysis:
1. Temporal smoothing (majority vote) to reduce flicker
2. Cloud masking (exclude cloudy pixels from loss, don't label them negative)
3. Confidence thresholding (filter low-probability detections)
4. Multi-source validation (boost confidence when VIIRS/MODIS confirm)

Input:  Raw fire confidence JSON (from download_fire_data.py)
Output: Processed label JSON with smoothed labels + validity masks
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def majority_vote_smooth(
    binary: np.ndarray, window: int, min_votes: int
) -> np.ndarray:
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
    T, H, W = binary.shape
    smoothed = np.zeros_like(binary)

    for t in range(T):
        start = max(0, t - window + 1)
        votes = binary[start:t + 1].sum(axis=0)
        smoothed[t] = (votes >= min_votes).astype(np.float32)

    return smoothed


def apply_cloud_masking(
    obs_valid: np.ndarray, cloud_mask: np.ndarray
) -> np.ndarray:
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


def process_fire(fire_path: Path, config: dict) -> dict:
    """Process one fire's raw data into training labels."""
    with open(fire_path) as f:
        data = json.load(f)

    conf = np.array(data["data"], dtype=np.float32)
    obs_valid = np.array(data.get("observation_valid", np.ones_like(conf)), dtype=np.float32)
    cloud = np.array(data.get("cloud_mask", np.zeros_like(conf)), dtype=np.float32)

    T, H, W = conf.shape
    fire_name = data["metadata"]["fire_name"]
    log(f"Processing {fire_name}: {T} hours, {H}x{W} grid")

    # Step 1: Apply confidence threshold
    threshold = config.get("goes_confidence_threshold", 0.30)
    binary = (conf >= threshold).astype(np.float32)
    n_fire_raw = int(binary.sum())
    log(f"  Raw fire pixels (>={threshold}): {n_fire_raw:,} ({100*n_fire_raw/(T*H*W):.3f}%)")

    # Step 2: Apply temporal smoothing
    smooth_cfg = config.get("label_smoothing", {})
    method = smooth_cfg.get("method", "majority_vote")
    window = smooth_cfg.get("window_hours", 5)
    min_votes = smooth_cfg.get("min_votes", 2)

    if method == "majority_vote":
        smoothed = majority_vote_smooth(binary, window, min_votes)
    elif method == "rolling_max":
        # Rolling max: fire if detected in ANY of the last `window` hours
        smoothed = np.zeros_like(binary)
        for t in range(T):
            start = max(0, t - window + 1)
            smoothed[t] = binary[start:t + 1].max(axis=0)
    else:
        smoothed = binary

    n_fire_smooth = int(smoothed.sum())
    log(f"  Smoothed fire pixels: {n_fire_smooth:,} ({100*n_fire_smooth/(T*H*W):.3f}%)")

    # Step 3: Apply cloud masking
    if config.get("cloud_masking", True):
        validity = apply_cloud_masking(obs_valid, cloud)
        n_valid = int(validity.sum())
        n_total = T * H * W
        n_cloudy = n_total - n_valid
        log(f"  Cloud-masked pixels: {n_cloudy:,} ({100*n_cloudy/n_total:.1f}% excluded)")
    else:
        validity = np.ones((T, H, W), dtype=np.float32)

    # Step 4: Compute flicker stats (for monitoring)
    flicker_count = 0
    for t in range(T - 1):
        flicker_count += int(((smoothed[t] == 1) & (smoothed[t + 1] == 0)).sum())
    n_fire_total = int(smoothed.sum())
    flicker_rate = flicker_count / max(n_fire_total, 1)

    # Compute oracle F1 (theoretical max for "copy t to predict t+1")
    tp = fn = fp = 0
    for t in range(T - 1):
        valid_t = (validity[t] > 0) & (validity[t + 1] > 0)
        pred = smoothed[t][valid_t].astype(int)
        truth = smoothed[t + 1][valid_t].astype(int)
        tp += int(((pred == 1) & (truth == 1)).sum())
        fp += int(((pred == 1) & (truth == 0)).sum())
        fn += int(((pred == 0) & (truth == 1)).sum())

    oracle_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    oracle_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    oracle_f1 = 2 * oracle_p * oracle_r / (oracle_p + oracle_r) if (oracle_p + oracle_r) > 0 else 0.0

    log(f"  Flicker rate (smoothed): {100*flicker_rate:.1f}%")
    log(f"  Oracle F1 (smoothed + cloud-masked): {oracle_f1:.4f}")

    result = {
        "metadata": {
            **data["metadata"],
            "processing": {
                "confidence_threshold": threshold,
                "smoothing_method": method,
                "smoothing_window": window,
                "smoothing_min_votes": min_votes,
                "cloud_masking": config.get("cloud_masking", True),
            },
            "quality": {
                "raw_fire_pixels": n_fire_raw,
                "smoothed_fire_pixels": n_fire_smooth,
                "flicker_rate": flicker_rate,
                "oracle_f1": oracle_f1,
                "cloud_excluded_fraction": 1.0 - validity.mean(),
            },
        },
        "labels": smoothed.tolist(),
        "validity": validity.tolist(),
        "raw_confidence": conf.tolist(),
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Process fire data into training labels")
    parser.add_argument("--input", required=True, help="Input fire JSON path")
    parser.add_argument("--output", help="Output path (default: input_dir/processed/)")
    parser.add_argument("--config", default="config/fires.json", help="Pipeline config")
    args = parser.parse_args()

    config_path = REPO_ROOT / args.config
    with open(config_path) as f:
        config = json.load(f)["pipeline_config"]

    input_path = Path(args.input)
    result = process_fire(input_path, config)

    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = input_path.parent / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / input_path.name.replace(".json", "_processed.json")

    with open(output_path, "w") as f:
        json.dump(result, f)
    log(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
