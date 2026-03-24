"""Shared test fixtures for the wildfire pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_fire_arrays():
    """Create synthetic fire detection arrays for testing."""
    T, H, W = 10, 5, 5
    rng = np.random.default_rng(42)

    # Create realistic-ish fire data: fire pixels in center, spreading
    confidence = np.zeros((T, H, W), dtype=np.float32)
    for t in range(T):
        # Fire grows from center
        radius = min(t + 1, 2)
        center = H // 2, W // 2
        for i in range(max(0, center[0] - radius), min(H, center[0] + radius + 1)):
            for j in range(max(0, center[1] - radius), min(W, center[1] + radius + 1)):
                confidence[t, i, j] = rng.uniform(0.3, 1.0)

    obs_valid = np.ones((T, H, W), dtype=np.float32)
    # Add some cloudy hours
    obs_valid[3] = 0.0  # Hour 3 fully cloudy
    obs_valid[7, :2, :] = 0.0  # Hour 7 partially cloudy

    cloud_mask = (1.0 - obs_valid).astype(np.float32)
    frp = (confidence * rng.uniform(0, 100, size=(T, H, W))).astype(np.float32)

    return {
        "data": confidence,
        "observation_valid": obs_valid,
        "cloud_mask": cloud_mask,
        "frp": frp,
    }


@pytest.fixture
def sample_metadata():
    """Create sample pipeline metadata."""
    return {
        "fire_name": "TestFire",
        "year": 2023,
        "start_utc": "2023-08-01T00:00:00Z",
        "n_hours": 10,
        "grid_shape": [5, 5],
        "aoi": [-122.0, 38.0, -121.0, 39.0],
        "pipeline": "wildfire-data-pipeline",
        "cloud_masking": True,
        "multi_source_fusion": True,
        "failed_hours": [],
        "failure_rate": 0.0,
    }


@pytest.fixture
def pipeline_config():
    """Default pipeline config dict (matching fires.json structure)."""
    return {
        "export_scale_m": 2004,
        "export_crs": "EPSG:3857",
        "goes_confidence_threshold": 0.30,
        "label_smoothing": {
            "method": "majority_vote",
            "window_hours": 5,
            "min_votes": 2,
        },
        "cloud_masking": True,
        "max_persistence_gap_hours": 3,
        "imputation_weight": 0.3,
        "download_features": True,
        "rtma_wind": True,
    }


@pytest.fixture
def config_path():
    """Path to the real fires.json config."""
    return REPO_ROOT / "config" / "fires.json"


@pytest.fixture
def saved_fire_npz(tmp_path, sample_fire_arrays, sample_metadata):
    """Save sample fire data to a temp npz file and return the path."""
    from wildfire_pipeline.processing.io import save_fire_data

    return save_fire_data(tmp_path / "test_fire", sample_fire_arrays, sample_metadata, fmt="npz")
