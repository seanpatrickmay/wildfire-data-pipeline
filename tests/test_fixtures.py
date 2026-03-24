"""Tests for conftest fixtures — verify the test foundation itself is correct.

If fixtures silently produce wrong shapes, dtypes, or value ranges, the
entire test suite gives false confidence. These tests validate the fixtures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path


class TestSampleFireArrays:
    """Verify the sample_fire_arrays fixture produces correct data."""

    def test_keys(self, sample_fire_arrays: dict[str, np.ndarray]) -> None:
        expected = {"data", "observation_valid", "cloud_mask", "frp"}
        assert set(sample_fire_arrays.keys()) == expected

    def test_shapes_consistent(self, sample_fire_arrays: dict[str, np.ndarray]) -> None:
        shapes = {k: v.shape for k, v in sample_fire_arrays.items()}
        assert len(set(shapes.values())) == 1, f"Shape mismatch: {shapes}"

    def test_shape_is_3d(self, sample_fire_arrays: dict[str, np.ndarray]) -> None:
        for name, arr in sample_fire_arrays.items():
            assert arr.ndim == 3, f"{name} is {arr.ndim}D, expected 3D"

    def test_dtypes_are_float32(self, sample_fire_arrays: dict[str, np.ndarray]) -> None:
        for name, arr in sample_fire_arrays.items():
            assert arr.dtype == np.float32, f"{name} dtype is {arr.dtype}"

    def test_confidence_in_valid_range(self, sample_fire_arrays: dict[str, np.ndarray]) -> None:
        conf = sample_fire_arrays["data"]
        assert conf.min() >= 0.0
        assert conf.max() <= 1.0

    def test_obs_valid_is_binary(self, sample_fire_arrays: dict[str, np.ndarray]) -> None:
        ov = sample_fire_arrays["observation_valid"]
        assert set(np.unique(ov)).issubset({0.0, 1.0})

    def test_cloud_mask_is_binary(self, sample_fire_arrays: dict[str, np.ndarray]) -> None:
        cm = sample_fire_arrays["cloud_mask"]
        assert set(np.unique(cm)).issubset({0.0, 1.0})

    def test_cloud_is_inverse_of_obs_valid(self, sample_fire_arrays: dict[str, np.ndarray]) -> None:
        ov = sample_fire_arrays["observation_valid"]
        cm = sample_fire_arrays["cloud_mask"]
        np.testing.assert_array_equal(cm, 1.0 - ov)

    def test_frp_non_negative(self, sample_fire_arrays: dict[str, np.ndarray]) -> None:
        assert sample_fire_arrays["frp"].min() >= 0.0

    def test_has_fire_pixels(self, sample_fire_arrays: dict[str, np.ndarray]) -> None:
        assert sample_fire_arrays["data"].max() > 0.0

    def test_has_cloudy_hours(self, sample_fire_arrays: dict[str, np.ndarray]) -> None:
        ov = sample_fire_arrays["observation_valid"]
        assert (ov == 0).any(), "Fixture should include cloudy hours for testing"


class TestSampleMetadata:
    """Verify the sample_metadata fixture has required fields."""

    def test_required_fields_present(self, sample_metadata: dict) -> None:
        required = {"fire_name", "year", "n_hours", "aoi"}
        assert required.issubset(set(sample_metadata.keys()))

    def test_fire_name_is_string(self, sample_metadata: dict) -> None:
        assert isinstance(sample_metadata["fire_name"], str)

    def test_n_hours_positive(self, sample_metadata: dict) -> None:
        assert sample_metadata["n_hours"] > 0

    def test_aoi_has_four_elements(self, sample_metadata: dict) -> None:
        assert len(sample_metadata["aoi"]) == 4


class TestPipelineConfig:
    """Verify the pipeline_config fixture matches the Pydantic model."""

    def test_is_valid_config(self, pipeline_config: dict) -> None:
        from wildfire_pipeline.config import PipelineConfig

        cfg = PipelineConfig.model_validate(pipeline_config)
        assert cfg.goes_confidence_threshold == pytest.approx(0.30)

    def test_has_new_quality_fields(self, pipeline_config: dict) -> None:
        assert "max_persistence_gap_hours" in pipeline_config
        assert "imputation_weight" in pipeline_config

    def test_has_smoothing_config(self, pipeline_config: dict) -> None:
        assert "label_smoothing" in pipeline_config
        assert "method" in pipeline_config["label_smoothing"]


class TestSavedFireNpz:
    """Verify the saved_fire_npz fixture produces a loadable file."""

    def test_file_exists(self, saved_fire_npz: Path) -> None:
        assert saved_fire_npz.exists()

    def test_file_is_npz(self, saved_fire_npz: Path) -> None:
        assert saved_fire_npz.suffix == ".npz"

    def test_loadable(self, saved_fire_npz: Path) -> None:
        from wildfire_pipeline.processing.io import load_fire_data

        arrays, metadata = load_fire_data(saved_fire_npz)
        assert "data" in arrays
        assert "fire_name" in metadata


class TestProgressLoggingFormula:
    """Verify the progress step calculation from download.py."""

    @staticmethod
    def _compute_step(n_hours: int) -> int:
        """Mirrors the formula: max(1, min(24, n_hours // 10))."""
        return max(1, min(24, n_hours // 10))

    def test_short_fire_logs_every_hour(self) -> None:
        assert self._compute_step(5) == 1
        assert self._compute_step(10) == 1

    def test_medium_fire_logs_at_10pct(self) -> None:
        assert self._compute_step(50) == 5
        assert self._compute_step(100) == 10

    def test_long_fire_caps_at_24(self) -> None:
        assert self._compute_step(300) == 24
        assert self._compute_step(1000) == 24

    def test_exact_boundary(self) -> None:
        assert self._compute_step(240) == 24
        assert self._compute_step(239) == 23
