"""End-to-end tests for process_fire — the main label-processing orchestration function."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from wildfire_pipeline.processing.io import save_fire_data
from wildfire_pipeline.processing.labels import process_fire

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_and_process(
    tmp_path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    config: dict[str, Any],
    fmt: str = "npz",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Save arrays to disk, run process_fire, return result."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    fire_path = save_fire_data(tmp_path / "test_fire", arrays, metadata, fmt=fmt)
    return process_fire(fire_path, config, fmt=fmt)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestProcessFireHappyPath:
    """process_fire with valid synthetic data returns correct structure."""

    def test_returns_expected_array_keys(
        self, tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
    ):
        out_arrays, _ = _save_and_process(
            tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
        )
        expected_keys = {
            "labels",
            "soft_labels",
            "fire_change",
            "validity",
            "prev_fire_state",
            "prev_distance_to_fire",
            "prev_fire_neighborhood",
            "loss_weights",
            "_diag_raw_confidence",
            "_diag_capped_frp",
            "_diag_frp_reliability",
            "_diag_was_imputed",
        }
        assert set(out_arrays.keys()) == expected_keys

    def test_returns_metadata_with_processing_section(
        self, tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
    ):
        _, out_meta = _save_and_process(
            tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
        )
        assert "processing" in out_meta
        assert "quality" in out_meta

    def test_array_shapes_match_input(
        self, tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
    ):
        out_arrays, _ = _save_and_process(
            tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
        )
        expected_shape = sample_fire_arrays["data"].shape
        for name, arr in out_arrays.items():
            assert arr.shape == expected_shape, f"{name} shape mismatch"

    def test_raw_confidence_equals_input_data(
        self, tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
    ):
        out_arrays, _ = _save_and_process(
            tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
        )
        np.testing.assert_array_equal(
            out_arrays["_diag_raw_confidence"],
            sample_fire_arrays["data"].astype(np.float32),
        )

    def test_output_saved_to_processed_dir(
        self, tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
    ):
        fire_path = save_fire_data(
            tmp_path / "test_fire", sample_fire_arrays, sample_metadata, fmt="npz"
        )
        process_fire(fire_path, pipeline_config, fmt="npz")
        processed_dir = tmp_path / "processed"
        assert processed_dir.exists()
        processed_files = list(processed_dir.glob("*_processed.npz"))
        assert len(processed_files) == 1

    def test_input_metadata_fields_preserved(
        self, tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
    ):
        _, out_meta = _save_and_process(
            tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
        )
        assert out_meta["fire_name"] == "TestFire"
        assert out_meta["year"] == 2023


# ---------------------------------------------------------------------------
# Output is binary
# ---------------------------------------------------------------------------


class TestOutputIsBinary:
    """Labels output should only contain 0.0 and 1.0."""

    def test_labels_are_binary(
        self, tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
    ):
        out_arrays, _ = _save_and_process(
            tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
        )
        unique_vals = np.unique(out_arrays["labels"])
        assert set(unique_vals).issubset({0.0, 1.0})

    def test_labels_dtype_is_float32(
        self, tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
    ):
        out_arrays, _ = _save_and_process(
            tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
        )
        assert out_arrays["labels"].dtype == np.float32


# ---------------------------------------------------------------------------
# Cloud masking applied
# ---------------------------------------------------------------------------


class TestCloudMaskingApplied:
    """Cloudy pixels should have validity=0 when cloud masking is enabled."""

    def test_fully_cloudy_hour_has_zero_validity(
        self, tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
    ):
        """Hour 3 is fully cloudy in sample data -> validity should be 0 everywhere."""
        out_arrays, _ = _save_and_process(
            tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
        )
        # obs_valid[3] = 0.0 and cloud_mask[3] = 1.0 -> validity[3] = 0 * (1-1) = 0
        np.testing.assert_array_equal(out_arrays["validity"][3], 0.0)

    def test_partially_cloudy_hour_has_mixed_validity(
        self, tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
    ):
        """Hour 7 is partially cloudy -> some pixels valid, some not."""
        out_arrays, _ = _save_and_process(
            tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
        )
        validity_h7 = out_arrays["validity"][7]
        # Top 2 rows invalid (obs_valid=0, cloud_mask=1)
        np.testing.assert_array_equal(validity_h7[:2, :], 0.0)
        # Bottom 3 rows valid (obs_valid=1, cloud_mask=0)
        np.testing.assert_array_equal(validity_h7[2:, :], 1.0)

    def test_clear_hours_have_full_validity(
        self, tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
    ):
        """Hours without clouds should have validity=1 everywhere."""
        out_arrays, _ = _save_and_process(
            tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
        )
        # Hour 0 has obs_valid=1 and cloud_mask=0 everywhere
        np.testing.assert_array_equal(out_arrays["validity"][0], 1.0)


# ---------------------------------------------------------------------------
# Smoothing reduces flicker
# ---------------------------------------------------------------------------


class TestSmoothingReducesFlicker:
    """Smoothing should reduce the flicker rate compared to raw thresholded data."""

    def test_smoothed_flicker_rate_lower_than_raw(self, tmp_path, pipeline_config):
        """Build a deliberately flickery pattern, verify smoothing helps."""
        T, H, W = 20, 3, 3
        # Alternating fire/no-fire pattern for center pixel
        confidence = np.zeros((T, H, W), dtype=np.float32)
        for t in range(0, T, 2):
            confidence[t, 1, 1] = 0.8  # fire every other hour

        arrays = {
            "data": confidence,
            "observation_valid": np.ones((T, H, W), dtype=np.float32),
            "cloud_mask": np.zeros((T, H, W), dtype=np.float32),
            "frp": np.zeros((T, H, W), dtype=np.float32),
        }
        metadata = {"fire_name": "FlickerTest", "n_hours": T}

        _, out_meta = _save_and_process(tmp_path, arrays, metadata, pipeline_config)

        # The raw binary pattern alternates 1/0, so flicker rate would be high.
        # After majority-vote smoothing, the flicker should be lower.
        # Specifically, with window=5/min_votes=2, alternating detections are
        # dense enough to mostly smooth to continuous fire.
        assert out_meta["quality"]["flicker_rate"] < 0.5

    def test_sustained_fire_has_zero_flicker(self, tmp_path, pipeline_config):
        """Continuous fire should have flicker_rate = 0."""
        T, H, W = 10, 2, 2
        confidence = np.full((T, H, W), 0.8, dtype=np.float32)
        arrays = {
            "data": confidence,
            "observation_valid": np.ones((T, H, W), dtype=np.float32),
            "cloud_mask": np.zeros((T, H, W), dtype=np.float32),
            "frp": np.zeros((T, H, W), dtype=np.float32),
        }
        metadata = {"fire_name": "SustainedFire", "n_hours": T}

        _, out_meta = _save_and_process(tmp_path, arrays, metadata, pipeline_config)

        assert out_meta["quality"]["flicker_rate"] == 0.0


# ---------------------------------------------------------------------------
# Confidence threshold respected
# ---------------------------------------------------------------------------


class TestConfidenceThresholdRespected:
    """No fire pixels where raw confidence < threshold."""

    def test_below_threshold_never_labeled_fire(self, tmp_path, pipeline_config):
        T, H, W = 10, 3, 3
        threshold = pipeline_config["goes_confidence_threshold"]  # 0.30

        # All confidence values just below threshold
        confidence = np.full((T, H, W), threshold - 0.01, dtype=np.float32)
        arrays = {
            "data": confidence,
            "observation_valid": np.ones((T, H, W), dtype=np.float32),
            "cloud_mask": np.zeros((T, H, W), dtype=np.float32),
            "frp": np.zeros((T, H, W), dtype=np.float32),
        }
        metadata = {"fire_name": "LowConf", "n_hours": T}

        out_arrays, _ = _save_and_process(tmp_path, arrays, metadata, pipeline_config)

        # Nothing should be labeled as fire
        assert out_arrays["labels"].sum() == 0.0

    def test_at_threshold_is_labeled_fire(self, tmp_path, pipeline_config):
        T, H, W = 10, 3, 3
        threshold = pipeline_config["goes_confidence_threshold"]

        # Confidence at exactly the threshold for all pixels, all timesteps
        # With majority_vote window=5, min_votes=2, need >= 2 consecutive to survive
        confidence = np.full((T, H, W), threshold, dtype=np.float32)
        arrays = {
            "data": confidence,
            "observation_valid": np.ones((T, H, W), dtype=np.float32),
            "cloud_mask": np.zeros((T, H, W), dtype=np.float32),
            "frp": np.zeros((T, H, W), dtype=np.float32),
        }
        metadata = {"fire_name": "ExactThreshold", "n_hours": T}

        out_arrays, _ = _save_and_process(tmp_path, arrays, metadata, pipeline_config)

        # All pixels are at threshold -> binary=1 everywhere, so after smoothing
        # (with enough timesteps) most will survive. At least t>=1 should be fire.
        assert out_arrays["labels"].sum() > 0

    def test_custom_threshold_respected(self, tmp_path, pipeline_config):
        """Verify a higher custom threshold filters out lower-confidence pixels."""
        T, H, W = 10, 3, 3
        config = {**pipeline_config, "goes_confidence_threshold": 0.80}

        # Confidence at 0.5 -- below the custom 0.80 threshold
        confidence = np.full((T, H, W), 0.5, dtype=np.float32)
        arrays = {
            "data": confidence,
            "observation_valid": np.ones((T, H, W), dtype=np.float32),
            "cloud_mask": np.zeros((T, H, W), dtype=np.float32),
            "frp": np.zeros((T, H, W), dtype=np.float32),
        }
        metadata = {"fire_name": "HighThreshold", "n_hours": T}

        out_arrays, _ = _save_and_process(tmp_path, arrays, metadata, config)

        assert out_arrays["labels"].sum() == 0.0


# ---------------------------------------------------------------------------
# Quality metrics computed
# ---------------------------------------------------------------------------


class TestQualityMetricsComputed:
    """Metadata should contain flicker_rate, oracle_f1, and processing params."""

    def test_quality_section_has_expected_keys(
        self, tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
    ):
        _, out_meta = _save_and_process(
            tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
        )
        quality = out_meta["quality"]
        assert "raw_fire_pixels" in quality
        assert "smoothed_fire_pixels" in quality
        assert "flicker_rate" in quality
        assert "oracle_f1_smoothed" in quality
        assert "cloud_excluded_fraction" in quality

    def test_processing_section_has_expected_keys(
        self, tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
    ):
        _, out_meta = _save_and_process(
            tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
        )
        processing = out_meta["processing"]
        assert processing["confidence_threshold"] == 0.30
        assert processing["smoothing_method"] == "majority_vote"
        assert processing["smoothing_window"] == 5
        assert processing["smoothing_min_votes"] == 2
        assert processing["cloud_masking"] is True

    def test_flicker_rate_is_non_negative(
        self, tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
    ):
        _, out_meta = _save_and_process(
            tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
        )
        assert out_meta["quality"]["flicker_rate"] >= 0.0

    def test_oracle_f1_in_valid_range(
        self, tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
    ):
        _, out_meta = _save_and_process(
            tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
        )
        f1 = out_meta["quality"]["oracle_f1_smoothed"]
        assert 0.0 <= f1 <= 1.0

    def test_cloud_excluded_fraction_matches_validity(
        self, tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
    ):
        out_arrays, out_meta = _save_and_process(
            tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
        )
        expected = float(1.0 - out_arrays["validity"].mean())
        assert out_meta["quality"]["cloud_excluded_fraction"] == pytest.approx(expected)

    def test_raw_fire_pixels_count_matches_threshold(
        self, tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
    ):
        _, out_meta = _save_and_process(
            tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
        )
        conf = sample_fire_arrays["data"]
        threshold = pipeline_config["goes_confidence_threshold"]
        expected_raw = int((conf >= threshold).sum())
        assert out_meta["quality"]["raw_fire_pixels"] == expected_raw


# ---------------------------------------------------------------------------
# No smoothing method (passthrough)
# ---------------------------------------------------------------------------


class TestNoSmoothingMethod:
    """With method='none', labels should be raw thresholded binary."""

    def test_none_method_is_passthrough(self, tmp_path, pipeline_config):
        T, H, W = 10, 3, 3
        config = {
            **pipeline_config,
            "label_smoothing": {"method": "none", "window_hours": 5, "min_votes": 2},
        }
        # Cluster of 2 adjacent pixels at t=5 (survives isolated pixel filter)
        confidence = np.zeros((T, H, W), dtype=np.float32)
        confidence[5, 1, 1] = 0.8
        confidence[5, 1, 2] = 0.8
        arrays = {
            "data": confidence,
            "observation_valid": np.ones((T, H, W), dtype=np.float32),
            "cloud_mask": np.zeros((T, H, W), dtype=np.float32),
            "frp": np.zeros((T, H, W), dtype=np.float32),
        }
        metadata = {"fire_name": "NoSmooth", "n_hours": T}

        out_arrays, out_meta = _save_and_process(tmp_path, arrays, metadata, config)

        # Without smoothing, the cluster should survive as-is
        assert out_arrays["labels"][5, 1, 1] == 1.0
        assert out_arrays["labels"][5, 1, 2] == 1.0
        assert out_arrays["labels"].sum() == 2.0
        assert out_meta["processing"]["smoothing_method"] == "none"

    def test_none_method_preserves_clustered_fire(self, tmp_path, pipeline_config):
        """Labels with method='none' should preserve fire pixels that have spatial neighbors."""
        T, H, W = 8, 4, 4
        config = {
            **pipeline_config,
            "label_smoothing": {"method": "none", "window_hours": 5, "min_votes": 2},
        }
        # Create fire clusters (not isolated) across multiple timesteps
        confidence = np.zeros((T, H, W), dtype=np.float32)
        for t in range(T):
            # 2x2 fire cluster in top-left
            confidence[t, 0:2, 0:2] = 0.8
        arrays = {
            "data": confidence,
            "observation_valid": np.ones((T, H, W), dtype=np.float32),
            "cloud_mask": np.zeros((T, H, W), dtype=np.float32),
            "frp": np.zeros((T, H, W), dtype=np.float32),
        }
        metadata = {"fire_name": "NonePassthrough", "n_hours": T}

        out_arrays, _ = _save_and_process(tmp_path, arrays, metadata, config)

        # All cluster pixels should survive (not isolated)
        for t in range(T):
            assert out_arrays["labels"][t, 0, 0] == 1.0
            assert out_arrays["labels"][t, 0, 1] == 1.0
            assert out_arrays["labels"][t, 1, 0] == 1.0
            assert out_arrays["labels"][t, 1, 1] == 1.0


# ---------------------------------------------------------------------------
# Rolling max method
# ---------------------------------------------------------------------------


class TestRollingMaxMethod:
    """With method='rolling_max', any detection in window keeps fire alive."""

    def test_single_detection_persists_through_window(self, tmp_path, pipeline_config):
        T, H, W = 10, 3, 3
        config = {
            **pipeline_config,
            "label_smoothing": {
                "method": "rolling_max",
                "window_hours": 4,
                "min_votes": 2,  # min_votes is ignored for rolling_max
            },
        }
        # 2-pixel cluster at t=3 (survives isolated pixel filter)
        confidence = np.zeros((T, H, W), dtype=np.float32)
        confidence[3, 0, 0] = 0.8
        confidence[3, 0, 1] = 0.8
        arrays = {
            "data": confidence,
            "observation_valid": np.ones((T, H, W), dtype=np.float32),
            "cloud_mask": np.zeros((T, H, W), dtype=np.float32),
            "frp": np.zeros((T, H, W), dtype=np.float32),
        }
        metadata = {"fire_name": "RollingMax", "n_hours": T}

        out_arrays, out_meta = _save_and_process(tmp_path, arrays, metadata, config)

        labels = out_arrays["labels"][:, 0, 0]
        # t=0,1,2: no fire in lookback window -> 0
        assert labels[0] == 0.0
        assert labels[1] == 0.0
        assert labels[2] == 0.0
        # t=3: fire detected -> 1
        assert labels[3] == 1.0
        # t=4: window [1..4] includes t=3 -> 1
        assert labels[4] == 1.0
        # t=5: window [2..5] includes t=3 -> 1
        assert labels[5] == 1.0
        # t=6: window [3..6] includes t=3 -> 1
        assert labels[6] == 1.0
        # t=7: window [4..7] does NOT include t=3 -> 0
        assert labels[7] == 0.0

        assert out_meta["processing"]["smoothing_method"] == "rolling_max"

    def test_rolling_max_more_aggressive_than_majority_vote(self, tmp_path, pipeline_config):
        """Rolling max should produce >= as many fire pixels as majority vote."""
        T, H, W = 10, 3, 3
        rng = np.random.default_rng(123)
        confidence = rng.choice([0.0, 0.8], size=(T, H, W)).astype(np.float32)

        arrays = {
            "data": confidence,
            "observation_valid": np.ones((T, H, W), dtype=np.float32),
            "cloud_mask": np.zeros((T, H, W), dtype=np.float32),
            "frp": np.zeros((T, H, W), dtype=np.float32),
        }
        metadata = {"fire_name": "CompareTest", "n_hours": T}

        config_mv = {
            **pipeline_config,
            "label_smoothing": {"method": "majority_vote", "window_hours": 5, "min_votes": 2},
        }
        config_rm = {
            **pipeline_config,
            "label_smoothing": {"method": "rolling_max", "window_hours": 5, "min_votes": 2},
        }

        out_mv, _ = _save_and_process(tmp_path / "mv", arrays, metadata, config_mv)
        out_rm, _ = _save_and_process(tmp_path / "rm", arrays, metadata, config_rm)

        # Rolling max (min_votes=1 equivalent) should be >= majority_vote (min_votes=2)
        assert out_rm["labels"].sum() >= out_mv["labels"].sum()


# ---------------------------------------------------------------------------
# Cloud masking disabled
# ---------------------------------------------------------------------------


class TestCloudMaskingDisabled:
    """With cloud_masking=False, all validity should be 1.0."""

    def test_all_validity_one_when_disabled(
        self, tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
    ):
        config = {**pipeline_config, "cloud_masking": False}
        out_arrays, out_meta = _save_and_process(
            tmp_path, sample_fire_arrays, sample_metadata, config
        )
        np.testing.assert_array_equal(out_arrays["validity"], 1.0)
        assert out_meta["processing"]["cloud_masking"] is False

    def test_cloudy_hours_still_valid_when_disabled(
        self, tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
    ):
        """Even hour 3 (fully cloudy in input) should have validity=1."""
        config = {**pipeline_config, "cloud_masking": False}
        out_arrays, _ = _save_and_process(tmp_path, sample_fire_arrays, sample_metadata, config)
        np.testing.assert_array_equal(out_arrays["validity"][3], 1.0)

    def test_cloud_excluded_fraction_zero_when_disabled(
        self, tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
    ):
        config = {**pipeline_config, "cloud_masking": False}
        _, out_meta = _save_and_process(tmp_path, sample_fire_arrays, sample_metadata, config)
        assert out_meta["quality"]["cloud_excluded_fraction"] == 0.0


# ---------------------------------------------------------------------------
# All zeros input
# ---------------------------------------------------------------------------


class TestAllZerosInput:
    """process_fire should handle data with no fire pixels gracefully."""

    def test_all_zeros_does_not_crash(self, tmp_path, pipeline_config):
        T, H, W = 10, 4, 4
        arrays = {
            "data": np.zeros((T, H, W), dtype=np.float32),
            "observation_valid": np.ones((T, H, W), dtype=np.float32),
            "cloud_mask": np.zeros((T, H, W), dtype=np.float32),
            "frp": np.zeros((T, H, W), dtype=np.float32),
        }
        metadata = {"fire_name": "EmptyFire", "n_hours": T}

        out_arrays, out_meta = _save_and_process(tmp_path, arrays, metadata, pipeline_config)

        # No fire pixels in output
        assert out_arrays["labels"].sum() == 0.0
        assert out_meta["quality"]["raw_fire_pixels"] == 0
        assert out_meta["quality"]["smoothed_fire_pixels"] == 0

    def test_all_zeros_flicker_rate_is_zero(self, tmp_path, pipeline_config):
        T, H, W = 10, 4, 4
        arrays = {
            "data": np.zeros((T, H, W), dtype=np.float32),
            "observation_valid": np.ones((T, H, W), dtype=np.float32),
            "cloud_mask": np.zeros((T, H, W), dtype=np.float32),
            "frp": np.zeros((T, H, W), dtype=np.float32),
        }
        metadata = {"fire_name": "EmptyFire", "n_hours": T}

        _, out_meta = _save_and_process(tmp_path, arrays, metadata, pipeline_config)

        assert out_meta["quality"]["flicker_rate"] == 0.0

    def test_all_zeros_oracle_f1_is_zero(self, tmp_path, pipeline_config):
        T, H, W = 10, 4, 4
        arrays = {
            "data": np.zeros((T, H, W), dtype=np.float32),
            "observation_valid": np.ones((T, H, W), dtype=np.float32),
            "cloud_mask": np.zeros((T, H, W), dtype=np.float32),
            "frp": np.zeros((T, H, W), dtype=np.float32),
        }
        metadata = {"fire_name": "EmptyFire", "n_hours": T}

        _, out_meta = _save_and_process(tmp_path, arrays, metadata, pipeline_config)

        assert out_meta["quality"]["oracle_f1_smoothed"] == 0.0

    def test_all_zeros_with_all_clouds(self, tmp_path, pipeline_config):
        """Zero fire + full cloud coverage should not crash."""
        T, H, W = 5, 3, 3
        arrays = {
            "data": np.zeros((T, H, W), dtype=np.float32),
            "observation_valid": np.zeros((T, H, W), dtype=np.float32),
            "cloud_mask": np.ones((T, H, W), dtype=np.float32),
            "frp": np.zeros((T, H, W), dtype=np.float32),
        }
        metadata = {"fire_name": "CloudyEmpty", "n_hours": T}

        out_arrays, out_meta = _save_and_process(tmp_path, arrays, metadata, pipeline_config)

        assert out_arrays["labels"].sum() == 0.0
        assert out_arrays["validity"].sum() == 0.0
        assert out_meta["quality"]["cloud_excluded_fraction"] == 1.0


# ---------------------------------------------------------------------------
# Type safety: PipelineConfig vs dict
# ---------------------------------------------------------------------------


class TestTypedConfig:
    """Verify process_fire works with both PipelineConfig objects and raw dicts."""

    def test_accepts_pipeline_config_object(self, tmp_path, sample_fire_arrays, sample_metadata):
        from wildfire_pipeline.config import PipelineConfig

        typed_config = PipelineConfig()
        out_arrays, out_meta = _save_and_process(
            tmp_path, sample_fire_arrays, sample_metadata, typed_config
        )
        assert "labels" in out_arrays
        assert out_meta["processing"]["confidence_threshold"] == 0.30

    def test_accepts_raw_dict(self, tmp_path, sample_fire_arrays, sample_metadata, pipeline_config):
        out_arrays, _ = _save_and_process(
            tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
        )
        assert "labels" in out_arrays

    def test_custom_typed_config_values_respected(
        self, tmp_path, sample_fire_arrays, sample_metadata
    ):
        from wildfire_pipeline.config import PipelineConfig

        config = PipelineConfig(
            goes_confidence_threshold=0.8,
            max_persistence_gap_hours=1,
            imputation_weight=0.5,
        )
        _out_arrays, out_meta = _save_and_process(
            tmp_path, sample_fire_arrays, sample_metadata, config
        )
        assert out_meta["processing"]["confidence_threshold"] == 0.8
        assert out_meta["processing"]["cloud_persistence_max_gap"] == 1
        assert out_meta["processing"]["imputation_weight"] == 0.5

    def test_invalid_dict_raises_validation_error(
        self, tmp_path, sample_fire_arrays, sample_metadata
    ):
        from pydantic import ValidationError

        bad_config = {"goes_confidence_threshold": 5.0}  # Out of [0,1] range
        with pytest.raises(ValidationError):
            _save_and_process(tmp_path, sample_fire_arrays, sample_metadata, bad_config)


# ---------------------------------------------------------------------------
# Missing/partial input arrays (backward compatibility with older formats)
# ---------------------------------------------------------------------------


class TestMissingInputArrays:
    """Verify process_fire gracefully handles missing optional arrays.

    Real-world scenario: older GEE exports or alternative data sources may
    not include FRP, cloud_mask, or observation_valid arrays.
    """

    def test_missing_frp_uses_zeros(self, tmp_path, pipeline_config):
        """process_fire should work when FRP is not in the input data."""
        T, H, W = 8, 4, 4
        arrays = {
            "data": np.full((T, H, W), 0.5, dtype=np.float32),
            "observation_valid": np.ones((T, H, W), dtype=np.float32),
            "cloud_mask": np.zeros((T, H, W), dtype=np.float32),
            # No "frp" key
        }
        metadata = {"fire_name": "NoFRP", "n_hours": T}

        out_arrays, _out_meta = _save_and_process(tmp_path, arrays, metadata, pipeline_config)

        assert "labels" in out_arrays
        assert "_diag_capped_frp" in out_arrays
        # Capped FRP should be all zeros since input was missing
        assert out_arrays["_diag_capped_frp"].sum() == 0.0
        # FRP reliability should still be computed (all 1.0 since no outliers)
        assert out_arrays["_diag_frp_reliability"].min() >= 0.0
        # Loss weights should still be valid
        assert out_arrays["loss_weights"].max() <= 1.0

    def test_missing_cloud_mask_uses_zeros(self, tmp_path, pipeline_config):
        """Missing cloud_mask should default to no clouds (all clear)."""
        T, H, W = 8, 4, 4
        arrays = {
            "data": np.full((T, H, W), 0.5, dtype=np.float32),
            "observation_valid": np.ones((T, H, W), dtype=np.float32),
            "frp": np.full((T, H, W), 50.0, dtype=np.float32),
            # No "cloud_mask" key
        }
        metadata = {"fire_name": "NoClouds", "n_hours": T}

        out_arrays, out_meta = _save_and_process(tmp_path, arrays, metadata, pipeline_config)

        assert "labels" in out_arrays
        # With no cloud mask, all pixels should be valid
        assert out_meta["quality"]["cloud_excluded_fraction"] == 0.0

    def test_missing_obs_valid_uses_ones(self, tmp_path, pipeline_config):
        """Missing observation_valid should default to all valid."""
        T, H, W = 8, 4, 4
        arrays = {
            "data": np.full((T, H, W), 0.5, dtype=np.float32),
            "cloud_mask": np.zeros((T, H, W), dtype=np.float32),
            "frp": np.full((T, H, W), 50.0, dtype=np.float32),
            # No "observation_valid" key
        }
        metadata = {"fire_name": "AllValid", "n_hours": T}

        out_arrays, _out_meta = _save_and_process(tmp_path, arrays, metadata, pipeline_config)

        assert "labels" in out_arrays
        assert out_arrays["validity"].sum() == T * H * W  # All valid

    def test_minimal_input_only_data(self, tmp_path, pipeline_config):
        """process_fire should work with only the 'data' array (absolute minimum)."""
        T, H, W = 6, 3, 3
        arrays = {
            "data": np.full((T, H, W), 0.5, dtype=np.float32),
            # No frp, no cloud_mask, no observation_valid
        }
        metadata = {"fire_name": "MinimalInput", "n_hours": T}

        out_arrays, _out_meta = _save_and_process(tmp_path, arrays, metadata, pipeline_config)

        assert "labels" in out_arrays
        assert "validity" in out_arrays
        assert "loss_weights" in out_arrays
        assert out_arrays["labels"].shape == (T, H, W)


# ---------------------------------------------------------------------------
# Previous fire state and derived features
# ---------------------------------------------------------------------------


class TestPrevFireState:
    """Verify prev_fire_state is correctly time-shifted."""

    def test_prev_fire_state_is_shifted_labels(
        self, tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
    ):
        out_arrays, _ = _save_and_process(
            tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
        )
        labels = out_arrays["labels"]
        prev = out_arrays["prev_fire_state"]
        # prev[0] should be all zeros (no history)
        assert prev[0].sum() == 0.0
        # prev[t] should equal labels[t-1] for t > 0
        if labels.shape[0] > 1:
            np.testing.assert_array_equal(prev[1:], labels[:-1])

    def test_fire_change_is_new_ignitions(
        self, tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
    ):
        out_arrays, _ = _save_and_process(
            tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
        )
        labels = out_arrays["labels"]
        prev = out_arrays["prev_fire_state"]
        change = out_arrays["fire_change"]
        # fire_change should be 1 only where labels=1 AND prev=0
        expected = ((labels == 1) & (prev == 0)).astype(np.float32)
        np.testing.assert_array_equal(change, expected)

    def test_prev_spatial_features_use_previous_labels(
        self, tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
    ):
        out_arrays, _ = _save_and_process(
            tmp_path, sample_fire_arrays, sample_metadata, pipeline_config
        )
        prev_dist = out_arrays["prev_distance_to_fire"]
        # At t=0, prev_fire_state is all zeros, so distance should be -1 (sentinel)
        assert (prev_dist[0] == -1.0).all()
