"""Tests for data quality functions: cloud-aware persistence, isolated pixel
filtering, FRP outlier detection, quality weighting, and gap statistics."""

from __future__ import annotations

import numpy as np
import pytest

from wildfire_pipeline.processing.quality import (
    FRP_DETECTION_FLOOR_MW,
    FRP_PRACTICAL_CEILING_MW,
    FRP_SATURATION_MW,
    cloud_aware_persistence,
    compute_gap_stats,
    compute_quality_weights,
    detect_frp_outliers,
    filter_isolated_pixels,
)

# ---------------------------------------------------------------------------
# cloud_aware_persistence
# ---------------------------------------------------------------------------


class TestCloudAwarePersistence:
    """Tests for forward-fill fire persistence through cloud gaps."""

    def test_fire_persists_through_1hour_gap(self) -> None:
        """Fire at t=0, invalid at t=1, fire at t=2 -> t=1 filled."""
        T, H, W = 5, 1, 1
        binary = np.zeros((T, H, W), dtype=np.float32)
        validity = np.ones((T, H, W), dtype=np.float32)

        binary[0, 0, 0] = 1.0
        binary[2, 0, 0] = 1.0
        validity[1, 0, 0] = 0.0  # 1-hour gap

        filled, was_imputed = cloud_aware_persistence(binary, validity, max_gap_hours=3)

        assert filled[1, 0, 0] == 1.0
        assert was_imputed[1, 0, 0] > 0

    def test_fire_persists_through_3hour_gap_at_max(self) -> None:
        """Fire persists through exactly max_gap_hours=3 invalid hours."""
        T, H, W = 8, 1, 1
        binary = np.zeros((T, H, W), dtype=np.float32)
        validity = np.ones((T, H, W), dtype=np.float32)

        binary[0, 0, 0] = 1.0  # fire before gap
        validity[1, 0, 0] = 0.0  # gap hours
        validity[2, 0, 0] = 0.0
        validity[3, 0, 0] = 0.0
        binary[4, 0, 0] = 1.0  # fire after gap

        filled, was_imputed = cloud_aware_persistence(binary, validity, max_gap_hours=3)

        # 3-hour gap should be fully filled
        assert filled[1, 0, 0] == 1.0
        assert filled[2, 0, 0] == 1.0
        assert filled[3, 0, 0] == 1.0
        assert was_imputed[1, 0, 0] > 0
        assert was_imputed[2, 0, 0] > 0
        assert was_imputed[3, 0, 0] > 0

    def test_fire_does_not_persist_through_4hour_gap(self) -> None:
        """A 4-hour gap exceeds max_gap_hours=3 -> last imputed hour cleared."""
        T, H, W = 8, 1, 1
        binary = np.zeros((T, H, W), dtype=np.float32)
        validity = np.ones((T, H, W), dtype=np.float32)

        binary[0, 0, 0] = 1.0
        validity[1, 0, 0] = 0.0
        validity[2, 0, 0] = 0.0
        validity[3, 0, 0] = 0.0
        validity[4, 0, 0] = 0.0  # 4-hour gap

        filled, was_imputed = cloud_aware_persistence(binary, validity, max_gap_hours=3)

        # First 3 hours should be filled, 4th hour should be cleared
        assert filled[1, 0, 0] == 1.0
        assert filled[2, 0, 0] == 1.0
        assert filled[3, 0, 0] == 1.0
        assert filled[4, 0, 0] == 0.0
        assert was_imputed[4, 0, 0] == 0.0

    def test_no_imputation_when_all_valid(self) -> None:
        """All valid observations -> no imputation needed."""
        T, H, W = 5, 2, 2
        binary = np.zeros((T, H, W), dtype=np.float32)
        binary[1, 0, 0] = 1.0
        binary[3, 0, 0] = 1.0
        validity = np.ones((T, H, W), dtype=np.float32)

        filled, was_imputed = cloud_aware_persistence(binary, validity)

        np.testing.assert_array_equal(filled, binary)
        assert was_imputed.sum() == 0

    def test_no_imputation_when_no_fire_before_gap(self) -> None:
        """Gap without fire before it -> no forward fill."""
        T, H, W = 5, 1, 1
        binary = np.zeros((T, H, W), dtype=np.float32)
        validity = np.ones((T, H, W), dtype=np.float32)

        validity[1, 0, 0] = 0.0  # gap at t=1
        binary[2, 0, 0] = 1.0  # fire only after gap

        filled, was_imputed = cloud_aware_persistence(binary, validity, max_gap_hours=3)

        # t=1 should NOT be filled because there's no fire at t=0
        assert filled[1, 0, 0] == 0.0
        assert was_imputed[1, 0, 0] == 0.0

    def test_was_imputed_marks_only_filled_pixels(self) -> None:
        """was_imputed should be nonzero only for pixels that were actually filled."""
        T, H, W = 6, 1, 2
        binary = np.zeros((T, H, W), dtype=np.float32)
        validity = np.ones((T, H, W), dtype=np.float32)

        # Pixel (0,0): fire at t=0, gap at t=1, fire at t=2
        binary[0, 0, 0] = 1.0
        validity[1, 0, 0] = 0.0
        binary[2, 0, 0] = 1.0

        # Pixel (0,1): no fire at all, gap at t=1
        validity[1, 0, 1] = 0.0

        _filled, was_imputed = cloud_aware_persistence(binary, validity, max_gap_hours=3)

        # Pixel (0,0) t=1 should be imputed
        assert was_imputed[1, 0, 0] > 0
        # Pixel (0,1) t=1 should NOT be imputed (no fire before gap)
        assert was_imputed[1, 0, 1] == 0.0
        # Original fire pixels should NOT be marked as imputed
        assert was_imputed[0, 0, 0] == 0.0
        assert was_imputed[2, 0, 0] == 0.0

    def test_imputation_does_not_create_fire_in_valid_non_fire(self) -> None:
        """Valid observations that are non-fire must remain non-fire."""
        T, H, W = 5, 1, 1
        binary = np.zeros((T, H, W), dtype=np.float32)
        validity = np.ones((T, H, W), dtype=np.float32)

        binary[0, 0, 0] = 1.0
        # t=1 is valid and non-fire -> should stay non-fire
        # t=2 is valid and non-fire -> should stay non-fire

        filled, _was_imputed = cloud_aware_persistence(binary, validity, max_gap_hours=3)

        # t=1 is valid, not a gap, so no imputation
        assert filled[1, 0, 0] == 0.0
        assert filled[2, 0, 0] == 0.0

    def test_multiple_pixels_fill_independently(self) -> None:
        """Each pixel's gap-fill is independent of its neighbors."""
        T, H, W = 5, 1, 3
        binary = np.zeros((T, H, W), dtype=np.float32)
        validity = np.ones((T, H, W), dtype=np.float32)

        # Pixel (0,0): fire->gap->fire
        binary[0, 0, 0] = 1.0
        validity[1, 0, 0] = 0.0
        binary[2, 0, 0] = 1.0

        # Pixel (0,1): no fire->gap->fire (no fill since no fire before gap)
        validity[1, 0, 1] = 0.0
        binary[2, 0, 1] = 1.0

        # Pixel (0,2): fire->gap->no fire (fills forward but no fire after)
        binary[0, 0, 2] = 1.0
        validity[1, 0, 2] = 0.0

        filled, _ = cloud_aware_persistence(binary, validity, max_gap_hours=3)

        assert filled[1, 0, 0] == 1.0  # fire before gap
        assert filled[1, 0, 1] == 0.0  # no fire before gap
        assert filled[1, 0, 2] == 1.0  # forward-fill from fire before gap

    def test_all_invalid_with_fire_fills_up_to_max_gap(self) -> None:
        """All-invalid validity with initial fire -> forward fill up to max_gap."""
        T, H, W = 6, 1, 1
        binary = np.zeros((T, H, W), dtype=np.float32)
        binary[0, 0, 0] = 1.0
        validity = np.zeros((T, H, W), dtype=np.float32)
        validity[0, 0, 0] = 1.0  # only t=0 is valid

        filled, _was_imputed = cloud_aware_persistence(binary, validity, max_gap_hours=3)

        # Forward fill should propagate for 3 hours then stop
        assert filled[0, 0, 0] == 1.0  # original
        assert filled[1, 0, 0] == 1.0  # imputed
        assert filled[2, 0, 0] == 1.0  # imputed
        assert filled[3, 0, 0] == 1.0  # imputed (3rd consecutive)
        assert filled[4, 0, 0] == 0.0  # exceeds max_gap
        assert filled[5, 0, 0] == 0.0  # exceeds max_gap

    def test_max_gap_hours_zero_no_imputation(self) -> None:
        """max_gap_hours=0 -> fill everything first then clear all imputed."""
        T, H, W = 5, 1, 1
        binary = np.zeros((T, H, W), dtype=np.float32)
        validity = np.ones((T, H, W), dtype=np.float32)

        binary[0, 0, 0] = 1.0
        validity[1, 0, 0] = 0.0

        filled, was_imputed = cloud_aware_persistence(binary, validity, max_gap_hours=0)

        # max_gap_hours=0 < T=5, so the enforcement pass runs with limit=0
        # The forward fill creates imputed at t=1, but then consecutive > 0
        # clears it immediately since 1 > 0
        assert filled[1, 0, 0] == 0.0
        assert was_imputed[1, 0, 0] == 0.0


# ---------------------------------------------------------------------------
# filter_isolated_pixels
# ---------------------------------------------------------------------------


class TestFilterIsolatedPixels:
    """Tests for spatial/temporal isolated pixel removal."""

    def test_isolated_single_pixel_removed(self) -> None:
        """A lone fire pixel with no spatial or temporal neighbors is removed."""
        T, H, W = 3, 5, 5
        binary = np.zeros((T, H, W), dtype=np.float32)
        binary[1, 2, 2] = 1.0  # single pixel at center, t=1

        result = filter_isolated_pixels(binary, min_spatial_neighbors=1)

        assert result[1, 2, 2] == 0.0

    def test_pixel_with_one_spatial_neighbor_kept(self) -> None:
        """A fire pixel with 1 spatial neighbor meets min_spatial_neighbors=1."""
        T, H, W = 3, 5, 5
        binary = np.zeros((T, H, W), dtype=np.float32)
        binary[1, 2, 2] = 1.0  # target pixel
        binary[1, 2, 3] = 1.0  # right neighbor

        result = filter_isolated_pixels(binary, min_spatial_neighbors=1)

        assert result[1, 2, 2] == 1.0
        assert result[1, 2, 3] == 1.0

    def test_pixel_with_temporal_but_no_spatial_neighbors_kept(self) -> None:
        """Spatially isolated but has temporal support -> kept."""
        T, H, W = 3, 5, 5
        binary = np.zeros((T, H, W), dtype=np.float32)
        binary[0, 2, 2] = 1.0  # temporal support at t=0
        binary[1, 2, 2] = 1.0  # target at t=1

        result = filter_isolated_pixels(
            binary, min_spatial_neighbors=1, require_temporal_support=True
        )

        # Has temporal support from t=0 -> kept
        assert result[1, 2, 2] == 1.0

    def test_pixel_with_spatial_neighbor_but_no_temporal_kept(self) -> None:
        """Has enough spatial neighbors -> kept regardless of temporal support."""
        T, H, W = 3, 5, 5
        binary = np.zeros((T, H, W), dtype=np.float32)
        binary[1, 2, 2] = 1.0  # target
        binary[1, 2, 3] = 1.0  # spatial neighbor
        # No fire at t=0 or t=2 for this pixel

        result = filter_isolated_pixels(
            binary, min_spatial_neighbors=1, require_temporal_support=True
        )

        assert result[1, 2, 2] == 1.0

    def test_cluster_of_fire_pixels_preserved(self) -> None:
        """A 2x2 cluster all have >= 2 spatial neighbors -> all kept."""
        T, H, W = 3, 5, 5
        binary = np.zeros((T, H, W), dtype=np.float32)
        binary[1, 1:3, 1:3] = 1.0  # 2x2 cluster

        result = filter_isolated_pixels(binary, min_spatial_neighbors=1)

        np.testing.assert_array_equal(result[1, 1:3, 1:3], 1.0)

    def test_empty_frame_unchanged(self) -> None:
        """All-zeros input should remain all-zeros."""
        T, H, W = 3, 4, 4
        binary = np.zeros((T, H, W), dtype=np.float32)

        result = filter_isolated_pixels(binary)

        np.testing.assert_array_equal(result, binary)

    def test_all_fire_frame_unchanged(self) -> None:
        """All-fire input: every pixel has 2-4 spatial neighbors -> all kept."""
        T, H, W = 3, 4, 4
        binary = np.ones((T, H, W), dtype=np.float32)

        result = filter_isolated_pixels(binary, min_spatial_neighbors=1)

        np.testing.assert_array_equal(result, binary)

    def test_require_temporal_false_only_checks_spatial(self) -> None:
        """With require_temporal_support=False, spatial neighbors alone decide."""
        T, H, W = 3, 5, 5
        binary = np.zeros((T, H, W), dtype=np.float32)
        # Single pixel with temporal support at t=0 but no spatial neighbors
        binary[0, 2, 2] = 1.0  # temporal support
        binary[1, 2, 2] = 1.0  # target pixel, no spatial neighbors

        result = filter_isolated_pixels(
            binary, min_spatial_neighbors=1, require_temporal_support=False
        )

        # No spatial neighbors AND temporal support ignored -> removed
        assert result[1, 2, 2] == 0.0

    def test_single_pixel_grid_not_filtered(self) -> None:
        """1x1 grid should skip isolated pixel filtering entirely."""
        binary = np.ones((5, 1, 1), dtype=np.float32)
        result = filter_isolated_pixels(
            binary, min_spatial_neighbors=1, require_temporal_support=True
        )
        np.testing.assert_array_equal(result, binary)

    def test_future_only_support_does_not_save_pixel(self) -> None:
        """A pixel with fire only at t+1 (future) should NOT provide temporal support."""
        T, H, W = 5, 3, 3
        binary = np.zeros((T, H, W), dtype=np.float32)
        binary[2, 1, 1] = 1.0  # isolated pixel at t=2
        binary[3, 1, 1] = 1.0  # fire at t=3 (future of t=2)
        # No fire at t=1 (past of t=2)

        result = filter_isolated_pixels(
            binary, min_spatial_neighbors=1, require_temporal_support=True
        )

        # t=2 pixel should be REMOVED — it has no past temporal support
        assert result[2, 1, 1] == 0.0
        # t=3 pixel has past support from original binary[2] -> kept
        assert result[3, 1, 1] == 1.0


# ---------------------------------------------------------------------------
# detect_frp_outliers
# ---------------------------------------------------------------------------


class TestDetectFrpOutliers:
    """Tests for FRP outlier detection and capping."""

    def test_normal_frp_reliability_all_one(self) -> None:
        """Normal FRP values -> reliability all 1.0, no capping."""
        T, H, W = 5, 3, 3
        frp = np.full((T, H, W), 100.0, dtype=np.float32)

        capped, reliability = detect_frp_outliers(frp)

        np.testing.assert_array_equal(reliability, 1.0)
        np.testing.assert_array_equal(capped, frp)

    def test_negative_frp_capped_to_zero_reliability_zero(self) -> None:
        """Negative FRP -> capped to 0, reliability = 0.0."""
        T, H, W = 3, 2, 2
        frp = np.full((T, H, W), 100.0, dtype=np.float32)
        frp[0, 0, 0] = -50.0

        capped, reliability = detect_frp_outliers(frp)

        assert capped[0, 0, 0] == 0.0
        assert reliability[0, 0, 0] == 0.0
        # Non-negative pixel should be unaffected
        assert reliability[1, 0, 0] == 1.0

    def test_frp_above_practical_ceiling_reliability_half(self) -> None:
        """FRP above 2000 MW -> reliability = 0.5."""
        T, H, W = 3, 2, 2
        frp = np.full((T, H, W), 100.0, dtype=np.float32)
        frp[0, 0, 0] = 3000.0  # above ceiling, below saturation

        _, reliability = detect_frp_outliers(frp)

        assert reliability[0, 0, 0] == pytest.approx(0.5)

    def test_frp_above_saturation_reliability_zero_and_capped(self) -> None:
        """FRP above 5000 MW -> reliability = 0.0, value capped at practical ceiling."""
        T, H, W = 3, 2, 2
        frp = np.full((T, H, W), 100.0, dtype=np.float32)
        frp[0, 0, 0] = 8000.0  # above saturation

        capped, reliability = detect_frp_outliers(frp)

        assert reliability[0, 0, 0] == 0.0
        # Fixed physical cap at FRP_PRACTICAL_CEILING_MW (2000)
        assert capped[0, 0, 0] == pytest.approx(FRP_PRACTICAL_CEILING_MW)

    def test_fixed_cap_at_practical_ceiling(self) -> None:
        """Values above practical ceiling are capped at 2000 MW (no percentile)."""
        T, H, W = 5, 4, 4
        frp = np.full((T, H, W), 100.0, dtype=np.float32)
        frp[0, 0, 0] = 4000.0  # above practical ceiling

        capped, _ = detect_frp_outliers(frp, cap_percentile=99.5)

        # Fixed physical cap at FRP_PRACTICAL_CEILING_MW — no temporal dependency
        assert capped[0, 0, 0] == pytest.approx(FRP_PRACTICAL_CEILING_MW)
        # Normal values below the cap are unchanged
        assert capped[1, 0, 0] == pytest.approx(100.0)

    def test_confidence_crosscheck_frp_positive_confidence_zero(self) -> None:
        """FRP > 0 but confidence == 0 -> reliability = 0.0."""
        T, H, W = 3, 2, 2
        frp = np.full((T, H, W), 100.0, dtype=np.float32)
        confidence = np.ones((T, H, W), dtype=np.float32)
        confidence[0, 0, 0] = 0.0  # zero confidence

        _, reliability = detect_frp_outliers(frp, confidence=confidence)

        assert reliability[0, 0, 0] == 0.0
        # Pixel with nonzero confidence should be fine
        assert reliability[1, 0, 0] == 1.0

    def test_all_zero_frp_no_changes(self) -> None:
        """All-zero FRP -> no capping needed, reliability all 1.0."""
        T, H, W = 5, 3, 3
        frp = np.zeros((T, H, W), dtype=np.float32)

        capped, reliability = detect_frp_outliers(frp)

        np.testing.assert_array_equal(capped, 0.0)
        np.testing.assert_array_equal(reliability, 1.0)

    def test_physical_constants_accessible(self) -> None:
        """Physical constants should be importable and have expected values."""
        assert FRP_DETECTION_FLOOR_MW >= 0
        assert FRP_PRACTICAL_CEILING_MW == 2000.0
        assert FRP_SATURATION_MW == 5000.0
        assert FRP_PRACTICAL_CEILING_MW < FRP_SATURATION_MW


# ---------------------------------------------------------------------------
# compute_quality_weights
# ---------------------------------------------------------------------------


class TestComputeQualityWeights:
    """Tests for per-pixel quality weight computation."""

    def test_all_valid_no_imputation_weights_one(self) -> None:
        """All valid, no imputation -> weights all 1.0."""
        T, H, W = 5, 3, 3
        validity = np.ones((T, H, W), dtype=np.float32)

        weights = compute_quality_weights(validity)

        np.testing.assert_array_equal(weights, 1.0)

    def test_invalid_pixels_weight_zero(self) -> None:
        """Invalid pixels -> weight 0.0."""
        T, H, W = 5, 3, 3
        validity = np.ones((T, H, W), dtype=np.float32)
        validity[0, 0, 0] = 0.0

        weights = compute_quality_weights(validity)

        assert weights[0, 0, 0] == 0.0
        assert weights[1, 0, 0] == 1.0

    def test_imputed_pixels_get_imputation_weight(self) -> None:
        """Imputed pixels -> weight = imputation_weight."""
        T, H, W = 5, 2, 2
        validity = np.ones((T, H, W), dtype=np.float32)
        was_imputed = np.zeros((T, H, W), dtype=np.float32)
        was_imputed[1, 0, 0] = 1.0

        weights = compute_quality_weights(validity, was_imputed=was_imputed)

        assert weights[1, 0, 0] == pytest.approx(0.3)  # default imputation_weight
        assert weights[0, 0, 0] == pytest.approx(1.0)

    def test_frp_reliability_reduces_weights(self) -> None:
        """FRP reliability multiplicatively reduces weights.

        Note: the implementation applies max(reliability, 0.3), so a
        reliability of 0.5 stays at 0.5, and reliability of 0.0 becomes 0.3.
        """
        T, H, W = 5, 2, 2
        validity = np.ones((T, H, W), dtype=np.float32)
        frp_reliability = np.ones((T, H, W), dtype=np.float32)
        frp_reliability[0, 0, 0] = 0.5

        weights = compute_quality_weights(validity, frp_reliability=frp_reliability)

        # weight = 1.0 * max(0.5, 0.3) = 0.5
        assert weights[0, 0, 0] == pytest.approx(0.5)
        # weight = 1.0 * max(1.0, 0.3) = 1.0
        assert weights[1, 0, 0] == pytest.approx(1.0)

    def test_custom_imputation_weight(self) -> None:
        """Custom imputation_weight should be respected."""
        T, H, W = 5, 2, 2
        validity = np.ones((T, H, W), dtype=np.float32)
        was_imputed = np.zeros((T, H, W), dtype=np.float32)
        was_imputed[0, 0, 0] = 1.0

        weights = compute_quality_weights(validity, was_imputed=was_imputed, imputation_weight=0.7)

        assert weights[0, 0, 0] == pytest.approx(0.7)

    def test_imputed_pixels_in_cloud_gaps_get_weight(self) -> None:
        """Critical: imputed pixels have validity=0 but should still get imputation_weight."""
        validity = np.zeros((5, 3, 3), dtype=np.float32)  # all invalid (cloud)
        was_imputed = np.zeros((5, 3, 3), dtype=np.float32)
        was_imputed[2, 1, 1] = 1.0  # pixel was forward-filled through gap

        weights = compute_quality_weights(validity, was_imputed=was_imputed)

        # The imputed pixel should get 0.3, not 0.0
        assert weights[2, 1, 1] == pytest.approx(0.3)
        # Non-imputed invalid pixels should remain 0
        assert weights[0, 0, 0] == 0.0

    def test_combination_valid_imputed_invalid(self) -> None:
        """Combination: valid observed=1.0, valid imputed=0.3, invalid=0.0."""
        T, H, W = 3, 1, 3
        validity = np.array([[[1.0, 1.0, 0.0]]], dtype=np.float32)  # (1, 1, 3)
        validity = np.broadcast_to(validity, (T, H, W)).copy()

        was_imputed = np.zeros((T, H, W), dtype=np.float32)
        was_imputed[:, 0, 1] = 1.0  # pixel 1 is imputed

        weights = compute_quality_weights(validity, was_imputed=was_imputed)

        # Pixel 0: valid, not imputed -> 1.0
        assert weights[0, 0, 0] == pytest.approx(1.0)
        # Pixel 1: valid, imputed -> 0.3
        assert weights[0, 0, 1] == pytest.approx(0.3)
        # Pixel 2: invalid -> 0.0
        assert weights[0, 0, 2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_gap_stats
# ---------------------------------------------------------------------------


class TestComputeGapStats:
    """Tests for observation gap statistics."""

    def test_all_valid_no_gaps(self) -> None:
        """All valid -> max_gap=0, gap_fraction=0.0."""
        T, H, W = 5, 3, 3
        validity = np.ones((T, H, W), dtype=np.float32)

        stats = compute_gap_stats(validity)

        assert stats["max_gap_overall"] == 0
        assert stats["gap_fraction"] == 0.0
        assert stats["mean_gap_length"] == 0.0
        np.testing.assert_array_equal(stats["max_gap_per_pixel"], 0)

    def test_all_invalid_full_gap(self) -> None:
        """All invalid -> max_gap=T, gap_fraction=1.0."""
        T, H, W = 5, 2, 2
        validity = np.zeros((T, H, W), dtype=np.float32)

        stats = compute_gap_stats(validity)

        assert stats["max_gap_overall"] == T
        assert stats["gap_fraction"] == pytest.approx(1.0)
        np.testing.assert_array_equal(stats["max_gap_per_pixel"], T)

    def test_single_3hour_gap(self) -> None:
        """Single contiguous 3-hour gap."""
        T, H, W = 8, 1, 1
        validity = np.ones((T, H, W), dtype=np.float32)
        validity[2:5, 0, 0] = 0.0  # 3-hour gap at t=2,3,4

        stats = compute_gap_stats(validity)

        assert stats["max_gap_overall"] == 3
        assert stats["max_gap_per_pixel"][0, 0] == 3
        assert stats["mean_gap_length"] == pytest.approx(3.0)

    def test_multiple_gaps_max_is_longest(self) -> None:
        """Multiple gaps -> max_gap is the longest one."""
        T, H, W = 10, 1, 1
        validity = np.ones((T, H, W), dtype=np.float32)
        validity[1:3, 0, 0] = 0.0  # 2-hour gap
        validity[5:9, 0, 0] = 0.0  # 4-hour gap

        stats = compute_gap_stats(validity)

        assert stats["max_gap_overall"] == 4
        assert stats["max_gap_per_pixel"][0, 0] == 4

    def test_per_pixel_gaps_differ(self) -> None:
        """Different pixels have different gap patterns."""
        T, H, W = 8, 1, 2
        validity = np.ones((T, H, W), dtype=np.float32)
        validity[1:3, 0, 0] = 0.0  # pixel 0: 2-hour gap
        validity[1:6, 0, 1] = 0.0  # pixel 1: 5-hour gap

        stats = compute_gap_stats(validity)

        assert stats["max_gap_per_pixel"][0, 0] == 2
        assert stats["max_gap_per_pixel"][0, 1] == 5
        assert stats["max_gap_overall"] == 5

    def test_mean_gap_length(self) -> None:
        """Mean gap length computed correctly across multiple gaps."""
        T, H, W = 10, 1, 1
        validity = np.ones((T, H, W), dtype=np.float32)
        # Two gaps: length 2 and length 4
        validity[1:3, 0, 0] = 0.0  # gap of 2
        validity[5:9, 0, 0] = 0.0  # gap of 4

        stats = compute_gap_stats(validity)

        # Mean of [2, 4] = 3.0
        assert stats["mean_gap_length"] == pytest.approx(3.0)

    def test_performance_large_array(self) -> None:
        """Gap stats on a large array should complete in under 2 seconds."""
        import time

        T, H, W = 100, 128, 128
        validity = np.ones((T, H, W), dtype=np.float32)
        # Scatter some gaps
        rng = np.random.default_rng(42)
        validity[rng.random((T, H, W)) < 0.1] = 0.0

        start = time.perf_counter()
        stats = compute_gap_stats(validity)
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"compute_gap_stats took {elapsed:.2f}s on {T}x{H}x{W} array"
        assert stats["gap_fraction"] > 0
        assert stats["max_gap_overall"] > 0
