"""Tests for data validation checks: validate_download, validate_labels."""

from __future__ import annotations

import numpy as np

from wildfire_pipeline.processing.validation import validate_download, validate_labels

# ---------------------------------------------------------------------------
# Helper: build valid arrays for validate_download
# ---------------------------------------------------------------------------


def _make_download_arrays(
    n_times: int = 24,
    n_height: int = 10,
    n_width: int = 10,
    *,
    confidence_fill: float = 0.5,
    frp_fill: float = 100.0,
) -> dict[str, np.ndarray]:
    """Return a dict of arrays that pass all validate_download checks."""
    shape = (n_times, n_height, n_width)
    return {
        "confidence": np.full(shape, confidence_fill, dtype=np.float32),
        "obs_valid": np.ones(shape, dtype=np.float32),
        "cloud_mask": np.zeros(shape, dtype=np.float32),
        "frp": np.full(shape, frp_fill, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Helper: build valid arrays for validate_labels
# ---------------------------------------------------------------------------


def _make_label_arrays(
    n_times: int = 24,
    n_height: int = 10,
    n_width: int = 10,
    *,
    fire_pixel_coords: list[tuple[int, int]] | None = None,
    fire_timesteps: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """Return a dict of arrays that pass all validate_labels checks.

    By default places a small fire region across all timesteps so the
    'no fire' warning is avoided.
    """
    shape = (n_times, n_height, n_width)
    labels = np.zeros(shape, dtype=np.float32)
    validity = np.ones(shape, dtype=np.float32)
    raw_confidence = np.full(shape, 0.5, dtype=np.float32)

    coords = fire_pixel_coords or [(0, 0), (0, 1)]
    timesteps = fire_timesteps or list(range(n_times))
    for t in timesteps:
        for r, c in coords:
            labels[t, r, c] = 1.0

    return {
        "labels": labels,
        "validity": validity,
        "raw_confidence": raw_confidence,
    }


# ---------------------------------------------------------------------------
# validate_download
# ---------------------------------------------------------------------------


class TestValidateDownloadPassing:
    """Cases where validate_download should pass with no errors."""

    def test_all_valid_data_passes(self) -> None:
        arrays = _make_download_arrays()
        result = validate_download(**arrays)
        assert result.passed is True
        assert result.errors == []

    def test_zero_confidence_and_frp_passes(self) -> None:
        arrays = _make_download_arrays(confidence_fill=0.0, frp_fill=0.0)
        result = validate_download(**arrays)
        assert result.passed is True

    def test_boundary_confidence_values_pass(self) -> None:
        arrays = _make_download_arrays()
        arrays["confidence"][0, 0, 0] = 0.0
        arrays["confidence"][1, 0, 0] = 1.0
        result = validate_download(**arrays)
        assert result.passed is True


class TestValidateDownloadErrors:
    """Cases where validate_download should report errors."""

    def test_shape_mismatch_reports_error(self) -> None:
        arrays = _make_download_arrays()
        arrays["frp"] = np.zeros((24, 5, 5), dtype=np.float32)
        result = validate_download(**arrays)
        assert result.passed is False
        assert any("Shape mismatch" in e for e in result.errors)

    def test_shape_mismatch_returns_early(self) -> None:
        """Shape mismatch should short-circuit; only 1 error expected."""
        arrays = _make_download_arrays()
        arrays["frp"] = np.zeros((24, 5, 5), dtype=np.float32)
        # Also inject NaN which would be caught by later checks
        arrays["confidence"][0, 0, 0] = np.nan
        result = validate_download(**arrays)
        assert len(result.errors) == 1

    def test_nan_in_confidence_reports_error(self) -> None:
        arrays = _make_download_arrays()
        arrays["confidence"][5, 3, 3] = np.nan
        result = validate_download(**arrays)
        assert result.passed is False
        assert any("confidence contains NaN" in e for e in result.errors)

    def test_nan_in_frp_reports_error(self) -> None:
        arrays = _make_download_arrays()
        arrays["frp"][0, 0, 0] = np.nan
        result = validate_download(**arrays)
        assert result.passed is False
        assert any("frp contains NaN" in e for e in result.errors)

    def test_nan_in_obs_valid_reports_error(self) -> None:
        arrays = _make_download_arrays()
        arrays["obs_valid"][0, 0, 0] = np.nan
        result = validate_download(**arrays)
        assert result.passed is False
        assert any("obs_valid contains NaN" in e for e in result.errors)

    def test_nan_in_cloud_mask_reports_error(self) -> None:
        arrays = _make_download_arrays()
        arrays["cloud_mask"][0, 0, 0] = np.nan
        result = validate_download(**arrays)
        assert result.passed is False
        assert any("cloud_mask contains NaN" in e for e in result.errors)

    def test_inf_in_frp_reports_error(self) -> None:
        arrays = _make_download_arrays()
        arrays["frp"][2, 1, 1] = np.inf
        result = validate_download(**arrays)
        assert result.passed is False
        assert any("frp contains Inf" in e for e in result.errors)

    def test_inf_in_confidence_reports_error(self) -> None:
        arrays = _make_download_arrays()
        arrays["confidence"][0, 0, 0] = np.inf
        result = validate_download(**arrays)
        assert result.passed is False
        assert any("confidence contains Inf" in e for e in result.errors)

    def test_negative_inf_reports_error(self) -> None:
        arrays = _make_download_arrays()
        arrays["frp"][0, 0, 0] = -np.inf
        result = validate_download(**arrays)
        assert result.passed is False
        assert any("frp contains Inf" in e for e in result.errors)

    def test_confidence_above_one_reports_error(self) -> None:
        arrays = _make_download_arrays()
        arrays["confidence"][0, 0, 0] = 1.01
        result = validate_download(**arrays)
        assert result.passed is False
        assert any("Confidence out of [0,1]" in e for e in result.errors)

    def test_confidence_below_zero_reports_error(self) -> None:
        arrays = _make_download_arrays()
        arrays["confidence"][0, 0, 0] = -0.01
        result = validate_download(**arrays)
        assert result.passed is False
        assert any("Confidence out of [0,1]" in e for e in result.errors)

    def test_negative_frp_reports_error(self) -> None:
        arrays = _make_download_arrays()
        arrays["frp"][0, 0, 0] = -1.0
        result = validate_download(**arrays)
        assert result.passed is False
        assert any("FRP has negative values" in e for e in result.errors)

    def test_too_many_failed_hours_reports_error(self) -> None:
        """More than 10% failed hours should be an error."""
        T = 100
        arrays = _make_download_arrays(n_times=T)
        failed = list(range(11))  # 11/100 = 11%
        result = validate_download(**arrays, failed_hours=failed)
        assert result.passed is False
        assert any("Too many failed hours" in e for e in result.errors)

    def test_over_50_percent_invalid_spatial_coverage_reports_error(self) -> None:
        """More than 50% of hours with zero valid observations."""
        T = 10
        arrays = _make_download_arrays(n_times=T)
        arrays["obs_valid"][:6] = 0.0  # 6/10 = 60% fully invalid
        result = validate_download(**arrays)
        assert result.passed is False
        assert any("Over 50% of hours have no valid observations" in e for e in result.errors)


class TestValidateDownloadWarnings:
    """Cases where validate_download should warn but still pass."""

    def test_high_frp_warns(self) -> None:
        arrays = _make_download_arrays()
        arrays["frp"][0, 0, 0] = 15000.0
        result = validate_download(**arrays)
        assert result.passed is True
        assert any("FRP unusually high" in w for w in result.warnings)

    def test_some_failed_hours_warns(self) -> None:
        """Between 5% and 10% failed hours should warn."""
        T = 100
        arrays = _make_download_arrays(n_times=T)
        failed = list(range(8))  # 8/100 = 8%
        result = validate_download(**arrays, failed_hours=failed)
        assert result.passed is True
        assert any("Some hours failed" in w for w in result.warnings)

    def test_exactly_5_percent_failure_no_warning(self) -> None:
        """Exactly 5% should not trigger the >5% warning."""
        T = 100
        arrays = _make_download_arrays(n_times=T)
        failed = list(range(5))  # 5/100 = 5%
        result = validate_download(**arrays, failed_hours=failed)
        assert result.passed is True
        assert not any("Some hours failed" in w for w in result.warnings)

    def test_partial_spatial_invalidity_warns(self) -> None:
        """Between 20% and 50% hours with no valid obs should warn."""
        T = 10
        arrays = _make_download_arrays(n_times=T)
        arrays["obs_valid"][:3] = 0.0  # 3/10 = 30%
        result = validate_download(**arrays)
        assert result.passed is True
        assert any("hours have no valid observations" in w for w in result.warnings)

    def test_no_failed_hours_no_warning(self) -> None:
        arrays = _make_download_arrays()
        result = validate_download(**arrays, failed_hours=None)
        assert result.passed is True
        assert result.warnings == []

    def test_empty_failed_hours_no_warning(self) -> None:
        arrays = _make_download_arrays()
        result = validate_download(**arrays, failed_hours=[])
        assert result.passed is True
        assert result.warnings == []


class TestValidateDownloadMultipleIssues:
    """Validate that multiple issues are accumulated correctly."""

    def test_nan_and_range_errors_both_reported(self) -> None:
        arrays = _make_download_arrays()
        arrays["confidence"][0, 0, 0] = np.nan
        arrays["frp"][0, 0, 0] = -5.0
        result = validate_download(**arrays)
        assert result.passed is False
        assert len(result.errors) >= 2


# ---------------------------------------------------------------------------
# Consecutive gap detection in validate_download
# ---------------------------------------------------------------------------


class TestConsecutiveGapDetection:
    """Tests for the consecutive invalid gap length warnings."""

    def test_no_gaps_no_warning(self) -> None:
        arrays = _make_download_arrays(n_times=20)
        result = validate_download(**arrays)
        gap_warnings = [w for w in result.warnings if "gap" in w.lower()]
        assert len(gap_warnings) == 0

    def test_short_gap_no_warning(self) -> None:
        """A 5-hour gap should not trigger any warning (threshold is 6)."""
        arrays = _make_download_arrays(n_times=20)
        arrays["obs_valid"][5:10] = 0.0  # 5-hour gap
        result = validate_download(**arrays)
        gap_warnings = [w for w in result.warnings if "gap" in w.lower()]
        assert len(gap_warnings) == 0

    def test_7_hour_gap_triggers_notable_warning(self) -> None:
        arrays = _make_download_arrays(n_times=20)
        arrays["obs_valid"][5:12] = 0.0  # 7-hour gap
        result = validate_download(**arrays)
        gap_warnings = [w for w in result.warnings if "notable" in w.lower() and "gap" in w.lower()]
        assert len(gap_warnings) == 1

    def test_13_hour_gap_triggers_longest_warning(self) -> None:
        arrays = _make_download_arrays(n_times=30)
        arrays["obs_valid"][5:18] = 0.0  # 13-hour gap
        result = validate_download(**arrays)
        gap_warnings = [w for w in result.warnings if "longest" in w.lower() and "gap" in w.lower()]
        assert len(gap_warnings) == 1
        assert "13" in gap_warnings[0]

    def test_per_pixel_gap_detection(self) -> None:
        """Gap at a single pixel should still be detected."""
        arrays = _make_download_arrays(n_times=20, n_height=3, n_width=3)
        # 8-hour gap at pixel (1,1) only
        arrays["obs_valid"][2:10, 1, 1] = 0.0
        result = validate_download(**arrays)
        gap_warnings = [w for w in result.warnings if "gap" in w.lower()]
        assert len(gap_warnings) == 1

    def test_single_timestep_skips_gap_check(self) -> None:
        """T=1 should skip gap detection (needs T>1)."""
        arrays = _make_download_arrays(n_times=1)
        arrays["obs_valid"][:] = 0.0
        result = validate_download(**arrays)
        gap_warnings = [w for w in result.warnings if "gap" in w.lower()]
        assert len(gap_warnings) == 0


# ---------------------------------------------------------------------------
# FRP distribution checks in validate_download
# ---------------------------------------------------------------------------


class TestFrpDistributionChecks:
    """Tests for FRP saturation and distribution warnings."""

    def test_normal_frp_no_warning(self) -> None:
        arrays = _make_download_arrays(frp_fill=200.0)
        result = validate_download(**arrays)
        frp_warnings = [w for w in result.warnings if "frp" in w.lower() or "saturat" in w.lower()]
        assert len(frp_warnings) == 0

    def test_frp_above_saturation_warns(self) -> None:
        """Pixels with FRP > 5000 MW should trigger saturation warning."""
        arrays = _make_download_arrays(n_times=20)
        arrays["frp"][0, 0, 0] = 6000.0
        result = validate_download(**arrays)
        sat_warnings = [w for w in result.warnings if "saturation" in w.lower()]
        assert len(sat_warnings) >= 1

    def test_frp_p99_above_saturation_warns(self) -> None:
        """If the 99th percentile FRP exceeds 5000, warn about p99."""
        arrays = _make_download_arrays(n_times=20, n_height=5, n_width=5)
        # Need >10 fire pixels (confidence > 0) for the check to activate
        arrays["confidence"][:] = 0.0
        arrays["frp"][:] = 0.0
        # 15 fire pixels with moderate FRP + 1 extreme outlier
        for t in range(16):
            arrays["confidence"][t, 0, 0] = 0.5
            arrays["frp"][t, 0, 0] = 100.0
        arrays["frp"][15, 0, 0] = 6000.0  # push p99 above 5000
        result = validate_download(**arrays)
        p99_warnings = [w for w in result.warnings if "p99" in w.lower()]
        assert len(p99_warnings) == 1

    def test_few_fire_pixels_skips_frp_distribution(self) -> None:
        """With <= 10 fire FRP pixels, distribution check is skipped."""
        arrays = _make_download_arrays(n_times=5, n_height=2, n_width=2)
        arrays["confidence"][:] = 0.0  # No fire
        arrays["frp"][:] = 0.0
        # Only 5 fire pixels (< 10 threshold)
        for t in range(5):
            arrays["confidence"][t, 0, 0] = 0.5
            arrays["frp"][t, 0, 0] = 8000.0  # Very high but too few to trigger
        result = validate_download(**arrays)
        p99_warnings = [w for w in result.warnings if "p99" in w.lower()]
        assert len(p99_warnings) == 0


# ---------------------------------------------------------------------------
# Confidence-FRP cross-check in validate_download
# ---------------------------------------------------------------------------


class TestConfidenceFrpCrossCheck:
    """Tests for the confidence-FRP cross-validation."""

    def test_consistent_data_no_warning(self) -> None:
        """FRP > 0 only where confidence > 0 should produce no warning."""
        arrays = _make_download_arrays()
        result = validate_download(**arrays)
        suspicious_warnings = [w for w in result.warnings if "suspicious" in w.lower()]
        assert len(suspicious_warnings) == 0

    def test_frp_without_confidence_is_suspicious(self) -> None:
        """Pixels with FRP > 0 but confidence == 0 should warn."""
        arrays = _make_download_arrays()
        # Set one pixel: FRP present but confidence absent
        arrays["confidence"][3, 2, 2] = 0.0
        arrays["frp"][3, 2, 2] = 150.0
        result = validate_download(**arrays)
        suspicious_warnings = [w for w in result.warnings if "suspicious" in w.lower()]
        assert len(suspicious_warnings) == 1
        assert "1 pixels" in suspicious_warnings[0] or "1 pixel" in suspicious_warnings[0]

    def test_multiple_suspicious_pixels_reported(self) -> None:
        arrays = _make_download_arrays(n_times=10)
        arrays["confidence"][0, 0, 0] = 0.0
        arrays["frp"][0, 0, 0] = 50.0
        arrays["confidence"][1, 1, 1] = 0.0
        arrays["frp"][1, 1, 1] = 75.0
        arrays["confidence"][2, 2, 2] = 0.0
        arrays["frp"][2, 2, 2] = 100.0
        result = validate_download(**arrays)
        suspicious_warnings = [w for w in result.warnings if "suspicious" in w.lower()]
        assert len(suspicious_warnings) == 1
        assert "3 pixels" in suspicious_warnings[0]

    def test_zero_frp_with_zero_confidence_not_suspicious(self) -> None:
        """FRP == 0 with confidence == 0 is normal (no fire), not suspicious."""
        arrays = _make_download_arrays()
        arrays["confidence"][0, 0, 0] = 0.0
        arrays["frp"][0, 0, 0] = 0.0
        result = validate_download(**arrays)
        suspicious_warnings = [w for w in result.warnings if "suspicious" in w.lower()]
        assert len(suspicious_warnings) == 0


# ---------------------------------------------------------------------------
# validate_labels
# ---------------------------------------------------------------------------


class TestValidateLabelsPassing:
    """Cases where validate_labels should pass with no errors."""

    def test_all_valid_labels_pass(self) -> None:
        arrays = _make_label_arrays()
        result = validate_labels(**arrays)
        assert result.passed is True
        assert result.errors == []

    def test_all_zeros_passes_with_warning(self) -> None:
        """All-zero labels should pass (just warns about no fire)."""
        arrays = _make_label_arrays()
        arrays["labels"][:] = 0.0
        result = validate_labels(**arrays)
        assert result.passed is True

    def test_single_timestep_passes(self) -> None:
        arrays = _make_label_arrays(n_times=1)
        result = validate_labels(**arrays)
        assert result.passed is True


class TestValidateLabelsErrors:
    """Cases where validate_labels should report errors."""

    def test_label_validity_shape_mismatch_reports_error(self) -> None:
        arrays = _make_label_arrays()
        arrays["validity"] = np.ones((12, 10, 10), dtype=np.float32)
        result = validate_labels(**arrays)
        assert result.passed is False
        assert any("Label/validity shape mismatch" in e for e in result.errors)

    def test_label_confidence_shape_mismatch_reports_error(self) -> None:
        arrays = _make_label_arrays()
        arrays["raw_confidence"] = np.ones((12, 10, 10), dtype=np.float32)
        result = validate_labels(**arrays)
        assert result.passed is False
        assert any("Label/confidence shape mismatch" in e for e in result.errors)

    def test_label_validity_mismatch_returns_early(self) -> None:
        """Shape mismatch between labels and validity should short-circuit."""
        arrays = _make_label_arrays()
        arrays["validity"] = np.ones((12, 10, 10), dtype=np.float32)
        # Also make labels non-binary, which would fail later checks
        arrays["labels"][0, 0, 0] = 0.5
        result = validate_labels(**arrays)
        assert len(result.errors) == 1

    def test_non_binary_labels_reports_error(self) -> None:
        arrays = _make_label_arrays()
        arrays["labels"][0, 0, 0] = 0.5
        result = validate_labels(**arrays)
        assert result.passed is False
        assert any("Labels not binary" in e for e in result.errors)

    def test_negative_labels_reports_error(self) -> None:
        arrays = _make_label_arrays()
        arrays["labels"][0, 0, 0] = -1.0
        result = validate_labels(**arrays)
        assert result.passed is False
        assert any("Labels not binary" in e for e in result.errors)

    def test_validity_above_one_reports_error(self) -> None:
        arrays = _make_label_arrays()
        arrays["validity"][0, 0, 0] = 1.5
        result = validate_labels(**arrays)
        assert result.passed is False
        assert any("Validity out of [0,1]" in e for e in result.errors)

    def test_validity_below_zero_reports_error(self) -> None:
        arrays = _make_label_arrays()
        arrays["validity"][0, 0, 0] = -0.1
        result = validate_labels(**arrays)
        assert result.passed is False
        assert any("Validity out of [0,1]" in e for e in result.errors)


class TestValidateLabelsWarnings:
    """Cases where validate_labels should warn but still pass."""

    def test_no_fire_pixels_warns(self) -> None:
        arrays = _make_label_arrays()
        arrays["labels"][:] = 0.0
        result = validate_labels(**arrays)
        assert result.passed is True
        assert any("No fire pixels" in w for w in result.warnings)

    def test_high_fire_fraction_warns(self) -> None:
        """Fire in >50% of pixels at a single timestep should warn."""
        arrays = _make_label_arrays(n_times=5, n_height=10, n_width=10)
        arrays["labels"][2, :6, :] = 1.0  # 60/100 = 60% at t=2
        result = validate_labels(**arrays)
        assert result.passed is True
        assert any("Fire fraction exceeds 50%" in w for w in result.warnings)

    def test_high_cloud_exclusion_warns(self) -> None:
        """Over 60% of pixels excluded should warn."""
        arrays = _make_label_arrays()
        arrays["validity"][:] = 0.3  # mean=0.3 -> excluded=70%
        result = validate_labels(**arrays)
        assert result.passed is True
        assert any("Over 60% of pixels excluded" in w for w in result.warnings)

    def test_high_flicker_rate_warns(self) -> None:
        """Fire that appears and disappears rapidly should warn."""
        T, H, W = 10, 4, 4
        labels = np.zeros((T, H, W), dtype=np.float32)
        validity = np.ones((T, H, W), dtype=np.float32)
        raw_confidence = np.full((T, H, W), 0.5, dtype=np.float32)

        # Alternating fire/no-fire at every pixel -> very high flicker
        for t in range(0, T, 2):
            labels[t, :, :] = 1.0

        result = validate_labels(labels, validity, raw_confidence)
        assert result.passed is True
        assert any("High flicker rate" in w for w in result.warnings)

    def test_no_flicker_no_warning(self) -> None:
        """Sustained fire should not trigger flicker warning."""
        arrays = _make_label_arrays(n_times=10)
        # Fire at (0,0) for all timesteps: never turns off -> flicker=0
        result = validate_labels(**arrays)
        assert not any("flicker" in w.lower() for w in result.warnings)

    def test_single_timestep_skips_flicker_check(self) -> None:
        """With T=1 flicker check should be skipped, no crash."""
        arrays = _make_label_arrays(n_times=1)
        result = validate_labels(**arrays)
        assert result.passed is True
        assert not any("flicker" in w.lower() for w in result.warnings)


class TestValidateLabelsEdgeCases:
    """Edge cases and boundary conditions for validate_labels."""

    def test_fire_fraction_exactly_50_percent_no_warning(self) -> None:
        """Exactly 50% fire should NOT trigger the >50% warning."""
        T, H, W = 2, 10, 10
        labels = np.zeros((T, H, W), dtype=np.float32)
        validity = np.ones((T, H, W), dtype=np.float32)
        raw_confidence = np.full((T, H, W), 0.5, dtype=np.float32)

        labels[0, :5, :] = 1.0  # exactly 50/100 = 50%

        result = validate_labels(labels, validity, raw_confidence)
        assert not any("Fire fraction exceeds 50%" in w for w in result.warnings)

    def test_cloud_exclusion_exactly_60_percent_no_warning(self) -> None:
        """Exactly 60% excluded should NOT trigger the >60% warning."""
        arrays = _make_label_arrays()
        arrays["validity"][:] = 0.4  # excluded = 60%
        result = validate_labels(**arrays)
        assert not any("Over 60% of pixels excluded" in w for w in result.warnings)

    def test_all_ones_labels_warns_high_fire(self) -> None:
        arrays = _make_label_arrays()
        arrays["labels"][:] = 1.0
        result = validate_labels(**arrays)
        assert result.passed is True
        assert any("Fire fraction exceeds 50%" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# ValidationResult dataclass
# ---------------------------------------------------------------------------


class TestValidationResult:
    """Direct tests for the ValidationResult dataclass."""

    def test_default_state_is_passed(self) -> None:
        from wildfire_pipeline.processing.validation import ValidationResult

        vr = ValidationResult()
        assert vr.passed is True
        assert vr.warnings == []
        assert vr.errors == []

    def test_add_warning_does_not_fail(self) -> None:
        from wildfire_pipeline.processing.validation import ValidationResult

        vr = ValidationResult()
        vr.add_warning("test warning")
        assert vr.passed is True
        assert "test warning" in vr.warnings

    def test_add_error_sets_passed_false(self) -> None:
        from wildfire_pipeline.processing.validation import ValidationResult

        vr = ValidationResult()
        vr.add_error("test error")
        assert vr.passed is False
        assert "test error" in vr.errors

    def test_multiple_errors_accumulate(self) -> None:
        from wildfire_pipeline.processing.validation import ValidationResult

        vr = ValidationResult()
        vr.add_error("error 1")
        vr.add_error("error 2")
        assert len(vr.errors) == 2
        assert vr.passed is False


# ---------------------------------------------------------------------------
# Temporal monotonicity checks in validate_labels
# ---------------------------------------------------------------------------


class TestTemporalMonotonicity:
    """Tests for the temporal monotonicity warning in validate_labels."""

    def test_steadily_growing_fire_no_warning(self) -> None:
        """Fire that only grows should produce no monotonicity warning."""
        T, H, W = 10, 5, 5
        labels = np.zeros((T, H, W), dtype=np.float32)
        # Fire grows each timestep: 1 pixel, 2 pixels, 3, ...
        for t in range(T):
            for i in range(min(t + 1, H)):
                labels[t, i, 0] = 1.0
        arrs = {"labels": labels, "validity": np.ones_like(labels), "raw_confidence": labels * 0.8}
        result = validate_labels(**arrs)
        mono_warnings = [
            w for w in result.warnings if "dropped" in w.lower() or "monoton" in w.lower()
        ]
        assert len(mono_warnings) == 0

    def test_large_drop_triggers_warning(self) -> None:
        """Fire area dropping >50% from peak should warn."""
        T, H, W = 10, 5, 5
        labels = np.zeros((T, H, W), dtype=np.float32)
        # Fire grows to 10 pixels at t=4, then drops to 2 pixels at t=5
        for t in range(5):
            for i in range(min(t * 2 + 2, H)):
                for j in range(min(2, W)):
                    labels[t, i, j] = 1.0
        # t=5: only 2 fire pixels (>50% drop from peak ~10)
        labels[5, 0, 0] = 1.0
        labels[5, 0, 1] = 1.0
        # t=6+: keep small fire so not all zeros
        for t in range(6, T):
            labels[t, 0, 0] = 1.0

        arrs = {"labels": labels, "validity": np.ones_like(labels), "raw_confidence": labels * 0.8}
        result = validate_labels(**arrs)
        mono_warnings = [w for w in result.warnings if "dropped" in w.lower()]
        assert len(mono_warnings) == 1
        assert "timestep" in mono_warnings[0].lower()

    def test_small_drop_no_warning(self) -> None:
        """A 30% drop should NOT trigger the >50% monotonicity warning."""
        T, H, W = 6, 10, 1
        labels = np.zeros((T, H, W), dtype=np.float32)
        # 10 fire pixels at peak, then 7 (30% drop)
        for t in range(3):
            labels[t, :10, 0] = 1.0
        for t in range(3, T):
            labels[t, :7, 0] = 1.0

        arrs = {"labels": labels, "validity": np.ones_like(labels), "raw_confidence": labels * 0.8}
        result = validate_labels(**arrs)
        mono_warnings = [w for w in result.warnings if "dropped" in w.lower()]
        assert len(mono_warnings) == 0

    def test_drop_in_cloudy_hours_ignored(self) -> None:
        """If the drop happens during fully cloudy hours, it should be ignored."""
        T, H, W = 8, 5, 5
        labels = np.zeros((T, H, W), dtype=np.float32)
        validity = np.ones((T, H, W), dtype=np.float32)

        # Fire at 10 pixels for t=0-3
        for t in range(4):
            labels[t, :2, :] = 1.0
        # t=4: cloudy (no valid obs) — fire labels drop to 0 but validity is 0
        labels[4] = 0.0
        validity[4] = 0.0
        # t=5-7: fire back at 8 pixels (within 50% of peak 10)
        for t in range(5, T):
            labels[t, :2, :4] = 1.0

        arrs = {"labels": labels, "validity": validity, "raw_confidence": labels * 0.8}
        result = validate_labels(**arrs)
        mono_warnings = [w for w in result.warnings if "dropped" in w.lower()]
        assert len(mono_warnings) == 0

    def test_no_fire_skips_monotonicity(self) -> None:
        """With zero fire pixels, temporal monotonicity check is skipped."""
        arrs = _make_label_arrays(n_times=5, fire_pixel_coords=[], fire_timesteps=[])
        # Override: remove all fire
        arrs["labels"] = np.zeros_like(arrs["labels"])
        result = validate_labels(**arrs)
        mono_warnings = [w for w in result.warnings if "dropped" in w.lower()]
        assert len(mono_warnings) == 0

    def test_single_timestep_skips_monotonicity(self) -> None:
        """T=1 should skip monotonicity check (needs T>2)."""
        labels = np.ones((1, 3, 3), dtype=np.float32)
        arrs = {"labels": labels, "validity": np.ones_like(labels), "raw_confidence": labels}
        result = validate_labels(**arrs)
        mono_warnings = [w for w in result.warnings if "dropped" in w.lower()]
        assert len(mono_warnings) == 0

    def test_two_timesteps_skips_monotonicity(self) -> None:
        """T=2 should skip monotonicity check (needs T>2)."""
        T, H, W = 2, 5, 5
        labels = np.zeros((T, H, W), dtype=np.float32)
        labels[0, :, :] = 1.0  # all fire
        labels[1, 0, 0] = 1.0  # almost no fire (>50% drop)
        arrs = {"labels": labels, "validity": np.ones_like(labels), "raw_confidence": labels}
        result = validate_labels(**arrs)
        mono_warnings = [w for w in result.warnings if "dropped" in w.lower()]
        assert len(mono_warnings) == 0
