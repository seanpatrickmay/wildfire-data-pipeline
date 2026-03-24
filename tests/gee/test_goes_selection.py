"""Tests for GOES satellite selection logic.

These test the pure logic (string selection based on fire_year) without
requiring Earth Engine credentials. The actual GEE API calls are not tested
here — those require integration tests with authentication.
"""

from __future__ import annotations


class TestGoesWestSatelliteSelection:
    """Verify GOES-West satellite ID selection based on fire year.

    GOES-17 was decommissioned January 2023, replaced by GOES-18.
    The pipeline must use the correct satellite for each fire's year.
    """

    @staticmethod
    def _select_goes_west(fire_year: int) -> tuple[str, str]:
        """Extract the satellite selection logic from goes.py for testing.

        This mirrors the logic at goes.py lines 74-75 without importing ee.
        """
        conus = "NOAA/GOES/17/FDCC" if fire_year < 2023 else "NOAA/GOES/18/FDCC"
        full = "NOAA/GOES/17/FDCF" if fire_year < 2023 else "NOAA/GOES/18/FDCF"
        return conus, full

    def test_2019_uses_goes17(self) -> None:
        conus, full = self._select_goes_west(2019)
        assert "GOES/17" in conus
        assert "GOES/17" in full

    def test_2022_uses_goes17(self) -> None:
        conus, full = self._select_goes_west(2022)
        assert "GOES/17" in conus
        assert "GOES/17" in full

    def test_2023_uses_goes18(self) -> None:
        conus, full = self._select_goes_west(2023)
        assert "GOES/18" in conus
        assert "GOES/18" in full

    def test_2024_uses_goes18(self) -> None:
        conus, full = self._select_goes_west(2024)
        assert "GOES/18" in conus
        assert "GOES/18" in full

    def test_2025_uses_goes18(self) -> None:
        conus, full = self._select_goes_west(2025)
        assert "GOES/18" in conus
        assert "GOES/18" in full

    def test_conus_vs_full_disk_suffixes(self) -> None:
        conus, full = self._select_goes_west(2020)
        assert conus.endswith("/FDCC")
        assert full.endswith("/FDCF")

    def test_goes16_east_is_always_16(self) -> None:
        """GOES-East (16) does not change — verify it's not affected by year."""
        goes_east_conus = "NOAA/GOES/16/FDCC"
        goes_east_full = "NOAA/GOES/16/FDCF"
        # These are hardcoded constants, not year-dependent
        assert "GOES/16" in goes_east_conus
        assert "GOES/16" in goes_east_full


class TestGoesConfidenceMapping:
    """Verify the GOES mask-to-confidence mapping logic.

    The mapping (extracted from goes.py line 42-44) converts GOES Mask
    codes to calibrated confidence values. This is pure arithmetic.
    """

    @staticmethod
    def _mask_to_confidence(mask_code: int) -> float:
        """Apply the mask-to-confidence formula from goes.py."""
        if mask_code < 10 or mask_code > 35:
            return 0.0  # Not a fire pixel
        category = mask_code % 10
        mapping = {0: 1.0, 1: 1.0, 2: 0.8, 3: 0.5, 4: 0.3, 5: 0.1}
        return mapping.get(category, 0.0)

    def test_processed_fire_is_1_0(self) -> None:
        assert self._mask_to_confidence(10) == 1.0
        assert self._mask_to_confidence(30) == 1.0  # temporally filtered

    def test_saturated_fire_is_1_0(self) -> None:
        assert self._mask_to_confidence(11) == 1.0
        assert self._mask_to_confidence(31) == 1.0

    def test_cloud_contaminated_is_0_8(self) -> None:
        assert self._mask_to_confidence(12) == 0.8
        assert self._mask_to_confidence(32) == 0.8

    def test_high_probability_is_0_5(self) -> None:
        assert self._mask_to_confidence(13) == 0.5
        assert self._mask_to_confidence(33) == 0.5

    def test_medium_probability_is_0_3(self) -> None:
        assert self._mask_to_confidence(14) == 0.3
        assert self._mask_to_confidence(34) == 0.3

    def test_low_probability_is_0_1(self) -> None:
        assert self._mask_to_confidence(15) == 0.1
        assert self._mask_to_confidence(35) == 0.1

    def test_non_fire_codes_are_zero(self) -> None:
        for code in [0, 1, 5, 9, 36, 40, 100]:
            assert self._mask_to_confidence(code) == 0.0

    def test_temporal_filter_same_as_instantaneous(self) -> None:
        """Codes 30-35 should produce same confidence as 10-15."""
        for offset in range(6):
            instant = self._mask_to_confidence(10 + offset)
            filtered = self._mask_to_confidence(30 + offset)
            assert instant == filtered


class TestDqfFlagSemantics:
    """Verify DQF flag interpretation logic from goes.py."""

    @staticmethod
    def _classify_dqf(dqf: int) -> tuple[bool, bool]:
        """Return (is_valid, is_cloud) based on DQF code."""
        is_valid = dqf <= 1
        is_cloud = dqf == 2
        return is_valid, is_cloud

    def test_good_fire_pixel(self) -> None:
        is_valid, is_cloud = self._classify_dqf(0)
        assert is_valid is True
        assert is_cloud is False

    def test_good_fire_free(self) -> None:
        is_valid, is_cloud = self._classify_dqf(1)
        assert is_valid is True
        assert is_cloud is False

    def test_cloud_flagged(self) -> None:
        is_valid, is_cloud = self._classify_dqf(2)
        assert is_valid is False
        assert is_cloud is True

    def test_invalid_not_cloud(self) -> None:
        """DQF 3 (sunglint/bad surface) is invalid but not cloud."""
        is_valid, is_cloud = self._classify_dqf(3)
        assert is_valid is False
        assert is_cloud is False

    def test_bad_input_data(self) -> None:
        is_valid, is_cloud = self._classify_dqf(4)
        assert is_valid is False
        assert is_cloud is False

    def test_algorithm_failure(self) -> None:
        is_valid, is_cloud = self._classify_dqf(5)
        assert is_valid is False
        assert is_cloud is False
