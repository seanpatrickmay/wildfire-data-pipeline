"""Tests for label processing functions: majority_vote_smooth, apply_cloud_masking."""

from __future__ import annotations

import numpy as np

from wildfire_pipeline.processing.labels import apply_cloud_masking, majority_vote_smooth

# ---------------------------------------------------------------------------
# majority_vote_smooth
# ---------------------------------------------------------------------------


class TestMajorityVoteSmooth:
    """Tests for temporal majority-vote smoothing."""

    def test_all_zeros_returns_all_zeros(self) -> None:
        binary = np.zeros((10, 4, 4), dtype=np.float32)
        result = majority_vote_smooth(binary, window=5, min_votes=2)
        np.testing.assert_array_equal(result, np.zeros_like(binary))

    def test_all_ones_returns_all_ones(self) -> None:
        """With min_votes=1, every timestep has >= 1 vote -> all ones."""
        binary = np.ones((10, 4, 4), dtype=np.float32)
        result = majority_vote_smooth(binary, window=5, min_votes=1)
        np.testing.assert_array_equal(result, np.ones_like(binary))

    def test_all_ones_early_timesteps_starved(self) -> None:
        """With min_votes=2, t=0 only has 1 vote even with all-ones input.

        The lookback window at t=0 is [0..0] (just 1 frame), so votes=1 < 2.
        Starting at t=1, the window [0..1] has 2 votes, so all remaining pass.
        """
        binary = np.ones((10, 2, 2), dtype=np.float32)
        result = majority_vote_smooth(binary, window=5, min_votes=2)

        # t=0: window has only 1 element -> votes=1 < 2 -> 0
        np.testing.assert_array_equal(result[0], np.zeros((2, 2), dtype=np.float32))
        # t=1 onward: window has >= 2 elements, all ones -> votes >= 2 -> 1
        np.testing.assert_array_equal(result[1:], np.ones((9, 2, 2), dtype=np.float32))

    def test_single_frame_fire_does_not_survive(self) -> None:
        """A single fire timestep cannot reach min_votes=2 in any window."""
        binary = np.zeros((10, 2, 2), dtype=np.float32)
        binary[5, :, :] = 1.0  # fire only at t=5

        result = majority_vote_smooth(binary, window=5, min_votes=2)

        # With only one detection, votes never reaches 2 in any window.
        np.testing.assert_array_equal(result, np.zeros_like(binary))

    def test_multi_frame_fire_survives(self) -> None:
        """Three consecutive fire frames should survive with window=5, min_votes=2."""
        binary = np.zeros((10, 2, 2), dtype=np.float32)
        binary[4, :, :] = 1.0
        binary[5, :, :] = 1.0
        binary[6, :, :] = 1.0

        result = majority_vote_smooth(binary, window=5, min_votes=2)

        # t=4: window [0..4], only t=4 has fire -> votes=1 < 2 -> 0
        assert result[4, 0, 0] == 0.0
        # t=5: window [1..5], t=4 and t=5 have fire -> votes=2 >= 2 -> 1
        assert result[5, 0, 0] == 1.0
        # t=6: window [2..6], t=4,5,6 have fire -> votes=3 >= 2 -> 1
        assert result[6, 0, 0] == 1.0
        # t=7: window [3..7], t=4,5,6 have fire -> votes=3 >= 2 -> 1
        assert result[7, 0, 0] == 1.0
        # t=8: window [4..8], t=4,5,6 have fire -> votes=3 >= 2 -> 1
        assert result[8, 0, 0] == 1.0
        # t=9: window [5..9], t=5,6 have fire -> votes=2 >= 2 -> 1
        assert result[9, 0, 0] == 1.0

    def test_flicker_pattern_fills_gaps(self) -> None:
        """fire, no-fire, fire, no-fire, fire with window=5, min_votes=2.

        The pattern [1,0,1,0,1] has enough density that the smoothing
        should fill in the gaps for middle timesteps.
        """
        binary = np.zeros((5, 1, 1), dtype=np.float32)
        binary[0, 0, 0] = 1.0
        binary[2, 0, 0] = 1.0
        binary[4, 0, 0] = 1.0

        result = majority_vote_smooth(binary, window=5, min_votes=2)

        # t=0: window [0..0], votes=1 < 2 -> 0
        assert result[0, 0, 0] == 0.0
        # t=1: window [0..1], votes=1 (t=0) < 2 -> 0
        assert result[1, 0, 0] == 0.0
        # t=2: window [0..2], votes=2 (t=0,t=2) >= 2 -> 1
        assert result[2, 0, 0] == 1.0
        # t=3: window [0..3], votes=2 (t=0,t=2) >= 2 -> 1
        assert result[3, 0, 0] == 1.0
        # t=4: window [0..4], votes=3 (t=0,t=2,t=4) >= 2 -> 1
        assert result[4, 0, 0] == 1.0

    def test_window_one_is_identity(self) -> None:
        """Window=1 means only the current timestep is considered.

        With min_votes=1 and window=1, output must equal input.
        """
        rng = np.random.default_rng(42)
        binary = rng.integers(0, 2, size=(8, 3, 3)).astype(np.float32)

        result = majority_vote_smooth(binary, window=1, min_votes=1)

        np.testing.assert_array_equal(result, binary)

    def test_min_votes_one_is_rolling_max(self) -> None:
        """min_votes=1 means any detection in the window survives.

        This is equivalent to a rolling max over the window.
        """
        binary = np.zeros((8, 1, 1), dtype=np.float32)
        binary[2, 0, 0] = 1.0  # single fire at t=2

        result = majority_vote_smooth(binary, window=4, min_votes=1)

        # t=0,1: window has no fire -> 0
        assert result[0, 0, 0] == 0.0
        assert result[1, 0, 0] == 0.0
        # t=2: window [0..2] includes t=2 -> 1
        assert result[2, 0, 0] == 1.0
        # t=3: window [0..3] includes t=2 -> 1
        assert result[3, 0, 0] == 1.0
        # t=4: window [1..4] includes t=2 -> 1
        assert result[4, 0, 0] == 1.0
        # t=5: window [2..5] includes t=2 -> 1
        assert result[5, 0, 0] == 1.0
        # t=6: window [3..6] does NOT include t=2 -> 0
        assert result[6, 0, 0] == 0.0
        # t=7: window [4..7] does NOT include t=2 -> 0
        assert result[7, 0, 0] == 0.0

    def test_spatial_independence(self) -> None:
        """Smoothing operates per-pixel: adjacent pixels don't influence each other."""
        binary = np.zeros((6, 1, 2), dtype=np.float32)
        # Pixel (0,0): fire at t=2,3 -> should survive with min_votes=2
        binary[2, 0, 0] = 1.0
        binary[3, 0, 0] = 1.0
        # Pixel (0,1): fire only at t=2 -> should NOT survive with min_votes=2
        binary[2, 0, 1] = 1.0

        result = majority_vote_smooth(binary, window=5, min_votes=2)

        # Pixel (0,0): at t=3, window [0..3] has t=2 and t=3 -> votes=2 -> 1
        assert result[3, 0, 0] == 1.0
        # Pixel (0,1): never gets 2 votes in any window -> all 0
        np.testing.assert_array_equal(result[:, 0, 1], 0.0)

    def test_single_timestep(self) -> None:
        """T=1: only one frame, so smoothing just applies the threshold."""
        binary = np.ones((1, 2, 2), dtype=np.float32)
        result = majority_vote_smooth(binary, window=5, min_votes=1)
        np.testing.assert_array_equal(result, binary)

    def test_single_timestep_min_votes_exceeds(self) -> None:
        """T=1 with min_votes=2: impossible to reach 2 votes -> all zeros."""
        binary = np.ones((1, 2, 2), dtype=np.float32)
        result = majority_vote_smooth(binary, window=5, min_votes=2)
        np.testing.assert_array_equal(result, np.zeros_like(binary))

    def test_window_larger_than_t(self) -> None:
        """Window exceeds T: clamps to available data, still works correctly."""
        binary = np.zeros((3, 1, 1), dtype=np.float32)
        binary[0, 0, 0] = 1.0
        binary[2, 0, 0] = 1.0

        result = majority_vote_smooth(binary, window=100, min_votes=2)

        # t=0: window covers [0..0], votes=1 < 2 -> 0
        assert result[0, 0, 0] == 0.0
        # t=1: window covers [0..1], votes=1 < 2 -> 0
        assert result[1, 0, 0] == 0.0
        # t=2: window covers [0..2], votes=2 >= 2 -> 1
        assert result[2, 0, 0] == 1.0

    def test_output_dtype_is_float32(self) -> None:
        binary = np.zeros((4, 2, 2), dtype=np.float32)
        result = majority_vote_smooth(binary, window=3, min_votes=1)
        assert result.dtype == np.float32

    def test_output_shape_matches_input(self) -> None:
        binary = np.zeros((7, 5, 3), dtype=np.float32)
        result = majority_vote_smooth(binary, window=4, min_votes=2)
        assert result.shape == binary.shape


# ---------------------------------------------------------------------------
# apply_cloud_masking
# ---------------------------------------------------------------------------


class TestApplyCloudMasking:
    """Tests for combined validity mask creation."""

    def test_all_valid_no_clouds(self) -> None:
        obs_valid = np.ones((5, 3, 3), dtype=np.float32)
        cloud_mask = np.zeros((5, 3, 3), dtype=np.float32)

        result = apply_cloud_masking(obs_valid, cloud_mask)

        np.testing.assert_array_equal(result, np.ones_like(obs_valid))

    def test_all_invalid_observation(self) -> None:
        obs_valid = np.zeros((5, 3, 3), dtype=np.float32)
        cloud_mask = np.zeros((5, 3, 3), dtype=np.float32)

        result = apply_cloud_masking(obs_valid, cloud_mask)

        np.testing.assert_array_equal(result, np.zeros_like(obs_valid))

    def test_cloud_pixels_become_zero(self) -> None:
        """Even if obs_valid=1, cloud_mask=1 should produce validity=0."""
        obs_valid = np.ones((3, 2, 2), dtype=np.float32)
        cloud_mask = np.ones((3, 2, 2), dtype=np.float32)

        result = apply_cloud_masking(obs_valid, cloud_mask)

        np.testing.assert_array_equal(result, np.zeros_like(obs_valid))

    def test_mixed_validity_and_clouds(self) -> None:
        """Verify pixel-level correctness with a mixed scenario."""
        obs_valid = np.array([[[1, 1], [0, 1]]], dtype=np.float32)  # (1, 2, 2)
        cloud_mask = np.array([[[0, 1], [0, 0]]], dtype=np.float32)  # (1, 2, 2)

        result = apply_cloud_masking(obs_valid, cloud_mask)

        expected = np.array([[[1, 0], [0, 1]]], dtype=np.float32)
        # (0,0): 1*(1-0)=1, (0,1): 1*(1-1)=0, (1,0): 0*(1-0)=0, (1,1): 1*(1-0)=1
        np.testing.assert_array_equal(result, expected)

    def test_formula_is_obs_times_one_minus_cloud(self) -> None:
        """Directly verify the formula on non-binary values."""
        rng = np.random.default_rng(99)
        obs_valid = rng.random((4, 3, 3)).astype(np.float32)
        cloud_mask = rng.random((4, 3, 3)).astype(np.float32)

        result = apply_cloud_masking(obs_valid, cloud_mask)
        expected = (obs_valid * (1.0 - cloud_mask)).astype(np.float32)

        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_output_dtype_is_float32(self) -> None:
        obs_valid = np.ones((2, 2, 2), dtype=np.float64)
        cloud_mask = np.zeros((2, 2, 2), dtype=np.float64)

        result = apply_cloud_masking(obs_valid, cloud_mask)

        assert result.dtype == np.float32

    def test_output_shape_matches_input(self) -> None:
        obs_valid = np.ones((6, 4, 5), dtype=np.float32)
        cloud_mask = np.zeros((6, 4, 5), dtype=np.float32)

        result = apply_cloud_masking(obs_valid, cloud_mask)

        assert result.shape == (6, 4, 5)

    def test_partial_cloud_fraction(self) -> None:
        """Cloud mask with fractional values (e.g. 0.5) should scale validity."""
        obs_valid = np.ones((1, 1, 1), dtype=np.float32)
        cloud_mask = np.full((1, 1, 1), 0.5, dtype=np.float32)

        result = apply_cloud_masking(obs_valid, cloud_mask)

        np.testing.assert_allclose(result, 0.5, rtol=1e-6)
