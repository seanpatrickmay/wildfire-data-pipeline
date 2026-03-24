"""Tests for ML readiness features: temporal encoding, class balance, normalization."""

from __future__ import annotations

import numpy as np
import pytest

from wildfire_pipeline.processing.quality import compute_normalization_stats


class TestNormalizationStats:
    def test_computes_expected_keys(self) -> None:
        arrays = {"a": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)}
        stats = compute_normalization_stats(arrays)
        assert "a" in stats
        assert set(stats["a"].keys()) == {"mean", "std", "min", "max", "p1", "p99"}

    def test_values_correct(self) -> None:
        arrays = {"x": np.arange(100, dtype=np.float32)}
        stats = compute_normalization_stats(arrays)
        assert stats["x"]["mean"] == pytest.approx(49.5, abs=0.1)
        assert stats["x"]["min"] == pytest.approx(0.0)
        assert stats["x"]["max"] == pytest.approx(99.0)

    def test_empty_arrays_skipped(self) -> None:
        arrays = {"empty": np.array([], dtype=np.float32)}
        stats = compute_normalization_stats(arrays)
        assert "empty" not in stats

    def test_multiple_channels(self) -> None:
        arrays = {
            "a": np.ones((10,), dtype=np.float32),
            "b": np.zeros((10,), dtype=np.float32) + 5.0,
        }
        stats = compute_normalization_stats(arrays)
        assert stats["a"]["mean"] == pytest.approx(1.0)
        assert stats["b"]["mean"] == pytest.approx(5.0)


class TestDistanceToFire:
    def test_fire_pixels_have_zero_distance(self) -> None:
        from wildfire_pipeline.processing.quality import compute_distance_to_fire

        labels = np.zeros((3, 5, 5), dtype=np.float32)
        labels[1, 2, 2] = 1.0
        dist = compute_distance_to_fire(labels)
        assert dist[1, 2, 2] == 0.0

    def test_adjacent_pixels_have_distance_one(self) -> None:
        from wildfire_pipeline.processing.quality import compute_distance_to_fire

        labels = np.zeros((1, 5, 5), dtype=np.float32)
        labels[0, 2, 2] = 1.0
        dist = compute_distance_to_fire(labels)
        assert dist[0, 2, 1] == pytest.approx(1.0)
        assert dist[0, 1, 2] == pytest.approx(1.0)

    def test_no_fire_returns_sentinel(self) -> None:
        from wildfire_pipeline.processing.quality import compute_distance_to_fire

        labels = np.zeros((2, 3, 3), dtype=np.float32)
        dist = compute_distance_to_fire(labels)
        assert (dist == -1.0).all()


class TestFireNeighborhood:
    def test_isolated_fire_has_low_fraction(self) -> None:
        from wildfire_pipeline.processing.quality import compute_fire_neighborhood

        labels = np.zeros((1, 9, 9), dtype=np.float32)
        labels[0, 4, 4] = 1.0
        frac = compute_fire_neighborhood(labels, kernel_size=5)
        assert frac[0, 4, 4] < 0.1  # 1/25 = 0.04

    def test_full_fire_has_high_fraction(self) -> None:
        from wildfire_pipeline.processing.quality import compute_fire_neighborhood

        labels = np.ones((1, 9, 9), dtype=np.float32)
        frac = compute_fire_neighborhood(labels, kernel_size=5)
        assert frac[0, 4, 4] == pytest.approx(1.0)
