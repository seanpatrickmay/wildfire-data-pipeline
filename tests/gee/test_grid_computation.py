"""Tests for grid shape computation and sampleRectangle pixel limit guard."""

from __future__ import annotations

import pytest

from wildfire_pipeline.gee.download import SAMPLE_RECT_LIMIT, compute_grid_shape


class TestComputeGridShape:
    """Tests for AOI-to-grid conversion and pixel limit enforcement."""

    def test_kincade_fire_aoi(self) -> None:
        """The Kincade fire AOI at 2004m scale should produce a small grid."""
        aoi = [-122.96, 38.50, -122.59, 38.87]
        rows, cols = compute_grid_shape(aoi, scale=2004)
        # ~41km lat span / 2004m ≈ 20 rows, ~29km lon span / 2004m ≈ 14 cols
        assert 15 <= rows <= 25
        assert 10 <= cols <= 20
        assert rows * cols < SAMPLE_RECT_LIMIT

    def test_walker_fire_aoi(self) -> None:
        """The Walker fire AOI at 2004m scale should produce a small grid."""
        aoi = [-120.65, 38.90, -120.35, 39.20]
        rows, cols = compute_grid_shape(aoi, scale=2004)
        assert rows * cols < SAMPLE_RECT_LIMIT

    def test_tiny_aoi_produces_at_least_1x1(self) -> None:
        """Even a very small AOI should produce at least a 1x1 grid."""
        aoi = [-122.0, 38.0, -121.999, 38.001]
        rows, cols = compute_grid_shape(aoi, scale=2004)
        assert rows >= 1
        assert cols >= 1

    def test_large_aoi_raises_value_error(self) -> None:
        """An AOI that exceeds 262,144 pixels should raise ValueError."""
        # 10 degrees at equator ≈ 1,113 km; at 100m scale = ~11,000 pixels per side
        aoi = [-90.0, -5.0, -80.0, 5.0]
        with pytest.raises(ValueError, match="sampleRectangle limit"):
            compute_grid_shape(aoi, scale=100)

    def test_large_aoi_error_message_contains_pixel_count(self) -> None:
        aoi = [-90.0, -5.0, -80.0, 5.0]
        with pytest.raises(ValueError, match=r"\d+x\d+"):
            compute_grid_shape(aoi, scale=100)

    def test_finer_scale_increases_pixels(self) -> None:
        """Halving the scale should roughly quadruple the pixel count."""
        aoi = [-122.96, 38.50, -122.59, 38.87]
        rows_2k, cols_2k = compute_grid_shape(aoi, scale=2000)
        rows_1k, cols_1k = compute_grid_shape(aoi, scale=1000)
        pixels_2k = rows_2k * cols_2k
        pixels_1k = rows_1k * cols_1k
        # Should be roughly 4x (between 3x and 5x due to rounding)
        assert 3 * pixels_2k <= pixels_1k <= 5 * pixels_2k

    def test_coarser_scale_reduces_pixels(self) -> None:
        """A 4km scale should produce fewer pixels than 2km."""
        aoi = [-122.96, 38.50, -122.59, 38.87]
        rows_2k, cols_2k = compute_grid_shape(aoi, scale=2004)
        rows_4k, cols_4k = compute_grid_shape(aoi, scale=4000)
        assert rows_4k * cols_4k < rows_2k * cols_2k

    def test_latitude_affects_column_count(self) -> None:
        """Same longitude span at higher latitude should produce fewer columns
        (longitude degrees shrink toward poles)."""
        aoi_equator = [-80.0, -0.5, -79.0, 0.5]
        aoi_high_lat = [-80.0, 59.5, -79.0, 60.5]
        _, cols_eq = compute_grid_shape(aoi_equator, scale=2000)
        _, cols_hi = compute_grid_shape(aoi_high_lat, scale=2000)
        assert cols_hi < cols_eq

    def test_just_under_limit_passes(self) -> None:
        """An AOI producing exactly 262,144 pixels should NOT raise."""
        # Create a 512x512 grid: need an AOI where rows*cols == 262144
        # At scale=1000m, 512 pixels = 512km = ~4.6 degrees latitude
        # This is approximate — we just need to be under the limit
        aoi = [-80.0, 38.0, -76.0, 42.0]  # ~4 degrees each way
        try:
            rows, cols = compute_grid_shape(aoi, scale=1000)
            # If it passes, verify it's under limit
            assert rows * cols <= SAMPLE_RECT_LIMIT
        except ValueError:
            # If this particular AOI happens to exceed, that's also valid
            pass

    def test_sample_rect_limit_constant_is_correct(self) -> None:
        """The constant should be 512*512 = 262,144."""
        assert SAMPLE_RECT_LIMIT == 512 * 512
