"""Tests for terrain/fuel feature extraction logic."""

from __future__ import annotations


class TestTerrainBands:
    def test_terrain_band_names(self) -> None:
        from wildfire_pipeline.gee.terrain import TERRAIN_BANDS

        expected = {
            "slope_deg",
            "aspect_sin",
            "aspect_cos",
            "tpi",
            "terrain_ruggedness",
            "elevation",
        }
        assert set(TERRAIN_BANDS) == expected

    def test_terrain_band_count(self) -> None:
        from wildfire_pipeline.gee.terrain import TERRAIN_BANDS

        assert len(TERRAIN_BANDS) == 6

    def test_fuel_band_names(self) -> None:
        from wildfire_pipeline.gee.terrain import FUEL_BANDS

        assert "fuel_load" in FUEL_BANDS
        assert "is_firebreak" in FUEL_BANDS
        assert "canopy_cover_pct" in FUEL_BANDS

    def test_vegetation_band_names(self) -> None:
        from wildfire_pipeline.gee.terrain import VEGETATION_BANDS

        assert len(VEGETATION_BANDS) == 3

    def test_history_band_names(self) -> None:
        from wildfire_pipeline.gee.terrain import HISTORY_BANDS

        assert "years_since_burn" in HISTORY_BANDS
        assert "burn_count" in HISTORY_BANDS

    def test_wui_bands(self) -> None:
        from wildfire_pipeline.gee.terrain import WUI_BANDS

        assert "population" in WUI_BANDS
        assert "built_up" in WUI_BANDS

    def test_static_feature_count(self) -> None:
        from wildfire_pipeline.gee.terrain import ALL_STATIC_BAND_COUNT

        assert ALL_STATIC_BAND_COUNT == 17

    def test_band_count_matches_sum(self) -> None:
        from wildfire_pipeline.gee.terrain import (
            ALL_STATIC_BAND_COUNT,
            FUEL_BANDS,
            HISTORY_BANDS,
            ROAD_BANDS,
            TERRAIN_BANDS,
            VEGETATION_BANDS,
            WUI_BANDS,
        )

        total = (
            len(TERRAIN_BANDS)
            + len(FUEL_BANDS)
            + len(VEGETATION_BANDS)
            + len(HISTORY_BANDS)
            + len(ROAD_BANDS)
            + len(WUI_BANDS)
        )
        assert total == ALL_STATIC_BAND_COUNT
