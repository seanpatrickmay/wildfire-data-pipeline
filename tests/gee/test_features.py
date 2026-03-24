"""Tests for slow-varying feature extraction logic."""

from __future__ import annotations


class TestNdviConfig:
    def test_ndvi_dataset_id(self) -> None:
        from wildfire_pipeline.gee.features import NDVI_DATASET

        assert NDVI_DATASET == "MODIS/061/MOD13Q1"

    def test_ndvi_bands(self) -> None:
        from wildfire_pipeline.gee.features import NDVI_BANDS

        assert "NDVI" in NDVI_BANDS
        assert "EVI" in NDVI_BANDS

    def test_ndvi_band_count(self) -> None:
        from wildfire_pipeline.gee.features import NDVI_BANDS

        assert len(NDVI_BANDS) == 2


class TestLstConfig:
    def test_lst_dataset_id(self) -> None:
        from wildfire_pipeline.gee.features import LST_DATASET

        assert LST_DATASET == "MODIS/061/MOD11A1"
