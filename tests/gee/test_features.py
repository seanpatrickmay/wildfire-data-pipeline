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


class TestSmokeAerosolConfig:
    def test_tropomi_dataset_id(self) -> None:
        from wildfire_pipeline.gee.features import TROPOMI_AAI_DATASET

        assert TROPOMI_AAI_DATASET == "COPERNICUS/S5P/OFFL/L3_AER_AI"


class TestNdwiConfig:
    """Verify NDWI/NDMI configuration for fuel moisture estimation."""

    def test_ndwi_dataset_id(self) -> None:
        from wildfire_pipeline.gee.features import NDWI_DATASET

        assert NDWI_DATASET == "MODIS/061/MOD09GA"

    def test_ndwi_bands(self) -> None:
        from wildfire_pipeline.gee.features import NDWI_BANDS

        assert "sur_refl_b02" in NDWI_BANDS  # NIR
        assert "sur_refl_b06" in NDWI_BANDS  # SWIR

    def test_ndwi_band_count(self) -> None:
        from wildfire_pipeline.gee.features import NDWI_BANDS

        assert len(NDWI_BANDS) == 2
