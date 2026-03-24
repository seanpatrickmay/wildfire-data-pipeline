"""Tests for weather feature extraction logic (no GEE auth needed)."""

from __future__ import annotations


class TestRtmaConfig:
    def test_rtma_dataset_id(self) -> None:
        from wildfire_pipeline.gee.weather import RTMA_DATASET

        assert RTMA_DATASET == "NOAA/NWS/RTMA"

    def test_rtma_bands_defined(self) -> None:
        from wildfire_pipeline.gee.weather import RTMA_BANDS

        assert "UGRD" in RTMA_BANDS
        assert "VGRD" in RTMA_BANDS
        assert "TMP" in RTMA_BANDS
        assert "GUST" in RTMA_BANDS

    def test_rtma_band_count(self) -> None:
        from wildfire_pipeline.gee.weather import RTMA_BANDS

        assert len(RTMA_BANDS) == 5


class TestGridmetConfig:
    def test_gridmet_fire_weather_bands(self) -> None:
        from wildfire_pipeline.gee.weather import GRIDMET_FIRE_BANDS

        expected = {"erc", "bi", "fm100", "fm1000", "vpd", "rmin", "rmax", "vs", "th"}
        assert set(GRIDMET_FIRE_BANDS) == expected

    def test_gridmet_dataset_id(self) -> None:
        from wildfire_pipeline.gee.weather import GRIDMET_DATASET

        assert GRIDMET_DATASET == "IDAHO_EPSCOR/GRIDMET"

    def test_gridmet_band_count(self) -> None:
        from wildfire_pipeline.gee.weather import GRIDMET_FIRE_BANDS

        assert len(GRIDMET_FIRE_BANDS) == 9


class TestEra5Config:
    def test_era5_dataset_id(self) -> None:
        from wildfire_pipeline.gee.weather import ERA5_LAND_DATASET

        assert ERA5_LAND_DATASET == "ECMWF/ERA5_LAND/HOURLY"

    def test_drought_dataset_id(self) -> None:
        from wildfire_pipeline.gee.weather import GRIDMET_DROUGHT_DATASET

        assert GRIDMET_DROUGHT_DATASET == "GRIDMET/DROUGHT"


class TestPrecipitationConfig:
    def test_gpm_dataset_id(self) -> None:
        from wildfire_pipeline.gee.weather import GPM_IMERG_DATASET

        assert GPM_IMERG_DATASET == "NASA/GPM_L3/IMERG_V07"
