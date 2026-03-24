"""Tests for the feature download orchestrator (pure logic, no GEE)."""

from __future__ import annotations


class TestFeatureChannelInventory:
    """Verify the expected channel counts for each feature category."""

    def test_hourly_weather_channels(self) -> None:
        from wildfire_pipeline.gee.weather import RTMA_BANDS

        assert len(RTMA_BANDS) + 1 == 6  # +1 for soil_moisture

    def test_daily_weather_channels(self) -> None:
        from wildfire_pipeline.gee.weather import GRIDMET_FIRE_BANDS

        assert len(GRIDMET_FIRE_BANDS) == 9

    def test_static_channels(self) -> None:
        from wildfire_pipeline.gee.terrain import ALL_STATIC_BAND_COUNT

        assert ALL_STATIC_BAND_COUNT == 17

    def test_slow_varying_channels(self) -> None:
        from wildfire_pipeline.gee.features import NDVI_BANDS

        assert len(NDVI_BANDS) == 2  # NDVI + EVI

    def test_download_features_importable(self) -> None:
        from wildfire_pipeline.gee.download import download_features

        assert callable(download_features)
