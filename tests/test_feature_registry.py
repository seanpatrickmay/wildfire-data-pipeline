"""Tests for the feature registry."""

from __future__ import annotations

from wildfire_pipeline.feature_registry import (
    FEATURE_REGISTRY,
    FeatureSpec,
    get_feature_spec,
    get_safe_input_features,
)


class TestFeatureRegistry:
    def test_registry_not_empty(self) -> None:
        assert len(FEATURE_REGISTRY) > 30

    def test_all_entries_are_feature_specs(self) -> None:
        for name, spec in FEATURE_REGISTRY.items():
            assert isinstance(spec, FeatureSpec), f"{name} is not a FeatureSpec"

    def test_get_feature_spec_direct(self) -> None:
        spec = get_feature_spec("erc")
        assert spec is not None
        assert spec.unit == "BTU/ft2"

    def test_get_feature_spec_with_prefix(self) -> None:
        spec = get_feature_spec("daily_erc")
        assert spec is not None
        assert spec.name == "erc"

    def test_unknown_feature_returns_none(self) -> None:
        assert get_feature_spec("nonexistent_feature") is None

    def test_all_have_valid_temporal(self) -> None:
        valid = {"hourly", "daily", "static", "slow"}
        for name, spec in FEATURE_REGISTRY.items():
            assert spec.temporal in valid, f"{name} has invalid temporal: {spec.temporal}"

    def test_all_have_valid_normalization(self) -> None:
        valid = {"zscore", "minmax", "none", "embedding"}
        for name, spec in FEATURE_REGISTRY.items():
            assert spec.normalization in valid, f"{name} has invalid normalization"

    def test_safe_input_features_excludes_targets(self) -> None:
        safe = get_safe_input_features()
        assert "labels" not in safe
        assert "soft_labels" not in safe
        assert "_diag_raw_confidence" not in safe
        assert "loss_weights" not in safe

    def test_safe_input_features_includes_weather(self) -> None:
        safe = get_safe_input_features()
        assert "erc" in safe
        assert "ugrd" in safe
        assert "slope_deg" in safe

    def test_prev_fire_features_safe_as_input(self) -> None:
        safe = get_safe_input_features()
        assert "prev_fire_state" in safe
        assert "prev_distance_to_fire" in safe
        assert "prev_fire_neighborhood" in safe

    def test_prev_fire_features_no_temporal_lag_required(self) -> None:
        from wildfire_pipeline.feature_registry import get_lagged_features

        lagged = get_lagged_features()
        # prev_* features are pre-shifted, so they do NOT require temporal lag
        assert "prev_fire_state" not in lagged
        assert "prev_distance_to_fire" not in lagged
        assert "prev_fire_neighborhood" not in lagged
        assert "erc" not in lagged

    def test_fire_change_not_safe_as_input(self) -> None:
        safe = get_safe_input_features()
        assert "fire_change" not in safe

    def test_smoke_features_registered(self) -> None:
        assert "is_smoke" in FEATURE_REGISTRY
        assert "btd_fire_smoke" in FEATURE_REGISTRY

    def test_smoke_features_are_hourly(self) -> None:
        assert FEATURE_REGISTRY["is_smoke"].temporal == "hourly"
        assert FEATURE_REGISTRY["btd_fire_smoke"].temporal == "hourly"

    def test_smoke_features_safe_as_input(self) -> None:
        safe = get_safe_input_features()
        assert "is_smoke" in safe
        assert "btd_fire_smoke" in safe

    def test_btd_fire_smoke_is_continuous(self) -> None:
        spec = FEATURE_REGISTRY["btd_fire_smoke"]
        assert spec.dtype_hint == "continuous"
        assert spec.unit == "K"
        assert spec.normalization == "zscore"

    def test_is_smoke_is_binary(self) -> None:
        spec = FEATURE_REGISTRY["is_smoke"]
        assert spec.dtype_hint == "binary"
        assert spec.normalization == "none"

    def test_fire_area_temp_registered(self) -> None:
        assert "fire_area_km2" in FEATURE_REGISTRY
        assert "fire_temp_k" in FEATURE_REGISTRY
        assert FEATURE_REGISTRY["fire_area_km2"].unit == "km2"
        assert FEATURE_REGISTRY["fire_temp_k"].unit == "K"

    def test_ndwi_registered(self) -> None:
        spec = FEATURE_REGISTRY["ndwi"]
        assert spec.source == "MODIS/061/MOD09GA"
        assert spec.temporal == "slow"
        assert spec.range_min == -1.0
        assert spec.range_max == 1.0

    def test_blue_swir_smoke_ratio_registered(self) -> None:
        spec = FEATURE_REGISTRY["blue_swir_smoke_ratio"]
        assert spec.unit == "ratio"
        assert spec.temporal == "hourly"

    def test_new_features_safe_as_input(self) -> None:
        safe = get_safe_input_features()
        for name in ["fire_area_km2", "fire_temp_k", "ndwi", "blue_swir_smoke_ratio"]:
            assert name in safe, f"{name} should be safe as model input"
