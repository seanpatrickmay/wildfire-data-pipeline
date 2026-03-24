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

    def test_lagged_features_identified(self) -> None:
        from wildfire_pipeline.feature_registry import get_lagged_features

        lagged = get_lagged_features()
        assert "_lagged_distance_to_fire" in lagged
        assert "_lagged_fire_neighborhood" in lagged
        assert "erc" not in lagged
