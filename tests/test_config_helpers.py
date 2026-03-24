"""Tests for config validation helper functions.

These guard the typed-config boundary: converting raw dicts to validated
Pydantic models at the entry points of both download and processing pipelines.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wildfire_pipeline.config import FiresConfig, PipelineConfig
from wildfire_pipeline.gee.download import _ensure_fires_config
from wildfire_pipeline.processing.labels import _ensure_pipeline_config

# ---------------------------------------------------------------------------
# _ensure_pipeline_config
# ---------------------------------------------------------------------------


class TestEnsurePipelineConfig:
    def test_passthrough_when_already_model(self) -> None:
        cfg = PipelineConfig()
        result = _ensure_pipeline_config(cfg)
        assert result is cfg  # same object, no copy

    def test_converts_valid_dict(self) -> None:
        d = {"goes_confidence_threshold": 0.5, "cloud_masking": False}
        result = _ensure_pipeline_config(d)
        assert isinstance(result, PipelineConfig)
        assert result.goes_confidence_threshold == pytest.approx(0.5)
        assert result.cloud_masking is False

    def test_empty_dict_uses_defaults(self) -> None:
        result = _ensure_pipeline_config({})
        assert isinstance(result, PipelineConfig)
        assert result.goes_confidence_threshold == pytest.approx(0.30)
        assert result.max_persistence_gap_hours == 3
        assert result.imputation_weight == pytest.approx(0.3)

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValidationError, match="goes_confidence_threshold"):
            _ensure_pipeline_config({"goes_confidence_threshold": 5.0})

    def test_invalid_imputation_weight_raises(self) -> None:
        with pytest.raises(ValidationError, match="imputation_weight"):
            _ensure_pipeline_config({"imputation_weight": -1.0})

    def test_invalid_max_gap_raises(self) -> None:
        with pytest.raises(ValidationError, match="max_persistence_gap_hours"):
            _ensure_pipeline_config({"max_persistence_gap_hours": 0})

    def test_wrong_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            _ensure_pipeline_config({"goes_confidence_threshold": "not a number"})

    def test_nested_smoothing_config(self) -> None:
        d = {"label_smoothing": {"method": "rolling_max", "window_hours": 3, "min_votes": 1}}
        result = _ensure_pipeline_config(d)
        assert result.label_smoothing.method == "rolling_max"
        assert result.label_smoothing.window_hours == 3


# ---------------------------------------------------------------------------
# _ensure_fires_config
# ---------------------------------------------------------------------------


class TestEnsureFiresConfig:
    @pytest.fixture()
    def valid_fires_dict(self) -> dict:
        return {
            "fires": {
                "TestFire": {
                    "year": 2023,
                    "aoi": [-122.0, 38.0, -121.0, 39.0],
                    "start_utc": "2023-08-01T00:00:00Z",
                    "n_hours": 24,
                }
            },
            "pipeline_config": {},
        }

    def test_passthrough_when_already_model(self, valid_fires_dict: dict) -> None:
        cfg = FiresConfig.model_validate(valid_fires_dict)
        result = _ensure_fires_config(cfg)
        assert result is cfg

    def test_converts_valid_dict(self, valid_fires_dict: dict) -> None:
        result = _ensure_fires_config(valid_fires_dict)
        assert isinstance(result, FiresConfig)
        assert "TestFire" in result.fires
        assert result.fires["TestFire"].year == 2023
        assert result.fires["TestFire"].n_hours == 24

    def test_fire_aoi_validated(self, valid_fires_dict: dict) -> None:
        valid_fires_dict["fires"]["TestFire"]["aoi"] = [-122.0, 38.0]  # too short
        with pytest.raises(ValidationError):
            _ensure_fires_config(valid_fires_dict)

    def test_missing_fires_key_raises(self) -> None:
        with pytest.raises(ValidationError, match="fires"):
            _ensure_fires_config({"pipeline_config": {}})

    def test_missing_pipeline_config_raises(self) -> None:
        with pytest.raises(ValidationError, match="pipeline_config"):
            _ensure_fires_config({"fires": {}})

    def test_n_hours_zero_raises(self, valid_fires_dict: dict) -> None:
        valid_fires_dict["fires"]["TestFire"]["n_hours"] = 0
        with pytest.raises(ValidationError, match="n_hours"):
            _ensure_fires_config(valid_fires_dict)

    def test_invalid_datetime_raises(self, valid_fires_dict: dict) -> None:
        valid_fires_dict["fires"]["TestFire"]["start_utc"] = "not-a-date"
        with pytest.raises(ValidationError):
            _ensure_fires_config(valid_fires_dict)

    def test_pipeline_config_defaults_applied(self, valid_fires_dict: dict) -> None:
        result = _ensure_fires_config(valid_fires_dict)
        assert result.pipeline_config.export_scale_m == 2004
        assert result.pipeline_config.goes_confidence_threshold == pytest.approx(0.30)

    def test_loads_real_config_file(self) -> None:
        from pathlib import Path

        from wildfire_pipeline.config import load_config

        config_path = Path(__file__).resolve().parent.parent / "config" / "fires.json"
        cfg = load_config(config_path)
        result = _ensure_fires_config(cfg)
        assert result is cfg
        assert "Kincade" in result.fires

    def test_datetime_fields_accessible(self, valid_fires_dict: dict) -> None:
        """Verify datetime parsing works through the ensure helper."""
        from datetime import datetime

        result = _ensure_fires_config(valid_fires_dict)
        fire = result.fires["TestFire"]
        assert isinstance(fire.start_utc, datetime)
        assert fire.start_utc.year == 2023
        assert fire.start_utc.month == 8

    def test_aoi_is_tuple(self, valid_fires_dict: dict) -> None:
        """Verify AOI list is coerced to tuple."""
        result = _ensure_fires_config(valid_fires_dict)
        fire = result.fires["TestFire"]
        assert isinstance(fire.aoi, tuple)
        assert len(fire.aoi) == 4


# ---------------------------------------------------------------------------
# Public API exports from wildfire_pipeline.__init__
# ---------------------------------------------------------------------------


class TestPublicApiExports:
    """Verify the package __init__.py exports work correctly."""

    def test_import_load_config(self) -> None:
        from wildfire_pipeline import load_config

        assert callable(load_config)

    def test_import_fires_config(self) -> None:
        from wildfire_pipeline import FiresConfig

        assert FiresConfig is not None
        cfg = FiresConfig(fires={}, pipeline_config={})
        assert isinstance(cfg, FiresConfig)

    def test_import_pipeline_config(self) -> None:
        from wildfire_pipeline import PipelineConfig

        assert PipelineConfig is not None
        cfg = PipelineConfig()
        assert cfg.goes_confidence_threshold == pytest.approx(0.30)

    def test_all_exports_listed(self) -> None:
        import wildfire_pipeline

        assert hasattr(wildfire_pipeline, "__all__")
        assert "load_config" in wildfire_pipeline.__all__
        assert "FiresConfig" in wildfire_pipeline.__all__
        assert "PipelineConfig" in wildfire_pipeline.__all__
