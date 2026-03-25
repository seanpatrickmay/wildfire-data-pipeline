from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from wildfire_pipeline.config import (
    FireEvent,
    FiresConfig,
    LabelSmoothing,
    PipelineConfig,
    load_config,
)

FIRES_JSON = Path(__file__).resolve().parent.parent / "config" / "fires.json"


class TestLoadRealConfig:
    def test_loads_fires_json(self):
        config = load_config(FIRES_JSON)
        assert isinstance(config, FiresConfig)
        assert "Kincade" in config.fires
        assert "Walker" in config.fires

    def test_fire_fields_parsed(self):
        config = load_config(FIRES_JSON)
        kincade = config.fires["Kincade"]
        assert kincade.year == 2019
        assert kincade.aoi == (-122.96, 38.50, -122.59, 38.87)
        assert kincade.start_utc == datetime(2019, 10, 24, 4, 27, tzinfo=UTC)
        assert kincade.n_hours == 160
        assert kincade.official_acres == 77758

    def test_pipeline_config_parsed(self):
        config = load_config(FIRES_JSON)
        pc = config.pipeline_config
        assert pc.export_scale_m == 2004
        assert pc.export_crs == "EPSG:3857"
        assert pc.goes_confidence_threshold == pytest.approx(0.30)
        assert pc.cloud_masking is True

    def test_label_smoothing_parsed(self):
        config = load_config(FIRES_JSON)
        ls = config.pipeline_config.label_smoothing
        assert ls.method == "majority_vote"
        assert ls.window_hours == 5
        assert ls.min_votes == 2


class TestFireEvent:
    def test_minimal_valid(self):
        event = FireEvent(
            year=2020,
            aoi=(-120.0, 38.0, -119.0, 39.0),
            start_utc="2020-01-01T00:00:00Z",
            n_hours=24,
        )
        assert event.official_acres is None

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            FireEvent(year=2020, aoi=(-120.0, 38.0, -119.0, 39.0), start_utc="2020-01-01T00:00:00Z")

    def test_n_hours_must_be_positive(self):
        with pytest.raises(ValidationError, match="n_hours"):
            FireEvent(
                year=2020,
                aoi=(-120.0, 38.0, -119.0, 39.0),
                start_utc="2020-01-01T00:00:00Z",
                n_hours=0,
            )

    def test_n_hours_negative(self):
        with pytest.raises(ValidationError, match="n_hours"):
            FireEvent(
                year=2020,
                aoi=(-120.0, 38.0, -119.0, 39.0),
                start_utc="2020-01-01T00:00:00Z",
                n_hours=-5,
            )

    def test_aoi_wrong_length(self):
        with pytest.raises(ValidationError):
            FireEvent(
                year=2020,
                aoi=(-120.0, 38.0, -119.0),
                start_utc="2020-01-01T00:00:00Z",
                n_hours=24,
            )

    def test_aoi_non_numeric(self):
        with pytest.raises(ValidationError):
            FireEvent(
                year=2020,
                aoi=("west", "south", "east", "north"),
                start_utc="2020-01-01T00:00:00Z",
                n_hours=24,
            )

    def test_invalid_datetime(self):
        with pytest.raises(ValidationError):
            FireEvent(
                year=2020,
                aoi=(-120.0, 38.0, -119.0, 39.0),
                start_utc="not-a-date",
                n_hours=24,
            )


class TestPipelineConfig:
    def test_defaults(self):
        pc = PipelineConfig()
        assert pc.export_scale_m == 2004
        assert pc.export_crs == "EPSG:3857"
        assert pc.goes_confidence_threshold == pytest.approx(0.30)
        assert pc.cloud_masking is True
        assert pc.label_smoothing.method == "majority_vote"
        assert pc.max_persistence_gap_hours == 3
        assert pc.imputation_weight == pytest.approx(0.3)
        assert pc.download_features is True
        assert pc.rtma_wind is True

    def test_confidence_threshold_too_high(self):
        with pytest.raises(ValidationError, match="goes_confidence_threshold"):
            PipelineConfig(goes_confidence_threshold=1.5)

    def test_confidence_threshold_negative(self):
        with pytest.raises(ValidationError, match="goes_confidence_threshold"):
            PipelineConfig(goes_confidence_threshold=-0.1)

    def test_confidence_threshold_boundaries(self):
        PipelineConfig(goes_confidence_threshold=0.0)
        PipelineConfig(goes_confidence_threshold=1.0)

    def test_max_persistence_gap_must_be_positive(self):
        with pytest.raises(ValidationError, match="max_persistence_gap_hours"):
            PipelineConfig(max_persistence_gap_hours=0)

    def test_max_persistence_gap_custom_value(self):
        pc = PipelineConfig(max_persistence_gap_hours=6)
        assert pc.max_persistence_gap_hours == 6

    def test_imputation_weight_must_be_in_range(self):
        with pytest.raises(ValidationError, match="imputation_weight"):
            PipelineConfig(imputation_weight=1.5)
        with pytest.raises(ValidationError, match="imputation_weight"):
            PipelineConfig(imputation_weight=-0.1)

    def test_imputation_weight_boundaries(self):
        PipelineConfig(imputation_weight=0.0)
        PipelineConfig(imputation_weight=1.0)

    def test_imputation_weight_custom_value(self):
        pc = PipelineConfig(imputation_weight=0.5)
        assert pc.imputation_weight == pytest.approx(0.5)


class TestLabelSmoothing:
    def test_defaults(self):
        ls = LabelSmoothing()
        assert ls.method == "majority_vote"
        assert ls.window_hours == 5
        assert ls.min_votes == 2

    def test_window_hours_must_be_positive(self):
        with pytest.raises(ValidationError, match="window_hours"):
            LabelSmoothing(window_hours=0)

    def test_min_votes_must_be_positive(self):
        with pytest.raises(ValidationError, match="min_votes"):
            LabelSmoothing(min_votes=0)


class TestFiresConfig:
    def test_missing_fires(self):
        with pytest.raises(ValidationError, match="fires"):
            FiresConfig(pipeline_config={})

    def test_missing_pipeline_config(self):
        with pytest.raises(ValidationError, match="pipeline_config"):
            FiresConfig(fires={})

    def test_empty_fires_dict_is_valid(self):
        config = FiresConfig(fires={}, pipeline_config={})
        assert config.fires == {}

    def test_description_defaults_to_empty(self):
        config = FiresConfig(fires={}, pipeline_config={})
        assert config.description == ""
