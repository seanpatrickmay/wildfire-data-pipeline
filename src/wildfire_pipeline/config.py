from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pathlib import Path


class FireEvent(BaseModel):
    """A single fire event configuration."""

    year: int
    aoi: tuple[float, float, float, float]  # west, south, east, north
    start_utc: datetime
    n_hours: int = Field(gt=0)
    official_acres: int | None = None


class LabelSmoothing(BaseModel):
    """Label temporal smoothing configuration."""

    method: str = "majority_vote"
    window_hours: int = Field(default=5, gt=0)
    min_votes: int = Field(default=2, gt=0)


class PipelineConfig(BaseModel):
    """Pipeline processing parameters."""

    export_scale_m: int = 2004
    export_crs: str = "EPSG:3857"
    goes_confidence_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    label_smoothing: LabelSmoothing = Field(default_factory=LabelSmoothing)
    cloud_masking: bool = True
    max_persistence_gap_hours: int = Field(
        default=3,
        gt=0,
        description="Max hours to forward-fill fire through cloud gaps",
    )
    imputation_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Training loss weight for imputed (forward-filled) pixels",
    )
    download_features: bool = Field(
        default=True,
        description="Download weather/terrain/vegetation features alongside fire labels",
    )
    rtma_wind: bool = Field(
        default=True,
        description="Use RTMA for hourly wind (2.5km). Falls back to GRIDMET daily if False.",
    )


class FiresConfig(BaseModel):
    """Root config matching fires.json structure."""

    description: str = ""
    fires: dict[str, FireEvent]
    pipeline_config: PipelineConfig


def load_config(config_path: Path) -> FiresConfig:
    """Load and validate pipeline config from JSON file."""
    import json

    with open(config_path) as f:
        raw = json.load(f)
    return FiresConfig.model_validate(raw)
