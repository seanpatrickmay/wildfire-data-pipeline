"""CLI entry point for the wildfire data pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="wildfire",
    help="Wildfire data pipeline — download, process, and validate fire detection data.",
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_config(config: Path) -> Path:
    """Resolve config path relative to repo root or as absolute."""
    if config.is_absolute():
        return config
    return REPO_ROOT / config


@app.command()
def download(
    fire: Annotated[str, typer.Argument(help="Fire name from config (e.g. 'Kincade')")],
    config: Annotated[Path, typer.Option(help="Config file path")] = Path("config/fires.json"),
    output: Annotated[Path, typer.Option(help="Output directory")] = Path("data"),
    fmt: Annotated[str, typer.Option("--format", help="Output format")] = "npz",
    features: Annotated[
        bool, typer.Option("--features/--no-features", help="Download input features")
    ] = True,
) -> None:
    """Download fire detection data from Google Earth Engine."""
    from wildfire_pipeline.config import load_config
    from wildfire_pipeline.gee.download import TooManyFailuresError, download_fire_stack
    from wildfire_pipeline.gee.retry import init_ee
    from wildfire_pipeline.logging import get_logger, setup_logging

    setup_logging()
    logger = get_logger()

    cfg = load_config(_resolve_config(config))

    if fire not in cfg.fires:
        logger.error("unknown_fire", fire=fire, available=list(cfg.fires.keys()))
        raise typer.Exit(1)

    init_ee()
    output_dir = REPO_ROOT / output

    try:
        download_fire_stack(fire, cfg, output_dir, fmt=fmt)
    except TooManyFailuresError as e:
        logger.error("download_failed", error=str(e))
        raise typer.Exit(1) from None

    if features:
        from wildfire_pipeline.gee.download import download_features

        try:
            download_features(fire, cfg, output_dir, fmt=fmt)
        except (ConnectionError, TimeoutError, OSError, ValueError) as e:
            logger.error("feature_download_failed", error=str(e))
            logger.warning("continuing_without_features")


@app.command()
def process(
    input_path: Annotated[
        Path, typer.Argument(help="Input fire data file (.npz, .zarr, or .json)")
    ],
    config: Annotated[Path, typer.Option(help="Pipeline config file")] = Path("config/fires.json"),
    output: Annotated[
        Path | None, typer.Option(help="Output path (default: input_dir/processed/)")
    ] = None,
    fmt: Annotated[str, typer.Option("--format", help="Output format")] = "npz",
) -> None:
    """Process raw fire data into training labels."""
    from wildfire_pipeline.config import load_config
    from wildfire_pipeline.logging import get_logger, setup_logging
    from wildfire_pipeline.processing.io import save_fire_data
    from wildfire_pipeline.processing.labels import process_fire

    setup_logging()
    logger = get_logger()

    cfg = load_config(_resolve_config(config))

    try:
        out_arrays, out_metadata = process_fire(input_path, cfg.pipeline_config, fmt=fmt)
    except FileNotFoundError:
        logger.error("input_not_found", path=str(input_path))
        raise typer.Exit(1) from None
    except Exception as e:
        logger.error("processing_failed", error=str(e), input=str(input_path))
        raise typer.Exit(1) from None

    if output is not None:
        actual_path = save_fire_data(output, out_arrays, out_metadata, fmt=fmt)
        logger.info("saved_copy", path=str(actual_path))


@app.command()
def validate(
    input_path: Annotated[
        Path, typer.Argument(help="Data file to validate (.npz, .zarr, or .json)")
    ],
) -> None:
    """Run data quality checks on fire data."""
    import numpy as np

    from wildfire_pipeline.logging import get_logger, setup_logging
    from wildfire_pipeline.processing.io import load_fire_data
    from wildfire_pipeline.processing.validation import validate_download, validate_labels

    setup_logging()
    logger = get_logger()

    arrays, _metadata = load_fire_data(input_path)

    # Determine which validation to run based on content
    if "data" in arrays or "confidence" in arrays:
        conf_key = "data" if "data" in arrays else "confidence"
        # Guard against missing keys — provide zero arrays as fallback with warning
        conf = arrays[conf_key]
        obs_valid = arrays.get("observation_valid", arrays.get("obs_valid"))
        cloud_mask = arrays.get("cloud_mask")
        frp = arrays.get("frp")

        if obs_valid is None:
            logger.warning("missing_key", key="observation_valid", using="ones")
            obs_valid = np.ones_like(conf)
        if cloud_mask is None:
            logger.warning("missing_key", key="cloud_mask", using="zeros")
            cloud_mask = np.zeros_like(conf)
        if frp is None:
            logger.warning("missing_key", key="frp", using="zeros")
            frp = np.zeros_like(conf)

        result = validate_download(
            confidence=conf,
            obs_valid=obs_valid,
            cloud_mask=cloud_mask,
            frp=frp,
        )
    elif "labels" in arrays:
        result = validate_labels(
            labels=arrays["labels"],
            validity=arrays["validity"],
            raw_confidence=arrays["_diag_raw_confidence"],
        )
    else:
        logger.error("unknown_data_format", keys=list(arrays.keys()))
        raise typer.Exit(1)

    if result.errors:
        for err in result.errors:
            logger.error("validation_error", message=err)
    if result.warnings:
        for warn in result.warnings:
            logger.warning("validation_warning", message=warn)

    if result.passed:
        logger.info("validation_passed", warnings=len(result.warnings))
    else:
        logger.error("validation_failed", errors=len(result.errors), warnings=len(result.warnings))
        raise typer.Exit(1)


@app.command()
def list_fires(
    config: Annotated[Path, typer.Option(help="Config file path")] = Path("config/fires.json"),
) -> None:
    """List available fire events from config."""
    from wildfire_pipeline.config import load_config

    cfg = load_config(_resolve_config(config))

    for name, fire in cfg.fires.items():
        typer.echo(f"  {name}: {fire.year}, {fire.n_hours}h, {fire.official_acres or '?'} acres")


if __name__ == "__main__":
    app()
