"""CLI entry point for fire label processing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from wildfire_pipeline.config import load_config  # noqa: E402
from wildfire_pipeline.logging import get_logger, setup_logging  # noqa: E402
from wildfire_pipeline.processing.labels import process_fire  # noqa: E402

logger = get_logger()


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Process fire data into training labels")
    parser.add_argument("--input", required=True, help="Input fire data path (npz, zarr, or json)")
    parser.add_argument("--output", help="Output path (default: input_dir/processed/)")
    parser.add_argument("--config", default="config/fires.json", help="Pipeline config")
    parser.add_argument(
        "--format",
        choices=["npz", "zarr", "json"],
        default="npz",
        help="Output format (default: npz)",
    )
    args = parser.parse_args()

    config = load_config(REPO_ROOT / args.config)

    input_path = Path(args.input)
    process_fire(input_path, config.pipeline_config, fmt=args.format)


if __name__ == "__main__":
    main()
