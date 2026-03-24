"""CLI entry point for fire data download."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from wildfire_pipeline.config import load_config  # noqa: E402
from wildfire_pipeline.gee.download import TooManyFailuresError, download_fire_stack  # noqa: E402
from wildfire_pipeline.gee.retry import init_ee  # noqa: E402
from wildfire_pipeline.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger()


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Download fire detection data from GEE")
    parser.add_argument("--fire", required=True, help="Fire name from config")
    parser.add_argument("--config", default="config/fires.json", help="Config file path")
    parser.add_argument("--output", default="data", help="Output directory")
    parser.add_argument(
        "--format",
        choices=["npz", "zarr", "json"],
        default="npz",
        help="Output format (default: npz)",
    )
    args = parser.parse_args()

    config = load_config(REPO_ROOT / args.config)

    if args.fire not in config.fires:
        logger.error("unknown_fire", fire=args.fire, available=list(config.fires.keys()))
        sys.exit(1)

    init_ee()
    output_dir = REPO_ROOT / args.output

    try:
        download_fire_stack(args.fire, config, output_dir, fmt=args.format)
    except TooManyFailuresError as e:
        logger.error(
            "too_many_failures",
            failed=e.failed,
            total=e.total,
            rate=e.rate,
            hours=e.hours,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
