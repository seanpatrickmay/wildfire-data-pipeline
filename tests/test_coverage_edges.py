"""Tests targeting specific uncovered lines identified by coverage report.

These cover edge cases in logging, CLI error handling, and single-timestep
processing that are rarely hit but should be validated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class TestLoggingEdgeCases:
    """Cover the two uncovered branches in logging.py."""

    def test_json_output_mode(self) -> None:
        """setup_logging with json_output=True should use JSONRenderer."""
        from wildfire_pipeline.logging import setup_logging

        # Should not raise
        setup_logging(json_output=True)

    def test_get_logger_with_initial_context(self) -> None:
        """get_logger with kwargs should bind context."""
        from wildfire_pipeline.logging import get_logger, setup_logging

        setup_logging()
        logger = get_logger(fire_name="Test", step="download")
        # Should not raise; context is bound
        assert logger is not None


class TestSingleTimestepProcessFire:
    """Cover the T=1 branch in process_fire quality metrics."""

    def test_single_timestep_produces_valid_output(
        self, tmp_path: Path, pipeline_config: dict
    ) -> None:
        from wildfire_pipeline.processing.io import save_fire_data
        from wildfire_pipeline.processing.labels import process_fire

        T, H, W = 1, 4, 4
        arrays = {
            "data": np.full((T, H, W), 0.5, dtype=np.float32),
            "observation_valid": np.ones((T, H, W), dtype=np.float32),
            "cloud_mask": np.zeros((T, H, W), dtype=np.float32),
            "frp": np.full((T, H, W), 50.0, dtype=np.float32),
        }
        metadata = {"fire_name": "SingleStep", "n_hours": T}
        path = save_fire_data(tmp_path / "single", arrays, metadata, fmt="npz")

        out_arrays, out_meta = process_fire(path, pipeline_config, fmt="npz")

        assert out_arrays["labels"].shape == (T, H, W)
        # T=1 means no flicker possible and no oracle F1
        assert out_meta["quality"]["flicker_rate"] == 0.0
        assert out_meta["quality"]["oracle_f1_smoothed"] == 0.0


class TestCliProcessExceptionHandler:
    """Cover the generic Exception handler in the process CLI command."""

    def test_process_with_corrupt_npz_exits_error(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from wildfire_pipeline.cli import app

        runner = CliRunner()

        corrupt_path = tmp_path / "corrupt.npz"
        corrupt_path.write_bytes(b"not a real npz file")

        repo_config = Path(__file__).resolve().parent.parent / "config" / "fires.json"

        result = runner.invoke(
            app,
            ["process", str(corrupt_path), "--config", str(repo_config)],
        )
        assert result.exit_code != 0
