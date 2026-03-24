"""Tests for the CLI commands (list-fires, validate, process).

Uses typer.testing.CliRunner to invoke commands without spawning subprocesses.
The download command requires GEE credentials and is only tested for error paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from wildfire_pipeline.cli import app
from wildfire_pipeline.processing.io import save_fire_data

runner = CliRunner()

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# list-fires command
# ---------------------------------------------------------------------------


class TestListFires:
    def test_lists_fires_from_real_config(self) -> None:
        result = runner.invoke(
            app, ["list-fires", "--config", str(REPO_ROOT / "config/fires.json")]
        )
        assert result.exit_code == 0
        assert "Kincade" in result.stdout
        assert "Walker" in result.stdout

    def test_shows_year_and_hours(self) -> None:
        result = runner.invoke(
            app, ["list-fires", "--config", str(REPO_ROOT / "config/fires.json")]
        )
        assert "2019" in result.stdout
        assert "160h" in result.stdout

    def test_invalid_config_path_exits_with_error(self) -> None:
        result = runner.invoke(app, ["list-fires", "--config", "/nonexistent/config.json"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# validate command
# ---------------------------------------------------------------------------


class TestValidateCommand:
    def test_validate_valid_download_data(self, tmp_path: Path) -> None:
        """Valid download data should pass validation."""
        T, H, W = 10, 5, 5
        arrays = {
            "data": np.full((T, H, W), 0.5, dtype=np.float32),
            "observation_valid": np.ones((T, H, W), dtype=np.float32),
            "cloud_mask": np.zeros((T, H, W), dtype=np.float32),
            "frp": np.full((T, H, W), 100.0, dtype=np.float32),
        }
        metadata = {"fire_name": "Test", "n_hours": T}
        path = save_fire_data(tmp_path / "test_fire", arrays, metadata, fmt="npz")

        result = runner.invoke(app, ["validate", str(path)])
        assert result.exit_code == 0

    def test_validate_valid_label_data(self, tmp_path: Path) -> None:
        """Valid processed label data should pass validation."""
        T, H, W = 10, 5, 5
        arrays = {
            "labels": np.zeros((T, H, W), dtype=np.float32),
            "validity": np.ones((T, H, W), dtype=np.float32),
            "raw_confidence": np.full((T, H, W), 0.2, dtype=np.float32),
        }
        # Add some fire pixels so we don't get "no fire" warning (which still passes)
        arrays["labels"][3:6, 1:3, 1:3] = 1.0
        metadata = {"fire_name": "Test"}
        path = save_fire_data(tmp_path / "test_labels", arrays, metadata, fmt="npz")

        result = runner.invoke(app, ["validate", str(path)])
        assert result.exit_code == 0

    def test_validate_corrupted_data_fails(self, tmp_path: Path) -> None:
        """Data with NaN values should fail validation."""
        T, H, W = 10, 5, 5
        arrays = {
            "data": np.full((T, H, W), 0.5, dtype=np.float32),
            "observation_valid": np.ones((T, H, W), dtype=np.float32),
            "cloud_mask": np.zeros((T, H, W), dtype=np.float32),
            "frp": np.full((T, H, W), 100.0, dtype=np.float32),
        }
        arrays["data"][0, 0, 0] = np.nan  # Inject corruption
        metadata = {"fire_name": "Corrupted"}
        path = save_fire_data(tmp_path / "bad_fire", arrays, metadata, fmt="npz")

        result = runner.invoke(app, ["validate", str(path)])
        assert result.exit_code != 0

    def test_validate_nonexistent_file(self) -> None:
        result = runner.invoke(app, ["validate", "/nonexistent/file.npz"])
        assert result.exit_code != 0

    def test_validate_unknown_format(self, tmp_path: Path) -> None:
        """A file with unrecognized keys should exit with error."""
        arrays = {"unknown_key": np.zeros((5, 3, 3), dtype=np.float32)}
        metadata = {}
        path = save_fire_data(tmp_path / "weird", arrays, metadata, fmt="npz")

        result = runner.invoke(app, ["validate", str(path)])
        assert result.exit_code != 0

    def test_validate_missing_obs_valid_uses_fallback(self, tmp_path: Path) -> None:
        """If observation_valid is missing, should use ones as fallback and still pass."""
        T, H, W = 5, 3, 3
        arrays = {
            "data": np.full((T, H, W), 0.5, dtype=np.float32),
            # Deliberately omit observation_valid, cloud_mask, frp
        }
        metadata = {"fire_name": "Minimal"}
        path = save_fire_data(tmp_path / "minimal", arrays, metadata, fmt="npz")

        result = runner.invoke(app, ["validate", str(path)])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# process command
# ---------------------------------------------------------------------------


class TestProcessCommand:
    @pytest.fixture()
    def fire_data_path(self, tmp_path: Path) -> Path:
        """Create a valid fire data npz file for processing."""
        T, H, W = 10, 5, 5
        rng = np.random.default_rng(42)
        confidence = np.zeros((T, H, W), dtype=np.float32)
        # Create a realistic fire cluster that survives isolated pixel filtering
        for t in range(T):
            confidence[t, 1:4, 1:4] = rng.uniform(0.3, 0.9, size=(3, 3)).astype(np.float32)

        arrays = {
            "data": confidence,
            "observation_valid": np.ones((T, H, W), dtype=np.float32),
            "cloud_mask": np.zeros((T, H, W), dtype=np.float32),
            "frp": (confidence * 50).astype(np.float32),
        }
        metadata = {"fire_name": "CLITest", "n_hours": T}
        return save_fire_data(tmp_path / "cli_test_fire", arrays, metadata, fmt="npz")

    def test_process_creates_output(self, fire_data_path: Path) -> None:
        config_path = REPO_ROOT / "config/fires.json"
        result = runner.invoke(
            app,
            ["process", str(fire_data_path), "--config", str(config_path)],
        )
        assert result.exit_code == 0
        # Should create processed/ subdirectory
        processed_dir = fire_data_path.parent / "processed"
        assert processed_dir.exists()
        processed_files = list(processed_dir.glob("*_processed*"))
        assert len(processed_files) > 0

    def test_process_with_custom_output(self, tmp_path: Path, fire_data_path: Path) -> None:
        config_path = REPO_ROOT / "config/fires.json"
        custom_output = tmp_path / "custom_output"
        result = runner.invoke(
            app,
            [
                "process",
                str(fire_data_path),
                "--config",
                str(config_path),
                "--output",
                str(custom_output),
            ],
        )
        assert result.exit_code == 0

    def test_process_nonexistent_input_exits_error(self) -> None:
        config_path = REPO_ROOT / "config/fires.json"
        result = runner.invoke(
            app,
            ["process", "/nonexistent/fire.npz", "--config", str(config_path)],
        )
        assert result.exit_code != 0

    def test_process_json_format(self, fire_data_path: Path) -> None:
        config_path = REPO_ROOT / "config/fires.json"
        result = runner.invoke(
            app,
            ["process", str(fire_data_path), "--config", str(config_path), "--format", "json"],
        )
        assert result.exit_code == 0
        processed_dir = fire_data_path.parent / "processed"
        json_files = list(processed_dir.glob("*.json"))
        assert len(json_files) > 0


# ---------------------------------------------------------------------------
# download command (error paths only — requires GEE credentials for success)
# ---------------------------------------------------------------------------


class TestDownloadCommand:
    def test_unknown_fire_exits_error(self) -> None:
        config_path = REPO_ROOT / "config/fires.json"
        result = runner.invoke(
            app,
            ["download", "NonexistentFire", "--config", str(config_path)],
        )
        assert result.exit_code != 0

    def test_invalid_config_exits_error(self) -> None:
        result = runner.invoke(
            app,
            ["download", "Kincade", "--config", "/nonexistent/config.json"],
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Round-trip integration: save → process → validate
# ---------------------------------------------------------------------------


class TestRoundTripIntegration:
    """Verify the full pipeline: save raw data → process_fire → validate output."""

    def test_save_process_validate_round_trip(self, tmp_path: Path) -> None:
        T, H, W = 12, 6, 6
        rng = np.random.default_rng(99)

        # Create realistic fire data with clusters
        confidence = np.zeros((T, H, W), dtype=np.float32)
        for t in range(T):
            r = min(t + 1, 3)
            confidence[t, 2 : 2 + r, 2 : 2 + r] = rng.uniform(0.4, 1.0, size=(r, r))

        obs_valid = np.ones((T, H, W), dtype=np.float32)
        obs_valid[4] = 0.0  # one cloudy hour
        cloud_mask = (1.0 - obs_valid).astype(np.float32)
        frp = (confidence * rng.uniform(10, 200, size=(T, H, W))).astype(np.float32)

        arrays = {
            "data": confidence,
            "observation_valid": obs_valid,
            "cloud_mask": cloud_mask,
            "frp": frp,
        }
        metadata = {"fire_name": "RoundTrip", "n_hours": T}

        # Step 1: Save raw fire data
        raw_path = save_fire_data(tmp_path / "roundtrip_raw", arrays, metadata, fmt="npz")

        # Step 2: Process via CLI
        config_path = REPO_ROOT / "config/fires.json"
        proc_result = runner.invoke(
            app,
            ["process", str(raw_path), "--config", str(config_path)],
        )
        assert proc_result.exit_code == 0, f"Process failed: {proc_result.output}"

        # Step 3: Find processed output
        processed_dir = raw_path.parent / "processed"
        processed_files = list(processed_dir.glob("*_processed*"))
        assert len(processed_files) > 0, "No processed output found"
        processed_path = processed_files[0]

        # Step 4: Validate processed output via CLI
        val_result = runner.invoke(app, ["validate", str(processed_path)])
        assert val_result.exit_code == 0, f"Validation failed: {val_result.output}"
