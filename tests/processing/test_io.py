"""Tests for fire data I/O: save_fire_data, load_fire_data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from wildfire_pipeline.processing.io import load_fire_data, save_fire_data

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_METADATA = {
    "fire_name": "TestFire",
    "year": 2023,
    "n_hours": 10,
    "grid_shape": [4, 5],
    "aoi": [-122.0, 38.0, -121.0, 39.0],
    "pipeline": "wildfire-data-pipeline",
    "nested": {"key": "value", "numbers": [1, 2, 3]},
}


def _make_arrays(n_times: int = 10, n_height: int = 4, n_width: int = 5) -> dict[str, np.ndarray]:
    """Build a set of realistic arrays for round-trip testing."""
    rng = np.random.default_rng(42)
    shape = (n_times, n_height, n_width)
    return {
        "data": rng.random(shape).astype(np.float32),
        "observation_valid": rng.integers(0, 2, size=shape).astype(np.float32),
        "cloud_mask": rng.integers(0, 2, size=shape).astype(np.float32),
        "frp": (rng.random(shape) * 500).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# NPZ format
# ---------------------------------------------------------------------------


class TestNpzRoundTrip:
    """NPZ save/load round-trip tests."""

    def test_arrays_preserved(self, tmp_path: Path) -> None:
        arrays = _make_arrays()
        out = save_fire_data(tmp_path / "fire", arrays, SAMPLE_METADATA, fmt="npz")
        loaded_arrays, _ = load_fire_data(out)

        assert set(loaded_arrays.keys()) == set(arrays.keys())
        for name in arrays:
            np.testing.assert_array_equal(loaded_arrays[name], arrays[name])

    def test_metadata_preserved(self, tmp_path: Path) -> None:
        arrays = _make_arrays()
        out = save_fire_data(tmp_path / "fire", arrays, SAMPLE_METADATA, fmt="npz")
        _, loaded_meta = load_fire_data(out)

        assert loaded_meta == SAMPLE_METADATA

    def test_extension_is_npz(self, tmp_path: Path) -> None:
        arrays = _make_arrays()
        out = save_fire_data(tmp_path / "fire", arrays, SAMPLE_METADATA, fmt="npz")
        assert out.suffix == ".npz"

    def test_extension_corrected_from_json(self, tmp_path: Path) -> None:
        arrays = _make_arrays()
        out = save_fire_data(tmp_path / "fire.json", arrays, SAMPLE_METADATA, fmt="npz")
        assert out.suffix == ".npz"

    def test_dtype_preserved(self, tmp_path: Path) -> None:
        arrays = {"vals": np.array([1.0, 2.0, 3.0], dtype=np.float64)}
        out = save_fire_data(tmp_path / "fire", arrays, {}, fmt="npz")
        loaded_arrays, _ = load_fire_data(out)
        assert loaded_arrays["vals"].dtype == np.float64

    def test_2d_array(self, tmp_path: Path) -> None:
        arrays = {"grid": np.ones((4, 5), dtype=np.float32)}
        out = save_fire_data(tmp_path / "fire", arrays, {}, fmt="npz")
        loaded_arrays, _ = load_fire_data(out)
        np.testing.assert_array_equal(loaded_arrays["grid"], arrays["grid"])

    def test_empty_metadata(self, tmp_path: Path) -> None:
        arrays = _make_arrays()
        out = save_fire_data(tmp_path / "fire", arrays, {}, fmt="npz")
        _, loaded_meta = load_fire_data(out)
        assert loaded_meta == {}


# ---------------------------------------------------------------------------
# JSON format (legacy)
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    """JSON save/load round-trip tests."""

    def test_arrays_preserved(self, tmp_path: Path) -> None:
        arrays = _make_arrays()
        out = save_fire_data(tmp_path / "fire", arrays, SAMPLE_METADATA, fmt="json")
        loaded_arrays, _ = load_fire_data(out)

        assert set(loaded_arrays.keys()) == set(arrays.keys())
        for name in arrays:
            np.testing.assert_allclose(
                loaded_arrays[name],
                arrays[name],
                rtol=1e-5,
                err_msg=f"Mismatch in array '{name}'",
            )

    def test_metadata_preserved(self, tmp_path: Path) -> None:
        arrays = _make_arrays()
        out = save_fire_data(tmp_path / "fire", arrays, SAMPLE_METADATA, fmt="json")
        _, loaded_meta = load_fire_data(out)

        assert loaded_meta == SAMPLE_METADATA

    def test_extension_is_json(self, tmp_path: Path) -> None:
        arrays = _make_arrays()
        out = save_fire_data(tmp_path / "fire", arrays, SAMPLE_METADATA, fmt="json")
        assert out.suffix == ".json"

    def test_json_arrays_load_as_float32(self, tmp_path: Path) -> None:
        """JSON loader always returns float32 arrays (legacy compat)."""
        arrays = {"vals": np.array([1.0, 2.0], dtype=np.float64)}
        out = save_fire_data(tmp_path / "fire", arrays, {}, fmt="json")
        loaded_arrays, _ = load_fire_data(out)
        assert loaded_arrays["vals"].dtype == np.float32


# ---------------------------------------------------------------------------
# Zarr format
# ---------------------------------------------------------------------------


class TestZarrRoundTrip:
    """Zarr save/load round-trip tests."""

    def test_arrays_preserved(self, tmp_path: Path) -> None:
        arrays = _make_arrays()
        out = save_fire_data(tmp_path / "fire", arrays, SAMPLE_METADATA, fmt="zarr")
        loaded_arrays, _ = load_fire_data(out)

        assert set(loaded_arrays.keys()) == set(arrays.keys())
        for name in arrays:
            np.testing.assert_allclose(
                loaded_arrays[name],
                arrays[name],
                rtol=1e-5,
                err_msg=f"Mismatch in array '{name}'",
            )

    def test_metadata_preserved(self, tmp_path: Path) -> None:
        arrays = _make_arrays()
        out = save_fire_data(tmp_path / "fire", arrays, SAMPLE_METADATA, fmt="zarr")
        _, loaded_meta = load_fire_data(out)

        assert loaded_meta == SAMPLE_METADATA

    def test_extension_is_zarr(self, tmp_path: Path) -> None:
        arrays = _make_arrays()
        out = save_fire_data(tmp_path / "fire", arrays, SAMPLE_METADATA, fmt="zarr")
        assert out.suffix == ".zarr"

    def test_2d_array(self, tmp_path: Path) -> None:
        arrays = {"grid": np.ones((4, 5), dtype=np.float32)}
        out = save_fire_data(tmp_path / "fire", arrays, {}, fmt="zarr")
        loaded_arrays, _ = load_fire_data(out)
        np.testing.assert_array_equal(loaded_arrays["grid"], arrays["grid"])


# ---------------------------------------------------------------------------
# Format auto-detection
# ---------------------------------------------------------------------------


class TestFormatAutoDetection:
    """load_fire_data should auto-detect format from file extension."""

    def test_detects_npz(self, tmp_path: Path) -> None:
        arrays = _make_arrays()
        out = save_fire_data(tmp_path / "fire", arrays, SAMPLE_METADATA, fmt="npz")
        loaded_arrays, loaded_meta = load_fire_data(out)
        assert "data" in loaded_arrays
        assert loaded_meta["fire_name"] == "TestFire"

    def test_detects_json(self, tmp_path: Path) -> None:
        arrays = _make_arrays()
        out = save_fire_data(tmp_path / "fire", arrays, SAMPLE_METADATA, fmt="json")
        loaded_arrays, loaded_meta = load_fire_data(out)
        assert "data" in loaded_arrays
        assert loaded_meta["fire_name"] == "TestFire"

    def test_detects_zarr(self, tmp_path: Path) -> None:
        arrays = _make_arrays()
        out = save_fire_data(tmp_path / "fire", arrays, SAMPLE_METADATA, fmt="zarr")
        loaded_arrays, loaded_meta = load_fire_data(out)
        assert "data" in loaded_arrays
        assert loaded_meta["fire_name"] == "TestFire"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error cases."""

    def test_unknown_save_format_raises(self, tmp_path: Path) -> None:
        arrays = _make_arrays()
        with pytest.raises(ValueError, match="Unknown format"):
            save_fire_data(tmp_path / "fire", arrays, {}, fmt="parquet")

    def test_unknown_load_extension_raises(self, tmp_path: Path) -> None:
        bad_path = tmp_path / "fire.xyz"
        bad_path.touch()
        with pytest.raises(ValueError, match="Unknown file format"):
            load_fire_data(bad_path)

    def test_save_creates_parent_dirs_not_needed(self, tmp_path: Path) -> None:
        """save_fire_data should work when parent dir already exists."""
        arrays = _make_arrays()
        out = save_fire_data(tmp_path / "fire", arrays, {}, fmt="npz")
        assert out.exists()


# ---------------------------------------------------------------------------
# Metadata edge cases
# ---------------------------------------------------------------------------


class TestMetadataEdgeCases:
    """Test that various metadata types survive round-trip."""

    def test_nested_dicts_preserved(self, tmp_path: Path) -> None:
        meta = {"outer": {"inner": {"deep": 42}}}
        arrays = {"x": np.array([1.0], dtype=np.float32)}
        out = save_fire_data(tmp_path / "fire", arrays, meta, fmt="npz")
        _, loaded_meta = load_fire_data(out)
        assert loaded_meta["outer"]["inner"]["deep"] == 42

    def test_list_values_preserved(self, tmp_path: Path) -> None:
        meta = {"coords": [1.0, 2.0, 3.0], "tags": ["a", "b"]}
        arrays = {"x": np.array([1.0], dtype=np.float32)}
        out = save_fire_data(tmp_path / "fire", arrays, meta, fmt="npz")
        _, loaded_meta = load_fire_data(out)
        assert loaded_meta["coords"] == [1.0, 2.0, 3.0]
        assert loaded_meta["tags"] == ["a", "b"]

    def test_boolean_metadata_preserved(self, tmp_path: Path) -> None:
        meta = {"cloud_masking": True, "debug": False}
        arrays = {"x": np.array([1.0], dtype=np.float32)}
        out = save_fire_data(tmp_path / "fire", arrays, meta, fmt="npz")
        _, loaded_meta = load_fire_data(out)
        assert loaded_meta["cloud_masking"] is True
        assert loaded_meta["debug"] is False

    def test_null_metadata_value_preserved(self, tmp_path: Path) -> None:
        meta = {"optional_field": None}
        arrays = {"x": np.array([1.0], dtype=np.float32)}
        out = save_fire_data(tmp_path / "fire", arrays, meta, fmt="npz")
        _, loaded_meta = load_fire_data(out)
        assert loaded_meta["optional_field"] is None


# ---------------------------------------------------------------------------
# Edge cases for coverage
# ---------------------------------------------------------------------------


class TestZarrNonStandardDims:
    """Zarr save/load for arrays with ndim != 2 or 3."""

    def test_1d_array_zarr_round_trip(self, tmp_path: Path) -> None:
        arrays = {"signal": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
        meta = {"name": "1d_test"}
        out = save_fire_data(tmp_path / "one_d", arrays, meta, fmt="zarr")
        loaded_arrays, loaded_meta = load_fire_data(out)
        np.testing.assert_array_almost_equal(loaded_arrays["signal"], arrays["signal"])
        assert loaded_meta["name"] == "1d_test"


class TestLegacyJsonFormat:
    """Load JSON files written in the old GOFER-compatible format (no _metadata key)."""

    def test_load_legacy_json_with_metadata_key(self, tmp_path: Path) -> None:
        """Old format: top-level 'metadata' dict + array lists."""
        import json

        legacy = {
            "metadata": {"fire_name": "Legacy", "year": 2019},
            "data": [[0.5, 0.6], [0.7, 0.8]],
            "observation_valid": [[1.0, 1.0], [1.0, 0.0]],
        }
        path = tmp_path / "legacy.json"
        with open(path, "w") as f:
            json.dump(legacy, f)

        arrays, metadata = load_fire_data(path)
        assert metadata["fire_name"] == "Legacy"
        assert metadata["year"] == 2019
        assert "data" in arrays
        assert arrays["data"].dtype == np.float32

    def test_load_legacy_json_with_scalar_metadata(self, tmp_path: Path) -> None:
        """Old format with scalar metadata mixed at top level."""
        import json

        legacy = {
            "fire_name": "OldFormat",
            "year": 2020,
            "data": [[0.1, 0.2], [0.3, 0.4]],
        }
        path = tmp_path / "old.json"
        with open(path, "w") as f:
            json.dump(legacy, f)

        arrays, metadata = load_fire_data(path)
        assert metadata["fire_name"] == "OldFormat"
        assert "data" in arrays
