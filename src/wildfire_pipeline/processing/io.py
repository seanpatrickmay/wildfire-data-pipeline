"""I/O utilities for pipeline data.

Supports multiple storage formats:
- npz: NumPy compressed archives (default, fast, simple)
- zarr: Chunked cloud-native format (best for ML dataloaders)
- json: Legacy format for backward compatibility
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_fire_data(
    path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    fmt: str = "npz",
) -> Path:
    """Save fire data arrays and metadata.

    Args:
        path: Output path (extension will be adjusted to match format)
        arrays: Named arrays to save (e.g. {"confidence": arr, "obs_valid": arr})
        metadata: Pipeline metadata dict
        fmt: Format - "npz", "zarr", or "json"

    Returns:
        Actual path written (may differ from input if extension changed)
    """
    path = Path(path)

    if fmt == "npz":
        out = path.with_suffix(".npz")
        # Merge arrays + metadata into a single kwargs dict for savez_compressed
        save_data: dict[str, np.ndarray] = {
            **arrays,
            "_metadata": np.array(json.dumps(metadata)),
        }
        np.savez_compressed(out, **save_data)  # type: ignore[arg-type]
        return out

    elif fmt == "zarr":
        try:
            import xarray as xr
        except ImportError as exc:
            raise ImportError(
                "Zarr format requires xarray and zarr. "
                "Install with: pip install wildfire-pipeline[zarr]"
            ) from exc

        out = path.with_suffix(".zarr")
        first_arr = next(iter(arrays.values()))
        dims: tuple[str, ...]
        if first_arr.ndim == 3:
            dims = ("time", "y", "x")
        elif first_arr.ndim == 2:
            dims = ("y", "x")
        else:
            dims = tuple(f"dim_{i}" for i in range(first_arr.ndim))

        data_vars = {name: (dims[: arr.ndim], arr) for name, arr in arrays.items()}
        # xarray attrs must be JSON-serializable scalars/lists/strings.
        # Nest everything under a single JSON string to avoid type issues.
        ds = xr.Dataset(data_vars, attrs={"_metadata_json": json.dumps(metadata)})
        ds.to_zarr(out, mode="w")
        return out

    elif fmt == "json":
        out = path.with_suffix(".json")
        result: dict[str, Any] = {"_metadata": metadata}
        for name, arr in arrays.items():
            result[name] = arr.tolist()
        with open(out, "w") as f:
            json.dump(result, f)
        return out

    else:
        raise ValueError(f"Unknown format: {fmt!r}. Use 'npz', 'zarr', or 'json'.")


def load_fire_data(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load fire data from any supported format.

    Returns:
        (arrays, metadata) tuple
    """
    path = Path(path)

    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        arrays: dict[str, np.ndarray] = {}
        metadata: dict[str, Any] = {}
        for key in data.files:
            if key == "_metadata":
                metadata = json.loads(data[key].item())
            else:
                arrays[key] = data[key]
        return arrays, metadata

    elif path.suffix == ".zarr" or (path.is_dir() and (path / ".zmetadata").exists()):
        try:
            import xarray as xr
        except ImportError as exc:
            raise ImportError(
                "Zarr format requires xarray and zarr. "
                "Install with: pip install wildfire-pipeline[zarr]"
            ) from exc

        ds = xr.open_zarr(path)
        arrays = {name: ds[name].values for name in ds.data_vars}
        metadata = json.loads(ds.attrs.get("_metadata_json", "{}"))
        ds.close()
        return arrays, metadata

    elif path.suffix == ".json":
        with open(path) as f:
            raw = json.load(f)
        # New format: metadata stored under "_metadata" key
        if "_metadata" in raw:
            metadata = raw.pop("_metadata")
            arrays = {k: np.array(v, dtype=np.float32) for k, v in raw.items()}
        else:
            # Legacy format: metadata mixed with arrays at top level.
            # Heuristic: nested lists (ndim >= 2) are arrays, everything else is metadata.
            metadata = {}
            arrays = {}
            for key, val in raw.items():
                if key == "metadata":
                    metadata = val
                elif isinstance(val, list):
                    arrays[key] = np.array(val, dtype=np.float32)
                else:
                    metadata[key] = val
        return arrays, metadata

    else:
        raise ValueError(f"Unknown file format: {path.suffix!r}")
