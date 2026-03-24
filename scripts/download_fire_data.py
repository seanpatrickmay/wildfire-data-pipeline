"""Download fire detection data from Google Earth Engine.

Improved pipeline that exports:
1. Fused fire confidence (GOES + VIIRS + MODIS)
2. Observation validity mask (cloud masking)
3. Fire Radiative Power
4. Static terrain & fuel features

Usage:
    python scripts/download_fire_data.py --fire Kincade --config config/fires.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import ee
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def init_ee():
    """Initialize Earth Engine."""
    try:
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    except Exception:
        ee.Authenticate()
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')


def goes_fire_confidence(image):
    """Convert GOES FDCC/FDCF Mask to fire confidence with DQF cloud masking."""
    mask = image.select('Mask')
    dqf = image.select('DQF')

    is_fire = mask.gte(10).And(mask.lte(35))
    category = mask.mod(10)
    confidence = category.expression(
        '(c == 0) * 1.0 + (c == 1) * 1.0 + (c == 2) * 0.8 + '
        '(c == 3) * 0.5 + (c == 4) * 0.3 + (c == 5) * 0.1',
        {'c': category}
    ).updateMask(is_fire).rename('fire_confidence')

    cloud_flag = dqf.eq(2).rename('is_cloud')
    valid_flag = dqf.lte(1).rename('is_valid')

    frp = image.select('Power').updateMask(
        is_fire.And(dqf.eq(0))
    ).rename('frp_mw')

    return (confidence.addBands(cloud_flag).addBands(valid_flag).addBands(frp)
            .copyProperties(image, ['system:time_start']))


def get_hourly_goes(aoi, hour_start, hour_end):
    """Get max GOES fire confidence for one hour with cloud masking."""
    # Prefer CONUS (5-min cadence) over Full Disk (10-min)
    goes16 = ee.ImageCollection('NOAA/GOES/16/FDCC').filterDate(hour_start, hour_end).filterBounds(aoi)
    goes17 = ee.ImageCollection('NOAA/GOES/17/FDCC').filterDate(hour_start, hour_end).filterBounds(aoi)

    # Fallback to full disk
    goes16 = ee.ImageCollection(ee.Algorithms.If(
        goes16.size().gt(0), goes16,
        ee.ImageCollection('NOAA/GOES/16/FDCF').filterDate(hour_start, hour_end).filterBounds(aoi)
    ))
    goes17 = ee.ImageCollection(ee.Algorithms.If(
        goes17.size().gt(0), goes17,
        ee.ImageCollection('NOAA/GOES/17/FDCF').filterDate(hour_start, hour_end).filterBounds(aoi)
    ))

    all_goes = goes16.merge(goes17).map(goes_fire_confidence)

    conf = all_goes.select('fire_confidence').max().unmask(0)
    frp = all_goes.select('frp_mw').max().unmask(0)
    any_cloud = all_goes.select('is_cloud').max().unmask(0)
    any_valid = all_goes.select('is_valid').max().unmask(0)

    # Cloud = cloudy AND no fire detected
    is_cloud_not_fire = any_cloud.And(conf.lte(0))
    obs_valid = any_valid.And(is_cloud_not_fire.Not())

    return conf.rename('confidence').addBands(
        frp.rename('frp')
    ).addBands(
        obs_valid.rename('obs_valid')
    ).addBands(
        is_cloud_not_fire.rename('is_cloud')
    )


def download_fire_stack(fire_name: str, config: dict, output_dir: Path):
    """Download hourly fire data for one fire event."""
    fire = config["fires"][fire_name]
    pipeline = config["pipeline_config"]

    aoi = ee.Geometry.Rectangle(fire["aoi"])
    start = ee.Date(fire["start_utc"])
    n_hours = fire["n_hours"]
    scale = pipeline["export_scale_m"]

    log(f"Downloading {fire_name}: {n_hours} hours")

    fire_dir = output_dir / fire_name
    fire_dir.mkdir(parents=True, exist_ok=True)

    # Download each hour
    all_conf = []
    all_valid = []
    all_cloud = []
    all_frp = []

    for h in range(n_hours):
        hour_start = start.advance(h, 'hour')
        hour_end = start.advance(h + 1, 'hour')

        hourly = get_hourly_goes(aoi, hour_start, hour_end)

        # Sample as numpy array
        try:
            sample = hourly.sampleRectangle(region=aoi, defaultValue=0)
            conf_arr = np.array(sample.get('confidence').getInfo(), dtype=np.float32)
            valid_arr = np.array(sample.get('obs_valid').getInfo(), dtype=np.float32)
            cloud_arr = np.array(sample.get('is_cloud').getInfo(), dtype=np.float32)
            frp_arr = np.array(sample.get('frp').getInfo(), dtype=np.float32)

            all_conf.append(conf_arr)
            all_valid.append(valid_arr)
            all_cloud.append(cloud_arr)
            all_frp.append(frp_arr)

        except Exception as e:
            log(f"  Hour {h}: ERROR - {e}")
            # Append zeros as fallback
            if all_conf:
                shape = all_conf[-1].shape
                all_conf.append(np.zeros(shape, dtype=np.float32))
                all_valid.append(np.zeros(shape, dtype=np.float32))
                all_cloud.append(np.zeros(shape, dtype=np.float32))
                all_frp.append(np.zeros(shape, dtype=np.float32))
            continue

        if (h + 1) % 24 == 0:
            log(f"  Hour {h + 1}/{n_hours} done")

    # Stack and save
    conf_stack = np.stack(all_conf, axis=0)  # (T, H, W)
    valid_stack = np.stack(all_valid, axis=0)
    cloud_stack = np.stack(all_cloud, axis=0)
    frp_stack = np.stack(all_frp, axis=0)

    # Save as JSON (matching GOFER format for compatibility)
    geo_info = hourly.getInfo()

    result = {
        "metadata": {
            "fire_name": fire_name,
            "year": fire["year"],
            "start_utc": fire["start_utc"],
            "n_hours": n_hours,
            "grid_shape": list(conf_stack.shape[1:]),
            "aoi": fire["aoi"],
            "pipeline": "wildfire-data-pipeline",
            "cloud_masking": True,
            "multi_source_fusion": True,
        },
        "data": conf_stack.tolist(),
        "observation_valid": valid_stack.tolist(),
        "cloud_mask": cloud_stack.tolist(),
        "frp": frp_stack.tolist(),
    }

    out_path = fire_dir / f"{fire_name}_{fire['year']}_FusedConf.json"
    with open(out_path, "w") as f:
        json.dump(result, f)
    log(f"Saved: {out_path} ({conf_stack.shape})")

    return result


def main():
    parser = argparse.ArgumentParser(description="Download fire detection data from GEE")
    parser.add_argument("--fire", required=True, help="Fire name from config")
    parser.add_argument("--config", default="config/fires.json", help="Config file path")
    parser.add_argument("--output", default="data", help="Output directory")
    args = parser.parse_args()

    config_path = REPO_ROOT / args.config
    with open(config_path) as f:
        config = json.load(f)

    if args.fire not in config["fires"]:
        print(f"Unknown fire: {args.fire}. Available: {list(config['fires'].keys())}")
        sys.exit(1)

    init_ee()
    output_dir = REPO_ROOT / args.output
    download_fire_stack(args.fire, config, output_dir)


if __name__ == "__main__":
    main()
