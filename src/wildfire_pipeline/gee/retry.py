from __future__ import annotations

import ee
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

RETRYABLE_EXCEPTIONS = (
    ee.EEException,
    ConnectionError,
    TimeoutError,
    OSError,
)


@retry(
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def safe_get_info(ee_object: ee.ComputedObject) -> dict:
    """Retrieve .getInfo() with retry on transient failures."""
    result: dict = ee_object.getInfo()  # type: ignore[assignment]
    return result


@retry(
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def safe_sample_rectangle(
    image: ee.Image, region: ee.Geometry, default_value: float = 0
) -> ee.Feature:  # sampleRectangle returns a Feature, not Dictionary
    """Sample rectangle with retry on transient failures."""
    return image.sampleRectangle(region=region, defaultValue=default_value)


def init_ee(high_volume: bool = True) -> None:
    """Initialize Earth Engine with retry on auth failure."""
    url = "https://earthengine-highvolume.googleapis.com" if high_volume else None
    try:
        ee.Initialize(opt_url=url)
    except ee.EEException:
        ee.Authenticate()
        ee.Initialize(opt_url=url)
