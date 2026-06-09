"""
gee_utils.py — Google Earth Engine utilities for satellite data fetching.

V3 pipeline:
  - GEE initialization (local dev + service-account prod)
  - 4-band Sentinel-2 patch extraction (B2, B3, B4, B8)
  - NDVI computation from fetched bands
  - RGB thumbnail URL generation
"""

import json
import os
import logging

import ee
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Sentinel-2 bands used by the V3 model
BANDS = ["B2", "B3", "B4", "B8"]


def init_gee() -> None:
    """
    Initialize Google Earth Engine.

    In production (Hugging Face Spaces / Docker), uses service account
    credentials from environment variables GEE_SA_EMAIL and GEE_CREDENTIALS.

    For local development, falls back to ee.Authenticate() + ee.Initialize().
    Run `earthengine authenticate` in your terminal first.
    """
    sa_email = os.environ.get("GEE_SA_EMAIL")
    sa_key = os.environ.get("GEE_CREDENTIALS")

    if sa_email and sa_key:
        # Production: service account
        logger.info("Initializing GEE with service account: %s", sa_email)
        credentials = ee.ServiceAccountCredentials(
            email=sa_email,
            key_data=sa_key,
        )
        ee.Initialize(credentials)
    else:
        # Local dev: assumes `earthengine authenticate` has been run
        logger.info("Initializing GEE with default credentials (local dev)")
        ee.Initialize()

    logger.info("GEE initialized successfully")


# def _get_sentinel_collection(
#     region: ee.Geometry,
#     year: int,
#     cloud_pct: int = 10,
# ) -> ee.ImageCollection:
#     """
#     Return a filtered Sentinel-2 SR Harmonized collection for a region/year.
#     """
#     return (
#         ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
#         .filterBounds(region)
#         .filterDate(f"{year}-01-01", f"{year}-12-31")
#         .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
#     )


def _get_sentinel_collection(
    region: ee.Geometry,
    year: int,
    cloud_pct: int = 10,
) -> ee.ImageCollection:
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
    )

    # Fallback: relax cloud filter if no images found
    if collection.size().getInfo() == 0:
        logger.warning(
            "No images found for year=%s with cloud_pct=%s, relaxing to 50%%",
            year, cloud_pct
        )
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
        )

    return collection


def _resize_band(arr: np.ndarray, size: int) -> np.ndarray:
    """Resize a 2D array to (size, size) using bilinear interpolation."""
    img = Image.fromarray(arr)
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.float32)


# def fetch_sentinel_patch(
#     lat: float,
#     lon: float,
#     year: int,
#     cloud_pct: int = 10,
#     patch_size: int = 256,
# ) -> np.ndarray:
#     """
#     Fetch a 4-band (B2, B3, B4, B8) Sentinel-2 patch as a numpy array.

#     Args:
#         lat:        Latitude of center point.
#         lon:        Longitude of center point.
#         year:       Calendar year to composite.
#         cloud_pct:  Max cloud pixel percentage filter.
#         patch_size: Output patch dimension (square).

#     Returns:
#         np.ndarray of shape (4, patch_size, patch_size), dtype float32.
#         Band order: [B2, B3, B4, B8].
#         Values are raw Sentinel-2 surface reflectance (typically 0–10000).
#     """
#     point = ee.Geometry.Point([lon, lat])
#     # ~2.56 km × 2.56 km at 10 m resolution → 256 pixels
#     # region = point.buffer(1280).bounds()
#     region = point.buffer(1200).bounds()  # was 1280, go slightly smaller

#     collection = _get_sentinel_collection(region, year, cloud_pct)
#     image = collection.median().clip(region)

#     # Guard: verify bands exist before selecting
#     band_names = image.bandNames().getInfo()
#     if not band_names:
#         raise ValueError(
#             f"No Sentinel-2 imagery available at ({lat}, {lon}) for year {year}"
#         )

#     # Select the 4 bands
#     multi_band = image.select(BANDS)

#     # Download as numpy via sampleRectangle
#     sample = multi_band.sampleRectangle(region=region, defaultValue=0)
#     properties = sample.getInfo()["properties"]

#     # Stack bands into (4, H, W) array, resizing each to patch_size
#     band_arrays = []
#     for band in BANDS:
#         arr = np.array(properties[band], dtype=np.float32)
#         arr = _resize_band(arr, patch_size)
#         band_arrays.append(arr)

#     return np.stack(band_arrays, axis=0)  # (4, 256, 256)



def fetch_sentinel_patch(
    lat: float,
    lon: float,
    year: int,
    cloud_pct: int = 10,
    patch_size: int = 256,
) -> np.ndarray:
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(1200).bounds()

    if year <= 2020:
        start, end = "2018-01-01", "2019-12-31"
    else:
        start, end = "2023-01-01", "2025-05-31"

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .map(lambda img: img.updateMask(
            img.select('QA60').bitwiseAnd(1 << 10).eq(0)
            .And(img.select('QA60').bitwiseAnd(1 << 11).eq(0))
        ))
    )

    if collection.size().getInfo() == 0:
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
        )

    image = collection.median().clip(region)

    band_names = image.bandNames().getInfo()
    if not band_names:
        raise ValueError(f"No imagery at ({lat}, {lon})")

    # ── NEW: use getDownloadURL instead of sampleRectangle ──
    
    import requests, tempfile
    import imageio.v3 as iio

    url = image.select(BANDS).getDownloadURL({
        "region": region,
        "scale": 10,
        "format": "GEO_TIFF",
        "crs": "EPSG:4326",
    })

    r = requests.get(url, timeout=60)
    r.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        f.write(r.content)
        tmp_path = f.name

    arr = iio.imread(tmp_path, plugin="tifffile")  # (H, W, 4)
    arr = arr.astype(np.float32).transpose(2, 0, 1)  # (4, H, W)

    logger.info("Downloaded patch shape: %s, min: %.1f, max: %.1f",
                arr.shape, arr.min(), arr.max())

    band_arrays = []
    for i in range(arr.shape[0]):
        band_arrays.append(_resize_band(arr[i], patch_size))

    return np.stack(band_arrays, axis=0)  # (4, 256, 256)



def compute_ndvi_from_patch(patch: np.ndarray) -> float:
    """
    Compute mean NDVI from a 4-band patch.

    Band order is [B2, B3, B4, B8], so:
      - B4 (Red) is at index 2
      - B8 (NIR) is at index 3

    Returns:
        Mean NDVI as a float, in range [-1, 1].
    """
    red = patch[2]   # B4
    nir = patch[3]   # B8
    # Avoid division by zero
    denominator = nir + red
    denominator = np.where(denominator == 0, 1e-10, denominator)
    ndvi = (nir - red) / denominator
    return float(np.mean(ndvi))


def fetch_rgb_thumbnail(
    lat: float,
    lon: float,
    year: int,
    cloud_pct: int = 10,
    dimensions: int = 256,
) -> str:
    """
    Generate a Sentinel-2 true-color (RGB) thumbnail URL.

    Args:
        lat:        Latitude of center point.
        lon:        Longitude of center point.
        year:       Calendar year to composite.
        cloud_pct:  Max cloud pixel percentage filter.
        dimensions: Thumbnail size in pixels (square).

    Returns:
        URL string pointing to the rendered PNG thumbnail.
    """
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(1200).bounds()  # was 1280, go slightly smaller

    collection = _get_sentinel_collection(region, year, cloud_pct)
    image = collection.median().clip(region)
    rgb = image.select(["B4", "B3", "B2"])

    url = rgb.getThumbURL({
        "region": region,
        "dimensions": dimensions,
        "format": "png",
        "min": 0,
        "max": 3000,
    })
    return url
