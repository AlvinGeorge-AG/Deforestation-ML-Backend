"""
main.py — FastAPI backend for the Kerala Deforestation Detector.

V3 pipeline:
  - Fetches 4 Sentinel-2 bands (B2, B3, B4, B8) for each year from GEE
  - Stacks into 8-channel input tensor
  - Runs U-Net inference (outputs logits → sigmoid → threshold)
  - Returns deforestation mask + statistics

Endpoints:
  GET  /health   → liveness check
  POST /analyze  → run deforestation analysis for a lat/lon + year pair
"""

import base64
import io
import logging

import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from gee_utils import (
    compute_ndvi_from_patch,
    fetch_rgb_thumbnail,
    fetch_sentinel_patch,
    init_gee,
)
from model import load_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()  # load .env if present (GEE_SA_EMAIL, GEE_CREDENTIALS, etc.)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Deforestation Detector API",
    description=(
        "Detects deforestation between two years using 8-band Sentinel-2 "
        "imagery (B2, B3, B4, B8 × 2 years) and a U-Net model trained on "
        "Dynamic World labels."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup: load model + initialize GEE once
# ---------------------------------------------------------------------------
model = None


# @app.on_event("startup")
# def startup() -> None:
#     """Load ML model and initialize Google Earth Engine on server start."""
#     global model
#     logger.info("Loading U-Net model (V3, 8-channel)…")
#     model = load_model("best_model.pth")
#     logger.info("Model loaded ✓")

#     logger.info("Initializing Google Earth Engine…")
#     init_gee()
#     logger.info("GEE initialized ✓")

@app.on_event("startup")
def startup():
    global model

    logger.info("Loading model...")
    model = load_model("best_model.pth")
    logger.info("Model loaded")

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    """Request body for the /analyze endpoint."""
    lat: float = Field(..., description="Latitude of the center point", ge=-90, le=90)
    lon: float = Field(..., description="Longitude of the center point", ge=-180, le=180)
    year_a: int = Field(..., description="Starting year (before)", ge=2015, le=2026)
    year_b: int = Field(..., description="Ending year (after)", ge=2015, le=2026)


class AnalyzeResponse(BaseModel):
    """Response body from the /analyze endpoint."""
    mask_image: str = Field(..., description="Base64-encoded RGBA PNG of the deforestation mask")
    pct_lost: float = Field(..., description="Percentage of area classified as deforested")
    area_sqkm: float = Field(..., description="Estimated deforested area in km²")
    ndvi_before: float = Field(..., description="Mean NDVI computed from B4/B8 for year A")
    ndvi_after: float = Field(..., description="Mean NDVI computed from B4/B8 for year B")
    thumbnail_a: str = Field(..., description="RGB satellite thumbnail URL for year A")
    thumbnail_b: str = Field(..., description="RGB satellite thumbnail URL for year B")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def mask_to_base64_png(mask: np.ndarray) -> str:
    """
    Convert a binary (0/1) mask to a red-overlay RGBA PNG, base64-encoded.

    Deforested pixels → red with slight transparency.
    Non-deforested    → fully transparent.
    """
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask == 1] = [255, 0, 0, 180]  # red with alpha=180
    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def normalize_bands(patch: np.ndarray) -> np.ndarray:
    """
    Normalize Sentinel-2 surface reflectance values to [0, 1].

    Sentinel-2 SR values are typically in [0, 10000].
    We clip and divide by 10000.

    Args:
        patch: Array of shape (4, 256, 256) with raw SR values.

    Returns:
        Normalized array of same shape, values in [0, 1].
    """
    return np.clip(patch, 0, 10000).astype(np.float32) / 10000.0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Kerala Deforestation Detector API"
    }

@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """
    Main analysis endpoint (V3 pipeline).

    1. Fetches 4-band Sentinel-2 patches (B2, B3, B4, B8) for both years
    2. Normalizes and stacks into 8-channel tensor
    3. Runs U-Net inference with sigmoid thresholding
    4. Computes NDVI from bands for FE display
    5. Returns the deforestation mask + statistics
    """
    # ---- Step 1: Fetch 4-band patches from GEE ----
    try:
        logger.info(
            "Fetching Sentinel-2 bands for (%.4f, %.4f) — years %d & %d",
            req.lat, req.lon, req.year_a, req.year_b,
        )
        patch_before = fetch_sentinel_patch(req.lat, req.lon, req.year_a)  # (4, 256, 256)
        patch_after = fetch_sentinel_patch(req.lat, req.lon, req.year_b)   # (4, 256, 256)
    except Exception as e:
        logger.error("GEE fetch failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Google Earth Engine fetch failed: {e}",
        )

    # ---- Step 2: Normalize bands from [0, 10000] to [0, 1] ----
    before_norm = normalize_bands(patch_before)  # (4, 256, 256)
    after_norm = normalize_bands(patch_after)     # (4, 256, 256)

    # ---- Step 3: Stack → 8-channel tensor and run inference ----
    stacked = np.concatenate([before_norm, after_norm], axis=0)  # (8, 256, 256)
    input_tensor = torch.tensor(
        stacked[np.newaxis],  # (1, 8, 256, 256)
        dtype=torch.float32,
    )

    with torch.no_grad():
        logits = model(input_tensor)      
        input_tensor = input_tensor.to(next(model.parameters()).device)                                # (1, 1, 256, 256)
        probs = torch.sigmoid(logits)                                     # sigmoid on logits
        binary_mask = (probs > 0.5).int().squeeze().numpy()               # (256, 256)

    # ---- Step 4: Compute statistics ----
    total_pixels = 256 * 256
    deforested_pixels = int(binary_mask.sum())

    # Sentinel-2 at 10 m resolution → each pixel ≈ 100 m²
    pixel_area_sqkm = (10 * 10) / 1_000_000  # 0.0001 km²
    area_sqkm = round(deforested_pixels * pixel_area_sqkm, 4)
    pct_lost = round((deforested_pixels / total_pixels) * 100, 2)

    # ---- Step 5: Compute NDVI from raw bands (for FE display) ----
    ndvi_before = compute_ndvi_from_patch(patch_before)
    ndvi_after = compute_ndvi_from_patch(patch_after)

    # ---- Step 6: Get RGB thumbnails ----
    try:
        thumb_a = fetch_rgb_thumbnail(req.lat, req.lon, req.year_a)
        thumb_b = fetch_rgb_thumbnail(req.lat, req.lon, req.year_b)
    except Exception as e:
        logger.warning("Thumbnail fetch failed, returning empty: %s", e)
        thumb_a = ""
        thumb_b = ""

    # ---- Step 7: Encode mask and return ----
    logger.info(
        "Analysis complete — %.2f%% deforested (%.4f km²), NDVI: %.4f → %.4f",
        pct_lost, area_sqkm, ndvi_before, ndvi_after,
    )

    return AnalyzeResponse(
        mask_image=mask_to_base64_png(binary_mask),
        pct_lost=pct_lost,
        area_sqkm=area_sqkm,
        ndvi_before=round(ndvi_before, 4),
        ndvi_after=round(ndvi_after, 4),
        thumbnail_a=thumb_a,
        thumbnail_b=thumb_b,
    )
