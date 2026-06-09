# Deforestation Detector — Backend (V3)

FastAPI backend that bridges the React frontend with a V3 U-Net deforestation detection model and Google Earth Engine satellite data.

## V3 Model Architecture

```
Input:  8 channels — B2, B3, B4, B8 (before) + B2, B3, B4, B8 (after)
Model:  U-Net with ResNet-34 encoder
Output: 1 channel (logits → sigmoid → binary mask)
Labels: Dynamic World forest cover
Loss:   BCEWithLogitsLoss
```

## Architecture

```
POST /analyze
  → GEE: fetch B2, B3, B4, B8 for year_a & year_b
  → Normalize [0, 10000] → [0, 1]
  → Stack → (1, 8, 256, 256) tensor
  → U-Net inference (logits → sigmoid → threshold 0.5)
  → Return mask + stats
```

## Setup

### 1. Python environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Google Earth Engine authentication (local dev)

```bash
pip install earthengine-api
earthengine authenticate
```

### 3. Model weights

Place `best_model.pth` in this directory.  
Generate it from the V3 training notebook (`deforestation_detector.ipynb`).

### 4. Run

```bash
uvicorn main:app --reload
```

Server starts at **http://localhost:8000**.  
Docs at **http://localhost:8000/docs**.

## API

### `GET /health`
Returns `{"status": "ok"}`.

### `POST /analyze`
**Request:**
```json
{
  "lat": 10.5,
  "lon": 76.2,
  "year_a": 2015,
  "year_b": 2023
}
```

**Response:**
```json
{
  "mask_image": "<base64 PNG>",
  "pct_lost": 12.5,
  "area_sqkm": 0.8192,
  "ndvi_before": 0.6234,
  "ndvi_after": 0.4521,
  "thumbnail_a": "https://earthengine.googleapis.com/...",
  "thumbnail_b": "https://earthengine.googleapis.com/..."
}
```

> Note: `ndvi_before` and `ndvi_after` are computed from B4/B8 bands for display purposes.

## Deploy to Hugging Face Spaces

1. Create a new Space → type: **Docker**
2. Upload all files including `best_model.pth`
3. Add secrets:
   - `GEE_SA_EMAIL` — service account email
   - `GEE_CREDENTIALS` — service account JSON key
4. The Dockerfile exposes port **7860** (HF default)

## Project Structure

```
BE/
├── main.py            # FastAPI app — 8-channel V3 pipeline
├── model.py           # U-Net loader (8 in_channels)
├── gee_utils.py       # GEE: 4-band patch fetch + NDVI + thumbnails
├── best_model.pth     # V3 trained weights (not in git)
├── requirements.txt   # Python dependencies
├── Dockerfile         # HF Spaces / Docker deployment
├── .env.example       # Environment variable template
└── .gitignore
```
