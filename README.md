---
title: Kerala Deforestation Detector
emoji: 🌿
colorFrom: green
colorTo: green
sdk: docker
sdk_version: 5.0.0
app_file: main.py
pinned: false
---

<h1 align="center">🌿 Kerala Deforestation Detector</h1>

<p align="center">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/U--Net-ResNet34-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Google%20Earth%20Engine-Sentinel--2-4285F4?style=flat-square&logo=google&logoColor=white"/>
  <img src="https://img.shields.io/badge/Deployed-HuggingFace%20Spaces-FFD21E?style=flat-square&logo=huggingface&logoColor=black"/>
</p>

<p align="center">
  FastAPI backend for pixel-level deforestation detection over Kerala using Sentinel-2 satellite imagery and a trained U-Net segmentation model.
</p>

---

## Model Architecture

```
Input:  8 channels — B2, B3, B4, B8 (before) + B2, B3, B4, B8 (after)
Model:  U-Net with ResNet-34 encoder
Output: 1 channel (logits → sigmoid → binary mask)
Labels: Dynamic World forest cover
Loss:   0.5 × DiceLoss + 0.5 × BCEWithLogitsLoss
Val IoU: ~0.36
```

## Inference Pipeline

```
POST /analyze
  → GEE: fetch B2, B3, B4, B8 composites for year_a & year_b
  → Normalize [0, 10000] → [0, 1]
  → Stack → (1, 8, 256, 256) tensor
  → U-Net inference → sigmoid → threshold 0.5
  → Return deforestation mask + statistics
```

---

## Local Setup

**1. Python environment**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Google Earth Engine authentication**

```bash
earthengine authenticate
```

**3. Model weights**

Place `best_model.pth` in the project root.
Generate it from the training notebook: `deforestation_detector.ipynb`

**4. Run**

```bash
uvicorn main:app --reload
```

API live at `http://localhost:8000` · Docs at `http://localhost:8000/docs`

---

## API Reference

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /analyze`

**Request**

```json
{
  "lat": 10.5,
  "lon": 76.2,
  "year_a": 2018,
  "year_b": 2024
}
```

> `year_a ≤ 2020` uses 2018–2019 composite. `year_b > 2020` uses 2023–2025 composite.

**Response**

```json
{
  "mask_image": "<base64 RGBA PNG>",
  "pct_lost": 12.5,
  "area_sqkm": 0.8192,
  "ndvi_before": 0.6234,
  "ndvi_after": 0.4521,
  "thumbnail_a": "https://earthengine.googleapis.com/...",
  "thumbnail_b": "https://earthengine.googleapis.com/..."
}
```

---

## Deploy to Hugging Face Spaces

1. Create a new Space → type: **Docker**
2. Upload all files including `best_model.pth`
3. Add Secrets:
   - `GEE_SA_EMAIL` — service account email
   - `GEE_CREDENTIALS` — service account JSON key
4. Dockerfile exposes port **7860**

---

## Project Structure

```
BE/
├── main.py            # FastAPI app — 8-channel V3 pipeline
├── model.py           # U-Net loader (8 in_channels)
├── gee_utils.py       # GEE: 4-band patch fetch + NDVI + thumbnails
├── best_model.pth     # Trained weights (not in git)
├── requirements.txt
├── Dockerfile
├── .env.example
└── .gitignore
```
