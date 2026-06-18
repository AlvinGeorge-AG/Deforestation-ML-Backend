---

## **LOW-LEVEL PIPELINE BREAKDOWN**

### **LAYER 1: THE DATA SOURCE**

**What raw data exists?**
- **14 district files** from Kerala, each split into 3 files:
  - `district_before.tif` — Sentinel-2 satellite image (2018-2019)
  - `district_after.tif` — Sentinel-2 satellite image (2023-2025)  
  - `district_dw_loss_mask.tif` — **Ground truth** binary label (Dynamic World labels)

**What is a `.tif` file?**
- GeoTIFF = georeferenced TIFF image. Stores satellite imagery with location metadata.
- Each `.tif` has **4 bands** (not RGB—these are Sentinel-2 spectral bands):
  - **B2** (Blue, 490nm) 
  - **B3** (Green, 560nm)
  - **B4** (Red, 665nm)
  - **B8** (Near-Infrared/NIR, 842nm)
- Dimensions: **Huge** (8000+ × 8000+ pixels). Each pixel = 10m × 10m on Earth.
- **Values**: Raw Sentinel-2 surface reflectance in range `[0, 10000]`.

**What is the mask file?**
- Binary image (0 = forest/non-deforested, 1 = deforested area).
- Labels come from **Dynamic World** (a Google AI dataset of land cover).
- Same spatial dimensions as the before/after images.

---

### **LAYER 2: PATCH GENERATION (Data Preprocessing)**

**Why?** The `.tif` files are too large (8000+ × 8000 pixels). We can't feed that to a neural network.

**What happens?**

```
For each district:
  1. Load before.tif (4 channels, huge size)
  2. Load after.tif  (4 channels, huge size)
  3. Load mask.tif   (1 channel, huge size)
  4. Normalize values from [0, 10000] → [0, 1]
  5. Slide a 256×256 window across the image with STRIDE=128
```

**Example:**
```
Image size: 8000×8000
Window size: 256×256
Stride: 128 pixels
→ Generates ~1000 patches per district
```

**What gets saved?**
```
/patches_v3/train/
  ├─ kollam_000000_before.npy   ← (4, 256, 256)
  ├─ kollam_000000_after.npy    ← (4, 256, 256)
  ├─ kollam_000000_mask.npy     ← (256, 256)
  ├─ kollam_000001_before.npy
  ├─ ...
```

**Key filtering:**
- Skip any patch where `mask.sum() == 0` (no deforestation at all).
- This avoids training on "negative only" patches, speeding up learning.
- Result: ~8 districts → train set, ~3 districts → val set, ~3 districts → test set.

---

### **LAYER 3: THE MODEL ARCHITECTURE**

**Model type: U-Net with ResNet-34 encoder**

**Architecture diagram:**
```
INPUT: 8-channel tensor (256, 256, 8)
  ↓
ENCODER (ResNet-34):
  - Conv layers that compress spatially, extract features
  - Outputs: feature maps at different scales (64, 128, 256, 512 channels)
  ↓
DECODER:
  - Upsampling layers that expand back to original size
  - Skip connections from encoder (crucial: preserve fine details)
  ↓
OUTPUT: 1-channel logits (256, 256, 1)
  ↓
SIGMOID activation → probabilities [0, 1]
  ↓
THRESHOLD at 0.5 → binary mask {0, 1}
```

**Input channels (WHY 8?):**
```
Channel 0 = B2_before
Channel 1 = B3_before
Channel 2 = B4_before
Channel 3 = B8_before (NIR before)
Channel 4 = B2_after
Channel 5 = B3_after
Channel 6 = B4_after
Channel 7 = B8_after (NIR after)
```

The model learns to detect **changes** by comparing before/after spectral signatures.

**Loss function (training only):**
```
Composite loss = 0.5 × Dice Loss + 0.5 × BCE-with-Logits Loss

Why two losses?
- Dice Loss: Good at handling class imbalance (deforestation is rare)
- BCE Loss: Standard binary classification loss
- Together: Robust training
```

**Evaluation metric: IoU (Intersection over Union)**
```
IoU = (True Positives) / (True Positives + False Positives + False Negatives)
      = overlap / union
Range: [0, 1]. Higher = better.
```

---

### **LAYER 4: TRAINING PROCESS**

**What happens during training? (Step by step)**

```
1. Load batch of 8 patches
   - Inputs: (8, 8, 256, 256)  ← batch_size × channels × height × width
   - Masks:  (8, 1, 256, 256)  ← batch_size × 1 × height × width

2. Data augmentation (train only):
   - Horizontal flip (50% chance)
   - Vertical flip (50% chance)
   - Rotate 90° (50% chance)
   - Shift/scale/rotate by small amounts (30% chance)
   - Why? Models learn rotation/flip invariance → generalize better

3. Forward pass:
   - Push inputs through U-Net
   - Get logits (raw predictions, unbounded)

4. Compute loss:
   - Compare logits vs ground truth masks
   - Loss tells us how wrong we were

5. Backward pass:
   - Compute gradients (how each weight affects loss)
   - Backpropagation through all layers

6. Optimizer step (AdamW):
   - Adjust weights to reduce loss
   - Learning rate = 1e-4 (very small steps)
   - Weight decay = 1e-5 (regularization, prevents overfitting)

7. Repeat for all training batches (1 epoch)

8. Validation phase (no augmentation, no gradients):
   - Run on validation set
   - Compute IoU (not used for training, just monitoring)
   - If val IoU improves → save model

9. Learning rate scheduler:
   - If val loss plateaus for 5 epochs → reduce learning rate by 50%
   - Helps escape local minima

10. Early stopping:
    - If val IoU doesn't improve for 10 epochs → stop training
    - Prevents overfitting
```

**Hyperparameters:**
```
Epochs: 50 (max)
Batch size: 8
Learning rate: 1e-4
Optimizer: AdamW
Scheduler: ReduceLROnPlateau
Early stopping patience: 10 epochs
```

---

### **LAYER 5: INFERENCE (Production)**

**When a user clicks on the map and selects a location:**

#### **Step 5a: Frontend (React)**
```javascript
User clicks map at (lat=10.123, lon=76.456)
↓
Frontend sends POST request to backend:
{
  "lat": 10.123,
  "lon": 76.456,
  "year_a": 2019,
  "year_b": 2024
}
```

#### **Step 5b: Backend receives request, fetches satellite data**

`gee_utils.fetch_sentinel_patch(lat, lon, year)`:

```
1. Create a point at (lat, lon)

2. Buffer it by 1200 meters in all directions
   → Creates a square region ~2.4 km × 2.4 km
   → At 10m resolution → 240 × 240 pixels
   → Resample to 256 × 256

3. Query Google Earth Engine:
   - Collection: "COPERNICUS/S2_SR_HARMONIZED"
   - Filter by region (the buffered square)
   - Filter by date range:
     * If year ≤ 2020: use composite "2018-01-01" to "2019-12-31"
     * If year > 2020: use composite "2023-01-01" to "2025-05-31"
   - Filter by cloud cover < 20%
   
4. Apply QA60 mask:
   - QA60 band encodes cloud/shadow information
   - Remove pixels that are clouds or shadows
   - Fallback: if no images found, relax to 50% cloud filter

5. Take median composite:
   - If multiple images pass filters, blend them (median)
   - Reduces noise, cloud artifacts

6. Download as GeoTIFF:
   - Use getDownloadURL() to get a temporary GCS URL
   - Download via HTTP request
   - Write to temp file

7. Read GeoTIFF → NumPy array:
   - Shape: (height, width, 4 channels)
   - Resize to (4, 256, 256)
   - Values still in [0, 10000]
   - Return
```

#### **Step 5c: Stack and normalize**

```python
patch_before = fetch_sentinel_patch(lat, lon, 2019)  # (4, 256, 256)
patch_after = fetch_sentinel_patch(lat, lon, 2024)   # (4, 256, 256)

# Normalize [0, 10000] → [0, 1]
before_norm = np.clip(patch_before, 0, 10000) / 10000
after_norm = np.clip(patch_after, 0, 10000) / 10000

# Stack into 8-channel tensor
stacked = np.concatenate([before_norm, after_norm], axis=0)  # (8, 256, 256)

# Convert to PyTorch tensor for GPU
input_tensor = torch.tensor(stacked[np.newaxis], dtype=torch.float32)
# Shape: (1, 8, 256, 256)  ← batch size 1, ready for model
```

#### **Step 5d: Run inference**

```python
model.eval()  # Set to evaluation mode (no dropout, batch norm uses running stats)

with torch.no_grad():  # Disable gradient computation (we don't need it, saves memory)
    input_tensor = input_tensor.to(device)  # Move to GPU if available
    
    logits = model(input_tensor)  # Forward pass
    # Output shape: (1, 1, 256, 256)
    
    probs = torch.sigmoid(logits)  # Convert logits → [0, 1] probabilities
    
    binary_mask = (probs > 0.5).int().squeeze().numpy()  # (256, 256), values {0, 1}
```

#### **Step 5e: Compute statistics**

```python
total_pixels = 256 × 256 = 65,536
deforested_pixels = binary_mask.sum()  # count of 1s

# Sentinel-2 resolution: 10m per pixel
pixel_area_sqkm = (10 × 10) / 1,000,000 = 0.0001 km²

area_sqkm = deforested_pixels × 0.0001
pct_lost = (deforested_pixels / total_pixels) × 100

# Example: if 5000 pixels are deforested:
# area_sqkm = 5000 × 0.0001 = 0.5 km²
# pct_lost = (5000 / 65536) × 100 = 7.63%
```

#### **Step 5f: Compute NDVI (vegetation index)**

```python
NDVI = (NIR - RED) / (NIR + RED)

# From patch:
# B4 (Red) = patch[2]
# B8 (NIR) = patch[3]

ndvi_before = (patch_before[3] - patch_before[2]) / (patch_before[3] + patch_before[2])
# Mean across all pixels: float value in [-1, 1]

# NDVI interpretation:
# -1 to 0: Water, built-up, barren
# 0.2 to 0.4: Sparse vegetation
# 0.4 to 0.6: Moderate vegetation
# 0.6 to 1.0: Dense forest
```

#### **Step 5g: Encode mask as base64 PNG**

```python
# Create RGBA image:
rgba = zeros((256, 256, 4), dtype=uint8)
# Set deforested pixels (mask==1) to red with transparency
rgba[mask == 1] = [255, 0, 0, 180]  # RGBA: red, semi-transparent

# Convert to PIL Image
img = Image.fromarray(rgba, "RGBA")

# Encode as PNG → bytes → base64 string
base64_png = base64.b64encode(png_bytes).decode("utf-8")
# Now it can be embedded in JSON response
```

#### **Step 5h: Get RGB thumbnails**

```python
# Generate visual preview (true-color RGB, not ML output)
# Select B4 (Red), B3 (Green), B2 (Blue)
rgb = image.select(["B4", "B3", "B2"])

# Get a public URL from GEE
url = rgb.getThumbURL({
    "region": region,
    "dimensions": 256,
    "format": "png",
    "min": 0,
    "max": 3000,  # clip to typical satellite values
})
# Returns a URL like: https://earthengine.googleapis.com/... that expires in hours
```

#### **Step 5i: Return response**

```python
return AnalyzeResponse(
    mask_image="data:image/png;base64,iVBORw0KGgoAAAANS...",
    pct_lost=7.63,
    area_sqkm=0.5,
    ndvi_before=0.52,
    ndvi_after=0.38,
    thumbnail_a="https://earthengine.googleapis.com/...",
    thumbnail_b="https://earthengine.googleapis.com/...",
)
# Serialized to JSON and sent to frontend
```

---

### **LAYER 6: FRONTEND DISPLAYS RESULTS**

```javascript
// api.js receives response
const result = await analyze(lat, lon, 2019, 2024)

// App.jsx stores it
setResult(result)

// ResultsPanel.jsx displays:
- mask_image as red overlay on map
- pct_lost and area_sqkm as text
- NDVI before/after as indicators
- thumbnail_a and thumbnail_b as preview images
- Mask opacity slider (0-1) to control transparency
```

---

## **SUMMARY: END-TO-END FLOW**

```
User clicks map
  ↓
Frontend sends: lat, lon, year_a, year_b
  ↓
Backend: fetch_sentinel_patch(lat, lon, 2019) via Google Earth Engine
  ↓
Backend: fetch_sentinel_patch(lat, lon, 2024) via Google Earth Engine
  ↓
Backend: Stack 8 channels, normalize [0,1]
  ↓
Backend: Feed through U-Net model (inference)
  ↓
Backend: Sigmoid + threshold → binary mask
  ↓
Backend: Compute area, NDVI, encode as PNG
  ↓
Backend: Return JSON response
  ↓
Frontend: Display mask + statistics
  ↓
User sees deforestation overlay on map
```

---