# 🌍 TerraSight AI — Civic Waste Detection & Spatial Hotspot Intelligence

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**TerraSight AI** is a single-stack, production-ready computer vision and geospatial analytics platform. It ingests smartphone and drone photographs, extracts EXIF GPS metadata, detects civic waste categories using **YOLOv8**, and clusters sightings into discrete actionable hotspots using **Haversine DBSCAN** spatial aggregation.

---

## 🏗️ System Architecture

```text
  📱 Mobile / Drone Image Upload
                 │
                 ▼
  🧭 EXIF Metadata Parser (Pillow / ExifTags)
        ├── GPS Found ──────────────► Use Embedded Coordinates
        └── Missing GPS ────────────► Interactive Manual Fallback (Lat/Lon)
                 │
                 ▼
  👁️ YOLOv8 Neural Vision Engine (@st.cache_resource)
        ├── Plastic Waste
        ├── Organic Waste
        └── Dry Leaves / Biomass
                 │
                 ▼
  🛰️ Haversine DBSCAN Spatial Clustering (eps = 50m)
        ├── Merge duplicate sightings into unified hotspots
        └── Compute cluster centroids & density weights
                 │
                 ▼
  🗺️ Interactive Folium GIS Command Map
        ├── 50m hotspot boundary circles
        ├── Color-coded category markers
        └── Popup cards with base64 thumbnail previews
                 │
                 ▼
  📊 Analytics, Priority Queue & Export (CSV / GeoJSON)
```

---

## 🚀 Key Features

1. **EXIF GPS & Timestamp Extraction (`utils/exif_extractor.py`)**:
   - Converts Degree/Minute/Second (DMS) tuples to Decimal Degrees ($DD = D + \frac{M}{60} + \frac{S}{3600}$).
   - Extracts camera hardware details (`Make`, `Model`) and capture timestamps.
   - Graceful fallback: If GPS metadata is absent, provides interactive number inputs with zero crashes.

2. **YOLOv8 Object Detection Pipeline (`utils/detector.py`)**:
   - Uses `@st.cache_resource` for zero-overhead model persistence across UI re-renders.
   - Loads custom fine-tuned weights (`weights/best.pt`) or automatically falls back to `yolov8n.pt`.
   - Real-time user tuning: Confidence Threshold and IoU NMS Threshold sliders.
   - Stylized bounding boxes with category-specific color branding and confidence pills.

3. **Spatial DBSCAN Hotspot Clustering (`utils/clustering.py`)**:
   - Converts coordinates into radians and applies `scikit-learn` DBSCAN with `metric='haversine'`.
   - Groups nearby sightings within an adjustable radius (default 50 meters).
   - Computes cluster centroids, dominant categories, and repeat report duplicate suppression rates.

4. **Interactive GIS Command Map**:
   - Leaflet/Folium integration via `streamlit-folium`.
   - Custom markers color-coded by category:
     - 🔵 **Blue**: Plastic Waste
     - 🟠 **Orange**: Organic Waste
     - 🟢 **Green**: Dry Leaves / Biomass
   - Pulsing/dashed 50m cluster boundary circles and badge markers.
   - Interactive popups with incident details and embedded base64 image thumbnails.

5. **Executive Analytics & Data Export**:
   - Top KPI cards: Total Reports, Active Hotspots, Duplicate Suppression Rate, Top Category.
   - One-click export to **CSV** and standard **GeoJSON FeatureCollection** for GIS systems.

---

## 📂 Repository Structure

```text
terrasight-ai/
├── .streamlit/
│   └── config.toml           # Theme, max upload size (25MB), headless server config
├── weights/
│   ├── .gitkeep              # Weights directory placeholder
│   └── best.pt               # Trained YOLOv8 model weights
├── utils/
│   ├── __init__.py           # Package exports
│   ├── exif_extractor.py     # DMS-to-DD conversion & EXIF parsing
│   ├── detector.py           # YOLOv8 inference wrapper with @st.cache_resource
│   └── clustering.py         # Haversine DBSCAN logic & centroid calculation
├── app.py                    # Main Streamlit dashboard
├── packages.txt              # Debian system dependencies for Streamlit Cloud (libgl1, etc.)
├── requirements.txt          # Pinned Python dependencies
├── Dockerfile                # Multi-stage production container
├── docker-compose.yml        # Docker Compose configuration
└── README.md                 # Documentation and deployment guide
```

---

## 💻 Local Quick Start

### 1. Clone & Navigate
```bash
git clone https://github.com/<your-username>/terrasight-ai.git
cd terrasight-ai
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Streamlit Community Cloud 1-Click Deployment

1. Push this repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/).
3. Click **"New app"**.
4. Select your repository, branch (`main`), and set the main file path to:
   ```text
   app.py
   ```
5. Click **"Deploy!"**.
   - Streamlit Cloud will automatically detect `packages.txt` to install `libgl1` and `requirements.txt` to install the Python stack.

---

## 🐳 Docker Deployment

### Run with Docker:
```bash
docker build -t terrasight-ai .
docker run -p 8501:8501 terrasight-ai
```

### Run with Docker Compose:
```bash
docker compose up -d
```
Access the application at `http://localhost:8501`.
