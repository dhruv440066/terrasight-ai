# CLEANPATH AI (TerraSight AI)
### AI-Powered Civic Waste Detection & Municipal Dispatch Platform

> **Tagline:** *Detect. Locate. Cluster. Prioritize. Dispatch.*

CleanPath AI (also referred to as **TerraSight AI**) is a complete, 100% locally runnable civic-tech platform that empowers citizens to report roadside and civic waste using photographs while providing municipal sanitation officers with an AI-driven GIS Command Center to detect, geolocate, deduplicate via 50-meter Haversine DBSCAN clustering, prioritize by mathematical severity, and dispatch sanitation crews.

---

## 1. Project Overview

Urban sanitation departments frequently suffer from fragmented citizen complaints, missing location data, duplicate reports of the same waste pile, and subjective severity assessment. **CleanPath AI** solves this end-to-end without relying on paid cloud APIs or proprietary mapping services:

1. **Automated Waste Detection & Classification:** Uses **YOLOv8n** trained on the **TACO (Trash Annotations in Context)** dataset (with a deterministic **OpenCV Analytical Fallback** engine when GPU/YOLO weights are unavailable) to classify litter into three functional municipal categories:
   - `Plastic Waste` (Hazard Multiplier: `1.0`)
   - `Organic Waste` (Hazard Multiplier: `1.2`)
   - `Dry Leaves / Biomass` (Hazard Multiplier: `0.7`)
2. **EXIF GPS Extraction & Manual Leaflet Fallback:** Extracts `GPSLatitude` and `GPSLongitude` DMS coordinates from photo EXIF metadata and converts them into decimal degrees. When EXIF GPS is stripped or unavailable, citizens can drag a marker on an interactive **Leaflet / OpenStreetMap** pin picker.
3. **Mathematical Severity Engine:** Calculates a normalized `[0, 100]` Severity Index from bounding-box area ratios weighted by functional category hazard multipliers, assigning `High (>60%)`, `Medium (25%–60%)`, or `Low (<25%)` priority tiers.
4. **50-Meter Haversine DBSCAN Hotspot Clustering:** Aggregates citizen reports within a 50-meter great-circle radius (`eps = 50 / 6371000` radians, `min_samples = 1`) into unified municipal **Hotspots**—suppressing duplicate dispatch orders while preserving every citizen report for auditability.
5. **Municipal Officer GIS Command Center:** Interactive Leaflet GIS map displaying color-coded hotspots (`🔴 High`, `🟡 Medium`, `🟢 Low`), 50-meter dashed cluster boundaries, a sorted Priority Queue, crew dispatch workflow, and live analytics.

---

## 2. Problem Statement

Municipal waste response teams face four core operational bottlenecks:
- **Unstructured & Unverified Reports:** Citizens submit text complaints without objective measurement of waste volume or hazard type.
- **Missing Geolocation:** Compressed or camera-captured photos often lack GPS tags, leaving sanitation crews unable to locate incidents.
- **Dispatch Duplication:** Ten citizens photographing the same overflowing street corner generate ten separate work orders, wasting fuel and crew capacity.
- **Lack of Priority Triage:** Without automated severity scoring, minor dry leaf litter and hazardous putrescible organic/plastic dumps sit in the same flat queue.

---

## 3. Proposed Solution

CleanPath AI introduces a unified **Citizen-to-Command-Center Pipeline**:
- Citizens upload or capture a waste photo → EXIF GPS is extracted (or pinned manually via Leaflet) → AI detects waste bounding boxes and computes a hazard-weighted Severity Score → Incident is stored in SQLite (`cleanpath.db`).
- Every new incident triggers **50m Haversine DBSCAN** clustering → Nearby complaints merge into a single **Hotspot** → Municipal officers view hotspots in a prioritized queue, inspect constituent citizen reports, and dispatch a single sanitation crew per hotspot.

---

## 4. Features

### Citizen Portal
- Role-based registration & JWT login, plus one-click **Continue as Guest** mode.
- Drag-and-drop image upload (`.jpg`, `.jpeg`, `.png`) and browser **Live Camera Capture**.
- Automatic EXIF GPS extraction with DMS-to-decimal conversion.
- Interactive **Leaflet Map Pin Picker** fallback when EXIF metadata is absent.
- Non-destructive SVG bounding-box overlay showing category name and confidence (e.g., `Plastic Waste 91%`).
- Visual Severity Score progress bar and color-coded Severity Tier (`High`, `Medium`, `Low`).
- Unique Incident ID generation (e.g., `CP-1001`) and 5-stage visual status timeline (`Submitted → Verified → Dispatched → In Progress → Resolved`).

### Municipal Officer Portal
- RBAC-protected Officer Command Center (`/officer/map` and `/officer/analytics`).
- Real-time **System Status Bar** (`AI Engine`, `Database`, `GIS Engine`, `Clustering Engine`).
- **GIS Command Map** with `🔴 High`, `🟡 Medium`, `🟢 Low` markers and **50-meter dashed radius circles**.
- **DBSCAN Deduplication Pipeline Strip** demonstrating `15 citizen reports → 10 unique hotspots → 5 dispatch operations` (`Duplicate reports suppressed: 5`).
- **Priority Queue** sorted by Severity Tier (`High → Medium → Low`) and maximum severity score descending.
- **Auditability (`View Reports`):** Inspect all individual citizen reports belonging to a hotspot and update hotspot or individual incident statuses.
- **Sanitation Crew Dispatch Modal** and comprehensive **Analytics Dashboard**.

---

## 5. Architecture

```text
                    CLEANPATH AI
                         │
              ┌──────────┴──────────┐
              │                     │
          CITIZEN               OFFICER
              │                     │
        Image Upload          Command Center
              │                     │
        EXIF Extraction       GIS Dashboard
              │                     │
        Manual Pin Picker     Priority Queue
              │                     │
              └──────────┬──────────┘
                         │
                    FastAPI API
                         │
             ┌───────────┼───────────┐
             │           │           │
          YOLOv8       SQLite      DBSCAN
             │           │           │
             └───────────┼───────────┘
                         │
                    Hotspot Engine
                         │
                  Dispatch System
```

---

## 6. Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend** | Python 3.11+, FastAPI, Uvicorn, SQLAlchemy, SQLite (`cleanpath.db`), Pydantic, PyJWT / `python-jose`, Passlib / Bcrypt |
| **Computer Vision** | Ultralytics YOLOv8 (`yolov8n.pt` / `best.pt`), OpenCV (`cv2`), Pillow (`PIL`), NumPy |
| **Geospatial & ML** | Scikit-learn (`DBSCAN` with Haversine metric), Pandas, Pillow EXIF Parser |
| **Frontend** | React 18, TypeScript, Vite, Tailwind CSS, Leaflet, React-Leaflet, Lucide React, Axios |

---

## 7. Installation

### Prerequisites
- **Python:** `3.11+`
- **Node.js & npm:** `18+`

### Clone / Open Repository
```bash
cd TACO
```

---

## 8. Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

- **Backend API URL:** `http://localhost:8000`
- **Interactive Swagger Docs:** `http://localhost:8000/docs`
- **ReDoc Documentation:** `http://localhost:8000/redoc`
- **Health Check:** `http://localhost:8000/health`

---

## 9. Frontend Setup

Open a second terminal:

```bash
cd frontend
npm install
npm run dev -- --port 3000
```

- **Frontend Application URL:** `http://localhost:3000`

---

## 10. Dataset Setup & 11. Model Training

The repository includes `backend/train_taco.py`, which serves as the single source of truth for converting COCO annotations from the **TACO dataset** (`data/annotations.json`) into normalized YOLOv8 format (`train/`, `val/`, `test/` splits + `data.yaml`) and mapping the 60 TACO subclasses into the three CleanPath functional categories.

### Convert TACO Dataset Only
```bash
cd backend
python train_taco.py --dataset-path ../data --convert-only
```

### Convert & Train YOLOv8n (CPU/GPU Compatible)
```bash
cd backend
python train_taco.py --dataset-path ../data --epochs 10 --image-size 640 --batch-size 8 --device cpu
```

Trained weights are output to `backend/runs/detect/cleanpath_yolov8n/` and automatically copied to `backend/models/best.pt`.

---

## 12. Demo Credentials

When the backend starts, it automatically initializes `cleanpath.db`, creates the demo accounts below, and seeds the 15-report demonstration dataset:

| Role | Username | Password | Access Scope |
| :--- | :--- | :--- | :--- |
| **Citizen (Demo)** | `citizen` | `citizen123` | Citizen Portal (`/citizen`, `/citizen/report`, `/citizen/reports`) |
| **Municipal Officer (Demo)** | `officer` | `officer123` | Officer GIS Command Center (`/officer`, `/officer/map`, `/officer/analytics`) |
| **Guest Citizen** | *Click "Continue as Guest"* | *None* | Submit & track waste reports locally without registration |

---

## 13. API Documentation

| Method | Endpoint | Access | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Public | Health check & subsystem status (`AI`, `DB`, `GIS`, `Clustering`) |
| `POST` | `/auth/register` | Public | Register a new citizen or officer account |
| `POST` | `/auth/login` | Public | Authenticate and receive a signed JWT access token |
| `GET` | `/auth/me` | Authenticated | Retrieve current user profile |
| `POST` | `/incidents/analyze` | Citizen / Guest | Upload image, extract EXIF GPS, run YOLOv8/Fallback vision & severity scoring |
| `POST` | `/incidents` | Citizen / Guest | Submit waste incident into SQLite and trigger 50m DBSCAN clustering |
| `GET` | `/incidents/my` | Citizen / Guest | Retrieve complaints submitted by current citizen or guest session |
| `GET` | `/incidents/{id}` | Citizen / Guest | Retrieve detailed status and timeline for a specific incident |
| `GET` | `/officer/incidents` | Officer (RBAC) | List all civic waste incidents across the municipality |
| `GET` | `/officer/hotspots` | Officer (RBAC) | Retrieve 50m DBSCAN clustered hotspots sorted by Priority Queue |
| `GET` | `/officer/analytics` | Officer (RBAC) | Retrieve severity distribution, category breakdown, and deduplication stats |
| `POST` | `/officer/dispatch/{hotspot_id}` | Officer (RBAC) | Dispatch a sanitation crew to a hotspot and update constituent incidents |
| `PATCH` | `/officer/incidents/{id}/status` | Officer (RBAC) | Update individual incident status and synchronize hotspot status |
| `PATCH` | `/officer/hotspots/{id}/status` | Officer (RBAC) | Update unified hotspot status and propagate to member incidents |
| `POST` | `/demo/seed` | Officer (RBAC) | Seed 15 realistic incidents → 10 DBSCAN hotspots → 5 dispatches |

---

## 14. GIS Architecture & 15. DBSCAN Explanation

CleanPath AI uses **OpenStreetMap** tiles rendered via **Leaflet** and **React-Leaflet**, paired with **Scikit-learn's DBSCAN** (`Density-Based Spatial Clustering of Applications with Noise`):

```python
coords_rad = np.radians(coords)

dbscan = DBSCAN(
    eps=50.0 / 6371000.0,
    min_samples=1,
    metric="haversine"
)

labels = dbscan.fit_predict(coords_rad)
```

- **Why `eps = 50 / 6371000`?** Earth's mean radius is $6,371,000\text{ meters}$. Dividing $50\text{ meters}$ by $6,371,000\text{ meters}$ converts the 50m municipal cluster radius into radians for spherical Haversine distance computation.
- **Why `min_samples = 1`?** Even a single isolated waste report must form a valid 1-report Hotspot so no citizen complaint is ever dropped as "noise."
- **Dispatch Deduplication vs. Data Deletion:** When 3 citizens report the same roadside dump within 50 meters, DBSCAN groups them into a single Hotspot (`report_count = 3`, `duplicate_reports_suppressed = 2`). All 3 individual reports remain intact and inspectable under **View Reports**, while only 1 sanitation crew needs to be dispatched.

---

## 16. Severity Formula

For each detected bounding box $i$ with dimensions $(w_i, h_i)$ in an image of resolution $(W, H)$:

$$\text{BBox Area}_i = w_i \times h_i, \quad r_i = \frac{\text{BBox Area}_i}{W \times H}$$

$$\text{Area\_Ratio} = \frac{\sum_{i=1}^{N} \text{BBox Area}_i}{W \times H}$$

Each functional category carries a municipal hazard multiplier $m_i$:
- `Plastic Waste`: $1.0$
- `Organic Waste`: $1.2$
- `Dry Leaves / Biomass`: $0.7$

For multi-category detections, the area-weighted effective multiplier is:

$$\text{effective\_multiplier} = \frac{\sum_{i=1}^{N} \text{BBox Area}_i \cdot m_i}{\sum_{i=1}^{N} \text{BBox Area}_i}$$

$$\text{Severity Index} = \min\left(100.0, \text{Area\_Ratio} \times \text{effective\_multiplier} \times 100\right)$$

### Severity Tiers
- **High (`🔴 Red`):** `Severity > 60%`
- **Medium (`🟡 Yellow`):** `25% <= Severity <= 60%`
- **Low (`🟢 Green`):** `Severity < 25%`

---

## 17. Screenshots Section

1. **Role-Based Login Portal (`/login`):** Switch seamlessly between Citizen Portal, Guest Citizen Mode, and Secure Municipal Officer Login.
2. **Citizen AI Waste Analysis (`/citizen/report`):** Non-destructive SVG bounding boxes, hazard-weighted Severity Card, and automatic EXIF GPS or interactive Leaflet manual pin picker.
3. **Citizen Complaint Timeline (`/citizen/reports`):** Track incident `#CP-XXXX` from `Submitted` through `Dispatched` and `Resolved`.
4. **Officer GIS Command Center (`/officer/map`):** Live 50m dashed cluster circles, priority queue, duplicate suppression banner (`15 reports → 10 hotspots → 5 dispatches`), and crew dispatch modal.
5. **Municipal Analytics Dashboard (`/officer/analytics`):** Real-time severity breakdown, waste category distribution, and full incident audit table.

---

## 18. Testing

Run the automated backend unit and integration test suite covering Authentication, RBAC, EXIF DMS-to-decimal conversion, Severity calculation, 50m Haversine DBSCAN clustering, and REST APIs:

```bash
cd backend
python test_cleanpath.py -v
```

---

## 19. Limitations

- **2D Area Approximation:** Bounding-box area ratio estimates surface coverage in 2D image space rather than 3D volumetric mass (which depends on camera distance and perspective).
- **EXIF Stripping by Messaging Apps:** Images shared over WhatsApp or social media often have EXIF tags stripped, requiring the interactive Leaflet Manual Pin Picker fallback.
- **SQLite Concurrency:** SQLite is ideal for local and single-node municipal deployments; city-scale high-concurrency deployments should migrate `DATABASE_URL` to PostgreSQL + PostGIS.

---

## 20. Future Scope

- Depth-sensor / monocular depth estimation for 3D waste volume and tonnage estimation.
- Route optimization (Vehicle Routing Problem / VRP) across top-priority hotspots for sanitation trucks.
- Before-and-after cleanup verification using citizen notifications when an officer marks a hotspot `RESOLVED`.

---

## 21. Viva Questions & Answers

### 1. Why YOLOv8?
YOLOv8 (You Only Look Once v8 Nano) is a single-stage anchor-free object detection architecture that predicts bounding boxes and class probabilities in a single forward pass. It provides high-speed real-time inference on both CPUs and GPUs, making it ideal for local and edge civic deployments without cloud GPU costs.

### 2. Why TACO?
TACO (*Trash Annotations in Context*) is an open-source benchmark dataset of real-world litter across diverse outdoor environments (roadsides, pavements, vegetation). Mapping its 60 fine-grained classes into 3 actionable municipal categories (`Plastic Waste`, `Organic Waste`, `Dry Leaves / Biomass`) aligns computer vision output directly with sanitation crew equipment needs.

### 3. Why DBSCAN?
Unlike K-Means, **DBSCAN** does not require specifying the number of clusters ($k$) in advance. Since a city does not know ahead of time how many distinct waste hotspots will emerge on a given day, DBSCAN naturally discovers arbitrary-shaped spatial clusters defined by a physical distance threshold (`50 meters`).

### 4. Why Haversine?
Latitude and longitude are spherical coordinates on Earth's curved surface, where $1^\circ$ of longitude varies with latitude. Standard Euclidean distance ($\sqrt{\Delta\text{lat}^2 + \Delta\text{lon}^2}$) distorts real-world meter distances. The **Haversine formula** accurately computes great-circle distances in meters over Earth's radius ($R = 6,371,000\text{ m}$).

### 5. Why EXIF?
Modern smartphones embed Exchangeable Image File Format (**EXIF**) metadata containing `GPSLatitude`, `GPSLatitudeRef`, `GPSLongitude`, and `GPSLongitudeRef` in Degrees/Minutes/Seconds (DMS). Extracting EXIF allows zero-click automatic localization of waste incidents at the exact spot the photograph was taken.

### 6. Why Manual Fallback?
Browser security policies, desktop uploads, screenshots, and compressed images frequently strip EXIF GPS tags. Providing an interactive **Leaflet Map Pin Picker** ensures citizens can still accurately pinpoint and submit waste complaints even when metadata is absent.
