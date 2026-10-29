# TerraSight AI — Render Production Deployment Guide

This guide details the complete production deployment of **TerraSight AI** to **Render** using GitHub.

---

## 1. Architecture Overview on Render

```text
                                 ┌──────────────────────────────────────────────┐
                                 │            User Web Browser                  │
                                 └──────────────┬───────────────────────────────┘
                                                │
                      ┌─────────────────────────┴────────────────────────┐
                      │                                                  │
                      ▼                                                  ▼
     ┌─────────────────────────────────┐                ┌─────────────────────────────────┐
     │      Frontend Static Site       │  VITE_API_URL  │       Backend Web Service       │
     │      (React + Vite + Leaflet)   ├───────────────►│    (FastAPI + YOLOv8 + DBSCAN)  │
     │   terrasight-frontend.onrender  │                │     terrasight-api.onrender     │
     └─────────────────────────────────┘                └────────────────┬────────────────┘
                                                                         │
                                                                         ▼
                                                        ┌─────────────────────────────────┐
                                                        │   Managed PostgreSQL Database   │
                                                        │          (terrasight-db)        │
                                                        └─────────────────────────────────┘
```

1. **Frontend (`terrasight-frontend`)**: Render Static Site built from `frontend/` using Vite. Serves the citizen reporting portal and officer GIS command center with SPA rewrite rules (`/*` -> `/index.html`).
2. **Backend (`terrasight-api`)**: Render Python Web Service executing FastAPI on port `$PORT`, hosting the YOLOv8 vision engine, EXIF GPS extractor, and DBSCAN clustering service.
3. **Database (`terrasight-db`)**: Render Managed PostgreSQL database linked automatically via `DATABASE_URL`.
4. **All-in-One Alternative**: The backend also natively serves the built React SPA from `frontend/dist` at root `/` when deployed as a single Web Service.

---

## 2. Step 1: Push Code to GitHub

Your target repository is: **`https://github.com/dhruv440066/terrasight-ai`**

Run the following commands in PowerShell from `C:\Users\LENOVO\TACO`:

```powershell
# 1. Stage all deployment configurations and source code
git add .

# 2. Commit the production deployment files
git commit -m "Configure TerraSight AI for Render production deployment"

# 3. Ensure branch is named main
git branch -M main

# 4. Point origin to your GitHub repository
git remote set-url origin https://github.com/dhruv440066/terrasight-ai.git

# 5. Push code to GitHub
git push -u origin main
```

*(If Git prompts for authentication, authenticate using your GitHub Personal Access Token or GitHub Desktop/CLI).*

---

## 3. Step 2: Deploy to Render via Blueprint (`render.yaml`)

The repository includes a ready-to-use [`render.yaml`](./render.yaml) Blueprint that sets up the database, backend, and frontend automatically.

1. Log in to [Render Dashboard](https://dashboard.render.com).
2. Click **Blueprints** in the top navigation bar.
3. Click **"New Blueprint Instance"**.
4. Connect your GitHub repository: **`dhruv440066/terrasight-ai`**.
5. Render will detect `render.yaml` and display the resources to create:
   - `terrasight-db` (PostgreSQL Database — Free)
   - `terrasight-api` (Web Service — Python 3.11 — Free)
   - `terrasight-frontend` (Static Site — Node/Vite — Free)
6. Click **"Apply"**.

Render will provision the database, build the FastAPI backend, and compile the React static site.

---

## 4. Alternative: Manual Service Setup on Render

If you prefer to configure the services manually through the Render UI instead of Blueprints:

### A. Create the PostgreSQL Database
1. Go to **Dashboard** -> **New** -> **PostgreSQL**.
2. **Name**: `terrasight-db`
3. **Database**: `terrasight`
4. **User**: `terrasight_user`
5. **Plan**: `Free`
6. Click **Create Database**.
7. Once created, copy the **Internal Database URL** (e.g. `postgres://...`).

### B. Create the Backend Web Service
1. Go to **Dashboard** -> **New** -> **Web Service**.
2. Connect `dhruv440066/terrasight-ai`.
3. **Name**: `terrasight-api`
4. **Language**: `Python 3`
5. **Region**: Same as database (e.g. `Oregon`)
6. **Branch**: `main`
7. **Root Directory**: Leave blank (repo root)
8. **Build Command**:
   ```bash
   pip install --upgrade pip && pip install -r backend/requirements.txt
   ```
9. **Start Command**:
   ```bash
   cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
10. **Plan**: `Free`
11. In **Environment Variables**, add:
    | Variable Name | Value | Description |
    | :--- | :--- | :--- |
    | `PYTHON_VERSION` | `3.11.9` | Ensures Python 3.11 runtime |
    | `DATABASE_URL` | Paste Internal Database URL from Step A | PostgreSQL connection string |
    | `JWT_SECRET_KEY` | *(Click Generate or enter secure random string)* | Auth signing key |
    | `DEMO_MODE` | `false` | Production mode (real citizen reports only) |
    | `MODEL_PATH` | `models/best.pt` | Path to trained YOLOv8 weights |
    | `CORS_ORIGINS` | `*` (or your frontend URL) | Allowed frontend domains |
    | `PORT` | `10000` (or leave default) | Render sets `$PORT` automatically |
12. Under **Advanced**, set **Health Check Path** to `/health`.
13. Click **Create Web Service**. Note your backend URL (e.g., `https://terrasight-api.onrender.com`).

### C. Create the Frontend Static Site
1. Go to **Dashboard** -> **New** -> **Static Site**.
2. Connect `dhruv440066/terrasight-ai`.
3. **Name**: `terrasight-frontend`
4. **Branch**: `main`
5. **Root Directory**: `frontend`
6. **Build Command**: `npm install && npm run build`
7. **Publish Directory**: `dist`
8. In **Environment Variables**, add:
   | Variable Name | Value |
   | :--- | :--- |
   | `VITE_API_URL` | `https://terrasight-api.onrender.com` *(your backend URL from Step B)* |
9. Under **Redirects/Rewrites**, add:
   - **Type**: `Rewrite`
   - **Source**: `/*`
   - **Destination**: `/index.html`
   *(This ensures React Router routes like `/citizen` and `/officer` reload cleanly).*
10. Click **Create Static Site**.

---

## 5. Production Environment Variables Reference

| Variable | Target Service | Default / Example | Purpose |
| :--- | :--- | :--- | :--- |
| `DATABASE_URL` | Backend | `sqlite:///./cleanpath.db` (local) / `postgresql://...` (prod) | Relational database connection string. Automatically normalizes `postgres://` to `postgresql://`. |
| `JWT_SECRET_KEY` | Backend | Random 64-char string | Cryptographic key for signing citizen & officer JWT access tokens. |
| `JWT_ALGORITHM` | Backend | `HS256` | JWT signature algorithm. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Backend | `1440` (24 hours) | Token validity lifetime. |
| `MODEL_PATH` | Backend | `models/best.pt` | Path to fine-tuned 3-class YOLOv8 weights. |
| `FALLBACK_MODEL_PATH` | Backend | `yolov8n.pt` | Fallback COCO weights if custom model missing. |
| `CONFIDENCE_THRESHOLD` | Backend | `0.20` | Detection confidence filter threshold. |
| `HOTSPOT_RADIUS_METERS` | Backend | `50.0` | Epsilon distance for Haversine DBSCAN clustering. |
| `CORS_ORIGINS` | Backend | Comma-separated URLs or `*` | Allowed CORS origins. Regex automatically permits all `*.onrender.com` origins. |
| `DEMO_MODE` | Backend | `false` | `false` = Production mode (no synthetic mock reports); `true` = Seed 15 demo incidents on startup. |
| `VITE_API_URL` | Frontend | `https://terrasight-api.onrender.com` | Base URL used by Axios in the React frontend to reach FastAPI. |

---

## 6. Pre-Flight Verification & Health Check

Once your services are deployed, test them using these endpoints:

1. **Backend Health Check**:
   ```bash
   curl -i https://<your-backend>.onrender.com/health
   ```
   Expected response:
   ```json
   {
     "status": "ok",
     "service": "CleanPath AI",
     "subsystems": {
       "ai_engine": "Online",
       "inference_mode": "YOLOv8",
       "weights_loaded": "models/best.pt",
       "database": "Online",
       "gis_engine": "Online",
       "clustering_engine": "Online"
     }
   }
   ```

2. **Swagger OpenAPI Documentation**:
   Navigate to `https://<your-backend>.onrender.com/docs` in your browser to inspect and test all endpoints interactively.

3. **Frontend Application**:
   Navigate to `https://<your-frontend>.onrender.com`:
   - Open `/citizen`: Upload an image, view EXIF GPS coordinate parsing (or pick a location manually), observe detected bounding boxes & severity score, and submit a complaint.
   - Open `/officer`: View live database statistics, 50-meter DBSCAN hotspot circles on the Leaflet GIS map, and dispatch sanitation crews.


---

## 8. Vercel Deployment Guide (100% Free, Global Edge CDN, Zero Cold Starts)

Deploying **TerraSight AI** to **Vercel** provides instant global CDN speeds and zero-latency loading for the React frontend, along with an optional built-in Serverless Python API.

### Why Vercel?
1. **Blazing Fast**: The Vite/React frontend builds in under 30 seconds and is served directly from Vercel's global edge network.
2. **Zero Frontend Cold Starts**: Unlike Render free-tier instances that sleep after 15 minutes of inactivity, Vercel static pages load immediately.
3. **No File Size Or Build Failures**: Vercel routes are pre-configured via `vercel.json` with SPA routing rewrites.

### Deployment Architecture Options on Vercel:

#### Option A: Full-Stack on Vercel (Frontend + Serverless API)
- Vercel builds the React frontend from `frontend/` and hosts it globally.
- Vercel automatically deploys `api/index.py` as a Python Serverless Function using `api/requirements.txt` (FastAPI + OpenCV + Scikit-Learn + SQLAlchemy without giant 1.5GB PyTorch).
- The vision engine runs the lightweight OpenCV Analytical Engine (<100ms inference, zero OOMs, 3-category waste detection, EXIF GPS, and 50m DBSCAN clustering).
- Zero external servers needed!

#### Option B: Hybrid (Vercel Frontend + Render/Railway/HF Backend)
- Vercel hosts the React frontend (`https://terrasight-ai.vercel.app`).
- Backend runs on Render (`https://terrasight-ai.onrender.com`) or Hugging Face Spaces (16GB RAM free) with full PyTorch YOLOv8 models.
- Set environment variable in Vercel: `VITE_API_URL=https://terrasight-ai.onrender.com`.

---

### Step-by-Step Instructions to Deploy on Vercel

1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Configure Vercel zero-config deployment"
   git push origin main
   ```

2. **Import Repository in Vercel**:
   - Go to [vercel.com](https://vercel.com) and log in with your GitHub account.
   - Click **"Add New..."** -> **"Project"**.
   - Select **`dhruv440066/terrasight-ai`** from your repository list and click **"Import"**.

3. **Verify Build Settings**:
   - Vercel will detect `vercel.json` automatically:
     - **Framework Preset**: Vite
     - **Build Command**: `cd frontend && npm install && npm run build`
     - **Output Directory**: `frontend/dist`
   - *(Optional)* Under **Environment Variables**, you can add:
     - `VITE_API_URL`: Your backend URL (e.g. `https://terrasight-ai.onrender.com`), or leave blank to use the built-in Vercel Serverless API.

4. **Click "Deploy"**:
   - Within 60 seconds, your site will be live at `https://terrasight-ai.vercel.app` (or your custom assigned Vercel URL)!

5. **Log In and Test**:
   - **Citizen Login**: `citizen` / `citizen123`
   - **Officer Login**: `officer` / `officer123`
   - Use the **"Server Config"** button in the navigation bar anytime to verify connection status or point to any custom API.
