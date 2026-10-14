#!/usr/bin/env bash
# ==============================================================================
# TerraSight AI — Unified Render Build Script
# Builds the React frontend and installs Python backend requirements
# so the entire platform runs on a single unified Render Web Service!
# ==============================================================================
set -o errexit

echo "===> [1/3] Upgrading pip and installing Python backend dependencies..."
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt

echo "===> [2/3] Building React Frontend SPA with Vite..."
cd frontend
npm install
npm run build
cd ..

echo "===> [3/3] Build completed successfully! Ready for launch."
