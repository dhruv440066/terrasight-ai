# ==============================================================================
# Multi-Stage Production Dockerfile for TerraSight AI
# Builds the React frontend and packages the FastAPI + YOLOv8 inference server
# ==============================================================================

# Stage 1: Build React Frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Stage 2: Production Python Backend & Inference Engine
FROM python:3.11-slim AS production

WORKDIR /app

# System dependencies for OpenCV, PyTorch, and geospatial utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r ./backend/requirements.txt

# Copy backend application source
COPY backend/ ./backend/
COPY weights/ ./weights/

# Copy built frontend assets from Stage 1 into frontend/dist
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Ensure runtime directories exist
RUN mkdir -p /app/backend/uploads /app/backend/models /app/backend/generated_reports

# Set environment defaults
ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    DEMO_MODE=false \
    MODEL_PATH=models/best.pt

EXPOSE 8000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start Uvicorn ASGI server
WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
