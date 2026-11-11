"""
Vercel Serverless Entrypoint for CleanPath AI (TerraSight AI).
Self-contained FastAPI ASGI entrypoint compatible with Vercel Services (`root: "api"`).
"""

import logging
import os
import sys
from pathlib import Path

# Mark runtime as Vercel Serverless so storage/DB paths use /tmp and stateless URIs
os.environ["VERCEL"] = "1"

# Resolve imports from this self-contained service directory first (fallback to ../backend if local)
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

backend_dir = current_dir.parent / "backend"
if backend_dir.exists() and str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

from main import app, ensure_app_bootstrapped  # noqa: E402

try:
    ensure_app_bootstrapped(run_demo_check=True)
except Exception as exc:
    logging.getLogger("cleanpath.vercel").warning("Serverless DB bootstrap note: %s", exc)

__all__ = ["app"]
