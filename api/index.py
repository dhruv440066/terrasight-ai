"""
Vercel Serverless Entrypoint for CleanPath AI (TerraSight AI)
Mounts the FastAPI application for execution on Vercel's serverless edge.
"""

import os
import sys
from pathlib import Path

# Mark as running in Vercel serverless environment
os.environ["VERCEL"] = "1"

# Add backend directory to sys.path so modules (auth, database, etc.) resolve cleanly
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Import main FastAPI application from backend
from main import app  # noqa: E402
from database import init_db, SessionLocal  # noqa: E402
from auth import seed_demo_users  # noqa: E402
from seed_demo import ensure_default_users  # noqa: E402

# Ensure database tables and demo accounts are initialized immediately
# in case the serverless execution environment bypasses ASGI lifespan events
try:
    init_db()
    db = SessionLocal()
    try:
        seed_demo_users(db)
        ensure_default_users(db)
    finally:
        db.close()
except Exception as exc:
    import logging
    logging.getLogger("cleanpath.vercel").warning("Serverless DB bootstrap note: %s", exc)

# Export app for Vercel Python runtime
__all__ = ["app"]
