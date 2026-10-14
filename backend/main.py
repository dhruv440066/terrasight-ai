"""
CleanPath AI (TerraSight AI) — FastAPI Backend Server
AI-Powered Civic Waste Detection, EXIF Geolocation, 50m Haversine DBSCAN
Clustering, and Municipal Sanitation Dispatch Platform.
"""

import io
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import text
from sqlalchemy.orm import Session

from auth import (
    create_access_token,
    get_current_user,
    get_optional_current_user,
    hash_password,
    require_role,
    seed_demo_users,
    verify_password,
)
from database import (
    AuditLog,
    Base,
    Dispatch,
    Hotspot,
    HotspotReport,
    Incident,
    ReportDetection,
    SessionLocal,
    User,
    engine,
    get_db,
    init_db,
    utc_now,
)
from exif_parser import extract_gps_from_image
from geospatial import (
    DEFAULT_CLUSTER_RADIUS_METERS,
    cluster_incidents_data,
    haversine_distance_meters,
    parse_categories_json,
    recompute_hotspots,
    reverse_geocode_coordinates,
)
from seed_demo import ensure_default_users, purge_synthetic_demo_incidents
from train_taco import (
    CATEGORY_MULTIPLIERS,
    FUNCTIONAL_CATEGORIES,
    map_taco_class_to_category,
)
from vision_engine import (
    analyze_waste_image,
    calculate_severity,
    determine_severity_tier,
    get_vision_status,
    map_category_to_db_enum,
    map_db_enum_to_category,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cleanpath.api")

BACKEND_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BACKEND_DIR / os.getenv("UPLOAD_DIR", "uploads")
MODELS_DIR = BACKEND_DIR / "models"
REPORTS_DIR = BACKEND_DIR / "generated_reports"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png"}
MAX_UPLOAD_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB

VALID_INCIDENT_STATUSES = {
    "SUBMITTED",
    "PENDING",
    "VERIFIED",
    "ACKNOWLEDGED",
    "DISPATCHED",
    "IN_PROGRESS",
    "RESOLVED",
    "REJECTED",
}

VALID_HOTSPOT_STATUSES = {
    "PENDING",
    "ACKNOWLEDGED",
    "DISPATCHED",
    "IN_PROGRESS",
    "RESOLVED",
}


# ==============================================================================
# PYDANTIC SCHEMAS
# ==============================================================================


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    name: Optional[str] = Field(default=None, max_length=120)
    email: Optional[str] = Field(default=None, max_length=255)
    phone: Optional[str] = Field(default=None, max_length=32)
    password: str = Field(..., min_length=6, max_length=128)
    role: str = Field(default="citizen")

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in ("citizen", "officer", "admin"):
            raise ValueError("Role must be 'citizen' or 'officer'")
        return "officer" if normalized == "admin" else normalized


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, description="Username or email address")
    password: str = Field(..., min_length=1)


class UserResponse(BaseModel):
    id: int
    name: Optional[str] = None
    username: str
    email: str
    role: str
    phone: Optional[str] = None
    created_at: datetime


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class IncidentCreateRequest(BaseModel):
    image_path: str = Field(..., min_length=1)
    original_filename: Optional[str] = Field(default=None)
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    location_source: str = Field(default="MANUAL")
    address: Optional[str] = Field(default=None, max_length=512)
    description: Optional[str] = Field(default=None, max_length=2000)
    detected_categories: Union[List[str], List[Dict[str, Any]], str] = Field(
        default_factory=list
    )
    detections: Optional[List[Dict[str, Any]]] = Field(default=None)
    detection_count: int = Field(default=1, ge=0)
    severity_score: float = Field(default=0.0, ge=0.0, le=100.0)
    severity_tier: Optional[str] = Field(default=None)


class DispatchCreateRequest(BaseModel):
    hotspot_id: Optional[int] = Field(default=None)
    crew_name: str = Field(..., min_length=2, max_length=200)
    vehicle_number: Optional[str] = Field(default=None, max_length=64)
    notes: Optional[str] = Field(default="", max_length=1000)


class StatusUpdateRequest(BaseModel):
    status: str = Field(..., min_length=2, max_length=32)
    notes: Optional[str] = Field(default=None, max_length=1000)


class DemoSeedRequest(BaseModel):
    center_latitude: float = Field(
        default=float(os.getenv("DEMO_CENTER_LAT", "28.6139")),
        ge=-90.0,
        le=90.0,
    )
    center_longitude: float = Field(
        default=float(os.getenv("DEMO_CENTER_LON", "77.2090")),
        ge=-180.0,
        le=180.0,
    )
    clear_existing: bool = Field(default=True)


# ==============================================================================
# HELPER UTILITIES
# ==============================================================================


def serialize_user(user: User) -> Dict[str, Any]:
    """Serialize a SQLAlchemy User object to a dictionary."""
    return {
        "id": user.id,
        "name": getattr(user, "name", None) or user.username,
        "username": user.username,
        "email": user.email,
        "role": user.role,
        "phone": getattr(user, "phone", None),
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


def serialize_audit_log(log: AuditLog) -> Dict[str, Any]:
    """Serialize an AuditLog record."""
    return {
        "id": log.id,
        "report_id": log.report_id,
        "hotspot_id": log.hotspot_id,
        "actor_id": log.actor_id,
        "actor_name": (log.actor.name or log.actor.username) if log.actor else (log.actor_role or "System"),
        "actor_role": log.actor_role or "SYSTEM",
        "action": log.action,
        "old_status": log.old_status,
        "new_status": log.new_status,
        "notes": log.notes or "",
        "timestamp": log.timestamp.isoformat() if log.timestamp else None,
    }


def _parse_stored_detection_payload(raw_text: Optional[str]) -> Dict[str, Any]:
    """
    Extract category names, counts, and full bounding-box detections from the
    Incident.detected_categories column.
    """
    if not raw_text:
        return {"categories": [], "detections": [], "counts": {}}
    try:
        loaded = json.loads(raw_text)
        if isinstance(loaded, dict):
            counts_dict = loaded.get("counts") or {}
            cats = loaded.get("categories") or loaded.get("detected_categories") or list(counts_dict.keys()) or []
            dets = loaded.get("detections") or []
            return {
                "categories": [str(c) for c in cats],
                "detections": dets,
                "counts": counts_dict,
            }
        if isinstance(loaded, list):
            cats_list: List[str] = []
            dets_list: List[Dict[str, Any]] = []
            for item in loaded:
                if isinstance(item, str):
                    if item not in cats_list:
                        cats_list.append(item)
                elif isinstance(item, dict):
                    dets_list.append(item)
                    cname = item.get("class_name")
                    if cname and str(cname) not in cats_list:
                        cats_list.append(str(cname))
            return {"categories": cats_list, "detections": dets_list, "counts": {}}
    except Exception:
        pass
    return {
        "categories": parse_categories_json(raw_text),
        "detections": [],
        "counts": {},
    }


def serialize_incident(
    incident: Incident, include_dispatches: bool = False
) -> Dict[str, Any]:
    """Serialize an Incident ORM record into a clean JSON-ready dictionary."""
    parsed_payload = _parse_stored_detection_payload(incident.detected_categories)
    categories = parsed_payload["categories"]
    detections = list(parsed_payload["detections"])

    # Also merge normalized ReportDetection rows if detections JSON is empty
    if not detections and getattr(incident, "report_detections", None):
        for rd in incident.report_detections:
            cat_label = map_db_enum_to_category(rd.category)
            if cat_label not in categories:
                categories.append(cat_label)
            detections.append(
                {
                    "id": rd.id,
                    "class_name": cat_label,
                    "category": rd.category,
                    "confidence": round(float(rd.confidence), 3),
                    "bbox": {
                        "x1": int(rd.bounding_box_x1),
                        "y1": int(rd.bounding_box_y1),
                        "x2": int(rd.bounding_box_x2),
                        "y2": int(rd.bounding_box_y2),
                    },
                    "object_count": rd.object_count or 1,
                }
            )

    category_counts: Dict[str, int] = dict(parsed_payload.get("counts") or {})
    category_confs: Dict[str, List[float]] = {}
    all_confs: List[float] = []

    for det in detections:
        cname = det.get("class_name", "Plastic Waste")
        conf = float(det.get("confidence", 0.85))
        all_confs.append(conf)
        category_confs.setdefault(cname, []).append(conf)

    if not category_counts:
        for det in detections:
            cname = det.get("class_name", "Plastic Waste")
            category_counts[cname] = category_counts.get(cname, 0) + 1
        if not category_counts and categories:
            for c in categories:
                category_counts[c] = 1

    category_stats: Dict[str, Dict[str, Any]] = {}
    for cat, cnt in category_counts.items():
        confs = category_confs.get(cat, [0.85])
        category_stats[cat] = {
            "count": cnt,
            "avg_confidence": round(sum(confs) / len(confs), 3),
            "max_confidence": round(max(confs), 3),
        }

    highest_confidence = round(max(all_confs), 3) if all_confs else 0.88
    average_confidence = round(sum(all_confs) / len(all_confs), 3) if all_confs else 0.85

    image_url = incident.image_path
    if image_url and not image_url.startswith("/") and not image_url.startswith("http"):
        image_url = f"/uploads/{Path(image_url).name}"

    lat_val = round(float(incident.latitude), 6)
    lon_val = round(float(incident.longitude), 6)
    address_val = getattr(incident, "address", None)
    if not address_val:
        address_val = reverse_geocode_coordinates(lat_val, lon_val, allow_network=False)["address"]

    # Determine primary vs duplicate status inside hotspot
    is_primary = True
    distance_from_centroid = 0.0
    if getattr(incident, "hotspot_links", None):
        link = incident.hotspot_links[0]
        is_primary = bool(link.is_primary)
        distance_from_centroid = round(float(link.distance_from_centroid or 0.0), 2)
    elif incident.hotspot and incident.hotspot.incidents:
        earliest_id = min(i.id for i in incident.hotspot.incidents)
        is_primary = incident.id == earliest_id
        distance_from_centroid = round(
            haversine_distance_meters(
                lat_val, lon_val, incident.hotspot.latitude, incident.hotspot.longitude
            ),
            2,
        )

    hotspot_code = f"HS-{incident.hotspot_id:04d}" if incident.hotspot_id else None
    duplicate_status = "PRIMARY" if is_primary else "DUPLICATE / CLUSTERED"

    citizen_name = "Guest Citizen"
    citizen_email = None
    if incident.citizen:
        citizen_name = getattr(incident.citizen, "name", None) or incident.citizen.username
        citizen_email = incident.citizen.email

    data: Dict[str, Any] = {
        "id": incident.id,
        "incident_uuid": incident.incident_uuid,
        "public_report_id": incident.incident_uuid,
        "citizen_id": incident.citizen_id,
        "citizen_username": incident.citizen.username if incident.citizen else "Guest Citizen",
        "citizen_name": citizen_name,
        "citizen_email": citizen_email,
        "image_path": incident.image_path,
        "image_url": image_url,
        "thumbnail_path": getattr(incident, "thumbnail_path", None) or image_url,
        "original_filename": getattr(incident, "original_filename", None) or Path(image_url or "waste.jpg").name,
        "latitude": lat_val,
        "longitude": lon_val,
        "location_source": incident.location_source,
        "address": address_val,
        "description": getattr(incident, "description", None) or "",
        "detected_categories": categories,
        "category_counts": category_counts,
        "category_stats": category_stats,
        "detections": detections,
        "detection_count": incident.detection_count,
        "highest_confidence": highest_confidence,
        "average_confidence": average_confidence,
        "severity_score": round(float(incident.severity_score), 2),
        "severity_tier": incident.severity_tier,
        "severity_level": incident.severity_tier.upper() if incident.severity_tier else "LOW",
        "status": incident.status,
        "hotspot_id": incident.hotspot_id,
        "hotspot_code": hotspot_code,
        "is_primary": is_primary,
        "duplicate_status": duplicate_status,
        "distance_from_centroid": distance_from_centroid,
        "created_at": incident.created_at.isoformat() if incident.created_at else None,
        "updated_at": incident.updated_at.isoformat() if incident.updated_at else None,
    }

    if include_dispatches:
        audit_entries: List[AuditLog] = list(getattr(incident, "audit_logs", []) or [])
        if incident.hotspot and getattr(incident.hotspot, "audit_logs", None):
            seen_ids = {a.id for a in audit_entries}
            for h_log in incident.hotspot.audit_logs:
                if h_log.id not in seen_ids:
                    audit_entries.append(h_log)
                    seen_ids.add(h_log.id)
        audit_entries.sort(key=lambda a: a.timestamp or datetime.min.replace(tzinfo=timezone.utc))
        data["audit_logs"] = [serialize_audit_log(a) for a in audit_entries]

        if incident.hotspot:
            data["hotspot"] = {
                "id": incident.hotspot.id,
                "hotspot_code": f"HS-{incident.hotspot.id:04d}",
                "latitude": incident.hotspot.latitude,
                "longitude": incident.hotspot.longitude,
                "report_count": incident.hotspot.report_count,
                "duplicate_count": getattr(incident.hotspot, "duplicate_count", max(0, incident.hotspot.report_count - 1)),
                "maximum_severity": incident.hotspot.maximum_severity,
                "severity_tier": incident.hotspot.severity_tier,
                "radius_meters": incident.hotspot.radius_meters,
                "status": incident.hotspot.status,
            }
            data["dispatches"] = [
                serialize_dispatch(d) for d in incident.hotspot.dispatches
            ]
        else:
            data["dispatches"] = []

    return data


def serialize_dispatch(dispatch: Dispatch) -> Dict[str, Any]:
    """Serialize a Dispatch ORM record."""
    return {
        "id": dispatch.id,
        "hotspot_id": dispatch.hotspot_id,
        "hotspot_code": f"HS-{dispatch.hotspot_id:04d}" if dispatch.hotspot_id else None,
        "officer_id": dispatch.officer_id,
        "officer_username": dispatch.officer.username if dispatch.officer else "Officer",
        "crew_name": dispatch.crew_name,
        "vehicle_number": getattr(dispatch, "vehicle_number", None),
        "status": dispatch.status,
        "dispatched_at": dispatch.dispatched_at.isoformat()
        if dispatch.dispatched_at
        else None,
        "started_at": dispatch.started_at.isoformat()
        if getattr(dispatch, "started_at", None)
        else None,
        "completed_at": dispatch.completed_at.isoformat()
        if dispatch.completed_at
        else None,
        "notes": dispatch.notes or "",
    }


def serialize_hotspot(hotspot: Hotspot) -> Dict[str, Any]:
    """Serialize a Hotspot ORM record including its constituent citizen reports and dispatches."""
    member_incidents = sorted(
        hotspot.incidents, key=lambda inc: (inc.created_at or datetime.min, inc.id)
    )
    aggregated_categories: List[str] = []
    for inc in member_incidents:
        for cat in _parse_stored_detection_payload(inc.detected_categories)["categories"]:
            if cat not in aggregated_categories:
                aggregated_categories.append(cat)

    dispatches_sorted = sorted(
        hotspot.dispatches,
        key=lambda d: d.dispatched_at or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    lat_val = round(float(hotspot.latitude), 6)
    lon_val = round(float(hotspot.longitude), 6)
    address_val = reverse_geocode_coordinates(lat_val, lon_val, allow_network=False)["address"]
    dup_count = max(0, hotspot.report_count - 1)
    radius_m = int(hotspot.radius_meters or 50)
    merge_explanation = (
        f"{hotspot.report_count} citizen report{'s' if hotspot.report_count != 1 else ''} "
        f"within {radius_m}m merged into 1 dispatch hotspot"
    )

    audit_entries = sorted(
        list(getattr(hotspot, "audit_logs", []) or []),
        key=lambda a: a.timestamp or datetime.min.replace(tzinfo=timezone.utc),
    )

    return {
        "id": hotspot.id,
        "hotspot_id": hotspot.id,
        "hotspot_code": f"HS-{hotspot.id:04d}",
        "latitude": lat_val,
        "longitude": lon_val,
        "address": address_val,
        "report_count": hotspot.report_count,
        "duplicate_count": dup_count,
        "duplicate_reports_suppressed": dup_count,
        "merge_explanation": merge_explanation,
        "maximum_severity": round(float(hotspot.maximum_severity), 2),
        "severity_tier": hotspot.severity_tier,
        "radius_meters": hotspot.radius_meters,
        "status": hotspot.status,
        "categories": aggregated_categories or ["Plastic Waste"],
        "created_at": hotspot.created_at.isoformat() if hotspot.created_at else None,
        "updated_at": hotspot.updated_at.isoformat() if hotspot.updated_at else None,
        "incidents": [serialize_incident(inc, include_dispatches=True) for inc in member_incidents],
        "dispatches": [serialize_dispatch(d) for d in dispatches_sorted],
        "audit_logs": [serialize_audit_log(a) for a in audit_entries],
    }


def validate_and_save_upload(upload_file: UploadFile) -> Tuple[Path, bytes]:
    """
    Validate file extension, MIME type, image integrity, and file size.
    Saves the file with a safe server-generated UUID filename inside UPLOAD_DIR.
    """
    if not upload_file or not upload_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file was provided.",
        )

    original_suffix = Path(upload_file.filename).suffix.lower()
    if original_suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{original_suffix}'. Allowed: .jpg, .jpeg, .png",
        )

    content_type = (upload_file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_MIME_TYPES and content_type != "application/octet-stream":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported MIME type '{content_type}'. Allowed: image/jpeg, image/png",
        )

    raw_bytes = upload_file.file.read()
    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(raw_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image exceeds the 15 MB maximum upload size limit.",
        )

    # Verify image can be decoded by Pillow
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img_check:
            img_check.verify()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid or uncorrupted image.",
        ) from exc

    safe_filename = f"waste_{uuid.uuid4().hex[:12]}{original_suffix}"
    saved_path = UPLOAD_DIR / safe_filename
    with open(saved_path, "wb") as out_f:
        out_f.write(raw_bytes)

    return saved_path, raw_bytes


def generate_incident_uuid(db: Session) -> str:
    """Generate a human-readable unique incident code (e.g., CP-2026-000123)."""
    year_str = datetime.now(timezone.utc).strftime("%Y")
    last_incident = db.query(Incident).order_by(Incident.id.desc()).first()
    next_num = 1 if last_incident is None else (last_incident.id + 1)
    candidate = f"CP-{year_str}-{next_num:06d}"
    while db.query(Incident).filter(Incident.incident_uuid == candidate).first() is not None:
        next_num += 1
        candidate = f"CP-{year_str}-{next_num:06d}"
    return candidate


def _create_synthetic_demo_image(
    filename: str,
    category: str,
    severity_tier: str,
) -> str:
    """Generate a clean visual sample image in uploads/ for demo incidents."""
    target_path = UPLOAD_DIR / filename
    if target_path.exists():
        return f"/uploads/{filename}"

    width, height = 640, 420
    img = Image.new("RGB", (width, height), color=(228, 234, 240))
    draw = ImageDraw.Draw(img)

    # Road / ground texture backdrop
    draw.rectangle([0, int(height * 0.52), width, height], fill=(185, 192, 198))
    draw.line([0, int(height * 0.52), width, int(height * 0.52)], fill=(140, 150, 160), width=3)

    if category == "Plastic Waste":
        box_color = (30, 136, 229)
        secondary_color = (245, 248, 252)
    elif category == "Organic Waste":
        box_color = (230, 126, 34)
        secondary_color = (180, 95, 25)
    else:
        box_color = (109, 76, 65)
        secondary_color = (76, 145, 65)

    if severity_tier == "High":
        draw.rectangle([70, 90, 570, 385], fill=box_color, outline=secondary_color, width=5)
        draw.ellipse([130, 130, 360, 340], fill=secondary_color)
        draw.rectangle([340, 150, 530, 355], fill=(210, 65, 65))
    elif severity_tier == "Medium":
        draw.rectangle([140, 135, 500, 355], fill=box_color, outline=secondary_color, width=4)
        draw.ellipse([190, 175, 350, 320], fill=secondary_color)
    else:
        draw.rectangle([220, 185, 420, 325], fill=box_color, outline=secondary_color, width=3)

    img.save(target_path, format="JPEG", quality=88)
    return f"/uploads/{filename}"


def seed_demo_incidents_and_dispatches(
    db: Session,
    center_lat: float = 28.6139,
    center_lon: float = 77.2090,
    clear_existing: bool = True,
) -> Dict[str, Any]:
    """
    Seed 15 realistic civic waste incidents (5 High, 5 Medium, 5 Low) around
    the specified center coordinates such that 50-meter Haversine DBSCAN
    produces exactly 10 unique hotspots (5 duplicate reports suppressed)
    and 5 municipal dispatch operations.
    """
    if clear_existing:
        db.query(AuditLog).delete()
        db.query(ReportDetection).delete()
        db.query(HotspotReport).delete()
        db.query(Dispatch).delete()
        db.query(Incident).delete()
        db.query(Hotspot).delete()
        db.commit()

    citizen_user = db.query(User).filter(User.username == "citizen").first()
    officer_user = db.query(User).filter(User.username == "officer").first()
    citizen_id = citizen_user.id if citizen_user else None
    officer_id = officer_user.id if officer_user else 1
    year_str = datetime.now(timezone.utc).strftime("%Y")

    # 15 carefully designed incidents:
    # - 5 High severity incidents (grouped into 2 hotspots: 3 reports + 2 reports)
    # - 5 Medium severity incidents (grouped into 3 hotspots: 2 reports + 2 reports + 1 report)
    # - 5 Low severity incidents (5 distinct hotspots: 1 report each)
    # Total: 15 citizen reports -> 10 unique hotspots (5 duplicates suppressed)
    sample_specs = [
        # Hotspot Cluster 1 (High - 3 reports within ~18m)
        {
            "dlat": 0.00000,
            "dlon": 0.00000,
            "source": "EXIF",
            "categories": ["Plastic Waste", "Organic Waste"],
            "count": 6,
            "score": 84.5,
            "tier": "High",
            "status": "DISPATCHED",
            "bbox": {"x1": 65, "y1": 85, "x2": 575, "y2": 385},
        },
        {
            "dlat": 0.00009,
            "dlon": 0.00008,
            "source": "EXIF",
            "categories": ["Plastic Waste"],
            "count": 5,
            "score": 76.2,
            "tier": "High",
            "status": "DISPATCHED",
            "bbox": {"x1": 80, "y1": 95, "x2": 560, "y2": 375},
        },
        {
            "dlat": -0.00008,
            "dlon": 0.00006,
            "source": "MANUAL",
            "categories": ["Organic Waste"],
            "count": 4,
            "score": 68.0,
            "tier": "High",
            "status": "DISPATCHED",
            "bbox": {"x1": 95, "y1": 100, "x2": 545, "y2": 370},
        },
        # Hotspot Cluster 2 (High - 2 reports within ~20m)
        {
            "dlat": 0.00260,
            "dlon": 0.00300,
            "source": "EXIF",
            "categories": ["Organic Waste", "Plastic Waste"],
            "count": 5,
            "score": 78.4,
            "tier": "High",
            "status": "DISPATCHED",
            "bbox": {"x1": 75, "y1": 90, "x2": 565, "y2": 380},
        },
        {
            "dlat": 0.00269,
            "dlon": 0.00309,
            "source": "EXIF",
            "categories": ["Plastic Waste"],
            "count": 4,
            "score": 64.5,
            "tier": "High",
            "status": "DISPATCHED",
            "bbox": {"x1": 100, "y1": 105, "x2": 535, "y2": 365},
        },
        # Hotspot Cluster 3 (Medium - 2 reports within ~16m)
        {
            "dlat": -0.00290,
            "dlon": -0.00350,
            "source": "EXIF",
            "categories": ["Plastic Waste", "Dry Leaves / Biomass"],
            "count": 3,
            "score": 54.0,
            "tier": "Medium",
            "status": "DISPATCHED",
            "bbox": {"x1": 135, "y1": 130, "x2": 505, "y2": 355},
        },
        {
            "dlat": -0.00282,
            "dlon": -0.00342,
            "source": "MANUAL",
            "categories": ["Plastic Waste"],
            "count": 3,
            "score": 42.5,
            "tier": "Medium",
            "status": "DISPATCHED",
            "bbox": {"x1": 150, "y1": 140, "x2": 490, "y2": 345},
        },
        # Hotspot Cluster 4 (Medium - 2 reports within ~16m)
        {
            "dlat": 0.00430,
            "dlon": -0.00300,
            "source": "EXIF",
            "categories": ["Organic Waste"],
            "count": 3,
            "score": 48.2,
            "tier": "Medium",
            "status": "IN_PROGRESS",
            "bbox": {"x1": 145, "y1": 135, "x2": 495, "y2": 350},
        },
        {
            "dlat": 0.00438,
            "dlon": -0.00292,
            "source": "MANUAL",
            "categories": ["Organic Waste", "Plastic Waste"],
            "count": 2,
            "score": 36.0,
            "tier": "Medium",
            "status": "IN_PROGRESS",
            "bbox": {"x1": 165, "y1": 150, "x2": 475, "y2": 340},
        },
        # Hotspot Cluster 5 (Medium - 1 report)
        {
            "dlat": -0.00440,
            "dlon": 0.00450,
            "source": "EXIF",
            "categories": ["Dry Leaves / Biomass", "Plastic Waste"],
            "count": 2,
            "score": 31.5,
            "tier": "Medium",
            "status": "RESOLVED",
            "bbox": {"x1": 175, "y1": 155, "x2": 465, "y2": 335},
        },
        # Hotspot Cluster 6 (Low - 1 report)
        {
            "dlat": 0.00130,
            "dlon": -0.00750,
            "source": "EXIF",
            "categories": ["Dry Leaves / Biomass"],
            "count": 2,
            "score": 21.0,
            "tier": "Low",
            "status": "SUBMITTED",
            "bbox": {"x1": 215, "y1": 180, "x2": 425, "y2": 325},
        },
        # Hotspot Cluster 7 (Low - 1 report)
        {
            "dlat": 0.00660,
            "dlon": 0.00150,
            "source": "MANUAL",
            "categories": ["Plastic Waste"],
            "count": 1,
            "score": 18.4,
            "tier": "Low",
            "status": "SUBMITTED",
            "bbox": {"x1": 225, "y1": 185, "x2": 415, "y2": 320},
        },
        # Hotspot Cluster 8 (Low - 1 report)
        {
            "dlat": -0.00640,
            "dlon": -0.00050,
            "source": "EXIF",
            "categories": ["Dry Leaves / Biomass"],
            "count": 1,
            "score": 14.8,
            "tier": "Low",
            "status": "VERIFIED",
            "bbox": {"x1": 235, "y1": 195, "x2": 405, "y2": 315},
        },
        # Hotspot Cluster 9 (Low - 1 report)
        {
            "dlat": -0.00110,
            "dlon": 0.00780,
            "source": "EXIF",
            "categories": ["Plastic Waste"],
            "count": 1,
            "score": 11.2,
            "tier": "Low",
            "status": "SUBMITTED",
            "bbox": {"x1": 240, "y1": 200, "x2": 395, "y2": 310},
        },
        # Hotspot Cluster 10 (Low - 1 report)
        {
            "dlat": 0.00510,
            "dlon": 0.00650,
            "source": "MANUAL",
            "categories": ["Dry Leaves / Biomass"],
            "count": 1,
            "score": 8.5,
            "tier": "Low",
            "status": "SUBMITTED",
            "bbox": {"x1": 250, "y1": 205, "x2": 385, "y2": 305},
        },
    ]

    for idx, spec in enumerate(sample_specs, start=1):
        primary_cat = spec["categories"][0]
        tier = str(spec["tier"])
        img_url = _create_synthetic_demo_image(
            filename=f"demo_sample_{idx:02d}.jpg",
            category=primary_cat,
            severity_tier=tier,
        )
        detections_payload = [
            {
                "class_name": cat,
                "confidence": round(0.92 - (i * 0.04), 2),
                "bbox": spec["bbox"],
                "area_ratio": round(float(spec["score"]) / 100.0 / len(spec["categories"]), 4),
            }
            for i, cat in enumerate(spec["categories"])
        ]
        stored_json = json.dumps(
            {
                "categories": spec["categories"],
                "detections": detections_payload,
            }
        )
        lat_val = round(center_lat + float(spec["dlat"]), 6)
        lon_val = round(center_lon + float(spec["dlon"]), 6)
        geo_info = reverse_geocode_coordinates(lat_val, lon_val, allow_network=False)
        public_id = f"CP-{year_str}-{idx:06d}"

        inc = Incident(
            incident_uuid=public_id,
            citizen_id=citizen_id,
            image_path=img_url,
            thumbnail_path=img_url,
            original_filename=f"demo_sample_{idx:02d}.jpg",
            latitude=lat_val,
            longitude=lon_val,
            location_source=str(spec["source"]),
            address=geo_info["address"],
            description=f"Municipal waste accumulation ({', '.join(spec['categories'])}) reported via {spec['source']}.",
            detected_categories=stored_json,
            detection_count=int(spec["count"]),
            severity_score=float(spec["score"]),
            severity_tier=tier,
            status=str(spec["status"]),
        )
        db.add(inc)
        db.flush()

        for det in detections_payload:
            bbox = det["bbox"]
            db.add(
                ReportDetection(
                    report_id=inc.id,
                    category=map_category_to_db_enum(det["class_name"]),
                    confidence=det["confidence"],
                    bounding_box_x1=bbox["x1"],
                    bounding_box_y1=bbox["y1"],
                    bounding_box_x2=bbox["x2"],
                    bounding_box_y2=bbox["y2"],
                    object_count=1,
                )
            )
        db.add(
            AuditLog(
                report_id=inc.id,
                actor_id=citizen_id,
                actor_role="CITIZEN",
                action="REPORT_CREATED",
                old_status=None,
                new_status=str(spec["status"]),
                notes=f"Demo report {public_id} created at {geo_info['address']}.",
            )
        )

    db.commit()

    # Run 50m Haversine DBSCAN to generate the 10 unified hotspots
    hotspots = recompute_hotspots(db)

    # Create 5 municipal dispatch operations on the top 5 priority hotspots
    crew_assignments = [
        (
            "North Zone Rapid Sanitation Unit A",
            "DISPATCHED",
            "High-density roadside plastic and organic accumulation; 3 duplicate citizen reports merged.",
        ),
        (
            "Central Sector Heavy Waste Team B",
            "DISPATCHED",
            "Commercial corridor overflow cluster; prioritized by severity queue.",
        ),
        (
            "East Ward Compactor Crew C",
            "DISPATCHED",
            "Mixed plastic & dry biomass cluster near transit stop.",
        ),
        (
            "West Zone Bio-Sanitation Squad D",
            "IN_PROGRESS",
            "Organic waste containment and curb sanitization in progress.",
        ),
        (
            "South Sector Green Sweeper Team E",
            "COMPLETED",
            "Dry biomass and litter cleared; post-cleanup verification complete.",
        ),
    ]

    for hotspot, (crew_name, disp_status, notes) in zip(hotspots[:5], crew_assignments):
        completed_time = utc_now() if disp_status == "COMPLETED" else None
        dispatch_record = Dispatch(
            hotspot_id=hotspot.id,
            officer_id=officer_id,
            crew_name=crew_name,
            status=disp_status,
            dispatched_at=utc_now(),
            completed_at=completed_time,
            notes=notes,
        )
        db.add(dispatch_record)
        if disp_status == "COMPLETED":
            hotspot.status = "RESOLVED"
        elif disp_status == "IN_PROGRESS":
            hotspot.status = "IN_PROGRESS"
        else:
            hotspot.status = "DISPATCHED"

    db.commit()

    total_incidents = db.query(Incident).count()
    total_hotspots = db.query(Hotspot).count()
    total_dispatches = db.query(Dispatch).count()

    return {
        "message": "CleanPath AI demo dataset seeded successfully.",
        "total_reports": total_incidents,
        "unique_hotspots": total_hotspots,
        "duplicate_reports_suppressed": max(0, total_incidents - total_hotspots),
        "dispatch_operations": total_dispatches,
    }


# ==============================================================================
# FASTAPI LIFESPAN & APPLICATION INITIALIZATION
# ==============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize SQLite schema, run non-destructive migrations, and seed default user accounts."""
    init_db()
    db = SessionLocal()
    try:
        seed_demo_users(db)
        ensure_default_users(db)
        demo_mode = os.getenv("DEMO_MODE", "false").strip().lower() in ("true", "1", "yes")
        if demo_mode:
            if db.query(Incident).count() == 0:
                seed_demo_incidents_and_dispatches(db, clear_existing=False)
        else:
            # In production mode (DEMO_MODE=false), remove synthetic demo_sample_*.jpg
            # rows so the platform starts with only real citizen reports.
            purge_synthetic_demo_incidents(db)
    except Exception as exc:
        logger.error("Startup initialization error: %s", exc)
    finally:
        db.close()
    yield


app = FastAPI(
    title="CleanPath AI (TerraSight AI) API",
    description=(
        "AI-Powered Civic Waste Detection, EXIF GPS Extraction, "
        "50m Haversine DBSCAN Hotspot Clustering, and Municipal Sanitation Dispatch Platform."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

raw_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000,http://localhost:5173,http://127.0.0.1:5173",
)
allowed_origins = [origin.strip().rstrip("/") for origin in raw_origins.split(",") if origin.strip()]

cors_kwargs: Dict[str, Any] = {
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}

if "*" in allowed_origins:
    cors_kwargs["allow_origins"] = ["*"]
    cors_kwargs["allow_credentials"] = False
else:
    cors_kwargs["allow_origins"] = allowed_origins
    cors_kwargs["allow_credentials"] = True
    # Allow all Render preview and custom onrender.com subdomains
    cors_kwargs["allow_origin_regex"] = r"^https:\/\/[a-zA-Z0-9_.-]+\.onrender\.com$"

app.add_middleware(
    CORSMiddleware,
    **cors_kwargs,
)

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

FRONTEND_DIST_DIR = BACKEND_DIR.parent / "frontend" / "dist"
if (FRONTEND_DIST_DIR / "assets").exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_DIST_DIR / "assets")),
        name="frontend_assets",
    )


# ==============================================================================
# HEALTH, GEOCODING & SYSTEM STATUS ENDPOINTS
# ==============================================================================


@app.get(
    "/health",
    tags=["System"],
    summary="Health check and subsystem status endpoint",
)
@app.get("/api/health", include_in_schema=False)
def health_check(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Verify CleanPath AI backend health and report status of AI, Database,
    GIS, and DBSCAN Clustering engines.
    """
    db_online = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_online = False

    vision_status = get_vision_status()
    return {
        "status": "ok",
        "service": "CleanPath AI",
        "subsystems": {
            "ai_engine": vision_status["ai_engine"],
            "inference_mode": vision_status["inference_mode"],
            "weights_loaded": vision_status.get("weights_loaded"),
            "database": "Online" if db_online else "Offline",
            "gis_engine": "Online",
            "clustering_engine": "Online",
        },
    }


@app.get(
    "/reverse-geocode",
    tags=["Geospatial"],
    summary="Reverse geocode latitude/longitude to a human-readable municipal address",
)
@app.get("/api/reverse-geocode", tags=["Geospatial"])
def reverse_geocode_endpoint(
    latitude: float = Query(..., ge=-90.0, le=90.0),
    longitude: float = Query(..., ge=-180.0, le=180.0),
) -> Dict[str, Any]:
    """Return human-readable street/ward address for coordinates with offline fallback."""
    return reverse_geocode_coordinates(latitude, longitude, allow_network=True)


# ==============================================================================
# AUTHENTICATION ENDPOINTS
# ==============================================================================


@app.post(
    "/auth/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Authentication"],
    summary="Register a new citizen or municipal officer account",
)
@app.post(
    "/api/auth/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Authentication"],
    include_in_schema=False,
)
def register_user(
    payload: RegisterRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Create a new user account with hashed password and return a signed JWT."""
    username_clean = payload.username.strip()
    email_clean = (
        payload.email.strip().lower()
        if payload.email
        else f"{username_clean.lower()}@cleanpath.local"
    )

    existing_user = (
        db.query(User)
        .filter((User.username == username_clean) | (User.email == email_clean))
        .first()
    )
    if existing_user is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email is already registered.",
        )

    new_user = User(
        name=(payload.name or username_clean).strip(),
        username=username_clean,
        email=email_clean,
        phone=payload.phone,
        password_hash=hash_password(payload.password),
        role=payload.role,
        is_active=True,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = create_access_token(
        data={"sub": new_user.username, "role": new_user.role, "user_id": new_user.id}
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": new_user,
    }


@app.post(
    "/auth/login",
    response_model=AuthResponse,
    tags=["Authentication"],
    summary="Authenticate user and obtain a JWT access token",
)
@app.post(
    "/api/auth/login",
    response_model=AuthResponse,
    tags=["Authentication"],
    include_in_schema=False,
)
def login_user(
    payload: LoginRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Authenticate a citizen or municipal sanitation officer by username or email
    and return a JWT access token.
    """
    identifier = payload.username.strip()
    user = (
        db.query(User)
        .filter((User.username == identifier) | (User.email == identifier.lower()))
        .first()
    )
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username/email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(
        data={"sub": user.username, "role": user.role, "user_id": user.id}
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user,
    }


@app.get(
    "/auth/me",
    response_model=UserResponse,
    tags=["Authentication"],
    summary="Get profile of currently authenticated user",
)
@app.get(
    "/api/auth/me",
    response_model=UserResponse,
    tags=["Authentication"],
    include_in_schema=False,
)
def get_me(current_user: User = Depends(get_current_user)) -> User:
    """Return the currently authenticated user's profile."""
    return current_user


# ==============================================================================
# CITIZEN INCIDENT / REPORT ENDPOINTS
# ==============================================================================


@app.post(
    "/incidents/analyze",
    tags=["Citizen Portal"],
    summary="Upload a waste photograph, extract EXIF GPS, and run AI waste detection & severity scoring",
)
@app.post("/api/reports/analyze", tags=["Citizen Portal"], include_in_schema=False)
def analyze_incident_image(
    file: UploadFile = File(..., description="Waste photograph (.jpg, .jpeg, .png)"),
) -> Dict[str, Any]:
    """
    Step 1 of the Citizen Reporting Pipeline:
    1. Validate and save uploaded image safely.
    2. Extract EXIF GPS coordinates (or flag MANUAL fallback if absent).
    3. Run YOLOv8 + Spectral ROI Waste Detection Engine.
    4. Compute functional categories, bounding boxes, confidence stats, and Severity Index.
    """
    saved_path, raw_bytes = validate_and_save_upload(file)
    gps_result = extract_gps_from_image(raw_bytes)
    vision_result = analyze_waste_image(raw_bytes)

    image_url = f"/uploads/{saved_path.name}"
    geo_info = reverse_geocode_coordinates(
        gps_result["latitude"], gps_result["longitude"], allow_network=False
    )

    return {
        "image_path": image_url,
        "image_url": image_url,
        "original_filename": file.filename or saved_path.name,
        "image_width": vision_result["image_width"],
        "image_height": vision_result["image_height"],
        "detections": vision_result["detections"],
        "category_counts": vision_result["category_counts"],
        "category_stats": vision_result.get("category_stats", {}),
        "detected_categories": vision_result["detected_categories"],
        "detection_count": vision_result["detection_count"],
        "highest_confidence": vision_result.get("highest_confidence", 0.0),
        "average_confidence": vision_result.get("average_confidence", 0.0),
        "severity_score": vision_result["severity_score"],
        "severity_tier": vision_result["severity_tier"],
        "severity_breakdown": vision_result.get("severity_breakdown", {}),
        "inference_mode": vision_result["inference_mode"],
        "gps": gps_result,
        "latitude": gps_result["latitude"],
        "longitude": gps_result["longitude"],
        "location_source": gps_result["source"],
        "address": geo_info["address"],
    }


@app.post(
    "/incidents",
    status_code=status.HTTP_201_CREATED,
    tags=["Citizen Portal"],
    summary="Submit a verified civic waste complaint with coordinates and AI analysis",
)
@app.post(
    "/api/reports",
    status_code=status.HTTP_201_CREATED,
    tags=["Citizen Portal"],
    include_in_schema=False,
)
def create_incident(
    payload: IncidentCreateRequest,
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Atomically persist a citizen (or guest) waste complaint into SQLite (`incidents`,
    `report_detections`, `audit_logs`) and immediately trigger 50-meter Haversine
    DBSCAN clustering to assign or create a municipal Hotspot (`hotspots`, `hotspot_reports`).
    """
    if not (-90.0 <= payload.latitude <= 90.0 and -180.0 <= payload.longitude <= 180.0):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid geographic coordinates provided.",
        )

    loc_source = (payload.location_source or "MANUAL").strip().upper()
    if loc_source not in ("EXIF", "MANUAL", "CURRENT_LOCATION"):
        loc_source = "MANUAL"

    categories_list: List[str] = []
    detections_list: List[Dict[str, Any]] = list(payload.detections or [])

    if isinstance(payload.detected_categories, str):
        categories_list = parse_categories_json(payload.detected_categories)
    elif isinstance(payload.detected_categories, list):
        for entry in payload.detected_categories:
            if isinstance(entry, str):
                mapped = map_taco_class_to_category(entry)
                if mapped not in categories_list:
                    categories_list.append(mapped)
            elif isinstance(entry, dict):
                detections_list.append(entry)
                cname = map_taco_class_to_category(str(entry.get("class_name", "Plastic Waste")))
                if cname not in categories_list:
                    categories_list.append(cname)

    if not categories_list and detections_list:
        for det in detections_list:
            cname = map_taco_class_to_category(str(det.get("class_name", "Plastic Waste")))
            if cname not in categories_list:
                categories_list.append(cname)

    if not categories_list:
        categories_list = ["Plastic Waste"]

    category_counts_map: Dict[str, int] = {}
    for det in detections_list:
        cname = map_taco_class_to_category(str(det.get("class_name", "Plastic Waste")))
        category_counts_map[cname] = category_counts_map.get(cname, 0) + 1
    if not category_counts_map:
        for c in categories_list:
            category_counts_map[c] = 1

    sev_score = round(float(payload.severity_score), 2)
    sev_tier = payload.severity_tier or determine_severity_tier(sev_score)
    if sev_tier not in ("High", "Medium", "Low"):
        sev_tier = determine_severity_tier(sev_score)

    det_count = max(
        int(payload.detection_count),
        len(detections_list) if detections_list else len(categories_list),
    )

    stored_categories_json = json.dumps(
        {
            "categories": categories_list,
            "counts": category_counts_map,
            "detections": detections_list,
        }
    )

    lat_val = round(float(payload.latitude), 6)
    lon_val = round(float(payload.longitude), 6)
    address_val = (payload.address or "").strip()
    if not address_val:
        address_val = reverse_geocode_coordinates(lat_val, lon_val, allow_network=False)["address"]

    try:
        incident_uuid = generate_incident_uuid(db)
        citizen_id = current_user.id if current_user is not None else None

        new_incident = Incident(
            incident_uuid=incident_uuid,
            citizen_id=citizen_id,
            image_path=payload.image_path,
            thumbnail_path=payload.image_path,
            original_filename=payload.original_filename or Path(payload.image_path).name,
            latitude=lat_val,
            longitude=lon_val,
            location_source=loc_source,
            address=address_val,
            description=(payload.description or "").strip(),
            detected_categories=stored_categories_json,
            detection_count=det_count,
            severity_score=sev_score,
            severity_tier=sev_tier,
            status="SUBMITTED",
        )
        db.add(new_incident)
        db.flush()

        # Persist normalized ReportDetection records
        for det in detections_list:
            bbox = det.get("bbox") or {}
            cname = map_taco_class_to_category(str(det.get("class_name", "Plastic Waste")))
            db.add(
                ReportDetection(
                    report_id=new_incident.id,
                    category=map_category_to_db_enum(cname),
                    confidence=float(det.get("confidence", 0.85)),
                    bounding_box_x1=float(bbox.get("x1", 0)),
                    bounding_box_y1=float(bbox.get("y1", 0)),
                    bounding_box_x2=float(bbox.get("x2", 100)),
                    bounding_box_y2=float(bbox.get("y2", 100)),
                    object_count=1,
                )
            )

        # Record creation AuditLog
        db.add(
            AuditLog(
                report_id=new_incident.id,
                actor_id=citizen_id,
                actor_role=(current_user.role.upper() if current_user else "CITIZEN"),
                action="REPORT_CREATED",
                old_status=None,
                new_status="SUBMITTED",
                notes=f"Waste report {incident_uuid} submitted via {loc_source} at {address_val}.",
            )
        )
        db.commit()
        db.refresh(new_incident)

        # Trigger 50m Haversine DBSCAN hotspot clustering
        recompute_hotspots(db)
        db.refresh(new_incident)

        if new_incident.hotspot_id:
            db.add(
                AuditLog(
                    report_id=new_incident.id,
                    hotspot_id=new_incident.hotspot_id,
                    actor_id=None,
                    actor_role="SYSTEM",
                    action="HOTSPOT_CLUSTERED",
                    old_status="SUBMITTED",
                    new_status=new_incident.status,
                    notes=f"Assigned to 50m DBSCAN Hotspot HS-{new_incident.hotspot_id:04d}.",
                )
            )
            db.commit()
            db.refresh(new_incident)

        return serialize_incident(new_incident, include_dispatches=True)
    except Exception as exc:
        db.rollback()
        logger.error("Failed to create incident atomically: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save report: {exc}",
        ) from exc


@app.get(
    "/incidents/my",
    tags=["Citizen Portal"],
    summary="List waste complaints submitted by the current citizen or guest session",
)
@app.get("/api/reports/my", tags=["Citizen Portal"], include_in_schema=False)
def get_my_incidents(
    guest_uuids: Optional[str] = Query(
        default=None,
        description="Optional comma-separated incident_uuids for guest mode tracking",
    ),
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """
    Return previously submitted complaints strictly isolated to the logged-in citizen,
    or matching explicit guest incident UUIDs when operating in Guest mode.
    Citizen A never sees Citizen B's reports.
    """
    query = db.query(Incident)

    if current_user is not None:
        incidents = (
            query.filter(Incident.citizen_id == current_user.id)
            .order_by(Incident.created_at.desc(), Incident.id.desc())
            .all()
        )
    elif guest_uuids:
        uuid_list = [u.strip().upper() for u in guest_uuids.split(",") if u.strip()]
        if not uuid_list:
            return []
        incidents = (
            query.filter(
                Incident.incident_uuid.in_(uuid_list),
                Incident.citizen_id.is_(None),
            )
            .order_by(Incident.created_at.desc(), Incident.id.desc())
            .all()
        )
    else:
        incidents = []

    return [serialize_incident(inc, include_dispatches=True) for inc in incidents]


@app.get(
    "/incidents/{incident_id}",
    tags=["Citizen Portal"],
    summary="Get detailed status and timeline for a specific incident by ID or UUID",
)
@app.get("/api/reports/{incident_id}", tags=["Citizen Portal"], include_in_schema=False)
@app.get("/api/officer/reports/{incident_id}", tags=["Officer Command Center"], include_in_schema=False)
def get_incident_by_id(
    incident_id: str,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Retrieve a single incident by numeric ID or incident_uuid (e.g. CP-2026-000001 or CP-1001)."""
    incident: Optional[Incident] = None
    if incident_id.isdigit():
        incident = db.query(Incident).filter(Incident.id == int(incident_id)).first()
    if incident is None:
        incident = (
            db.query(Incident)
            .filter(Incident.incident_uuid == incident_id.strip().upper())
            .first()
        )
    if incident is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident '{incident_id}' not found.",
        )

    return serialize_incident(incident, include_dispatches=True)


# ==============================================================================
# MUNICIPAL OFFICER COMMAND CENTER ENDPOINTS (RBAC PROTECTED)
# ==============================================================================


@app.get(
    "/officer/incidents",
    tags=["Officer Command Center"],
    summary="Get all civic waste incidents (Officer RBAC required)",
)
@app.get("/api/officer/reports", tags=["Officer Command Center"], include_in_schema=False)
@app.get("/api/map/reports", tags=["Officer Command Center"], include_in_schema=False)
def get_officer_incidents(
    status_filter: Optional[str] = Query(default=None),
    severity_tier: Optional[str] = Query(default=None),
    category: Optional[str] = Query(default=None),
    officer: User = Depends(require_role("officer")),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Return all citizen waste reports across the municipality."""
    recompute_hotspots(db)
    query = db.query(Incident)
    if status_filter:
        query = query.filter(Incident.status == status_filter.strip().upper())
    if severity_tier:
        query = query.filter(Incident.severity_tier == severity_tier.strip().capitalize())

    incidents = query.order_by(Incident.created_at.desc(), Incident.id.desc()).all()
    serialized = [serialize_incident(inc, include_dispatches=True) for inc in incidents]
    if category:
        cat_norm = category.strip().lower()
        serialized = [
            item
            for item in serialized
            if any(cat_norm in c.lower() for c in item.get("detected_categories", []))
        ]
    return serialized


@app.get(
    "/officer/hotspots",
    tags=["Officer Command Center"],
    summary="Get 50m Haversine DBSCAN clustered hotspots ordered by priority queue",
)
@app.get("/api/hotspots", tags=["Officer Command Center"], include_in_schema=False)
@app.get("/api/map/hotspots", tags=["Officer Command Center"], include_in_schema=False)
def get_officer_hotspots(
    officer: User = Depends(require_role("officer")),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """
    Recompute and return all 50-meter Haversine DBSCAN waste hotspots sorted by:
        1. Severity Tier (High -> Medium -> Low)
        2. Maximum Severity descending
        3. Report Count descending
    Includes all underlying citizen reports for auditability.
    """
    hotspots = recompute_hotspots(db)
    return [serialize_hotspot(h) for h in hotspots]


@app.get(
    "/api/hotspots/{hotspot_id}",
    tags=["Officer Command Center"],
    summary="Get detailed information for a specific 50m DBSCAN hotspot",
)
def get_hotspot_by_id(
    hotspot_id: int,
    officer: User = Depends(require_role("officer")),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Return a single Hotspot record with its clustered reports, dispatches, and audit logs."""
    recompute_hotspots(db)
    hotspot = db.query(Hotspot).filter(Hotspot.id == hotspot_id).first()
    if hotspot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hotspot #{hotspot_id} not found.",
        )
    return serialize_hotspot(hotspot)


@app.get(
    "/officer/analytics",
    tags=["Officer Command Center"],
    summary="Get municipal waste analytics, deduplication metrics, and system status",
)
@app.get("/api/dashboard/statistics", tags=["Officer Command Center"], include_in_schema=False)
def get_officer_analytics(
    officer: User = Depends(require_role("officer")),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Compute comprehensive municipal analytics from the real SQLite database
    including severity distributions, functional waste category counts,
    incident/hotspot status metrics, reports today, and DBSCAN duplicate suppression.
    """
    hotspots = recompute_hotspots(db)
    incidents = db.query(Incident).all()
    dispatches = db.query(Dispatch).all()

    total_reports = len(incidents)
    non_rejected_reports = sum(1 for i in incidents if i.status != "REJECTED")
    total_hotspots = len(hotspots)
    active_hotspots = sum(1 for h in hotspots if h.status != "RESOLVED")
    duplicate_suppressed = max(0, non_rejected_reports - total_hotspots)

    today_date = datetime.now(timezone.utc).date()
    reports_today = sum(
        1
        for i in incidents
        if i.created_at and i.created_at.date() == today_date
    )

    high_incidents = sum(1 for i in incidents if i.severity_tier == "High")
    medium_incidents = sum(1 for i in incidents if i.severity_tier == "Medium")
    low_incidents = sum(1 for i in incidents if i.severity_tier == "Low")

    high_hotspots = sum(1 for h in hotspots if h.severity_tier == "High")
    medium_hotspots = sum(1 for h in hotspots if h.severity_tier == "Medium")
    low_hotspots = sum(1 for h in hotspots if h.severity_tier == "Low")

    pending_incidents = sum(
        1 for i in incidents if i.status in ("SUBMITTED", "PENDING", "VERIFIED", "ACKNOWLEDGED")
    )
    acknowledged_incidents = sum(
        1 for i in incidents if i.status in ("VERIFIED", "ACKNOWLEDGED")
    )
    dispatched_incidents = sum(
        1 for i in incidents if i.status in ("DISPATCHED", "IN_PROGRESS")
    )
    in_progress_incidents = sum(1 for i in incidents if i.status == "IN_PROGRESS")
    resolved_incidents = sum(1 for i in incidents if i.status == "RESOLVED")
    rejected_incidents = sum(1 for i in incidents if i.status == "REJECTED")

    active_dispatches = sum(
        1 for d in dispatches if d.status not in ("COMPLETED", "RESOLVED")
    )

    category_counts: Dict[str, int] = {cat: 0 for cat in FUNCTIONAL_CATEGORIES}
    for inc in incidents:
        parsed = _parse_stored_detection_payload(inc.detected_categories)
        counts_map = parsed.get("counts") or {}
        if counts_map:
            for cat, cnt in counts_map.items():
                mapped = map_taco_class_to_category(cat)
                category_counts[mapped] = category_counts.get(mapped, 0) + int(cnt)
        else:
            for cat in parsed["categories"]:
                mapped = map_taco_class_to_category(cat)
                category_counts[mapped] = category_counts.get(mapped, 0) + 1

    vision_status = get_vision_status()

    return {
        "total_reports": total_reports,
        "reports_today": reports_today,
        "total_hotspots": total_hotspots,
        "active_hotspots": active_hotspots,
        "duplicate_reports_suppressed": duplicate_suppressed,
        "dispatch_operations": len(dispatches),
        "active_dispatches": active_dispatches,
        "cluster_radius_meters": int(DEFAULT_CLUSTER_RADIUS_METERS),
        "high_severity": high_incidents,
        "medium_severity": medium_incidents,
        "low_severity": low_incidents,
        "hotspot_severity_breakdown": {
            "High": high_hotspots,
            "Medium": medium_hotspots,
            "Low": low_hotspots,
        },
        "pending": pending_incidents,
        "acknowledged": acknowledged_incidents,
        "dispatched": dispatched_incidents,
        "in_progress": in_progress_incidents,
        "resolved": resolved_incidents,
        "rejected": rejected_incidents,
        "category_breakdown": category_counts,
        "system_status": {
            "ai_engine": vision_status["ai_engine"],
            "inference_mode": vision_status["inference_mode"],
            "database": "Online",
            "gis_engine": "Online",
            "clustering_engine": "Online",
        },
    }


def _execute_hotspot_dispatch(
    hotspot_id: int,
    payload: DispatchCreateRequest,
    officer: User,
    db: Session,
) -> Dict[str, Any]:
    hotspot = db.query(Hotspot).filter(Hotspot.id == hotspot_id).first()
    if hotspot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hotspot #{hotspot_id} not found.",
        )

    crew_name_clean = payload.crew_name.strip()
    if not crew_name_clean:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Crew name cannot be empty.",
        )

    old_hotspot_status = hotspot.status
    dispatch_record = Dispatch(
        hotspot_id=hotspot.id,
        officer_id=officer.id,
        crew_name=crew_name_clean,
        vehicle_number=(payload.vehicle_number or "").strip() or None,
        status="DISPATCHED",
        dispatched_at=utc_now(),
        notes=(payload.notes or "").strip(),
    )
    db.add(dispatch_record)

    hotspot.status = "DISPATCHED"
    hotspot.updated_at = utc_now()

    db.add(
        AuditLog(
            hotspot_id=hotspot.id,
            actor_id=officer.id,
            actor_role="OFFICER",
            action="CREW_DISPATCHED",
            old_status=old_hotspot_status,
            new_status="DISPATCHED",
            notes=f"Dispatched '{crew_name_clean}' to Hotspot HS-{hotspot.id:04d}. {payload.notes or ''}".strip(),
        )
    )

    for inc in hotspot.incidents:
        if inc.status not in ("RESOLVED", "REJECTED"):
            prev_status = inc.status
            inc.status = "DISPATCHED"
            inc.updated_at = utc_now()
            db.add(
                AuditLog(
                    report_id=inc.id,
                    hotspot_id=hotspot.id,
                    actor_id=officer.id,
                    actor_role="OFFICER",
                    action="CREW_DISPATCHED",
                    old_status=prev_status,
                    new_status="DISPATCHED",
                    notes=f"Sanitation crew '{crew_name_clean}' dispatched via Hotspot HS-{hotspot.id:04d}.",
                )
            )

    db.commit()
    db.refresh(dispatch_record)
    db.refresh(hotspot)

    return {
        "message": f"Sanitation crew '{crew_name_clean}' dispatched to Hotspot #{hotspot.id}.",
        "dispatch": serialize_dispatch(dispatch_record),
        "hotspot": serialize_hotspot(hotspot),
    }


@app.post(
    "/officer/dispatch/{hotspot_id}",
    status_code=status.HTTP_201_CREATED,
    tags=["Officer Command Center"],
    summary="Dispatch a municipal sanitation crew to a clustered hotspot",
)
def dispatch_sanitation_crew(
    hotspot_id: int,
    payload: DispatchCreateRequest,
    officer: User = Depends(require_role("officer")),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Create a sanitation dispatch order for a unified DBSCAN hotspot, update the
    hotspot status to DISPATCHED, and update all constituent incidents accordingly.
    """
    return _execute_hotspot_dispatch(hotspot_id, payload, officer, db)


@app.post(
    "/api/dispatch",
    status_code=status.HTTP_201_CREATED,
    tags=["Officer Command Center"],
    summary="Dispatch a sanitation crew (via body hotspot_id)",
)
def dispatch_sanitation_crew_body(
    payload: DispatchCreateRequest,
    officer: User = Depends(require_role("officer")),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    if not payload.hotspot_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="hotspot_id is required.",
        )
    return _execute_hotspot_dispatch(payload.hotspot_id, payload, officer, db)


@app.patch(
    "/officer/incidents/{incident_id}/status",
    tags=["Officer Command Center"],
    summary="Update status of an individual incident and synchronize its parent hotspot",
)
@app.patch(
    "/api/officer/reports/{incident_id}/status",
    tags=["Officer Command Center"],
    include_in_schema=False,
)
def update_incident_status(
    incident_id: str,
    payload: StatusUpdateRequest,
    officer: User = Depends(require_role("officer")),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Update an incident's lifecycle status and log the transition in AuditLog."""
    new_status = payload.status.strip().upper()
    if new_status not in VALID_INCIDENT_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid incident status '{new_status}'. Allowed: {', '.join(sorted(VALID_INCIDENT_STATUSES))}",
        )

    incident: Optional[Incident] = None
    if incident_id.isdigit():
        incident = db.query(Incident).filter(Incident.id == int(incident_id)).first()
    if incident is None:
        incident = (
            db.query(Incident)
            .filter(Incident.incident_uuid == incident_id.strip().upper())
            .first()
        )
    if incident is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident '{incident_id}' not found.",
        )

    old_status = incident.status
    incident.status = new_status
    incident.updated_at = utc_now()

    db.add(
        AuditLog(
            report_id=incident.id,
            hotspot_id=incident.hotspot_id,
            actor_id=officer.id,
            actor_role="OFFICER",
            action="STATUS_UPDATED",
            old_status=old_status,
            new_status=new_status,
            notes=(payload.notes or f"Officer updated report status from {old_status} to {new_status}.").strip(),
        )
    )

    # Synchronize parent hotspot status if linked
    if incident.hotspot is not None:
        sibling_statuses = {
            sib.status for sib in incident.hotspot.incidents if sib.status != "REJECTED"
        }
        if sibling_statuses and sibling_statuses == {"RESOLVED"}:
            incident.hotspot.status = "RESOLVED"
            for disp in incident.hotspot.dispatches:
                disp.status = "COMPLETED"
                if disp.completed_at is None:
                    disp.completed_at = utc_now()
        elif "IN_PROGRESS" in sibling_statuses:
            incident.hotspot.status = "IN_PROGRESS"
        elif "DISPATCHED" in sibling_statuses:
            incident.hotspot.status = "DISPATCHED"
        elif "ACKNOWLEDGED" in sibling_statuses or "VERIFIED" in sibling_statuses:
            incident.hotspot.status = "ACKNOWLEDGED"
        else:
            incident.hotspot.status = "PENDING"
        incident.hotspot.updated_at = utc_now()

    db.commit()
    db.refresh(incident)
    return serialize_incident(incident, include_dispatches=True)


@app.patch(
    "/officer/hotspots/{hotspot_id}/status",
    tags=["Officer Command Center"],
    summary="Update status of an entire hotspot and all its constituent incidents",
)
@app.patch(
    "/api/hotspots/{hotspot_id}/status",
    tags=["Officer Command Center"],
    include_in_schema=False,
)
@app.patch(
    "/api/dispatch/{hotspot_id}/status",
    tags=["Officer Command Center"],
    include_in_schema=False,
)
def update_hotspot_status(
    hotspot_id: int,
    payload: StatusUpdateRequest,
    officer: User = Depends(require_role("officer")),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Update a hotspot's status (PENDING, ACKNOWLEDGED, DISPATCHED, IN_PROGRESS, RESOLVED) and propagate to its incidents."""
    new_status = payload.status.strip().upper()
    if new_status not in VALID_HOTSPOT_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid hotspot status '{new_status}'. Allowed: {', '.join(sorted(VALID_HOTSPOT_STATUSES))}",
        )

    hotspot = db.query(Hotspot).filter(Hotspot.id == hotspot_id).first()
    if hotspot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hotspot #{hotspot_id} not found.",
        )

    old_hotspot_status = hotspot.status
    hotspot.status = new_status
    hotspot.updated_at = utc_now()

    incident_target_status = {
        "PENDING": "VERIFIED",
        "ACKNOWLEDGED": "ACKNOWLEDGED",
        "DISPATCHED": "DISPATCHED",
        "IN_PROGRESS": "IN_PROGRESS",
        "RESOLVED": "RESOLVED",
    }[new_status]

    db.add(
        AuditLog(
            hotspot_id=hotspot.id,
            actor_id=officer.id,
            actor_role="OFFICER",
            action="HOTSPOT_STATUS_UPDATED",
            old_status=old_hotspot_status,
            new_status=new_status,
            notes=(payload.notes or f"Hotspot HS-{hotspot.id:04d} status updated to {new_status}.").strip(),
        )
    )

    for inc in hotspot.incidents:
        if inc.status != "REJECTED":
            prev_inc_status = inc.status
            inc.status = incident_target_status
            inc.updated_at = utc_now()
            db.add(
                AuditLog(
                    report_id=inc.id,
                    hotspot_id=hotspot.id,
                    actor_id=officer.id,
                    actor_role="OFFICER",
                    action="STATUS_UPDATED",
                    old_status=prev_inc_status,
                    new_status=incident_target_status,
                    notes=(payload.notes or f"Status updated to {incident_target_status} via Hotspot HS-{hotspot.id:04d}.").strip(),
                )
            )

    if new_status == "IN_PROGRESS":
        for disp in hotspot.dispatches:
            if disp.status != "COMPLETED":
                disp.status = "IN_PROGRESS"
                if getattr(disp, "started_at", None) is None:
                    disp.started_at = utc_now()
    elif new_status == "RESOLVED":
        for disp in hotspot.dispatches:
            disp.status = "COMPLETED"
            if disp.completed_at is None:
                disp.completed_at = utc_now()

    db.commit()
    db.refresh(hotspot)
    return serialize_hotspot(hotspot)


# ==============================================================================
# DEMO MODE SEED / CLEAR ENDPOINTS (RBAC PROTECTED)
# ==============================================================================


@app.post(
    "/demo/seed",
    tags=["Demo & Simulation"],
    summary="Seed 15 realistic civic waste incidents (10 DBSCAN hotspots, 5 dispatches)",
)
def seed_demo_data_endpoint(
    payload: Optional[DemoSeedRequest] = None,
    officer: User = Depends(require_role("officer")),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Generate 15 sample incidents (5 High, 5 Medium, 5 Low) around a configurable
    center coordinate that cluster via 50m Haversine DBSCAN into 10 unique
    hotspots and 5 dispatch operations. Protected so only officers can invoke it.
    """
    req = payload or DemoSeedRequest()
    return seed_demo_incidents_and_dispatches(
        db=db,
        center_lat=req.center_latitude,
        center_lon=req.center_longitude,
        clear_existing=req.clear_existing,
    )


@app.post(
    "/demo/clear",
    tags=["Demo & Simulation"],
    summary="Clear synthetic demo incidents and keep only real citizen reports",
)
def clear_demo_data_endpoint(
    officer: User = Depends(require_role("officer")),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Remove synthetic demo_sample_*.jpg incidents and recompute hotspots."""
    return purge_synthetic_demo_incidents(db)


# ==============================================================================
# ROOT & SPA FRONTEND SERVING (Allows opening http://127.0.0.1:8000 directly)
# ==============================================================================

from fastapi.responses import FileResponse, JSONResponse, Response


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    """Serve a clean SVG leaf icon so browsers never log 404 on /favicon.ico."""
    svg_icon = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<rect width="32" height="32" rx="8" fill="#16A34A"/>'
        '<path d="M9 23C9 14 16 9 23 9C23 16 18 23 9 23Z" fill="#FFFFFF"/>'
        "</svg>"
    )
    return Response(content=svg_icon, media_type="image/svg+xml")


def _serve_spa_or_api_info() -> Response:
    index_file = FRONTEND_DIST_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse(
        {
            "service": "CleanPath AI (TerraSight AI)",
            "status": "ok",
            "frontend_url": "http://localhost:3000",
            "api_docs": "http://localhost:8000/docs",
            "health": "http://localhost:8000/health",
        }
    )


@app.get("/", include_in_schema=False)
def root_index() -> Response:
    """Serve CleanPath AI web application at root URL."""
    return _serve_spa_or_api_info()


@app.get("/login", include_in_schema=False)
@app.get("/citizen", include_in_schema=False)
@app.get("/citizen/{rest_path:path}", include_in_schema=False)
@app.get("/officer", include_in_schema=False)
@app.get("/officer/map", include_in_schema=False)
@app.get("/officer/{rest_path:path}", include_in_schema=False)
@app.get("/map", include_in_schema=False)
def spa_frontend_routes(rest_path: Optional[str] = None) -> Response:
    """Serve React SPA index.html for client-side routes on port 8000."""
    return _serve_spa_or_api_info()


if __name__ == "__main__":
    import uvicorn

    server_port = int(os.getenv("PORT", "8000"))
    logger.info("Starting TerraSight AI server on 0.0.0.0:%d", server_port)
    uvicorn.run("main:app", host="0.0.0.0", port=server_port, reload=False)


