"""
CleanPath AI / TerraSight AI — Optional Demo Seed Script
========================================================
Seeds or clears optional municipal demonstration incidents across New Delhi
with full relational entities (`Incident`, `ReportDetection`, `Hotspot`,
`HotspotReport`, `Dispatch`, `AuditLog`).

IMPORTANT:
In production mode (`DEMO_MODE=false`, the default), the backend does NOT
auto-seed fake reports on startup. Only real citizen reports are displayed.
Run this script manually if you want pre-populated demo data:

    cd backend
    python seed_demo.py          # Seed 15 realistic clustered incidents
    python seed_demo.py --clear  # Remove synthetic demo incidents
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw
from sqlalchemy.orm import Session

from auth import get_password_hash
from database import (
    AuditLog,
    Dispatch,
    Hotspot,
    HotspotReport,
    Incident,
    IncidentStatus,
    LocationSource,
    ReportDetection,
    SessionLocal,
    SeverityTier,
    User,
    UserRole,
    init_db,
)
from geospatial import recompute_hotspots, reverse_geocode_coordinates
from vision_engine import map_category_to_db_enum

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_default_users(db: Session) -> Dict[str, User]:
    """Ensure default Citizen and Officer accounts exist for login."""
    citizen = db.query(User).filter(User.username == "citizen_demo").first()
    if not citizen:
        citizen = User(
            name="Aarav Sharma (Citizen)",
            username="citizen_demo",
            email="citizen@cleanpath.ai",
            password_hash=get_password_hash("citizen123"),
            role=UserRole.CITIZEN.value,
            phone="+91-98101-11223",
            is_active=True,
        )
        db.add(citizen)

    citizen_b = db.query(User).filter(User.username == "citizen_priya").first()
    if not citizen_b:
        citizen_b = User(
            name="Priya Verma (Citizen)",
            username="citizen_priya",
            email="priya@cleanpath.ai",
            password_hash=get_password_hash("citizen123"),
            role=UserRole.CITIZEN.value,
            phone="+91-98101-44556",
            is_active=True,
        )
        db.add(citizen_b)

    officer = db.query(User).filter(User.username == "officer_admin").first()
    if not officer:
        officer = User(
            name="Rajeshwari Nair (Sanitation Officer)",
            username="officer_admin",
            email="officer@cleanpath.ai",
            password_hash=get_password_hash("officer123"),
            role=UserRole.OFFICER.value,
            phone="+91-11-2322-0900",
            is_active=True,
        )
        db.add(officer)

    db.commit()
    db.refresh(citizen)
    db.refresh(citizen_b)
    db.refresh(officer)
    return {"citizen": citizen, "citizen_b": citizen_b, "officer": officer}


def _generate_demo_image(filename: str, variant: int = 0) -> str:
    """Create a realistic waste sample image in uploads/ if not already present."""
    filepath = UPLOADS_DIR / filename
    if filepath.exists():
        return f"/uploads/{filename}"

    img = Image.new("RGB", (640, 480), color=(78, 82, 86))
    draw = ImageDraw.Draw(img)

    for y in range(0, 480, 24):
        shade = 70 + (y // 24) * 2
        draw.rectangle([0, y, 640, y + 24], fill=(shade, shade + 2, shade + 4))

    draw.rectangle([65, 130, 195, 270], fill=(35, 135, 225), outline=(220, 240, 255), width=2)
    draw.rectangle([205, 175, 315, 295], fill=(235, 240, 245), outline=(180, 195, 210), width=2)
    if variant % 2 == 0:
        draw.rectangle([90, 285, 240, 395], fill=(25, 165, 195), outline=(200, 235, 245), width=2)

    draw.ellipse([330, 190, 470, 325], fill=(165, 118, 42), outline=(120, 82, 25), width=2)
    draw.ellipse([365, 220, 445, 295], fill=(210, 160, 45))

    draw.polygon([(475, 245), (595, 215), (615, 355), (490, 375)], fill=(95, 130, 48), outline=(70, 95, 32))
    img.save(filepath, "JPEG", quality=88)
    return f"/uploads/{filename}"


def purge_synthetic_demo_incidents(db: Session) -> Dict[str, Any]:
    """
    Remove synthetic `demo_sample_*.jpg` incidents and recompute hotspots
    so that only real citizen-submitted reports remain in the database.
    """
    synthetic_incidents = (
        db.query(Incident)
        .filter(Incident.image_path.like("%/demo_sample_%"))
        .all()
    )
    if not synthetic_incidents:
        return {"removed_incidents": 0}

    synthetic_ids = [inc.id for inc in synthetic_incidents]
    db.query(ReportDetection).filter(ReportDetection.report_id.in_(synthetic_ids)).delete(
        synchronize_session=False
    )
    db.query(HotspotReport).filter(HotspotReport.report_id.in_(synthetic_ids)).delete(
        synchronize_session=False
    )
    db.query(AuditLog).filter(AuditLog.report_id.in_(synthetic_ids)).delete(
        synchronize_session=False
    )
    for inc in synthetic_incidents:
        db.delete(inc)
    db.commit()

    # Recompute hotspots from remaining real citizen reports
    recompute_hotspots(db)
    return {"removed_incidents": len(synthetic_ids)}


def seed_demo_data(db: Session, reset_existing: bool = False) -> Dict[str, Any]:
    """
    Seed 15 realistic municipal waste incidents across New Delhi forming 4 DBSCAN
    50m hotspots + isolated incidents, complete with ReportDetection, HotspotReport,
    Dispatch, and AuditLog records.
    """
    if reset_existing:
        db.query(AuditLog).delete()
        db.query(ReportDetection).delete()
        db.query(HotspotReport).delete()
        db.query(Dispatch).delete()
        db.query(Incident).delete()
        db.query(Hotspot).delete()
        db.commit()

    users = ensure_default_users(db)
    citizen = users["citizen"]
    officer = users["officer"]

    existing_demo_count = (
        db.query(Incident).filter(Incident.image_path.like("%/demo_sample_%")).count()
    )
    if existing_demo_count >= 10 and not reset_existing:
        hotspots = recompute_hotspots(db)
        return {
            "seeded": False,
            "message": "Demo data already present",
            "incidents": db.query(Incident).count(),
            "hotspots": len(hotspots),
        }

    img_1 = _generate_demo_image("demo_sample_1.jpg", 0)
    img_2 = _generate_demo_image("demo_sample_2.jpg", 1)
    img_3 = _generate_demo_image("demo_sample_3.jpg", 2)

    demo_samples = [
        # Cluster 1: Connaught Place Outer Circle (4 reports within ~30m -> High Severity Hotspot)
        (28.63150, 77.21670, LocationSource.EXIF.value, {"Plastic Waste": 4, "Organic Waste": 2}, 84.5, SeverityTier.HIGH.value, img_1, "Overflowing plastic bottles and food containers near Gate 2"),
        (28.63162, 77.21681, LocationSource.EXIF.value, {"Plastic Waste": 3, "Dry Leaves / Biomass": 1}, 76.2, SeverityTier.HIGH.value, img_2, "Repeated dumping of single-use plastic bags by street vendors"),
        (28.63142, 77.21660, LocationSource.MANUAL.value, {"Plastic Waste": 5, "Organic Waste": 1}, 88.4, SeverityTier.HIGH.value, img_1, "Uncollected roadside pile blocking pedestrian walkway"),
        (28.63155, 77.21675, LocationSource.EXIF.value, {"Plastic Waste": 2, "Organic Waste": 2}, 68.0, SeverityTier.MEDIUM.value, img_3, "Secondary report confirming uncollected municipal waste pile"),

        # Cluster 2: Karol Bagh Market (3 reports within ~25m -> High Severity Hotspot)
        (28.65190, 77.19090, LocationSource.EXIF.value, {"Organic Waste": 4, "Plastic Waste": 2}, 78.0, SeverityTier.HIGH.value, img_2, "Rotting vegetable waste and plastic packaging outside market lane"),
        (28.65202, 77.19101, LocationSource.MANUAL.value, {"Organic Waste": 3, "Dry Leaves / Biomass": 2}, 64.5, SeverityTier.MEDIUM.value, img_3, "Organic waste attracting stray animals near metro pillar"),
        (28.65185, 77.19082, LocationSource.EXIF.value, {"Plastic Waste": 3, "Organic Waste": 2}, 72.1, SeverityTier.HIGH.value, img_1, "Market waste spillover onto main arterial road"),

        # Cluster 3: Chandni Chowk Heritage Corridor (3 reports within ~35m -> Medium Severity Hotspot)
        (28.65060, 77.23030, LocationSource.EXIF.value, {"Plastic Waste": 2, "Dry Leaves / Biomass": 2}, 58.4, SeverityTier.MEDIUM.value, img_3, "Plastic wrappers and dry leaves accumulated near shopfronts"),
        (28.65071, 77.23042, LocationSource.MANUAL.value, {"Organic Waste": 2, "Plastic Waste": 1}, 52.0, SeverityTier.MEDIUM.value, img_2, "Food waste left beside overflowing dustbin"),
        (28.65052, 77.23021, LocationSource.EXIF.value, {"Dry Leaves / Biomass": 3, "Plastic Waste": 1}, 46.8, SeverityTier.MEDIUM.value, img_3, "Sweepings and packaging debris piled on corner"),

        # Cluster 4: Lajpat Nagar Central Market (2 reports within ~20m -> Low/Medium Hotspot)
        (28.56770, 77.24330, LocationSource.MANUAL.value, {"Dry Leaves / Biomass": 3}, 34.2, SeverityTier.LOW.value, img_3, "Dry garden clippings and leaves near park boundary"),
        (28.56781, 77.24339, LocationSource.EXIF.value, {"Plastic Waste": 1, "Dry Leaves / Biomass": 2}, 41.5, SeverityTier.MEDIUM.value, img_2, "Mixed leaf litter and plastic bottles"),

        # Isolated incidents
        (28.61290, 77.22950, LocationSource.EXIF.value, {"Plastic Waste": 2}, 38.0, SeverityTier.MEDIUM.value, img_1, "Plastic water bottles discarded along footpath"),
        (28.59210, 77.20460, LocationSource.MANUAL.value, {"Dry Leaves / Biomass": 2}, 22.5, SeverityTier.LOW.value, img_3, "Dry leaves pile near bus stop"),
        (28.63580, 77.22450, LocationSource.EXIF.value, {"Plastic Waste": 3, "Organic Waste": 1}, 69.5, SeverityTier.MEDIUM.value, img_2, "Roadside plastic and organic litter near Barakhamba Road"),
    ]

    sample_boxes = {
        "Plastic Waste": {"x1": 65, "y1": 130, "x2": 195, "y2": 270},
        "Organic Waste": {"x1": 330, "y1": 190, "x2": 470, "y2": 325},
        "Dry Leaves / Biomass": {"x1": 475, "y1": 215, "x2": 615, "y2": 375},
    }

    created_incidents: List[Incident] = []
    year_str = datetime.now(timezone.utc).strftime("%Y")

    for idx, (lat, lon, src, cats, sev_score, sev_tier, img_path, desc) in enumerate(demo_samples, start=1):
        det_list = []
        for cat_name, count in cats.items():
            for i in range(count):
                base_box = sample_boxes.get(cat_name, {"x1": 80, "y1": 120, "x2": 220, "y2": 260})
                offset = i * 18
                det_list.append(
                    {
                        "class_id": 0 if cat_name == "Plastic Waste" else (1 if cat_name == "Organic Waste" else 2),
                        "class_name": cat_name,
                        "confidence": round(0.89 - (i * 0.03), 3),
                        "bbox": {
                            "x1": base_box["x1"] + offset,
                            "y1": base_box["y1"] + offset,
                            "x2": min(630, base_box["x2"] + offset),
                            "y2": min(470, base_box["y2"] + offset),
                        },
                        "area_ratio": 0.042,
                    }
                )

        geo_info = reverse_geocode_coordinates(lat, lon, allow_network=False)
        public_id = f"CP-{year_str}-{idx:06d}"

        incident = Incident(
            incident_uuid=public_id,
            citizen_id=citizen.id,
            image_path=img_path,
            thumbnail_path=img_path,
            original_filename=Path(img_path).name,
            latitude=lat,
            longitude=lon,
            location_source=src,
            address=geo_info["address"],
            description=desc,
            detected_categories=json.dumps({"counts": cats, "detections": det_list}),
            detection_count=sum(cats.values()),
            severity_score=sev_score,
            severity_tier=sev_tier,
            status=IncidentStatus.SUBMITTED.value,
        )
        db.add(incident)
        db.flush()

        for det in det_list:
            bbox = det["bbox"]
            db.add(
                ReportDetection(
                    report_id=incident.id,
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
                report_id=incident.id,
                actor_id=citizen.id,
                actor_role=UserRole.CITIZEN.value,
                action="REPORT_CREATED",
                old_status=None,
                new_status=IncidentStatus.SUBMITTED.value,
                notes=f"Citizen submitted waste report ({public_id}) via {src}.",
            )
        )
        created_incidents.append(incident)

    db.commit()
    hotspots = recompute_hotspots(db)

    if hotspots:
        top_hotspot = hotspots[0]
        existing_dispatch = db.query(Dispatch).filter(Dispatch.hotspot_id == top_hotspot.id).first()
        if not existing_dispatch:
            dispatch = Dispatch(
                hotspot_id=top_hotspot.id,
                officer_id=officer.id,
                crew_name="Alpha Rapid Sanitation Unit #04",
                vehicle_number="DL-1G-4821",
                status=IncidentStatus.DISPATCHED.value,
                notes="Priority dispatch triggered by 4 clustered citizen reports within 50m radius.",
            )
            top_hotspot.status = IncidentStatus.DISPATCHED.value
            for inc in db.query(Incident).filter(Incident.hotspot_id == top_hotspot.id).all():
                inc.status = IncidentStatus.DISPATCHED.value
                db.add(
                    AuditLog(
                        report_id=inc.id,
                        hotspot_id=top_hotspot.id,
                        actor_id=officer.id,
                        actor_role=UserRole.OFFICER.value,
                        action="CREW_DISPATCHED",
                        old_status=IncidentStatus.SUBMITTED.value,
                        new_status=IncidentStatus.DISPATCHED.value,
                        notes="Dispatched Alpha Rapid Sanitation Unit #04 (DL-1G-4821).",
                    )
                )
            db.add(dispatch)
            db.commit()

    return {
        "seeded": True,
        "incidents": db.query(Incident).count(),
        "hotspots": len(hotspots),
        "dispatches": db.query(Dispatch).count(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CleanPath AI Demo Data Seeder")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Remove synthetic demo incidents and keep only real citizen reports",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset entire database and re-seed fresh demo incidents",
    )
    args = parser.parse_args()

    init_db()
    session = SessionLocal()
    try:
        ensure_default_users(session)
        if args.clear:
            result = purge_synthetic_demo_incidents(session)
            print(f"[CleanPath AI] Purged synthetic demo incidents: {result}")
        else:
            result = seed_demo_data(session, reset_existing=args.reset)
            print(f"[CleanPath AI] Demo seed complete: {result}")
    finally:
        session.close()
