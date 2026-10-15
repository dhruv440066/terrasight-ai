"""
Database configuration, SQLAlchemy ORM models, and non-destructive schema
migrations for CleanPath AI / TerraSight AI.

Supports PostgreSQL (via DATABASE_URL) with automatic local SQLite fallback
(cleanpath.db). Defines Users, Incidents (Reports), ReportDetections, Hotspots,
HotspotReports, Dispatches, and AuditLogs entities.
"""

import logging
import os
from datetime import datetime, timezone
from enum import Enum as PyEnum
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

load_dotenv()

logger = logging.getLogger("cleanpath.database")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "cleanpath.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH.as_posix()}")

if DATABASE_URL == "sqlite:///./cleanpath.db":
    DATABASE_URL = f"sqlite:///{DEFAULT_DB_PATH.as_posix()}"

# Normalize Render/Heroku postgres:// and generic postgresql:// to use psycopg2 driver
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://") and not DATABASE_URL.startswith("postgresql+"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

engine_kwargs = {}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    engine_kwargs["pool_pre_ping"] = True
    engine_kwargs["pool_recycle"] = 300

engine = create_engine(DATABASE_URL, **engine_kwargs)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy declarative models."""


class UserRole(str, PyEnum):
    CITIZEN = "citizen"
    OFFICER = "officer"
    ADMIN = "admin"


class LocationSource(str, PyEnum):
    EXIF = "EXIF"
    MANUAL = "MANUAL"
    CURRENT_LOCATION = "CURRENT_LOCATION"


class IncidentStatus(str, PyEnum):
    SUBMITTED = "SUBMITTED"
    VERIFIED = "VERIFIED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    DISPATCHED = "DISPATCHED"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    REJECTED = "REJECTED"


class SeverityTier(str, PyEnum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


def utc_now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc)


class User(Base):
    """Registered platform user (CITIZEN, OFFICER, or ADMIN)."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    name = Column(String(150), nullable=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(32), nullable=False, default="citizen")  # citizen | officer | admin
    phone = Column(String(40), nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    incidents = relationship("Incident", back_populates="citizen")
    dispatches = relationship("Dispatch", back_populates="officer")
    audit_logs = relationship(
        "AuditLog", back_populates="actor", foreign_keys="AuditLog.actor_id"
    )


class Hotspot(Base):
    """Geospatially clustered waste hotspot generated via 50m Haversine DBSCAN."""

    __tablename__ = "hotspots"
    __table_args__ = (
        Index("ix_hotspots_status", "status"),
    )

    id = Column(Integer, primary_key=True, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    report_count = Column(Integer, nullable=False, default=1)
    duplicate_count = Column(Integer, nullable=False, default=0)
    maximum_severity = Column(Float, nullable=False, default=0.0)
    severity_tier = Column(String(32), nullable=False, default="Low")  # High | Medium | Low
    radius_meters = Column(Float, nullable=False, default=50.0)
    status = Column(
        String(32), nullable=False, default="PENDING"
    )  # PENDING | ACKNOWLEDGED | DISPATCHED | IN_PROGRESS | RESOLVED
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    incidents = relationship("Incident", back_populates="hotspot")
    hotspot_reports = relationship(
        "HotspotReport", back_populates="hotspot", cascade="all, delete-orphan"
    )
    dispatches = relationship(
        "Dispatch", back_populates="hotspot", cascade="all, delete-orphan"
    )
    audit_logs = relationship("AuditLog", back_populates="hotspot")


class Incident(Base):
    """
    Individual citizen-submitted civic waste report (Single Source of Truth).
    Mapped to 'incidents' table for backward compatibility and aliased as Report.
    """

    __tablename__ = "incidents"
    __table_args__ = (
        Index("ix_incidents_status", "status"),
        Index("ix_incidents_created_at", "created_at"),
        Index("ix_incidents_lat_lon", "latitude", "longitude"),
    )

    id = Column(Integer, primary_key=True, index=True)
    incident_uuid = Column(String(64), unique=True, index=True, nullable=False)
    citizen_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    image_path = Column(String(512), nullable=False)
    thumbnail_path = Column(String(512), nullable=True)
    original_filename = Column(String(255), nullable=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    location_source = Column(
        String(32), nullable=False, default="MANUAL"
    )  # EXIF | MANUAL | CURRENT_LOCATION
    address = Column(String(512), nullable=True)
    description = Column(Text, nullable=True)
    detected_categories = Column(Text, nullable=False, default="[]")  # JSON string
    detection_count = Column(Integer, nullable=False, default=0)
    severity_score = Column(Float, nullable=False, default=0.0)
    severity_tier = Column(String(32), nullable=False, default="Low")  # High | Medium | Low
    status = Column(
        String(32),
        nullable=False,
        default="SUBMITTED",
    )  # SUBMITTED | VERIFIED | ACKNOWLEDGED | DISPATCHED | IN_PROGRESS | RESOLVED | REJECTED
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)
    hotspot_id = Column(Integer, ForeignKey("hotspots.id"), nullable=True, index=True)

    citizen = relationship("User", back_populates="incidents")
    hotspot = relationship("Hotspot", back_populates="incidents")
    report_detections = relationship(
        "ReportDetection", back_populates="report", cascade="all, delete-orphan"
    )
    hotspot_links = relationship(
        "HotspotReport", back_populates="report", cascade="all, delete-orphan"
    )
    audit_logs = relationship(
        "AuditLog", back_populates="report", cascade="all, delete-orphan"
    )

    @property
    def public_report_id(self) -> str:
        """Standardized public report identifier (e.g., CP-2026-000123)."""
        return self.incident_uuid


# Alias Incident as Report so modules using either name work seamlessly
Report = Incident


class ReportDetection(Base):
    """Normalized AI bounding-box detection belonging to a citizen waste report."""

    __tablename__ = "report_detections"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("incidents.id"), nullable=False, index=True)
    category = Column(
        String(64), nullable=False
    )  # Plastic Waste | Organic Waste | Dry Leaves / Biomass
    confidence = Column(Float, nullable=False, default=0.0)
    bounding_box_x1 = Column(Integer, nullable=False, default=0)
    bounding_box_y1 = Column(Integer, nullable=False, default=0)
    bounding_box_x2 = Column(Integer, nullable=False, default=0)
    bounding_box_y2 = Column(Integer, nullable=False, default=0)
    object_count = Column(Integer, nullable=False, default=1)
    area_ratio = Column(Float, nullable=False, default=0.0)

    report = relationship("Incident", back_populates="report_detections")


class HotspotReport(Base):
    """
    Relational join table linking a Hotspot to its constituent Citizen Reports.
    Tracks whether each report is the PRIMARY report or a DUPLICATE / CLUSTERED
    submission and stores its exact Haversine distance from the hotspot centroid.
    """

    __tablename__ = "hotspot_reports"
    __table_args__ = (
        Index("ix_hotspot_reports_hotspot_id", "hotspot_id"),
        Index("ix_hotspot_reports_report_id", "report_id"),
    )

    id = Column(Integer, primary_key=True, index=True)
    hotspot_id = Column(Integer, ForeignKey("hotspots.id"), nullable=False)
    report_id = Column(Integer, ForeignKey("incidents.id"), nullable=False)
    is_primary = Column(Boolean, nullable=False, default=False)
    distance_from_centroid = Column(Float, nullable=False, default=0.0)

    hotspot = relationship("Hotspot", back_populates="hotspot_reports")
    report = relationship("Incident", back_populates="hotspot_links")


class Dispatch(Base):
    """Municipal sanitation crew dispatch record linked to a unified hotspot."""

    __tablename__ = "dispatches"

    id = Column(Integer, primary_key=True, index=True)
    hotspot_id = Column(Integer, ForeignKey("hotspots.id"), nullable=False, index=True)
    officer_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    crew_name = Column(String(255), nullable=False)
    vehicle_number = Column(String(64), nullable=True)
    status = Column(
        String(32), nullable=False, default="DISPATCHED"
    )  # DISPATCHED | IN_PROGRESS | COMPLETED
    dispatched_at = Column(DateTime, default=utc_now, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)

    hotspot = relationship("Hotspot", back_populates="dispatches")
    officer = relationship("User", back_populates="dispatches")


class AuditLog(Base):
    """Immutable audit log tracking every report, hotspot, and dispatch state transition."""

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    actor_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    actor_role = Column(String(32), nullable=True, default="SYSTEM")
    report_id = Column(Integer, ForeignKey("incidents.id"), nullable=True, index=True)
    hotspot_id = Column(Integer, ForeignKey("hotspots.id"), nullable=True, index=True)
    action = Column(String(128), nullable=False)
    old_status = Column(String(64), nullable=True)
    new_status = Column(String(64), nullable=True)
    notes = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=utc_now, nullable=False, index=True)
    metadata_json = Column(Text, nullable=True)

    actor = relationship("User", back_populates="audit_logs", foreign_keys=[actor_id])
    user = relationship("User", foreign_keys=[user_id], overlaps="actor,audit_logs")
    report = relationship("Incident", back_populates="audit_logs")
    hotspot = relationship("Hotspot", back_populates="audit_logs")


def run_schema_migrations() -> None:
    """
    Create any missing tables and non-destructively add new columns/indexes to
    existing SQLite/PostgreSQL tables so existing data is preserved.
    """
    Base.metadata.create_all(bind=engine)

    alter_statements = [
        ("users", "name", "VARCHAR(150)"),
        ("users", "phone", "VARCHAR(40)"),
        ("users", "is_active", "BOOLEAN DEFAULT 1"),
        ("users", "updated_at", "DATETIME"),
        ("incidents", "thumbnail_path", "VARCHAR(512)"),
        ("incidents", "original_filename", "VARCHAR(255)"),
        ("incidents", "address", "VARCHAR(512)"),
        ("incidents", "description", "TEXT"),
        ("hotspots", "duplicate_count", "INTEGER DEFAULT 0"),
        ("dispatches", "vehicle_number", "VARCHAR(64)"),
        ("dispatches", "started_at", "DATETIME"),
        ("audit_logs", "actor_id", "INTEGER"),
        ("audit_logs", "actor_role", "VARCHAR(32)"),
        ("audit_logs", "notes", "TEXT"),
    ]

    with engine.begin() as conn:
        if DATABASE_URL.startswith("sqlite"):
            for table_name, col_name, col_type in alter_statements:
                try:
                    rows = conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
                    existing_cols = {r[1] for r in rows}
                    if col_name not in existing_cols:
                        conn.execute(
                            text(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}")
                        )
                        logger.info("Migrated column %s.%s", table_name, col_name)
                except Exception as exc:
                    logger.debug("Migration check for %s.%s skipped: %s", table_name, col_name, exc)

        # Ensure performance indexes exist on SQLite and PostgreSQL
        index_statements = [
            "CREATE INDEX IF NOT EXISTS ix_incidents_status ON incidents (status)",
            "CREATE INDEX IF NOT EXISTS ix_incidents_created_at ON incidents (created_at)",
            "CREATE INDEX IF NOT EXISTS ix_incidents_lat_lon ON incidents (latitude, longitude)",
            "CREATE INDEX IF NOT EXISTS ix_hotspots_status ON hotspots (status)",
            "CREATE INDEX IF NOT EXISTS ix_hotspot_reports_hotspot_id ON hotspot_reports (hotspot_id)",
            "CREATE INDEX IF NOT EXISTS ix_hotspot_reports_report_id ON hotspot_reports (report_id)",
        ]
        for stmt in index_statements:
            try:
                conn.execute(text(stmt))
            except Exception:
                pass


# Backward-compatible alias used across main.py, seed_demo.py, and test_cleanpath.py
init_db = run_schema_migrations


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a SQLAlchemy database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
