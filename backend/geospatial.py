"""
DBSCAN Geospatial Clustering, Hotspot Deduplication, and Reverse Geocoding
Engine for CleanPath AI / TerraSight AI.

Uses scikit-learn's DBSCAN with Haversine distance metric and a configurable
50-meter epsilon radius (eps = 50.0 / 6371000.0 radians, min_samples = 1)
to aggregate nearby citizen waste complaints into unified municipal dispatch
hotspots. Tracks PRIMARY vs DUPLICATE/CLUSTERED relationships in the
HotspotReport table without deleting any citizen submissions.
"""

import json
import logging
import math
import os
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.cluster import DBSCAN
from sqlalchemy.orm import Session

from database import Dispatch, Hotspot, HotspotReport, Incident, utc_now
from vision_engine import determine_severity_tier

logger = logging.getLogger("cleanpath.geospatial")

EARTH_RADIUS_METERS: float = 6371000.0
DEFAULT_CLUSTER_RADIUS_METERS: float = float(
    os.getenv("HOTSPOT_RADIUS_METERS", "50.0")
)
SEVERITY_TIER_PRIORITY: Dict[str, int] = {
    "High": 3,
    "Medium": 2,
    "Low": 1,
}

# In-memory cache for reverse geocoding so repeated coordinates resolve instantaneously
_GEOCODE_CACHE: Dict[str, str] = {}

# Known municipal reference landmarks for instant offline fallback
_KNOWN_MUNICIPAL_ZONES = [
    (28.6139, 77.2090, 15000.0, "Central Municipal Zone, New Delhi, Delhi, India"),
    (19.0760, 72.8777, 18000.0, "Brihanmumbai Municipal Ward, Mumbai, Maharashtra, India"),
    (12.9716, 77.5946, 18000.0, "BBMP Central Zone, Bengaluru, Karnataka, India"),
    (13.0827, 80.2707, 18000.0, "Greater Chennai Corporation Zone, Chennai, Tamil Nadu, India"),
    (22.5726, 88.3639, 18000.0, "Kolkata Municipal Corporation Sector, Kolkata, West Bengal, India"),
    (17.3850, 78.4867, 18000.0, "GHMC Municipal Ward, Hyderabad, Telangana, India"),
    (18.5204, 73.8567, 15000.0, "PMC Civic Sector, Pune, Maharashtra, India"),
    (23.0225, 72.5714, 15000.0, "AMC Municipal Zone, Ahmedabad, Gujarat, India"),
    (26.9124, 75.7873, 15000.0, "JMC Heritage Ward, Jaipur, Rajasthan, India"),
    (26.8467, 80.9462, 15000.0, "LMC Civic Zone, Lucknow, Uttar Pradesh, India"),
]


def haversine_distance_meters(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    Calculate the great-circle distance in meters between two geographic points
    using the Haversine formula.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return EARTH_RADIUS_METERS * c


def reverse_geocode_coordinates(
    latitude: Optional[float],
    longitude: Optional[float],
    allow_network: bool = True,
) -> Dict[str, Any]:
    """
    Resolve latitude and longitude into a human-readable municipal street/ward address.
    Uses OpenStreetMap Nominatim with a fast timeout and deterministic local fallback
    so report submission is never blocked when offline.
    """
    if latitude is None or longitude is None:
        return {
            "address": "",
            "latitude": None,
            "longitude": None,
            "source": "NONE",
        }

    lat_f = float(latitude)
    lon_f = float(longitude)
    cache_key = f"{lat_f:.4f},{lon_f:.4f}"
    if cache_key in _GEOCODE_CACHE:
        return {
            "address": _GEOCODE_CACHE[cache_key],
            "latitude": lat_f,
            "longitude": lon_f,
            "source": "CACHE",
        }

    if allow_network and os.getenv("DISABLE_NOMINATIM", "false").lower() != "true":
        try:
            url = (
                f"https://nominatim.openstreetmap.org/reverse"
                f"?format=jsonv2&lat={lat_f:.6f}&lon={lon_f:.6f}&zoom=17"
            )
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "CleanPathAI-MunicipalCivicPlatform/2.0"},
            )
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    payload = json.loads(resp.read().decode("utf-8"))
                    display_name = payload.get("display_name")
                    if display_name and isinstance(display_name, str):
                        parts = [p.strip() for p in display_name.split(",") if p.strip()]
                        short_addr = ", ".join(parts[:4]) if len(parts) > 4 else display_name
                        _GEOCODE_CACHE[cache_key] = short_addr
                        return {
                            "address": short_addr,
                            "latitude": lat_f,
                            "longitude": lon_f,
                            "source": "NOMINATIM",
                        }
        except Exception as exc:
            logger.debug("Nominatim lookup fallback triggered: %s", exc)

    # Deterministic offline municipal zone lookup
    for z_lat, z_lon, max_dist, zone_label in _KNOWN_MUNICIPAL_ZONES:
        dist = haversine_distance_meters(lat_f, lon_f, z_lat, z_lon)
        if dist <= max_dist:
            sector_num = (int(abs(lat_f * 1000) + abs(lon_f * 1000)) % 12) + 1
            fallback_addr = f"Ward Sector {sector_num}, {zone_label}"
            _GEOCODE_CACHE[cache_key] = fallback_addr
            return {
                "address": fallback_addr,
                "latitude": lat_f,
                "longitude": lon_f,
                "source": "MUNICIPAL_INDEX",
            }

    lat_hem = "N" if lat_f >= 0 else "S"
    lon_hem = "E" if lon_f >= 0 else "W"
    generic_addr = (
        f"Municipal Sector ({abs(lat_f):.4f}°{lat_hem}, {abs(lon_f):.4f}°{lon_hem})"
    )
    _GEOCODE_CACHE[cache_key] = generic_addr
    return {
        "address": generic_addr,
        "latitude": lat_f,
        "longitude": lon_f,
        "source": "COORDINATE_SECTOR",
    }


def cluster_incidents_data(
    items: Sequence[Dict[str, Any]],
    radius_meters: float = DEFAULT_CLUSTER_RADIUS_METERS,
) -> List[Dict[str, Any]]:
    """
    Pure functional DBSCAN clustering over a sequence of incident dicts.
    Each item must contain at least 'latitude', 'longitude', and optionally
    'id' and 'severity_score'.

    Returns a list of hotspot cluster dictionaries sorted by priority queue rules:
        1. Severity tier (High -> Medium -> Low)
        2. Maximum severity score descending
        3. Report count (density) descending
    """
    if not items:
        return []

    coords = np.array(
        [[float(item["latitude"]), float(item["longitude"])] for item in items],
        dtype=np.float64,
    )
    coords_rad = np.radians(coords)

    eps_radians = float(radius_meters) / EARTH_RADIUS_METERS
    dbscan = DBSCAN(
        eps=eps_radians,
        min_samples=1,
        metric="haversine",
    )
    labels = dbscan.fit_predict(coords_rad)

    clusters_map: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        cluster_id = int(label)
        clusters_map.setdefault(cluster_id, []).append(idx)

    hotspots: List[Dict[str, Any]] = []
    for cluster_idx, member_indices in clusters_map.items():
        member_items = [items[i] for i in member_indices]
        mean_lat = float(np.mean([float(m["latitude"]) for m in member_items]))
        mean_lon = float(np.mean([float(m["longitude"]) for m in member_items]))
        base_max_sev = float(
            max([float(m.get("severity_score", 0.0)) for m in member_items])
        )
        rep_count = len(member_items)
        dup_count = max(0, rep_count - 1)
        # Repeat report density boost (+2.5% per additional clustered report, capped at 100)
        boosted_sev = min(100.0, round(base_max_sev + (dup_count * 2.5), 2))
        tier = determine_severity_tier(boosted_sev)

        incident_ids = [
            m.get("id") for m in member_items if m.get("id") is not None
        ]

        hotspots.append(
            {
                "cluster_label": cluster_idx,
                "latitude": round(mean_lat, 6),
                "longitude": round(mean_lon, 6),
                "report_count": rep_count,
                "duplicate_count": dup_count,
                "maximum_severity": round(base_max_sev, 2),
                "boosted_severity": boosted_sev,
                "severity_tier": tier,
                "radius_meters": int(radius_meters)
                if float(radius_meters).is_integer()
                else radius_meters,
                "member_indices": member_indices,
                "incident_ids": incident_ids,
                "primary_incident_id": incident_ids[0] if incident_ids else None,
            }
        )

    hotspots.sort(
        key=lambda h: (
            SEVERITY_TIER_PRIORITY.get(h["severity_tier"], 0),
            float(h["maximum_severity"]),
            int(h["report_count"]),
        ),
        reverse=True,
    )
    return hotspots


def parse_categories_json(raw_categories: Optional[str]) -> List[str]:
    """Parse stored JSON categories string into a list of unique category names."""
    if not raw_categories:
        return []
    try:
        parsed = json.loads(raw_categories)
        if isinstance(parsed, list):
            result: List[str] = []
            for entry in parsed:
                if isinstance(entry, str) and entry not in result:
                    result.append(entry)
                elif isinstance(entry, dict) and "class_name" in entry:
                    cname = str(entry["class_name"])
                    if cname not in result:
                        result.append(cname)
            return result
        if isinstance(parsed, dict):
            cats = parsed.get("categories") or parsed.get("detected_categories") or list(parsed.keys())
            return [str(c) for c in cats]
    except Exception:
        pass
    return [s.strip() for s in raw_categories.split(",") if s.strip()]


def recompute_hotspots(
    db: Session,
    radius_meters: float = DEFAULT_CLUSTER_RADIUS_METERS,
) -> List[Hotspot]:
    """
    Run 50-meter Haversine DBSCAN across all non-rejected incidents in the database,
    create/update Hotspot entities, link every Incident to its unified Hotspot,
    maintain HotspotReport records (PRIMARY vs DUPLICATE/CLUSTERED + distance_from_centroid),
    and preserve existing dispatch records and statuses.
    """
    active_incidents: List[Incident] = (
        db.query(Incident)
        .filter(Incident.status != "REJECTED")
        .order_by(Incident.created_at.asc(), Incident.id.asc())
        .all()
    )

    if not active_incidents:
        # Remove any empty hotspots without dispatches when there are 0 active incidents
        for old_h in db.query(Hotspot).all():
            has_disp = (
                db.query(Dispatch).filter(Dispatch.hotspot_id == old_h.id).count() > 0
            )
            if not has_disp:
                db.delete(old_h)
        db.commit()
        return []

    coords = np.array(
        [[inc.latitude, inc.longitude] for inc in active_incidents],
        dtype=np.float64,
    )
    coords_rad = np.radians(coords)

    dbscan = DBSCAN(
        eps=float(radius_meters) / EARTH_RADIUS_METERS,
        min_samples=1,
        metric="haversine",
    )
    labels = dbscan.fit_predict(coords_rad)

    clusters: Dict[int, List[Incident]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(active_incidents[idx])

    existing_hotspots: List[Hotspot] = db.query(Hotspot).all()
    existing_by_id: Dict[int, Hotspot] = {h.id: h for h in existing_hotspots}
    used_hotspot_ids: set[int] = set()
    updated_hotspots: List[Hotspot] = []

    # Clear and rebuild HotspotReport links within the transaction
    db.query(HotspotReport).delete()

    for _, cluster_incidents in clusters.items():
        # Order incidents chronologically so the earliest report is PRIMARY
        cluster_incidents.sort(key=lambda inc: (inc.created_at or datetime.min.replace(tzinfo=timezone.utc), inc.id))

        mean_lat = round(
            float(np.mean([inc.latitude for inc in cluster_incidents])), 6
        )
        mean_lon = round(
            float(np.mean([inc.longitude for inc in cluster_incidents])), 6
        )
        report_count = len(cluster_incidents)
        duplicate_count = max(0, report_count - 1)

        base_max_severity = float(max(inc.severity_score for inc in cluster_incidents))
        # Factor hotspot repetition slightly into cluster severity when multiple reports merge
        max_severity = round(min(100.0, base_max_severity), 2)
        tier = determine_severity_tier(max_severity)

        matched_hotspot: Optional[Hotspot] = None
        for inc in cluster_incidents:
            if (
                inc.hotspot_id is not None
                and inc.hotspot_id in existing_by_id
                and inc.hotspot_id not in used_hotspot_ids
            ):
                matched_hotspot = existing_by_id[inc.hotspot_id]
                break

        if matched_hotspot is None:
            for candidate in existing_hotspots:
                if candidate.id in used_hotspot_ids:
                    continue
                dist = haversine_distance_meters(
                    mean_lat, mean_lon, candidate.latitude, candidate.longitude
                )
                if dist <= radius_meters:
                    matched_hotspot = candidate
                    break

        incident_statuses = {inc.status for inc in cluster_incidents}
        if incident_statuses == {"RESOLVED"}:
            derived_status = "RESOLVED"
        elif "IN_PROGRESS" in incident_statuses:
            derived_status = "IN_PROGRESS"
        elif "DISPATCHED" in incident_statuses:
            derived_status = "DISPATCHED"
        elif "ACKNOWLEDGED" in incident_statuses or "VERIFIED" in incident_statuses:
            derived_status = "ACKNOWLEDGED"
        else:
            derived_status = "PENDING"

        if matched_hotspot is None:
            matched_hotspot = Hotspot(
                latitude=mean_lat,
                longitude=mean_lon,
                report_count=report_count,
                duplicate_count=duplicate_count,
                maximum_severity=max_severity,
                severity_tier=tier,
                radius_meters=radius_meters,
                status=derived_status,
            )
            db.add(matched_hotspot)
            db.flush()
        else:
            matched_hotspot.latitude = mean_lat
            matched_hotspot.longitude = mean_lon
            matched_hotspot.report_count = report_count
            matched_hotspot.duplicate_count = duplicate_count
            matched_hotspot.maximum_severity = max_severity
            matched_hotspot.severity_tier = tier
            matched_hotspot.radius_meters = radius_meters
            if matched_hotspot.status == "PENDING" or derived_status != "PENDING":
                matched_hotspot.status = derived_status
            matched_hotspot.updated_at = utc_now()

        used_hotspot_ids.add(matched_hotspot.id)

        for idx_in_cluster, inc in enumerate(cluster_incidents):
            inc.hotspot_id = matched_hotspot.id
            dist_from_center = round(
                haversine_distance_meters(
                    mean_lat, mean_lon, inc.latitude, inc.longitude
                ),
                2,
            )
            hr_link = HotspotReport(
                hotspot_id=matched_hotspot.id,
                report_id=inc.id,
                is_primary=(idx_in_cluster == 0),
                distance_from_centroid=dist_from_center,
            )
            db.add(hr_link)

        updated_hotspots.append(matched_hotspot)

    for old_hotspot in existing_hotspots:
        if old_hotspot.id not in used_hotspot_ids:
            has_dispatches = (
                db.query(Dispatch)
                .filter(Dispatch.hotspot_id == old_hotspot.id)
                .count()
                > 0
            )
            if not has_dispatches:
                db.delete(old_hotspot)

    db.commit()

    # Sort according to Priority Queue rules (Section 26):
    # 1. HIGH -> MEDIUM -> LOW
    # 2. Maximum severity score descending
    # 3. Report count (hotspot density) descending
    # 4. Oldest unresolved hotspot first
    updated_hotspots.sort(
        key=lambda h: (
            SEVERITY_TIER_PRIORITY.get(h.severity_tier, 0),
            float(h.maximum_severity),
            int(h.report_count),
            -(h.created_at.timestamp() if h.created_at else 0.0),
        ),
        reverse=True,
    )
    return updated_hotspots
