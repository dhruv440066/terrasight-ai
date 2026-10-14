"""
DBSCAN Geospatial Clustering & Hotspot Intelligence Engine
==========================================================
Groups spatially adjacent citizen and drone waste sightings into discrete
actionable hotspots using scikit-learn's DBSCAN with Haversine great-circle distance.
Calculates cluster centroids, repeat report counts, and priority metrics.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

EARTH_RADIUS_METERS: float = 6371008.8
EARTH_RADIUS_KM: float = 6371.0088


def cluster_incidents_dbscan(
    incidents: List[Dict[str, Any]],
    epsilon_meters: float = 50.0,
    min_samples: int = 1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Cluster spatial sightings using Haversine DBSCAN.
    
    Args:
        incidents: List of dicts, each with 'latitude', 'longitude', and optional detection info.
        epsilon_meters: Clustering distance threshold in meters (default 50m).
        min_samples: Minimum points required to form a core DBSCAN cluster.
        
    Returns:
        Tuple of:
            - updated_incidents: Copy of incidents with assigned 'cluster_id' and 'is_hotspot'
            - hotspots: Aggregated cluster summaries (centroids, counts, dominant category)
    """
    if not incidents:
        return [], []

    updated = [dict(item) for item in incidents]

    # Filter out incidents missing valid coordinates
    valid_indices = []
    coords_list = []
    for idx, inc in enumerate(updated):
        lat = inc.get("latitude")
        lon = inc.get("longitude")
        if lat is not None and lon is not None:
            try:
                coords_list.append([float(lat), float(lon)])
                valid_indices.append(idx)
            except (ValueError, TypeError):
                inc["cluster_id"] = -1
                inc["is_hotspot"] = False
        else:
            inc["cluster_id"] = -1
            inc["is_hotspot"] = False

    if not coords_list:
        return updated, []

    # Convert coordinates to radians: [lat, lon] -> [radians(lat), radians(lon)]
    coords_deg = np.array(coords_list, dtype=np.float64)
    coords_rad = np.radians(coords_deg)

    # Convert epsilon meters to radians
    eps_rad = float(epsilon_meters) / EARTH_RADIUS_METERS

    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
    labels = db.fit_predict(coords_rad)

    cluster_groups: Dict[int, List[int]] = {}
    for local_idx, label in enumerate(labels):
        orig_idx = valid_indices[local_idx]
        cluster_id = int(label)
        updated[orig_idx]["cluster_id"] = cluster_id
        cluster_groups.setdefault(cluster_id, []).append(orig_idx)

    hotspots: List[Dict[str, Any]] = []

    for cluster_id, member_indices in cluster_groups.items():
        member_items = [updated[i] for i in member_indices]
        lats = [float(item["latitude"]) for item in member_items]
        lons = [float(item["longitude"]) for item in member_items]

        centroid_lat = round(float(np.mean(lats)), 6)
        centroid_lon = round(float(np.mean(lons)), 6)
        count = len(member_items)

        # Flag as hotspot if clustered (>=2 reports) or if single high-density point
        is_hotspot = count >= 2 or cluster_id >= 0

        for i in member_indices:
            updated[i]["is_hotspot"] = is_hotspot
            updated[i]["cluster_centroid"] = (centroid_lat, centroid_lon)

        # Tally categories
        cat_counts: Dict[str, int] = {}
        total_detections = 0
        for item in member_items:
            det_cats = item.get("category_counts") or {}
            for c, cnt in det_cats.items():
                cat_counts[c] = cat_counts.get(c, 0) + cnt
                total_detections += cnt

        dominant_cat = (
            max(cat_counts.items(), key=lambda x: x[1])[0]
            if cat_counts
            else member_items[0].get("primary_category", "Waste")
        )

        # Calculate severity (0 - 100 scale)
        base_sev = sum(item.get("severity_score", 45.0) for item in member_items) / max(1, count)
        boosted_sev = min(100.0, round(base_sev + (max(0, count - 1) * 3.5), 1))

        if boosted_sev >= 65:
            severity_tier = "High"
        elif boosted_sev >= 35:
            severity_tier = "Medium"
        else:
            severity_tier = "Low"

        # Hotspot summary dictionary
        hotspots.append({
            "cluster_id": cluster_id,
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "report_count": count,
            "duplicate_count": max(0, count - 1),
            "dominant_category": dominant_cat,
            "category_breakdown": cat_counts,
            "severity_score": boosted_sev,
            "severity_tier": severity_tier,
            "radius_meters": epsilon_meters,
            "member_indices": member_indices,
            "is_hotspot": is_hotspot,
        })

    # Sort hotspots: High severity and high report counts first
    hotspots.sort(
        key=lambda h: (
            {"High": 3, "Medium": 2, "Low": 1}.get(h["severity_tier"], 0),
            h["report_count"],
            h["severity_score"]
        ),
        reverse=True
    )

    return updated, hotspots


def compute_cluster_metrics(
    incidents: List[Dict[str, Any]],
    hotspots: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute executive analytics and KPI metrics for dashboard display.
    """
    total_reports = len(incidents)
    active_hotspots = len([h for h in hotspots if h.get("report_count", 0) >= 2])
    total_duplicates = sum(h.get("duplicate_count", 0) for h in hotspots)

    suppression_rate = (
        round((total_duplicates / total_reports) * 100, 1)
        if total_reports > 0
        else 0.0
    )

    # Category tally across all incidents
    all_cats: Dict[str, int] = {}
    for inc in incidents:
        for cat, cnt in (inc.get("category_counts") or {}).items():
            all_cats[cat] = all_cats.get(cat, 0) + cnt

    top_cat = (
        max(all_cats.items(), key=lambda x: x[1])[0]
        if all_cats
        else "None"
    )

    return {
        "total_reports": total_reports,
        "active_hotspots": active_hotspots,
        "total_clusters": len(hotspots),
        "suppression_rate": suppression_rate,
        "top_category": top_cat,
        "category_distribution": all_cats,
    }
