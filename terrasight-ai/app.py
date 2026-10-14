"""
TerraSight AI — AI-Powered Civic Issue & Waste Monitoring Platform
==================================================================
Streamlit Production Application with:
1. EXIF GPS and capture timestamp extraction (with interactive manual fallback)
2. YOLOv8 object detection with cached weights and user-tunable thresholds
3. DBSCAN 50m Haversine spatial clustering for duplicate suppression and hotspot discovery
4. Interactive Folium GIS mapping with cluster radius circles and annotated popups
5. Executive analytics, KPI metrics, and CSV/GeoJSON export
"""

import base64
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import folium
from folium import plugins
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_folium import st_folium

# Utilities
from utils.clustering import cluster_incidents_dbscan, compute_cluster_metrics
from utils.detector import load_yolo_model, run_detection
from utils.exif_extractor import extract_exif_metadata

# Page Configuration
st.set_page_config(
    page_title="TerraSight AI — Civic Waste & Hotspot Monitoring",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Glassmorphic UI with vibrant accents)
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #10B981 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        background-color: #1E293B;
        color: #94A3B8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #16A34A !important;
        color: #FFFFFF !important;
    }
    .status-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-green { background: #064E3B; color: #34D399; border: 1px solid #059669; }
    .badge-blue { background: #1E3A8A; color: #60A5FA; border: 1px solid #2563EB; }
    .badge-orange { background: #7C2D12; color: #FB923C; border: 1px solid #EA580C; }
    .badge-red { background: #7F1D1D; color: #F87171; border: 1px solid #DC2626; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Sample Civic Data Generator for Instant One-Click Demonstration
def get_sample_delhi_incidents() -> List[Dict[str, Any]]:
    """Generate realistic demonstration incidents clustered across New Delhi."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    samples = [
        # Hotspot 1: Connaught Place Outer Circle (3 reports within ~25m -> Cluster 0)
        {
            "id": "TS-2026-001",
            "latitude": 28.63150,
            "longitude": 77.21670,
            "timestamp": now_str,
            "source": "EXIF GPS",
            "primary_category": "Plastic Waste",
            "category_counts": {"Plastic Waste": 4, "Organic Waste": 1},
            "detection_count": 5,
            "severity_score": 84.5,
            "severity_tier": "High",
            "notes": "Severe accumulation of single-use bottles and packaging outside Metro Gate 2.",
            "thumbnail_b64": None,
        },
        {
            "id": "TS-2026-002",
            "latitude": 28.63162,
            "longitude": 77.21682,
            "timestamp": now_str,
            "source": "EXIF GPS",
            "primary_category": "Plastic Waste",
            "category_counts": {"Plastic Waste": 3, "Dry Leaves / Biomass": 1},
            "detection_count": 4,
            "severity_score": 76.2,
            "severity_tier": "High",
            "notes": "Secondary report: Vendor waste spilling onto pedestrian sidewalk.",
            "thumbnail_b64": None,
        },
        {
            "id": "TS-2026-003",
            "latitude": 28.63142,
            "longitude": 77.21661,
            "timestamp": now_str,
            "source": "Manual Pin",
            "primary_category": "Organic Waste",
            "category_counts": {"Organic Waste": 3, "Plastic Waste": 2},
            "detection_count": 5,
            "severity_score": 81.0,
            "severity_tier": "High",
            "notes": "Food waste containers mixed with beverage bottles.",
            "thumbnail_b64": None,
        },
        # Hotspot 2: Karol Bagh Market (3 reports within ~20m -> Cluster 1)
        {
            "id": "TS-2026-004",
            "latitude": 28.65190,
            "longitude": 77.19090,
            "timestamp": now_str,
            "source": "EXIF GPS",
            "primary_category": "Organic Waste",
            "category_counts": {"Organic Waste": 5, "Plastic Waste": 2},
            "detection_count": 7,
            "severity_score": 79.0,
            "severity_tier": "High",
            "notes": "Market vegetable residue and discard plastic packaging.",
            "thumbnail_b64": None,
        },
        {
            "id": "TS-2026-005",
            "latitude": 28.65201,
            "longitude": 77.19102,
            "timestamp": now_str,
            "source": "EXIF GPS",
            "primary_category": "Organic Waste",
            "category_counts": {"Organic Waste": 4},
            "detection_count": 4,
            "severity_score": 68.5,
            "severity_tier": "High",
            "notes": "Fruit stalls disposal pile near drainage culvert.",
            "thumbnail_b64": None,
        },
        {
            "id": "TS-2026-006",
            "latitude": 28.65184,
            "longitude": 77.19082,
            "timestamp": now_str,
            "source": "Manual Pin",
            "primary_category": "Plastic Waste",
            "category_counts": {"Plastic Waste": 3, "Organic Waste": 2},
            "detection_count": 5,
            "severity_score": 72.4,
            "severity_tier": "High",
            "notes": "Overnight garbage spillover on road bend.",
            "thumbnail_b64": None,
        },
        # Hotspot 3: Chandni Chowk (2 reports within ~30m -> Cluster 2)
        {
            "id": "TS-2026-007",
            "latitude": 28.65060,
            "longitude": 77.23030,
            "timestamp": now_str,
            "source": "EXIF GPS",
            "primary_category": "Dry Leaves / Biomass",
            "category_counts": {"Dry Leaves / Biomass": 3, "Plastic Waste": 1},
            "detection_count": 4,
            "severity_score": 52.0,
            "severity_tier": "Medium",
            "notes": "Dry garden clippings and wrappers near historical walkway.",
            "thumbnail_b64": None,
        },
        {
            "id": "TS-2026-008",
            "latitude": 28.65072,
            "longitude": 77.23041,
            "timestamp": now_str,
            "source": "Manual Pin",
            "primary_category": "Dry Leaves / Biomass",
            "category_counts": {"Dry Leaves / Biomass": 4},
            "detection_count": 4,
            "severity_score": 46.5,
            "severity_tier": "Medium",
            "notes": "Street sweeping pile awaiting municipal tractor collection.",
            "thumbnail_b64": None,
        },
        # Isolated Incident 1: India Gate Lawns
        {
            "id": "TS-2026-009",
            "latitude": 28.61290,
            "longitude": 77.22950,
            "timestamp": now_str,
            "source": "EXIF GPS",
            "primary_category": "Plastic Waste",
            "category_counts": {"Plastic Waste": 2},
            "detection_count": 2,
            "severity_score": 38.0,
            "severity_tier": "Medium",
            "notes": "Scattered beverage bottles along tourist pathway.",
            "thumbnail_b64": None,
        },
        # Isolated Incident 2: Lodhi Garden Edge
        {
            "id": "TS-2026-010",
            "latitude": 28.59210,
            "longitude": 77.20460,
            "timestamp": now_str,
            "source": "Manual Pin",
            "primary_category": "Dry Leaves / Biomass",
            "category_counts": {"Dry Leaves / Biomass": 2},
            "detection_count": 2,
            "severity_score": 24.0,
            "severity_tier": "Low",
            "notes": "Dry organic leaf pile on lawn boundary.",
            "thumbnail_b64": None,
        },
    ]
    return samples


# Initialize Streamlit Session State
if "incidents" not in st.session_state:
    st.session_state["incidents"] = get_sample_delhi_incidents()

if "last_processed" not in st.session_state:
    st.session_state["last_processed"] = None


# Helper to convert PIL Image to base64 for HTML Folium popups
def pil_to_base64_thumbnail(img: Image.Image, max_dim: int = 180) -> str:
    """Generate lightweight base64 thumbnail string for popup rendering."""
    try:
        thumb = img.copy()
        thumb.thumbnail((max_dim, max_dim))
        buf = BytesIO()
        thumb.save(buf, format="JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return ""


# ==============================================================================
# SIDEBAR CONTROLS & PARAMETERS
# ==============================================================================
with st.sidebar:
    st.markdown("### 🌍 **TerraSight AI**")
    st.markdown(
        """
        <div style="font-size: 0.82rem; color: #94A3B8; margin-top: -8px; margin-bottom: 12px;">
        AI-Powered Civic Waste Detection & Spatial Hotspot Intelligence
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Model Initialization
    with st.spinner("Initializing YOLOv8 Engine..."):
        try:
            model_instance, model_desc = load_yolo_model("weights/best.pt")
            st.markdown(
                f'<span class="status-badge badge-green">✓ {model_desc}</span>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Model Error: {e}")
            model_instance = None

    st.markdown("---")
    st.markdown("#### ⚙️ **Inference Parameters**")
    conf_thresh = st.slider("Confidence Threshold", 0.10, 1.00, 0.25, 0.05)
    iou_thresh = st.slider("IoU NMS Threshold", 0.10, 0.90, 0.45, 0.05)

    st.markdown("---")
    st.markdown("#### 🛰️ **DBSCAN Clustering**")
    cluster_radius = st.slider(
        "Hotspot Radius (Epsilon Meters)", 10, 200, 50, 5,
        help="Haversine distance within which multiple sightings merge into a unified hotspot."
    )
    min_samples = st.slider(
        "Min Sightings for Cluster", 1, 5, 1,
        help="Minimum sightings needed to form a core DBSCAN cluster."
    )

    st.markdown("---")
    st.markdown("#### 🧪 **Data & Presets**")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("🔄 Reset Demo", use_container_width=True):
            st.session_state["incidents"] = get_sample_delhi_incidents()
            st.rerun()
    with col_s2:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state["incidents"] = []
            st.rerun()

    st.markdown("---")
    st.markdown("#### 📦 **Export Intelligence**")

    # Perform clustering over current incidents
    clustered_incidents, hotspots = cluster_incidents_dbscan(
        st.session_state["incidents"],
        epsilon_meters=cluster_radius,
        min_samples=min_samples,
    )

    if st.session_state["incidents"]:
        # Prepare CSV
        csv_df = pd.DataFrame(clustered_incidents)
        if "thumbnail_b64" in csv_df.columns:
            csv_df = csv_df.drop(columns=["thumbnail_b64"])
        csv_data = csv_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"terrasight_incidents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Prepare GeoJSON
        geojson_features = []
        for inc in clustered_incidents:
            if inc.get("latitude") is not None and inc.get("longitude") is not None:
                feat = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(inc["longitude"]), float(inc["latitude"])],
                    },
                    "properties": {
                        "id": inc.get("id"),
                        "timestamp": inc.get("timestamp"),
                        "category": inc.get("primary_category"),
                        "severity_score": inc.get("severity_score"),
                        "severity_tier": inc.get("severity_tier"),
                        "cluster_id": inc.get("cluster_id"),
                        "is_hotspot": inc.get("is_hotspot"),
                    },
                }
                geojson_features.append(feat)

        geojson_obj = {"type": "FeatureCollection", "features": geojson_features}
        st.download_button(
            label="🌍 Download GeoJSON",
            data=json.dumps(geojson_obj, indent=2),
            file_name=f"terrasight_hotspots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson",
            mime="application/geo+json",
            use_container_width=True,
        )


# ==============================================================================
# MAIN DASHBOARD CONTENT
# ==============================================================================
st.markdown('<div class="main-header">TerraSight AI Command Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AI-Powered Mobile & Drone Waste Detection, EXIF Geotagging, and DBSCAN Hotspot Prioritization</div>',
    unsafe_allow_html=True,
)

# Compute Top KPI Metrics
kpis = compute_cluster_metrics(st.session_state["incidents"], hotspots)

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{kpis['total_reports']}</div>
            <div class="metric-label">Total Sightings Ingested</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_m2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #F59E0B;">{kpis['active_hotspots']}</div>
            <div class="metric-label">Active Hotspots (>=2 Reports)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_m3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #10B981;">{kpis['suppression_rate']}%</div>
            <div class="metric-label">Duplicate Suppression Rate</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_m4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #60A5FA;">{kpis['top_category']}</div>
            <div class="metric-label">Top Detected Category</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Main Navigation Tabs
tab_map, tab_upload, tab_analytics = st.tabs(
    ["🗺️ Geospatial Command Map", "📸 Upload & AI Detection Lab", "📊 Hotspots & Priority Queue"]
)


# ==============================================================================
# TAB 1: GEOSPATIAL COMMAND MAP (Folium + DBSCAN Circles + Popups)
# ==============================================================================
with tab_map:
    st.markdown("#### 🌐 **Municipal GIS Hotspot Map**")
    st.caption(
        f"Displays individual sightings and DBSCAN cluster boundary circles (Radius = {cluster_radius}m). "
        "Markers are color-coded by category: 🔵 Blue = Plastic Waste | 🟠 Orange = Organic Waste | 🟢 Green = Biomass | 🟣 Purple = Other."
    )

    # Determine map center
    if st.session_state["incidents"]:
        valid_lats = [inc["latitude"] for inc in st.session_state["incidents"] if inc.get("latitude") is not None]
        valid_lons = [inc["longitude"] for inc in st.session_state["incidents"] if inc.get("longitude") is not None]
        center_lat = float(np.mean(valid_lats)) if valid_lats else 28.6139
        center_lon = float(np.mean(valid_lons)) if valid_lons else 77.2090
    else:
        center_lat, center_lon = 28.6139, 77.2090

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # 1. Render DBSCAN Cluster Circles (50m radius)
    for h in hotspots:
        if h.get("report_count", 0) >= 2 or h.get("is_hotspot"):
            c_lat = h["centroid_lat"]
            c_lon = h["centroid_lon"]
            r_meters = h["radius_meters"]
            count = h["report_count"]
            tier = h["severity_tier"]
            dom_cat = h["dominant_category"]

            circle_color = "#DC2626" if tier == "High" else ("#F59E0B" if tier == "Medium" else "#10B981")

            folium.Circle(
                location=[c_lat, c_lon],
                radius=r_meters,
                color=circle_color,
                weight=2,
                fill=True,
                fill_color=circle_color,
                fill_opacity=0.22,
                dash_array="6, 6",
                tooltip=f"<b>Hotspot Cluster #{h['cluster_id']}</b><br>Sightings: {count}<br>Category: {dom_cat}<br>Severity: {h['severity_score']}/100 ({tier})",
            ).add_to(m)

            # Center hotspot badge marker
            folium.map.Marker(
                [c_lat, c_lon],
                icon=folium.DivIcon(
                    html=f"""
                    <div style="
                        background-color: {circle_color};
                        color: white;
                        border-radius: 50%;
                        width: 28px;
                        height: 28px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 11px;
                        font-weight: 700;
                        box-shadow: 0 0 10px rgba(0,0,0,0.4);
                        border: 2px solid white;
                    ">{count}</div>
                    """
                ),
            ).add_to(m)

    # 2. Render Individual Incident Markers
    for inc in clustered_incidents:
        lat = inc.get("latitude")
        lon = inc.get("longitude")
        if lat is None or lon is None:
            continue

        cat = inc.get("primary_category", "Plastic Waste")
        color_map = {
            "Plastic Waste": "blue",
            "Organic Waste": "orange",
            "Dry Leaves / Biomass": "green",
            "Metal Debris": "purple",
        }
        marker_color = color_map.get(cat, "red")

        cluster_tag = (
            f"<span style='color: #EF4444; font-weight: bold;'>Cluster #{inc.get('cluster_id')} (HOTSPOT)</span>"
            if inc.get("is_hotspot")
            else "<span>Isolated Sighting</span>"
        )

        thumb_html = ""
        if inc.get("thumbnail_b64"):
            thumb_html = f"""
            <div style="margin-top: 8px; text-align: center;">
                <img src="data:image/jpeg;base64,{inc['thumbnail_b64']}" style="max-width: 170px; border-radius: 6px; border: 1px solid #CBD5E1;" />
            </div>
            """

        popup_content = f"""
        <div style="font-family: sans-serif; font-size: 12px; width: 190px;">
            <b style="font-size: 13px; color: #1E293B;">Incident {inc.get('id', 'N/A')}</b><br>
            <div style="margin: 4px 0;">{cluster_tag}</div>
            <b>Category:</b> {cat}<br>
            <b>Severity:</b> {inc.get('severity_score', 50.0)}/100 ({inc.get('severity_tier', 'Medium')})<br>
            <b>Detections:</b> {inc.get('detection_count', 1)}<br>
            <b>Coords:</b> {lat:.5f}, {lon:.5f}<br>
            <b>Source:</b> {inc.get('source', 'EXIF GPS')}<br>
            <span style="font-size: 10px; color: #64748B;">{inc.get('timestamp', '')}</span>
            {thumb_html}
        </div>
        """

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_content, max_width=240),
            tooltip=f"{inc.get('id', 'Incident')} — {cat} ({inc.get('severity_tier')})",
            icon=folium.Icon(color=marker_color, icon="trash", prefix="fa"),
        ).add_to(m)

    # Render Folium Map in Streamlit
    st_folium(m, width=None, height=520, use_container_width=True)


# ==============================================================================
# TAB 2: UPLOAD & AI DETECTION LAB
# ==============================================================================
with tab_upload:
    st.markdown("#### 📸 **Image Ingestion & Real-Time Computer Vision**")
    st.caption("Upload geotagged mobile or drone imagery. EXIF metadata will be automatically parsed with interactive manual coordinate fallback.")

    col_up1, col_up2 = st.columns([1.1, 0.9])

    with col_up1:
        uploaded_file = st.file_uploader(
            "Upload Waste Image (.jpg, .jpeg, .png)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            help="High-resolution mobile or drone camera photograph.",
        )

        if uploaded_file is not None:
            # Load PIL image
            pil_img = Image.open(uploaded_file).convert("RGB")

            # Extract EXIF metadata
            exif_data = extract_exif_metadata(uploaded_file)

            st.markdown("##### 📍 **Geotagging & Metadata Status**")

            if exif_data["has_gps"]:
                st.success(
                    f"✓ EXIF GPS Verified: Lat {exif_data['latitude']:.6f}, Lon {exif_data['longitude']:.6f}"
                )
                default_lat = exif_data["latitude"]
                default_lon = exif_data["longitude"]
                source_tag = "EXIF GPS"
            else:
                st.warning(
                    f"⚠️ {exif_data.get('error', 'No GPS tags found')}. Please verify coordinates below:"
                )
                default_lat = 28.6315
                default_lon = 77.2167
                source_tag = "Manual Input"

            col_lat, col_lon = st.columns(2)
            with col_lat:
                input_lat = st.number_input("Latitude (°N)", -90.0, 90.0, float(default_lat), format="%.6f")
            with col_lon:
                input_lon = st.number_input("Longitude (°E)", -180.0, 180.0, float(default_lon), format="%.6f")

            input_notes = st.text_input("Incident Notes / Citizen Remarks", placeholder="e.g. Overflowing garbage near park gate")

            if st.button("🚀 Run YOLOv8 Detection & Log Incident", type="primary", use_container_width=True):
                if model_instance is None:
                    st.error("YOLO model not loaded.")
                else:
                    with st.spinner("Executing YOLOv8 neural network inference..."):
                        annotated_img, detections, cat_counts = run_detection(
                            pil_img,
                            model_instance,
                            conf_threshold=conf_thresh,
                            iou_threshold=iou_thresh,
                        )

                        # Determine primary category and compute severity score
                        if cat_counts:
                            primary_cat = max(cat_counts.items(), key=lambda x: x[1])[0]
                        else:
                            primary_cat = "Plastic Waste"

                        det_count = len(detections) if detections else 1

                        # Severity calculation (0 - 100)
                        sev_weight = {"Plastic Waste": 18, "Organic Waste": 14, "Dry Leaves / Biomass": 10}.get(primary_cat, 12)
                        raw_sev = min(100.0, round(sev_weight * (1 + (det_count * 0.45)), 1))
                        tier = "High" if raw_sev >= 65 else ("Medium" if raw_sev >= 35 else "Low")

                        # Generate thumbnail
                        thumb_b64 = pil_to_base64_thumbnail(annotated_img)

                        # Create incident record
                        new_id = f"TS-{datetime.now().strftime('%Y')}-{len(st.session_state['incidents']) + 1:03d}"
                        incident_record = {
                            "id": new_id,
                            "latitude": float(input_lat),
                            "longitude": float(input_lon),
                            "timestamp": exif_data.get("timestamp") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                            "source": source_tag,
                            "primary_category": primary_cat,
                            "category_counts": cat_counts if cat_counts else {primary_cat: 1},
                            "detection_count": det_count,
                            "severity_score": raw_sev,
                            "severity_tier": tier,
                            "notes": input_notes or "Citizen reported waste sighting via TerraSight AI portal.",
                            "thumbnail_b64": thumb_b64,
                        }

                        # Persist to session
                        st.session_state["incidents"].append(incident_record)
                        st.session_state["last_processed"] = {
                            "original": pil_img,
                            "annotated": annotated_img,
                            "detections": detections,
                            "cat_counts": cat_counts,
                            "record": incident_record,
                        }

                        st.success(f"✓ Incident {new_id} detected and logged to database! Switching to view...")
                        st.rerun()

    with col_up2:
        st.markdown("##### 🔍 **Inference Preview**")
        if st.session_state["last_processed"]:
            lp = st.session_state["last_processed"]
            st.image(lp["annotated"], caption=f"YOLOv8 Output ({lp['record']['id']})", use_container_width=True)

            st.markdown(
                f"""
                **Detected:** {lp['record']['primary_category']} | **Objects:** {lp['record']['detection_count']} | **Severity:** {lp['record']['severity_score']}/100 ({lp['record']['severity_tier']})
                """
            )

            # Detection details table
            if lp["detections"]:
                det_df = pd.DataFrame(lp["detections"])[["label", "confidence", "box"]]
                st.dataframe(det_df, use_container_width=True, height=180)
        elif uploaded_file is not None:
            st.image(pil_img, caption="Uploaded Original Image", use_container_width=True)
        else:
            st.info("Upload an image on the left to see instant EXIF analysis and YOLOv8 neural bounding boxes.")


# ==============================================================================
# TAB 3: HOTSPOTS & PRIORITY QUEUE
# ==============================================================================
with tab_analytics:
    st.markdown("#### ⚡ **DBSCAN Waste Hotspot Priority Queue**")
    st.caption(
        f"Aggregates reports within {cluster_radius} meters into high-priority municipal dispatch hotspots."
    )

    if hotspots:
        hotspot_rows = []
        for h in hotspots:
            hotspot_rows.append({
                "Hotspot ID": f"HS-{h['cluster_id']:02d}" if h['cluster_id'] >= 0 else "Isolated",
                "Centroid Lat": h["centroid_lat"],
                "Centroid Lon": h["centroid_lon"],
                "Sightings Clustered": h["report_count"],
                "Duplicate Reports Suppressed": h["duplicate_count"],
                "Dominant Category": h["dominant_category"],
                "Severity Score": f"{h['severity_score']}/100",
                "Priority Tier": h["severity_tier"],
                "Status": "DISPATCH REQUIRED" if h["report_count"] >= 2 else "MONITORING",
            })

        hotspot_df = pd.DataFrame(hotspot_rows)
        st.dataframe(hotspot_df, use_container_width=True, hide_index=True)
    else:
        st.info("No hotspots generated yet. Upload images or click 'Reset Demo' in the sidebar.")

    st.markdown("---")
    st.markdown("#### 📋 **Ingested Incidents Ledger**")
    if st.session_state["incidents"]:
        inc_df = pd.DataFrame(st.session_state["incidents"])
        display_cols = [c for c in ["id", "timestamp", "latitude", "longitude", "source", "primary_category", "detection_count", "severity_score", "severity_tier", "notes"] if c in inc_df.columns]
        st.dataframe(inc_df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("Incidents table is currently empty.")
