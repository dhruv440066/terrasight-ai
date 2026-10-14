"""
Comprehensive Test Suite for CleanPath AI Backend.
Covers Authentication & RBAC, EXIF GPS Extraction, Mathematical Severity Engine,
50m Haversine DBSCAN Clustering, Normalized Relational Entities, Citizen Privacy
Isolation, and End-to-End FastAPI Endpoints.
"""

import io
import unittest

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from auth import seed_demo_users
from database import (
    AuditLog,
    Base,
    Dispatch,
    Hotspot,
    HotspotReport,
    Incident,
    ReportDetection,
    SessionLocal,
    engine,
    init_db,
)
from exif_parser import dms_to_decimal, extract_gps_from_image, parse_gps_info
from geospatial import cluster_incidents_data, haversine_distance_meters
from main import app
from seed_demo import ensure_default_users, purge_synthetic_demo_incidents
from vision_engine import calculate_severity, determine_severity_tier


def create_test_jpeg_bytes(width: int = 400, height: int = 300) -> bytes:
    """Create an in-memory valid JPEG image with synthetic waste objects."""
    img = Image.new("RGB", (width, height), color=(235, 240, 245))
    draw = ImageDraw.Draw(img)
    draw.rectangle([60, 60, 280, 220], fill=(25, 118, 210), outline=(10, 50, 120), width=4)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestCleanPathAI(unittest.TestCase):
    """End-to-end and unit test suite for CleanPath AI."""

    @classmethod
    def setUpClass(cls) -> None:
        init_db()
        db = SessionLocal()
        try:
            seed_demo_users(db)
            ensure_default_users(db)
        finally:
            db.close()
        cls.client = TestClient(app)

    def _get_token(self, username: str, password: str) -> str:
        res = self.client.post(
            "/auth/login",
            json={"username": username, "password": password},
        )
        self.assertEqual(res.status_code, 200, res.text)
        return res.json()["access_token"]

    # ==========================================================================
    # 1. AUTHENTICATION & RBAC TESTS
    # ==========================================================================

    def test_auth_valid_login_citizen_and_officer(self) -> None:
        """Test valid login for both demo citizen and demo officer accounts."""
        cit_res = self.client.post(
            "/auth/login",
            json={"username": "citizen", "password": "citizen123"},
        )
        self.assertEqual(cit_res.status_code, 200)
        cit_data = cit_res.json()
        self.assertIn("access_token", cit_data)
        self.assertEqual(cit_data["user"]["role"], "citizen")

        off_res = self.client.post(
            "/auth/login",
            json={"username": "officer", "password": "officer123"},
        )
        self.assertEqual(off_res.status_code, 200)
        off_data = off_res.json()
        self.assertIn("access_token", off_data)
        self.assertEqual(off_data["user"]["role"], "officer")

    def test_auth_invalid_login(self) -> None:
        """Test login rejection with wrong password or non-existent user."""
        bad_res = self.client.post(
            "/auth/login",
            json={"username": "officer", "password": "wrongpassword"},
        )
        self.assertEqual(bad_res.status_code, 401)

    def test_rbac_citizen_cannot_access_officer_endpoints(self) -> None:
        """Ensure a citizen JWT is strictly forbidden (403) from officer endpoints."""
        citizen_token = self._get_token("citizen", "citizen123")
        headers = {"Authorization": f"Bearer {citizen_token}"}

        for endpoint in ("/officer/incidents", "/officer/hotspots", "/officer/analytics"):
            res = self.client.get(endpoint, headers=headers)
            self.assertEqual(res.status_code, 403, f"Expected 403 on {endpoint}")

    def test_rbac_officer_access_allowed(self) -> None:
        """Ensure an officer JWT can access officer endpoints."""
        officer_token = self._get_token("officer", "officer123")
        headers = {"Authorization": f"Bearer {officer_token}"}

        res = self.client.get("/officer/analytics", headers=headers)
        self.assertEqual(res.status_code, 200)
        self.assertIn("total_reports", res.json())

    # ==========================================================================
    # 2. EXIF GPS EXTRACTION TESTS
    # ==========================================================================

    def test_exif_gps_present_and_hemispheres(self) -> None:
        """Test DMS to decimal conversion across N, S, E, and W hemispheres."""
        gps_ne = {
            "GPSLatitude": ((28, 1), (36, 1), (5004, 100)),
            "GPSLatitudeRef": "N",
            "GPSLongitude": ((77, 1), (12, 1), (3240, 100)),
            "GPSLongitudeRef": "E",
        }
        parsed_ne = parse_gps_info(gps_ne)
        self.assertEqual(parsed_ne["source"], "EXIF")
        self.assertAlmostEqual(parsed_ne["latitude"], 28.6139, places=4)
        self.assertAlmostEqual(parsed_ne["longitude"], 77.2090, places=4)

        lat_s = dms_to_decimal([(33, 1), (52, 1), (8, 1)], "S")
        lon_w = dms_to_decimal([(70, 1), (40, 1), (9, 1)], "W")
        self.assertLess(lat_s, 0.0)
        self.assertLess(lon_w, 0.0)
        self.assertAlmostEqual(lat_s, -33.868889, places=4)
        self.assertAlmostEqual(lon_w, -70.669167, places=4)

    def test_exif_gps_absent_fallback(self) -> None:
        """Verify image without EXIF metadata returns MANUAL source without crashing."""
        img_bytes = create_test_jpeg_bytes()
        result = extract_gps_from_image(img_bytes)
        self.assertIsNone(result["latitude"])
        self.assertIsNone(result["longitude"])
        self.assertEqual(result["source"], "MANUAL")

    # ==========================================================================
    # 3. SEVERITY ENGINE TESTS
    # ==========================================================================

    def test_severity_low_medium_high_tiers(self) -> None:
        """Verify Low (<25%), Medium (25%-60%), and High (>60%) severity calculations."""
        low_res = calculate_severity(
            detections=[
                {
                    "class_name": "Plastic Waste",
                    "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
                }
            ],
            image_width=1000,
            image_height=100,
        )
        self.assertAlmostEqual(low_res["severity_score"], 10.0, places=2)
        self.assertEqual(low_res["severity_tier"], "Low")

        med_res = calculate_severity(
            detections=[
                {
                    "class_name": "Plastic Waste",
                    "bbox": {"x1": 0, "y1": 0, "x2": 400, "y2": 100},
                }
            ],
            image_width=1000,
            image_height=100,
        )
        self.assertAlmostEqual(med_res["severity_score"], 40.0, places=2)
        self.assertEqual(med_res["severity_tier"], "Medium")

        high_res = calculate_severity(
            detections=[
                {
                    "class_name": "Organic Waste",
                    "bbox": {"x1": 0, "y1": 0, "x2": 600, "y2": 100},
                }
            ],
            image_width=1000,
            image_height=100,
        )
        self.assertAlmostEqual(high_res["severity_score"], 72.0, places=2)
        self.assertEqual(high_res["severity_tier"], "High")

        self.assertEqual(determine_severity_tier(24.9), "Low")
        self.assertEqual(determine_severity_tier(25.0), "Medium")
        self.assertEqual(determine_severity_tier(60.0), "Medium")
        self.assertEqual(determine_severity_tier(60.1), "High")

    # ==========================================================================
    # 4. DBSCAN GEOSPATIAL CLUSTERING TESTS
    # ==========================================================================

    def test_dbscan_clustering_within_and_beyond_50m(self) -> None:
        """Test DBSCAN <50m aggregation, >50m separation, multiple clusters, and single incident."""
        single = cluster_incidents_data(
            [{"id": 1, "latitude": 28.613900, "longitude": 77.209000, "severity_score": 45.0}]
        )
        self.assertEqual(len(single), 1)
        self.assertEqual(single[0]["report_count"], 1)

        pt_a = (28.613900, 77.209000)
        pt_b = (28.613950, 77.209020)
        pt_c = (28.620000, 77.215000)

        self.assertLess(haversine_distance_meters(pt_a[0], pt_a[1], pt_b[0], pt_b[1]), 50.0)
        self.assertGreater(haversine_distance_meters(pt_a[0], pt_a[1], pt_c[0], pt_c[1]), 50.0)

        clusters = cluster_incidents_data(
            [
                {"id": 101, "latitude": pt_a[0], "longitude": pt_a[1], "severity_score": 65.0},
                {"id": 102, "latitude": pt_b[0], "longitude": pt_b[1], "severity_score": 82.0},
                {"id": 103, "latitude": pt_c[0], "longitude": pt_c[1], "severity_score": 20.0},
            ]
        )
        self.assertEqual(len(clusters), 2)
        self.assertEqual(clusters[0]["report_count"], 2)
        self.assertEqual(clusters[0]["maximum_severity"], 82.0)
        self.assertEqual(clusters[0]["severity_tier"], "High")
        self.assertEqual(clusters[1]["report_count"], 1)
        self.assertEqual(clusters[1]["severity_tier"], "Low")

    # ==========================================================================
    # 5. END-TO-END API & CITIZEN -> OFFICER WORKFLOW TESTS
    # ==========================================================================

    def test_api_health_analyze_submit_dashboard_and_dispatch(self) -> None:
        """Test health check, image analysis, incident creation, demo seed, officer hotspots, and dispatch."""
        health_res = self.client.get("/health")
        self.assertEqual(health_res.status_code, 200)
        self.assertEqual(health_res.json()["status"], "ok")
        self.assertEqual(health_res.json()["service"], "CleanPath AI")

        img_bytes = create_test_jpeg_bytes()
        analyze_res = self.client.post(
            "/incidents/analyze",
            files={"file": ("sample_waste.jpg", img_bytes, "image/jpeg")},
        )
        self.assertEqual(analyze_res.status_code, 200, analyze_res.text)
        analyze_data = analyze_res.json()
        self.assertIn("detections", analyze_data)
        self.assertIn("severity_score", analyze_data)
        self.assertIn("severity_tier", analyze_data)
        self.assertIn("inference_mode", analyze_data)
        self.assertIn("address", analyze_data)

        citizen_token = self._get_token("citizen", "citizen123")
        cit_headers = {"Authorization": f"Bearer {citizen_token}"}
        create_res = self.client.post(
            "/incidents",
            headers=cit_headers,
            json={
                "image_path": analyze_data["image_path"],
                "latitude": 28.61392,
                "longitude": 77.20901,
                "location_source": "MANUAL",
                "detected_categories": analyze_data["detected_categories"],
                "detections": analyze_data["detections"],
                "detection_count": analyze_data["detection_count"],
                "severity_score": 74.5,
                "severity_tier": "High",
            },
        )
        self.assertEqual(create_res.status_code, 201, create_res.text)
        created_inc = create_res.json()
        self.assertTrue(created_inc["incident_uuid"].startswith("CP-"))
        self.assertIsNotNone(created_inc["hotspot_id"])

        my_res = self.client.get("/incidents/my", headers=cit_headers)
        self.assertEqual(my_res.status_code, 200)
        self.assertGreaterEqual(len(my_res.json()), 1)

        officer_token = self._get_token("officer", "officer123")
        off_headers = {"Authorization": f"Bearer {officer_token}"}

        seed_res = self.client.post(
            "/demo/seed",
            headers=off_headers,
            json={"center_latitude": 28.6139, "center_longitude": 77.2090, "clear_existing": True},
        )
        self.assertEqual(seed_res.status_code, 200, seed_res.text)
        seed_info = seed_res.json()
        self.assertEqual(seed_info["total_reports"], 15)
        self.assertEqual(seed_info["unique_hotspots"], 10)
        self.assertEqual(seed_info["duplicate_reports_suppressed"], 5)
        self.assertEqual(seed_info["dispatch_operations"], 5)

        hotspots_res = self.client.get("/officer/hotspots", headers=off_headers)
        self.assertEqual(hotspots_res.status_code, 200)
        hotspots_list = hotspots_res.json()
        self.assertEqual(len(hotspots_list), 10)

        target_hotspot_id = hotspots_list[-1]["id"]
        disp_res = self.client.post(
            f"/officer/dispatch/{target_hotspot_id}",
            headers=off_headers,
            json={
                "crew_name": "North Zone Sanitation Team A",
                "notes": "Automated test dispatch verification.",
            },
        )
        self.assertEqual(disp_res.status_code, 201, disp_res.text)
        self.assertEqual(disp_res.json()["hotspot"]["status"], "DISPATCHED")

    def test_real_citizen_to_officer_workflow_and_privacy_isolation(self) -> None:
        """
        Verify complete end-to-end production workflow:
        1. Citizen A submits 2 reports within 20m -> clustered into 1 hotspot (PRIMARY + DUPLICATE).
        2. Citizen B logs in -> must NOT see Citizen A's reports.
        3. Officer sees Citizen A's reports, dispatches crew, and resolves hotspot.
        4. Citizen A sees RESOLVED status and full AuditLog history.
        """
        officer_token = self._get_token("officer", "officer123")
        off_headers = {"Authorization": f"Bearer {officer_token}"}

        # Clear synthetic demo incidents so we test real citizen submissions
        db = SessionLocal()
        try:
            db.query(AuditLog).delete()
            db.query(ReportDetection).delete()
            db.query(HotspotReport).delete()
            db.query(Dispatch).delete()
            db.query(Incident).delete()
            db.query(Hotspot).delete()
            db.commit()
        finally:
            db.close()

        citizen_a_token = self._get_token("citizen_demo", "citizen123")
        citizen_b_token = self._get_token("citizen_priya", "citizen123")
        headers_a = {"Authorization": f"Bearer {citizen_a_token}"}
        headers_b = {"Authorization": f"Bearer {citizen_b_token}"}

        # Citizen A submits Report #1
        r1 = self.client.post(
            "/api/reports",
            headers=headers_a,
            json={
                "image_path": "/uploads/test_real_1.jpg",
                "original_filename": "street_waste_1.jpg",
                "latitude": 28.631500,
                "longitude": 77.216700,
                "location_source": "EXIF",
                "description": "Plastic bottles and banana peels near metro exit",
                "detected_categories": ["Plastic Waste", "Organic Waste"],
                "detections": [
                    {
                        "class_name": "Plastic Waste",
                        "confidence": 0.91,
                        "bbox": {"x1": 50, "y1": 60, "x2": 180, "y2": 210},
                    },
                    {
                        "class_name": "Organic Waste",
                        "confidence": 0.86,
                        "bbox": {"x1": 210, "y1": 120, "x2": 340, "y2": 260},
                    },
                ],
                "detection_count": 2,
                "severity_score": 78.5,
                "severity_tier": "High",
            },
        )
        self.assertEqual(r1.status_code, 201, r1.text)
        rep1 = r1.json()
        self.assertTrue(rep1["incident_uuid"].startswith("CP-2026-"))
        self.assertEqual(rep1["duplicate_status"], "PRIMARY")
        self.assertTrue(rep1["is_primary"])
        self.assertTrue(len(rep1["address"]) > 5)

        # Citizen A submits Report #2 ~15m away (<50m)
        r2 = self.client.post(
            "/api/reports",
            headers=headers_a,
            json={
                "image_path": "/uploads/test_real_2.jpg",
                "original_filename": "street_waste_2.jpg",
                "latitude": 28.631580,
                "longitude": 77.216770,
                "location_source": "CURRENT_LOCATION",
                "description": "Second angle of same waste pile",
                "detected_categories": ["Plastic Waste"],
                "detections": [
                    {
                        "class_name": "Plastic Waste",
                        "confidence": 0.88,
                        "bbox": {"x1": 70, "y1": 80, "x2": 200, "y2": 220},
                    }
                ],
                "detection_count": 1,
                "severity_score": 64.0,
                "severity_tier": "High",
            },
        )
        self.assertEqual(r2.status_code, 201, r2.text)
        rep2 = r2.json()
        self.assertEqual(rep1["hotspot_id"], rep2["hotspot_id"])
        self.assertFalse(rep2["is_primary"])
        self.assertEqual(rep2["duplicate_status"], "DUPLICATE / CLUSTERED")

        # Citizen B must NOT see Citizen A's reports
        my_b = self.client.get("/api/reports/my", headers=headers_b)
        self.assertEqual(my_b.status_code, 200)
        self.assertEqual(len(my_b.json()), 0)

        # Unauthenticated request without guest_uuids must return []
        unauth_my = self.client.get("/api/reports/my")
        self.assertEqual(unauth_my.status_code, 200)
        self.assertEqual(len(unauth_my.json()), 0)

        # Officer sees both reports and 1 unified hotspot with 1 duplicate suppressed
        stats_res = self.client.get("/api/dashboard/statistics", headers=off_headers)
        self.assertEqual(stats_res.status_code, 200)
        stats = stats_res.json()
        self.assertEqual(stats["total_reports"], 2)
        self.assertEqual(stats["total_hotspots"], 1)
        self.assertEqual(stats["duplicate_reports_suppressed"], 1)

        hotspot_id = rep1["hotspot_id"]
        disp_res = self.client.post(
            "/api/dispatch",
            headers=off_headers,
            json={
                "hotspot_id": hotspot_id,
                "crew_name": "Central Rapid Response Unit #1",
                "vehicle_number": "DL-1G-9988",
                "notes": "Dispatched to clear Connaught Place cluster",
            },
        )
        self.assertEqual(disp_res.status_code, 201, disp_res.text)

        # Officer updates hotspot status to RESOLVED
        resolve_res = self.client.patch(
            f"/api/hotspots/{hotspot_id}/status",
            headers=off_headers,
            json={"status": "RESOLVED", "notes": "Site cleaned and verified."},
        )
        self.assertEqual(resolve_res.status_code, 200, resolve_res.text)

        # Citizen A checks My Reports and sees RESOLVED + AuditLog history
        my_a = self.client.get("/api/reports/my", headers=headers_a)
        self.assertEqual(my_a.status_code, 200)
        reports_a = my_a.json()
        self.assertEqual(len(reports_a), 2)
        self.assertEqual(reports_a[0]["status"], "RESOLVED")
        self.assertGreaterEqual(len(reports_a[0].get("audit_logs", [])), 3)


if __name__ == "__main__":
    unittest.main()
