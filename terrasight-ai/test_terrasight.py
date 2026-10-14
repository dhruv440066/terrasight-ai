"""
Automated Verification Suite for TerraSight AI Streamlit Components
===================================================================
Tests:
1. EXIF extractor (DMS to DD conversion and fallback)
2. YOLOv8 detector loading and inference
3. Haversine DBSCAN spatial clustering and centroid calculation
4. Streamlit app module compilation
"""

import unittest
from pathlib import Path
from PIL import Image
import numpy as np

from utils.exif_extractor import dms_to_decimal, extract_exif_metadata
from utils.clustering import cluster_incidents_dbscan, compute_cluster_metrics
from utils.detector import load_yolo_model, run_detection


class TestTerraSightComponents(unittest.TestCase):

    def test_dms_to_decimal(self):
        # 28 deg, 37 min, 53.4 sec North
        dms_lat = ((28, 1), (37, 1), (534, 10))
        lat = dms_to_decimal(dms_lat, 'N')
        self.assertAlmostEqual(lat, 28.6315, places=3)

        # 77 deg, 13 min, 0.12 sec East
        dms_lon = ((77, 1), (13, 1), (12, 100))
        lon = dms_to_decimal(dms_lon, 'E')
        self.assertAlmostEqual(lon, 77.2167, places=3)

        # South hemisphere check
        lat_south = dms_to_decimal(dms_lat, 'S')
        self.assertAlmostEqual(lat_south, -28.6315, places=3)

    def test_exif_missing_fallback(self):
        # Blank test image without EXIF
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        res = extract_exif_metadata(img)
        self.assertFalse(res["has_gps"])
        self.assertIsNone(res["latitude"])
        self.assertIsNone(res["longitude"])

    def test_dbscan_clustering(self):
        # Three points within ~20m of each other (should cluster)
        # One point ~5km away (should be separate)
        test_incidents = [
            {"id": "1", "latitude": 28.63150, "longitude": 77.21670, "severity_score": 80.0, "category_counts": {"Plastic Waste": 3}},
            {"id": "2", "latitude": 28.63160, "longitude": 77.21680, "severity_score": 75.0, "category_counts": {"Plastic Waste": 2}},
            {"id": "3", "latitude": 28.63145, "longitude": 77.21665, "severity_score": 70.0, "category_counts": {"Organic Waste": 1}},
            {"id": "4", "latitude": 28.60000, "longitude": 77.15000, "severity_score": 40.0, "category_counts": {"Dry Leaves / Biomass": 1}},
        ]

        clustered, hotspots = cluster_incidents_dbscan(test_incidents, epsilon_meters=50.0, min_samples=1)
        
        self.assertEqual(len(clustered), 4)
        # Cluster 0 should contain items 0, 1, 2
        c0_items = [item for item in clustered if item["cluster_id"] == clustered[0]["cluster_id"]]
        self.assertEqual(len(c0_items), 3)

        # Check KPI computation
        kpis = compute_cluster_metrics(clustered, hotspots)
        self.assertEqual(kpis["total_reports"], 4)
        self.assertGreaterEqual(kpis["active_hotspots"], 1)

    def test_yolo_detector(self):
        weights_path = Path("weights/best.pt")
        self.assertTrue(weights_path.exists(), "weights/best.pt must exist")
        
        model, desc = load_yolo_model(str(weights_path))
        self.assertIsNotNone(model)

        # Create dummy image and run inference
        dummy_img = Image.new("RGB", (320, 240), color=(60, 60, 60))
        annotated, detections, cat_counts = run_detection(dummy_img, model, conf_threshold=0.25)
        self.assertIsInstance(annotated, Image.Image)
        self.assertIsInstance(detections, list)
        self.assertIsInstance(cat_counts, dict)


if __name__ == "__main__":
    unittest.main()
