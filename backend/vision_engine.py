"""
Computer Vision & Severity Engine for CleanPath AI / TerraSight AI.

Unified Hybrid Dual-YOLOv8 + Spectral ROI Waste Detection Engine supporting:
1. Fast ONNX Runtime (`onnxruntime` + Pillow + NumPy) for BOTH Vercel Serverless
   (`api/`) and local deployments with < 200 MB total bundle size and zero OpenCV
   or PyTorch dependency in serverless environments.
2. Ultralytics PyTorch (`.pt`) fallback support for local GPU/CPU environments
   if `.onnx` exports are absent or newer `.pt` weights are configured.
3. Multi-pass inference:
   - Pass 1: Official COCO YOLOv8n (`yolov8n.onnx` / `.pt`) for human/portrait
     & vehicle/indoor false-positive suppression + recognizable waste items
     (bottles, cups, food scraps).
   - Pass 2: Custom 13-class TACO YOLOv8n (`models/best.onnx` / `.pt`) across
     full-frame, foreground crop, and 2x2 overlapping quadrant tiles for sharp
     multi-object localization on dense street dumps and collages.
   - Pass 3: Pure-NumPy HSV complementary organic peel & dry leaf cluster
     verification inside confirmed multi-object waste scenes.
4. Deterministic OpenCV / NumPy Analytical Fallback ONLY when YOLO weights
   and runtimes are completely unavailable.
5. Exact CleanPath Severity Engine preserved across all 3 functional categories
   (`Plastic Waste`, `Organic Waste`, `Dry Leaves / Biomass`) while attaching
   specific TACO item labels (`subclass`) to every detection box.
"""

import ast
import io
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    HAS_CV2 = False

try:
    import onnxruntime as ort

    HAS_ORT = True
except ImportError:
    ort = None  # type: ignore[assignment]
    HAS_ORT = False

import numpy as np
import sys
from dotenv import load_dotenv
from PIL import Image, ImageOps

_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from train_taco import (
    CATEGORY_MULTIPLIERS,
    FUNCTIONAL_CATEGORIES,
    GRANULAR_TACO_CLASSES,
    ID_TO_CATEGORY_MAP,
    ID_TO_GRANULAR_CATEGORY_MAP,
    map_taco_class_to_category,
    map_taco_class_to_granular,
)

load_dotenv()

logger = logging.getLogger("cleanpath.vision")

BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_CUSTOM_MODEL = BACKEND_DIR / os.getenv("MODEL_PATH", "models/best.onnx")
DEFAULT_BASE_MODEL = BACKEND_DIR / os.getenv("FALLBACK_MODEL_PATH", "models/yolov8n.onnx")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.18"))

COCO_CLASS_NAMES: Tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)

# COCO classes mapped to (functional_category, display_subclass, min_conf)
# Note: Personal belongings (backpack, handbag, umbrella, suitcase) and fragile items
# like vases/wine glasses are strictly excluded to prevent transparent plastic misclassification
COCO_WASTE_CLASS_MAP: Dict[str, Tuple[str, str, float]] = {
    "bottle": ("Plastic Waste", "Plastic bottle", 0.22),
    "cup": ("Plastic Waste", "Disposable cup", 0.24),
    "fork": ("Plastic Waste", "Bottle cap, straw & lid", 0.28),
    "knife": ("Plastic Waste", "Bottle cap, straw & lid", 0.28),
    "spoon": ("Plastic Waste", "Bottle cap, straw & lid", 0.28),
    "banana": ("Organic Waste", "Food & organic waste", 0.22),
    "apple": ("Organic Waste", "Food & organic waste", 0.24),
    "sandwich": ("Organic Waste", "Food & organic waste", 0.24),
    "orange": ("Organic Waste", "Food & organic waste", 0.24),
    "broccoli": ("Organic Waste", "Food & organic waste", 0.24),
    "carrot": ("Organic Waste", "Food & organic waste", 0.24),
    "hot dog": ("Organic Waste", "Food & organic waste", 0.26),
    "pizza": ("Organic Waste", "Food & organic waste", 0.24),
    "donut": ("Organic Waste", "Food & organic waste", 0.24),
    "cake": ("Organic Waste", "Food & organic waste", 0.26),
    "potted plant": ("Dry Leaves / Biomass", "Dry leaves & foliage", 0.34),
    "book": ("Dry Leaves / Biomass", "Paper & paper bag", 0.30),
}

# COCO classes that represent vehicles, humans, pets, or indoor electronics/furniture
COCO_VEHICLE_CLASSES = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "train",
    "airplane",
    "boat",
}

COCO_HUMAN_AND_PET_CLASSES = {
    "person",
    "cat",
    "dog",
    "bird",
    "horse",
    "sheep",
    "cow",
}

COCO_FURNITURE_ELECTRONICS = {
    "laptop",
    "tv",
    "cell phone",
    "keyboard",
    "mouse",
    "couch",
    "bed",
}

COCO_NON_WASTE_CLASSES = (
    COCO_VEHICLE_CLASSES
    | COCO_HUMAN_AND_PET_CLASSES
    | COCO_FURNITURE_ELECTRONICS
    | {"backpack", "handbag", "suitcase", "umbrella", "tie", "wine glass", "vase"}
)

INDOOR_PORTRAIT_ANCHORS = {
    "laptop",
    "couch",
    "tv",
    "cell phone",
}


def determine_severity_tier(severity_score: float) -> str:
    """
    Map a numerical severity score [0, 100] to a municipal priority tier:
        - Critical: severity >= 80.0
        - High:     55.0 <= severity < 80.0
        - Medium:   25.0 <= severity < 55.0
        - Low:      severity < 25.0
    """
    if severity_score >= 80.0:
        return "Critical"
    if severity_score >= 55.0:
        return "High"
    if severity_score >= 25.0:
        return "Medium"
    return "Low"


def _compute_iou(box_a: Dict[str, int], box_b: Dict[str, int]) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes."""
    ix1 = max(box_a["x1"], box_b["x1"])
    iy1 = max(box_a["y1"], box_b["y1"])
    ix2 = min(box_a["x2"], box_b["x2"])
    iy2 = min(box_a["y2"], box_b["y2"])

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = float((ix2 - ix1) * (iy2 - iy1))
    area_a = float(max(1, (box_a["x2"] - box_a["x1"]) * (box_a["y2"] - box_a["y1"])))
    area_b = float(max(1, (box_b["x2"] - box_b["x1"]) * (box_b["y2"] - box_b["y1"])))
    return inter / (area_a + area_b - inter)


def _compute_containment(inner: Dict[str, int], outer: Dict[str, int]) -> float:
    """Return fraction of `inner` box area that lies inside `outer` box."""
    ix1 = max(inner["x1"], outer["x1"])
    iy1 = max(inner["y1"], outer["y1"])
    ix2 = min(inner["x2"], outer["x2"])
    iy2 = min(inner["y2"], outer["y2"])

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = float((ix2 - ix1) * (iy2 - iy1))
    inner_area = float(
        max(1, (inner["x2"] - inner["x1"]) * (inner["y2"] - inner["y1"]))
    )
    return inter / inner_area


def _numpy_nms(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = 0.45,
) -> List[int]:
    """Pure NumPy greedy Non-Maximum Suppression on (N, 4) xyxy boxes."""
    if len(scores) == 0:
        return []
    order = np.argsort(-scores)
    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    keep: List[int] = []
    while order.size > 0:
        idx = int(order[0])
        keep.append(idx)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[idx], x1[rest])
        yy1 = np.maximum(y1[idx], y1[rest])
        xx2 = np.minimum(x2[idx], x2[rest])
        yy2 = np.minimum(y2[idx], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = np.maximum(1e-6, areas[idx] + areas[rest] - inter)
        iou = inter / union
        order = rest[iou <= iou_thresh]
    return keep


def _extract_connected_components(
    mask: np.ndarray,
    total_area: float,
    min_area_ratio: float,
    max_area_ratio: float,
    img_width: int,
    img_height: int,
    min_y_ratio: float = 0.0,
    min_fill_ratio: float = 0.0,
    max_aspect: float = 6.0,
) -> List[Tuple[int, int, int, int, float]]:
    """Extract bounding boxes from a boolean downsampled mask using pure NumPy + DFS."""
    grid_h, grid_w = mask.shape
    scale_x = img_width / float(max(1, grid_w))
    scale_y = img_height / float(max(1, grid_h))
    visited = np.zeros_like(mask, dtype=bool)
    min_box_area = total_area * min_area_ratio
    max_box_area = total_area * max_area_ratio
    min_y_px = int(img_height * min_y_ratio)
    candidates: List[Tuple[int, int, int, int, float]] = []

    ys, xs = np.nonzero(mask)
    for sy_raw, sx_raw in zip(ys.tolist(), xs.tolist()):
        if visited[sy_raw, sx_raw]:
            continue
        stack = [(sy_raw, sx_raw)]
        visited[sy_raw, sx_raw] = True
        min_x = max_x = sx_raw
        min_y = max_y = sy_raw
        pixel_count = 0
        while stack:
            cy, cx = stack.pop()
            pixel_count += 1
            if cx < min_x:
                min_x = cx
            if cx > max_x:
                max_x = cx
            if cy < min_y:
                min_y = cy
            if cy > max_y:
                max_y = cy
            for ny in range(max(0, cy - 1), min(grid_h, cy + 2)):
                for nx in range(max(0, cx - 1), min(grid_w, cx + 2)):
                    if mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

        grid_box_area = max(1, (max_x - min_x + 1) * (max_y - min_y + 1))
        if min_fill_ratio > 0.0 and (pixel_count / float(grid_box_area)) < min_fill_ratio:
            continue

        x1 = max(0, int(min_x * scale_x))
        y1 = max(0, int(min_y * scale_y))
        x2 = min(img_width, int((max_x + 1) * scale_x))
        y2 = min(img_height, int((max_y + 1) * scale_y))
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        if max(bw, bh) / float(min(bw, bh)) > max_aspect:
            continue
        box_area = float(bw * bh)
        if y1 >= min_y_px and min_box_area <= box_area <= max_box_area:
            candidates.append((x1, y1, x2, y2, box_area))

    candidates.sort(key=lambda item: item[4], reverse=True)
    return candidates


def _compute_raster_union_ratio(
    boxes: List[Dict[str, Any]],
    width: int,
    height: int,
) -> float:
    """Compute exact non-overlapping bounding-box union area fraction on a compact raster grid."""
    if not boxes or width <= 0 or height <= 0:
        return 0.0
    gw, gh = min(128, max(1, width)), min(96, max(1, height))
    mask = np.zeros((gh, gw), dtype=bool)
    sx = gw / float(width)
    sy = gh / float(height)
    for b in boxes:
        x1 = max(0, min(gw - 1, int(float(b.get("x1", 0.0)) * sx)))
        y1 = max(0, min(gh - 1, int(float(b.get("y1", 0.0)) * sy)))
        x2 = max(x1 + 1, min(gw, int(np.ceil(float(b.get("x2", 0.0)) * sx))))
        y2 = max(y1 + 1, min(gh, int(np.ceil(float(b.get("y2", 0.0)) * sy))))
        mask[y1:y2, x1:x2] = True
    return float(np.mean(mask))


# Scientifically weighted environmental & public health risk factors (Pillar 3)
CATEGORY_RISK_WEIGHTS: Dict[str, float] = {
    "Hazardous Waste": 1.50,
    "Organic Waste": 1.30,
    "Plastic Waste": 1.10,
    "Metal": 1.00,
    "Dry Leaves / Biomass": 0.75,
    "Paper / Cardboard": 0.75,
}


def calculate_severity(
    detections: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    repeat_reports: int = 0,
    legacy_mode: bool = False,
) -> Dict[str, Any]:
    """
    Calculate the CleanPath AI Severity Index and Severity Tier from detected bounding boxes.
    Supports 5-pillar scientific environmental assessment (default) or linear legacy model.
    """
    if image_width <= 0 or image_height <= 0 or not detections:
        return {
            "severity_score": 0.0,
            "severity_tier": "Low",
            "prediction_reliability": "High",
            "total_area_ratio": 0.0,
            "effective_multiplier": 0.0,
            "object_count": 0,
            "repeat_report_boost": 0.0,
            "breakdown": {
                "quantity": {"weight_pct": 25, "raw_score": 0.0, "contribution": 0.0, "label": "0 detected waste objects"},
                "accumulation": {"weight_pct": 25, "raw_score": 0.0, "contribution": 0.0, "label": "0.0% visual coverage"},
                "risk": {"weight_pct": 20, "raw_score": 0.0, "contribution": 0.0, "label": "No hazard detected"},
                "confidence": {"weight_pct": 15, "raw_score": 0.0, "contribution": 0.0, "label": "N/A"},
                "hotspot": {"weight_pct": 15, "raw_score": 0.0, "contribution": 0.0, "label": f"{repeat_reports} prior report(s)"},
            },
            "explanation": "No waste detected in target scene. Environmental severity is zero.",
            "formula_explanation": (
                "5-Pillar severity model: 25% Quantity + 25% Accumulation Area + 20% Category Hazard "
                "+ 15% Confidence Evidence + 15% Hotspot History."
            ),
        }

    total_image_area = float(image_width * image_height)
    box_coords: List[Dict[str, float]] = []
    category_list: List[str] = []
    confidence_list: List[float] = []
    total_bbox_area = 0.0
    weighted_multiplier_sum = 0.0

    for det in detections:
        cls_name = det.get("class_name", "Plastic Waste")
        functional_cat = map_taco_class_to_category(cls_name)
        category_list.append(functional_cat)
        conf = float(det.get("confidence", 0.85))
        confidence_list.append(conf)

        if "area_ratio" in det and det["area_ratio"] is not None and "bbox" not in det:
            box_area_ratio = max(0.0, float(det["area_ratio"]))
            box_area = box_area_ratio * total_image_area
            side = math.sqrt(max(1.0, box_area))
            cx, cy = image_width / 2.0, image_height / 2.0
            box_dict = {
                "x1": max(0.0, cx - side / 2.0),
                "y1": max(0.0, cy - side / 2.0),
                "x2": min(float(image_width), cx + side / 2.0),
                "y2": min(float(image_height), cy + side / 2.0),
            }
        else:
            bbox = det.get("bbox", {})
            x1 = float(bbox.get("x1", 0.0))
            y1 = float(bbox.get("y1", 0.0))
            x2 = float(bbox.get("x2", 0.0))
            y2 = float(bbox.get("y2", 0.0))
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            box_area = w * h
            box_area_ratio = box_area / total_image_area
            if "area_ratio" not in det or det["area_ratio"] is None:
                det["area_ratio"] = round(box_area_ratio, 4)
            box_dict = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

        box_coords.append(box_dict)
        total_bbox_area += box_area
        multiplier = CATEGORY_MULTIPLIERS.get(functional_cat, 1.0)
        weighted_multiplier_sum += box_area * multiplier

    # Legacy linear formula if explicitly requested
    if legacy_mode:
        if total_bbox_area <= 0.0:
            return {
                "severity_score": 0.0,
                "severity_tier": "Low",
                "total_area_ratio": 0.0,
                "effective_multiplier": 0.0,
                "object_count": len(detections),
                "repeat_report_boost": 0.0,
                "formula_explanation": "Legacy linear area-ratio model.",
            }
        area_ratio = total_bbox_area / total_image_area
        effective_multiplier = weighted_multiplier_sum / total_bbox_area
        repeat_boost = max(0, int(repeat_reports)) * 2.5
        raw_severity = (area_ratio * effective_multiplier * 100.0) + repeat_boost
        score = round(min(100.0, max(0.0, raw_severity)), 2)
        tier = determine_severity_tier(score)
        return {
            "severity_score": score,
            "severity_tier": tier,
            "total_area_ratio": round(area_ratio, 4),
            "effective_multiplier": round(effective_multiplier, 4),
            "object_count": len(detections),
            "repeat_report_boost": round(repeat_boost, 2),
            "formula_explanation": "Legacy linear area-ratio model.",
        }

    # 1. Pillar 1: Waste Quantity (25%)
    n_count = len(detections)
    s_quantity = 100.0 * (1.0 - math.exp(-n_count / 7.5))

    # 2. Pillar 2: Waste Accumulation / Non-overlapping Coverage (25%)
    union_ratio = _compute_raster_union_ratio(box_coords, image_width, image_height)
    s_accumulation = 100.0 * (1.0 - math.exp(-union_ratio / 0.18))

    # 3. Pillar 3: Category Risk (20%)
    cat_risk_sum = sum(CATEGORY_RISK_WEIGHTS.get(c, 1.0) for c in category_list)
    avg_cat_risk = cat_risk_sum / max(1, len(category_list))
    distinct_categories = len(set(category_list))
    diversity_bonus = 1.0 + 0.08 * (distinct_categories - 1)
    effective_risk = avg_cat_risk * diversity_bonus
    s_risk = min(100.0, max(0.0, ((effective_risk - 0.70) / (1.40 - 0.70)) * 100.0))

    # 4. Pillar 4: Confidence-Adjusted Evidence (15%)
    avg_conf = float(np.mean(confidence_list)) if confidence_list else 0.85
    evidence_intensity = (s_quantity + s_accumulation) / 2.0
    s_confidence = min(100.0, max(0.0, avg_conf * evidence_intensity))

    # 5. Pillar 5: Geospatial Hotspot Context (15%)
    repeats = max(0, int(repeat_reports))
    s_hotspot = min(100.0, 100.0 * (1.0 - math.exp(-repeats / 4.5)))

    # Weighted composite score
    c_q = 0.25 * s_quantity
    c_a = 0.25 * s_accumulation
    c_r = 0.20 * s_risk
    c_c = 0.15 * s_confidence
    c_h = 0.15 * s_hotspot

    raw_total = c_q + c_a + c_r + c_c + c_h
    severity_score = round(min(100.0, max(0.0, raw_total)), 2)
    severity_tier = determine_severity_tier(severity_score)

    reliability = "High" if avg_conf >= 0.80 else ("Moderate" if avg_conf >= 0.55 else "Preliminary")

    top_cat = max(set(category_list), key=category_list.count) if category_list else "Waste"
    explanation_parts = [
        f"Assessed as {severity_tier} severity ({severity_score}/100) with {n_count} detected object(s) "
        f"covering {round(union_ratio * 100.0, 1)}% non-overlapping visual footprint.",
        f"Primary waste profile is {top_cat} (risk index {round(s_risk, 1)}/100).",
        f"Model detection reliability is {reliability} ({round(avg_conf * 100.0, 1)}% average confidence).",
    ]
    if repeats > 0:
        explanation_parts.append(f"Amplified by {repeats} prior report(s) in this hotspot vicinity (+{round(c_h, 1)} pts).")
    explanation = " ".join(explanation_parts)

    return {
        "severity_score": severity_score,
        "severity_tier": severity_tier,
        "prediction_reliability": reliability,
        "breakdown": {
            "quantity": {
                "weight_pct": 25,
                "raw_score": round(s_quantity, 1),
                "contribution": round(c_q, 2),
                "object_count": n_count,
                "label": f"{n_count} detected waste object(s)",
            },
            "accumulation": {
                "weight_pct": 25,
                "raw_score": round(s_accumulation, 1),
                "contribution": round(c_a, 2),
                "coverage_pct": round(union_ratio * 100.0, 1),
                "label": f"{round(union_ratio * 100.0, 1)}% visual accumulation footprint",
            },
            "risk": {
                "weight_pct": 20,
                "raw_score": round(s_risk, 1),
                "contribution": round(c_r, 2),
                "dominant_category": top_cat,
                "label": f"Hazard risk: {top_cat} ({round(s_risk, 1)}/100)",
            },
            "confidence": {
                "weight_pct": 15,
                "raw_score": round(s_confidence, 1),
                "contribution": round(c_c, 2),
                "avg_confidence_pct": round(avg_conf * 100.0, 1),
                "label": f"{round(avg_conf * 100.0, 1)}% detection confidence evidence",
            },
            "hotspot": {
                "weight_pct": 15,
                "raw_score": round(s_hotspot, 1),
                "contribution": round(c_h, 2),
                "repeat_count": repeats,
                "label": f"{repeats} prior nearby reports",
            },
        },
        "explanation": explanation,
        # Backward-compatible metadata
        "total_area_ratio": round(union_ratio, 4),
        "effective_multiplier": round(effective_risk, 4),
        "object_count": n_count,
        "repeat_report_boost": round(c_h, 2),
        "formula_explanation": (
            "Multi-factor severity model: 25% Quantity + 25% Accumulation Area (raster union) + "
            "20% Hazard Risk + 15% Confidence Evidence + 15% Geospatial Hotspot History."
        ),
    }


class VisionEngine:
    """
    Manages Dual-YOLOv8 inference across ONNX Runtime (`best.onnx` + `yolov8n.onnx`)
    and Ultralytics PyTorch (`best.pt` + `yolov8n.pt`), multi-scale tiled detection,
    human/indoor portrait suppression, spectral ROI verification, and deterministic
    analytical fallback when YOLO assets are absent.
    """

    def __init__(self) -> None:
        self.model: Any = None
        self.model_meta: Dict[str, Any] = {}
        self.custom_backend: Optional[str] = None
        self.coco_model: Any = None
        self.coco_meta: Dict[str, Any] = {}
        self.coco_backend: Optional[str] = None
        self.model_path_used: Optional[str] = None
        self.coco_path_used: Optional[str] = None
        self.device: str = "cpu"
        self.yolo_available: bool = False
        self._initialize_model()

    def _load_onnx_session(self, onnx_path: Path) -> Tuple[Any, Dict[str, Any]]:
        """Initialize an ONNX Runtime session and extract input dimensions & class names."""
        opts = ort.SessionOptions()
        cpu_cnt = os.cpu_count() or 2
        opts.intra_op_num_threads = max(1, min(4, cpu_cnt))
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            str(onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        inp = session.get_inputs()[0]
        input_name = inp.name
        shape = inp.shape or [1, 3, 640, 640]
        inp_h = int(shape[2]) if len(shape) >= 4 and isinstance(shape[2], int) else 640
        inp_w = int(shape[3]) if len(shape) >= 4 and isinstance(shape[3], int) else 640

        names: Dict[int, str] = {}
        try:
            meta_map = session.get_modelmeta().custom_metadata_map or {}
            if "names" in meta_map:
                parsed = ast.literal_eval(meta_map["names"])
                if isinstance(parsed, dict):
                    names = {int(k): str(v) for k, v in parsed.items()}
                elif isinstance(parsed, list):
                    names = {idx: str(v) for idx, v in enumerate(parsed)}
        except Exception:
            names = {}

        return session, {
            "input_name": input_name,
            "input_hw": (inp_h, inp_w),
            "names": names,
        }

    def _unique_existing_paths(self, candidates: List[Path]) -> List[Path]:
        seen: set[str] = set()
        out: List[Path] = []
        for c in candidates:
            try:
                res = str(c.resolve())
            except Exception:
                res = str(c)
            if res not in seen:
                seen.add(res)
                out.append(c)
        return out

    def _initialize_model(self) -> None:
        """Load ONNX Runtime models first (serverless-safe), with PyTorch fallback."""
        self.model = None
        self.coco_model = None
        self.custom_backend = None
        self.coco_backend = None
        self.model_path_used = None
        self.coco_path_used = None
        self.yolo_available = False
        self.device = "cpu"

        repo_root = BACKEND_DIR.parent
        env_model_raw = os.getenv("MODEL_PATH", "").strip()
        custom_onnx_candidates: List[Path] = []
        custom_pt_candidates: List[Path] = []

        if env_model_raw:
            p = BACKEND_DIR / env_model_raw
            if p.suffix.lower() == ".onnx":
                custom_onnx_candidates.append(p)
            elif p.suffix.lower() == ".pt":
                custom_pt_candidates.append(p)
                custom_onnx_candidates.append(p.with_suffix(".onnx"))

        custom_onnx_candidates.extend(
            [
                BACKEND_DIR / "models" / "best.onnx",
                repo_root / "api" / "models" / "best.onnx",
                repo_root / "backend" / "models" / "best.onnx",
                repo_root / "weights" / "best.onnx",
            ]
        )
        custom_pt_candidates.extend(
            [
                BACKEND_DIR / "models" / "best.pt",
                repo_root / "backend" / "models" / "best.pt",
                repo_root / "weights" / "best.pt",
                repo_root / "terrasight-ai" / "weights" / "best.pt",
            ]
        )

        coco_onnx_candidates = [
            BACKEND_DIR / "models" / "yolov8n.onnx",
            BACKEND_DIR / "yolov8n.onnx",
            repo_root / "api" / "models" / "yolov8n.onnx",
            repo_root / "backend" / "models" / "yolov8n.onnx",
        ]
        coco_pt_candidates = [
            BACKEND_DIR / "yolov8n.pt",
            BACKEND_DIR / "models" / "yolov8n.pt",
            repo_root / "yolov8n.pt",
        ]

        # Check whether a local .pt checkpoint is noticeably newer than exported .onnx
        best_onnx = next(
            (p for p in self._unique_existing_paths(custom_onnx_candidates) if p.exists() and p.is_file() and p.stat().st_size > 1_000_000),
            None,
        )
        best_pt = next(
            (p for p in self._unique_existing_paths(custom_pt_candidates) if p.exists() and p.is_file() and p.stat().st_size > 1_000_000),
            None,
        )
        pt_is_newer = False
        if best_onnx is not None and best_pt is not None:
            if best_pt.stat().st_mtime > best_onnx.stat().st_mtime + 10.0:
                pt_is_newer = True

        # 1. Try ONNX Runtime first (unless a significantly newer local .pt exists)
        if HAS_ORT and ort is not None and best_onnx is not None and not pt_is_newer:
            try:
                sess, meta = self._load_onnx_session(best_onnx)
                self.model = sess
                self.model_meta = meta
                self.custom_backend = "onnx"
                self.model_path_used = str(best_onnx)
                self.yolo_available = True
                logger.info("Loaded custom YOLOv8 ONNX model from %s", best_onnx)
            except Exception as exc:
                logger.warning("Failed to load ONNX model %s: %s", best_onnx, exc)

        # 2. Load COCO auxiliary ONNX model
        if HAS_ORT and ort is not None:
            coco_onnx = next(
                (p for p in self._unique_existing_paths(coco_onnx_candidates) if p.exists() and p.is_file() and p.stat().st_size > 1_000_000),
                None,
            )
            if coco_onnx is not None:
                try:
                    csess, cmeta = self._load_onnx_session(coco_onnx)
                    if not cmeta.get("names"):
                        cmeta["names"] = {i: n for i, n in enumerate(COCO_CLASS_NAMES)}
                    self.coco_model = csess
                    self.coco_meta = cmeta
                    self.coco_backend = "onnx"
                    self.coco_path_used = str(coco_onnx)
                    self.yolo_available = True
                    logger.info("Loaded COCO YOLOv8n ONNX model from %s", coco_onnx)
                except Exception as exc:
                    logger.warning("Failed to load COCO ONNX model %s: %s", coco_onnx, exc)

        # 3. PyTorch / Ultralytics fallback (local backend when ONNX not used)
        if self.model is None or self.coco_model is None:
            try:
                import torch
                from ultralytics import YOLO

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                if self.device == "cpu":
                    try:
                        torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
                    except Exception:
                        pass

                if self.model is None and best_pt is not None:
                    self.model = YOLO(str(best_pt))
                    self.model_meta = {"names": getattr(self.model, "names", {})}
                    self.custom_backend = "pt"
                    self.model_path_used = str(best_pt)
                    self.yolo_available = True
                    logger.info("Loaded custom YOLOv8 PyTorch weights from %s", best_pt)

                if self.coco_model is None:
                    coco_pt = next(
                        (p for p in self._unique_existing_paths(coco_pt_candidates) if p.exists() and p.is_file() and p.stat().st_size > 1_000_000),
                        None,
                    )
                    if coco_pt is not None:
                        self.coco_model = YOLO(str(coco_pt))
                        self.coco_meta = {
                            "names": getattr(
                                self.coco_model,
                                "names",
                                {i: n for i, n in enumerate(COCO_CLASS_NAMES)},
                            )
                        }
                        self.coco_backend = "pt"
                        self.coco_path_used = str(coco_pt)
                        self.yolo_available = True
                        logger.info("Loaded COCO YOLOv8n PyTorch weights from %s", coco_pt)
            except Exception as exc:
                if not self.yolo_available:
                    logger.info(
                        "YOLO runtimes unavailable (%s). Analytical Fallback enabled.",
                        exc,
                    )

        # If only COCO loaded and no custom model yet, promote COCO as fallback primary
        if self.model is None and self.coco_model is not None and not self.yolo_available:
            self.yolo_available = True

    def reload_models(self) -> Dict[str, Any]:
        """Re-initialize models from disk (useful after training or ONNX export)."""
        self._initialize_model()
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """Return current vision engine status for system health & officer dashboard."""
        loaded_parts = []
        if self.model_path_used:
            loaded_parts.append(Path(self.model_path_used).name)
        if self.coco_path_used:
            loaded_parts.append(Path(self.coco_path_used).name)

        if self.custom_backend == "onnx" or self.coco_backend == "onnx":
            runtime_label = "ONNX Runtime"
        elif self.custom_backend == "pt" or self.coco_backend == "pt":
            runtime_label = "Ultralytics PyTorch"
        else:
            runtime_label = "Analytical"

        return {
            "ai_engine": "Online" if self.yolo_available else "Fallback Mode",
            "inference_mode": "YOLOv8" if self.yolo_available else "Analytical Fallback",
            "runtime": runtime_label,
            "device": self.device,
            "model_path": self.model_path_used or self.coco_path_used,
            "weights_loaded": " + ".join(loaded_parts) if loaded_parts else "Analytical Fallback",
            "categories": FUNCTIONAL_CATEGORIES,
            "granular_classes": GRANULAR_TACO_CLASSES,
        }

    def _predict_onnx(
        self,
        session: Any,
        meta: Dict[str, Any],
        pil_crop: Image.Image,
        conf_thresh: float,
        iou_thresh: float = 0.45,
        class_agnostic_nms: bool = False,
    ) -> List[Tuple[int, str, float, int, int, int, int]]:
        """Run single YOLO ONNX forward pass with Pillow letterbox & pure-NumPy NMS."""
        crop_w, crop_h = pil_crop.size
        inp_h, inp_w = meta.get("input_hw", (640, 640))
        scale = min(inp_w / float(max(1, crop_w)), inp_h / float(max(1, crop_h)))
        new_w = max(1, int(round(crop_w * scale)))
        new_h = max(1, int(round(crop_h * scale)))
        dw = (inp_w - new_w) / 2.0
        dh = (inp_h - new_h) / 2.0
        pad_x = int(round(dw - 0.1))
        pad_y = int(round(dh - 0.1))

        resized = pil_crop.resize((new_w, new_h), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (inp_w, inp_h), (114, 114, 114))
        canvas.paste(resized, (pad_x, pad_y))

        tensor = np.asarray(canvas, dtype=np.float32) * (1.0 / 255.0)
        tensor = np.ascontiguousarray(np.transpose(tensor, (2, 0, 1))[None, ...])

        input_name = meta.get("input_name") or session.get_inputs()[0].name
        outputs = session.run(None, {input_name: tensor})
        raw_out = outputs[0]

        if raw_out.ndim == 3:
            if raw_out.shape[1] < raw_out.shape[2]:
                preds = np.transpose(raw_out[0], (1, 0))
            else:
                preds = raw_out[0]
        else:
            preds = raw_out

        if preds.shape[1] <= 4:
            return []

        nc = preds.shape[1] - 4
        cls_scores = preds[:, 4:]
        cls_ids = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(len(preds)), cls_ids]

        mask = scores >= conf_thresh
        if not np.any(mask):
            return []

        boxes_xywh = preds[mask, :4]
        filt_scores = scores[mask]
        filt_cls = cls_ids[mask]

        cx = boxes_xywh[:, 0]
        cy = boxes_xywh[:, 1]
        bw = boxes_xywh[:, 2]
        bh = boxes_xywh[:, 3]

        x1 = np.clip((cx - bw * 0.5 - pad_x) / scale, 0.0, float(crop_w))
        y1 = np.clip((cy - bh * 0.5 - pad_y) / scale, 0.0, float(crop_h))
        x2 = np.clip((cx + bw * 0.5 - pad_x) / scale, 0.0, float(crop_w))
        y2 = np.clip((cy + bh * 0.5 - pad_y) / scale, 0.0, float(crop_h))
        xyxy = np.stack([x1, y1, x2, y2], axis=1)

        if class_agnostic_nms:
            keep_indices = _numpy_nms(xyxy, filt_scores, iou_thresh=iou_thresh)
        else:
            keep_indices = []
            for unique_cid in np.unique(filt_cls).tolist():
                c_mask = np.nonzero(filt_cls == unique_cid)[0]
                kept_local = _numpy_nms(
                    xyxy[c_mask], filt_scores[c_mask], iou_thresh=iou_thresh
                )
                keep_indices.extend(int(c_mask[li]) for li in kept_local)
            keep_indices.sort(key=lambda idx: float(filt_scores[idx]), reverse=True)

        names_map = meta.get("names") or {}
        if not names_map:
            if nc == len(GRANULAR_TACO_CLASSES):
                names_map = ID_TO_GRANULAR_CATEGORY_MAP
            elif nc == len(FUNCTIONAL_CATEGORIES):
                names_map = ID_TO_CATEGORY_MAP
            elif nc == len(COCO_CLASS_NAMES):
                names_map = {i: n for i, n in enumerate(COCO_CLASS_NAMES)}

        results: List[Tuple[int, str, float, int, int, int, int]] = []
        for idx in keep_indices:
            cid = int(filt_cls[idx])
            conf = float(filt_scores[idx])
            bx1 = int(round(float(xyxy[idx, 0])))
            by1 = int(round(float(xyxy[idx, 1])))
            bx2 = int(round(float(xyxy[idx, 2])))
            by2 = int(round(float(xyxy[idx, 3])))
            if bx2 > bx1 and by2 > by1:
                raw_name = str(names_map.get(cid, f"Class {cid}"))
                results.append((cid, raw_name, conf, bx1, by1, bx2, by2))
        return results

    def _predict_pt(
        self,
        model: Any,
        meta: Dict[str, Any],
        pil_crop: Image.Image,
        conf_thresh: float,
        imgsz: int = 512,
    ) -> List[Tuple[int, str, float, int, int, int, int]]:
        """Run Ultralytics PyTorch inference on a Pillow crop."""
        try:
            import torch

            ctx = torch.inference_mode()
        except Exception:
            from contextlib import nullcontext

            ctx = nullcontext()

        crop_w, crop_h = pil_crop.size
        with ctx:
            pt_results = model.predict(
                source=pil_crop,
                device=self.device,
                imgsz=imgsz,
                conf=conf_thresh,
                verbose=False,
            )

        out: List[Tuple[int, str, float, int, int, int, int]] = []
        for res in pt_results:
            names_map = getattr(res, "names", None) or meta.get("names") or {}
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                bx1, by1, bx2, by2 = [int(round(v)) for v in box.xyxy[0].tolist()]
                bx1 = max(0, min(crop_w, bx1))
                by1 = max(0, min(crop_h, by1))
                bx2 = max(0, min(crop_w, bx2))
                by2 = max(0, min(crop_h, by2))
                if bx2 <= bx1 or by2 <= by1:
                    continue
                conf = float(box.conf[0].item())
                cid = int(box.cls[0].item())
                raw_name = (
                    str(names_map.get(cid, ID_TO_CATEGORY_MAP.get(cid, "Plastic Waste")))
                    if isinstance(names_map, dict)
                    else str(names_map[cid] if cid < len(names_map) else cid)
                )
                out.append((cid, raw_name, conf, bx1, by1, bx2, by2))
        return out

    def _infer_crop(
        self,
        backend: Optional[str],
        model_obj: Any,
        meta: Dict[str, Any],
        pil_crop: Image.Image,
        conf_thresh: float,
        imgsz: int = 512,
        class_agnostic_nms: bool = False,
    ) -> List[Tuple[int, str, float, int, int, int, int]]:
        if model_obj is None or backend is None:
            return []
        if backend == "onnx":
            return self._predict_onnx(
                model_obj,
                meta,
                pil_crop,
                conf_thresh=conf_thresh,
                class_agnostic_nms=class_agnostic_nms,
            )
        return self._predict_pt(
            model_obj,
            meta,
            pil_crop,
            conf_thresh=conf_thresh,
            imgsz=imgsz,
        )

    def _compute_boxes_union_ratio(
        self,
        boxes: List[Dict[str, int]],
        width: int,
        height: int,
    ) -> float:
        """Compute exact union area fraction on a compact raster grid."""
        if not boxes or width <= 0 or height <= 0:
            return 0.0
        gw, gh = min(128, width), min(96, height)
        mask = np.zeros((gh, gw), dtype=bool)
        sx = gw / float(width)
        sy = gh / float(height)
        for b in boxes:
            x1 = max(0, min(gw - 1, int(b["x1"] * sx)))
            y1 = max(0, min(gh - 1, int(b["y1"] * sy)))
            x2 = max(x1 + 1, min(gw, int(np.ceil(b["x2"] * sx))))
            y2 = max(y1 + 1, min(gh, int(np.ceil(b["y2"] * sy))))
            mask[y1:y2, x1:x2] = True
        return float(np.mean(mask))

    def _refine_detection_from_roi(
        self,
        roi_hsv: np.ndarray,
        initial_category: str,
        initial_subclass: str,
        confidence: float,
        is_granular_model: bool,
    ) -> Tuple[str, str, float]:
        """
        Inspect HSV pixel statistics inside a detected box to upgrade coarse or
        ambiguous labels (`Mixed litter` / legacy `Plastic Waste`) when obvious
        organic food peels or plant foliage occupy the bounding box.
        """
        if roi_hsv.size == 0:
            return initial_category, initial_subclass, confidence

        # Trust explicit high-confidence granular item predictions directly
        high_trust_subclasses = {
            "Plastic bottle",
            "Drink & metal can",
            "Disposable cup",
            "Styrofoam piece",
            "Cardboard carton",
            "Paper & paper bag",
            "Cigarette",
            "Food & takeout container",
            "Bottle cap, straw & lid",
            "Plastic wrapper",
        }
        if is_granular_model and initial_subclass in high_trust_subclasses and confidence >= 0.20:
            return initial_category, initial_subclass, confidence

        if initial_subclass == "Glass bottle & jar":
            # Glass bottle/jar requires high confidence (>= 0.45) to avoid false positives on transparent plastic.
            # Below 0.45, it is safely classified as Plastic bottle.
            if confidence >= 0.45:
                return "Plastic Waste", "Glass bottle & jar", confidence
            else:
                return "Plastic Waste", "Plastic bottle", confidence

        h = roi_hsv[:, :, 0].astype(np.float32)
        s = roi_hsv[:, :, 1].astype(np.float32)
        v = roi_hsv[:, :, 2].astype(np.float32)
        total_px = float(h.size)

        blue_plastic_ratio = float(np.sum((h >= 95) & (h <= 135) & (s >= 65) & (v >= 50))) / total_px
        white_foam_ratio = float(np.sum((s <= 45) & (v >= 190))) / total_px
        black_bag_ratio = float(np.sum((v <= 55) & (s <= 95))) / total_px
        if blue_plastic_ratio > 0.22 or white_foam_ratio > 0.34 or black_bag_ratio > 0.42:
            if initial_subclass in ("Mixed litter", "Plastic Waste") and black_bag_ratio > 0.42:
                return "Plastic Waste", "Plastic bag & film", max(confidence, 0.72)
            if initial_subclass in ("Mixed litter", "Plastic Waste") and white_foam_ratio > 0.34:
                return "Plastic Waste", "Styrofoam piece", max(confidence, 0.70)
            return initial_category, initial_subclass, confidence

        # Kraft cardboard check (warm brown matte tones, moderate saturation)
        cardboard_mask = (h >= 10) & (h <= 24) & (s >= 35) & (s <= 140) & (v >= 70) & (v <= 190)
        cardboard_ratio = float(np.sum(cardboard_mask)) / total_px
        if cardboard_ratio > 0.42 and initial_subclass in ("Mixed litter", "Plastic Waste"):
            return "Dry Leaves / Biomass", "Cardboard carton", max(confidence, 0.68)

        organic_mask = ((h >= 15) & (h <= 26) & (s >= 150) & (v >= 110)) | (
            (h >= 7) & (h < 15) & (s >= 165) & (v >= 115)
        )
        organic_ratio = float(np.sum(organic_mask)) / total_px

        leaf_mask = (h >= 21) & (h <= 35) & (s >= 68) & (s <= 148) & (v >= 48) & (v <= 165)
        leaf_ratio = float(np.sum(leaf_mask)) / total_px

        if (
            initial_subclass in ("Mixed litter", "Plastic Waste", "Plastic bag & film", "Food & organic waste")
            and organic_ratio >= 0.34
            and organic_ratio >= leaf_ratio
        ):
            return "Organic Waste", "Food & organic waste", round(min(0.76, max(confidence, confidence + 0.12)), 2)

        if (
            initial_subclass in ("Mixed litter", "Dry Leaves / Biomass", "Dry leaves & foliage")
            and leaf_ratio >= 0.48
        ):
            return "Dry Leaves / Biomass", "Dry leaves & foliage", round(min(0.74, max(confidence, confidence + 0.10)), 2)

        return initial_category, initial_subclass, confidence

    def _segment_complementary_waste_numpy(
        self,
        pil_rgb: Image.Image,
        existing_dets: List[Dict[str, Any]],
        non_waste_boxes: List[Dict[str, int]],
        width: int,
        height: int,
    ) -> List[Dict[str, Any]]:
        """
        Pure-NumPy + Pillow spectral segmentation for missed Organic food scraps
        and Dry Leaves inside ALREADY-CONFIRMED multi-object waste piles.
        Requires >= 4 YOLO detections so clean tabletop/isolated shots never trigger false positives.
        """
        if len(existing_dets) < 4 or width <= 0 or height <= 0:
            return []

        total_area = float(width * height)
        # Compute expanded waste-pile bounding envelope from confirmed YOLO boxes
        xs1 = [d["bbox"]["x1"] for d in existing_dets]
        ys1 = [d["bbox"]["y1"] for d in existing_dets]
        xs2 = [d["bbox"]["x2"] for d in existing_dets]
        ys2 = [d["bbox"]["y2"] for d in existing_dets]
        pad_x = int(width * 0.09)
        pad_y = int(height * 0.09)
        pile_x1 = max(0, min(xs1) - pad_x)
        pile_y1 = max(int(height * 0.20), min(ys1) - pad_y)
        pile_x2 = min(width, max(xs2) + pad_x)
        pile_y2 = min(height, max(ys2) + pad_y)

        grid_w = min(112, max(32, width // 6))
        grid_h = min(84, max(24, height // 6))
        small_rgb = pil_rgb.resize((grid_w, grid_h), Image.Resampling.BILINEAR)
        hsv = np.asarray(small_rgb.convert("HSV"), dtype=np.uint8).copy()
        hsv[:, :, 0] = (hsv[:, :, 0].astype(np.uint16) * 180 // 256).astype(np.uint8)

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        # Restrict to waste pile envelope on grid
        gz = np.zeros((grid_h, grid_w), dtype=bool)
        gx1 = max(0, int(pile_x1 * grid_w / float(width)))
        gy1 = max(0, int(pile_y1 * grid_h / float(height)))
        gx2 = min(grid_w, int(np.ceil(pile_x2 * grid_w / float(width))))
        gy2 = min(grid_h, int(np.ceil(pile_y2 * grid_h / float(height))))
        gz[gy1:gy2, gx1:gx2] = True

        supplemental: List[Dict[str, Any]] = []

        def _count_nearby_yolo_trash(box_dict: Dict[str, int], max_gap_frac: float = 0.06) -> int:
            """Count distinct confirmed YOLO trash detections adjacent to this region."""
            gx = width * max_gap_frac
            gy = height * max_gap_frac
            hits = 0
            for eb in existing_dets:
                kb = eb["bbox"]
                dx = max(0, max(box_dict["x1"] - kb["x2"], kb["x1"] - box_dict["x2"]))
                dy = max(0, max(box_dict["y1"] - kb["y2"], kb["y1"] - box_dict["y2"]))
                if dx <= gx and dy <= gy:
                    hits += 1
            return hits

        # 1. Organic food peels / warm organic clusters inside confirmed waste pile
        org_mask = gz & (
            ((h >= 15) & (h <= 25) & (s >= 166) & (v >= 115) & (v <= 245))
            | ((h >= 8) & (h < 15) & (s >= 176) & (v >= 120) & (v <= 240))
        )
        org_cands = _extract_connected_components(
            org_mask,
            total_area=total_area,
            min_area_ratio=0.0028,
            max_area_ratio=0.075,
            img_width=width,
            img_height=height,
            min_y_ratio=0.24,
            min_fill_ratio=0.22,
            max_aspect=3.6,
        )
        for x1, y1, x2, y2, b_area in org_cands[:3]:
            cand_box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            if _count_nearby_yolo_trash(cand_box, max_gap_frac=0.05) < 1:
                continue
            if any(_compute_containment(cand_box, nw) > 0.45 for nw in non_waste_boxes):
                continue
            if any(
                _compute_iou(cand_box, eb["bbox"]) > 0.28 and eb["class_name"] == "Organic Waste"
                for eb in existing_dets + supplemental
            ):
                continue
            supplemental.append(
                {
                    "class_name": "Organic Waste",
                    "subclass": "Food & organic waste",
                    "confidence": round(min(0.87, 0.68 + (b_area / total_area) * 3.0), 2),
                    "bbox": cand_box,
                    "area_ratio": round(b_area / total_area, 4),
                }
            )

        # 2. Dry leaves & foliage clusters strictly inside dense pile (excludes green trees/weeds)
        if len(existing_dets) >= 5:
            leaf_mask = gz & (h >= 21) & (h <= 34) & (s >= 68) & (s <= 148) & (v >= 48) & (v <= 165)
            leaf_cands = _extract_connected_components(
                leaf_mask,
                total_area=total_area,
                min_area_ratio=0.0038,
                max_area_ratio=0.036,
                img_width=width,
                img_height=height,
                min_y_ratio=0.45,
                min_fill_ratio=0.28,
                max_aspect=2.6,
            )
            for x1, y1, x2, y2, b_area in leaf_cands[:2]:
                cand_box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                if _count_nearby_yolo_trash(cand_box, max_gap_frac=0.035) < 2:
                    continue
                if any(_compute_containment(cand_box, nw) > 0.45 for nw in non_waste_boxes):
                    continue
                if any(
                    _compute_iou(cand_box, eb["bbox"]) > 0.28
                    and eb["class_name"] == "Dry Leaves / Biomass"
                    for eb in existing_dets + supplemental
                ):
                    continue
                supplemental.append(
                    {
                        "class_name": "Dry Leaves / Biomass",
                        "subclass": "Dry leaves & foliage",
                        "confidence": round(min(0.85, 0.66 + (b_area / total_area) * 3.0), 2),
                        "bbox": cand_box,
                        "area_ratio": round(b_area / total_area, 4),
                    }
                )

        return supplemental

    def _run_hybrid_yolo_inference(
        self, pil_img: Image.Image
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Execute multi-pass YOLOv8 ONNX/PyTorch detection:
          1. COCO context + non-waste gate + identifiable waste objects.
          2. Custom TACO detector across full frame, foreground region, and 2x2 tiles.
          3. Spectral ROI refinement & pure-NumPy complementary cluster check.
        Returns (detections, is_portrait_or_indoor).
        """
        width, height = pil_img.size
        total_area = float(max(1, width * height))

        # Precompute HSV map once in Pillow -> OpenCV Hue scale [0..179]
        hsv_full = np.asarray(pil_img.convert("HSV"), dtype=np.uint8).copy()
        hsv_full[:, :, 0] = (hsv_full[:, :, 0].astype(np.uint16) * 180 // 256).astype(np.uint8)

        non_waste_boxes: List[Dict[str, int]] = []
        person_boxes: List[Dict[str, int]] = []
        indoor_boxes: List[Dict[str, int]] = []
        coco_waste_candidates: List[Dict[str, Any]] = []

        # ------------------------------------------------------------------
        # PASS 1: COCO Auxiliary Context, Portrait Suppression & Common Items
        # ------------------------------------------------------------------
        if self.coco_model is not None and self.coco_backend is not None:
            try:
                coco_preds = self._infer_crop(
                    self.coco_backend,
                    self.coco_model,
                    self.coco_meta,
                    pil_img,
                    conf_thresh=0.22,
                    imgsz=640,
                    class_agnostic_nms=False,
                )
                for _, raw_cls, conf, x1, y1, x2, y2 in coco_preds:
                    cls_low = raw_cls.strip().lower()
                    box_dict = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                    box_area = float(max(0, x2 - x1) * max(0, y2 - y1))

                    if cls_low in COCO_NON_WASTE_CLASSES:
                        # Only living subjects (person, cat, dog) suppress detections inside their body
                        # Furniture, vehicles, and backgrounds (bed/couch/car/table) NEVER suppress waste
                        if cls_low in COCO_HUMAN_AND_PET_CLASSES:
                            non_waste_boxes.append(box_dict)
                        if cls_low == "person":
                            person_boxes.append(box_dict)
                        if cls_low in INDOOR_PORTRAIT_ANCHORS:
                            indoor_boxes.append(box_dict)
                        continue

                    if cls_low in COCO_WASTE_CLASS_MAP:
                        mapped_cat, mapped_sub, min_req_conf = COCO_WASTE_CLASS_MAP[cls_low]
                        if conf < min_req_conf:
                            continue
                        if box_area < total_area * 0.0002 or box_area > total_area * 0.75:
                            continue
                        coco_waste_candidates.append(
                            {
                                "class_name": mapped_cat,
                                "subclass": mapped_sub,
                                "confidence": round(float(conf), 2),
                                "bbox": box_dict,
                                "area_ratio": round(box_area / total_area, 4),
                                "_score_boost": 0.0,
                            }
                        )
            except Exception as exc:
                logger.warning("Auxiliary COCO YOLO pass error: %s", exc)

        person_union = self._compute_boxes_union_ratio(person_boxes, width, height)

        raw_detections: List[Dict[str, Any]] = []
        for cw in coco_waste_candidates:
            if any(_compute_containment(cw["bbox"], nw) > 0.60 for nw in non_waste_boxes):
                continue
            raw_detections.append(cw)

        # ------------------------------------------------------------------
        # PASS 2: Custom Trained TACO YOLOv8n (Full Frame + Foreground + Tiles)
        # ------------------------------------------------------------------
        if self.model is not None and self.custom_backend is not None:
            names_map = self.model_meta.get("names") or {}
            is_granular = len(names_map) >= 10 or (not names_map)

            passes: List[Tuple[Image.Image, int, int, float, float, float]] = [
                (pil_img, 0, 0, max(0.15, CONFIDENCE_THRESHOLD - 0.03), 0.03, 0.85)
            ]

            # Add foreground crop to zoom into dense roadside piles
            if max(width, height) >= 420:
                fx1 = int(width * 0.05)
                fy1 = int(height * 0.20)
                fx2 = int(width * 0.95)
                fy2 = int(height * 0.96)
                if fx2 - fx1 >= 240 and fy2 - fy1 >= 200:
                    passes.append(
                        (
                            pil_img.crop((fx1, fy1, fx2, fy2)),
                            fx1,
                            fy1,
                            max(0.18, CONFIDENCE_THRESHOLD),
                            0.0,
                            0.75,
                        )
                    )

            # Add 2x2 overlapping tiles on large images / multi-scene collages
            if width >= 540 and height >= 400:
                tx_mid_r = int(width * 0.56)
                tx_mid_l = int(width * 0.44)
                ty_mid_b = int(height * 0.56)
                ty_mid_t = int(height * 0.44)
                tile_coords = (
                    (0, 0, tx_mid_r, ty_mid_b),
                    (tx_mid_l, 0, width, ty_mid_b),
                    (0, ty_mid_t, tx_mid_r, height),
                    (tx_mid_l, ty_mid_t, width, height),
                )
                for tx1, ty1, tx2, ty2 in tile_coords:
                    passes.append(
                        (
                            pil_img.crop((tx1, ty1, tx2, ty2)),
                            tx1,
                            ty1,
                            max(0.16, CONFIDENCE_THRESHOLD),
                            0.0,
                            0.65,
                        )
                    )

            for idx_pass, (crop_im, ox, oy, pass_conf, rank_boost, max_area_frac) in enumerate(passes):
                try:
                    preds = self._infer_crop(
                        self.custom_backend,
                        self.model,
                        self.model_meta,
                        crop_im,
                        conf_thresh=pass_conf,
                        imgsz=512,
                        class_agnostic_nms=True,
                    )
                    for _, raw_name, conf, x1, y1, x2, y2 in preds:
                        gx1 = max(0, min(width, x1 + ox))
                        gy1 = max(0, min(height, y1 + oy))
                        gx2 = max(0, min(width, x2 + ox))
                        gy2 = max(0, min(height, y2 + oy))
                        bw = gx2 - gx1
                        bh = gy2 - gy1
                        if bw < 8 or bh < 8:
                            continue

                        box_area = float(bw * bh)
                        area_frac = box_area / total_area
                        # Allow small litter down to 0.00015 (captures small wrappers, caps, straws)
                        max_allowed = 0.85 if idx_pass == 0 and conf >= 0.35 else max_area_frac
                        if area_frac < 0.00015 or area_frac > max_allowed:
                            continue

                        bbox_dict = {"x1": gx1, "y1": gy1, "x2": gx2, "y2": gy2}
                        # Reject boxes overlapping people or pets (waste near vehicles is retained)
                        if any(_compute_containment(bbox_dict, nw) > 0.60 for nw in non_waste_boxes):
                            continue

                        raw_sub = (
                            map_taco_class_to_granular(raw_name)
                            if is_granular
                            else str(raw_name)
                        )
                        initial_cat = map_taco_class_to_category(raw_sub)
                        roi_hsv = hsv_full[gy1:gy2, gx1:gx2]
                        refined_cat, refined_sub, refined_conf = self._refine_detection_from_roi(
                            roi_hsv,
                            initial_cat,
                            raw_sub,
                            conf,
                            is_granular_model=is_granular,
                        )

                        raw_detections.append(
                            {
                                "class_name": refined_cat,
                                "subclass": refined_sub,
                                "confidence": round(refined_conf, 2),
                                "bbox": bbox_dict,
                                "area_ratio": round(area_frac, 4),
                                "_score_boost": rank_boost,
                            }
                        )
                except Exception as exc:
                    logger.warning("TACO pass %d error: %s", idx_pass, exc)

        # ------------------------------------------------------------------
        # PASS 3: Unified Confidence & Containment NMS on Distinct YOLO Objects
        # ------------------------------------------------------------------
        raw_detections.sort(
            key=lambda d: float(d["confidence"]) + float(d.get("_score_boost", 0.0)),
            reverse=True,
        )
        final_detections: List[Dict[str, Any]] = []

        for cand in raw_detections:
            duplicate = False
            cand_area = float(
                max(1, (cand["bbox"]["x2"] - cand["bbox"]["x1"]) * (cand["bbox"]["y2"] - cand["bbox"]["y1"]))
            )
            for kept in final_detections:
                iou = _compute_iou(cand["bbox"], kept["bbox"])
                containment = _compute_containment(cand["bbox"], kept["bbox"])
                reverse_contain = _compute_containment(kept["bbox"], cand["bbox"])
                same_sub = cand.get("subclass") == kept.get("subclass")
                same_cat = cand["class_name"] == kept["class_name"]
                kept_area = float(
                    max(1, (kept["bbox"]["x2"] - kept["bbox"]["x1"]) * (kept["bbox"]["y2"] - kept["bbox"]["y1"]))
                )

                if iou > 0.42:
                    duplicate = True
                    break
                if same_cat and (containment > 0.72 or (same_sub and reverse_contain > 0.78)):
                    duplicate = True
                    break
                if (
                    kept["confidence"] >= 0.72
                    and containment >= 0.88
                    and cand_area < kept_area * 0.16
                    and cand["confidence"] < 0.45
                    and cand.get("subclass") in ("Mixed litter", "Dry leaves & foliage", "Food & organic waste")
                ):
                    duplicate = True
                    break
            if not duplicate:
                clean_det = {
                    k: v for k, v in cand.items() if k != "_score_boost"
                }
                final_detections.append(clean_det)
            if len(final_detections) >= 24:
                break

        # ------------------------------------------------------------------
        # PASS 4: Complementary Pure-NumPy Spectral Check (Dense Piles Only)
        # Uses deduplicated distinct YOLO objects so repeated tile boxes never
        # inflate object count on simple 1-3 object scenes.
        # ------------------------------------------------------------------
        if len(final_detections) >= 4:
            supplemental = self._segment_complementary_waste_numpy(
                pil_img, final_detections, non_waste_boxes, width, height
            )
            for item in supplemental:
                if len(final_detections) >= 24:
                    break
                dup_sup = any(
                    _compute_iou(item["bbox"], kept["bbox"]) > 0.42
                    or (
                        item["class_name"] == kept["class_name"]
                        and _compute_containment(item["bbox"], kept["bbox"]) > 0.65
                    )
                    for kept in final_detections
                )
                if not dup_sup:
                    final_detections.append(item)

        final_detections.sort(key=lambda d: float(d["confidence"]), reverse=True)
        return final_detections, False

    def _classify_region_heuristically(
        self, roi_bgr: np.ndarray, roi_hsv: np.ndarray
    ) -> Tuple[str, str, float]:
        """
        Deterministically classify a candidate waste region into one of the three
        functional categories based on HSV chromaticity and luminance statistics.
        """
        if roi_bgr.size == 0 or roi_hsv.size == 0:
            return "Plastic Waste", "Mixed litter", 0.78

        mean_h = float(np.mean(roi_hsv[:, :, 0]))
        mean_s = float(np.mean(roi_hsv[:, :, 1]))
        mean_v = float(np.mean(roi_hsv[:, :, 2]))
        std_v = float(np.std(roi_hsv[:, :, 2]))

        if (5.0 <= mean_h <= 28.0 and mean_s > 95.0) or (mean_h > 160.0 and mean_s > 110.0):
            conf = min(0.92, 0.72 + (mean_s / 1000.0))
            return "Organic Waste", "Food & organic waste", round(conf, 2)

        if 15.0 <= mean_h <= 65.0 and 40.0 <= mean_s <= 145.0 and mean_v < 185.0:
            conf = min(0.90, 0.70 + (std_v / 300.0))
            return "Dry Leaves / Biomass", "Dry leaves & foliage", round(conf, 2)

        conf = min(0.93, 0.74 + (std_v / 280.0))
        return "Plastic Waste", "Mixed litter", round(conf, 2)

    def _run_analytical_fallback_numpy(
        self, img_bgr: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Pure NumPy + Pillow analytical segmentation pipeline used ONLY when
        neither ONNX Runtime nor PyTorch YOLO weights are available.
        """
        height, width = img_bgr.shape[:2]
        total_area = float(width * height)
        if total_area <= 0:
            return []

        img_rgb = img_bgr[:, :, ::-1]
        pil_rgb = Image.fromarray(img_rgb, mode="RGB")
        hsv_arr = np.array(pil_rgb.convert("HSV"), dtype=np.uint8)
        hsv_arr[:, :, 0] = (hsv_arr[:, :, 0].astype(np.uint16) * 180 // 256).astype(np.uint8)

        grid_w = min(96, max(24, width // 6))
        grid_h = min(72, max(18, height // 6))
        small_rgb = np.array(
            pil_rgb.resize((grid_w, grid_h), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        small_gray = (
            0.299 * small_rgb[:, :, 0]
            + 0.587 * small_rgb[:, :, 1]
            + 0.114 * small_rgb[:, :, 2]
        )

        border_rgb = np.concatenate(
            [small_rgb[0, :], small_rgb[-1, :], small_rgb[:, 0], small_rgb[:, -1]],
            axis=0,
        )
        bg_median = np.median(border_rgb, axis=0)
        color_dist = np.linalg.norm(small_rgb - bg_median, axis=2)

        gx = np.abs(np.diff(small_gray, axis=1, prepend=small_gray[:, :1]))
        gy = np.abs(np.diff(small_gray, axis=0, prepend=small_gray[:1, :]))
        saliency = color_dist + 0.65 * (gx + gy)
        thresh = max(22.0, min(65.0, float(np.percentile(saliency, 68))))
        fg_mask = saliency > thresh

        padded = np.pad(fg_mask, 1, mode="constant", constant_values=False)
        neighbor_count = sum(
            padded[dy : dy + grid_h, dx : dx + grid_w].astype(np.uint8)
            for dy in range(3)
            for dx in range(3)
        )
        fg_mask = neighbor_count >= 3

        candidate_boxes = _extract_connected_components(
            fg_mask,
            total_area=total_area,
            min_area_ratio=0.012,
            max_area_ratio=0.85,
            img_width=width,
            img_height=height,
        )

        selected_boxes: List[Tuple[int, int, int, int, float]] = []
        for box in candidate_boxes:
            x1, y1, x2, y2, area = box
            overlaps = False
            for sx1, sy1, sx2, sy2, _ in selected_boxes:
                ix1 = max(x1, sx1)
                iy1 = max(y1, sy1)
                ix2 = min(x2, sx2)
                iy2 = min(y2, sy2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = float((ix2 - ix1) * (iy2 - iy1))
                    if inter / min(area, float((sx2 - sx1) * (sy2 - sy1))) > 0.45:
                        overlaps = True
                        break
            if not overlaps:
                selected_boxes.append(box)
            if len(selected_boxes) >= 8:
                break

        if not selected_boxes:
            pad_w = int(width * 0.18)
            pad_h = int(height * 0.18)
            x1, y1 = pad_w, pad_h
            x2, y2 = max(x1 + 20, width - pad_w), max(y1 + 20, height - pad_h)
            selected_boxes.append((x1, y1, x2, y2, float((x2 - x1) * (y2 - y1))))

        detections: List[Dict[str, Any]] = []
        for x1, y1, x2, y2, box_area in selected_boxes:
            roi_bgr = img_bgr[y1:y2, x1:x2]
            roi_hsv = hsv_arr[y1:y2, x1:x2]
            cat_name, sub_name, confidence = self._classify_region_heuristically(roi_bgr, roi_hsv)
            detections.append(
                {
                    "class_name": cat_name,
                    "subclass": sub_name,
                    "confidence": confidence,
                    "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                    "area_ratio": round(box_area / total_area, 4),
                }
            )
        return detections

    def _run_analytical_fallback(
        self, img_bgr: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Deterministic OpenCV contour and color/texture segmentation pipeline.
        Only executed when YOLO weights or runtimes are missing.
        """
        if not HAS_CV2 or cv2 is None:
            return self._run_analytical_fallback_numpy(img_bgr)

        height, width = img_bgr.shape[:2]
        total_area = float(width * height)
        if total_area <= 0:
            return []

        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blurred, 45, 135)
        _, thresh_otsu = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        border_pixels = np.concatenate(
            [gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]]
        )
        if float(np.mean(border_pixels)) < 110.0:
            _, thresh_otsu = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.bitwise_or(closed_edges, thresh_otsu)
        combined_mask = cv2.morphologyEx(
            combined_mask, cv2.MORPH_OPEN, kernel, iterations=1
        )

        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidate_boxes: List[Tuple[int, int, int, int, float]] = []
        min_box_area = total_area * 0.012
        max_box_area = total_area * 0.85

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            box_area = float(w * h)
            if min_box_area <= box_area <= max_box_area:
                candidate_boxes.append((x, y, x + w, y + h, box_area))

        candidate_boxes.sort(key=lambda b: b[4], reverse=True)
        selected_boxes: List[Tuple[int, int, int, int, float]] = []

        for box in candidate_boxes:
            x1, y1, x2, y2, area = box
            overlaps = False
            for sx1, sy1, sx2, sy2, _ in selected_boxes:
                ix1 = max(x1, sx1)
                iy1 = max(y1, sy1)
                ix2 = min(x2, sx2)
                iy2 = min(y2, sy2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = float((ix2 - ix1) * (iy2 - iy1))
                    if inter / min(area, float((sx2 - sx1) * (sy2 - sy1))) > 0.45:
                        overlaps = True
                        break
            if not overlaps:
                selected_boxes.append(box)
            if len(selected_boxes) >= 8:
                break

        if not selected_boxes:
            pad_w = int(width * 0.18)
            pad_h = int(height * 0.18)
            x1, y1 = pad_w, pad_h
            x2, y2 = max(x1 + 20, width - pad_w), max(y1 + 20, height - pad_h)
            selected_boxes.append((x1, y1, x2, y2, float((x2 - x1) * (y2 - y1))))

        detections: List[Dict[str, Any]] = []
        for x1, y1, x2, y2, box_area in selected_boxes:
            roi_bgr = img_bgr[y1:y2, x1:x2]
            roi_hsv = img_hsv[y1:y2, x1:x2]
            cat_name, sub_name, confidence = self._classify_region_heuristically(roi_bgr, roi_hsv)
            detections.append(
                {
                    "class_name": cat_name,
                    "subclass": sub_name,
                    "confidence": confidence,
                    "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                    "area_ratio": round(box_area / total_area, 4),
                }
            )

        return detections

    def analyze_image(
        self, image_source: Union[str, Path, bytes, Image.Image]
    ) -> Dict[str, Any]:
        """
        Run waste detection and severity analysis on an image.
        Returns bounding boxes, functional categories + specific TACO subclasses,
        severity score, severity tier, and inference mode ("YOLOv8" or "Analytical Fallback").
        """
        if isinstance(image_source, Image.Image):
            pil_img = ImageOps.exif_transpose(image_source).convert("RGB")
        elif isinstance(image_source, (bytes, bytearray)):
            with Image.open(io.BytesIO(image_source)) as opened:
                pil_img = ImageOps.exif_transpose(opened).convert("RGB")
        else:
            with Image.open(Path(image_source)) as opened:
                pil_img = ImageOps.exif_transpose(opened).convert("RGB")

        # Memory optimization: resize high-resolution images to max 1024px to prevent OOM
        max_dim = max(pil_img.size)
        if max_dim > 1024:
            scale = 1024.0 / max_dim
            new_w = max(32, int(round(pil_img.size[0] * scale)))
            new_h = max(32, int(round(pil_img.size[1] * scale)))
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        img_width, img_height = pil_img.size
        detections: List[Dict[str, Any]] = []
        inference_mode = "Analytical Fallback"

        if self.yolo_available and (self.model is not None or self.coco_model is not None):
            try:
                detections, _ = self._run_hybrid_yolo_inference(pil_img)
                inference_mode = "YOLOv8"
            except Exception as exc:
                logger.warning("YOLOv8 inference error, switching to fallback: %s", exc)
                img_rgb = np.array(pil_img)
                img_bgr = img_rgb[:, :, ::-1].copy()
                detections = self._run_analytical_fallback(img_bgr)
                inference_mode = "Analytical Fallback"
        else:
            img_rgb = np.array(pil_img)
            img_bgr = img_rgb[:, :, ::-1].copy()
            detections = self._run_analytical_fallback(img_bgr)
            inference_mode = "Analytical Fallback"

        severity_data = calculate_severity(detections, img_width, img_height)

        category_counts: Dict[str, int] = {}
        category_confidences: Dict[str, List[float]] = {}
        all_confidences: List[float] = []
        for det in detections:
            cat = det["class_name"]
            conf = float(det.get("confidence", 0.0))
            category_counts[cat] = category_counts.get(cat, 0) + 1
            category_confidences.setdefault(cat, []).append(conf)
            all_confidences.append(conf)

        category_stats: Dict[str, Dict[str, Any]] = {}
        for cat, count in category_counts.items():
            confs = category_confidences.get(cat, [0.0])
            category_stats[cat] = {
                "count": count,
                "avg_confidence": round(sum(confs) / len(confs), 3),
                "max_confidence": round(max(confs), 3),
            }

        highest_conf = round(max(all_confidences), 3) if all_confidences else 0.0
        avg_conf = round(sum(all_confidences) / len(all_confidences), 3) if all_confidences else 0.0

        return {
            "image_width": img_width,
            "image_height": img_height,
            "detections": detections,
            "category_counts": category_counts,
            "category_stats": category_stats,
            "detected_categories": list(category_counts.keys()),
            "detection_count": len(detections),
            "highest_confidence": highest_conf,
            "average_confidence": avg_conf,
            "severity_score": severity_data["severity_score"],
            "severity_tier": severity_data["severity_tier"],
            "severity_breakdown": severity_data,
            "inference_mode": inference_mode,
        }


CATEGORY_PLASTIC = "Plastic Waste"
CATEGORY_ORGANIC = "Organic Waste"
CATEGORY_DRY_BIOMASS = "Dry Leaves / Biomass"

_CATEGORY_TO_DB_ENUM = {
    CATEGORY_PLASTIC: "PLASTIC_WASTE",
    CATEGORY_ORGANIC: "ORGANIC_WASTE",
    CATEGORY_DRY_BIOMASS: "DRY_LEAVES_BIOMASS",
    "PLASTIC_WASTE": "PLASTIC_WASTE",
    "ORGANIC_WASTE": "ORGANIC_WASTE",
    "DRY_LEAVES_BIOMASS": "DRY_LEAVES_BIOMASS",
}

_DB_ENUM_TO_CATEGORY = {
    "PLASTIC_WASTE": CATEGORY_PLASTIC,
    "ORGANIC_WASTE": CATEGORY_ORGANIC,
    "DRY_LEAVES_BIOMASS": CATEGORY_DRY_BIOMASS,
    CATEGORY_PLASTIC: CATEGORY_PLASTIC,
    CATEGORY_ORGANIC: CATEGORY_ORGANIC,
    CATEGORY_DRY_BIOMASS: CATEGORY_DRY_BIOMASS,
}


def map_category_to_db_enum(category: str) -> str:
    """Convert a functional category name into its normalized DB enum string."""
    if not category:
        return "PLASTIC_WASTE"
    if category in _CATEGORY_TO_DB_ENUM:
        return _CATEGORY_TO_DB_ENUM[category]
    low = category.lower()
    if "organic" in low or "food" in low:
        return "ORGANIC_WASTE"
    if "dry" in low or "leaf" in low or "leaves" in low or "biomass" in low:
        return "DRY_LEAVES_BIOMASS"
    return "PLASTIC_WASTE"


def map_db_enum_to_category(db_enum: str) -> str:
    """Convert a DB enum string back to its display functional category name."""
    if not db_enum:
        return CATEGORY_PLASTIC
    return _DB_ENUM_TO_CATEGORY.get(db_enum, CATEGORY_PLASTIC)


# Singleton vision engine instance
vision_engine = VisionEngine()


def analyze_waste_image(
    image_source: Union[str, Path, bytes, Image.Image],
) -> Dict[str, Any]:
    """Module-level convenience function to run waste detection & severity scoring."""
    return vision_engine.analyze_image(image_source)


def get_vision_status() -> Dict[str, Any]:
    """Module-level convenience function to retrieve AI engine status."""
    return vision_engine.get_status()
