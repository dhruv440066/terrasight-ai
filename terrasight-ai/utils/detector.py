"""
YOLOv8 Computer Vision Detection Pipeline
=========================================
Loads and caches YOLOv8 object detection models using Streamlit's
@st.cache_resource decorator. Performs inference, computes bounding boxes,
class distributions, and renders annotated visual outputs.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from ultralytics import YOLO

# Category color palette (RGB)
CATEGORY_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Plastic Waste": (37, 99, 235),       # Blue
    "Organic Waste": (234, 88, 12),       # Orange
    "Dry Leaves / Biomass": (22, 163, 74),# Green
    "Metal Debris": (168, 85, 247),       # Purple
    "General Waste": (220, 38, 38),       # Red
}
DEFAULT_BOX_COLOR: Tuple[int, int, int] = (147, 51, 234)

# COCO class mapping fallback if generic yolov8n is used
COCO_FALLBACK_MAP: Dict[str, str] = {
    "bottle": "Plastic Waste",
    "cup": "Plastic Waste",
    "fork": "Plastic Waste",
    "knife": "Plastic Waste",
    "spoon": "Plastic Waste",
    "bowl": "Plastic Waste",
    "banana": "Organic Waste",
    "apple": "Organic Waste",
    "sandwich": "Organic Waste",
    "orange": "Organic Waste",
    "broccoli": "Organic Waste",
    "carrot": "Organic Waste",
    "pizza": "Organic Waste",
    "donut": "Organic Waste",
    "cake": "Organic Waste",
    "potted plant": "Dry Leaves / Biomass",
}


@st.cache_resource(show_spinner=False)
def load_yolo_model(weights_path: str = "weights/best.pt") -> Tuple[YOLO, str]:
    """
    Load YOLOv8 model with Streamlit resource caching to prevent re-instantiation.
    Falls back to pretrained yolov8n.pt if custom weights are missing.
    
    Returns:
        Tuple of (YOLO model instance, model description string)
    """
    w_path = Path(weights_path)
    
    if w_path.exists() and w_path.stat().st_size > 1000:
        try:
            model = YOLO(str(w_path))
            return model, f"Custom Civic Model ({w_path.name})"
        except Exception as e:
            st.warning(f"Could not load custom weights '{weights_path}': {e}. Falling back to standard yolov8n.")

    # Alternative local search
    alt_paths = [
        Path("weights/best.pt"),
        Path("../backend/models/best.pt"),
        Path("yolov8n.pt"),
        Path("../backend/yolov8n.pt"),
    ]
    for p in alt_paths:
        if p.exists() and p.stat().st_size > 1000:
            try:
                model = YOLO(str(p))
                return model, f"Local Weights ({p.name})"
            except Exception:
                pass

    # Automatic fallback download of standard YOLOv8 nano
    try:
        model = YOLO("yolov8n.pt")
        return model, "YOLOv8 Nano (Pretrained COCO Fallback)"
    except Exception as e:
        raise RuntimeError(f"Failed to initialize any YOLO model: {e}")


def run_detection(
    image: Image.Image,
    model: YOLO,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> Tuple[Image.Image, List[Dict[str, Any]], Dict[str, int]]:
    """
    Execute YOLOv8 inference and draw stylized bounding boxes.
    
    Args:
        image: PIL Image object
        model: Loaded YOLO model
        conf_threshold: Minimum confidence score (0.0 to 1.0)
        iou_threshold: Intersection-over-Union threshold for NMS
        
    Returns:
        Tuple of:
            - annotated_image (PIL Image)
            - detections_list (list of dicts with box, label, confidence)
            - category_counts (dict mapping label -> count)
    """
    img_rgb = image.convert("RGB")
    width, height = img_rgb.size

    # Run inference
    results = model.predict(
        source=img_rgb,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )

    detections: List[Dict[str, Any]] = []
    category_counts: Dict[str, int] = {}

    annotated = img_rgb.copy()
    draw = ImageDraw.Draw(annotated)

    # Attempt to load a scalable font
    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, int(min(width, height) * 0.025)))
    except Exception:
        font = ImageFont.load_default()

    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            x1, y1, x2, y2 = [int(coord) for coord in xyxy]
            conf = float(box.conf[0].cpu().numpy())
            cls_idx = int(box.cls[0].cpu().numpy())

            raw_label = model.names.get(cls_idx, f"Class {cls_idx}")
            label = COCO_FALLBACK_MAP.get(raw_label, raw_label)

            color = CATEGORY_COLORS.get(label, DEFAULT_BOX_COLOR)

            # Record detection item
            det_item = {
                "label": label,
                "confidence": round(conf, 3),
                "box": [x1, y1, x2, y2],
                "box_normalized": [
                    round(x1 / width, 4),
                    round(y1 / height, 4),
                    round(x2 / width, 4),
                    round(y2 / height, 4),
                ],
                "area_ratio": round(((x2 - x1) * (y2 - y1)) / (width * height), 4),
            }
            detections.append(det_item)
            category_counts[label] = category_counts.get(label, 0) + 1

            # Draw bounding box (thick outline)
            box_width = max(2, int(min(width, height) * 0.005))
            for i in range(box_width):
                draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)

            # Draw label pill
            tag_text = f"{label} {int(conf * 100)}%"
            text_bbox = draw.textbbox((x1, y1), tag_text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            pill_y0 = max(0, y1 - text_h - 8)
            pill_y1 = y1
            pill_x0 = x1
            pill_x1 = x1 + text_w + 10

            draw.rectangle([pill_x0, pill_y0, pill_x1, pill_y1], fill=color)
            draw.text((pill_x0 + 5, pill_y0 + 2), tag_text, fill=(255, 255, 255), font=font)

    return annotated, detections, category_counts
