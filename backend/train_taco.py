"""
TACO Dataset Conversion & YOLOv8n Training Pipeline for CleanPath AI.

This module is also the single source of truth for functional waste categories,
TACO subclass mappings, and municipal hazard multipliers.
"""

import argparse
import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger("cleanpath.train_taco")

# ==============================================================================
# 1. FUNCTIONAL WASTE CATEGORIES & HAZARD MULTIPLIERS (Single Source of Truth)
# ==============================================================================

FUNCTIONAL_CATEGORIES: List[str] = [
    "Plastic Waste",
    "Organic Waste",
    "Dry Leaves / Biomass",
]

CATEGORY_ID_MAP: Dict[str, int] = {
    "Plastic Waste": 0,
    "Organic Waste": 1,
    "Dry Leaves / Biomass": 2,
}

ID_TO_CATEGORY_MAP: Dict[int, str] = {
    idx: name for name, idx in CATEGORY_ID_MAP.items()
}

CATEGORY_MULTIPLIERS: Dict[str, float] = {
    "Plastic Waste": 1.0,
    "Organic Waste": 1.2,
    "Dry Leaves / Biomass": 0.7,
}

# Complete mapping dictionary from all 60 TACO dataset classes to the 3 functional categories
TACO_TO_FUNCTIONAL_MAPPING: Dict[str, str] = {
    # Plastic, Synthetic, Container & Packaging Waste -> Plastic Waste
    "Clear plastic bottle": "Plastic Waste",
    "Other plastic bottle": "Plastic Waste",
    "Plastic bottle cap": "Plastic Waste",
    "Metal bottle cap": "Plastic Waste",
    "Disposable plastic cup": "Plastic Waste",
    "Other plastic cup": "Plastic Waste",
    "Plastic lid": "Plastic Waste",
    "Single-use carrier bag": "Plastic Waste",
    "Polypropylene bag": "Plastic Waste",
    "Plastic film": "Plastic Waste",
    "Crisp packet": "Plastic Waste",
    "Other plastic wrapper": "Plastic Waste",
    "Other plastic": "Plastic Waste",
    "Other plastic container": "Plastic Waste",
    "Plastic glooves": "Plastic Waste",
    "Plastic utensils": "Plastic Waste",
    "Plastic straw": "Plastic Waste",
    "Six pack rings": "Plastic Waste",
    "Squeezable tube": "Plastic Waste",
    "Styrofoam piece": "Plastic Waste",
    "Foam cup": "Plastic Waste",
    "Foam food container": "Organic Waste",
    "Disposable food container": "Organic Waste",
    "Drink can": "Plastic Waste",
    "Food can": "Organic Waste",
    "Aerosol": "Plastic Waste",
    "Glass bottle": "Plastic Waste",
    "Broken glass": "Plastic Waste",
    "Glass cup": "Plastic Waste",
    "Glass jar": "Plastic Waste",
    "Pop tab": "Plastic Waste",
    "Metal lid": "Plastic Waste",
    "Scrap metal": "Plastic Waste",
    "Shoe": "Plastic Waste",
    "Battery": "Plastic Waste",
    "Blister pack": "Plastic Waste",
    "Unlabeled litter": "Plastic Waste",
    # Food & Putrescible Waste -> Organic Waste
    "Food waste": "Organic Waste",
    "Pizza box": "Organic Waste",
    "Meal carton": "Organic Waste",
    "Drink carton": "Organic Waste",
    "Spread tub": "Organic Waste",
    "Tupperware": "Organic Waste",
    "Aluminium blister pack": "Plastic Waste",
    "Aluminium foil": "Organic Waste",
    # Paper, Cardboard, Yard, Wood & Natural Biomass -> Dry Leaves / Biomass
    "Normal paper": "Dry Leaves / Biomass",
    "Paper bag": "Dry Leaves / Biomass",
    "Paper cup": "Dry Leaves / Biomass",
    "Paper straw": "Dry Leaves / Biomass",
    "Magazine paper": "Dry Leaves / Biomass",
    "Tissues": "Dry Leaves / Biomass",
    "Wrapping paper": "Dry Leaves / Biomass",
    "Corrugated carton": "Dry Leaves / Biomass",
    "Egg carton": "Dry Leaves / Biomass",
    "Other carton": "Dry Leaves / Biomass",
    "Toilet tube": "Dry Leaves / Biomass",
    "Rope & strings": "Dry Leaves / Biomass",
    "Cigarette": "Dry Leaves / Biomass",
    "Carded blister pack": "Plastic Waste",
    "Garbage bag": "Plastic Waste",
}


def map_taco_class_to_category(class_name: str) -> str:
    """
    Map any TACO subclass (or general YOLO/COCO class name) into one of the three
    CleanPath AI functional categories:
        - "Plastic Waste"
        - "Organic Waste"
        - "Dry Leaves / Biomass"
    """
    if not class_name:
        return "Plastic Waste"

    # Direct lookup if already a functional category
    if class_name in CATEGORY_MULTIPLIERS:
        return class_name

    # Exact TACO class lookup
    if class_name in TACO_TO_FUNCTIONAL_MAPPING:
        return TACO_TO_FUNCTIONAL_MAPPING[class_name]

    # Case-insensitive lookup
    lower_name = class_name.strip().lower()
    for taco_cls, target_cat in TACO_TO_FUNCTIONAL_MAPPING.items():
        if taco_cls.lower() == lower_name:
            return target_cat

    # Keyword heuristic fallback (handles COCO classes or custom labels)
    organic_keywords = (
        "food",
        "organic",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "fruit",
        "vegetable",
        "peel",
        "compost",
        "meal",
    )
    biomass_keywords = (
        "leaf",
        "leaves",
        "biomass",
        "paper",
        "carton",
        "cardboard",
        "wood",
        "branch",
        "plant",
        "potted plant",
        "book",
        "tissue",
        "grass",
        "twig",
        "cigarette",
        "rope",
    )
    plastic_keywords = (
        "plastic",
        "bottle",
        "cup",
        "bag",
        "wrapper",
        "container",
        "lid",
        "straw",
        "can",
        "foam",
        "styrofoam",
        "utensil",
        "handbag",
        "backpack",
        "suitcase",
        "umbrella",
        "vase",
        "bowl",
        "wine glass",
        "fork",
        "knife",
        "spoon",
        "cell phone",
    )

    if any(k in lower_name for k in organic_keywords):
        return "Organic Waste"
    if any(k in lower_name for k in biomass_keywords):
        return "Dry Leaves / Biomass"
    if any(k in lower_name for k in plastic_keywords):
        return "Plastic Waste"

    return "Plastic Waste"


def map_taco_class_to_id(class_name: str) -> int:
    """Return integer class ID (0, 1, or 2) for a given TACO or raw class name."""
    category = map_taco_class_to_category(class_name)
    return CATEGORY_ID_MAP[category]


# ==============================================================================
# 2. TACO COCO TO YOLOv8 FORMAT CONVERSION
# ==============================================================================


def locate_taco_dataset(dataset_path: Optional[Path] = None) -> Path:
    """Locate the TACO dataset directory containing annotations.json."""
    candidates: List[Path] = []
    if dataset_path is not None:
        candidates.append(Path(dataset_path))

    repo_root = Path(__file__).resolve().parent.parent
    candidates.extend(
        [
            repo_root / "data",
            Path(__file__).resolve().parent / "data",
            Path.cwd() / "data",
        ]
    )

    for candidate in candidates:
        ann_file = candidate / "annotations.json"
        if ann_file.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate TACO annotations.json. Checked: "
        + ", ".join(str(c) for c in candidates)
    )


def coco_bbox_to_yolo(
    bbox: List[float], img_width: float, img_height: float
) -> Tuple[float, float, float, float]:
    """
    Convert COCO [x_min, y_min, width, height] into normalized YOLO
    [x_center, y_center, norm_width, norm_height] clipped to [0.0, 1.0].
    """
    if img_width <= 0 or img_height <= 0:
        raise ValueError("Image dimensions must be positive")

    x_min, y_min, width, height = [float(v) for v in bbox]
    x_center = (x_min + width / 2.0) / img_width
    y_center = (y_min + height / 2.0) / img_height
    norm_w = width / img_width
    norm_h = height / img_height

    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    norm_w = max(0.0, min(1.0, norm_w))
    norm_h = max(0.0, min(1.0, norm_h))

    return x_center, y_center, norm_w, norm_h


def convert_taco_to_yolo(
    dataset_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    train_ratio: float = 0.75,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Path:
    """
    Convert TACO COCO annotations into YOLO dataset structure with
    train/, val/, and test/ splits plus data.yaml.
    """
    taco_dir = locate_taco_dataset(dataset_path)
    annotations_file = taco_dir / "annotations.json"

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "taco_yolo_dataset"
    output_dir = Path(output_dir).resolve()

    with open(annotations_file, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Map COCO category_id -> CleanPath functional class ID (0, 1, 2)
    category_id_lookup: Dict[int, int] = {}
    for cat in coco_data.get("categories", []):
        cat_id = int(cat["id"])
        cat_name = str(cat.get("name", ""))
        category_id_lookup[cat_id] = map_taco_class_to_id(cat_name)

    # Create split directories
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Group annotations by image_id
    annotations_by_image: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco_data.get("annotations", []):
        img_id = int(ann["image_id"])
        annotations_by_image.setdefault(img_id, []).append(ann)

    images = list(coco_data.get("images", []))
    rng = random.Random(seed)
    rng.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    converted_counts = {"train": 0, "val": 0, "test": 0}

    for idx, img_info in enumerate(images):
        if idx < train_end:
            split = "train"
        elif idx < val_end:
            split = "val"
        else:
            split = "test"

        img_id = int(img_info["id"])
        rel_file = img_info["file_name"]
        src_img_path = taco_dir / rel_file

        if not src_img_path.exists():
            continue

        img_w = float(img_info.get("width", 0))
        img_h = float(img_info.get("height", 0))
        if img_w <= 0 or img_h <= 0:
            continue

        dest_stem = f"{img_id:06d}"
        ext = src_img_path.suffix or ".jpg"
        dest_img_path = output_dir / "images" / split / f"{dest_stem}{ext}"
        dest_lbl_path = output_dir / "labels" / split / f"{dest_stem}.txt"

        shutil.copy2(src_img_path, dest_img_path)

        label_lines: List[str] = []
        for ann in annotations_by_image.get(img_id, []):
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            target_cls_id = category_id_lookup.get(int(ann["category_id"]), 0)
            xc, yc, nw, nh = coco_bbox_to_yolo(bbox, img_w, img_h)
            label_lines.append(f"{target_cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        with open(dest_lbl_path, "w", encoding="utf-8") as lf:
            lf.write("\n".join(label_lines))

        converted_counts[split] += 1

    # Write data.yaml
    data_yaml_path = output_dir / "data.yaml"
    data_yaml_content = {
        "path": output_dir.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(FUNCTIONAL_CATEGORIES),
        "names": {idx: name for idx, name in enumerate(FUNCTIONAL_CATEGORIES)},
    }
    with open(data_yaml_path, "w", encoding="utf-8") as yf:
        yaml.safe_dump(data_yaml_content, yf, sort_keys=False)

    logger.info(
        "TACO conversion complete at %s (train=%d, val=%d, test=%d)",
        output_dir,
        converted_counts["train"],
        converted_counts["val"],
        converted_counts["test"],
    )
    return data_yaml_path


# ==============================================================================
# 3. YOLOv8n TRAINING WORKFLOW
# ==============================================================================


def select_compute_device(requested_device: Optional[str] = None) -> str:
    """Automatically select CUDA if available, otherwise default safely to CPU."""
    if requested_device:
        return requested_device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def train_yolov8_model(
    data_yaml_path: Path,
    epochs: int = 10,
    image_size: int = 640,
    batch_size: int = 8,
    device: Optional[str] = None,
    base_weights: str = "yolov8n.pt",
) -> Path:
    """
    Train YOLOv8n on the converted TACO dataset and save the best checkpoint
    to backend/models/best.pt.
    """
    from ultralytics import YOLO

    compute_device = select_compute_device(device)
    backend_dir = Path(__file__).resolve().parent
    models_dir = backend_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_target_path = models_dir / "best.pt"

    project_runs_dir = backend_dir / "runs" / "detect"
    model = YOLO(base_weights)

    model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=compute_device,
        project=str(project_runs_dir),
        name="cleanpath_yolov8n",
        exist_ok=True,
    )

    trained_best = project_runs_dir / "cleanpath_yolov8n" / "weights" / "best.pt"
    if trained_best.exists():
        shutil.copy2(trained_best, best_target_path)
        logger.info("Saved best YOLOv8 weights to %s", best_target_path)
    else:
        last_weights = project_runs_dir / "cleanpath_yolov8n" / "weights" / "last.pt"
        if last_weights.exists():
            shutil.copy2(last_weights, best_target_path)
            logger.info("Saved last YOLOv8 weights to %s", best_target_path)

    return best_target_path


def main() -> None:
    """CLI entrypoint for converting TACO and training YOLOv8n."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="CleanPath AI: Convert TACO dataset and train YOLOv8n waste detector."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to TACO data directory containing annotations.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for YOLO-formatted dataset",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5 for college laptop compatibility)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Training image resolution (default: 640)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device: 'cpu' or 'cuda' (auto-detected if omitted)",
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Only convert TACO annotations to YOLO format without starting training",
    )

    args = parser.parse_args()
    ds_path = Path(args.dataset_path) if args.dataset_path else None
    out_path = Path(args.output_dir) if args.output_dir else None

    yaml_path = convert_taco_to_yolo(dataset_path=ds_path, output_dir=out_path)
    print(f"Dataset conversion complete. YAML generated at: {yaml_path}")

    if not args.convert_only:
        best_pt = train_yolov8_model(
            data_yaml_path=yaml_path,
            epochs=args.epochs,
            image_size=args.image_size,
            batch_size=args.batch_size,
            device=args.device,
        )
        print(f"Training complete. Best weights saved at: {best_pt}")


if __name__ == "__main__":
    main()
