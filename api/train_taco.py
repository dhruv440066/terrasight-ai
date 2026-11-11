"""
TACO Dataset Conversion & YOLOv8n Training Pipeline for CleanPath AI.

This module is also the single source of truth for functional waste categories,
consolidated TACO subclass mappings, and municipal hazard multipliers.
"""

import argparse
import json
import logging
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps

try:
    import yaml
except ImportError:  # Optional in serverless API runtime where only category maps are used
    yaml = None  # type: ignore[assignment]

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

# ==============================================================================
# 1B. CONSOLIDATED TACO ITEM CLASSES (Visually Coherent 13-Class Taxonomy)
# ==============================================================================
# Merging TACO's 60 raw classes (many with 1-10 samples) into 13 visually
# consistent object categories (>=107 instances each across all 4,784 boxes)
# allows YOLOv8 to learn crisp per-object bounding boxes and recognizable item
# labels while mapping 1-to-1 into CleanPath's 3 Municipal Functional Categories.

GRANULAR_TACO_CLASSES: List[str] = [
    "Plastic bottle",
    "Glass bottle & jar",
    "Drink & metal can",
    "Plastic bag & film",
    "Plastic wrapper",
    "Disposable cup",
    "Styrofoam piece",
    "Bottle cap, straw & lid",
    "Food & takeout container",
    "Cardboard carton",
    "Paper & paper bag",
    "Cigarette",
    "Mixed litter",
]

GRANULAR_CATEGORY_ID_MAP: Dict[str, int] = {
    name: idx for idx, name in enumerate(GRANULAR_TACO_CLASSES)
}

ID_TO_GRANULAR_CATEGORY_MAP: Dict[int, str] = {
    idx: name for name, idx in GRANULAR_CATEGORY_ID_MAP.items()
}

GRANULAR_TO_FUNCTIONAL_MAPPING: Dict[str, str] = {
    "Plastic bottle": "Plastic Waste",
    "Glass bottle & jar": "Plastic Waste",
    "Drink & metal can": "Plastic Waste",
    "Plastic bag & film": "Plastic Waste",
    "Plastic wrapper": "Plastic Waste",
    "Disposable cup": "Plastic Waste",
    "Styrofoam piece": "Plastic Waste",
    "Bottle cap, straw & lid": "Plastic Waste",
    "Food & takeout container": "Organic Waste",
    "Cardboard carton": "Dry Leaves / Biomass",
    "Paper & paper bag": "Dry Leaves / Biomass",
    "Cigarette": "Dry Leaves / Biomass",
    "Mixed litter": "Plastic Waste",
}

TACO_TO_GRANULAR_MAPPING: Dict[str, str] = {
    # 0. Plastic bottle (335)
    "Clear plastic bottle": "Plastic bottle",
    "Other plastic bottle": "Plastic bottle",
    # 1. Glass bottle & jar (116)
    "Glass bottle": "Glass bottle & jar",
    "Glass jar": "Glass bottle & jar",
    "Glass cup": "Glass bottle & jar",
    # 2. Drink & metal can (273)
    "Drink can": "Drink & metal can",
    "Food Can": "Drink & metal can",
    "Food can": "Drink & metal can",
    "Aerosol": "Drink & metal can",
    # 3. Plastic bag & film (550)
    "Plastic film": "Plastic bag & film",
    "Single-use carrier bag": "Plastic bag & film",
    "Garbage bag": "Plastic bag & film",
    "Polypropylene bag": "Plastic bag & film",
    "Plastified paper bag": "Plastic bag & film",
    "Plastic glooves": "Plastic bag & film",
    # 4. Plastic wrapper (312)
    "Other plastic wrapper": "Plastic wrapper",
    "Crisp packet": "Plastic wrapper",
    "Six pack rings": "Plastic wrapper",
    "Squeezable tube": "Plastic wrapper",
    "Aluminium blister pack": "Plastic wrapper",
    "Carded blister pack": "Plastic wrapper",
    "Blister pack": "Plastic wrapper",
    # 5. Disposable cup (192)
    "Disposable plastic cup": "Disposable cup",
    "Paper cup": "Disposable cup",
    "Foam cup": "Disposable cup",
    "Other plastic cup": "Disposable cup",
    # 6. Styrofoam piece (112)
    "Styrofoam piece": "Styrofoam piece",
    # 7. Bottle cap, straw & lid (673)
    "Plastic bottle cap": "Bottle cap, straw & lid",
    "Metal bottle cap": "Bottle cap, straw & lid",
    "Plastic lid": "Bottle cap, straw & lid",
    "Metal lid": "Bottle cap, straw & lid",
    "Pop tab": "Bottle cap, straw & lid",
    "Plastic straw": "Bottle cap, straw & lid",
    "Paper straw": "Bottle cap, straw & lid",
    "Plastic utensils": "Bottle cap, straw & lid",
    # 8. Food & takeout container (107)
    "Food waste": "Food & takeout container",
    "Disposable food container": "Food & takeout container",
    "Foam food container": "Food & takeout container",
    "Spread tub": "Food & takeout container",
    "Tupperware": "Food & takeout container",
    "Meal carton": "Food & takeout container",
    "Pizza box": "Food & takeout container",
    # 9. Cardboard carton (218)
    "Corrugated carton": "Cardboard carton",
    "Other carton": "Cardboard carton",
    "Drink carton": "Cardboard carton",
    "Egg carton": "Cardboard carton",
    "Toilet tube": "Cardboard carton",
    # 10. Paper & paper bag (175)
    "Normal paper": "Paper & paper bag",
    "Tissues": "Paper & paper bag",
    "Wrapping paper": "Paper & paper bag",
    "Magazine paper": "Paper & paper bag",
    "Paper bag": "Paper & paper bag",
    # 11. Cigarette (667)
    "Cigarette": "Cigarette",
    # 12. Mixed litter (1054)
    "Other plastic container": "Mixed litter",
    "Other plastic": "Mixed litter",
    "Unlabeled litter": "Mixed litter",
    "Broken glass": "Mixed litter",
    "Aluminium foil": "Mixed litter",
    "Scrap metal": "Mixed litter",
    "Rope & strings": "Mixed litter",
    "Shoe": "Mixed litter",
    "Battery": "Mixed litter",
}

# Complete mapping dictionary from all 60 TACO dataset classes + consolidated names to the 3 functional categories
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
    "Plastified paper bag": "Plastic Waste",
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
    "Food Can": "Organic Waste",
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
    "Carded blister pack": "Plastic Waste",
    "Garbage bag": "Plastic Waste",
    "Aluminium blister pack": "Plastic Waste",
    # Food & Putrescible Waste -> Organic Waste
    "Food waste": "Organic Waste",
    "Pizza box": "Organic Waste",
    "Meal carton": "Organic Waste",
    "Drink carton": "Organic Waste",
    "Spread tub": "Organic Waste",
    "Tupperware": "Organic Waste",
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
    # Include consolidated granular TACO labels directly
    **GRANULAR_TO_FUNCTIONAL_MAPPING,
}


def map_taco_class_to_granular(class_name: str) -> str:
    """Map any raw TACO category name into its consolidated 13-class waste label."""
    if not class_name:
        return "Mixed litter"
    cleaned = class_name.strip()
    if cleaned in GRANULAR_CATEGORY_ID_MAP:
        return cleaned
    if cleaned in TACO_TO_GRANULAR_MAPPING:
        return TACO_TO_GRANULAR_MAPPING[cleaned]
    lower_name = cleaned.lower()
    for raw_k, target in TACO_TO_GRANULAR_MAPPING.items():
        if raw_k.lower() == lower_name:
            return target
    return "Mixed litter"


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

    # Exact TACO / granular class lookup
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
        "takeout",
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
        "glass",
        "litter",
    )

    if any(k in lower_name for k in organic_keywords):
        return "Organic Waste"
    if any(k in lower_name for k in biomass_keywords):
        return "Dry Leaves / Biomass"
    if any(k in lower_name for k in plastic_keywords):
        return "Plastic Waste"

    return "Plastic Waste"


def map_taco_class_to_id(class_name: str) -> int:
    """Return integer functional class ID (0, 1, or 2) for a given TACO or raw class name."""
    category = map_taco_class_to_category(class_name)
    return CATEGORY_ID_MAP[category]


# ==============================================================================
# 2. TACO COCO TO YOLOv8 FORMAT CONVERSION & OBJECT-ZOOM AUGMENTATION
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


def _save_exif_corrected_resized(
    src_path: Path,
    dest_path: Path,
    max_side: int = 768,
) -> Tuple[int, int]:
    """
    Open image safely, apply EXIF Orientation (153 TACO images use EXIF orientation 3/6/8),
    convert to RGB, resize down proportionally so max(W, H) <= max_side, and save JPEG.
    Returns the original EXIF-transposed (W, H) before resizing so COCO bboxes align cleanly.
    """
    with Image.open(src_path) as im:
        im_t = ImageOps.exif_transpose(im).convert("RGB")
        orig_w, orig_h = im_t.size
        if max_side > 0 and max(orig_w, orig_h) > max_side:
            scale = max_side / float(max(orig_w, orig_h))
            new_w = max(32, int(round(orig_w * scale)))
            new_h = max(32, int(round(orig_h * scale)))
            im_out = im_t.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            im_out = im_t
        im_out.save(dest_path, format="JPEG", quality=92)
    return orig_w, orig_h


def _generate_train_zoom_crops(
    src_path: Path,
    orig_w: int,
    orig_h: int,
    boxes_cls: List[Tuple[int, float, float, float, float]],
    dest_img_dir: Path,
    dest_lbl_dir: Path,
    stem: str,
    rng: random.Random,
    rare_class_ids: set,
    max_crops: int = 2,
    crop_max_side: int = 640,
) -> int:
    """
    Generate leak-free object-centered zoom crops ONLY from training images.
    TACO objects have median area < 1% of image area; 2x zoom crops centered
    around small/rare waste objects boost YOLOv8 recall dramatically without
    corrupting validation or test evaluations.
    """
    if not boxes_cls or orig_w < 300 or orig_h < 300:
        return 0

    img_area = float(orig_w * orig_h)
    # Candidate anchor boxes: prefer small/medium objects and under-represented classes
    candidates: List[Tuple[float, Tuple[int, float, float, float, float]]] = []
    for item in boxes_cls:
        cid, bx, by, bw, bh = item
        area_r = (bw * bh) / img_area
        if area_r > 0.16 or bw <= 4 or bh <= 4:
            continue
        priority = 2.0 if cid in rare_class_ids else 1.0
        if 0.0005 <= area_r <= 0.04:
            priority += 1.0
        candidates.append((priority + rng.random() * 0.25, item))

    if not candidates:
        return 0

    candidates.sort(key=lambda t: t[0], reverse=True)
    generated = 0
    used_centers: List[Tuple[float, float]] = []

    with Image.open(src_path) as im_raw:
        im_full = ImageOps.exif_transpose(im_raw).convert("RGB")
        full_w, full_h = im_full.size
        aspect = full_w / float(max(1, full_h))

        for _, anchor in candidates:
            if generated >= max_crops:
                break
            _, ax, ay, aw, ah = anchor
            acx = ax + aw * 0.5
            acy = ay + ah * 0.5

            # Avoid generating nearly identical overlapping crops for the same region
            if any(
                abs(acx - ux) < full_w * 0.18 and abs(acy - uy) < full_h * 0.18
                for ux, uy in used_centers
            ):
                continue

            # Crop side around 45-58% of frame so objects double in apparent pixel scale
            crop_scale = rng.uniform(0.45, 0.58)
            crop_h = max(320, min(full_h, int(round(full_h * crop_scale))))
            crop_w = max(320, min(full_w, int(round(crop_h * aspect))))
            if crop_w > full_w:
                crop_w = full_w

            jitter_x = rng.uniform(-0.14, 0.14) * crop_w
            jitter_y = rng.uniform(-0.14, 0.14) * crop_h
            cx1 = int(max(0, min(full_w - crop_w, round(acx - crop_w * 0.5 + jitter_x))))
            cy1 = int(max(0, min(full_h - crop_h, round(acy - crop_h * 0.5 + jitter_y))))
            cx2 = cx1 + crop_w
            cy2 = cy1 + crop_h

            crop_labels: List[str] = []
            for cid, bx, by, bw, bh in boxes_cls:
                bx2 = bx + bw
                by2 = by + bh
                ix1 = max(float(cx1), bx)
                iy1 = max(float(cy1), by)
                ix2 = min(float(cx2), bx2)
                iy2 = min(float(cy2), by2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                orig_box_area = max(1.0, bw * bh)
                if inter_area / orig_box_area < 0.60:
                    continue

                nw = (ix2 - ix1) / float(crop_w)
                nh = (iy2 - iy1) / float(crop_h)
                if nw < 0.015 or nh < 0.015:
                    continue
                nxc = ((ix1 + ix2) * 0.5 - cx1) / float(crop_w)
                nyc = ((iy1 + iy2) * 0.5 - cy1) / float(crop_h)
                nxc = max(0.0, min(1.0, nxc))
                nyc = max(0.0, min(1.0, nyc))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))
                crop_labels.append(f"{cid} {nxc:.6f} {nyc:.6f} {nw:.6f} {nh:.6f}")

            if not crop_labels:
                continue

            cropped_im = im_full.crop((cx1, cy1, cx2, cy2))
            if max(cropped_im.size) > crop_max_side:
                s = crop_max_side / float(max(cropped_im.size))
                cropped_im = cropped_im.resize(
                    (max(32, int(round(crop_w * s))), max(32, int(round(crop_h * s)))),
                    Image.Resampling.LANCZOS,
                )

            crop_name = f"{stem}_z{generated}"
            cropped_im.save(dest_img_dir / f"{crop_name}.jpg", format="JPEG", quality=92)
            (dest_lbl_dir / f"{crop_name}.txt").write_text(
                "\n".join(crop_labels), encoding="utf-8"
            )
            used_centers.append((acx, acy))
            generated += 1

    return generated


def _stratified_taco_split(
    images: List[Dict[str, Any]],
    annotations_by_image: Dict[int, List[Dict[str, Any]]],
    category_id_lookup: Dict[int, int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[int, str]:
    """
    Assign each image deterministically to 'train', 'val', or 'test' while preserving
    proportional class balance across rare consolidated waste categories.
    """
    rng = random.Random(seed)
    total = len(images)
    targets = {
        "train": int(round(total * train_ratio)),
        "val": int(round(total * val_ratio)),
    }
    targets["test"] = max(0, total - targets["train"] - targets["val"])

    total_cls_counts: Counter[int] = Counter()
    img_cls_counts: Dict[int, Counter[int]] = {}
    for im in images:
        iid = int(im["id"])
        c: Counter[int] = Counter()
        for ann in annotations_by_image.get(iid, []):
            cid = category_id_lookup.get(int(ann["category_id"]), 0)
            c[cid] += 1
            total_cls_counts[cid] += 1
        img_cls_counts[iid] = c

    # Sort images by rarity of their rarest class (deterministic tie-break shuffle)
    shuffled = list(images)
    rng.shuffle(shuffled)

    def rarity_key(im: Dict[str, Any]) -> Tuple[int, int]:
        c = img_cls_counts[int(im["id"])]
        if not c:
            return (10**9, int(im["id"]))
        min_freq = min(total_cls_counts[cid] for cid in c)
        return (min_freq, -len(c))

    ordered = sorted(shuffled, key=rarity_key)

    split_img_counts: Counter[str] = Counter()
    split_cls_counts: Dict[str, Counter[int]] = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }
    desired_ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": max(0.05, 1.0 - train_ratio - val_ratio),
    }

    assignment: Dict[int, str] = {}
    for im in ordered:
        iid = int(im["id"])
        available = [s for s in ("train", "val", "test") if split_img_counts[s] < targets[s]]
        if not available:
            available = ["train"]

        img_c = img_cls_counts[iid]
        if len(available) == 1 or not img_c:
            # Fill split proportionally if image has no boxes or only 1 bucket left
            best_split = max(
                available,
                key=lambda s: (targets[s] - split_img_counts[s]) / max(1, targets[s]),
            )
        else:
            # Choose split maximizing deficit on the rarest class inside this image
            rarest_cls = min(img_c.keys(), key=lambda k: total_cls_counts[k])

            def split_score(s: str) -> Tuple[float, float]:
                desired_cls = total_cls_counts[rarest_cls] * desired_ratios[s]
                cls_deficit = desired_cls - split_cls_counts[s][rarest_cls]
                img_deficit = (targets[s] - split_img_counts[s]) / max(1.0, targets[s])
                return (cls_deficit, img_deficit)

            best_split = max(available, key=split_score)

        assignment[iid] = best_split
        split_img_counts[best_split] += 1
        for k, v in img_c.items():
            split_cls_counts[best_split][k] += v

    return assignment


def convert_taco_to_yolo(
    dataset_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    train_ratio: float = 0.75,
    val_ratio: float = 0.15,
    seed: int = 42,
    granular: bool = True,
    max_side: int = 768,
    zoom_crops: bool = True,
    clean_output: bool = True,
) -> Path:
    """
    Convert TACO COCO annotations into YOLO dataset structure with
    EXIF-corrected pre-resized images, stratified train/val/test splits,
    leak-free object-zoom crops in train/, and data.yaml.
    """
    taco_dir = locate_taco_dataset(dataset_path)
    annotations_file = taco_dir / "annotations.json"

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "taco_yolo_dataset"
    output_dir = Path(output_dir).resolve()

    if clean_output and output_dir.exists():
        shutil.rmtree(output_dir)

    with open(annotations_file, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    class_names = GRANULAR_TACO_CLASSES if granular else FUNCTIONAL_CATEGORIES
    category_id_lookup: Dict[int, int] = {}
    for cat in coco_data.get("categories", []):
        cat_id = int(cat["id"])
        cat_name = str(cat.get("name", ""))
        if granular:
            g_name = map_taco_class_to_granular(cat_name)
            category_id_lookup[cat_id] = GRANULAR_CATEGORY_ID_MAP[g_name]
        else:
            category_id_lookup[cat_id] = map_taco_class_to_id(cat_name)

    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    annotations_by_image: Dict[int, List[Dict[str, Any]]] = {}
    class_totals: Counter[int] = Counter()
    for ann in coco_data.get("annotations", []):
        img_id = int(ann["image_id"])
        annotations_by_image.setdefault(img_id, []).append(ann)
        cid = category_id_lookup.get(int(ann["category_id"]), 0)
        class_totals[cid] += 1

    median_count = sorted(class_totals.values())[len(class_totals) // 2] if class_totals else 0
    rare_class_ids = {cid for cid, cnt in class_totals.items() if cnt <= median_count}

    images = list(coco_data.get("images", []))
    split_assignment = _stratified_taco_split(
        images=images,
        annotations_by_image=annotations_by_image,
        category_id_lookup=category_id_lookup,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    rng = random.Random(seed)
    converted_counts = {"train": 0, "val": 0, "test": 0, "train_crops": 0}

    for img_info in images:
        img_id = int(img_info["id"])
        split = split_assignment.get(img_id, "train")
        rel_file = img_info["file_name"]
        src_img_path = taco_dir / rel_file

        if not src_img_path.exists():
            continue

        img_w = float(img_info.get("width", 0))
        img_h = float(img_info.get("height", 0))
        if img_w <= 0 or img_h <= 0:
            continue

        dest_stem = f"{img_id:06d}"
        dest_img_dir = output_dir / "images" / split
        dest_lbl_dir = output_dir / "labels" / split
        dest_img_path = dest_img_dir / f"{dest_stem}.jpg"
        dest_lbl_path = dest_lbl_dir / f"{dest_stem}.txt"

        orig_w, orig_h = _save_exif_corrected_resized(
            src_img_path, dest_img_path, max_side=max_side
        )

        label_lines: List[str] = []
        boxes_cls: List[Tuple[int, float, float, float, float]] = []
        for ann in annotations_by_image.get(img_id, []):
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            target_cls_id = category_id_lookup.get(int(ann["category_id"]), 0)
            xc, yc, nw, nh = coco_bbox_to_yolo(bbox, orig_w, orig_h)
            label_lines.append(f"{target_cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            boxes_cls.append(
                (target_cls_id, float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            )

        dest_lbl_path.write_text("\n".join(label_lines), encoding="utf-8")
        converted_counts[split] += 1

        # Strictly generate zoom crops ONLY on the training split (zero val/test leakage)
        if split == "train" and zoom_crops:
            converted_counts["train_crops"] += _generate_train_zoom_crops(
                src_path=src_img_path,
                orig_w=orig_w,
                orig_h=orig_h,
                boxes_cls=boxes_cls,
                dest_img_dir=dest_img_dir,
                dest_lbl_dir=dest_lbl_dir,
                stem=dest_stem,
                rng=rng,
                rare_class_ids=rare_class_ids,
                max_crops=1,
                crop_max_side=640,
            )

    # Write data.yaml
    data_yaml_path = output_dir / "data.yaml"
    data_yaml_content = {
        "path": output_dir.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": {idx: name for idx, name in enumerate(class_names)},
    }
    with open(data_yaml_path, "w", encoding="utf-8") as yf:
        if yaml is None:
            json.dump(data_yaml_content, yf, indent=2)
        else:
            yaml.safe_dump(data_yaml_content, yf, sort_keys=False)

    logger.info(
        "TACO conversion complete at %s (train=%d + %d zoom crops, val=%d, test=%d, nc=%d)",
        output_dir,
        converted_counts["train"],
        converted_counts["train_crops"],
        converted_counts["val"],
        converted_counts["test"],
        len(class_names),
    )
    return data_yaml_path


# ==============================================================================
# 3. YOLOv8n TRAINING & ONNX EXPORT WORKFLOW
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
    epochs: int = 18,
    image_size: int = 416,
    batch_size: int = 16,
    device: Optional[str] = None,
    base_weights: str = "yolov8n.pt",
    export_onnx: bool = True,
) -> Path:
    """
    Train YOLOv8n on the converted TACO dataset, save best checkpoint to
    backend/models/best.pt (and mirror to weights/ and api/models/ as ONNX).
    """
    import torch
    from ultralytics import YOLO

    compute_device = select_compute_device(device)
    if compute_device == "cpu":
        cpu_threads = os.cpu_count() or 8
        torch.set_num_threads(cpu_threads)

    backend_dir = Path(__file__).resolve().parent
    repo_root = backend_dir.parent
    models_dir = backend_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_target_path = models_dir / "best.pt"

    base_path = Path(base_weights)
    if not base_path.exists() and (backend_dir / base_weights).exists():
        base_weights = str(backend_dir / base_weights)

    project_runs_dir = backend_dir / "runs" / "detect"
    model = YOLO(base_weights)

    model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=compute_device,
        workers=0 if compute_device == "cpu" else 4,
        cache="ram",
        erasing=0.0,
        mosaic=0.65,
        close_mosaic=min(6, max(3, epochs // 4)),
        degrees=10.0,
        flipud=0.2,
        scale=0.5,
        cls=0.8,
        warmup_epochs=1.5,
        cos_lr=True,
        patience=15,
        lr0=0.0018,
        optimizer="AdamW",
        project=str(project_runs_dir),
        name="cleanpath_yolov8n",
        exist_ok=True,
    )

    run_weights_dir = project_runs_dir / "cleanpath_yolov8n" / "weights"
    trained_best = run_weights_dir / "best.pt"
    source_pt = (
        trained_best
        if trained_best.exists()
        else (run_weights_dir / "last.pt")
    )
    if source_pt.exists():
        shutil.copy2(source_pt, best_target_path)
        logger.info("Saved best YOLOv8 weights to %s", best_target_path)

        # Mirror updated .pt weights cleanly across repo entrypoints
        for extra_dir in (
            repo_root / "weights",
            repo_root / "terrasight-ai" / "weights",
        ):
            if extra_dir.exists():
                shutil.copy2(best_target_path, extra_dir / "best.pt")

        if export_onnx:
            export_model = YOLO(str(best_target_path))
            onnx_out = export_model.export(
                format="onnx",
                imgsz=512,
                opset=17,
                simplify=True,
            )
            onnx_path = Path(str(onnx_out))
            if onnx_path.exists():
                target_onnx = models_dir / "best.onnx"
                if onnx_path.resolve() != target_onnx.resolve():
                    shutil.move(str(onnx_path), str(target_onnx))
                api_models_dir = repo_root / "api" / "models"
                api_models_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target_onnx, api_models_dir / "best.onnx")
                logger.info(
                    "Exported best ONNX model to %s and %s",
                    target_onnx,
                    api_models_dir / "best.onnx",
                )

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
        default=18,
        help="Number of training epochs (default: 18)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=416,
        help="Training image resolution (default: 416)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16)",
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
    parser.add_argument(
        "--force-convert",
        action="store_true",
        help="Rebuild YOLO dataset even if data.yaml already exists",
    )
    parser.add_argument(
        "--no-crops",
        action="store_true",
        help="Disable object-centered zoom crops during train split generation",
    )
    parser.add_argument(
        "--legacy-3class",
        action="store_true",
        help="Convert using 3 functional categories instead of 13 granular TACO classes",
    )

    args = parser.parse_args()
    ds_path = Path(args.dataset_path) if args.dataset_path else None
    out_path = Path(args.output_dir) if args.output_dir else (Path(__file__).resolve().parent / "taco_yolo_dataset")
    existing_yaml = Path(out_path).resolve() / "data.yaml"

    if existing_yaml.exists() and not args.force_convert and not args.convert_only:
        yaml_path = existing_yaml
        print(f"Using existing converted YOLO dataset at: {yaml_path}")
    else:
        yaml_path = convert_taco_to_yolo(
            dataset_path=ds_path,
            output_dir=out_path,
            granular=not args.legacy_3class,
            zoom_crops=not args.no_crops,
        )
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
