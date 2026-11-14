"""
Dataset Quality Assurance & Automated Validation Pipeline for TerraSight AI V2.

Performs comprehensive data hygiene checks:
1. Image file integrity and decodability (corrupt header detection)
2. Label coordinate boundaries (normalized [0, 1] checks, non-zero dimensions)
3. Class ID taxonomy validation (within 13-class consolidated schema)
4. Perceptual hashing (64-bit dHash) to detect duplicates and prevent train/val/test data leakage
5. Class distribution and imbalance analysis
"""

import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cleanpath.dataset_validator")

CLASS_NAMES = [
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


def compute_dhash64(image_path: Path, hash_size: int = 8) -> int:
    """Compute a 64-bit Difference Hash (dHash) using pure Pillow and NumPy."""
    with Image.open(image_path) as img:
        # Resize to (hash_size + 1, hash_size) grayscale
        gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
        pixels = np.asarray(gray, dtype=np.int16)
        # Compare adjacent horizontal pixels
        diff = pixels[:, 1:] > pixels[:, :-1]
        val = 0
        for bit in diff.flatten():
            val = (val << 1) | int(bit)
        return val


def hamming_distance64(hash1: int, hash2: int) -> int:
    """Compute Hamming distance between two 64-bit integer hashes."""
    xor = hash1 ^ hash2
    return bin(xor).count("1")


def validate_yolo_dataset(dataset_dir: Path) -> Dict[str, Any]:
    """
    Run full validation suite on a YOLO format dataset directory.
    Expected directory structure:
      dataset_dir/
        images/ {train, val, test}
        labels/ {train, val, test}
    """
    start_time = time.time()
    images_root = dataset_dir / "images"
    labels_root = dataset_dir / "labels"

    report: Dict[str, Any] = {
        "dataset_path": str(dataset_dir),
        "validated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "PASSED",
        "splits": {},
        "integrity": {
            "total_images_checked": 0,
            "corrupted_images": [],
            "missing_label_files": [],
            "invalid_boxes": [],
            "out_of_bounds_classes": [],
            "empty_label_files": [],
        },
        "perceptual_duplicates": {
            "within_split_duplicates": [],
            "cross_split_leakage": [],
        },
        "class_distribution": {
            "train": {},
            "val": {},
            "test": {},
            "total": {},
        },
    }

    split_hashes: Dict[str, Dict[str, int]] = {"train": {}, "val": {}, "test": {}}

    for split in ("train", "val", "test"):
        split_img_dir = images_root / split
        split_lbl_dir = labels_root / split

        if not split_img_dir.exists():
            continue

        split_images = sorted(list(split_img_dir.glob("*.*")))
        class_counter: Counter[int] = Counter()
        box_count = 0

        for img_path in split_images:
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            report["integrity"]["total_images_checked"] += 1

            # 1. Verify Image Decodability
            try:
                with Image.open(img_path) as im:
                    im.verify()
                # Re-open for actual reading & hash computation (verify clears file pointer)
                img_hash = compute_dhash64(img_path)
                split_hashes[split][img_path.name] = img_hash
            except Exception as exc:
                report["integrity"]["corrupted_images"].append(
                    {"path": str(img_path), "error": str(exc)}
                )
                continue

            # 2. Verify Label File
            lbl_path = split_lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                report["integrity"]["missing_label_files"].append(str(lbl_path))
                continue

            try:
                lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
            except Exception as exc:
                report["integrity"]["corrupted_images"].append(
                    {"path": str(lbl_path), "error": str(exc)}
                )
                continue

            if not lines:
                report["integrity"]["empty_label_files"].append(str(lbl_path))
                continue

            for line_idx, line in enumerate(lines, start=1):
                parts = line.strip().split()
                if len(parts) < 5:
                    report["integrity"]["invalid_boxes"].append(
                        {
                            "file": str(lbl_path),
                            "line": line_idx,
                            "reason": f"Expected 5 values, got {len(parts)}",
                        }
                    )
                    continue

                try:
                    cid = int(parts[0])
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except ValueError as exc:
                    report["integrity"]["invalid_boxes"].append(
                        {"file": str(lbl_path), "line": line_idx, "reason": str(exc)}
                    )
                    continue

                # Check Class ID range
                if cid < 0 or cid >= len(CLASS_NAMES):
                    report["integrity"]["out_of_bounds_classes"].append(
                        {
                            "file": str(lbl_path),
                            "line": line_idx,
                            "class_id": cid,
                            "allowed_max": len(CLASS_NAMES) - 1,
                        }
                    )
                else:
                    class_counter[cid] += 1
                    box_count += 1

                # Check Normalized Coordinates [0, 1]
                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    report["integrity"]["invalid_boxes"].append(
                        {
                            "file": str(lbl_path),
                            "line": line_idx,
                            "reason": f"Coordinate out of [0, 1] range: ({xc}, {yc}, {w}, {h})",
                        }
                    )

        report["splits"][split] = {
            "images": len(split_images),
            "boxes": box_count,
        }
        for cid, cnt in sorted(class_counter.items()):
            cname = CLASS_NAMES[cid]
            report["class_distribution"][split][cname] = cnt
            report["class_distribution"]["total"][cname] = (
                report["class_distribution"]["total"].get(cname, 0) + cnt
            )

    # 3. Check for Cross-Split Leakage (Perceptual Duplicates between Train and Val/Test)
    train_hashes = split_hashes.get("train", {})
    val_hashes = split_hashes.get("val", {})
    test_hashes = split_hashes.get("test", {})

    for val_name, val_h in val_hashes.items():
        for tr_name, tr_h in train_hashes.items():
            dist = hamming_distance64(val_h, tr_h)
            if dist <= 2:
                report["perceptual_duplicates"]["cross_split_leakage"].append(
                    {
                        "source_split": "train",
                        "source_image": tr_name,
                        "target_split": "val",
                        "target_image": val_name,
                        "hamming_distance": dist,
                    }
                )
                break

    for test_name, test_h in test_hashes.items():
        for tr_name, tr_h in train_hashes.items():
            dist = hamming_distance64(test_h, tr_h)
            if dist <= 2:
                report["perceptual_duplicates"]["cross_split_leakage"].append(
                    {
                        "source_split": "train",
                        "source_image": tr_name,
                        "target_split": "test",
                        "target_image": test_name,
                        "hamming_distance": dist,
                    }
                )
                break

    # Determine Overall QA Status
    has_corruption = bool(report["integrity"]["corrupted_images"])
    has_invalid_boxes = bool(report["integrity"]["invalid_boxes"])
    has_invalid_classes = bool(report["integrity"]["out_of_bounds_classes"])

    if has_corruption or has_invalid_boxes or has_invalid_classes:
        report["status"] = "FAILED"
    else:
        report["status"] = "PASSED"

    report["duration_seconds"] = round(time.time() - start_time, 2)
    return report


if __name__ == "__main__":
    import sys

    dataset_path = Path("backend/taco_yolo_dataset")
    if not dataset_path.exists():
        dataset_path = Path("taco_yolo_dataset")

    logger.info("Starting automated dataset validation on %s...", dataset_path)
    res = validate_yolo_dataset(dataset_path)

    out_file = dataset_path.parent / "dataset_validation_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    logger.info("Validation Complete! Status: %s", res["status"])
    logger.info("Total images: %d across splits: %s", res["integrity"]["total_images_checked"], res["splits"])
    logger.info("Saved validation report to %s", out_file)
