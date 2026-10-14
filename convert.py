import json
import os
import shutil
from pathlib import Path
import numpy as np

# Paths
TACO_DATA_DIR = Path("data")
ANNOTATIONS_FILE = TACO_DATA_DIR / "annotations.json"
OUTPUT_DIR = Path("../cleanpath-dataset") # Or your chosen project data path

# 3 Target Functional Categories for CleanPath AI
TARGET_CLASSES = {
    "Plastic Waste": 0,
    "Organic Waste": 1,
    "Dry Leaves / Biomass": 2
}

def map_taco_category(cat_name):
    cat = cat_name.lower()
    if any(w in cat for w in ["plastic", "bottle", "bag", "cup", "can", "wrapper", "lid", "container"]):
        return TARGET_CLASSES["Plastic Waste"]
    elif any(w in cat for w in ["food", "organic", "bio", "fruit", "peel"]):
        return TARGET_CLASSES["Organic Waste"]
    else:
        return TARGET_CLASSES["Dry Leaves / Biomass"]

with open(ANNOTATIONS_FILE, 'r') as f:
    coco = json.load(f)

# Build category lookup
cat_id_to_target = {}
for c in coco['categories']:
    cat_id_to_target[c['id']] = map_taco_category(c['name'])

# Create train/val structure
for split in ['train', 'val']:
    (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

# Organize annotations by image_id
image_annotations = {}
for ann in coco['annotations']:
    img_id = ann['image_id']
    if img_id not in image_annotations:
        image_annotations[img_id] = []
    image_annotations[img_id].append(ann)

images = coco['images']
np.random.seed(42)
np.random.shuffle(images)
split_idx = int(len(images) * 0.8) # 80% train, 20% val

for idx, img_info in enumerate(images):
    split = "train" if idx < split_idx else "val"
    img_id = img_info['id']
    rel_path = img_info['file_name'] # e.g. "batch_1/000001.jpg"
    src_img = TACO_DATA_DIR / rel_path
    
    if not src_img.exists():
        continue

    dest_img_name = f"{img_info['id']:06d}.jpg"
    dest_img = OUTPUT_DIR / "images" / split / dest_img_name
    shutil.copy(src_img, dest_img)

    # Convert bounding boxes
    img_w = float(img_info['width'])
    img_h = float(img_info['height'])
    label_path = OUTPUT_DIR / "labels" / split / f"{img_info['id']:06d}.txt"

    lines = []
    for ann in image_annotations.get(img_id, []):
        cat_id = ann['category_id']
        target_cls = cat_id_to_target.get(cat_id, 0)
        # COCO bbox: [x_min, y_min, width, height]
        bx, by, bw, bh = ann['bbox']
        
        # YOLO normalized: [x_center, y_center, width, height]
        x_center = (bx + bw / 2.0) / img_w
        y_center = (by + bh / 2.0) / img_h
        norm_w = bw / img_w
        norm_h = bh / img_h

        lines.append(f"{target_cls} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    with open(label_path, 'w') as out_f:
        out_f.write("\n".join(lines))

print(f"Conversion complete! Dataset saved at: {OUTPUT_DIR.resolve()}")