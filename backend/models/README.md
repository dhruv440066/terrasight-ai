# CleanPath AI — YOLOv8 Waste Detection Models Guide

CleanPath AI uses a multi-stage **YOLOv8 + Spectral ROI Waste Detection Engine** (`backend/vision_engine.py`) trained on the **TACO (Trash Annotations in Context)** dataset mapped to three municipal sanitation functional categories:

| Class ID | Functional Category | DB Enum | Weight ($w_c$) | Target Waste Types |
| :--- | :--- | :--- | :--- | :--- |
| `0` | **Plastic Waste** | `PLASTIC_WASTE` | `1.0` | Clear/colored PET bottles, plastic bags, wrappers, cups, straws, lids, sachets, synthetic containers |
| `1` | **Organic Waste** | `ORGANIC_WASTE` | `0.8` | Food scraps, banana/fruit peels, vegetable waste, rotting organic matter, food-contaminated containers |
| `2` | **Dry Leaves / Biomass** | `DRY_LEAVES_BIOMASS` | `0.5` | Fallen leaves, twigs, garden clippings, yard trimmings, paper/cardboard biomass |

---

## 1. Model Weights Placement

Place your custom trained YOLOv8 weights at:

```text
backend/models/best.pt
```

When the backend starts, `VisionEngine` automatically loads weights in the following priority order:
1. Custom environment variable `MODEL_PATH` (or `CLEANPATH_YOLO_WEIGHTS`) if specified in `.env`
2. `backend/models/best.pt` (Custom 3-class TACO YOLOv8 checkpoint)
3. `backend/yolov8n.pt` (Official COCO YOLOv8n auxiliary/fallback checkpoint)

Both `backend/models/best.pt` and `backend/yolov8n.pt` are loaded simultaneously when available so that:
- `best.pt` detects street-litter patterns learned from TACO across full-frame (`1024px`) and foreground-cropped tiles.
- `yolov8n.pt` provides complementary objectness and person/portrait suppression.
- Spectral ROI refinement (`_refine_category_from_roi`) and complementary waste region segmentation (`_segment_complementary_waste_regions`) ensure balanced recall across **Plastic Waste**, **Organic Waste**, and **Dry Leaves / Biomass** even when classes are imbalanced in raw street datasets.

---

## 2. Environment Configuration (`.env`)

You can customize the model path and confidence threshold via environment variables in `backend/.env`:

```env
MODEL_PATH=models/best.pt
CONFIDENCE_THRESHOLD=0.25
```

---

## 3. Training / Retraining on TACO Dataset

To train or fine-tune `models/best.pt` on the local TACO dataset (`data/TACO-master`):

```bash
cd backend
python train_taco.py --epochs 30 --imgsz 640 --batch 16
```

What `train_taco.py` does:
1. Converts `data/TACO-master/data/annotations.json` into YOLO format under `backend/datasets/taco_yolo/`.
2. Maps all 60 TACO categories into the 3 functional municipal categories (`Plastic Waste`, `Organic Waste`, `Dry Leaves / Biomass`), including mapping food-contaminated containers to `Organic Waste` for better class balance.
3. Trains YOLOv8 and copies the best checkpoint directly to `backend/models/best.pt`.

---

## 4. Evaluating Model Performance

Run the evaluation suite to compute **Precision**, **Recall**, **mAP@50**, **mAP@50-95**, **F1 Score**, **Class-wise Detection Metrics**, and the **Confusion Matrix**:

```bash
cd backend
python evaluate_model.py
```

Evaluation artifacts are saved automatically to:
- `backend/evaluation/evaluation_report.json`
- `backend/evaluation/evaluation_summary.txt`
