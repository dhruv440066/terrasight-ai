# CleanPath AI — YOLOv8 Waste Detection Models Guide

CleanPath AI uses a unified **Dual-YOLOv8 + Spectral ROI Waste Detection Engine** (`backend/vision_engine.py` & `api/vision_engine.py`) trained on the **TACO (Trash Annotations in Context)** dataset across **13 visually coherent granular trash classes** mapped 1-to-1 into three municipal sanitation functional categories:

| Functional Category | DB Enum | Weight ($w_c$) | Mapped TACO Item Subclasses (`subclass`) |
| :--- | :--- | :--- | :--- |
| **Plastic Waste** | `PLASTIC_WASTE` | `1.0` | `Plastic bottle`, `Glass bottle & jar`, `Drink & metal can`, `Plastic bag & film`, `Plastic wrapper`, `Disposable cup`, `Styrofoam piece`, `Bottle cap, straw & lid`, `Mixed litter` |
| **Organic Waste** | `ORGANIC_WASTE` | `1.2` | `Food & takeout container`, `Food & organic waste` |
| **Dry Leaves / Biomass** | `DRY_LEAVES_BIOMASS` | `0.7` | `Cardboard carton`, `Paper & paper bag`, `Dry leaves & foliage`, `Cigarette` |

---

## 1. Unified ONNX Runtime & PyTorch Weight Placement

Both Vercel Serverless (`api/models/`) and local FastAPI (`backend/models/`) run real YOLOv8 inference via lightweight **ONNX Runtime (`onnxruntime`)**:

```text
api/models/best.onnx        # Exported 13-class TACO YOLOv8n ONNX graph
api/models/yolov8n.onnx     # Official COCO YOLOv8n ONNX graph (context + portrait gate)
backend/models/best.onnx    # Mirrored ONNX graph for local parity
backend/models/yolov8n.onnx # Mirrored COCO ONNX graph
backend/models/best.pt      # Native PyTorch checkpoint (mirrored to weights/best.pt)
```

When the vision engine initializes:
1. It loads `best.onnx` + `yolov8n.onnx` via `onnxruntime` CPUExecutionProvider (unless a newer `.pt` checkpoint or PyTorch environment overrides it).
2. `yolov8n.onnx` runs first to identify common waste items (`bottle`, `cup`, food scraps) AND suppress non-waste scenes (portraits/selfies, vehicles, indoor furniture/laptops).
3. `best.onnx` runs across full-frame, foreground crop, and overlapping 2x2 quadrant tiles to produce tight per-item bounding boxes with granular item names (`Plastic bottle`, `Plastic bag & film`, `Styrofoam piece`, `Plastic wrapper`, `Bottle cap, straw & lid`, `Cardboard carton`, etc.).
4. Spectral HSV verification (`_refine_detection_from_roi`) and pile-bounded cluster verification (`_segment_complementary_waste_numpy`) boost recall on yellow/amber organic food scraps and leaf/paper debris inside confirmed multi-object piles.

---

## 2. Environment Configuration (`.env`)

You can customize the model path and confidence threshold via environment variables in `backend/.env`:

```env
MODEL_PATH=models/best.onnx
CONFIDENCE_THRESHOLD=0.18
```

---

## 3. Training / Retraining on TACO Dataset

To re-convert `data/` (`annotations.json`) with EXIF orientation correction, stratified class-balanced train/val/test splits, and leak-free training zoom crops, then train and export `best.pt` + `best.onnx`:

```bash
python backend/train_taco.py --epochs 18 --image-size 416 --batch-size 16
```

What [`backend/train_taco.py`](../train_taco.py) does:
1. Converts all 1,500 TACO images into `backend/taco_yolo_dataset/` (`nc=13`, EXIF-transposed `max_side=768`, plus object-centered zoom crops strictly in `train/`).
2. Trains YOLOv8n (`AdamW`, `cos_lr=True`, `mosaic=0.65`, `erasing=0.0`) and copies `best.pt` to `backend/models/best.pt`, `weights/best.pt`, and `terrasight-ai/weights/best.pt`.
3. Exports simplified ONNX weights (`opset=17`) to `backend/models/best.onnx` and `api/models/best.onnx`.

---

## 4. Honest Model Evaluation

Run [`backend/evaluate_model.py`](../evaluate_model.py) to evaluate both raw 13-class Ultralytics validation metrics (`mAP@50`, `mAP@50-95`) and end-to-end 3-class VisionEngine metrics on the held-out `val` split:

```bash
cd backend
python evaluate_model.py
```

Evaluation artifacts are saved automatically to:
- `backend/evaluation/evaluation_report.json`
- `backend/evaluation/evaluation_summary.txt`
