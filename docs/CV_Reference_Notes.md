# PartSight — Computer Vision Reference Notes
### A living document — updated every session as new topics are covered
### Author: Aditya Manoj Madgulkar | Project: PartSight (KLA Interview Prep)

---

## How To Use This Document
This is your single source of truth for CV theory learned while building PartSight.
Every new concept taught gets added here — nothing lives only in chat.
Review this before interviews. Each topic has: Definition → Why it matters → Math → PartSight application → Interview answer.

---

## Progress Tracker

| # | Topic | Status |
|---|---|---|
| 1 | Image Fundamentals (pixels, shape, channels) | ✅ Covered |
| 2 | Color Spaces (BGR/RGB/Grayscale) | ✅ Covered |
| 3 | Bounding Boxes | ✅ Covered |
| 4 | Pascal VOC XML Format | ✅ Covered |
| 5 | YOLO TXT Format & Normalization | ✅ Covered |
| 6 | Convolution Operation | ✅ Covered |
| 7 | Stride | ✅ Covered |
| 8 | Padding | ✅ Covered |
| 9 | Pooling (Max/Average) | ✅ Covered |
| 10 | Output Size Formula | ✅ Covered |
| 11 | Backpropagation | ✅ Covered |
| 12 | Loss Function | ✅ Covered |
| 13 | Pretrained Weights & Transfer Learning | ✅ Covered |
| 14 | Gradient Descent & Learning Rate | ✅ Covered |
| 15 | YOLO Architecture (Backbone/Neck/Head) | ⚠️ Covered — needs re-review |
| 16 | Data Augmentation (Albumentations) | ✅ Covered |
| 17 | IoU (Intersection over Union) | ✅ Covered |
| 18 | Precision & Recall | ✅ Covered |
| 19 | AP & mAP | ✅ Covered |
| 20 | mAP@50 vs mAP@50-95 | ✅ Covered |
| 21 | NMS (Non-Max Suppression) | ⏳ To revisit |
| 22 | ONNX Export | ⏳ Not yet taught |
| 23 | FastAPI Deployment | ⏳ Not yet taught |

---

## MODULE 1: Image Fundamentals

### 1.1 What Is An Image To A Computer?
An image is a 3D matrix of numbers — **Height × Width × Channels**.

**Each number = pixel intensity, range 0–255.**

**PartSight application:** NEU-DET images are grayscale (200×200×1) — metal surface defects don't need color information, so grayscale is simpler and faster with no accuracy loss.

### 1.2 Color Spaces — BGR vs RGB vs Grayscale
OpenCV reads images as **BGR** by default. Matplotlib displays as **RGB**. Mismatch = wrong colors on screen (not wrong data).

```python
img_bgr  = cv2.imread(path)                       # BGR by default
img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # convert for display
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
```

**Interview answer:** "OpenCV loads images in BGR channel order, so I always convert to RGB with cvtColor before visualizing with matplotlib, otherwise colors appear swapped."

---

## MODULE 2: Bounding Boxes & Data Formats

### 2.1 What Is A Bounding Box?
A rectangle drawn around a defect, telling the model **where** it is in the image.

### 2.2 Pascal VOC XML Format (raw NEU-DET format)
Stores **absolute pixel coordinates**:
```xml
<bndbox>
  <xmin>26</xmin>   <!-- left edge -->
  <ymin>12</ymin>   <!-- top edge -->
  <xmax>43</xmax>   <!-- right edge -->
  <ymax>171</ymax>  <!-- bottom edge -->
</bndbox>
```

### 2.3 YOLO TXT Format (what YOLOv11 needs)
Stores **normalized coordinates (0 to 1)**:

### 2.4 Conversion Formula (XML → YOLO)
**Why normalize?** A pixel coordinate like x=50 means something different on a 200×200 image vs a 1920×1080 image. A normalized value like 0.25 always means "25% from the left," regardless of resolution — making the model resolution-independent.

### 2.5 Reverse Formula (YOLO → pixels, for visualization)
**PartSight application:** Built `xml_to_yolo()` converter processing 1799 NEU-DET samples — verified visually by drawing boxes back on images using the reverse formula.

**Interview answer:** "I parsed Pascal VOC XML using Python's ElementTree, extracted pixel bounding boxes, and normalized them to YOLO format by dividing by image dimensions — making annotations resolution-independent for production deployment."

---

## MODULE 3: CNN Fundamentals

### 3.1 Convolution — The Core Operation
A **kernel** (small grid of numbers, e.g. 3×3) slides across the image. At each position: multiply overlapping numbers element-wise, then sum → one output number. Repeating this across the whole image produces a **feature map**.

Different kernels detect different things (edges, textures). **The model learns the best kernel values automatically via backpropagation** — they are not hand-designed.

**PartSight application:** Early YOLOv11 backbone layers detect edges of parts; deeper layers detect defect-specific shapes (scratches, cracks).

### 3.2 Output Size Formula
Always round DOWN when the result has a decimal (a kernel can't sit "half on" a pixel grid).

### 3.3 Stride
How many pixels the kernel moves per step.
**PartSight application:** YOLOv11 uses stride 2 in deeper layers to reduce computation — critical for real-time production-line inspection.

### 3.4 Padding
Adding a border of zeros around the image so convolution output doesn't shrink and edge information isn't lost.
**PartSight application:** Same padding preserves edge-of-part defects that would otherwise be lost near the image boundary.

### 3.5 Pooling
Reduces feature map size while keeping the most important signal.
**Interview answer:** "Stride controls how far the kernel moves per step, padding preserves spatial dimensions and edge information, and max pooling reduces feature map size while retaining the strongest activations — which for defect detection correspond to anomaly signals like scratches or cracks."

---

## MODULE 4: Training Mechanics

### 4.1 Loss Function
Measures how wrong a prediction is. Perfect prediction → loss = 0. Terrible prediction → high loss.
YOLOv11 uses a **combined loss**:

### 4.2 Backpropagation
How the network learns from its mistakes.
Analogy: like a chef adjusting a recipe after tasting — "too salty" (loss) traces back to "too much salt in step 3" (backprop), then reduce salt next time (gradient descent). Repeated over many attempts (epochs) → recipe converges to perfect.

### 4.3 Gradient Descent & Learning Rate
Minimizing loss = walking downhill on a "loss landscape" toward the lowest point (minimum loss).

### 4.4 Pretrained Weights & Transfer Learning
Pretrained weights = kernel values already learned from a huge dataset (YOLOv11 trained on COCO). They already understand edges, shapes, textures.

**Transfer learning** = start from these weights, freeze/reuse most layers, and only fine-tune the detection head on your own small dataset.

**PartSight application:** Training log confirmed — "Transferred 448/499 items from pretrained weights" — proving transfer learning was actually applied, not training from scratch.

**Interview answer:** "I used transfer learning by loading YOLOv11's COCO-pretrained weights, which already encode general visual features like edges and shapes. I fine-tuned only the detection head on our 1799-sample defect dataset for 50 epochs — this is why strong results were achievable with a relatively small dataset."

---

## MODULE 5: YOLO Architecture

### 5.1 Why "You Only Look Once"?
Older detectors (R-CNN family) proposed thousands of regions and classified each separately — slow. YOLO looks at the whole image **once** and predicts all boxes, classes, and confidences simultaneously in a single forward pass — fast enough for real-time inspection.

### 5.2 Grid-Based Prediction
The image is divided into a grid; each cell predicts:

### 5.3 Three Architecture Components

### 5.4 NMS — Non-Maximum Suppression (to revisit in depth)
After prediction, multiple overlapping boxes may cover the same defect. NMS keeps the highest-confidence box and removes others that overlap it above an IoU threshold, leaving one clean box per defect.

**Interview answer:** "YOLO divides the image into a grid and predicts bounding boxes, class probabilities, and objectness scores for every cell in a single forward pass, making it far faster than two-stage detectors. YOLOv11 has a convolutional backbone for feature extraction, an FPN neck for multi-scale detection, and a detection head for final predictions, followed by NMS to remove duplicate boxes."

---

## MODULE 6: Data Augmentation

### 6.1 Why Augment?
1440 training images is borderline for deep learning. Augmentation creates realistic variations without collecting new data — same defect identity, different appearance — preventing the model from memorizing (overfitting) and forcing it to learn real patterns.

### 6.2 Transforms Used In PartSight
| Transform | Purpose |
|---|---|
| ToGray | Metal inspection images don't need color |
| HorizontalFlip | Simulates viewing part from opposite side |
| Rotate ±10° | Simulates camera angle variation |
| RandomBrightnessContrast ±15% | Simulates different factory lighting |
| GaussianBlur | Simulates camera focus variation |
| GaussNoise | Simulates sensor noise |

### 6.3 Why Albumentations Over Manual OpenCV
Albumentations **automatically recalculates bounding box coordinates** when the image is transformed. Manual OpenCV augmentation requires recalculating coordinates yourself — complex and error-prone.

### 6.4 Horizontal Flip Math

**Output** "I built an augmentation pipeline using Albumentations with horizontal flips, ±10° rotation, ±15% brightness/contrast variation, blur, and noise injection. Albumentations automatically transforms bounding box coordinates alongside the image, which is critical for detection tasks — manual coordinate recalculation would be error-prone."

---

## MODULE 7: Evaluation Metrics

### 7.1 IoU — Intersection over Union
Measures how well a predicted box overlaps the ground-truth box.

### 7.2 TP / FP / FN

### 7.3 Precision & Recall

They trade off against each other via the **confidence threshold** (how sure the model must be before flagging a defect).

### 7.4 AP (Average Precision) — Per Class
Plot Precision (y) vs Recall (x) across all confidence thresholds → Precision-Recall curve. **AP = area under this curve.**

### 7.5 mAP (mean Average Precision) — All Classes
Simple average of AP across every class.

### 7.6 mAP@50 vs mAP@50-95

mAP@50-95 is always lower — it demands much tighter, pixel-precise boxes.

**Interview answer:** "mAP is computed by first calculating IoU between predicted and ground-truth boxes to classify detections as true or false positives, then building a precision-recall curve per class and taking the area under it as AP. mAP@50 averages AP across all classes at a 0.5 IoU threshold, while mAP@50-95 is stricter, averaging across ten thresholds from 0.5 to 0.95."

---

## MODULE 8: PartSight Actual Results (Day 4 Training)

### 8.1 Training Configuration

### 8.2 Final Metrics
| Metric | Score |
|---|---|
| mAP@50 | **0.756** |
| mAP@50-95 | 0.407 |
| Precision | 0.708 |
| Recall | 0.663 |

### 8.3 Per-Class mAP@50 Breakdown
| Class | mAP@50 | Notes |
|---|---|---|
| coating_damage | 0.926 | Strongest — compact, well-defined boundaries |
| surface_scratch | 0.884 | Strong — thin, elongated but distinct shape |
| material_inclusion | 0.854 | Strong — clear vertical streak pattern |
| corrosion_pit | 0.824 | Good — clustered dot patterns |
| surface_irregularity | 0.584 | Weaker — irregular, less consistent shape |
| surface_crack_network | 0.466 | Weakest — box spans ~95% of image (crazing spreads everywhere), making precise localization inherently harder |

**Key insight to remember for interview:** The weakest class isn't a bug — it's an inherent property of the defect type. Crazing (crack networks) don't have a compact, localizable shape like a scratch does, so bounding-box regression is fundamentally harder for it. This is a nuanced, self-aware observation that shows real understanding, not just running code.

---

## MODULE 9: To Be Covered Next

- [ ] NMS in depth (currently only summary-level understanding)
- [ ] ONNX export — model optimization for inference
- [ ] FastAPI deployment — serving the model as an API
- [ ] LLM reporting layer — connecting detection output to natural language reports
- [ ] Confusion matrix interpretation
- [ ] Precision-Recall curve reading (actual chart from training)

---

## Code Review Sessions (Tracking)

| Session | Cells Covered | Status |
|---|---|---|
| Session 2 | Cells 1-4 (setup, drive, exploration, XML→YOLO) | ✅ Done (Day 5) |
| Session 3 | Cells 5-7 (pipeline, data.yaml, validation) | ⏳ Pending |
| Session 4 | Cells 8-9 (visualization, augmentation) | ⏳ Pending |
| Session 5 | Cell 10 (training) + connect to mAP results | ⏳ Pending |

*Each session: Mady explains the code in his own words first, then gets corrected — locking in real understanding rather than passive reading.*

---

## Session Log — Day 5: Code Review Session 1 (Cells 1-4)

### Cell 1 — Environment Setup
**Why check `torch.cuda.is_available()`?**
GPU training is 10–50x faster than CPU. Our 50-epoch training run
took 11.5 minutes on a T4 GPU — on CPU this could take several hours.
Always verify GPU is active before starting a training run.

### Cell 2 — Mount Drive + Path Validation
**Why check `os.path.exists()` before proceeding?**
If skipped, later cells (like dataset exploration) would crash with a
confusing "file not found" error deep in the pipeline. Checking early
gives a clear, immediate error message instead of a cryptic crash later —
this is a general good-engineering habit, not just a CV-specific one.

### Cell 3 — Dataset Exploration
Purpose: understand the data BEFORE writing any processing pipeline.
Confirms image counts per class, folder structure, and file formats —
catches surprises (missing files, format mismatches) before they cause
wasted training time.

### Cell 4 — XML to YOLO Converter (`xml_to_yolo()`)
**Summary:** Takes one Pascal VOC XML file as input, reads the pixel
bounding-box coordinates (xmin, ymin, xmax, ymax), and converts them to
YOLO normalized format by dividing by the image's width and height.
This is the single most important function in the data pipeline — it's
the bridge between the raw dataset format and what YOLOv11 can actually
train on.

**Status:** Core understanding solid. Full line-by-line walkthrough
(the `ET.parse()` mechanics, `CLASS_MAPPING` dict, loop over `<object>`
tags) to be revisited for deeper recall before interview.
