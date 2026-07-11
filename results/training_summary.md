# PartSight — YOLOv11 Training Results

## Training Configuration
- Model: YOLOv11n (nano)
- Epochs: 50
- Image size: 200x200 (auto-adjusted to 224)
- Batch size: 16
- Training time: 11.5 minutes (T4 GPU)

## Final Metrics
| Metric | Score |
|---|---|
| mAP@50 | 0.756 |
| mAP@50-95 | 0.407 |
| Precision | 0.708 |
| Recall | 0.663 |

## Per-Class mAP@50
| Class | mAP@50 |
|---|---|
| coating_damage | 0.926 |
| surface_scratch | 0.884 |
| material_inclusion | 0.854 |
| corrosion_pit | 0.824 |
| surface_irregularity | 0.584 |
| surface_crack_network | 0.466 |

## Analysis
Weakest class (crazing/crack_network) has ground-truth boxes
spanning nearly the entire image, making precise bounding-box
localization harder than compact defects like scratches or
coating damage.
