# PartSight
Aircraft Engine Part Inspection System using Computer Vision 


# PartSight — Aircraft Engine Parts Inspection System

An end-to-end Computer Vision pipeline for automated
detection and classification of surface defects on
aircraft engine components built using YOLOv11.

## Problem Statement
Manual visual inspection of aircraft engine parts is
slow, inconsistent, and expensive. PartSight automates
this using Computer Vision and Deep Learning.

## Pipeline Architecture

Image Input → OpenCV Preprocessing → YOLOv11 Detection

→ ONNX Inference → FastAPI Endpoint → LLM Repor

## Tech Stack

- YOLOv11 (Ultralytics)
- OpenCV
- Albumentations  
- ONNX Runtime
- FastAPI
- PyTorch

## Dataset
NEU-DET Surface Defect Dataset used as surrogate
for proprietary aerospace parts imagery.

## Defect Classes
| Class | Description |

| surface_scratch | Linear surface damage |
| structural_crack | Fracture in material |
| corrosion_pit | Localized corrosion damage |
| coating_damage | Surface coating failure |
| material_inclusion | Foreign material embedded |
| surface_irregularity | Non-uniform surface finish |

## Project Progress
- [x] Day 1 - Project setup + data pipeline
- [ ] Day 2 - OpenCV preprocessing
- [ ] Day 3 - YOLOv11 training
- [ ] Day 4 - Model evaluation
- [ ] Day 5 - ONNX export
- [ ] Day 6 - FastAPI endpoint
- [ ] Day 7 - LLM report layer
- [ ] Day 8 - Full pipeline
- [ ] Day 9 - Documentation
- [ ] Day 10 - Final cleanup

## Results
*To be updated after Day 4 training*
