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
