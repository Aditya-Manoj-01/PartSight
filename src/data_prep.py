# PartSight - Data Preparation Pipeline
# Day 1 - Dataset Exploration Complete
#
# WHAT WE LEARNED TODAY:
# - Dataset: NEU-DET, 1800 images, 6 defect classes
# - 240 train + 60 val images per class (balanced)
# - Labels in Pascal VOC XML format
# - Need to convert XML → YOLO TXT format
#
# BOUNDING BOX CONCEPT:
# XML format  → xmin, ymin, xmax, ymax (pixel coords)
# YOLO format → x_center, y_center, width, height
#               (normalized 0-1 relative to image size)
#
# CONVERSION FORMULA:
# x_center = (xmin + xmax) / 2 / image_width
# y_center = (ymin + ymax) / 2 / image_height
# width    = (xmax - xmin) / image_width
# height   = (ymax - ymin) / image_height
#
# DEFECT CLASSES:
# 0: crazing          → surface_crack_network
# 1: inclusion        → material_inclusion
# 2: patches          → coating_damage
# 3: pitted_surface   → corrosion_pit
# 4: rolled-in_scale  → surface_irregularity
# 5: scratches        → surface_scratch
#
# Day 2: Write full XML→YOLO conversion code

import os
import glob
import xml.etree.ElementTree as ET
from pathlib import Path

# Class mapping - aerospace terminology
CLASS_MAPPING = {
    'crazing':          0,
    'inclusion':        1,
    'patches':          2,
    'pitted_surface':   3,
    'rolled-in_scale':  4,
    'scratches':        5
}

DATASET_PATH = 'data/raw/NEU-DET'
OUTPUT_PATH  = 'data/processed'
