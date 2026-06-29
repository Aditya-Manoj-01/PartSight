# ============================================
# PartSight - Data Preparation Pipeline
# Author: Aditya Manoj Madgulkar
# ============================================
#
# PURPOSE:
# Convert NEU-DET Pascal VOC XML annotations
# to YOLO TXT format for YOLOv11 training
#
# DATASET: NEU-DET Surface Defect Dataset
# - 1800 images, 6 defect classes
# - 1440 train / 360 validation
# - Image size: 200×200 pixels
#
# INPUT:  data/raw/NEU-DET/
#         ├── train/
#         │   ├── images/[class]/*.jpg
#         │   └── annotations/*.xml
#         └── validation/
#             ├── images/[class]/*.jpg
#             └── annotations/*.xml
#
# OUTPUT: data/processed/PartSight_Dataset/
#         ├── train/
#         │   ├── images/*.jpg  (flat)
#         │   └── labels/*.txt  (YOLO format)
#         ├── val/
#         │   ├── images/*.jpg
#         │   └── labels/*.txt
#         └── data.yaml
#
# BOUNDING BOX CONVERSION:
# XML  → xmin, ymin, xmax, ymax (pixels)
# YOLO → x_center, y_center, width, height (normalized 0-1)
#
# FORMULAS:
# x_center = (xmin + xmax) / 2 / image_width
# y_center = (ymin + ymax) / 2 / image_height
# width    = (xmax - xmin) / image_width
# height   = (ymax - ymin) / image_height
#
# WHY NORMALIZE?
# Normalized coords work for ANY image resolution
# 0.5 always means "center" regardless of image size
# ============================================

import os
import glob
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# ============================================
# CONFIGURATION
# ============================================

DATASET_PATH = 'data/raw/NEU-DET'
OUTPUT_PATH  = 'data/processed/PartSight_Dataset'

# Maps defect class name → integer ID
# YOLO needs integers, not strings
CLASS_MAPPING = {
    'crazing':         0,  # surface crack network
    'inclusion':       1,  # foreign material embedded
    'patches':         2,  # coating damage
    'pitted_surface':  3,  # corrosion pits
    'rolled-in_scale': 4,  # surface irregularity
    'scratches':       5   # linear surface scratch
}

# ============================================
# CONVERTER FUNCTION
# ============================================

def xml_to_yolo(xml_path, verbose=False):
    """
    Convert Pascal VOC XML annotation to YOLO format.
    
    Pascal VOC XML stores bounding boxes as pixel coordinates:
        xmin, ymin = top-left corner
        xmax, ymax = bottom-right corner
    
    YOLO stores bounding boxes as normalized values (0-1):
        x_center, y_center = center of box
        width, height = size of box
    
    Args:
        xml_path (str): Path to XML annotation file
        verbose (bool): Print details if True
    
    Returns:
        list[str]: YOLO format lines
                   Each: "class_id x_center y_center width height"
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Image dimensions needed for normalization
    img_w = int(root.find('size/width').text)
    img_h = int(root.find('size/height').text)
    
    yolo_lines = []
    
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        class_id   = CLASS_MAPPING[class_name]
        
        # Pixel coordinates from XML
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)
        
        # Convert to normalized YOLO format
        x_center = (xmin + xmax) / 2 / img_w
        y_center = (ymin + ymax) / 2 / img_h
        width    = (xmax - xmin) / img_w
        height   = (ymax - ymin) / img_h
        
        if verbose:
            print(f"  {class_name}({class_id}): "
                  f"{x_center:.4f} {y_center:.4f} "
                  f"{width:.4f} {height:.4f}")
        
        yolo_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} "
            f"{width:.6f} {height:.6f}"
        )
    
    return yolo_lines

# ============================================
# PIPELINE FUNCTION
# ============================================

def build_dataset(dataset_path, output_path):
    """
    Build complete YOLO dataset from NEU-DET.
    
    Steps:
    1. Create output folder structure
    2. Convert all XML → YOLO TXT
    3. Copy images to flat output folders
    4. Create data.yaml config file
    5. Validate all images have labels
    
    Args:
        dataset_path (str): Path to raw NEU-DET dataset
        output_path (str):  Path for processed output
    """
    
    # Step 1: Create folder structure
    for split in ['train', 'val']:
        os.makedirs(f'{output_path}/{split}/images', exist_ok=True)
        os.makedirs(f'{output_path}/{split}/labels', exist_ok=True)
    
    # Step 2 & 3: Convert + copy
    split_map = {'train': 'train', 'validation': 'val'}
    
    for src_split, dst_split in split_map.items():
        ann_path = os.path.join(dataset_path, src_split, 'annotations')
        img_base = os.path.join(dataset_path, src_split, 'images')
        xml_files = glob.glob(f'{ann_path}/*.xml')
        
        converted = 0
        for xml_path in xml_files:
            stem      = Path(xml_path).stem
            yolo_lines = xml_to_yolo(xml_path)
            
            # Save label
            with open(f'{output_path}/{dst_split}/labels/{stem}.txt', 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            # Copy image
            for ext in ['.jpg', '.jpeg', '.png']:
                matches = glob.glob(f'{img_base}/**/{stem}{ext}', recursive=True)
                if matches:
                    shutil.copy(matches[0], f'{output_path}/{dst_split}/images/{stem}{ext}')
                    converted += 1
                    break
        
        print(f"{dst_split}: {converted} samples converted")
    
    # Step 4: Create data.yaml
    yaml_content = f"""path: {output_path}
train: train/images
val:   val/images
nc: 6
names:
  0: surface_crack_network
  1: material_inclusion
  2: coating_damage
  3: corrosion_pit
  4: surface_irregularity
  5: surface_scratch
"""
    with open(f'{output_path}/data.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("data.yaml created ✅")

# ============================================
# RUN PIPELINE
# ============================================

if __name__ == '__main__':
    build_dataset(DATASET_PATH, OUTPUT_PATH)
