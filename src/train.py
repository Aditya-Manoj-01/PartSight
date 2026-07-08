"""
PartSight - YOLOv11 Training Script
Fine-tunes YOLOv11 on NEU-DET defect dataset
using transfer learning from COCO pretrained weights.
"""

from ultralytics import YOLO


def train_model(data_yaml, epochs=50, imgsz=200, batch=16):
    """
    Fine-tune YOLOv11 on defect detection dataset.
    
    Uses transfer learning: starts from pretrained
    COCO weights, adapts final layers to our 6
    aerospace defect classes.
    
    Args:
        data_yaml (str): path to dataset config file
        epochs (int): number of training epochs
        imgsz (int): input image size
        batch (int): batch size
    
    Returns:
        Training results object with metrics
    """
    model = YOLO('yolo11n.pt')
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=15,
        project='PartSight_Training',
        name='run1',
        exist_ok=True
    )
    
    return results


if __name__ == '__main__':
    train_model('data/processed/PartSight_Dataset/data.yaml')
