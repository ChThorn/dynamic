#!/usr/bin/env python3
"""
Clean YOLO object detection class for RealSense capture pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union
import numpy as np


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    label: str


class YOLODetector:
    """Simple YOLO detection wrapper."""
    
    def __init__(self, model_path: Union[str, Path] = "yolov8n.pt", confidence_threshold: float = 0.5, target_classes: Optional[List[int]] = None):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO model file (str or Path)
            confidence_threshold: Minimum confidence for detections
            target_classes: List of class IDs to filter for (None = detect all classes)
        """
        self.model_path = str(model_path)  # Convert Path to string for ultralytics
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes
        self.model: Optional[Any] = None
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load YOLO model. Returns True if successful."""
        try:
            from ultralytics import YOLO
            
            # Check if model file exists
            model_path = Path(self.model_path)
            if not model_path.exists() and not model_path.name.startswith("yolov8"):
                print(f"✗ YOLO model file not found: {self.model_path}")
                print("  Consider running: python -m detection.models.model_manager --setup-basic")
                return False
            
            self.model = YOLO(self.model_path)
            class_info = f" (filtering for classes: {self.target_classes})" if self.target_classes else ""
            print(f"✓ Loaded YOLO model: {model_path.name}{class_info}")
            return True
            
        except ImportError:
            print("✗ YOLO not available. Install with: pip install ultralytics")
            return False
        except Exception as exc:
            print(f"✗ Failed to load YOLO model: {exc}")
            if "model file not found" in str(exc).lower():
                print("  Consider running: python -m detection.models.model_manager --setup-basic")
            return False
    
    def is_available(self) -> bool:
        """Check if YOLO model is loaded and ready."""
        return self.model is not None
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Run YOLO detection on image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of Detection objects
        """
        if not self.is_available():
            return []
        
        try:
            # Pass target classes to YOLO inference if specified
            if self.target_classes:
                results = self.model(image, conf=self.confidence_threshold, classes=self.target_classes, verbose=False)
            else:
                results = self.model(image, conf=self.confidence_threshold, verbose=False)
                
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label = self.model.names[cls_id] if hasattr(self.model, 'names') else f"class_{cls_id}"
                        
                        detections.append(Detection(
                            x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                            confidence=conf, label=label
                        ))
            
            return detections
            
        except Exception as exc:
            print(f"YOLO inference failed: {exc}")
            return []
    
    def get_bounding_boxes(self, detections: List[Detection]) -> np.ndarray:
        """Convert detections to numpy array of bounding boxes."""
        if not detections:
            return np.empty((0, 4), dtype=np.float32)
        return np.array([[det.x1, det.y1, det.x2, det.y2] for det in detections], dtype=np.float32)