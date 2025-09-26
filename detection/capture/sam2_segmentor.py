#!/usr/bin/env python3
"""
Clean SAM2 segmentation class for mask generation from bounding boxes.
"""

from pathlib import Path
from typing import Any, Optional, List, Union
import numpy as np
import cv2


class SAM2Segmentor:
    """Simple SAM2 segmentation wrapper."""
    
    def __init__(
        self, 
        config_path: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None, 
        device: str = "cuda"
    ):
        """
        Initialize SAM2 segmentor.
        
        Args:
            config_path: Path to SAM2 config YAML
            checkpoint_path: Path to SAM2 checkpoint
            device: Device for inference (cuda, cuda:0, cpu)
        """
        self.config_path = Path(config_path) if config_path else None
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.device = device
        self.predictor: Optional[Any] = None
        
        if self.config_path and self.checkpoint_path:
            self._load_model()
    
    def _load_model(self) -> bool:
        """Load SAM2 model. Returns True if successful."""
        # Check if files exist
        if not self.config_path.exists():
            print(f"✗ SAM2 config not found: {self.config_path}")
            print("  Consider running: python -m detection.models.model_manager --setup-basic")
            return False
            
        if not self.checkpoint_path.exists():
            print(f"✗ SAM2 checkpoint not found: {self.checkpoint_path}")
            print("  Consider running: python -m detection.models.model_manager --setup-basic")
            return False
        
        try:
            from sam2.build_sam import build_sam2  # type: ignore
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
        except ImportError as exc:
            print("✗ SAM2 not available. Install with:")
            print("  pip install 'git+https://github.com/facebookresearch/sam2.git'")
            return False

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            # Use SAM2 (not 2.1) config with fallback to available checkpoint
            config_name = "sam2_hiera_t.yaml"  # Use SAM2.0 config
            
            # Try loading with the SAM2.0 config and our checkpoint
            try:
                model = build_sam2(config_name, str(self.checkpoint_path), device=self.device)
                self.predictor = SAM2ImagePredictor(model)
                print(f"✓ Loaded SAM2: {self.checkpoint_path.name} on {self.device}")
                return True
            except Exception as config_err:
                print(f"✗ Failed to load SAM2 with config '{config_name}': {config_err}")
                # If config loading fails, we'll fall back to bounding box masks
                return False
        except Exception as exc:
            print(f"✗ Failed to load SAM2: {exc}")
            return False
    
    def is_available(self) -> bool:
        """Check if SAM2 model is loaded and ready."""
        return self.predictor is not None
    
    def generate_masks(self, image: np.ndarray, bounding_boxes: np.ndarray) -> np.ndarray:
        """
        Generate segmentation masks from bounding boxes.
        
        Args:
            image: Input image (BGR format)  
            bounding_boxes: Array of [x1, y1, x2, y2] boxes
            
        Returns:
            Combined binary mask (0 or 255)
        """
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        if not self.is_available() or len(bounding_boxes) == 0:
            return mask
        
        try:
            # Convert BGR to RGB for SAM2
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(rgb_image)
            
            # Process each bounding box
            for box in bounding_boxes:
                outputs = self.predictor.predict(box=box, multimask_output=False)
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    masks = outputs[0]
                else:
                    masks = outputs
                
                # Convert to numpy if needed
                if hasattr(masks, "cpu"):
                    masks_np = masks.cpu().numpy()
                else:
                    masks_np = np.asarray(masks)
                
                # Extract mask from different dimensions
                if masks_np.ndim == 3:
                    mask_candidate = masks_np[0]
                else:
                    mask_candidate = masks_np
                
                if mask_candidate.ndim == 2:
                    # Combine with existing mask
                    mask = np.maximum(mask, (mask_candidate > 0).astype(np.uint8) * 255)
            
            # Reset predictor state
            if hasattr(self.predictor, 'reset_image'):
                self.predictor.reset_image()
            
            return mask
            
        except Exception as exc:
            print(f"SAM2 mask generation failed: {exc}")
            # Reset predictor on error
            if hasattr(self.predictor, 'reset_image'):
                self.predictor.reset_image()
            return mask
    
    def generate_mask_from_detections(self, image: np.ndarray, detections) -> np.ndarray:
        """
        Generate mask from detection objects with fallback to bounding boxes.
        
        Args:
            image: Input image (BGR format)
            detections: List of Detection objects with x1,y1,x2,y2 attributes
            
        Returns:
            Binary mask (0 or 255)
        """
        if not detections:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Convert detections to bounding boxes
        bounding_boxes = np.array([
            [det.x1, det.y1, det.x2, det.y2] for det in detections
        ], dtype=np.float32)
        
        # Try SAM2 segmentation first
        if self.is_available():
            mask = self.generate_masks(image, bounding_boxes)
            if np.any(mask):  # If SAM2 generated any mask
                return mask
        
        # Fallback to bounding box masks
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for det in detections:
            cv2.rectangle(mask, (det.x1, det.y1), (det.x2, det.y2), color=255, thickness=-1)
        
        return mask