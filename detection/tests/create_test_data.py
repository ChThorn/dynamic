#!/usr/bin/env python3
"""
Generate synthetic LINEMOD-style test data for validating the point cloud pipeline.
"""

import json
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime


def create_test_dataset(output_dir: Path, num_frames: int = 3):
    """Create synthetic LINEMOD-style test data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    for subdir in ["rgb", "depth", "mask", "meta"]:
        (output_dir / subdir).mkdir(exist_ok=True)
    
    # Synthetic camera intrinsics (similar to RealSense D456)
    intrinsics = {
        "width": 848,
        "height": 480,
        "fx": 421.612,
        "fy": 421.612,
        "ppx": 424.0,
        "ppy": 240.0,
    }
    
    depth_scale = 0.001  # 1mm = 0.001m
    
    for i in range(num_frames):
        frame_id = f"test_{i:04d}"
        
        # Create synthetic RGB image (simple gradient with colored rectangle)
        rgb = np.zeros((480, 848, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.linspace(0, 100, 848).astype(np.uint8)  # Red gradient
        rgb[:, :, 1] = 50  # Green constant
        rgb[:, :, 2] = np.linspace(100, 200, 480).reshape(-1, 1).astype(np.uint8)  # Blue gradient
        
        # Add a colored rectangle (simulating object)
        x1, y1, x2, y2 = 300 + i*20, 180 + i*10, 500 + i*20, 320 + i*10
        rgb[y1:y2, x1:x2] = [255, 128, 64]  # Orange rectangle
        
        # Create synthetic depth image
        depth = np.full((480, 848), 800, dtype=np.uint16)  # Background at 800mm
        # Object closer at 500mm + some variation
        depth[y1:y2, x1:x2] = 500 + np.random.randint(-20, 20, (y2-y1, x2-x1))
        
        # Create mask (object region)
        mask = np.zeros((480, 848), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        
        # Save images
        cv2.imwrite(str(output_dir / "rgb" / f"{frame_id}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_dir / "depth" / f"{frame_id}.png"), depth)
        cv2.imwrite(str(output_dir / "mask" / f"{frame_id}.png"), mask)
        
        # Create metadata
        metadata = {
            "label": "test_object",
            "timestamp": frame_id,
            "intrinsics": intrinsics,
            "depth_scale_m": depth_scale,
            "detections": [
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": 0.95,
                    "label": "test_object"
                }
            ]
        }
        
        with open(output_dir / "meta" / f"{frame_id}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Created test frame {frame_id}")
    
    print(f"Test dataset created in {output_dir}")
    return output_dir


if __name__ == "__main__":
    test_dir = Path("test_linemod_data")
    create_test_dataset(test_dir, 3)