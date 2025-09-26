#!/usr/bin/env python3
"""
Debug the LINEMOD to point cloud conversion process.
"""

import cv2
import numpy as np
import json
from pathlib import Path

def debug_linemod_frame(dataset_dir: Path, frame_id: str):
    """Debug a single LINEMOD frame to see what's happening."""
    print(f"=== Debugging frame {frame_id} ===")
    
    # Load data
    rgb_path = dataset_dir / "rgb" / f"{frame_id}.png"
    depth_path = dataset_dir / "depth" / f"{frame_id}.png"
    mask_path = dataset_dir / "mask" / f"{frame_id}.png"
    meta_path = dataset_dir / "meta" / f"{frame_id}.json"
    
    rgb = cv2.imread(str(rgb_path))
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    with meta_path.open('r') as f:
        metadata = json.load(f)
    
    print(f"RGB shape: {rgb.shape if rgb is not None else 'None'}")
    print(f"Depth shape: {depth.shape if depth is not None else 'None'}")
    print(f"Depth dtype: {depth.dtype if depth is not None else 'None'}")
    print(f"Depth range: {depth.min() if depth is not None else 'None'} - {depth.max() if depth is not None else 'None'}")
    print(f"Mask shape: {mask.shape if mask is not None else 'None'}")
    print(f"Mask range: {mask.min() if mask is not None else 'None'} - {mask.max() if mask is not None else 'None'}")
    
    depth_scale = metadata.get("depth_scale_m", 0.001)
    print(f"Depth scale: {depth_scale}")
    
    if mask is not None and depth is not None:
        mask_bool = mask > 127
        print(f"Mask pixels > 127: {mask_bool.sum()}")
        
        # Apply mask to depth
        masked_depth = depth.copy().astype(np.float32)
        masked_depth *= depth_scale  # Convert to meters
        masked_depth[~mask_bool] = 0.0
        
        print(f"Masked depth range: {masked_depth[masked_depth > 0].min() if np.any(masked_depth > 0) else 'None'} - {masked_depth[masked_depth > 0].max() if np.any(masked_depth > 0) else 'None'}")
        print(f"Non-zero masked depth points: {np.sum(masked_depth > 0)}")

if __name__ == "__main__":
    debug_linemod_frame(Path("test_linemod_data"), "test_0000")