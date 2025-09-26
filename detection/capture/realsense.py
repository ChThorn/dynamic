#!/usr/bin/env python3
"""
Basic RealSense D456 streaming utilities and helper functions.

This module provides simple streaming capabilities and utility functions
used by the more advanced capture scripts.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
from typing import Tuple, Optional


def get_realsense_pipeline(
    width: int = 640, 
    height: int = 480, 
    fps: int = 30
) -> Tuple[rs.pipeline, rs.config]:
    """Create and configure a RealSense pipeline."""
    pipeline = rs.pipeline()
    config = rs.config()
    
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    
    return pipeline, config


def get_device_info(pipeline: rs.pipeline) -> dict:
    """Get RealSense device information."""
    profile = pipeline.get_active_profile()
    device = profile.get_device()
    
    return {
        'name': device.get_info(rs.camera_info.name),
        'serial': device.get_info(rs.camera_info.serial_number),
        'firmware': device.get_info(rs.camera_info.firmware_version),
    }


def basic_stream_viewer(width: int = 640, height: int = 480) -> None:
    """Simple viewer for RealSense streams."""
    pipeline, config = get_realsense_pipeline(width, height)
    
    try:
        pipeline.start(config)
        print(f"RealSense D456 streaming at {width}x{height}")
        print("Press 'q' to quit")
        
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue

            # Convert to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Create depth visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.05),
                cv2.COLORMAP_JET
            )

            # Display side by side
            images = np.hstack((color_image, depth_colormap))
            cv2.imshow('RealSense D456 Stream (Color | Depth)', images)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    basic_stream_viewer()