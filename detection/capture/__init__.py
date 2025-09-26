"""
RealSense camera capture with YOLO detection and SAM2 segmentation.

This module provides tools for capturing RGB-D data with object detection
and segmentation, saving data in LINEMOD-compatible format.
"""

from .realsense_yolo_848x480 import main as capture_main
from .yolo_detector import YOLODetector, Detection
from .sam2_segmentor import SAM2Segmentor
from .realsense import (
    get_realsense_pipeline,
    get_device_info,
    basic_stream_viewer
)

__all__ = [
    "capture_main", 
    "YOLODetector", 
    "SAM2Segmentor", 
    "Detection",
    "get_realsense_pipeline", 
    "get_device_info", 
    "basic_stream_viewer"
]