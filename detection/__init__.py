"""
Detection package for RealSense-based object detection and 3D reconstruction.

Modules:
- capture: RealSense camera capture with YOLO detection and SAM2 segmentation
- processing: Point cloud generation and registration from LINEMOD-style data
- reconstruction: 3D mesh reconstruction from aligned point clouds  
- utils: Common utilities and helper functions
- tests: Test data generation and debugging tools
"""

from . import capture
from . import processing
from . import reconstruction
from . import utils

__version__ = "1.0.0"
__all__ = ["capture", "processing", "reconstruction", "utils"]