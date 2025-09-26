"""
Test utilities and debugging tools.

Tools for generating test data and debugging the detection pipeline.
"""

from .create_test_data import create_test_dataset
from .debug_linemod import debug_linemod_frame

__all__ = ["create_test_dataset", "debug_linemod_frame"]