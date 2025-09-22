"""
Robot Driver Interface - Examples and Demonstrations
===================================================

This package contains examples, tests, and demonstration scripts for the 
robot driver interface system.

Components:
- Test scripts: test_*.py files for validation and testing
- Demo scripts: *demo*.py files for demonstrations
- Visualization: motion quality analysis and comparison tools
- Validation: comprehensive system validation scripts

Usage:
    python examples/test_polynomial_trajectory.py
    python examples/safety_first_example.py
    python examples/architecture_summary.py

Author: Robot Control Team  
Date: September 2025
"""

# Note: Examples package - import parent src package for demos
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

__version__ = "1.0.0"