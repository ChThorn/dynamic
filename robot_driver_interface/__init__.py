"""
Robot Driver Interface Package
==============================

Advanced robot control system with two-step smoothing architecture:
1. Planning Module: Collision-free waypoint generation
2. Execution Module: Quintic polynomial smoothing for jerk-free motion

Package Structure:
- src/: Core source code modules
- examples/: Test scripts, demonstrations, and validation tools

Main Components:
- PlanningDynamicExecutor: Advanced robot controller
- Two-step smoothing: Planning + Execution smoothing
- Safety-first operation modes
- Chunked waypoint processing

Author: Robot Control Team
Date: September 2025
"""

__version__ = "1.0.0"
__author__ = "Robot Control Team"

# Import main components from src package
try:
    from .src import (
        PlanningDynamicExecutor,
        PlanningTarget,
        ExecutionWaypoint,
        create_planning_target
    )
    
    # Main exports
    __all__ = [
        'PlanningDynamicExecutor',
        'PlanningTarget', 
        'ExecutionWaypoint',
        'create_planning_target'
    ]
except ImportError:
    # Fallback for when src components are not available
    __all__ = []