#!/usr/bin/env python3
"""
Robot Motion Planning Package

A professional motion planning library for 6-DOF robot manipulators that builds
upon the constraint-free kinematics foundation to provide high-level planning
capabilities with safety and optimization.

This package provides:
- Path planning with constraint checking and collision avoidance
- Trajectory planning with smoothing and optimization
- Motion planning coordination and execution monitoring
- Integration with the kinematics package for mathematical computation

Architecture:
- Uses the constraint-free kinematics package for FK/IK computation
- Handles all constraint checking, safety validation, and optimization
- Provides high-level planning interfaces for robotics applications

Author: Robot Control Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Robot Control Team"

# Core planning classes
from .path_planner import PathPlanner, ConstraintsChecker
from .trajectory_planner import TrajectoryPlanner
from .motion_planner import MotionPlanner

# Export all public classes
__all__ = [
    'PathPlanner',
    'ConstraintsChecker', 
    'TrajectoryPlanner',
    'MotionPlanner'
]

# Package metadata
__title__ = "robot_planning"
__description__ = "Professional motion planning library for 6-DOF robot manipulators"
__license__ = "MIT"
