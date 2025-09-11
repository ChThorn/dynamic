#!/usr/bin/env python3
"""
Robot Kinematics Package - Source Module

A clean kinematics library for 6-DOF robot manipulators using
Product of Exponentials (PoE) formulation.

This package provides:
- Forward kinematics computation
- Inverse kinematics solving with damped least squares
- Comprehensive validation utilities

Author: Robot Control Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Robot Control Team"

# Core kinematics classes
from .forward_kinematic import ForwardKinematics
from .inverse_kinematic import InverseKinematics
from .kinematics_validation import KinematicsValidator

# Package metadata
__title__ = "robot_kinematics"
__description__ = "Clean kinematics library for 6-DOF robot manipulators"
__license__ = "MIT"