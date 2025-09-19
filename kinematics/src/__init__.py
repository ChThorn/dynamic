#!/usr/bin/env python3
"""
Robot Kinematics Package - Source Module

A high-performance kinematics library for 6-DOF robot manipulators using
Product of Exponentials (PoE) formulation, optimized for production applications.

This package provides:
- Fast and accurate forward kinematics computation
- Time-budgeted inverse kinematics solving for real-time applications
- Production-ready IK solver with high success rate
- Comprehensive validation utilities

Author: Robot Control Team
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "Robot Control Team"

# Core kinematics classes
from .forward_kinematic import ForwardKinematics
from .inverse_kinematic import FastIK
from .kinematics_validation import KinematicsValidator

# Package metadata
__title__ = "robot_kinematics"
__description__ = "Production-ready kinematics library for 6-DOF robot manipulators"
__license__ = "MIT"