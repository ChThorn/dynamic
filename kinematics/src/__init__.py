#!/usr/bin/env python3
"""
Robot Kinematics Package - Source Module

A production-ready kinematics library for 6-DOF robot manipulators using
Product of Exponentials (PoE) formulation.

This package provides:
- Forward kinematics computation
- Inverse kinematics solving with damped least squares
- Robot controller interface with safety features
- Comprehensive validation utilities
- Configuration management

Author: Robot Control Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Robot Control Team"

# Core kinematics classes
from .robot_kinematics import RobotKinematics, RobotKinematicsError
from .forward_kinematic import ForwardKinematics
from .inverse_kinematic import InverseKinematics
from .robot_controller import RobotController

# Configuration management
from .config import KinematicsConfig, get_config, set_config_file

# Validation utilities
from .kinematics_validation import KinematicsValidator, run_comprehensive_validation

# Export all public classes and functions
__all__ = [
    # Core classes
    'RobotKinematics',
    'RobotKinematicsError',
    'ForwardKinematics', 
    'InverseKinematics',
    'RobotController',
    
    # Configuration
    'KinematicsConfig',
    'get_config',
    'set_config_file',
    
    # Validation
    'KinematicsValidator',
    'run_comprehensive_validation'
]

# Package metadata
__title__ = "robot_kinematics"
__description__ = "Production-ready kinematics library for 6-DOF robot manipulators"
__url__ = "https://github.com/company/robot_core_control"
__license__ = "MIT"