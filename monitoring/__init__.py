"""
Monitoring Package for Robot Motion Planning System

This package provides visualization and monitoring tools for the robot motion planning system,
including interactive pose definition, trajectory visualization, and system monitoring.

Modules:
    - AdvancedPoseVisualizer: Interactive 3D pose control and visualization tool
    - (Future) TrajectoryMonitor: Real-time trajectory execution monitoring
    - (Future) PlanningAnalyzer: Motion planning performance analysis
    - (Future) SystemDashboard: Comprehensive system monitoring dashboard

Author: Robot Motion Planning Team
Date: September 16, 2025
"""

__version__ = "1.0.0"
__author__ = "Robot Motion Planning Team"

# Import main classes for easy access
try:
    from .AdvancedPoseVisualizer import AdvancedPoseVisualizer
    __all__ = ['AdvancedPoseVisualizer']
except ImportError:
    # Graceful handling if dependencies are missing
    __all__ = []

def get_version():
    """Get the package version."""
    return __version__

def get_available_tools():
    """Get list of available monitoring tools."""
    tools = []
    if 'AdvancedPoseVisualizer' in __all__:
        tools.append("AdvancedPoseVisualizer - Interactive 3D pose control")
    return tools