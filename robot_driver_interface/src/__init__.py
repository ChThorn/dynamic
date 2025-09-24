"""
Robot Driver Interface - Core Source Code
=========================================

This package contains the core robot control and motion planning execution modules.

Main components:
- planning_dynamic_executor: Advanced robot controller with two-step smoothing
- visualizer_tool_TCP: Interactive 3D pose selection and visualization
- pose_integration_bridge: Seamless integration between visualizer and executor
- Additional core modules for robot control and trajectory generation
"""

from .planning_dynamic_executor import (
    PlanningDynamicExecutor,
    PlanningTarget, 
    ExecutionWaypoint,
    create_planning_target
)

from .visualizer_tool_TCP import visualizer_tool_TCP

from .pose_integration_bridge import (
    PoseIntegrationBridge,
    create_integrated_workflow
)

__all__ = [
    'PlanningDynamicExecutor',
    'PlanningTarget',
    'ExecutionWaypoint', 
    'create_planning_target',
    'visualizer_tool_TCP',
    'PoseIntegrationBridge',
    'create_integrated_workflow'
]

__version__ = "1.0.0"