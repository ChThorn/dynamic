#!/usr/bin/env python3
"""
Gripper Tool Module for Robot Kinematics

This module provides gripper tool support for the robot kinematics system.
It handles the transformation from TCP (Tool Center Point) to the gripper's
functional point (typically the center between gripper fingers).

Key Features:
- Tool offset transformations (TCP to functional point)
- Gripper geometry modeling for collision checking
- Multiple gripper type support
- Tool configuration management
- Integration with existing kinematics modules

Author: Robot Control Team
"""

import os
import yaml
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional
from numpy.linalg import norm

logger = logging.getLogger(__name__)

class GripperToolError(Exception):
    """Custom exception for gripper tool errors."""
    pass

class GripperTool:
    """
    Gripper tool class for handling tool transformations and geometry.
    
    This class manages the transformation between the robot's TCP and the
    gripper's functional working point.
    """
    
    def __init__(self, tool_config: Optional[str] = None, tool_name: str = "default_gripper"):
        """
        Initialize gripper tool with configuration.
        
        Args:
            tool_config: Path to tool configuration file (uses default if None)
            tool_name: Name of the tool configuration to load
        """
        self.tool_name = tool_name
        self.config_path = tool_config or self._get_default_config_path()
        
        # Load tool configuration
        self.config = self._load_tool_config()
        
        # Extract tool parameters
        self.tool_info = self.config.get('tools', {}).get(tool_name, {})
        if not self.tool_info:
            raise GripperToolError(f"Tool '{tool_name}' not found in configuration")
        
        # Parse tool offset transformation
        self.tcp_to_tool_transform = self._parse_tool_offset()
        
        # Tool geometry for collision checking
        self.geometry = self.tool_info.get('geometry', {})
        
        # Tool operation parameters
        self.operation = self.tool_info.get('operation', {})
        
        logger.info(f"Gripper tool '{tool_name}' initialized successfully")
        logger.info(f"Tool offset: {self.get_tool_offset_summary()}")
    
    def _get_default_config_path(self) -> str:
        """Get default path to configuration file."""
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "..", "config", "constraints.yaml"),
            os.path.join(os.path.dirname(__file__), "config", "constraints.yaml"),
            os.path.join(os.path.dirname(__file__), "..", "..", "config", "constraints.yaml")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        return possible_paths[0]  # Return first as default
    
    def _load_tool_config(self) -> Dict[str, Any]:
        """Load tool configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise GripperToolError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise GripperToolError(f"Failed to load configuration: {e}")
    
    def _parse_tool_offset(self) -> np.ndarray:
        """
        Parse tool offset transformation from configuration.
        
        Returns:
            4x4 homogeneous transformation matrix from TCP to tool functional point
        """
        offset_config = self.tool_info.get('tcp_to_tool_offset', {})
        
        # Translation (TCP to tool functional point)
        translation = np.array(offset_config.get('translation', [0.0, 0.0, 0.0]))
        
        # Rotation (RPY in degrees, convert to radians)
        rotation_deg = np.array(offset_config.get('rotation', [0.0, 0.0, 0.0]))
        rotation_rad = np.deg2rad(rotation_deg)
        
        # Create transformation matrix
        T_tcp_to_tool = self._create_transformation_matrix(translation, rotation_rad)
        
        return T_tcp_to_tool
    
    @staticmethod
    def _create_transformation_matrix(translation: np.ndarray, rpy: np.ndarray) -> np.ndarray:
        """
        Create 4x4 transformation matrix from translation and RPY angles.
        
        Args:
            translation: Translation vector [x, y, z]
            rpy: Roll-Pitch-Yaw angles in radians [r, p, y]
            
        Returns:
            4x4 homogeneous transformation matrix
        """
        # Create rotation matrix from RPY
        roll, pitch, yaw = rpy
        
        # Rotation matrices
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation: R = R_z * R_y * R_x
        R = R_z @ R_y @ R_x
        
        # Create homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = translation
        
        return T
    
    def transform_tcp_to_tool(self, T_tcp: np.ndarray) -> np.ndarray:
        """
        Transform pose from TCP frame to tool functional point frame.
        
        Args:
            T_tcp: 4x4 transformation matrix at TCP
            
        Returns:
            4x4 transformation matrix at tool functional point
        """
        # Apply tool offset transformation
        T_tool = T_tcp @ self.tcp_to_tool_transform
        return T_tool
    
    def transform_tool_to_tcp(self, T_tool: np.ndarray) -> np.ndarray:
        """
        Transform pose from tool functional point frame to TCP frame.
        
        Args:
            T_tool: 4x4 transformation matrix at tool functional point
            
        Returns:
            4x4 transformation matrix at TCP
        """
        # Apply inverse tool offset transformation
        # If T_tool = T_tcp @ tool_offset, then T_tcp = T_tool @ inv(tool_offset)
        T_tcp_to_tool_inv = np.linalg.inv(self.tcp_to_tool_transform)
        T_tcp = T_tool @ T_tcp_to_tool_inv
        return T_tcp
    
    def get_tool_offset_translation(self) -> np.ndarray:
        """Get tool offset translation vector."""
        return self.tcp_to_tool_transform[:3, 3]
    
    def get_tool_offset_rotation(self) -> np.ndarray:
        """Get tool offset rotation matrix."""
        return self.tcp_to_tool_transform[:3, :3]
    
    def get_tool_offset_summary(self) -> str:
        """Get human-readable summary of tool offset."""
        translation = self.get_tool_offset_translation()
        return f"Translation: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}] m"
    
    def get_gripper_geometry(self) -> Dict[str, Any]:
        """Get gripper geometry information."""
        return self.geometry.copy()
    
    def get_operation_parameters(self) -> Dict[str, Any]:
        """Get gripper operation parameters."""
        return self.operation.copy()
    
    def check_workspace_collision(self, position: np.ndarray, 
                                 workspace_bounds: Dict[str, Any]) -> bool:
        """
        Check if gripper geometry collides with workspace boundaries.
        
        Args:
            position: Tool functional point position [x, y, z]
            workspace_bounds: Workspace boundary definitions
            
        Returns:
            True if collision detected, False otherwise
        """
        # Get gripper body dimensions
        body_size = self.geometry.get('body_size', [0.0, 0.0, 0.0])
        finger_length = self.geometry.get('finger_length', 0.0)
        
        # Calculate gripper envelope (conservative bounding box)
        # Account for gripper body and extended fingers
        envelope_x = max(body_size[0], finger_length)
        envelope_y = max(body_size[1], self.geometry.get('max_opening', 0.0))
        envelope_z = body_size[2]
        
        # Check if gripper envelope exceeds workspace bounds
        x_min = workspace_bounds.get('x_min', -float('inf'))
        x_max = workspace_bounds.get('x_max', float('inf'))
        y_min = workspace_bounds.get('y_min', -float('inf'))
        y_max = workspace_bounds.get('y_max', float('inf'))
        z_min = workspace_bounds.get('z_min', -float('inf'))
        z_max = workspace_bounds.get('z_max', float('inf'))
        
        # Check boundaries with gripper envelope
        if (position[0] - envelope_x/2 < x_min or position[0] + envelope_x/2 > x_max or
            position[1] - envelope_y/2 < y_min or position[1] + envelope_y/2 > y_max or
            position[2] - envelope_z/2 < z_min or position[2] + envelope_z/2 > z_max):
            return True
        
        return False
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get complete tool information."""
        return {
            'name': self.tool_info.get('name', self.tool_name),
            'type': self.tool_info.get('type', 'unknown'),
            'enabled': self.tool_info.get('enabled', False),
            'offset_translation': self.get_tool_offset_translation().tolist(),
            'offset_rotation_matrix': self.get_tool_offset_rotation().tolist(),
            'geometry': self.get_gripper_geometry(),
            'operation': self.get_operation_parameters()
        }
    
    @staticmethod
    def list_available_tools(config_path: Optional[str] = None) -> Dict[str, str]:
        """
        List all available tools in the configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary mapping tool names to descriptions
        """
        if config_path is None:
            # Use default path logic
            possible_paths = [
                os.path.join(os.path.dirname(__file__), "..", "config", "constraints.yaml"),
                os.path.join(os.path.dirname(__file__), "config", "constraints.yaml")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
        
        if not config_path or not os.path.exists(config_path):
            return {}
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            tools = config.get('tools', {})
            return {name: info.get('name', name) for name, info in tools.items()}
        
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            return {}

# Convenience functions for easy tool management
def create_gripper_tool(tool_name: str = "default_gripper") -> GripperTool:
    """Create a gripper tool instance."""
    return GripperTool(tool_name=tool_name)

def get_active_tool_name(config_path: Optional[str] = None) -> str:
    """Get the name of the currently active tool."""
    if config_path is None:
        # Use default path
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "..", "config", "constraints.yaml"),
            os.path.join(os.path.dirname(__file__), "config", "constraints.yaml")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if not config_path or not os.path.exists(config_path):
        return "default_gripper"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config.get('active_tool', {}).get('current', 'default_gripper')
    
    except Exception:
        return "default_gripper"
