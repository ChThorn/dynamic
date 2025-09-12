#!/usr/bin/env python3
"""
Enhanced Collision Checker Module

This module provides minimal, essential collision detection for robot motion planning:
- Self-collision detection between robot links
- Floor constraint (robot base at z=0)
- Wood surface constraint (60mm working surface)
- Workspace limit enforcement (robot reach boundaries)
- Intermediate path collision checking

Clean, minimal implementation focusing on the 3 essential environmental constraints.

Author: Robot Control Team
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import yaml

logger = logging.getLogger(__name__)

class CollisionType(Enum):
    """Types of collision detection."""
    NONE = "none"
    SELF_COLLISION = "self_collision"
    ENVIRONMENT_COLLISION = "environment_collision" 
    FLOOR_COLLISION = "floor_collision"
    WORKSPACE_VIOLATION = "workspace_violation"
    JOINT_LIMIT_VIOLATION = "joint_limit_violation"

@dataclass
class CollisionResult:
    """Result of collision checking."""
    is_collision: bool
    collision_type: CollisionType
    details: str
    collision_point: Optional[List[float]] = None
    link_names: Optional[List[str]] = None

class EnhancedCollisionChecker:
    """
    Enhanced collision checker with self-collision and environment detection.
    """
    
    def __init__(self, config_path: str):
        """Initialize collision checker with configuration."""
        self.config_path = config_path
        self.load_configuration()
        self.setup_robot_model()
        
    def load_configuration(self):
        """Load collision checking configuration."""
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            
            # Extract key parameters for 3 essential constraints
            self.workspace = self.config.get('workspace', {})
            self.safety_margins = self.config.get('safety_margins', {})
            self.joint_limits = self.config.get('joint_limits', {})
            
            logger.info(f"Clean collision checker loaded config from: {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load collision config: {e}")
            raise
    
    def setup_robot_model(self):
        """Setup collision checking based on RB3-730ES-U URDF parameters."""
        # Get self-collision config (fallback to URDF-derived values)
        self_collision_config = self.config.get('self_collision', {})
        
        if self_collision_config.get('enabled', True):
            # Critical joint pairs based on actual robot geometry
            # Joint indices: base=0, shoulder=1, elbow=2, wrist1=3, wrist2=4, wrist3=5, tcp=6
            self.critical_joint_pairs = [
                (1, 4),  # Shoulder vs Wrist2 - upper arm can hit wrist assembly
                (1, 5),  # Shoulder vs Wrist3 - upper arm vs wrist3
                (1, 6),  # Shoulder vs TCP - upper arm vs end effector
                (2, 0),  # Elbow vs Base - forearm can hit base when folded
                (3, 0),  # Wrist1 vs Base - wrist assembly vs base
                (4, 0),  # Wrist2 vs Base - wrist2 vs base
            ]
            
            # Load distances from config or use URDF-derived defaults
            critical_pairs = self_collision_config.get('critical_pairs', {})
            self.min_joint_distances = {
                (1, 4): critical_pairs.get('shoulder_wrist2', 0.12),
                (1, 5): critical_pairs.get('shoulder_wrist3', 0.10), 
                (1, 6): critical_pairs.get('shoulder_tcp', 0.15),
                (2, 0): critical_pairs.get('elbow_base', 0.18),
                (3, 0): critical_pairs.get('wrist1_base', 0.16),
                (4, 0): critical_pairs.get('wrist2_base', 0.16),
            }
            
            robot_model = self_collision_config.get('robot_model', 'RB3-730ES-U')
            logger.info(f"{robot_model} collision model: {len(self.critical_joint_pairs)} critical pairs (URDF-derived)")
        else:
            self.critical_joint_pairs = []
            self.min_joint_distances = {}
            logger.info("Self-collision detection disabled")
        
    def check_joint_limits(self, joint_angles: np.ndarray) -> CollisionResult:
        """Check if joint configuration violates joint limits."""
        joint_names = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6']
        
        for i, (name, angle) in enumerate(zip(joint_names, joint_angles)):
            if name in self.joint_limits:
                limits = self.joint_limits[name]
                angle_deg = np.degrees(angle)
                
                if angle_deg < limits['min'] or angle_deg > limits['max']:
                    return CollisionResult(
                        is_collision=True,
                        collision_type=CollisionType.JOINT_LIMIT_VIOLATION,
                        details=f"Joint {name} ({angle_deg:.1f}°) exceeds limits [{limits['min']:.1f}°, {limits['max']:.1f}°]"
                    )
        
        return CollisionResult(False, CollisionType.NONE, "Joint limits OK")
    
    def check_workspace_limits(self, tcp_position: np.ndarray) -> CollisionResult:
        """Check if TCP position violates workspace boundaries."""
        x, y, z = tcp_position
        
        # Apply safety margins if enabled
        margins = self.safety_margins if self.safety_margins.get('enabled', False) else {}
        margin_x = margins.get('margin_x', 0.0)
        margin_y = margins.get('margin_y', 0.0) 
        margin_z = margins.get('margin_z', 0.0)
        
        # Effective workspace with margins
        x_min = self.workspace.get('x_min', -0.7) + margin_x
        x_max = self.workspace.get('x_max', 0.7) - margin_x
        y_min = self.workspace.get('y_min', -0.7) + margin_y
        y_max = self.workspace.get('y_max', 0.7) - margin_y
        z_min = self.workspace.get('z_min', 0.06) + margin_z  # Table height + margin
        z_max = self.workspace.get('z_max', 1.1) - margin_z
        
        if not (x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max):
            return CollisionResult(
                is_collision=True,
                collision_type=CollisionType.WORKSPACE_VIOLATION,
                details=f"TCP position ({x:.3f}, {y:.3f}, {z:.3f}) outside workspace",
                collision_point=[x, y, z]
            )
        
        return CollisionResult(False, CollisionType.NONE, "Workspace limits OK")
    
    def check_floor_collision(self, tcp_position: np.ndarray, joint_positions: List[np.ndarray]) -> CollisionResult:
        """Check collision with floor and wood surface constraints."""
        # Get environment constraints from config
        env_config = self.config.get('environment', {})
        floor_level = env_config.get('floor_level', 0.0)        # Robot base at z=0
        wood_thickness = env_config.get('wood_thickness', 0.06)  # 60mm wood surface
        
        # Check if TCP is below wood surface (minimum working height)
        if tcp_position[2] < wood_thickness:
            return CollisionResult(
                is_collision=True,
                collision_type=CollisionType.FLOOR_COLLISION,
                details=f"TCP below wood surface: {tcp_position[2]*1000:.1f}mm < {wood_thickness*1000:.1f}mm",
                collision_point=tcp_position.tolist()
            )
        
        # Check if any joint position is below floor level (robot base is at z=0)
        for i, pos in enumerate(joint_positions):
            if pos[2] < floor_level:
                return CollisionResult(
                    is_collision=True,
                    collision_type=CollisionType.FLOOR_COLLISION,
                    details=f"Joint {i+1} below floor level: {pos[2]*1000:.1f}mm < 0mm",
                    collision_point=pos.tolist()
                )
        
        return CollisionResult(False, CollisionType.NONE, "Floor/surface collision OK")
    
    def check_self_collision(self, joint_angles: np.ndarray, joint_positions: List[np.ndarray]) -> CollisionResult:
        """Check for self-collision using joint positions from screw theory FK."""
        # Check critical joint pairs for minimum distance violations
        for joint1_idx, joint2_idx in self.critical_joint_pairs:
            # Ensure we have enough joint positions
            if joint1_idx < len(joint_positions) and joint2_idx < len(joint_positions):
                pos1 = joint_positions[joint1_idx]
                pos2 = joint_positions[joint2_idx]
                
                # Calculate distance between joints
                distance = np.linalg.norm(pos1 - pos2)
                
                # Get minimum safe distance for this joint pair
                min_distance = self.min_joint_distances.get((joint1_idx, joint2_idx), 0.08)
                
                if distance < min_distance:
                    joint_names = [f"J{joint1_idx}", f"J{joint2_idx}"]
                    return CollisionResult(
                        is_collision=True,
                        collision_type=CollisionType.SELF_COLLISION,
                        details=f"Self-collision between {joint_names[0]} and {joint_names[1]}: {distance*1000:.1f}mm < {min_distance*1000:.1f}mm",
                        link_names=joint_names
                    )
        
        return CollisionResult(False, CollisionType.NONE, "Self-collision OK")
    
    def check_environment_collision(self, joint_positions: List[np.ndarray], tcp_position: np.ndarray) -> CollisionResult:
        """Check collision with wood surface (already handled in floor_collision check)."""
        # Environment collision is now handled directly in floor_collision check
        # This method kept for interface compatibility but no additional checks needed
        return CollisionResult(False, CollisionType.NONE, "Environment collision OK")
    
    def check_path_collision(self, joint_path: List[np.ndarray], fk_function) -> CollisionResult:
        """Check collision along the entire joint path with intermediate points."""
        resolution = self.config.get('validation', {}).get('path_resolution', 0.01)  # 1cm resolution
        
        for i in range(len(joint_path) - 1):
            q_start = joint_path[i]
            q_end = joint_path[i + 1]
            
            # Calculate number of intermediate points needed
            joint_diff = np.linalg.norm(q_end - q_start)
            num_points = max(2, int(joint_diff / 0.1))  # At least 2 points, more for larger movements
            
            # Check intermediate points
            for j in range(num_points):
                t = j / (num_points - 1)
                q_interp = (1 - t) * q_start + t * q_end
                
                # Get forward kinematics for interpolated configuration
                T = fk_function(q_interp)
                tcp_pos = T[:3, 3]
                
                # Check all collision types for this intermediate point
                result = self.check_configuration_collision(q_interp, tcp_pos, fk_function)
                if result.is_collision:
                    result.details = f"Path collision at waypoint {i}->{i+1}, t={t:.2f}: {result.details}"
                    return result
        
        return CollisionResult(False, CollisionType.NONE, "Path collision OK")
    
    def check_configuration_collision(self, joint_angles: np.ndarray, tcp_position: np.ndarray, 
                                    fk_function=None) -> CollisionResult:
        """Comprehensive collision check for a single robot configuration."""
        
        # 1. Check joint limits
        result = self.check_joint_limits(joint_angles)
        if result.is_collision:
            return result
        
        # 2. Check workspace limits
        result = self.check_workspace_limits(tcp_position)
        if result.is_collision:
            return result
        
        # 3. Check floor/table collision (simplified with TCP only for now)
        joint_positions = [tcp_position]  # Simplified: use TCP position
        result = self.check_floor_collision(tcp_position, joint_positions)
        if result.is_collision:
            return result
        
        # 4. Check self-collision (get joint positions from FK if provided)
        if fk_function is not None:
            joint_positions = self._get_joint_positions(joint_angles, fk_function)
        result = self.check_self_collision(joint_angles, joint_positions)
        if result.is_collision:
            return result
        
        # 5. Check environment collision
        result = self.check_environment_collision(joint_positions, tcp_position)
        if result.is_collision:
            return result
        
        return CollisionResult(False, CollisionType.NONE, "Configuration collision-free")
    
    def _get_joint_positions(self, joint_angles: np.ndarray, fk_function) -> List[np.ndarray]:
        """Get joint positions using forward kinematics (RB3-730ES-U structure)."""
        # Compute joint positions based on URDF kinematic chain
        joint_positions = []
        
        # Base position (link0 - robot base)
        joint_positions.append(np.array([0.0, 0.0, 0.0]))
        
        # Compute intermediate joint positions using partial FK
        for i in range(len(joint_angles)):
            # Compute FK up to joint i+1
            q_partial = np.zeros_like(joint_angles)
            q_partial[:i+1] = joint_angles[:i+1]
            T_i = fk_function(q_partial)
            joint_positions.append(T_i[:3, 3])  # Extract 3D position
        
        # Add TCP position (final transformation)
        T_final = fk_function(joint_angles)
        joint_positions.append(T_final[:3, 3])
        
        return joint_positions
    
    def get_collision_summary(self) -> Dict[str, Any]:
        """Get summary of collision checker configuration."""
        return {
            'critical_joint_pairs': len(self.critical_joint_pairs),
            'safety_margins_enabled': self.safety_margins.get('enabled', False),
            'workspace_bounds': self.workspace,
            'path_resolution': self.config.get('validation', {}).get('path_resolution', 0.01),
            'constraints': ['floor_constraint', 'wood_surface_constraint', 'workspace_limits', 'self_collision']
        }
