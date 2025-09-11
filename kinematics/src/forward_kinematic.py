#!/usr/bin/env python3
"""
Forward Kinematics Module for 6-DOF Robot Manipulator

This module implements forward kinematics using the Product of Exponentials (PoE)
formulation with screw theory. It provides efficient computation of end-effector
pose given joint angles.

Key Features:
- Product of Exponentials (PoE) formulation
- Screw theory implementation
- Matrix exponential computation
- Workspace and constraint checking
- Performance optimizations for real-time use

Author: Robot Control Team
"""

import numpy as np
from numpy.linalg import norm
import logging
from typing import Tuple, Optional
import os
import yaml

logger = logging.getLogger(__name__)

class ForwardKinematicsError(Exception):
    """Custom exception for forward kinematics errors."""
    pass

class ForwardKinematics:
    """Forward kinematics implementation using Product of Exponentials."""
    
    def __init__(self, constraints_path: Optional[str] = None):
        """
        Initialize forward kinematics with robot parameters.
        
        Args:
            constraints_path: Path to constraints YAML file
        """
        self.constraints_path = constraints_path or self._get_default_constraints_path()
        self.constraints = self._load_constraints()
        
        # Get hardcoded robot parameters
        self.S, self.M, self.joint_limits = self._get_robot_parameters()
        self.n_joints = self.S.shape[1]
        
        if self.n_joints == 0:
            raise ForwardKinematicsError("No active joints found in the kinematic chain.")
            
        logger.info(f"Forward kinematics initialized with {self.n_joints} joints")
    
    def _get_default_constraints_path(self) -> str:
        """Get default path to constraints file."""
        # Try multiple possible locations
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "..", "config", "constraints.yaml"),
            os.path.join(os.path.dirname(__file__), "..", "..", "config", "constraints.yaml"),
            os.path.join(os.path.dirname(__file__), "constraints.yaml")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        # Return first path as default even if it doesn't exist
        return possible_paths[0]
    
    def _load_constraints(self) -> dict:
        """Load workspace and orientation constraints."""
        if not os.path.exists(self.constraints_path):
            logger.warning(f"Constraints file not found: {self.constraints_path}")
            return self._get_default_constraints()
        
        try:
            with open(self.constraints_path, "r") as f:
                constraints = yaml.safe_load(f)
            logger.info(f"Constraints loaded from: {self.constraints_path}")
            return constraints or {}
        except Exception as e:
            logger.error(f"Failed to load constraints from {self.constraints_path}: {e}")
            return self._get_default_constraints()
    
    def _get_default_constraints(self) -> dict:
        """Get default workspace constraints."""
        return {
            "workspace": {
                "x_min": -1.0, "x_max": 1.0,
                "y_min": -1.0, "y_max": 1.0, 
                "z_min": 0.0, "z_max": 2.0,
                "safety_margins": {
                    "enabled": False,
                    "margin_x": 0.0,
                    "margin_y": 0.0,
                    "margin_z": 0.0
                }
            },
            "orientation_limits": {
                "roll_min": -180, "roll_max": 180,
                "pitch_min": -180, "pitch_max": 180,
                "yaw_min": -180, "yaw_max": 180
            },
            "obstacles": {
                "enabled": False,
                "list": []
            }
        }
    
    def _get_robot_parameters(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get hardcoded robot parameters for RB3-730ES-U.
        
        Returns:
            S: Screw axes matrix (6 x n_joints)
            M: Home configuration matrix (4 x 4)
            joint_limits: Joint limits array (2 x n_joints)
        """
        # Screw axes (S) for the 6 joints, in the base frame
        # Each column is a screw axis [w_x, w_y, w_z, v_x, v_y, v_z].T
        S = np.array([
            # Joint:    1      2         3         4          5         6
            [0.,     0.,      0.,       0.,       0.,       0.      ],  # ω_x
            [0.,     1.,      1.,       0.,       1.,       0.      ],  # ω_y
            [1.,     0.,      0.,       1.,       0.,       1.      ],  # ω_z
            [0.,    -0.1453, -0.4313,  -0.00645, -0.7753,  -0.00645 ],  # v_x
            [0.,     0.,      0.,       0.,       0.,       0.      ],  # v_y
            [0.,     0.,      0.,       0.,       0.,       0.      ]   # v_z
        ])

        # Home configuration (M): Transformation from base to end-effector
        # when all joint angles are zero
        M = np.array([
            [1., 0., 0.,  0.0     ],
            [0., 1., 0., -0.00645],
            [0., 0., 1.,  0.8753  ],
            [0., 0., 0.,  1.0     ]
        ])

        # Joint limits (radians)
        joint_limits = self._load_joint_limits()

        return S, M, joint_limits
    
    def _load_joint_limits(self) -> np.ndarray:
        """Load joint limits from configuration or use defaults."""
        try:
            joint_config = self.constraints.get('joint_limits', {})
            if not joint_config:
                logger.warning("No joint limits found in config, using default ±π limits")
                return self._get_default_joint_limits()

            # Read limits for each joint
            joint_names = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6']
            lower_limits = []
            upper_limits = []
            
            for joint_name in joint_names:
                if joint_name not in joint_config:
                    logger.warning(f"Joint {joint_name} not found in config, using ±π")
                    lower_limits.append(-np.pi)
                    upper_limits.append(np.pi)
                else:
                    joint_limits = joint_config[joint_name]
                    # Convert from degrees to radians
                    min_rad = np.deg2rad(joint_limits['min'])
                    max_rad = np.deg2rad(joint_limits['max'])
                    lower_limits.append(min_rad)
                    upper_limits.append(max_rad)
            
            joint_limits = np.array([lower_limits, upper_limits])
            logger.info("Joint limits loaded from configuration")
            return joint_limits
            
        except Exception as e:
            logger.error(f"Failed to load joint limits: {e}")
            return self._get_default_joint_limits()
    
    def _get_default_joint_limits(self) -> np.ndarray:
        """Get default joint limits."""
        pi = np.pi
        return np.array([
            [-pi, -pi, -pi, -pi, -pi, -pi],  # Lower limits
            [ pi,  pi,  pi,  pi,  pi,  pi]   # Upper limits
        ])
    
    @staticmethod
    def skew_symmetric(w: np.ndarray) -> np.ndarray:
        """
        Compute skew-symmetric matrix from 3D vector.
        
        Args:
            w: 3D vector
            
        Returns:
            3x3 skew-symmetric matrix
        """
        return np.array([
            [0, -w[2], w[1]], 
            [w[2], 0, -w[0]], 
            [-w[1], w[0], 0]
        ])
    
    @staticmethod
    def matrix_exp6(xi_theta: np.ndarray) -> np.ndarray:
        """
        Compute matrix exponential of a 6D screw vector.
        
        Uses the closed-form solution for SE(3) matrix exponential:
        exp([ξ]θ) = [exp([ω]θ)  G·v·θ]
                    [0         1     ]
        
        Args:
            xi_theta: 6D screw vector [ω·θ, v·θ]
            
        Returns:
            4x4 homogeneous transformation matrix
        """
        w_theta, v_theta = xi_theta[:3], xi_theta[3:]
        theta = norm(w_theta)
        
        T = np.eye(4)
        
        if theta < 1e-12:
            # Small angle approximation
            T[:3, 3] = v_theta
            return T
        
        w = w_theta / theta
        v = v_theta / theta
        w_hat = ForwardKinematics.skew_symmetric(w)
        w_hat2 = w_hat @ w_hat
        
        # Rotation matrix using Rodrigues' formula
        R = np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * w_hat2
        
        # Translation using G matrix
        G = (np.eye(3) * theta + 
             (1 - np.cos(theta)) * w_hat + 
             (theta - np.sin(theta)) * w_hat2)
        p = G @ v
        
        T[:3, :3] = R
        T[:3, 3] = p
        return T
    
    def compute_forward_kinematics(self, q: np.ndarray, 
                                 suppress_warnings: bool = False) -> np.ndarray:
        """
        Compute forward kinematics using Product of Exponentials.
        
        The forward kinematics is computed as:
        T(q) = exp([S₁]q₁) · exp([S₂]q₂) · ... · exp([Sₙ]qₙ) · M
        
        Args:
            q: Joint angles in radians (n_joints,)
            suppress_warnings: If True, suppress workspace/constraint warnings
            
        Returns:
            4x4 homogeneous transformation matrix of end-effector pose
            
        Raises:
            ForwardKinematicsError: If input dimensions are invalid
        """
        if not isinstance(q, np.ndarray) or q.ndim != 1 or q.shape[0] != self.n_joints:
            raise ForwardKinematicsError(
                f"Input q must be a numpy array of shape ({self.n_joints},), "
                f"got shape {q.shape if hasattr(q, 'shape') else 'invalid'}"
            )
        
        # Initialize with identity matrix
        T = np.eye(4)
        
        # Apply each joint transformation using PoE
        for i in range(self.n_joints):
            xi_theta = self.S[:, i] * q[i]
            T = T @ self.matrix_exp6(xi_theta)
        
        # Apply home configuration
        T = T @ self.M
        
        # Check constraints if warnings are enabled
        if not suppress_warnings:
            self._check_constraints(T)
        
        return T
    
    def _check_constraints(self, T: np.ndarray):
        """Check workspace and orientation constraints."""
        pos = T[:3, 3]
        rpy = self.matrix_to_rpy(T[:3, :3])
        
        if not self._check_workspace(pos):
            logger.warning(f"FK: Position {pos} out of workspace bounds")
        
        if not self._check_orientation(rpy):
            logger.warning(f"FK: Orientation {rpy} out of limits")
        
        if not self._check_obstacles(pos):
            logger.warning(f"FK: Position {pos} collides with obstacle")
    
    def _check_workspace(self, pos: np.ndarray) -> bool:
        """Check if position is within workspace bounds."""
        ws = self.constraints.get("workspace", {})
        if not ws:
            return True
        
        x_min = ws.get("x_min", -1.0)
        x_max = ws.get("x_max", 1.0)
        y_min = ws.get("y_min", -1.0)
        y_max = ws.get("y_max", 1.0)
        z_min = ws.get("z_min", 0.0)
        z_max = ws.get("z_max", 2.0)

        # Apply safety margins if enabled (support both nested and top-level keys)
        safety = ws.get("safety_margins", {})
        if not safety or not safety.get("enabled", False):
            # Fallback to top-level safety_margins from YAML
            safety = self.constraints.get("safety_margins", {})
        if safety.get("enabled", False):
            margin_x = safety.get("margin_x", 0.0)
            margin_y = safety.get("margin_y", 0.0)
            margin_z = safety.get("margin_z", 0.0)
            
            x_min += margin_x
            x_max -= margin_x
            y_min += margin_y
            y_max -= margin_y
            z_min += margin_z
            z_max -= margin_z

        return (x_min <= pos[0] <= x_max and
                y_min <= pos[1] <= y_max and
                z_min <= pos[2] <= z_max)
    
    def _check_orientation(self, rpy: np.ndarray) -> bool:
        """Check if orientation is within limits."""
        limits = self.constraints.get("orientation_limits", {})
        if not limits:
            return True
            
        roll, pitch, yaw = np.degrees(rpy)
        return (limits.get("roll_min", -180) <= roll <= limits.get("roll_max", 180) and
                limits.get("pitch_min", -180) <= pitch <= limits.get("pitch_max", 180) and
                limits.get("yaw_min", -180) <= yaw <= limits.get("yaw_max", 180))
    
    def _check_obstacles(self, pos: np.ndarray) -> bool:
        """Check if position collides with obstacles."""
        obstacles_config = self.constraints.get("obstacles", {})
        
        if isinstance(obstacles_config, dict):
            if not obstacles_config.get("enabled", False):
                return True
            obstacles = obstacles_config.get("list", [])
        elif isinstance(obstacles_config, list):
            obstacles = obstacles_config
        else:
            return True
        
        for obs in obstacles:
            if not isinstance(obs, dict):
                continue
                
            obs_type = obs.get("type", "")
            
            if obs_type == "box":
                center = np.array(obs.get("center", [0, 0, 0]))
                size = np.array(obs.get("size", [0, 0, 0])) / 2.0
                
                # Convert to meters if needed
                if np.max(center) > 10:  # Likely in mm
                    center = center / 1000.0
                    size = size / 1000.0
                    
                if np.all(np.abs(pos - center) <= size):
                    return False
                    
            elif obs_type == "cylinder":
                center = np.array(obs.get("center", [0, 0, 0]))
                radius = obs.get("radius", 0)
                height = obs.get("height", 0) / 2.0
                
                # Convert to meters if needed
                if np.max(center) > 10:
                    center = center / 1000.0
                    radius = radius / 1000.0
                    height = height / 1000.0
                    
                horizontal_dist = np.linalg.norm(pos[:2] - center[:2])
                vertical_dist = abs(pos[2] - center[2])
                
                if horizontal_dist <= radius and vertical_dist <= height:
                    return False
        
        return True
    
    @staticmethod
    def matrix_to_rpy(R: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to RPY angles (XYZ convention).
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            RPY angles [roll, pitch, yaw] in radians
        """
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2,1], R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else:
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0
            
        return np.array([x, y, z])
    
    @staticmethod
    def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
        """
        Convert RPY angles to rotation matrix (XYZ convention).
        
        Args:
            rpy: RPY angles [roll, pitch, yaw] in radians
            
        Returns:
            3x3 rotation matrix
        """
        roll, pitch, yaw = rpy
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        
        return Rz @ Ry @ Rx
    
    def get_screw_axes(self) -> np.ndarray:
        """Get the screw axes matrix."""
        return self.S.copy()
    
    def get_home_configuration(self) -> np.ndarray:
        """Get the home configuration matrix."""
        return self.M.copy()
    
    def get_joint_limits(self) -> np.ndarray:
        """Get joint limits."""
        return self.joint_limits.copy()