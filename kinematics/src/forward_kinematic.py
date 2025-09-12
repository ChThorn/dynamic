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
        Initialize constraint-free forward kinematics with robot parameters.
        
        Args:
            constraints_path: Path to constraints YAML file (used only for joint limits)
        """
        self.constraints_path = constraints_path or self._get_default_constraints_path()
        
        # Get hardcoded robot parameters (only joint limits loaded from config)
        self.S, self.M, self.joint_limits = self._get_robot_parameters()
        self.n_joints = self.S.shape[1]
        
        if self.n_joints == 0:
            raise ForwardKinematicsError("No active joints found in the kinematic chain.")
            
        logger.info(f"Constraint-free forward kinematics initialized with {self.n_joints} joints")
    
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
    
    def _load_joint_limits_from_config(self) -> dict:
        """Load joint limits configuration only."""
        if not os.path.exists(self.constraints_path):
            logger.warning(f"Constraints file not found: {self.constraints_path}, using default joint limits")
            return {}
        
        try:
            with open(self.constraints_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Joint limits loaded from: {self.constraints_path}")
            return config.get('joint_limits', {}) if config else {}
        except Exception as e:
            logger.error(f"Failed to load joint limits from {self.constraints_path}: {e}")
            return {}
    
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
            joint_config = self._load_joint_limits_from_config()
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
    
    def compute_forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics using Product of Exponentials.
        
        Pure mathematical computation without constraint checking (Approach 1).
        The forward kinematics is computed as:
        T(q) = exp([S₁]q₁) · exp([S₂]q₂) · ... · exp([Sₙ]qₙ) · M
        
        Args:
            q: Joint angles in radians (n_joints,)
            
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
        
        return T
    
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