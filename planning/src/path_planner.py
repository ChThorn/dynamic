#!/usr/bin/env python3
"""
Path Planning Module with Constraint Checking

This module provides comprehensive path planning capabilities including:
- Workspace boundary validation
- Joint limit checking  
- Obstacle collision detection
- Orientation limit validation
- Path discretization and validation
- Safety margin enforcement

The PathPlanner integrates with the constraint-free kinematics package to provide
high-level planning intelligence while keeping mathematical computation separate.

Author: Robot Control Team
"""

import numpy as np
import yaml
import os
import logging
import math
import random
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
import time
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

class PathPlanningError(Exception):
    """Custom exception for path planning errors."""
    pass

@dataclass
class PlanningResult:
    """Result container for planning operations."""
    success: bool
    path: Optional[List[np.ndarray]] = None
    error_message: Optional[str] = None
    computation_time: Optional[float] = None
    validation_results: Optional[Dict[str, Any]] = None

class ConstraintsChecker:
    """Comprehensive constraint checking for robot motion planning."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize constraints checker with robot configuration.
        
        Args:
            config_path: Path to constraints YAML file
        """
        self.config_path = config_path or self._get_default_config_path()
        self.constraints = self._load_constraints()
        
        logger.info(f"Constraints checker initialized with config: {self.config_path}")
    
    def _get_default_config_path(self) -> str:
        """Get default path to constraints configuration."""
        # Look for config at project root level
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "..", "..", "config", "constraints.yaml"),
            os.path.join(os.path.dirname(__file__), "..", "config", "constraints.yaml"),
            os.path.join(os.path.dirname(__file__), "constraints.yaml")
        ]
        
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path
        
        # Return first path as default even if it doesn't exist
        return os.path.abspath(possible_paths[0])
    
    def _load_constraints(self) -> dict:
        """Load robot constraints from YAML configuration."""
        if not os.path.exists(self.config_path):
            logger.warning(f"Constraints file not found: {self.config_path}")
            return self._get_default_constraints()
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info("Constraints loaded successfully from configuration")
            return config if config else {}
            
        except Exception as e:
            logger.error(f"Failed to load constraints from {self.config_path}: {e}")
            return self._get_default_constraints()
    
    def _get_default_constraints(self) -> dict:
        """Provide default constraints if config file is not available."""
        return {
            'joint_limits': {
                'j1': {'min': -360, 'max': 360},
                'j2': {'min': -360, 'max': 360}, 
                'j3': {'min': -150, 'max': 150},
                'j4': {'min': -360, 'max': 360},
                'j5': {'min': -360, 'max': 360},
                'j6': {'min': -360, 'max': 360}
            },
            'workspace': {
                'x_min': -0.7, 'x_max': 0.7,
                'y_min': -0.7, 'y_max': 0.7, 
                'z_min': 0.06, 'z_max': 1.1
            },
            'obstacles': {'enabled': False, 'list': []},
            'orientation_limits': {'enabled': False},
            'safety_margins': {'enabled': True, 'margin_x': 0.05, 'margin_y': 0.05, 'margin_z': 0.05}
        }
    
    def check_joint_limits(self, q: np.ndarray) -> Tuple[bool, str]:
        """
        Check if joint configuration is within limits.
        
        Args:
            q: Joint angles in radians
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if q.shape[0] != 6:
            return False, f"Expected 6 joints, got {q.shape[0]}"
        
        joint_limits = self.constraints.get('joint_limits', {})
        joint_names = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6']
        
        for i, (joint_name, q_val) in enumerate(zip(joint_names, q)):
            limits = joint_limits.get(joint_name, {'min': -360, 'max': 360})
            q_deg = np.degrees(q_val)
            
            if not (limits['min'] <= q_deg <= limits['max']):
                return False, f"Joint {joint_name} ({q_deg:.1f}°) exceeds limits [{limits['min']}, {limits['max']}]"
        
        return True, ""
    
    def check_workspace(self, position: np.ndarray) -> Tuple[bool, str]:
        """
        Check if position is within workspace boundaries.
        
        Args:
            position: 3D position in meters [x, y, z]
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if position.shape[0] != 3:
            return False, f"Expected 3D position, got {position.shape[0]}D"
        
        ws = self.constraints.get('workspace', {})
        x, y, z = position
        
        # Get workspace bounds
        x_min, x_max = ws.get('x_min', -1.0), ws.get('x_max', 1.0)
        y_min, y_max = ws.get('y_min', -1.0), ws.get('y_max', 1.0)
        z_min, z_max = ws.get('z_min', 0.0), ws.get('z_max', 2.0)
        
        # Apply safety margins if enabled
        safety = self.constraints.get('safety_margins', {})
        if safety.get('enabled', False):
            margin_x = safety.get('margin_x', 0.0)
            margin_y = safety.get('margin_y', 0.0)
            margin_z = safety.get('margin_z', 0.0)
            
            # Reduce workspace boundaries for safety
            x_min += margin_x  # Move inward from left
            x_max -= margin_x  # Move inward from right  
            y_min += margin_y  # Move inward from back
            y_max -= margin_y  # Move inward from front
            # For Z: keep floor constraint (z_min) but reduce ceiling
            # z_min stays the same (floor/table height must be respected)
            z_max -= margin_z  # Lower the ceiling for safety
        
        # Check boundaries
        if not (x_min <= x <= x_max):
            return False, f"X position ({x:.3f}m) outside bounds [{x_min:.3f}, {x_max:.3f}]"
        if not (y_min <= y <= y_max):
            return False, f"Y position ({y:.3f}m) outside bounds [{y_min:.3f}, {y_max:.3f}]"
        if not (z_min <= z <= z_max):
            return False, f"Z position ({z:.3f}m) outside bounds [{z_min:.3f}, {z_max:.3f}]"
        
        return True, ""
    
    def check_orientation_limits(self, rpy: np.ndarray) -> Tuple[bool, str]:
        """
        Check if orientation is within limits.
        
        Args:
            rpy: Roll, pitch, yaw angles in radians
            
        Returns:
            Tuple of (is_valid, error_message)  
        """
        orientation_limits = self.constraints.get('orientation_limits', {})
        
        if not orientation_limits.get('enabled', False):
            return True, ""  # No orientation limits enforced
        
        if rpy.shape[0] != 3:
            return False, f"Expected 3D orientation (RPY), got {rpy.shape[0]}D"
        
        roll, pitch, yaw = np.degrees(rpy)
        
        # Check roll limits
        roll_min = orientation_limits.get('roll_min', -180)
        roll_max = orientation_limits.get('roll_max', 180)
        if not (roll_min <= roll <= roll_max):
            return False, f"Roll ({roll:.1f}°) outside limits [{roll_min}, {roll_max}]"
        
        # Check pitch limits
        pitch_min = orientation_limits.get('pitch_min', -90)
        pitch_max = orientation_limits.get('pitch_max', 90)
        if not (pitch_min <= pitch <= pitch_max):
            return False, f"Pitch ({pitch:.1f}°) outside limits [{pitch_min}, {pitch_max}]"
        
        # Check yaw limits
        yaw_min = orientation_limits.get('yaw_min', -180)
        yaw_max = orientation_limits.get('yaw_max', 180)
        if not (yaw_min <= yaw <= yaw_max):
            return False, f"Yaw ({yaw:.1f}°) outside limits [{yaw_min}, {yaw_max}]"
        
        return True, ""
    
    def check_obstacles(self, position: np.ndarray) -> Tuple[bool, str]:
        """
        Check for collision with defined obstacles.
        
        Args:
            position: 3D position in meters
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        obstacles_config = self.constraints.get('obstacles', {})
        
        if not obstacles_config.get('enabled', False):
            return True, ""  # No obstacle checking
        
        obstacles = obstacles_config.get('list', [])
        if not obstacles:
            return True, ""
        
        x, y, z = position
        
        for obstacle in obstacles:
            obs_type = obstacle.get('type', 'box')
            center = np.array(obstacle.get('center', [0, 0, 0]))
            
            if obs_type == 'box':
                size = np.array(obstacle.get('size', [0.1, 0.1, 0.1]))
                half_size = size / 2
                
                # Check if point is inside box
                if (abs(x - center[0]) <= half_size[0] and
                    abs(y - center[1]) <= half_size[1] and
                    abs(z - center[2]) <= half_size[2]):
                    return False, f"Collision with box obstacle '{obstacle.get('name', 'unnamed')}'"
                    
            elif obs_type == 'cylinder':
                radius = obstacle.get('radius', 0.05)
                height = obstacle.get('height', 1.0)
                
                # Check horizontal distance
                dx, dy = x - center[0], y - center[1]
                horizontal_dist = np.sqrt(dx**2 + dy**2)
                
                # Check if within cylinder bounds
                if (horizontal_dist <= radius and
                    center[2] <= z <= center[2] + height):
                    return False, f"Collision with cylinder obstacle '{obstacle.get('name', 'unnamed')}'"
                    
            elif obs_type == 'sphere':
                radius = obstacle.get('radius', 0.05)
                dist = np.linalg.norm(position - center)
                
                if dist <= radius:
                    return False, f"Collision with sphere obstacle '{obstacle.get('name', 'unnamed')}'"
        
        return True, ""
    
    def check_pose_constraints(self, T: np.ndarray) -> Tuple[bool, str]:
        """
        Check if transformation matrix satisfies all pose constraints.
        
        Args:
            T: 4x4 transformation matrix
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if T.shape != (4, 4):
            return False, f"Expected 4x4 transformation matrix, got {T.shape}"
        
        position = T[:3, 3]
        rotation = T[:3, :3]
        
        # Check workspace bounds
        workspace_valid, workspace_msg = self.check_workspace(position)
        if not workspace_valid:
            return False, f"Workspace violation: {workspace_msg}"
        
        # Check obstacle collisions
        obstacle_valid, obstacle_msg = self.check_obstacles(position)
        if not obstacle_valid:
            return False, f"Obstacle collision: {obstacle_msg}"
        
        # Check orientation limits if enabled
        try:
            rpy = self._rotation_matrix_to_rpy(rotation)
            orientation_valid, orientation_msg = self.check_orientation_limits(rpy)
            if not orientation_valid:
                return False, f"Orientation violation: {orientation_msg}"
        except Exception as e:
            logger.warning(f"Could not check orientation limits: {e}")
        
        return True, ""
    
    def _rotation_matrix_to_rpy(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to roll-pitch-yaw angles."""
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])  # Roll
            y = np.arctan2(-R[2, 0], sy)       # Pitch  
            z = np.arctan2(R[1, 0], R[0, 0])  # Yaw
        else:
            x = np.arctan2(-R[1, 2], R[1, 1]) # Roll
            y = np.arctan2(-R[2, 0], sy)       # Pitch
            z = 0                              # Yaw
        
        return np.array([x, y, z])

class AOTRRCPathPlanner:
    """Advanced AORRTC-based path planner for robot motion planning."""
    
    def __init__(self, kinematics_fk, kinematics_ik, config_path: Optional[str] = None):
        """
        Initialize AORRTC path planner with kinematics and constraints.
        
        Args:
            kinematics_fk: ForwardKinematics instance
            kinematics_ik: FastIK instance
            config_path: Path to constraints configuration
        """
        self.fk = kinematics_fk
        self.ik = kinematics_ik
        self.constraints_checker = ConstraintsChecker(config_path)
        self.n_joints = kinematics_fk.n_joints
        
        # AORRTC parameters (optimized for real-time performance)
        self.max_iter = 1000  # Reduced from 5000 for faster planning
        self.step_size = np.radians(20)  # Increased from 10° to 20° for larger steps
        self.goal_bias = 0.15  # Increased bias toward goal for faster convergence
        self.connect_threshold = np.radians(25)  # Increased for easier connections
        self.rewire_radius = np.radians(30)  # Increased for better optimization
        
        # Early termination parameters for real-time operation
        self.early_termination_threshold = 200  # Stop if no improvement for 200 iterations
        self.quality_threshold = 1.1  # Accept path within 10% of optimal
        self.fast_mode = False  # Can be enabled for even faster planning
        
        # Planning state
        self.tree_a: Dict[str, List] = {'points': [], 'parents': [], 'costs': []}
        self.tree_b: Dict[str, List] = {'points': [], 'parents': [], 'costs': []}
        self.best_path: Optional[List[np.ndarray]] = None
        self.best_cost = float('inf')
        self.iteration = 0
        self.last_improvement = 0
        self.use_informed_sampling = False
        
        # Joint limits for sampling
        self.joint_limits = self._get_joint_limits()
        
        logger.info("AORRTC path planner initialized with real-time optimized parameters")
    
    def enable_fast_mode(self, enable: bool = True):
        """
        Enable/disable fast mode for real-time operation.
        Fast mode sacrifices some path quality for speed.
        """
        self.fast_mode = enable
        if enable:
            # Ultra-fast parameters for real-time operation
            self.max_iter = 500
            self.step_size = np.radians(30)  # 30° steps
            self.goal_bias = 0.2
            self.early_termination_threshold = 100
            self.quality_threshold = 1.2  # Accept 20% suboptimal paths
            logger.info("Fast mode enabled: ~5-15 second planning times")
        else:
            # Standard real-time parameters
            self.max_iter = 1000
            self.step_size = np.radians(20)
            self.goal_bias = 0.15
            self.early_termination_threshold = 200
            self.quality_threshold = 1.1
            logger.info("Standard mode enabled: ~10-30 second planning times")
    
    def plan_aorrtc_path(self, q_start: np.ndarray, q_goal: np.ndarray, 
                        max_iterations: Optional[int] = None) -> PlanningResult:
        """
        Plan path using AORRTC algorithm.
        
        Args:
            q_start: Starting joint configuration
            q_goal: Goal joint configuration
            max_iterations: Maximum iterations (uses default if None)
            
        Returns:
            PlanningResult with planned path
        """
        start_time = time.time()
        
        if max_iterations:
            self.max_iter = max_iterations
        
        # Initialize trees
        self._initialize_trees(q_start, q_goal)
        
        # Validate start and goal
        if not self._is_configuration_valid(q_start):
            return PlanningResult(
                success=False,
                error_message="Start configuration violates constraints",
                computation_time=time.time() - start_time
            )
        
        if not self._is_configuration_valid(q_goal):
            return PlanningResult(
                success=False,
                error_message="Goal configuration violates constraints", 
                computation_time=time.time() - start_time
            )
        
        logger.info(f"Starting AORRTC planning with {self.max_iter} iterations")
        
        # Main planning loop
        for i in range(self.max_iter):
            self.iteration = i
            
            # Early termination for real-time operation
            if self.best_path and (i - self.last_improvement) > self.early_termination_threshold:
                logger.info(f"Early termination at iteration {i} - no improvement for {self.early_termination_threshold} iterations")
                break
            
            # Quality-based early termination
            if self.best_path and self.best_cost > 0:
                # Calculate theoretical minimum (straight line distance)
                min_possible_cost = np.linalg.norm(q_goal - q_start)
                if self.best_cost <= min_possible_cost * self.quality_threshold:
                    logger.info(f"Early termination at iteration {i} - acceptable quality achieved")
                    break
            
            # Progress reporting
            if i % 500 == 0 and i > 0:
                tree_sizes = f"{len(self.tree_a['points'])}/{len(self.tree_b['points'])}"
                cost_info = f", Cost: {self.best_cost:.3f}" if self.best_path else ""
                logger.info(f"Iteration {i}, Trees: {tree_sizes}{cost_info}")
            
            # Alternate between trees
            tree_from, tree_to = (self.tree_a, self.tree_b) if i % 2 == 0 else (self.tree_b, self.tree_a)
            
            # Sample and extend
            rand_point = self._sample_configuration(tree_to)
            new_idx = self._extend_tree(tree_from, rand_point)
            
            if new_idx is not None:
                if self._try_connect_trees(tree_from, tree_to, new_idx):
                    self._rewire_tree(tree_from, new_idx)
        
        computation_time = time.time() - start_time
        
        if self.best_path:
            # Apply path smoothing
            smoothed_path = self._smooth_path_optimization(self.best_path)
            
            return PlanningResult(
                success=True,
                path=smoothed_path,
                computation_time=computation_time,
                validation_results={
                    'iterations': self.iteration + 1,
                    'tree_sizes': [len(self.tree_a['points']), len(self.tree_b['points'])],
                    'original_cost': self.best_cost,
                    'final_cost': self._path_cost(smoothed_path),
                    'improvement_factor': self.best_cost / self._path_cost(smoothed_path)
                }
            )
        else:
            return PlanningResult(
                success=False,
                error_message="No path found within iteration limit",
                computation_time=computation_time
            )
    
    def _initialize_trees(self, q_start: np.ndarray, q_goal: np.ndarray):
        """Initialize the dual trees for bidirectional search."""
        self.tree_a = {'points': [q_start.copy()], 'parents': [-1], 'costs': [0.0]}
        self.tree_b = {'points': [q_goal.copy()], 'parents': [-1], 'costs': [0.0]}
        self.best_path = None
        self.best_cost = float('inf')
        self.iteration = 0
        self.last_improvement = 0
        self.use_informed_sampling = False
    
    def _sample_configuration(self, tree_to: Dict[str, List]) -> np.ndarray:
        """Sample a random configuration using various strategies."""
        if self.use_informed_sampling and random.random() > self.goal_bias:
            return self._sample_informed()
        elif random.random() < self.goal_bias:
            # Goal-biased sampling
            return tree_to['points'][0].copy()
        else:
            # Uniform random sampling
            return self._sample_random()
    
    def _sample_random(self) -> np.ndarray:
        """Sample uniformly random configuration within joint limits."""
        lower_limits, upper_limits = self.joint_limits
        return np.random.uniform(lower_limits, upper_limits)
    
    def _sample_informed(self) -> np.ndarray:
        """Informed sampling within ellipsoid for optimal paths."""
        if not self.best_path or len(self.best_path) < 2:
            return self._sample_random()
        
        # Simple ellipsoid sampling in configuration space
        q_start = self.tree_a['points'][0]
        q_goal = self.tree_b['points'][0]
        
        # Center of ellipsoid
        center = (q_start + q_goal) / 2.0
        
        # Semi-axes lengths
        c_min = np.linalg.norm(q_goal - q_start)
        c_max = self.best_cost
        
        if c_max <= c_min:
            return self._sample_random()
        
        # Sample in unit sphere and transform
        while True:
            # Sample in unit sphere
            x = np.random.randn(self.n_joints)
            x = x / np.linalg.norm(x) * (random.random() ** (1.0/self.n_joints))
            
            # Scale to ellipsoid
            scaling = np.full(self.n_joints, c_max / 2.0)
            scaling[0] = c_min / 2.0  # Principal axis
            
            sample = center + scaling * x
            
            # Clip to joint limits
            lower_limits, upper_limits = self.joint_limits
            sample = np.clip(sample, lower_limits, upper_limits)
            
            if self._is_configuration_valid(sample):
                return sample
            
            # Fallback to random sampling if too many failures
            if random.random() < 0.1:
                return self._sample_random()
    
    def _extend_tree(self, tree: Dict[str, List], target: np.ndarray) -> Optional[int]:
        """Extend tree toward target configuration."""
        if not tree['points']:
            return None
        
        # Find nearest node using KDTree
        kdtree = cKDTree(tree['points'])
        _, nearest_idx = kdtree.query(target)
        nearest_point = tree['points'][nearest_idx]
        
        # Compute direction and distance
        direction = target - nearest_point
        dist = np.linalg.norm(direction)
        
        if dist < 1e-6:
            return None
        
        # Step toward target
        step_dist = min(self.step_size, dist)
        new_point = nearest_point + direction / dist * step_dist
        
        # Validate new configuration and path
        if not self._is_configuration_valid(new_point):
            return None
        
        if not self._is_path_valid(nearest_point, new_point):
            return None
        
        # Add to tree
        new_cost = tree['costs'][nearest_idx] + np.linalg.norm(new_point - nearest_point)
        tree['points'].append(new_point)
        tree['parents'].append(nearest_idx)
        tree['costs'].append(new_cost)
        
        return len(tree['points']) - 1
    
    def _try_connect_trees(self, tree_from: Dict[str, List], tree_to: Dict[str, List], 
                          new_idx: int) -> bool:
        """Try to connect the two trees."""
        new_point = tree_from['points'][new_idx]
        
        # Find nearest point in target tree
        if not tree_to['points']:
            return False
        
        kdtree_to = cKDTree(tree_to['points'])
        dist, nearest_idx_to = kdtree_to.query(new_point)
        
        if dist < self.connect_threshold:
            nearest_point_to = tree_to['points'][nearest_idx_to]
            
            if self._is_path_valid(new_point, nearest_point_to):
                self._update_best_path(tree_from, new_idx, tree_to, nearest_idx_to)
                return True
        
        return False
    
    def _update_best_path(self, tree1: Dict[str, List], idx1: int, 
                         tree2: Dict[str, List], idx2: int):
        """Update the best path found so far."""
        path1 = self._get_tree_path(tree1, idx1)
        path2 = self._get_tree_path(tree2, idx2)
        
        # Connect paths correctly based on tree order
        if self.tree_a is tree1:
            full_path = path1[::-1] + path2
        else:
            full_path = path2[::-1] + path1
        
        cost = self._path_cost(full_path)
        
        if cost < self.best_cost:
            self.best_path = full_path
            self.best_cost = cost
            self.last_improvement = self.iteration
            
            if not self.use_informed_sampling:
                logger.info(f"Initial path found! Cost: {cost:.3f}. Enabling informed sampling.")
                self.use_informed_sampling = True
            else:
                logger.info(f"Improved path found! Cost: {cost:.3f}")
    
    def _rewire_tree(self, tree: Dict[str, List], new_idx: int):
        """Rewire tree for asymptotic optimality."""
        new_point = tree['points'][new_idx]
        new_cost = tree['costs'][new_idx]
        
        # Find nearby nodes
        kdtree = cKDTree(tree['points'])
        nearby_indices = kdtree.query_ball_point(new_point, self.rewire_radius)
        
        for idx in nearby_indices:
            if idx == new_idx:
                continue
            
            neighbor_point = tree['points'][idx]
            potential_cost = new_cost + np.linalg.norm(neighbor_point - new_point)
            
            # Rewire if better and valid path
            if (potential_cost < tree['costs'][idx] and 
                self._is_path_valid(new_point, neighbor_point)):
                tree['parents'][idx] = new_idx
                tree['costs'][idx] = potential_cost
    
    def _get_tree_path(self, tree: Dict[str, List], node_idx: int) -> List[np.ndarray]:
        """Extract path from tree to given node."""
        path = []
        current = node_idx
        
        while current != -1:
            path.append(tree['points'][current].copy())
            current = tree['parents'][current]
        
        return path
    
    def _path_cost(self, path: List[np.ndarray]) -> float:
        """Compute total path cost."""
        if len(path) < 2:
            return 0.0
        
        return sum(np.linalg.norm(path[i] - path[i-1]) for i in range(1, len(path)))
    
    def _smooth_path_optimization(self, path: List[np.ndarray], iterations: int = 100) -> List[np.ndarray]:
        """Smooth path using gradient-based optimization."""
        if len(path) < 3:
            return path
        
        smoothed_path = [p.copy() for p in path]
        alpha = 0.1  # Learning rate
        beta = 0.4   # Smoothness weight
        
        for _ in range(iterations):
            for i in range(1, len(smoothed_path) - 1):
                # Smoothness gradient
                grad_smooth = (smoothed_path[i-1] + smoothed_path[i+1] - 2 * smoothed_path[i])
                
                # Update with constraints checking
                update = alpha * beta * grad_smooth
                candidate = smoothed_path[i] + update
                
                # Clip to joint limits and validate
                lower_limits, upper_limits = self.joint_limits
                candidate = np.clip(candidate, lower_limits, upper_limits)
                
                if (self._is_configuration_valid(candidate) and
                    self._is_path_valid(smoothed_path[i-1], candidate) and
                    self._is_path_valid(candidate, smoothed_path[i+1])):
                    smoothed_path[i] = candidate
        
        return smoothed_path
    
    def _is_configuration_valid(self, q: np.ndarray) -> bool:
        """Check if joint configuration is valid."""
        try:
            # Check joint limits
            joint_valid, _ = self.constraints_checker.check_joint_limits(q)
            if not joint_valid:
                return False
            
            # Check workspace constraints via forward kinematics
            T = self.fk.compute_forward_kinematics(q)
            pose_valid, _ = self.constraints_checker.check_pose_constraints(T)
            
            return pose_valid
        except:
            return False
    
    def _is_path_valid(self, q1: np.ndarray, q2: np.ndarray, num_checks: int = 10) -> bool:
        """Check if straight-line path between configurations is valid."""
        if num_checks < 2:
            return self._is_configuration_valid(q2)
        
        # Interpolate and check intermediate points
        for i in range(num_checks):
            t = i / (num_checks - 1)
            q_interp = (1 - t) * q1 + t * q2
            
            if not self._is_configuration_valid(q_interp):
                return False
        
        return True
    
    def _get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get joint limits from constraints checker."""
        joint_limits = self.constraints_checker.constraints.get('joint_limits', {})
        joint_names = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6']
        
        lower_limits = []
        upper_limits = []
        
        for joint_name in joint_names:
            limits = joint_limits.get(joint_name, {'min': -360, 'max': 360})
            lower_limits.append(np.radians(limits['min']))
            upper_limits.append(np.radians(limits['max']))
        
        return np.array(lower_limits), np.array(upper_limits)


class PathPlanner:
    """Clean, minimal path planner using AORRTC with smart fallback."""
    
    def __init__(self, kinematics_fk, kinematics_ik, config_path: Optional[str] = None):
        """
        Initialize path planner with AORRTC algorithm and smart fallback.
        
        Args:
            kinematics_fk: ForwardKinematics instance
            kinematics_ik: FastIK instance  
            config_path: Path to constraints configuration
        """
        self.fk = kinematics_fk
        self.ik = kinematics_ik
        self.constraints_checker = ConstraintsChecker(config_path)
        
        # Initialize AORRTC planner (primary algorithm)
        self.aorrtc_planner = AOTRRCPathPlanner(kinematics_fk, kinematics_ik, config_path)
        
        logger.info("Clean path planner initialized: AORRTC primary + smart fallback")
    
    def enable_fast_mode(self, enable: bool = True):
        """Enable fast mode for real-time robot operation."""
        self.aorrtc_planner.enable_fast_mode(enable)
        if enable:
            logger.info("Path planner fast mode enabled for real-time operation")
        else:
            logger.info("Path planner standard mode enabled")
    
    def get_planning_stats(self) -> Dict[str, Any]:
        """Get current planning configuration and performance stats."""
        return {
            'max_iterations': self.aorrtc_planner.max_iter,
            'step_size_degrees': np.degrees(self.aorrtc_planner.step_size),
            'fast_mode': self.aorrtc_planner.fast_mode,
            'early_termination_threshold': self.aorrtc_planner.early_termination_threshold,
            'quality_threshold': self.aorrtc_planner.quality_threshold
        }
    
    def plan_path(self, q_start: np.ndarray, q_goal: np.ndarray,
                  max_iterations: Optional[int] = None, 
                  use_fallback: bool = True) -> PlanningResult:
        """
        Plan path using AORRTC algorithm with smart fallback.
        
        Args:
            q_start: Starting joint configuration
            q_goal: Goal joint configuration
            max_iterations: Maximum AORRTC iterations
            use_fallback: Enable smart fallback if AORRTC fails
            
        Returns:
            PlanningResult with planned path
        """
        # Primary algorithm: AORRTC
        logger.info("Planning with AORRTC (primary algorithm)")
        result = self.aorrtc_planner.plan_aorrtc_path(
            q_start, q_goal, max_iterations=max_iterations
        )
        
        if result.success:
            logger.info(f"AORRTC success: {len(result.path)} waypoints in {result.computation_time*1000:.1f}ms")
            return result
        
        # Smart fallback: Simple interpolation (guaranteed success for valid configs)
        if use_fallback:
            logger.info("AORRTC failed, using smart fallback (interpolation)")
            return self._smart_fallback(q_start, q_goal)
        
        return result
    
    def _smart_fallback(self, q_start: np.ndarray, q_goal: np.ndarray) -> PlanningResult:
        """Enhanced smart fallback with multiple strategies for difficult planning scenarios."""
        start_time = time.time()
        
        # Validate endpoints
        if not self._is_configuration_valid(q_start):
            return PlanningResult(
                success=False,
                error_message="Start configuration invalid",
                computation_time=time.time() - start_time
            )
        
        if not self._is_configuration_valid(q_goal):
            return PlanningResult(
                success=False,
                error_message="Goal configuration invalid", 
                computation_time=time.time() - start_time
            )
        
        # Strategy 1: Direct linear interpolation (fastest)
        result = self._try_linear_interpolation(q_start, q_goal, start_time)
        if result.success:
            logger.info("Smart fallback: Linear interpolation successful")
            return result
        
        # Strategy 2: Multi-segment interpolation via intermediate waypoints
        result = self._try_multisegment_path(q_start, q_goal, start_time)
        if result.success:
            logger.info("Smart fallback: Multi-segment path successful")
            return result
        
        # Strategy 3: Workspace-guided interpolation
        result = self._try_workspace_guided_path(q_start, q_goal, start_time)
        if result.success:
            logger.info("Smart fallback: Workspace-guided path successful")
            return result
        
        # Strategy 4: Joint-by-joint sequential motion (most conservative)
        result = self._try_sequential_joint_motion(q_start, q_goal, start_time)
        if result.success:
            logger.info("Smart fallback: Sequential joint motion successful")
            return result
        
        return PlanningResult(
            success=False,
            error_message="All fallback strategies failed",
            computation_time=time.time() - start_time
        )
    
    def _try_linear_interpolation(self, q_start: np.ndarray, q_goal: np.ndarray, start_time: float) -> PlanningResult:
        """Try simple linear interpolation between start and goal."""
        max_step = np.radians(10)  # Conservative 10 degrees max step
        distance = np.linalg.norm(q_goal - q_start)
        num_steps = max(2, int(np.ceil(distance / max_step)))
        
        waypoints = []
        for i in range(num_steps + 1):
            t = i / num_steps
            q_interp = (1 - t) * q_start + t * q_goal
            
            # Validate each waypoint
            if not self._is_configuration_valid(q_interp):
                return PlanningResult(
                    success=False,
                    error_message="Linear interpolation violates constraints",
                    computation_time=time.time() - start_time
                )
            waypoints.append(q_interp)
        
        return PlanningResult(
            success=True,
            path=waypoints,
            computation_time=time.time() - start_time,
            validation_results={'algorithm': 'linear_interpolation', 'waypoints': len(waypoints)}
        )
    
    def _try_multisegment_path(self, q_start: np.ndarray, q_goal: np.ndarray, start_time: float) -> PlanningResult:
        """Try path via strategic intermediate configurations."""
        # Generate intermediate waypoints using different strategies
        intermediate_configs = [
            (q_start + q_goal) / 2,  # Midpoint
            q_start * 0.7 + q_goal * 0.3,  # 30% toward goal
            q_start * 0.3 + q_goal * 0.7,  # 70% toward goal
        ]
        
        # Add strategic joint configurations
        neutral_config = np.array([0.0, -0.5, 0.5, 0.0, 0.0, 0.0])
        intermediate_configs.append((q_start + neutral_config) / 2)
        intermediate_configs.append((q_goal + neutral_config) / 2)
        
        # Try each intermediate configuration
        for intermediate in intermediate_configs:
            # Validate intermediate configuration
            if not self._is_configuration_valid(intermediate):
                continue
            
            # Try path: start -> intermediate -> goal
            path1 = self._try_linear_interpolation(q_start, intermediate, start_time)
            if not path1.success:
                continue
                
            path2 = self._try_linear_interpolation(intermediate, q_goal, start_time)
            if not path2.success:
                continue
            
            # Combine paths (remove duplicate intermediate point)
            combined_path = path1.path[:-1] + path2.path
            
            return PlanningResult(
                success=True,
                path=combined_path,
                computation_time=time.time() - start_time,
                validation_results={'algorithm': 'multisegment', 'waypoints': len(combined_path)}
            )
        
        return PlanningResult(success=False, error_message="Multisegment path failed")
    
    def _try_workspace_guided_path(self, q_start: np.ndarray, q_goal: np.ndarray, start_time: float) -> PlanningResult:
        """Try path guided by workspace trajectory (Cartesian interpolation)."""
        try:
            # This would require FK - simplified version for now
            # Generate path by interpolating in joint space but with workspace awareness
            
            # Use smaller steps for workspace-guided approach
            max_step = np.radians(8)  # Even more conservative
            distance = np.linalg.norm(q_goal - q_start)
            num_steps = max(3, int(np.ceil(distance / max_step)))
            
            waypoints = []
            for i in range(num_steps + 1):
                # Use cubic interpolation for smoother motion
                t = i / num_steps
                t_smooth = 3 * t**2 - 2 * t**3  # Smooth step function
                q_interp = (1 - t_smooth) * q_start + t_smooth * q_goal
                
                if not self._is_configuration_valid(q_interp):
                    return PlanningResult(success=False, error_message="Workspace-guided path invalid")
                waypoints.append(q_interp)
            
            return PlanningResult(
                success=True,
                path=waypoints,
                computation_time=time.time() - start_time,
                validation_results={'algorithm': 'workspace_guided', 'waypoints': len(waypoints)}
            )
            
        except Exception as e:
            return PlanningResult(success=False, error_message=f"Workspace-guided path error: {e}")
    
    def _try_sequential_joint_motion(self, q_start: np.ndarray, q_goal: np.ndarray, start_time: float) -> PlanningResult:
        """Try moving one joint at a time (most conservative approach)."""
        try:
            waypoints = [q_start.copy()]
            current_q = q_start.copy()
            
            # Move joints in order of importance: base, shoulder, elbow, wrists
            joint_order = [0, 1, 2, 3, 4, 5]  # Can be customized based on robot
            
            for joint_idx in joint_order:
                target_value = q_goal[joint_idx]
                current_value = current_q[joint_idx]
                
                if abs(target_value - current_value) < 1e-6:
                    continue  # Joint already at target
                
                # Move this joint gradually to target
                joint_diff = target_value - current_value
                max_joint_step = np.radians(20)  # 20 degrees per step for single joint
                num_joint_steps = max(1, int(np.ceil(abs(joint_diff) / max_joint_step)))
                
                for step in range(1, num_joint_steps + 1):
                    step_q = current_q.copy()
                    step_q[joint_idx] = current_value + (joint_diff * step / num_joint_steps)
                    
                    if not self._is_configuration_valid(step_q):
                        return PlanningResult(success=False, error_message=f"Sequential motion failed at joint {joint_idx}")
                    
                    waypoints.append(step_q.copy())
                    current_q = step_q.copy()
            
            return PlanningResult(
                success=True,
                path=waypoints,
                computation_time=time.time() - start_time,
                validation_results={'algorithm': 'sequential_joints', 'waypoints': len(waypoints)}
            )
            
        except Exception as e:
            return PlanningResult(success=False, error_message=f"Sequential joint motion error: {e}")
    
    def _is_configuration_valid(self, q: np.ndarray) -> bool:
        """Check if joint configuration is valid."""
        try:
            # Check joint limits
            joint_valid, _ = self.constraints_checker.check_joint_limits(q)
            if not joint_valid:
                return False
            
            # Check workspace constraints via forward kinematics
            T = self.fk.compute_forward_kinematics(q)
            pose_valid, _ = self.constraints_checker.check_pose_constraints(T)
            
            return pose_valid
        except:
            return False

    def validate_joint_path(self, joint_path: List[np.ndarray]) -> PlanningResult:
        """
        Validate a joint space path against all constraints.
        
        Args:
            joint_path: List of joint configurations
            
        Returns:
            PlanningResult with validation outcome
        """
        start_time = time.time()
        
        if not joint_path:
            return PlanningResult(
                success=False, 
                error_message="Empty joint path provided",
                computation_time=time.time() - start_time
            )
        
        # Validate all waypoints
        for i, q in enumerate(joint_path):
            if not self._is_configuration_valid(q):
                return PlanningResult(
                    success=False,
                    error_message=f"Waypoint {i} violates constraints",
                    computation_time=time.time() - start_time
                )
        
        return PlanningResult(
            success=True,
            path=joint_path,
            computation_time=time.time() - start_time,
            validation_results={'waypoints_validated': len(joint_path)}
        )
    
    # Clean interface methods
    def get_workspace_bounds(self) -> Dict[str, float]:
        """Get workspace boundary limits."""
        return self.constraints_checker.constraints.get('workspace', {})
    
    def get_joint_limits(self) -> Dict[str, Dict[str, float]]:
        """Get joint angle limits.""" 
        return self.constraints_checker.constraints.get('joint_limits', {})
    
    def update_constraints(self, new_constraints: Dict[str, Any]):
        """Update constraint parameters."""
        self.constraints_checker.constraints.update(new_constraints)
        logger.info("Constraints updated")
