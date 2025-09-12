#!/usr/bin/env python3
"""
Configuration Space Analysis Module

This module provides tools for analyzing the robot's configuration space (C-space),
including:
- Pre-computation of reachability maps
- Singularity detection and analysis
- Configuration space sampling and clustering
- C-space to workspace mapping
- Probabilistic roadmap generation

These analysis tools improve motion planning performance by creating a deeper
understanding of the robot's movement capabilities and limitations.

Author: Robot Control Team
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict
import os
import pickle
from dataclasses import dataclass

# For visualization if available
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Module-level constants / thresholds
# ------------------------------------------------------------
REACHABILITY_DISTANCE = 0.05  # meters (FK sample within 5cm of ws point)
REACHABILITY_DISTANCE_VIS = 0.10  # meters for visualization heatmap
IK_SEED_NEIGHBOR_COUNT = 5
SINGULARITY_THRESHOLD = 0.1
HIGH_MANIP_THRESHOLD = 0.8
MAX_SEED_PERTURB_TRIALS = 10
SINGULARITY_REPULSION_RADIUS = 1.0
SINGULARITY_PATH_STEP_SIZE = 0.1
SINGULARITY_PATH_MAX_ITERS = 100

# Safe acos to avoid NaNs from slight numeric drift
def _safe_acos(x: float) -> float:
    return np.arccos(np.clip(x, -1.0, 1.0))


@dataclass
class CSpaceRegion:
    """Represents a region in configuration space."""
    # Bounds in joint space (min and max for each joint)
    bounds: np.ndarray  # Shape: (2, n_joints)
    # Estimated reachability (0-1, higher is more reachable)
    reachability: float
    # Estimated manipulability (0-1, higher is more manipulable)
    manipulability: float
    # List of workspace positions reachable from this region
    workspace_samples: List[np.ndarray]
    # Metadata
    properties: Dict[str, Any] = None


class ConfigurationSpaceAnalyzer:
    """
    Analyzes robot's configuration space for motion planning optimization.
    
    This class provides tools to:
    1. Generate a discretized map of the C-space
    2. Identify singularity and high-manipulability regions
    3. Pre-compute reachability information
    4. Build probabilistic roadmaps for faster planning
    """
    
    def __init__(self, forward_kinematics, inverse_kinematics, 
                 collision_checker=None, resolution=10):
        """
        Initialize configuration space analyzer.
        
        Args:
            forward_kinematics: ForwardKinematics instance
            inverse_kinematics: InverseKinematics instance
            collision_checker: CollisionChecker instance (optional)
            resolution: Resolution of C-space discretization per dimension
        """
        self.fk = forward_kinematics
        self.ik = inverse_kinematics
        self.collision_checker = collision_checker
        self.resolution = resolution
        self.n_joints = forward_kinematics.n_joints
        
        # Joint limits
        self.joint_limits = forward_kinematics.get_joint_limits()
        
        # Results storage
        self.c_space_map = None
        self.reachability_map = None
        self.roadmap = None
        self.singularity_regions = []
        self.high_manipulability_regions = []
        
        logger.info(f"Configuration space analyzer initialized with resolution {resolution}")
    
    def build_reachability_map(self, 
                               workspace_samples: int = 1000, 
                               c_space_samples: int = 5000,
                               save_path: Optional[str] = None,
                               use_cache: bool = True) -> Dict[str, Any]:
        """
        Build a reachability map of the workspace.
        
        Args:
            workspace_samples: Number of workspace points to sample
            c_space_samples: Number of configuration space points to sample
            save_path: Path to save the reachability map (optional)
            
        Returns:
            Dictionary with reachability analysis results
        """
        start_time = time.time()

        # Attempt to load cached map if requested
        if use_cache and save_path and os.path.exists(save_path):
            loaded = self._load_reachability_map(save_path)
            if loaded:
                logger.info(f"Loaded cached reachability map from {save_path}")
                return self.reachability_map

        logger.info(f"Building reachability map with {workspace_samples} workspace samples")
        
        # 1. Generate workspace samples (grid + sphere)
        ws_points = self._generate_workspace_samples(workspace_samples)
        logger.info(f"Generated {len(ws_points)} workspace points")
        
        # 2. Generate C-space samples
        c_space_points = self._generate_c_space_samples(c_space_samples)
        logger.info(f"Generated {len(c_space_points)} C-space points")
        
        # 3. Compute forward kinematics for all C-space points
        ws_from_cs = []
        valid_configs = []
        manipulability = []
        
        for q in c_space_points:
            # Check if configuration is valid (collision-free)
            if self.collision_checker and not self.collision_checker.is_valid_joint_position(q):
                continue
                
            valid_configs.append(q)
            
            # Compute end-effector pose
            T = self.fk.compute_forward_kinematics(q)
            ws_from_cs.append(T[:3, 3])  # Store position only
            
            # Compute manipulability at this configuration
            J = self._compute_jacobian(q)
            m = self._compute_manipulability_measure(J)
            manipulability.append(m)
        
        ws_from_cs = np.array(ws_from_cs)
        valid_configs = np.array(valid_configs)
        manipulability = np.array(manipulability)
        
        # 4. Build the reachability map
        reachability_map = self._build_reachability_map(ws_points, ws_from_cs, valid_configs, manipulability)
        
        # 5. Identify singularity regions
        singularity_regions = self._identify_singularity_regions(valid_configs, manipulability)
        
        # 6. Identify high-manipulability regions
        high_manip_regions = self._identify_high_manipulability_regions(valid_configs, manipulability)
        
        self.reachability_map = {
            'workspace_points': ws_points,
            'c_space_points': valid_configs,
            'workspace_from_cspace': ws_from_cs,
            'manipulability': manipulability,
            'singularity_regions': singularity_regions,
            'high_manipulability_regions': high_manip_regions,
            'map_data': reachability_map,
            'generation_time': time.time() - start_time
        }
        
        # Save if requested
        if save_path:
            self._save_reachability_map(save_path)
        
        logger.info(f"Reachability map built in {time.time() - start_time:.1f} seconds")
        logger.info(f"Found {len(singularity_regions)} singularity regions and " 
                   f"{len(high_manip_regions)} high-manipulability regions")
        
        return self.reachability_map
    
    def build_probabilistic_roadmap(self, 
                                   num_samples: int = 1000,
                                   k_neighbors: int = 10,
                                   max_edge_distance: float = 0.5,
                                   save_path: Optional[str] = None,
                                   use_cache: bool = True) -> Dict[str, Any]:
        """
        Build a probabilistic roadmap (PRM) for faster path planning.
        
        Args:
            num_samples: Number of C-space samples to use
            k_neighbors: Number of neighbors to connect per node
            max_edge_distance: Maximum edge length in C-space
            save_path: Path to save the roadmap (optional)
            
        Returns:
            Dictionary with roadmap data
        """
        start_time = time.time()
        # Derived cache path
        if save_path and use_cache and os.path.exists(save_path):
            try:
                with open(save_path, 'rb') as f:
                    self.roadmap = pickle.load(f)
                logger.info(f"Loaded cached roadmap from {save_path}")
                return self.roadmap
            except Exception as e:
                logger.warning(f"Failed to load cached roadmap, rebuilding. Reason: {e}")

        logger.info(f"Building probabilistic roadmap with {num_samples} samples")
        
        # 1. Generate C-space samples
        samples = self._generate_c_space_samples(num_samples)
        valid_samples = []
        
        # 2. Filter out invalid samples
        for q in samples:
            if self.collision_checker and not self.collision_checker.is_valid_joint_position(q):
                continue
            valid_samples.append(q)
        
        valid_samples = np.array(valid_samples)
        logger.info(f"Found {len(valid_samples)} valid configurations")
        
        # 3. Build the roadmap
        roadmap = self._build_roadmap(valid_samples, k_neighbors, max_edge_distance)
        
        self.roadmap = {
            'nodes': valid_samples,
            'edges': roadmap,
            'k_neighbors': k_neighbors,
            'max_edge_distance': max_edge_distance,
            'generation_time': time.time() - start_time
        }
        
        # Save if requested
        if save_path:
            try:
                with open(save_path, 'wb') as f:
                    pickle.dump(self.roadmap, f)
                logger.info(f"Roadmap saved to: {save_path}")
            except Exception as e:
                logger.error(f"Failed to save roadmap: {e}")
        
        logger.info(f"Probabilistic roadmap built in {time.time() - start_time:.1f} seconds")
        logger.info(f"Roadmap has {len(valid_samples)} nodes and {sum(len(edges) for edges in roadmap)} edges")
        
        return self.roadmap
    
    def get_best_ik_region(self, target_position: np.ndarray, 
                          orientation_preference: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Get the best C-space region for IK solving for a target position.
        
        Args:
            target_position: Target end-effector position (3D)
            orientation_preference: Preferred orientation as RPY angles (optional)
            
        Returns:
            Reference configuration for IK initialization, or None if not found
        """
        if self.reachability_map is None:
            logger.warning("Reachability map not built, cannot get best IK region")
            return None
        
        # Find closest workspace point in reachability map
        ws_points = self.reachability_map['workspace_points']
        distances = np.linalg.norm(ws_points - target_position, axis=1)
        nearest_indices = np.argsort(distances)[:IK_SEED_NEIGHBOR_COUNT]
        
        best_q = None
        best_score = float('-inf')
        
        for idx in nearest_indices:
            # Look up C-space points that can reach this workspace point
            ws_point = ws_points[idx]
            
            # Find C-space points that can reach this workspace region
            ws_from_cs = self.reachability_map['workspace_from_cspace']
            reach_dists = np.linalg.norm(ws_from_cs - ws_point, axis=1)
            
            # Get configurations that are close to this workspace point
            close_indices = np.where(reach_dists < REACHABILITY_DISTANCE_VIS)[0]
            
            if len(close_indices) == 0:
                continue
                
            # Get manipulability scores for these configurations
            manip_scores = self.reachability_map['manipulability'][close_indices]
            
            # Find configuration with highest manipulability
            best_idx = close_indices[np.argmax(manip_scores)]
            q = self.reachability_map['c_space_points'][best_idx]
            score = manip_scores[np.argmax(manip_scores)]
            
            if score > best_score:
                best_score = score
                best_q = q.copy()
        
        return best_q
    
    def get_singularity_free_path(self, start_q: np.ndarray, goal_q: np.ndarray, 
                                 discretization: int = 20) -> List[np.ndarray]:
        """
        Get a path from start to goal that avoids singularities.
        
        Args:
            start_q: Start configuration
            goal_q: Goal configuration
            discretization: Number of waypoints for path discretization
            
        Returns:
            List of configurations forming a path, or empty list if not found
        """
        if not self.singularity_regions:
            logger.warning("No singularity regions identified, using straight path")
            # Return straight-line path
            return [start_q + t * (goal_q - start_q) for t in np.linspace(0, 1, discretization)]
        
        # Try to find a path using the roadmap if available
        if self.roadmap is not None:
            path = self._find_path_in_roadmap(start_q, goal_q)
            if path:
                return path
        
        # Otherwise, use potential field approach to avoid singularities
        return self._compute_singularity_free_path(start_q, goal_q, discretization)
    
    def suggest_optimal_configuration(self, target_pose: np.ndarray, 
                                     q_current: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Suggest optimal configuration to reach a target pose.
        
        Uses pre-computed C-space analysis to find a configuration that:
        1. Reaches the target pose
        2. Has high manipulability
        3. Avoids singularities
        4. Is close to current configuration (if provided)
        
        Args:
            target_pose: Target end-effector pose as 4x4 matrix
            q_current: Current robot configuration (optional)
            
        Returns:
            Optimal joint configuration
        """
        # Extract position from pose
        position = target_pose[:3, 3]
        
        # 1. Use reachability map to get a good IK seed
        q_seed = self.get_best_ik_region(position)
        
        # If no good seed found, use current configuration or default
        if q_seed is None:
            if q_current is not None:
                q_seed = q_current
            else:
                q_seed = np.zeros(self.n_joints)
        
        # 2. Solve IK with the seed
        q_sol, converged = self.ik.solve(target_pose, q_init=q_seed)
        
        if not converged:
            logger.warning("Failed to find optimal configuration, using best seed")
            return q_seed
        
        # 3. Check if solution is near singularity
        J = self._compute_jacobian(q_sol)
        manip = self._compute_manipulability_measure(J)
        
        if manip < SINGULARITY_THRESHOLD:  # Near singularity
            logger.info("Solution is near singularity, trying alternatives")
            
            # Try alternative IK solutions with random seeds
            best_q = q_sol
            best_manip = manip
            
            for _ in range(MAX_SEED_PERTURB_TRIALS):
                # Random perturbation of seed
                q_perturbed = q_seed + np.random.normal(0, 0.2, self.n_joints)
                q_perturbed = np.clip(q_perturbed, self.joint_limits[0], self.joint_limits[1])
                
                q_alt, conv = self.ik.solve(target_pose, q_init=q_perturbed)
                
                if conv:
                    J_alt = self._compute_jacobian(q_alt)
                    manip_alt = self._compute_manipulability_measure(J_alt)
                    
                    if manip_alt > best_manip:
                        best_manip = manip_alt
                        best_q = q_alt
            
            q_sol = best_q
        
        return q_sol
    
    def visualize_workspace_reachability(self, resolution=20) -> Optional[Any]:
        """
        Visualize the reachability of the workspace as a 3D heatmap.
        
        Args:
            resolution: Resolution of the visualization grid
            
        Returns:
            Figure object if matplotlib is available, None otherwise
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, cannot visualize workspace reachability")
            return None
            
        if self.reachability_map is None:
            logger.warning("Reachability map not built, cannot visualize")
            return None
        
        # Create a visualization grid
        x_range = [-0.7, 0.7]
        y_range = [-0.7, 0.7]
        z_range = [0.1, 1.0]
        
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        z = np.linspace(z_range[0], z_range[1], resolution)
        
        grid_points = np.array([(xi, yi, zi) 
                              for xi in x 
                              for yi in y 
                              for zi in z])
        
        # Compute reachability score for each grid point
        ws_from_cs = self.reachability_map['workspace_from_cspace']
        reachability = np.zeros(len(grid_points))
        
        # For each grid point, count how many C-space points can reach it
        for i, point in enumerate(grid_points):
            distances = np.linalg.norm(ws_from_cs - point, axis=1)
            reachability[i] = np.sum(distances < REACHABILITY_DISTANCE_VIS)
        
        # Normalize reachability
        if np.max(reachability) > 0:
            reachability = reachability / np.max(reachability)
        
        # Reshape for visualization
        reachability_3d = reachability.reshape(resolution, resolution, resolution)
        
        # Create 3D visualization
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot workspace points with color based on reachability
        sc = ax.scatter(grid_points[:, 0], grid_points[:, 1], grid_points[:, 2], 
                      c=reachability, cmap='viridis', alpha=0.2, s=10)
        
        # Plot singularity regions
        if self.singularity_regions:
            for region in self.singularity_regions:
                center = region['center']
                ax.scatter(center[0], center[1], center[2], 
                          color='red', marker='x', s=100)
        
        # Plot high-manipulability regions
        if self.high_manipulability_regions:
            for region in self.high_manipulability_regions:
                center = region['center']
                ax.scatter(center[0], center[1], center[2], 
                          color='green', marker='o', s=50)
        
        plt.colorbar(sc, label='Reachability')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Workspace Reachability Map')
        
        return fig
    
    def visualize_cspace_regions(self) -> Optional[Any]:
        """
        Visualize C-space regions and their properties.
        
        Returns:
            Figure object if matplotlib is available, None otherwise
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, cannot visualize C-space regions")
            return None
            
        if self.reachability_map is None:
            logger.warning("Reachability map not built, cannot visualize")
            return None
        
        # Get data
        c_space_points = self.reachability_map['c_space_points']
        manipulability = self.reachability_map['manipulability']
        
        # Create visualization for first 2 joints (for simplicity)
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot configurations colored by manipulability
        sc = ax.scatter(np.rad2deg(c_space_points[:, 0]), 
                       np.rad2deg(c_space_points[:, 1]), 
                       c=manipulability, cmap='viridis', alpha=0.7)
        
        # Mark singularity regions
        if self.singularity_regions:
            for region in self.singularity_regions:
                bounds = region['bounds']
                x_center = np.rad2deg((bounds[0, 0] + bounds[1, 0]) / 2)
                y_center = np.rad2deg((bounds[0, 1] + bounds[1, 1]) / 2)
                ax.scatter(x_center, y_center, marker='x', color='red', s=100)
        
        # Mark high-manipulability regions
        if self.high_manipulability_regions:
            for region in self.high_manipulability_regions:
                bounds = region['bounds']
                x_center = np.rad2deg((bounds[0, 0] + bounds[1, 0]) / 2)
                y_center = np.rad2deg((bounds[0, 1] + bounds[1, 1]) / 2)
                ax.scatter(x_center, y_center, marker='o', color='green', s=50)
        
        plt.colorbar(sc, label='Manipulability')
        ax.set_xlabel('Joint 1 (degrees)')
        ax.set_ylabel('Joint 2 (degrees)')
        ax.set_title('C-space Manipulability Map')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _generate_workspace_samples(self, num_samples: int) -> np.ndarray:
        """Generate representative workspace samples."""
        # Robot workspace parameters (estimated)
        center = np.array([0, 0, 0.5])  # Center of workspace
        max_reach = 0.73  # Maximum reach
        
        # Generate samples using mixed strategy
        samples = []
        
        # Strategy 1: Grid-based samples (60%)
        n_grid = int(num_samples * 0.6)
        grid_points = self._generate_grid_samples(n_grid)
        samples.extend(grid_points)
        
        # Strategy 2: Spherical samples (40%)
        n_sphere = num_samples - len(samples)
        for _ in range(n_sphere):
            # Random direction
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            
            # Random radius (biased toward max reach)
            # Beta(2,1) biases toward outer shell but still samples interior
            radius = np.random.beta(2, 1) * max_reach
            
            # Generate point
            point = center + direction * radius
            
            # Ensure z > 0.1 (above table)
            if point[2] < 0.1:
                point[2] = 0.1
            
            samples.append(point)
        
        return np.array(samples)
    
    def _generate_grid_samples(self, num_points: int) -> List[np.ndarray]:
        """Generate grid-based workspace samples."""
        # Workspace boundaries
        x_range = [-0.7, 0.7]
        y_range = [-0.7, 0.7]
        z_range = [0.1, 1.0]
        
        # Calculate approximately cubic grid dimensions
        total_volume = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0]) * (z_range[1] - z_range[0])
        point_volume = total_volume / num_points
        point_side = point_volume ** (1/3)
        
        nx = max(2, int((x_range[1] - x_range[0]) / point_side))
        ny = max(2, int((y_range[1] - y_range[0]) / point_side))
        nz = max(2, int((z_range[1] - z_range[0]) / point_side))
        
        # Generate grid
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        z = np.linspace(z_range[0], z_range[1], nz)
        
        points = []
        for xi in x:
            for yi in y:
                for zi in z:
                    # Exclude points outside spherical workspace
                    dist_from_center = np.linalg.norm([xi, yi, zi - 0.5])
                    if dist_from_center <= 0.73:  # Max reach
                        points.append(np.array([xi, yi, zi]))
                        
                        # Stop if we have enough points
                        if len(points) >= num_points:
                            return points
        
        return points
    
    def _generate_c_space_samples(self, num_samples: int) -> np.ndarray:
        """Generate representative C-space samples."""
        lower, upper = self.joint_limits[0], self.joint_limits[1]
        
        # Mix of sampling strategies
        samples = []
        
        # Strategy 1: Pure random sampling (40%)
        n_random = int(num_samples * 0.4)
        for _ in range(n_random):
            q = np.random.uniform(lower, upper, size=self.n_joints)
            samples.append(q)
        
        # Strategy 2: Near-zero configurations (30%)
        n_near_zero = int(num_samples * 0.3)
        for _ in range(n_near_zero):
            # Sample more densely near zero configuration
            q = np.random.normal(0, 0.5, size=self.n_joints)
            q = np.clip(q, lower, upper)
            samples.append(q)
        
        # Strategy 3: Grid-based sampling for first 2 joints (30%)
        n_grid = num_samples - len(samples)
        
        # Create grid for first 2 joints
        resolution = int(np.sqrt(n_grid))
        j1_vals = np.linspace(lower[0], upper[0], resolution)
        j2_vals = np.linspace(lower[1], upper[1], resolution)
        
        for j1 in j1_vals:
            for j2 in j2_vals:
                # Random values for other joints
                q = np.random.uniform(lower, upper, size=self.n_joints)
                q[0] = j1
                q[1] = j2
                samples.append(q)
                
                # Stop if we have enough samples
                if len(samples) >= num_samples:
                    break
        
        return np.array(samples)
    
    def _compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        """Compute the manipulator Jacobian at the given configuration."""
        # This is a simplified numerical Jacobian computation
        # For production, you would use analytical Jacobian from forward kinematics
        delta = 1e-6
        J = np.zeros((6, self.n_joints))
        
        T_base = self.fk.compute_forward_kinematics(q)
        
        for i in range(self.n_joints):
            q_delta = q.copy()
            q_delta[i] += delta
            
            T_delta = self.fk.compute_forward_kinematics(q_delta)
            
            # Position difference
            dp = (T_delta[:3, 3] - T_base[:3, 3]) / delta
            
            # Rotation difference (simplified)
            R_base = T_base[:3, :3]
            R_delta = T_delta[:3, :3]
            R_diff = R_delta @ R_base.T
            
            # Convert to axis-angle
            angle = _safe_acos((np.trace(R_diff) - 1) / 2)
            if abs(angle) < 1e-10:
                axis = np.array([0, 0, 1])
            else:
                axis = np.array([
                    R_diff[2, 1] - R_diff[1, 2],
                    R_diff[0, 2] - R_diff[2, 0],
                    R_diff[1, 0] - R_diff[0, 1]
                ])
                axis = axis / (2 * np.sin(angle))
            
            dr = axis * angle / delta
            
            # Combine into Jacobian column
            J[:3, i] = dr
            J[3:, i] = dp
        
        return J
    
    def _compute_manipulability_measure(self, J: np.ndarray) -> float:
        """
        Compute the manipulability measure from the Jacobian.
        
        Uses the Yoshikawa manipulability measure: sqrt(det(J*J^T))
        """
        try:
            # Use SVD for numerical stability
            u, s, vh = np.linalg.svd(J, full_matrices=False)
            
            # Manipulability is the product of singular values
            manip = np.prod(s)
            
            # Normalize based on theoretical maximum
            max_manip = 1.0  # This should be calibrated for your specific robot
            norm_manip = min(1.0, manip / max_manip)
            
            return norm_manip
            
        except np.linalg.LinAlgError:
            # In case of numerical issues
            return 0.0
    
    def _build_reachability_map(self, ws_points: np.ndarray, 
                              ws_from_cs: np.ndarray, 
                              c_space_points: np.ndarray,
                              manipulability: np.ndarray) -> Dict[str, Any]:
        """
        Build a reachability map linking workspace points to C-space regions.
        
        Args:
            ws_points: Workspace points
            ws_from_cs: Workspace points mapped from C-space samples
            c_space_points: C-space points
            manipulability: Manipulability at each C-space point
            
        Returns:
            Reachability map data structure
        """
        # For each workspace point, find which C-space points can reach it
        reach_map = {}
        
        for i, ws_point in enumerate(ws_points):
            # Find C-space points that can reach this workspace point (within 5cm)
            distances = np.linalg.norm(ws_from_cs - ws_point, axis=1)
            reachable_indices = np.where(distances < REACHABILITY_DISTANCE)[0]
            
            if len(reachable_indices) > 0:
                # Store information about this workspace point
                reach_map[i] = {
                    'ws_point': ws_point,
                    'reachable_indices': reachable_indices,
                    'reachability': len(reachable_indices) / len(c_space_points),
                    'best_manip_index': reachable_indices[np.argmax(manipulability[reachable_indices])],
                    'best_manip': np.max(manipulability[reachable_indices])
                }
        
        # Create C-space regions using clustering
        c_space_regions = self._cluster_c_space_points(c_space_points, ws_from_cs, manipulability)
        
        return {
            'workspace_reach_map': reach_map,
            'c_space_regions': c_space_regions
        }
    
    def _cluster_c_space_points(self, c_space_points: np.ndarray,
                              ws_points: np.ndarray,
                              manipulability: np.ndarray,
                              num_clusters: int = 10) -> List[Dict[str, Any]]:
        """
        Cluster C-space points into regions with similar properties.
        
        This is a simplified clustering based on spatial distribution.
        """
        if len(c_space_points) < num_clusters:
            num_clusters = max(1, len(c_space_points) // 2)
        
        # Simple grid-based clustering
        regions = []
        
        # Divide the first two joints into a grid
        j1_min, j1_max = np.min(c_space_points[:, 0]), np.max(c_space_points[:, 0])
        j2_min, j2_max = np.min(c_space_points[:, 1]), np.max(c_space_points[:, 1])
        
        j1_step = (j1_max - j1_min) / int(np.sqrt(num_clusters))
        j2_step = (j2_max - j2_min) / int(np.sqrt(num_clusters))
        
        # Create grid cells
        for j1_start in np.arange(j1_min, j1_max, j1_step):
            j1_end = j1_start + j1_step
            
            for j2_start in np.arange(j2_min, j2_max, j2_step):
                j2_end = j2_start + j2_step
                
                # Find points in this grid cell
                mask = ((c_space_points[:, 0] >= j1_start) & 
                       (c_space_points[:, 0] < j1_end) &
                       (c_space_points[:, 1] >= j2_start) & 
                       (c_space_points[:, 1] < j2_end))
                
                if np.sum(mask) > 0:
                    # Get points in this region
                    region_points = c_space_points[mask]
                    region_ws_points = ws_points[mask]
                    region_manip = manipulability[mask]
                    
                    # Create region bounds for all joints
                    region_bounds = np.zeros((2, self.n_joints))
                    for j in range(self.n_joints):
                        region_bounds[0, j] = np.min(region_points[:, j])
                        region_bounds[1, j] = np.max(region_points[:, j])
                    
                    regions.append({
                        'bounds': region_bounds,
                        'points': region_points,
                        'ws_points': region_ws_points,
                        'manipulability': region_manip,
                        'avg_manipulability': np.mean(region_manip),
                        'point_count': len(region_points)
                    })
        
        return regions
    
    def _identify_singularity_regions(self, c_space_points: np.ndarray, 
                                    manipulability: np.ndarray,
                                    threshold: float = SINGULARITY_THRESHOLD) -> List[Dict[str, Any]]:
        """
        Identify regions in C-space that are near singularities.
        
        Args:
            c_space_points: C-space points
            manipulability: Manipulability measure at each point
            threshold: Manipulability threshold below which a point is considered singular
            
        Returns:
            List of singularity regions
        """
        # Find points with low manipulability
        singular_indices = np.where(manipulability < threshold)[0]
        
        if len(singular_indices) == 0:
            return []
            
        singular_points = c_space_points[singular_indices]
        
        # Cluster singular points (simple distance-based clustering)
        clusters = []
        assigned = np.zeros(len(singular_points), dtype=bool)
        
        for i, point in enumerate(singular_points):
            if assigned[i]:
                continue
                
            # Create new cluster
            cluster_points = [i]
            assigned[i] = True
            
            # Find nearby points
            for j in range(i+1, len(singular_points)):
                if assigned[j]:
                    continue
                    
                # Check if point j is close to point i
                if np.linalg.norm(singular_points[i] - singular_points[j]) < 0.5:
                    cluster_points.append(j)
                    assigned[j] = True
            
            if len(cluster_points) >= 5:  # Only keep clusters with at least 5 points
                clusters.append({
                    'indices': cluster_points,
                    'center': np.mean(singular_points[cluster_points], axis=0),
                    'bounds': np.array([
                        np.min(singular_points[cluster_points], axis=0),
                        np.max(singular_points[cluster_points], axis=0)
                    ]),
                    'size': len(cluster_points)
                })
        
        # Compute workspace position for each singularity region
        for cluster in clusters:
            # Compute mean workspace position
            center_q = cluster['center']
            T = self.fk.compute_forward_kinematics(center_q)
            cluster['ws_center'] = T[:3, 3]
        
        self.singularity_regions = clusters
        return clusters
    
    def _identify_high_manipulability_regions(self, c_space_points: np.ndarray, 
                                            manipulability: np.ndarray,
                                            threshold: float = HIGH_MANIP_THRESHOLD) -> List[Dict[str, Any]]:
        """
        Identify regions in C-space with high manipulability.
        
        Args:
            c_space_points: C-space points
            manipulability: Manipulability measure at each point
            threshold: Manipulability threshold above which a point is considered high-quality
            
        Returns:
            List of high-manipulability regions
        """
        # Find points with high manipulability
        high_manip_indices = np.where(manipulability > threshold)[0]
        
        if len(high_manip_indices) == 0:
            return []
            
        high_manip_points = c_space_points[high_manip_indices]
        high_manip_values = manipulability[high_manip_indices]
        
        # Cluster high-manipulability points
        clusters = []
        assigned = np.zeros(len(high_manip_points), dtype=bool)
        
        for i, point in enumerate(high_manip_points):
            if assigned[i]:
                continue
                
            # Create new cluster
            cluster_points = [i]
            assigned[i] = True
            
            # Find nearby points
            for j in range(i+1, len(high_manip_points)):
                if assigned[j]:
                    continue
                    
                # Check if point j is close to point i
                if np.linalg.norm(high_manip_points[i] - high_manip_points[j]) < 0.5:
                    cluster_points.append(j)
                    assigned[j] = True
            
            if len(cluster_points) >= 5:  # Only keep clusters with at least 5 points
                clusters.append({
                    'indices': cluster_points,
                    'center': np.mean(high_manip_points[cluster_points], axis=0),
                    'bounds': np.array([
                        np.min(high_manip_points[cluster_points], axis=0),
                        np.max(high_manip_points[cluster_points], axis=0)
                    ]),
                    'avg_manipulability': np.mean(high_manip_values[cluster_points]),
                    'size': len(cluster_points)
                })
        
        # Compute workspace position for each high-manipulability region
        for cluster in clusters:
            # Compute mean workspace position
            center_q = cluster['center']
            T = self.fk.compute_forward_kinematics(center_q)
            cluster['ws_center'] = T[:3, 3]
        
        self.high_manipulability_regions = clusters
        return clusters
    
    def _build_roadmap(self, points: np.ndarray, k: int, max_distance: float) -> List[List[Tuple[int, float]]]:
        """
        Build a probabilistic roadmap from sample points.
        
        Args:
            points: C-space sample points
            k: Number of neighbors to connect per node
            max_distance: Maximum edge length
            
        Returns:
            List of adjacency lists, where each entry is a list of (neighbor_idx, distance) tuples
        """
        n = len(points)
        roadmap = [[] for _ in range(n)]
        
        # Build KD-tree for efficient nearest neighbor search
        try:
            from scipy.spatial import cKDTree
            kdtree = cKDTree(points)
            
            # Connect each point to its k nearest neighbors
            for i, point in enumerate(points):
                # Find k+1 nearest neighbors (including self)
                distances, indices = kdtree.query(point, k=k+1)
                
                # Skip the first index (self)
                for j, idx in enumerate(indices[1:], 1):
                    if distances[j] > max_distance:
                        continue
                        
                    # Check if path is collision-free
                    if self.collision_checker:
                        path_valid = self._check_edge_validity(point, points[idx])
                        if not path_valid:
                            continue
                    
                    # Add edge to roadmap (bidirectional)
                    roadmap[i].append((idx, distances[j]))
                    roadmap[idx].append((i, distances[j]))
            
        except ImportError:
            # Fallback if SciPy is not available
            logger.warning("SciPy not available, using slower roadmap construction")
            
            # Connect each point to its nearest neighbors
            for i, point1 in enumerate(points):
                # Compute distances to all other points
                distances = []
                for j, point2 in enumerate(points):
                    if i != j:
                        dist = np.linalg.norm(point1 - point2)
                        distances.append((j, dist))
                
                # Sort by distance
                distances.sort(key=lambda x: x[1])
                
                # Connect to k nearest neighbors
                for j, dist in distances[:k]:
                    if dist > max_distance:
                        continue
                        
                    # Check if path is collision-free
                    if self.collision_checker:
                        path_valid = self._check_edge_validity(point1, points[j])
                        if not path_valid:
                            continue
                    
                    # Add edge to roadmap (bidirectional)
                    roadmap[i].append((j, dist))
                    roadmap[j].append((i, dist))
        
        return roadmap
    
    def _check_edge_validity(self, q1: np.ndarray, q2: np.ndarray, 
                           num_checks: int = 10) -> bool:
        """
        Check if a straight-line edge in C-space is collision-free.
        
        Args:
            q1, q2: Endpoints of the edge
            num_checks: Number of intermediate points to check
            
        Returns:
            True if edge is valid, False otherwise
        """
        if self.collision_checker is None:
            return True
            
        for t in np.linspace(0, 1, num_checks):
            q_interp = q1 + t * (q2 - q1)
            if not self.collision_checker.is_valid_joint_position(q_interp):
                return False
        
        return True
    
    def _find_path_in_roadmap(self, start_q: np.ndarray, 
                             goal_q: np.ndarray,
                             max_nearest: int = 5) -> Optional[List[np.ndarray]]:
        """
        Find a path from start to goal using the pre-computed roadmap.
        
        Uses A* search with Euclidean distance heuristic.
        
        Args:
            start_q: Start configuration
            goal_q: Goal configuration
            max_nearest: Maximum nearest neighbors to consider
            
        Returns:
            List of configurations forming a path, or None if not found
        """
        if self.roadmap is None:
            return None
            
        roadmap_nodes = self.roadmap['nodes']
        roadmap_edges = self.roadmap['edges']
        
        # Find nearest nodes to start and goal
        start_dists = np.linalg.norm(roadmap_nodes - start_q, axis=1)
        goal_dists = np.linalg.norm(roadmap_nodes - goal_q, axis=1)
        
        start_nearest = np.argsort(start_dists)[:max_nearest]
        goal_nearest = np.argsort(goal_dists)[:max_nearest]
        
        # Check direct connection from start to nearest nodes
        valid_start_indices = []
        for idx in start_nearest:
            if self.collision_checker is None or self._check_edge_validity(start_q, roadmap_nodes[idx]):
                valid_start_indices.append(idx)
        
        if not valid_start_indices:
            logger.warning("No valid connection from start to roadmap")
            return None
            
        # Check direct connection from nearest nodes to goal
        valid_goal_indices = []
        for idx in goal_nearest:
            if self.collision_checker is None or self._check_edge_validity(roadmap_nodes[idx], goal_q):
                valid_goal_indices.append(idx)
        
        if not valid_goal_indices:
            logger.warning("No valid connection from roadmap to goal")
            return None
        
        # Find paths from each valid start to each valid goal
        best_path = None
        best_cost = float('inf')
        
        for start_idx in valid_start_indices:
            for goal_idx in valid_goal_indices:
                # Use A* to find path
                path_indices = self._astar_search(roadmap_edges, start_idx, goal_idx, roadmap_nodes, goal_q)
                
                if path_indices:
                    # Compute path cost
                    cost = start_dists[start_idx] + goal_dists[goal_idx]
                    for i in range(len(path_indices) - 1):
                        cost += np.linalg.norm(roadmap_nodes[path_indices[i]] - 
                                              roadmap_nodes[path_indices[i+1]])
                    
                    if cost < best_cost:
                        best_cost = cost
                        # Convert indices to configurations
                        path = [start_q] + [roadmap_nodes[idx] for idx in path_indices] + [goal_q]
                        best_path = path
        
        return best_path
    
    def _astar_search(self, roadmap: List[List[Tuple[int, float]]], 
                     start_idx: int, goal_idx: int, 
                     nodes: np.ndarray, goal_q: np.ndarray) -> Optional[List[int]]:
        """
        A* search algorithm for finding path in roadmap.
        
        Args:
            roadmap: List of adjacency lists, where each entry is a list of (neighbor_idx, distance) tuples
            start_idx: Index of start node in roadmap
            goal_idx: Index of goal node in roadmap
            nodes: Array of configuration points
            goal_q: Goal configuration
            
        Returns:
            List of node indices forming a path, or None if not found
        """
        import heapq
        
        # Heuristic function (Euclidean distance to goal)
        def heuristic(idx):
            return np.linalg.norm(nodes[idx] - nodes[goal_idx])
        
        # Initialize
        open_set = [(0, start_idx)]  # (f_score, node_idx)
        came_from = {}
        g_score = {start_idx: 0}  # Cost from start
        f_score = {start_idx: heuristic(start_idx)}  # Estimated total cost
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal_idx:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_idx)
                path.reverse()
                return path
            
            for neighbor, distance in roadmap[current]:
                tentative_g = g_score[current] + distance
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This path to neighbor is better than previous
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor)
                    
                    # Add to open set if not already there
                    for _, idx in open_set:
                        if idx == neighbor:
                            break
                    else:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # No path found
    
    def _compute_singularity_free_path(self, start_q: np.ndarray, 
                                      goal_q: np.ndarray, 
                                      discretization: int = 20) -> List[np.ndarray]:
        """
        Compute a path from start to goal that avoids singularities.
        
        Uses a potential field approach where singularities create repulsive forces.
        
        Args:
            start_q: Start configuration
            goal_q: Goal configuration
            discretization: Number of waypoints for path discretization
            
        Returns:
            List of configurations forming a path
        """
        if not self.singularity_regions:
            # No singularities, return straight-line path
            return [start_q + t * (goal_q - start_q) for t in np.linspace(0, 1, discretization)]
        
        # Parameters
        max_iterations = SINGULARITY_PATH_MAX_ITERS
        step_size = SINGULARITY_PATH_STEP_SIZE
        singularity_repulsion = 1.0
        goal_attraction = 0.5
        
        # Initialize path with straight line
        path = [start_q + t * (goal_q - start_q) for t in np.linspace(0, 1, discretization)]
        
        # Iteratively improve path
        for iteration in range(max_iterations):
            path_changed = False
            
            # For each waypoint (except start and goal)
            for i in range(1, discretization - 1):
                q = path[i]
                
                # Compute forces
                force = np.zeros(self.n_joints)
                
                # Attraction to goal
                force += goal_attraction * (goal_q - q)
                
                # Repulsion from singularities
                for region in self.singularity_regions:
                    center = region['center']
                    bounds = region['bounds']
                    
                    # Check if q is near this singularity
                    distance = np.linalg.norm(q - center)
                    
                    if distance < SINGULARITY_REPULSION_RADIUS:  # Only consider nearby singularities
                        # Compute repulsion vector (away from singularity)
                        repulsion = q - center
                        
                        # Normalize and scale by distance
                        if np.linalg.norm(repulsion) > 0:
                            repulsion = repulsion / np.linalg.norm(repulsion)
                            
                        # Stronger repulsion when closer
                        scale = singularity_repulsion / (distance + 0.1) ** 2
                        force += scale * repulsion
                
                # Apply force to update waypoint
                if np.linalg.norm(force) > 0.01:
                    # Normalize force and apply step
                    force = force / np.linalg.norm(force) * step_size
                    new_q = q + force
                    
                    # Enforce joint limits
                    new_q = np.clip(new_q, self.joint_limits[0], self.joint_limits[1])
                    
                    # Check if configuration is valid
                    if self.collision_checker is None or self.collision_checker.is_valid_joint_position(new_q):
                        path[i] = new_q
                        path_changed = True
            
            # Stop if path no longer changes
            if not path_changed:
                break
        
        return path
    
    def _save_reachability_map(self, save_path: str):
        """Save reachability map to disk."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save using pickle
            with open(save_path, 'wb') as f:
                # Don't save huge arrays
                # Persist full arrays required for IK seeding. For large robots
                # this could be memory-heavy; compression / down-sampling could
                # be added later.
                save_data = {
                    'workspace_points': self.reachability_map['workspace_points'],
                    'c_space_points': self.reachability_map['c_space_points'],
                    'workspace_from_cspace': self.reachability_map['workspace_from_cspace'],
                    'manipulability': self.reachability_map['manipulability'],
                    'singularity_regions': self.reachability_map['singularity_regions'],
                    'high_manipulability_regions': self.reachability_map['high_manipulability_regions'],
                    'map_data': self.reachability_map['map_data'],
                    'generation_time': self.reachability_map['generation_time']
                }
                pickle.dump(save_data, f)
                
            logger.info(f"Reachability map saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save reachability map: {e}")

    def _load_reachability_map(self, path: str) -> bool:
        """Load a previously saved (compressed) reachability map.

        Note: For memory reasons only summary data is stored; detailed arrays
        that would require recomputation (like full c_space_points) are not
        persisted. This function reconstructs a minimal structure so that
        seed selection still works; if full data is needed it must be rebuilt.
        """
        try:
            with open(path, 'rb') as f:
                save_data = pickle.load(f)
            required_keys = {'workspace_points','c_space_points','workspace_from_cspace','manipulability','map_data'}
            if not required_keys.issubset(save_data.keys()):
                logger.warning("Saved reachability map missing required arrays; rebuilding will be needed.")
                return False
            self.reachability_map = save_data
            logger.info("Reachability map loaded successfully.")
            return True
        except Exception as e:
            logger.warning(f"Could not load reachability map from {path}: {e}")
            return False

    # ------------------------------------------------------------------
    # Public API wrappers
    # ------------------------------------------------------------------
    def load_reachability_map(self, path: str) -> bool:
        """Public wrapper for loading a reachability map.

        Args:
            path: File path to previously saved reachability map pickle.

        Returns:
            True if load succeeded, else False.
        """
        return self._load_reachability_map(path)


# Demo function to show how to use the analyzer
def demo_configuration_space_analysis(fk, ik, save_path=None):
    """
    Demonstrate configuration space analysis with the given kinematics modules.
    
    Args:
        fk: Forward kinematics instance
        ik: Inverse kinematics instance
        save_path: Path to save results (optional)
    """
    # Create analyzer
    analyzer = ConfigurationSpaceAnalyzer(fk, ik)
    
    # Build reachability map (this can take some time)
    reachability_map = analyzer.build_reachability_map(
        workspace_samples=500,
        c_space_samples=2000,
        save_path=os.path.join(save_path, 'reachability_map.pkl') if save_path else None
    )
    
    # Build probabilistic roadmap
    roadmap = analyzer.build_probabilistic_roadmap(
        num_samples=1000,
        k_neighbors=10,
        save_path=os.path.join(save_path, 'roadmap.pkl') if save_path else None
    )
    
    # Visualize results if matplotlib is available
    if MATPLOTLIB_AVAILABLE:
        # Workspace reachability visualization
        fig1 = analyzer.visualize_workspace_reachability()
        if save_path and fig1:
            fig1.savefig(os.path.join(save_path, 'workspace_reachability.png'), dpi=300, bbox_inches='tight')
        
        # C-space regions visualization
        fig2 = analyzer.visualize_cspace_regions()
        if save_path and fig2:
            fig2.savefig(os.path.join(save_path, 'cspace_regions.png'), dpi=300, bbox_inches='tight')
    
    print("\nConfiguration Space Analysis Results:")
    print("-" * 50)
    print(f"Workspace points analyzed: {len(reachability_map['workspace_points'])}")
    print(f"C-space configurations analyzed: {len(reachability_map['c_space_points'])}")
    print(f"Singularity regions identified: {len(analyzer.singularity_regions)}")
    print(f"High-manipulability regions identified: {len(analyzer.high_manipulability_regions)}")
    print(f"Probabilistic roadmap nodes: {len(roadmap['nodes'])}")
    print(f"Probabilistic roadmap edges: {sum(len(edges) for edges in roadmap['edges'])}")
    print("-" * 50)
    
    return analyzer
