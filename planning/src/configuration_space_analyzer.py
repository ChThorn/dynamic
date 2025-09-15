#!/usr/bin/env python3
"""
Configuration Space Analysis Module - Production Optimized

Minimal configuration space analysis for motion planning optimization.
Contains only the essential methods used by the motion planner:
- Reachability map building for IK seed optimization
- Best IK region identification for improved convergence

Removed unused features:
- Visualization methods (176 lines removed)
- Probabilistic roadmap generation
- Singularity analysis
- Path planning in C-space
- Complex clustering algorithms

Author: Robot Control Team
"""

import numpy as np
import logging
import pickle
import os
import threading
import time
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Production constants
REACHABILITY_DISTANCE = 0.05  # meters - FK sample within 5cm of workspace point
IK_SEED_NEIGHBOR_COUNT = 5    # Number of neighboring configurations to consider

@dataclass
class CSpaceRegion:
    """Represents a region in configuration space with reachability information."""
    bounds: np.ndarray  # Shape: (2, n_joints) - min/max bounds for each joint
    reachability: float  # 0-1, higher means more reachable workspace positions
    workspace_samples: List[np.ndarray]  # Workspace positions reachable from this region

class ConfigurationSpaceAnalyzer:
    """
    Enhanced configuration space analyzer for motion planning optimization.
    
    Provides C-space analysis to improve IK convergence by:
    1. Building reachability maps that associate workspace regions with good joint configurations
    2. Providing optimal IK seeds based on target workspace positions
    3. Background processing and caching for performance
    4. Progress feedback for long operations
    """
    
    def __init__(self, forward_kinematics, inverse_kinematics, cache_dir='./cache'):
        """
        Initialize configuration space analyzer.
        
        Args:
            forward_kinematics: ForwardKinematics instance
            inverse_kinematics: InverseKinematics instance
            cache_dir: Directory for caching reachability maps
        """
        self.fk = forward_kinematics
        self.ik = inverse_kinematics
        self.n_joints = forward_kinematics.n_joints
        self.cache_dir = cache_dir
        
        # Get joint limits - ForwardKinematics returns numpy array (2, n_joints)
        self.joint_limits = forward_kinematics.get_joint_limits()
        
        # Reachability map: maps workspace positions to good joint configurations
        self.reachability_map = {}
        
        # Background processing
        self._build_thread = None
        self._build_progress = 0.0
        self._build_status = "idle"
        self._progress_callback = None
        
        logger.info("Enhanced configuration space analyzer initialized")
    
    def build_reachability_map(self, workspace_samples=500, c_space_samples=2000, save_path=None):
        """
        Build reachability map for IK seed optimization with caching support.
        
        Creates a mapping from workspace positions to joint configurations that
        can reach those positions. This improves IK convergence by providing
        good initial seeds.
        
        Args:
            workspace_samples: Number of workspace positions to sample
            c_space_samples: Number of joint configurations to sample
            save_path: Optional path to save the reachability map
        """
        # Check for cached reachability map first
        cache_name = f"reachability_map_{workspace_samples}_{c_space_samples}.pkl"
        cache_path = os.path.join(self.cache_dir, cache_name)
        
        if os.path.exists(cache_path):
            logger.info(f"Loading cached reachability map from {cache_path}")
            if self.load_reachability_map(cache_path):
                logger.info("Cached reachability map loaded successfully")
                return
        
        logger.info(f"Building reachability map: {workspace_samples} workspace, {c_space_samples} C-space samples")
        self._build_status = "building"
        self._build_progress = 0.0
        
        # Generate workspace samples (target positions)
        self._update_progress(10.0, "Generating workspace samples")
        workspace_points = self._generate_workspace_samples(workspace_samples)
        
        # Generate C-space samples (joint configurations)  
        self._update_progress(20.0, "Generating C-space samples")
        c_space_points = self._generate_c_space_samples(c_space_samples)
        
        # Build the reachability mapping
        self._update_progress(30.0, "Building reachability mapping")
        self.reachability_map = {}
        
        for i, q_sample in enumerate(c_space_points):
            progress = 30.0 + (i / len(c_space_points)) * 60.0
            if i % 100 == 0:
                self._update_progress(progress, f"Processing C-space samples ({100*i//len(c_space_points)}%)")
            
            try:
                # Compute forward kinematics for this configuration
                T = self.fk.compute_forward_kinematics(q_sample)
                tcp_position = T[:3, 3]
                
                # Find nearby workspace points
                for ws_point in workspace_points:
                    distance = np.linalg.norm(tcp_position - ws_point)
                    
                    if distance <= REACHABILITY_DISTANCE:
                        # Create a key for this workspace region
                        ws_key = tuple(np.round(ws_point, 2))  # Round to 1cm precision
                        
                        if ws_key not in self.reachability_map:
                            self.reachability_map[ws_key] = []
                        
                        self.reachability_map[ws_key].append({
                            'joint_config': q_sample.copy(),
                            'tcp_position': tcp_position.copy(),
                            'distance': distance
                        })
                        
            except Exception as e:
                logger.debug(f"FK computation failed for sample {i}: {e}")
                continue
        
        # Sort configurations by distance for each workspace point
        self._update_progress(90.0, "Optimizing reachability data")
        for ws_key in self.reachability_map:
            self.reachability_map[ws_key].sort(key=lambda x: x['distance'])
            # Keep only the best configurations to save memory
            self.reachability_map[ws_key] = self.reachability_map[ws_key][:IK_SEED_NEIGHBOR_COUNT]
        
        self._update_progress(100.0, "Complete")
        self._build_status = "complete"
        
        logger.info(f"Reachability map built: {len(self.reachability_map)} workspace regions mapped")
        
        # Save if requested or to cache
        save_path = save_path or cache_path
        if save_path:
            self._save_reachability_map(save_path)
    
    def get_best_ik_region(self, target_position: np.ndarray) -> Optional[np.ndarray]:
        """
        Get the best joint configuration seed for IK at target position.
        
        Args:
            target_position: 3D target position in workspace
            
        Returns:
            Joint configuration that's likely to converge for this target, or None
        """
        if not self.reachability_map:
            logger.warning("Reachability map not built - cannot provide IK seed")
            return None
        
        # Find the closest workspace region
        target_key = tuple(np.round(target_position, 2))
        
        # Direct lookup first
        if target_key in self.reachability_map:
            best_config = self.reachability_map[target_key][0]
            return best_config['joint_config']
        
        # Find nearest region if direct lookup fails
        min_distance = float('inf')
        best_region = None
        
        for ws_key, configs in self.reachability_map.items():
            ws_position = np.array(ws_key)
            distance = np.linalg.norm(target_position - ws_position)
            
            if distance < min_distance:
                min_distance = distance
                best_region = configs
        
        if best_region and min_distance < 0.1:  # Within 10cm
            return best_region[0]['joint_config']
        
        logger.debug(f"No good IK seed found for position {target_position}")
        return None
    
    def _generate_workspace_samples(self, num_samples: int) -> List[np.ndarray]:
        """Generate sample points in the robot's workspace."""
        # Define workspace bounds based on typical robot reach
        # These should be adjusted based on the specific robot
        x_range = (-0.8, 0.8)  # meters
        y_range = (-0.8, 0.8)  # meters  
        z_range = (0.1, 1.2)   # meters (above table surface)
        
        samples = []
        for _ in range(num_samples):
            x = np.random.uniform(*x_range)
            y = np.random.uniform(*y_range)
            z = np.random.uniform(*z_range)
            samples.append(np.array([x, y, z]))
        
        return samples
    
    def _generate_c_space_samples(self, num_samples: int) -> List[np.ndarray]:
        """Generate sample joint configurations within joint limits."""
        samples = []
        
        for _ in range(num_samples):
            q_sample = np.random.uniform(
                self.joint_limits[0],  # min limits
                self.joint_limits[1]   # max limits
            )
            samples.append(q_sample)
        
        return samples
    
    def _save_reachability_map(self, save_path: str):
        """Save reachability map to file."""
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(self.reachability_map, f)
            logger.info(f"Reachability map saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save reachability map: {e}")
    
    def load_reachability_map(self, load_path: str) -> bool:
        """Load reachability map from file."""
        try:
            with open(load_path, 'rb') as f:
                self.reachability_map = pickle.load(f)
            logger.info(f"Reachability map loaded from {load_path}: {len(self.reachability_map)} regions")
            return True
        except Exception as e:
            logger.error(f"Failed to load reachability map: {e}")
            return False
    
    def build_reachability_map_async(self, workspace_samples=500, c_space_samples=2000, cache_name='reachability_map') -> bool:
        """Build reachability map in background thread."""
        if self._build_thread and self._build_thread.is_alive():
            logger.warning("Reachability map building already in progress")
            return False
        
        cache_path = os.path.join(self.cache_dir, f"{cache_name}.pkl")
        
        def build_worker():
            self.build_reachability_map(workspace_samples, c_space_samples, cache_path)
        
        self._build_thread = threading.Thread(target=build_worker)
        self._build_thread.start()
        logger.info("Started background reachability map building")
        return True
    
    def get_build_progress(self) -> Tuple[float, str]:
        """Get current build progress."""
        return self._build_progress, self._build_status
    
    def is_map_ready(self) -> bool:
        """Check if reachability map is ready for use."""
        return len(self.reachability_map) > 0 and self._build_status != "building"
    
    def wait_for_completion(self, timeout: float = 60.0) -> bool:
        """Wait for background building to complete."""
        if self._build_thread:
            self._build_thread.join(timeout)
            return not self._build_thread.is_alive()
        return True
    
    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """Set callback for progress updates."""
        self._progress_callback = callback
    
    def _update_progress(self, progress: float, status: str):
        """Update build progress and notify callback."""
        self._build_progress = progress
        self._build_status = status
        if self._progress_callback:
            self._progress_callback(progress, status)
    
    @staticmethod
    def print_progress_callback(progress: float, status: str):
        """Default progress callback that prints to console."""
        bar_length = 30
        filled = int(bar_length * progress / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\r🔄 Building C-space map: |{bar}| {progress:.1f}% ({status})", end='', flush=True)
        if progress >= 100:
            print()  # New line when complete

