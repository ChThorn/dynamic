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

# Production constants - ENHANCED for better coverage
REACHABILITY_DISTANCE = 0.08  # meters - Increased from 5cm to 8cm for better coverage
IK_SEED_NEIGHBOR_COUNT = 8    # Increased from 5 to 8 neighboring configurations
WORKSPACE_GRID_RESOLUTION = 0.05  # meters - Grid resolution for workspace sampling
MIN_SAMPLES_PER_REGION = 3   # Minimum samples required to create a region

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
        
        # Build the reachability mapping - ENHANCED for better coverage
        self._update_progress(30.0, "Building reachability mapping")
        self.reachability_map = {}
        workspace_coverage = {}  # Track coverage statistics
        
        for i, q_sample in enumerate(c_space_points):
            progress = 30.0 + (i / len(c_space_points)) * 60.0
            if i % 50 == 0:  # More frequent updates (was 100)
                self._update_progress(progress, f"Processing C-space samples ({100*i//len(c_space_points)}%)")
            
            try:
                # Compute forward kinematics for this configuration
                T = self.fk.compute_forward_kinematics(q_sample)
                tcp_position = T[:3, 3]
                
                # Find nearby workspace points using improved clustering
                for ws_point in workspace_points:
                    distance = np.linalg.norm(tcp_position - ws_point)
                    
                    if distance <= REACHABILITY_DISTANCE:
                        # Create a key for this workspace region with finer resolution
                        ws_key = tuple(np.round(ws_point / WORKSPACE_GRID_RESOLUTION) * WORKSPACE_GRID_RESOLUTION)
                        
                        if ws_key not in self.reachability_map:
                            self.reachability_map[ws_key] = []
                            workspace_coverage[ws_key] = 0
                        
                        # Add configuration with quality metrics
                        config_quality = self._evaluate_configuration_quality(q_sample, tcp_position, ws_point)
                        
                        self.reachability_map[ws_key].append({
                            'joint_config': q_sample.copy(),
                            'tcp_position': tcp_position.copy(),
                            'distance': distance,
                            'quality': config_quality  # NEW: Quality metric
                        })
                        workspace_coverage[ws_key] += 1
                        
            except Exception as e:
                logger.debug(f"FK computation failed for sample {i}: {e}")
                continue
        
        # Filter regions with insufficient samples and sort by quality
        self._update_progress(90.0, "Optimizing reachability data")
        filtered_map = {}
        
        for ws_key in self.reachability_map:
            configs = self.reachability_map[ws_key]
            
            # Only keep regions with sufficient samples
            if len(configs) >= MIN_SAMPLES_PER_REGION:
                # Sort by quality (higher is better) then by distance (lower is better)
                configs.sort(key=lambda x: (-x['quality'], x['distance']))
                # Keep only the best configurations to save memory
                filtered_map[ws_key] = configs[:IK_SEED_NEIGHBOR_COUNT]
        
        self.reachability_map = filtered_map
        
        # Log coverage statistics
        total_regions = len(self.reachability_map)
        if total_regions > 0:
            avg_configs_per_region = np.mean([len(configs) for configs in self.reachability_map.values()])
            logger.info(f"Reachability map coverage: {total_regions} regions, "
                       f"avg {avg_configs_per_region:.1f} configs/region")
        
        self._update_progress(100.0, "Complete")
        self._build_status = "complete"
        
        logger.info(f"Reachability map built: {len(self.reachability_map)} workspace regions mapped")
        
        # Save if requested or to cache
        save_path = save_path or cache_path
        if save_path:
            self._save_reachability_map(save_path)
    
    def _evaluate_configuration_quality(self, joint_config: np.ndarray, tcp_position: np.ndarray, 
                                       target_position: np.ndarray) -> float:
        """Evaluate the quality of a joint configuration for IK seeding.
        
        Higher scores indicate better configurations for IK convergence.
        """
        quality = 1.0
        
        # 1. Distance to target (closer is better)
        distance = np.linalg.norm(tcp_position - target_position)
        distance_score = max(0.0, 1.0 - distance / REACHABILITY_DISTANCE)
        quality *= distance_score
        
        # 2. Configuration singularity avoidance
        # Avoid configurations near joint limits
        joint_ranges = self.joint_limits[1] - self.joint_limits[0]  # max - min
        joint_centers = (self.joint_limits[1] + self.joint_limits[0]) / 2  # center positions
        
        # Penalty for joints near limits
        for i, (q, center, range_val) in enumerate(zip(joint_config, joint_centers, joint_ranges)):
            normalized_pos = abs(q - center) / (range_val / 2)  # 0 (center) to 1 (limit)
            if normalized_pos > 0.8:  # Near joint limit
                quality *= (1.0 - (normalized_pos - 0.8) / 0.2 * 0.5)  # Reduce quality by up to 50%
        
        # 3. Manipulability measure (avoid singular configurations)
        # Simple heuristic: penalize extreme joint angles
        extreme_penalty = 0.0
        for q in joint_config:
            if abs(q) > np.pi * 0.8:  # Beyond 80% of ±π
                extreme_penalty += 0.1
        quality *= max(0.1, 1.0 - extreme_penalty)
        
        # 4. Configuration complexity (simpler is better for IK convergence)
        complexity = np.linalg.norm(joint_config) / np.sqrt(len(joint_config))
        complexity_score = max(0.2, 1.0 - complexity / np.pi)  # Normalize by π
        quality *= complexity_score
        
        return max(0.01, quality)  # Ensure minimum quality
    
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
        """Generate sample points in the robot's workspace with improved distribution."""
        # Define workspace bounds based on RB3-730ES-U specifications
        x_range = (-0.7, 0.7)   # meters (conservative for 730mm reach)
        y_range = (-0.7, 0.7)   # meters  
        z_range = (0.1, 1.0)    # meters (above table surface, realistic height)
        
        samples = []
        
        # Generate samples using mixed strategy for better coverage
        uniform_samples = int(num_samples * 0.6)  # 60% uniform distribution
        focused_samples = int(num_samples * 0.4)  # 40% focused on common work areas
        
        # 1. Uniform random sampling
        for _ in range(uniform_samples):
            x = np.random.uniform(*x_range)
            y = np.random.uniform(*y_range)
            z = np.random.uniform(*z_range)
            samples.append(np.array([x, y, z]))
        
        # 2. Focused sampling on typical work areas
        work_zones = [
            # Front center zone (most common)
            {'center': [0.4, 0.0, 0.5], 'radius': 0.15, 'weight': 0.4},
            # Right side zone
            {'center': [0.2, 0.3, 0.4], 'radius': 0.12, 'weight': 0.25},
            # Left side zone  
            {'center': [0.2, -0.3, 0.4], 'radius': 0.12, 'weight': 0.25},
            # High reach zone
            {'center': [0.3, 0.0, 0.8], 'radius': 0.1, 'weight': 0.1}
        ]
        
        for _ in range(focused_samples):
            # Select zone based on weights
            zone_weights = [z['weight'] for z in work_zones]
            zone_idx = np.random.choice(len(work_zones), p=zone_weights)
            zone = work_zones[zone_idx]
            
            # Sample within the zone
            center = np.array(zone['center'])
            radius = zone['radius']
            
            # Random direction and distance
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            distance = np.random.uniform(0, radius)
            
            sample = center + direction * distance
            
            # Clamp to workspace bounds
            sample[0] = np.clip(sample[0], *x_range)
            sample[1] = np.clip(sample[1], *y_range)
            sample[2] = np.clip(sample[2], *z_range)
            
            samples.append(sample)
        
        return samples
    
    def _generate_c_space_samples(self, num_samples: int) -> List[np.ndarray]:
        """Generate sample joint configurations within joint limits with improved distribution."""
        samples = []
        
        # Generate samples using mixed strategy
        uniform_samples = int(num_samples * 0.5)  # 50% uniform
        focused_samples = int(num_samples * 0.3)  # 30% near home/neutral poses
        boundary_samples = int(num_samples * 0.2) # 20% near workspace boundaries
        
        # 1. Uniform random sampling
        for _ in range(uniform_samples):
            q_sample = np.random.uniform(
                self.joint_limits[0],  # min limits
                self.joint_limits[1]   # max limits
            )
            samples.append(q_sample)
        
        # 2. Focused sampling near neutral/home configurations
        neutral_configs = [
            np.zeros(self.n_joints),  # Home position
            np.array([0, -0.5, 0.5, 0, 0, 0]),  # Neutral working pose
            np.array([0, -0.8, 1.2, 0, 0.6, 0])  # Extended reach pose
        ]
        
        for _ in range(focused_samples):
            # Choose a neutral config as base
            base_config = neutral_configs[np.random.randint(len(neutral_configs))]
            
            # Add small random perturbation
            noise_scale = 0.3  # radians
            noise = np.random.normal(0, noise_scale, self.n_joints)
            q_sample = base_config + noise
            
            # Clamp to joint limits
            q_sample = np.clip(q_sample, self.joint_limits[0], self.joint_limits[1])
            samples.append(q_sample)
        
        # 3. Boundary sampling for edge case coverage
        for _ in range(boundary_samples):
            q_sample = np.random.uniform(
                self.joint_limits[0],
                self.joint_limits[1]
            )
            
            # Push some joints towards their limits
            for i in range(self.n_joints):
                if np.random.random() < 0.3:  # 30% chance per joint
                    if np.random.random() < 0.5:
                        # Push towards upper limit
                        q_sample[i] = np.random.uniform(
                            self.joint_limits[1][i] * 0.7,
                            self.joint_limits[1][i] * 0.95
                        )
                    else:
                        # Push towards lower limit
                        q_sample[i] = np.random.uniform(
                            self.joint_limits[0][i] * 0.95,
                            self.joint_limits[0][i] * 0.7
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

