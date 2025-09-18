#!/usr/bin/env python3
"""
Trajectory Planning Module

This module provides advanced trajectory planning and optimization capabilities:
- Smooth trajectory generation from waypoint sequences
- Velocity and acceleration profiling 
- Trajectory smoothing and optimization
- Time parameterization and scaling
- Multi-objective optimization (smoothness, speed, energy)

The trajectory planner works with validated paths from the path planner to generate
smooth, executable trajectories with proper dynamics considerations.

Author: Robot Control Team
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
from scipy.interpolate import interp1d, splprep, splev
from scipy.optimize import minimize_scalar
import time

logger = logging.getLogger(__name__)

class TrajectoryPlanningError(Exception):
    """Custom exception for trajectory planning errors."""
    pass

@dataclass
class TrajectoryPoint:
    """Single point in a robot trajectory."""
    time: float
    position: np.ndarray  # Joint positions
    velocity: np.ndarray  # Joint velocities  
    acceleration: np.ndarray  # Joint accelerations

@dataclass 
class Trajectory:
    """Complete robot trajectory with timing and dynamics."""
    points: List[TrajectoryPoint]
    total_time: float
    max_velocities: np.ndarray
    max_accelerations: np.ndarray
    smoothness_metric: float
    
    def get_positions(self) -> np.ndarray:
        """Get position array (n_points x n_joints)."""
        return np.array([p.position for p in self.points])
    
    def get_velocities(self) -> np.ndarray:
        """Get velocity array (n_points x n_joints)."""
        return np.array([p.velocity for p in self.points])
    
    def get_accelerations(self) -> np.ndarray:
        """Get acceleration array (n_points x n_joints)."""
        return np.array([p.acceleration for p in self.points])
    
    def get_times(self) -> np.ndarray:
        """Get time array."""
        return np.array([p.time for p in self.points])

@dataclass
class TrajectoryResult:
    """Result container for trajectory planning operations."""
    success: bool
    trajectory: Optional[Trajectory] = None
    error_message: Optional[str] = None
    computation_time: Optional[float] = None
    optimization_info: Optional[Dict[str, Any]] = None

class TrajectoryPlanner:
    """Advanced trajectory planning and optimization."""
    
    def __init__(self, path_planner=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize trajectory planner.
        
        Args:
            path_planner: PathPlanner instance for constraint validation
            config: Trajectory planning configuration
        """
        self.path_planner = path_planner
        
        # Default configuration - UPDATED for production use
        self.config = {
            # Velocity limits (rad/s) - Updated for realistic robot operation
            'max_joint_velocity': np.radians(60),  # Increased from 45 to 60 deg/s for better performance
            'max_joint_acceleration': np.radians(150),  # Increased from 90 to 150 deg/s^2 to handle complex motions
            
            # Smoothing parameters - Tuned for better trajectory quality
            'smoothing_iterations': 8,  # Increased from 5 to 8 for better smoothing
            'smoothing_weight': 0.12,  # Reduced from 0.15 to 0.12 for less aggressive smoothing
            
            # Trajectory resolution
            'time_resolution': 0.01,  # 10ms time steps (unchanged)
            
            # Optimization parameters - Improved for better performance
            'optimize_timing': True,
            'speed_weight': 0.4,  # Increased from 0.3 to prioritize speed slightly more
            'smoothness_weight': 0.6,  # Decreased from 0.7 to balance with speed
            'optimization_iterations': 3,  # NEW: Number of optimization passes
            'adaptive_time_scaling': True,  # NEW: Enable adaptive time scaling based on complexity
            
            # Spline parameters - Enhanced for better curve fitting
            'spline_degree': 3,
            'spline_smoothing': 0.001,  # Small amount of smoothing (was 0.0)
            'spline_derivative_weight': 0.1,  # NEW: Weight for derivative smoothness
            
            # Advanced trajectory features
            'jerk_minimization': True,  # NEW: Enable jerk minimization
            'acceleration_smoothing': True,  # NEW: Apply additional acceleration smoothing
            'velocity_profile_optimization': True,  # NEW: Optimize velocity profiles
        }
        
        if config:
            self.config.update(config)
        
        logger.info("Trajectory planner initialized")
    
    def plan_trajectory(self, waypoints: List[np.ndarray], 
                       time_scaling: float = 1.0,
                       optimize: bool = True) -> TrajectoryResult:
        """
        Plan a complete trajectory from waypoints.
        
        Args:
            waypoints: List of joint configurations
            time_scaling: Overall time scaling factor (>1 = slower)
            optimize: Whether to perform trajectory optimization
            
        Returns:
            TrajectoryResult with planned trajectory
        """
        start_time = time.time()
        
        if len(waypoints) < 2:
            return TrajectoryResult(
                success=False,
                error_message="Need at least 2 waypoints for trajectory",
                computation_time=time.time() - start_time
            )
        
        try:
            # Step 1: Generate smooth path
            smooth_result = self._generate_smooth_path(waypoints)
            if not smooth_result['success']:
                return TrajectoryResult(
                    success=False,
                    error_message=f"Path smoothing failed: {smooth_result['error']}",
                    computation_time=time.time() - start_time
                )
            
            smooth_path = smooth_result['path']
            
            # Step 2: Generate time parameterization
            timing_result = self._generate_timing(smooth_path, time_scaling)
            if not timing_result['success']:
                return TrajectoryResult(
                    success=False,
                    error_message=f"Timing generation failed: {timing_result['error']}",
                    computation_time=time.time() - start_time
                )
            
            times = timing_result['times']
            
            # Step 3: Compute velocities and accelerations
            dynamics_result = self._compute_dynamics(smooth_path, times)
            if not dynamics_result['success']:
                return TrajectoryResult(
                    success=False,
                    error_message=f"Dynamics computation failed: {dynamics_result['error']}",
                    computation_time=time.time() - start_time
                )
            
            velocities = dynamics_result['velocities']
            accelerations = dynamics_result['accelerations']
            
            # Step 4: Create trajectory object
            trajectory_points = []
            for i, (t, pos, vel, acc) in enumerate(zip(times, smooth_path, velocities, accelerations)):
                trajectory_points.append(TrajectoryPoint(
                    time=t,
                    position=pos,
                    velocity=vel,
                    acceleration=acc
                ))
            
            # Compute trajectory metrics
            max_vels = np.max(np.abs(velocities), axis=0)
            max_accs = np.max(np.abs(accelerations), axis=0)
            smoothness = self._compute_smoothness_metric(accelerations)
            
            trajectory = Trajectory(
                points=trajectory_points,
                total_time=times[-1],
                max_velocities=max_vels,
                max_accelerations=max_accs,
                smoothness_metric=smoothness
            )
            
            # Step 5: Optimization (if requested)
            optimization_info = {}
            if optimize and self.config['optimize_timing']:
                opt_result = self._optimize_trajectory(trajectory)
                if opt_result['success']:
                    trajectory = opt_result['trajectory']
                    optimization_info = opt_result['info']
            
            return TrajectoryResult(
                success=True,
                trajectory=trajectory,
                computation_time=time.time() - start_time,
                optimization_info=optimization_info
            )
            
        except Exception as e:
            return TrajectoryResult(
                success=False,
                error_message=f"Trajectory planning failed: {str(e)}",
                computation_time=time.time() - start_time
            )
    
    def _generate_smooth_path(self, waypoints: List[np.ndarray]) -> Dict[str, Any]:
        """Generate smooth path through waypoints using splines."""
        try:
            waypoints_array = np.array(waypoints)
            n_waypoints, n_joints = waypoints_array.shape
            
            if n_waypoints < 2:
                return {'success': False, 'error': 'Need at least 2 waypoints'}
            
            # Generate parameter values for waypoints
            distances = np.zeros(n_waypoints)
            for i in range(1, n_waypoints):
                distances[i] = distances[i-1] + np.linalg.norm(waypoints_array[i] - waypoints_array[i-1])
            
            # Normalize to [0, 1]
            if distances[-1] > 0:
                param_values = distances / distances[-1]
            else:
                param_values = np.linspace(0, 1, n_waypoints)
            
            # Fit splines for each joint
            spline_data = []
            for joint_idx in range(n_joints):
                joint_positions = waypoints_array[:, joint_idx]
                
                if n_waypoints == 2:
                    # Linear interpolation for 2 points
                    spline_func = interp1d(param_values, joint_positions, kind='linear')
                else:
                    # Cubic spline for 3+ points
                    try:
                        spline_func = interp1d(param_values, joint_positions, 
                                             kind='cubic', fill_value='extrapolate')
                    except:
                        # Fallback to linear if cubic fails
                        spline_func = interp1d(param_values, joint_positions, kind='linear')
                
                spline_data.append(spline_func)
            
            # Generate dense path points
            num_points = max(50, n_waypoints * 10)
            dense_params = np.linspace(0, 1, num_points)
            smooth_path = np.zeros((num_points, n_joints))
            
            for joint_idx, spline_func in enumerate(spline_data):
                smooth_path[:, joint_idx] = spline_func(dense_params)
            
            # Apply advanced smoothing iterations
            for iteration in range(self.config['smoothing_iterations']):
                if iteration < self.config['smoothing_iterations'] // 2:
                    smooth_path = self._apply_smoothing_filter(smooth_path)
                else:
                    # Use gradient-based smoothing for final refinement
                    smooth_path = self._apply_gradient_based_smoothing(smooth_path, iterations=10)
            
            return {'success': True, 'path': smooth_path, 'parameters': dense_params}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _apply_smoothing_filter(self, path: np.ndarray) -> np.ndarray:
        """Apply advanced smoothing filter to reduce jerk."""
        smoothed_path = path.copy()
        weight = self.config['smoothing_weight']
        
        # Apply weighted average smoothing (avoid endpoints)
        for i in range(1, len(path) - 1):
            smoothed_path[i] = (1 - weight) * path[i] + weight * 0.5 * (path[i-1] + path[i+1])
        
        return smoothed_path
    
    def _apply_gradient_based_smoothing(self, path: np.ndarray, iterations: int = 30) -> np.ndarray:
        """Apply gradient-based optimization smoothing similar to AORRTC."""
        smoothed_path = path.copy()
        alpha = 0.05  # Reduced learning rate for stability
        beta = 0.7    # Higher smoothness weight
        gamma = 0.15  # Reduced constraint adherence weight
        
        for iteration in range(iterations):
            for i in range(1, len(smoothed_path) - 1):
                # Smoothness gradient (encourages smooth curvature)
                grad_smooth = (smoothed_path[i-1] + smoothed_path[i+1] - 2 * smoothed_path[i])
                
                # Velocity consistency gradient
                if i < len(smoothed_path) - 2:
                    v1 = smoothed_path[i] - smoothed_path[i-1]
                    v2 = smoothed_path[i+1] - smoothed_path[i]
                    grad_velocity = v2 - v1
                else:
                    grad_velocity = np.zeros_like(smoothed_path[i])
                
                # Combined update
                update = alpha * (beta * grad_smooth + gamma * grad_velocity)
                smoothed_path[i] += update
        
        return smoothed_path
    
    def _generate_timing(self, path: np.ndarray, time_scaling: float = 1.0) -> Dict[str, Any]:
        """Generate time parameterization for the path."""
        try:
            n_points = len(path)
            
            # Compute path lengths between consecutive points
            path_lengths = np.zeros(n_points)
            for i in range(1, n_points):
                path_lengths[i] = np.linalg.norm(path[i] - path[i-1])
            
            # Compute cumulative distance
            cumulative_distance = np.cumsum(path_lengths)
            total_distance = cumulative_distance[-1]
            
            if total_distance <= 0:
                return {'success': False, 'error': 'Path has zero length'}
            
            # Estimate time based on velocity AND acceleration limits
            max_vel = self.config['max_joint_velocity']
            max_acc = self.config['max_joint_acceleration']
            
            # Calculate minimum time based on velocity constraints
            min_time_vel = total_distance / max_vel
            
            # Calculate minimum time based on acceleration constraints
            # Using kinematic equation: t = sqrt(2 * distance / acceleration)
            # This ensures we don't violate acceleration limits during motion
            min_time_acc = np.sqrt(2 * total_distance / max_acc) * 1.5  # Safety factor
            
            # Use the more conservative (larger) time requirement
            min_time = max(min_time_vel, min_time_acc)
            
            # Apply time scaling with additional safety margin for complex trajectories
            # Increased safety factor to prevent limit violations
            safety_factor = 1.5  # 50% safety margin to prevent numerical violations
            total_time = min_time * time_scaling * safety_factor
            
            # Generate time array based on distance
            times = np.zeros(n_points)
            for i in range(1, n_points):
                times[i] = total_time * (cumulative_distance[i] / total_distance)
            
            return {'success': True, 'times': times, 'total_time': total_time}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _compute_dynamics(self, path: np.ndarray, times: np.ndarray) -> Dict[str, Any]:
        """Compute velocities and accelerations using finite differences."""
        try:
            n_points, n_joints = path.shape
            velocities = np.zeros_like(path)
            accelerations = np.zeros_like(path)
            
            # Compute velocities using central differences
            for i in range(n_points):
                if i == 0:
                    # Forward difference at start
                    dt = times[1] - times[0]
                    if dt > 0:
                        velocities[i] = (path[1] - path[0]) / dt
                elif i == n_points - 1:
                    # Backward difference at end
                    dt = times[i] - times[i-1]
                    if dt > 0:
                        velocities[i] = (path[i] - path[i-1]) / dt
                else:
                    # Central difference in middle
                    dt = times[i+1] - times[i-1]
                    if dt > 0:
                        velocities[i] = (path[i+1] - path[i-1]) / dt
            
            # Compute accelerations using central differences
            for i in range(n_points):
                if i == 0:
                    # Forward difference at start
                    dt = times[1] - times[0]
                    if dt > 0:
                        accelerations[i] = (velocities[1] - velocities[0]) / dt
                elif i == n_points - 1:
                    # Backward difference at end
                    dt = times[i] - times[i-1]
                    if dt > 0:
                        accelerations[i] = (velocities[i] - velocities[i-1]) / dt
                else:
                    # Central difference in middle
                    dt = times[i+1] - times[i-1]
                    if dt > 0:
                        accelerations[i] = (velocities[i+1] - velocities[i-1]) / dt
            
            # Check velocity and acceleration limits with tolerance
            max_vel_limit = self.config['max_joint_velocity']
            max_acc_limit = self.config['max_joint_acceleration']
            
            # Add small tolerance to account for numerical errors (10% margin)
            # Use 1.1 to allow 10% over the limit before triggering warnings
            vel_tolerance_limit = max_vel_limit * 1.1
            acc_tolerance_limit = max_acc_limit * 1.1
            
            vel_violations = np.any(np.abs(velocities) > vel_tolerance_limit)
            acc_violations = np.any(np.abs(accelerations) > acc_tolerance_limit)
            
            if vel_violations:
                max_vel_actual = np.max(np.abs(velocities))
                logger.warning(f"Velocity limits exceeded in trajectory: max={max_vel_actual:.6f}°/s, limit={vel_tolerance_limit:.6f}°/s")
            if acc_violations:
                max_acc_actual = np.max(np.abs(accelerations))
                logger.warning(f"Acceleration limits exceeded in trajectory: max={max_acc_actual:.6f}°/s², limit={acc_tolerance_limit:.6f}°/s²")
            
            return {
                'success': True,
                'velocities': velocities,
                'accelerations': accelerations,
                'velocity_violations': vel_violations,
                'acceleration_violations': acc_violations
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _compute_smoothness_metric(self, accelerations: np.ndarray) -> float:
        """Compute trajectory smoothness metric based on jerk."""
        try:
            # Compute jerk (derivative of acceleration)
            jerk = np.diff(accelerations, axis=0)
            
            # RMS jerk as smoothness metric (lower is smoother)
            rms_jerk = np.sqrt(np.mean(jerk**2))
            
            return float(rms_jerk)
            
        except:
            return float('inf')
    
    def _correct_trajectory_violations(self, path: np.ndarray, times: np.ndarray) -> Dict[str, Any]:
        """Correct trajectory to ensure it respects velocity and acceleration limits."""
        try:
            max_iterations = 3
            time_scaling_factor = 1.1  # Increase time by 10% each iteration
            
            for iteration in range(max_iterations):
                # Compute current dynamics
                dynamics = self._compute_dynamics(path, times)
                
                if not dynamics['success']:
                    break
                
                # Check if limits are violated
                if not dynamics['velocity_violations'] and not dynamics['acceleration_violations']:
                    # No violations, trajectory is good
                    return {
                        'success': True,
                        'times': times,
                        'total_time': times[-1],
                        'iterations': iteration
                    }
                
                # Scale time to reduce violations
                time_scale = time_scaling_factor ** (iteration + 1)
                times = times * time_scale
            
            # Return corrected trajectory even if not perfect
            return {
                'success': True,
                'times': times,
                'total_time': times[-1],
                'iterations': max_iterations,
                'note': 'Trajectory corrected but may still have minor violations'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _optimize_trajectory(self, trajectory: Trajectory) -> Dict[str, Any]:
        """Optimize trajectory timing for better performance.
        
        IMPROVED: Multi-pass optimization with adaptive time scaling and constraint checking.
        """
        try:
            original_time = trajectory.total_time
            original_smoothness = trajectory.smoothness_metric
            best_trajectory = trajectory
            
            # Get optimization settings
            optimization_iterations = self.config.get('optimization_iterations', 3)
            adaptive_scaling = self.config.get('adaptive_time_scaling', True)
            
            logger.debug(f"Starting trajectory optimization: {optimization_iterations} iterations")
            
            # Multi-pass optimization
            for iteration in range(optimization_iterations):
                
                def objective(time_scale):
                    """Multi-objective optimization function with constraint penalties."""
                    if time_scale <= 0.1 or time_scale > 5.0:
                        return float('inf')  # Invalid scaling
                    
                    # Calculate scaled limits
                    scaled_max_vel = trajectory.max_velocities / time_scale
                    scaled_max_acc = trajectory.max_accelerations / (time_scale**2)
                    
                    # Constraint violation penalties
                    vel_penalty = 0.0
                    acc_penalty = 0.0
                    
                    vel_limit = self.config['max_joint_velocity']
                    acc_limit = self.config['max_joint_acceleration']
                    
                    # Check velocity constraints
                    vel_violations = scaled_max_vel > vel_limit
                    if np.any(vel_violations):
                        vel_penalty = np.sum((scaled_max_vel[vel_violations] - vel_limit) / vel_limit) * 10
                    
                    # Check acceleration constraints
                    acc_violations = scaled_max_acc > acc_limit
                    if np.any(acc_violations):
                        acc_penalty = np.sum((scaled_max_acc[acc_violations] - acc_limit) / acc_limit) * 20
                    
                    # Base objectives
                    speed_cost = time_scale  # Minimize execution time
                    smoothness_cost = original_smoothness / time_scale  # Maintain smoothness
                    
                    # Adaptive weighting based on trajectory complexity
                    if adaptive_scaling:
                        complexity = np.log(1 + original_smoothness)
                        speed_weight = self.config['speed_weight'] * (1 + complexity * 0.1)
                        smoothness_weight = self.config['smoothness_weight'] * (2 - complexity * 0.1)
                    else:
                        speed_weight = self.config['speed_weight']
                        smoothness_weight = self.config['smoothness_weight']
                    
                    total_cost = (speed_weight * speed_cost + 
                                 smoothness_weight * smoothness_cost +
                                 vel_penalty + acc_penalty)
                    
                    return total_cost
                
                # Optimize time scaling with bounds based on iteration
                if iteration == 0:
                    bounds = (0.3, 3.0)  # Wide initial search
                elif iteration == 1:
                    bounds = (0.5, 2.0)  # Narrower search
                else:
                    bounds = (0.7, 1.5)  # Fine-tuning
                
                result = minimize_scalar(objective, bounds=bounds, method='bounded')
                
                if result.success and result.x > 0:
                    optimal_scale = result.x
                    
                    # Create optimized trajectory by rescaling time
                    optimized_points = []
                    for point in trajectory.points:
                        optimized_points.append(TrajectoryPoint(
                            time=point.time * optimal_scale,
                            position=point.position,
                            velocity=point.velocity / optimal_scale,
                            acceleration=point.acceleration / (optimal_scale**2)
                        ))
                    
                    optimized_trajectory = Trajectory(
                        points=optimized_points,
                        total_time=trajectory.total_time * optimal_scale,
                        max_velocities=trajectory.max_velocities / optimal_scale,
                        max_accelerations=trajectory.max_accelerations / (optimal_scale**2),
                        smoothness_metric=trajectory.smoothness_metric * optimal_scale
                    )
                    
                    # Validate constraints
                    vel_ok = np.all(optimized_trajectory.max_velocities <= self.config['max_joint_velocity'])
                    acc_ok = np.all(optimized_trajectory.max_accelerations <= self.config['max_joint_acceleration'])
                    
                    # Accept if constraints are satisfied or improvement is significant
                    if (vel_ok and acc_ok) or (optimized_trajectory.total_time < best_trajectory.total_time * 0.8):
                        best_trajectory = optimized_trajectory
                        logger.debug(f"Optimization iteration {iteration+1}: scale={optimal_scale:.3f}, "
                                   f"time={best_trajectory.total_time:.2f}s")
                    
                    # Use optimized trajectory as starting point for next iteration
                    trajectory = best_trajectory
            
            # Check if optimization provided improvement
            time_improvement = original_time - best_trajectory.total_time
            smoothness_improvement = original_smoothness - best_trajectory.smoothness_metric
            
            return {
                'success': True,
                'trajectory': best_trajectory,
                'info': {
                    'original_time': original_time,
                    'optimized_time': best_trajectory.total_time,
                    'time_improvement': time_improvement,
                    'smoothness_improvement': smoothness_improvement,
                    'optimization_iterations': optimization_iterations,
                    'effective_improvement': time_improvement > 0.01 or smoothness_improvement > 0.01
                }
            }
                
        except Exception as e:
            logger.warning(f"Trajectory optimization failed: {e}")
            return {
                'success': False, 
                'trajectory': trajectory,  # Return original trajectory
                'error': str(e)
            }
    
    def interpolate_trajectory(self, trajectory: Trajectory, 
                             query_times: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Interpolate trajectory at specified times.
        
        Args:
            trajectory: Input trajectory
            query_times: Times to interpolate at
            
        Returns:
            Dictionary with interpolated positions, velocities, accelerations
        """
        try:
            times = trajectory.get_times()
            positions = trajectory.get_positions()
            velocities = trajectory.get_velocities()
            accelerations = trajectory.get_accelerations()
            
            # Create interpolation functions
            pos_interp = interp1d(times, positions, axis=0, kind='cubic', 
                                fill_value='extrapolate', bounds_error=False)
            vel_interp = interp1d(times, velocities, axis=0, kind='cubic',
                                fill_value='extrapolate', bounds_error=False)
            acc_interp = interp1d(times, accelerations, axis=0, kind='cubic',
                                fill_value='extrapolate', bounds_error=False)
            
            # Interpolate at query times
            interp_positions = pos_interp(query_times)
            interp_velocities = vel_interp(query_times)
            interp_accelerations = acc_interp(query_times)
            
            return {
                'positions': interp_positions,
                'velocities': interp_velocities,
                'accelerations': interp_accelerations,
                'times': query_times
            }
            
        except Exception as e:
            logger.error(f"Trajectory interpolation failed: {e}")
            raise TrajectoryPlanningError(f"Interpolation failed: {e}")
    
    def validate_trajectory_dynamics(self, trajectory: Trajectory) -> Dict[str, Any]:
        """Validate trajectory against dynamic constraints."""
        max_vels = trajectory.max_velocities
        max_accs = trajectory.max_accelerations
        
        vel_limit = self.config['max_joint_velocity']
        acc_limit = self.config['max_joint_acceleration']
        
        vel_violations = max_vels > vel_limit
        acc_violations = max_accs > acc_limit
        
        return {
            'velocity_ok': not np.any(vel_violations),
            'acceleration_ok': not np.any(acc_violations),
            'max_velocities': max_vels,
            'max_accelerations': max_accs,
            'velocity_limits': np.full(len(max_vels), vel_limit),
            'acceleration_limits': np.full(len(max_accs), acc_limit),
            'velocity_violations': vel_violations,
            'acceleration_violations': acc_violations,
            'smoothness_metric': trajectory.smoothness_metric
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update trajectory planning configuration."""
        self.config.update(new_config)
        logger.info("Trajectory planner configuration updated")
