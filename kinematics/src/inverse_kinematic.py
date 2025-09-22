#!/usr/bin/env python3
"""
Production-Ready Inverse Kinematics Module for 6-DOF Robot Manipulator

This module implements a high-performance, time-budgeted inverse kinematics solver
using damped least squares method with optimizations for real-time industrial applications.
Designed for real-time pick-and-place operations with strict timing constraints.

Key Features:
- Time-budgeted Damped Least Squares (DLS) method
- Solution caching and warm-starting for similar poses
- Nullspace optimization for joint limit avoidance
- Adaptive damping for singularity handling
- Early termination strategies for real-time guarantees
- Vectorized operations and LRU caching for performance
- Pure mathematical computation (no workspace/constraint checking)
- Production-ready with robust error handling

Author: Robot Control Team
"""

import numpy as np
from numpy.linalg import norm, inv, pinv
import logging
from typing import Tuple, Optional, Dict, Any, List, Union, Callable
import time
import collections
from functools import lru_cache

logger = logging.getLogger(__name__)

class InverseKinematicsError(Exception):
    """Custom exception for inverse kinematics errors."""
    pass

class FastIK:
    """
    Time-budgeted fast inverse kinematics solver optimized for real-time pick-and-place operations.
    
    This class implements a highly optimized IK solver that uses:
    - Time-budget constraints (can return partial solutions when deadline approaches)
    - Solution caching and warm-starting for similar target poses
    - Nullspace optimization to avoid joint limits
    - Vectorized operations for performance
    - Adaptive damping with rapid convergence for real-time robotics
    
    Designed for scenarios where:
    1. Low latency is critical (real-time pick-and-place)
    2. Target poses are often nearby the previous pose
    3. Exact precision can be traded for speed in appropriate contexts
    """
    
    def __init__(self, forward_kinematics, default_params: Optional[Dict[str, Any]] = None,
                 cache_size: int = 50):
        """
        Initialize time-budgeted fast inverse kinematics solver.
        
        Args:
            forward_kinematics: ForwardKinematics instance
            default_params: Default IK parameters
            cache_size: Size of solution cache for warm-starts (default: 50)
        """
        self.fk = forward_kinematics
        self.S = forward_kinematics.S
        self.M = forward_kinematics.M
        self.joint_limits = forward_kinematics.joint_limits
        self.n_joints = forward_kinematics.n_joints
        
        # Performance-optimized parameters for real-time applications
        self.default_params = default_params or {
            'time_budget': 0.025,       # MODERATE: 25ms time budget for extended reach
            'pos_tol': 1.5e-3,          # MODERATE: 1.5mm position tolerance (balance of accuracy/reach)
            'rot_tol': 6e-3,            # MODERATE: ~0.35° rotation tolerance
            'max_iters': 120,           # MODERATE: More iterations for extended reach
            'damping_min': 8e-6,        # MODERATE: Slightly lower minimum damping
            'damping_max': 0.6,         # MODERATE: Higher maximum damping for stability
            'step_scale': 0.6,          # MODERATE: Balanced step size
            'dq_max': 0.35,             # MODERATE: Balanced joint steps
            'early_exit_improvement': 8e-5, # MODERATE: Balanced convergence
            'max_attempts': 6,          # MODERATE: More attempts for difficult poses
            'use_warm_start': True,     # Use warm-starting from previous solutions
            'adaptive_damping': True,   # Adjust damping based on error progress
            'nullspace_weight': 0.12,   # MODERATE: Balanced nullspace optimization
            'cache_invalidation': 0.35, # MODERATE: Balanced cache sensitivity
            'acceptable_error': 0.005   # MODERATE: 5mm acceptable for extended reach
        }
        
        # Update parameters with user-provided values
        if default_params:
            self.default_params.update(default_params)
        
        # Solution cache for warm-starting (pose hash -> joint solution)
        self.solution_cache = collections.OrderedDict()
        self.cache_size = cache_size
        
        # Performance tracking
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'cache_hits': 0,
            'early_exits': 0
        }
        
        # Precomputed identity matrix for performance
        self._identity = np.eye(4)
        
        # Initialize cache for Jacobian calculation (improves performance)
        self._compute_body_jacobian = lru_cache(maxsize=32)(self._compute_body_jacobian_impl)
        
        logger.info("Fast IK solver initialized with time budget of "
                   f"{self.default_params['time_budget']*1000:.1f}ms")
    
    def solve(self, T_target: np.ndarray, q_init: Optional[np.ndarray] = None, 
              use_tool_frame: bool = False, **kwargs) -> Tuple[Optional[np.ndarray], bool]:
        """
        Solve the inverse kinematics for a given target pose with time budget constraints.
        
        Args:
            T_target: Target transformation matrix (4x4)
            q_init: Initial guess for joint angles (optional, will use cache/defaults if None)
            use_tool_frame: If True and tool is attached, T_target is the desired tool pose
                           If False, T_target is the desired TCP pose
            **kwargs: Additional parameters to override defaults
                     'time_budget': Override default time budget (seconds)
        
        Returns:
            q_solution: Solution joint angles (None if no solution found within time budget)
            success: Boolean indicating if a solution was found that meets tolerances
        """
        # Start timing
        start_time = time.time()
        self.stats['total_calls'] += 1
        
        # Merge parameters
        params = self.default_params.copy()
        params.update(kwargs)
        
        # Convert tool target to TCP target if needed
        if use_tool_frame and self.fk.tool:
            T_tcp_target = self.fk.tool.transform_tool_to_tcp(T_target)
        else:
            T_tcp_target = T_target
        
        # Try to get a warm start from cache
        q_init = self._get_warm_start(T_tcp_target, q_init, params)
        
        # Execute time-budgeted IK solving
        q_solution, success, info = self._solve_with_time_budget(T_tcp_target, q_init, params)
        
        # Update cache with new solution if successful
        if success and params['use_warm_start']:
            self._update_solution_cache(T_tcp_target, q_solution)
        
        # Update statistics
        solve_time = time.time() - start_time
        self.stats['total_time'] += solve_time
        if success:
            self.stats['successful_calls'] += 1
        
        # Update average time
        if self.stats['total_calls'] > 0:
            self.stats['average_time'] = self.stats['total_time'] / self.stats['total_calls']
            
        # Log performance metrics for monitoring
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"FastIK solved in {solve_time*1000:.2f}ms, success: {success}, "
                        f"iterations: {info.get('iterations', 0)}")
        
        return q_solution, success
    
    def _hash_pose(self, T: np.ndarray) -> str:
        """Generate a hash for pose matrix for cache lookups."""
        # Round to reduce sensitivity to tiny changes
        pos = tuple(np.round(T[:3, 3], 5))
        
        # Extract rotation representation (more stable than using full matrix)
        # Get rotation angle
        angle = np.arccos((np.trace(T[:3, :3]) - 1) / 2)
        if abs(angle) < 1e-6:
            axis = (0, 0, 1)  # Default axis for identity rotation
        else:
            # Extract axis using eigenvalue decomposition
            vals, vecs = np.linalg.eig(T[:3, :3])
            real_idx = np.argmin(np.abs(vals - 1.0))
            axis = tuple(np.round(np.real(vecs[:, real_idx]), 3))
        
        return str((pos, angle, axis))
    
    def _get_warm_start(self, T_target: np.ndarray, q_init: Optional[np.ndarray],
                        params: Dict[str, Any]) -> np.ndarray:
        """
        Get initial configuration for IK solving with warm-starting when appropriate.
        
        Prioritizes:
        1. User-provided initial guess (q_init)
        2. Cached solution for similar poses
        3. Strategic seed positions
        """
        if q_init is not None:
            return q_init
            
        # Try cache lookup if enabled
        if params['use_warm_start']:
            pose_hash = self._hash_pose(T_target)
            if pose_hash in self.solution_cache:
                self.stats['cache_hits'] += 1
                logger.debug("Using cached solution for warm-start")
                return self.solution_cache[pose_hash].copy()
        
        # Fall back to reasonable defaults
        # Try a few strategic positions for faster convergence
        seed_configs = [
            # Home position with small offsets to avoid singularity
            np.array([0.01, -0.01, 0.01, 0.01, -0.01, 0.01]),
            
            # Position biased for common picking poses
            np.array([0.0, -0.3, 0.6, 0.0, 0.3, 0.0])
        ]
        
        # Choose config closest to home position
        # In real-time pick-and-place, most operations happen in front workspace
        return seed_configs[1].copy()
    
    def _update_solution_cache(self, T_target: np.ndarray, q_solution: np.ndarray):
        """Update solution cache with new pose -> solution mapping."""
        pose_hash = self._hash_pose(T_target)
        
        # Add to cache (or update existing entry)
        self.solution_cache[pose_hash] = q_solution.copy()
        
        # Keep cache size limited
        while len(self.solution_cache) > self.cache_size:
            self.solution_cache.popitem(last=False)  # Remove oldest entry (FIFO)
    
    def _solve_with_time_budget(self, T_des: np.ndarray, q_init: np.ndarray,
                              params: Dict[str, Any]) -> Tuple[Optional[np.ndarray], bool, Dict[str, Any]]:
        """
        Solve IK with strict time budget constraints.
        
        Returns:
            q_solution: Solution joint configuration (or None if failed)
            success: Whether a solution was found that meets tolerances
            info: Dictionary with additional solving info (iterations, etc.)
        """
        time_budget = params['time_budget']
        start_time = time.time()
        deadline = start_time + time_budget
        
        # Setup for multiple attempts
        max_attempts = params['max_attempts']
        attempt = 0
        best_q = None
        best_error = float('inf')
        
        # Performance tracking
        info = {
            'iterations': 0,
            'time_used': 0.0,
            'early_exits': 0,
            'attempts': 0
        }
        
        # Generate initial configurations for attempts
        base_configs = self._generate_fast_attempt_configs(q_init, max_attempts)
        
        # Main solving loop with time budget enforcement
        while attempt < max_attempts:
            # Check time budget
            time_now = time.time()
            time_remaining = deadline - time_now
            
            # Exit if we're out of time or close to deadline
            # Leave 10% of budget for final cleanup
            if time_remaining <= time_budget * 0.1:
                logger.debug(f"Time budget nearly exceeded after {attempt} attempts")
                break
                
            # Get next configuration to try
            q_attempt = base_configs[attempt].copy()
            attempt += 1
            info['attempts'] += 1
            
            # Solve with remaining time budget (minus safety margin)
            q_sol, converged, solve_info = self._time_budgeted_dls_solve(
                T_des, q_attempt, params, time_remaining * 0.9)
            
            # Update info
            info['iterations'] += solve_info['iterations']
            info['early_exits'] += solve_info.get('early_exit', 0)
            
            if q_sol is not None:
                # Evaluate solution quality
                pos_err, rot_err = self._compute_pose_error(T_des, q_sol)
                total_err = pos_err + rot_err * 0.2  # Weight rotation less
                
                # Update best solution
                if total_err < best_error:
                    best_error = total_err
                    best_q = q_sol.copy()
                    
                    # Early exit if error is acceptable for pick and place
                    if pos_err < params['acceptable_error'] and rot_err < params['rot_tol']:
                        logger.debug(f"Acceptable solution found on attempt {attempt}")
                        self.stats['early_exits'] += 1
                        info['early_exits'] += 1
                        break
                    
                    # Also exit if fully converged
                    if converged:
                        logger.debug(f"Converged solution on attempt {attempt}")
                        break
        
        # Finalize results
        info['time_used'] = time.time() - start_time
        
        # Check if solution meets basic tolerances
        success = False
        if best_q is not None:
            pos_err, rot_err = self._compute_pose_error(T_des, best_q)
            success = (pos_err < params['pos_tol'] and rot_err < params['rot_tol'])
            
            # For real-time pick-and-place, we can be more lenient
            # This is a common practice - allow slightly reduced accuracy for speed
            if not success and pos_err < params['acceptable_error'] and rot_err < params['rot_tol'] * 1.5:
                logger.debug(f"Using acceptable solution with pos_err={pos_err:.4f}, rot_err={rot_err:.4f}")
                success = True
        
        return best_q, success, info
    
    def _generate_fast_attempt_configs(self, q_init: np.ndarray, max_attempts: int) -> list:
        """
        Generate a small set of diverse initial configurations for fast attempts.
        
        Unlike the full IK solver, we use a much smaller set focused on likely
        pick-and-place configurations. The first attempt is always the provided
        initial guess (or cache hit).
        """
        configs = [q_init.copy()]
        limits_lower, limits_upper = self.joint_limits[0], self.joint_limits[1]
        
        if max_attempts <= 1:
            return configs
            
        # Add strategic pick-and-place configs
        # These are joint configurations that commonly work well for top-down
        # and angled grasping, which are common in pick-and-place
        
        seed_configs = [
            # Top-down picking pose (slightly to front)
            np.array([0.0, -0.5, 1.1, 0.0, 0.3, 0.0]),
            
            # Angled picking pose 
            np.array([0.4, -0.4, 0.8, 0.7, 0.4, 0.0]),
            
            # Alternative angle
            np.array([-0.4, -0.4, 0.8, -0.7, 0.4, 0.0]),
            
            # Low front pose
            np.array([0.0, -0.1, 0.2, 0.0, 0.1, 0.0])
        ]
        
        for seed in seed_configs[:max_attempts-1]:
            if len(configs) < max_attempts:
                # Ensure joint limits are respected
                seed = np.clip(seed, limits_lower, limits_upper)
                configs.append(seed)
                
        return configs
    
    def _time_budgeted_dls_solve(self, T_des: np.ndarray, q0: np.ndarray, params: Dict[str, Any],
                               time_budget: float) -> Tuple[Optional[np.ndarray], bool, Dict[str, Any]]:
        """
        Damped Least Squares IK solver with strict time budget.
        
        Uses aggressive step sizes, adaptive damping, and nullspace optimization
        for joint limit avoidance. Will return best partial solution when time budget
        is about to expire.
        
        Args:
            T_des: Desired end-effector pose
            q0: Initial joint configuration
            params: Solver parameters
            time_budget: Time budget in seconds
            
        Returns:
            q_solution: Best solution found (even if not fully converged)
            converged: Whether solution fully converged to desired tolerances
            info: Dictionary with solving info (iterations, etc.)
        """
        q = q0.copy()
        limits_lower, limits_upper = self.joint_limits[0], self.joint_limits[1]
        
        # Initialize damping with adaptive bounds
        damping_min = params['damping_min']
        damping_max = params['damping_max']
        damping = 0.01  # Start with moderate damping
        
        # Use aggressive step scaling for speed
        step_scale = params['step_scale']
        dq_max = params['dq_max']
        
        # Performance tracking
        info = {
            'iterations': 0,
            'early_exit': 0
        }
        
        # Timing
        start_time = time.time()
        deadline = start_time + time_budget
        
        # Main iteration loop
        best_q = q.copy()
        best_error = float('inf')
        no_improvement_count = 0
        
        # For real-time performance, we use more aggressive parameters
        max_iters = min(params['max_iters'], 100)  # Cap iterations for real-time
        
        for iteration in range(max_iters):
            # Check time budget (check every iteration)
            if time.time() >= deadline:
                logger.debug(f"Time budget exceeded at iteration {iteration}")
                break
            
            info['iterations'] += 1
            
            # Compute current pose error
            T_cur = self.fk.compute_forward_kinematics(q)
            error_twist = self._compute_error_twist(T_des, T_cur)
            
            pos_err = norm(error_twist[3:])
            rot_err = norm(error_twist[:3])
            total_error = pos_err + rot_err * 0.2  # Weight rotation less for pick-and-place
            
            # Update best solution
            if total_error < best_error:
                improvement = best_error - total_error
                best_error = total_error
                best_q = q.copy()
                no_improvement_count = 0
                
                # Early exit for small improvements
                if improvement < params['early_exit_improvement'] and iteration > 10:
                    logger.debug(f"Early exit at iteration {iteration}, small improvement")
                    info['early_exit'] = 1
                    break
            else:
                no_improvement_count += 1
                
                # Early termination if stuck with no improvement
                if no_improvement_count > 10:
                    logger.debug(f"Early exit at iteration {iteration}, no improvement")
                    info['early_exit'] = 1
                    break
            
            # Check convergence
            # For pick-and-place, we can use more relaxed tolerances
            acceptable_error = params.get('acceptable_error', params['pos_tol'])
            if pos_err < acceptable_error and rot_err < params['rot_tol']:
                logger.debug(f"Converged at iteration {iteration}")
                return q, True, info
            
            # Compute Jacobian - use cached version for performance
            Jb = self._compute_body_jacobian(tuple(q))
            
            # Adaptive damping based on error and manipulability
            if params['adaptive_damping']:
                manipulability = np.sqrt(abs(np.linalg.det(Jb @ Jb.T) + 1e-10))
                if manipulability < 0.01:  # Near singularity
                    damping = min(damping_max, damping * 1.5)
                elif no_improvement_count > 5:
                    damping = min(damping_max, damping * 1.2)
                else:
                    damping = max(damping_min, damping * 0.9)  # Reduce damping for faster convergence
            
            # Compute step with damped least squares
            # Fast implementation optimized for real-time
            JtJ = Jb.T @ Jb
            reg_term = (damping ** 2) * np.eye(self.n_joints)
            dq_raw = np.linalg.solve(JtJ + reg_term, Jb.T @ error_twist)
            
            # Add nullspace optimization for joint limit avoidance
            if params['nullspace_weight'] > 0:
                # Compute nullspace projection
                nullspace_proj = np.eye(self.n_joints) - pinv(Jb) @ Jb
                
                # Joint limit gradient (push away from limits)
                limit_gradient = self._joint_limit_gradient(q)
                
                # Add weighted nullspace component
                dq_nullspace = nullspace_proj @ limit_gradient
                dq_raw += params['nullspace_weight'] * dq_nullspace
            
            # Limit maximum step size
            dq_norm = norm(dq_raw)
            if dq_norm > dq_max:
                dq = dq_raw * (dq_max / dq_norm)
            else:
                dq = dq_raw
            
            # Apply step with scaling
            q_new = q + step_scale * dq
            
            # Apply joint limits
            q = np.clip(q_new, limits_lower, limits_upper)
        
        # Return best solution found, even if not fully converged
        return best_q, False, info
    
    def _joint_limit_gradient(self, q: np.ndarray) -> np.ndarray:
        """
        Compute gradient to push joints away from limits.
        
        Returns a vector that points away from joint limits.
        Used for nullspace optimization.
        """
        limits_lower, limits_upper = self.joint_limits[0], self.joint_limits[1]
        mid_point = (limits_lower + limits_upper) / 2
        range_half = (limits_upper - limits_lower) / 2
        
        # Normalize joint positions to [-1, 1] range where:
        # -1 = lower limit, 0 = midpoint, 1 = upper limit
        q_norm = (q - mid_point) / (range_half + 1e-10)
        
        # Compute gradient (higher when closer to limits)
        # Uses cubic function to create stronger gradient near limits
        gradient = -q_norm * (1.0 - q_norm**2)
        
        return gradient
    
    def _compute_body_jacobian_impl(self, q_tuple: Tuple[float, ...]) -> np.ndarray:
        """
        Compute body Jacobian at given configuration.
        
        Implementation separated for LRU caching.
        """
        q = np.array(q_tuple)
        J_s = np.zeros((6, self.n_joints))
        T_temp = self._identity.copy()
        
        for i in range(self.n_joints):
            if i == 0:
                J_s[:, i] = self.S[:, i]
            else:
                J_s[:, i] = self._adjoint_matrix(T_temp) @ self.S[:, i]
            T_temp = T_temp @ self.fk.matrix_exp6(self.S[:, i] * q[i])
        
        T_final = self.fk.compute_forward_kinematics(q)
        return self._adjoint_matrix(inv(T_final)) @ J_s
    
    def _compute_error_twist(self, T_des: np.ndarray, T_cur: np.ndarray) -> np.ndarray:
        """
        Compute the error twist between desired and current pose.
        
        Uses the se(3) logarithm of the relative transformation.
        Optimized implementation for real-time performance.
        
        Returns:
            6D error twist [angular_error, position_error]
        """
        # Compute relative transformation
        T_rel = inv(T_cur) @ T_des
        
        # Extract rotation and translation
        R, p = T_rel[:3, :3], T_rel[:3, 3]
        
        # Compute angular error using logarithm of rotation
        trace_R = np.trace(R)
        cos_theta = np.clip((trace_R - 1.0) * 0.5, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        if theta < 1e-6:
            # Small angle approximation
            omega = np.zeros(3)
        else:
            sin_theta = np.sin(theta)
            if abs(sin_theta) < 1e-6:
                # Near pi rotation - use eigenvalue decomposition
                try:
                    eigvals, eigvecs = np.linalg.eig(R)
                    idx = np.argmin(np.abs(eigvals - 1.0))
                    omega_hat = np.real(eigvecs[:, idx])
                    omega = omega_hat * theta / (norm(omega_hat) + 1e-12)
                except:
                    # Fallback
                    omega = np.array([0., 0., theta])
            else:
                # Standard case
                omega_hat = (R - R.T) * (0.5 / sin_theta)
                omega = np.array([
                    omega_hat[2, 1], 
                    omega_hat[0, 2], 
                    omega_hat[1, 0]
                ]) * theta
                
        # Compute linear error (simplified for real-time)
        return np.hstack([omega, p])
    
    @staticmethod
    def _adjoint_matrix(T: np.ndarray) -> np.ndarray:
        """
        Compute adjoint matrix for SE(3) transformation.
        
        Fast implementation for performance.
        """
        R, p = T[:3, :3], T[:3, 3]
        p_skew = np.array([
            [0, -p[2], p[1]], 
            [p[2], 0, -p[0]], 
            [-p[1], p[0], 0]
        ])
        
        adj = np.zeros((6, 6))
        adj[:3, :3] = R
        adj[3:, 3:] = R
        adj[3:, :3] = p_skew @ R
        
        return adj
    
    def _compute_pose_error(self, T_des: np.ndarray, q: np.ndarray) -> Tuple[float, float]:
        """
        Compute position and orientation error between desired pose and configuration.
        
        Args:
            T_des: Desired end-effector pose
            q: Joint configuration
            
        Returns:
            pos_err: Position error (Euclidean norm)
            rot_err: Rotation error (angle of relative rotation)
        """
        # Compute forward kinematics
        T_actual = self.fk.compute_forward_kinematics(q)
        
        # Position error
        pos_err = norm(T_actual[:3, 3] - T_des[:3, 3])
        
        # Rotation error (angle of relative rotation)
        R_err = T_actual[:3, :3].T @ T_des[:3, :3]
        cos_angle = np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0)
        rot_err = np.arccos(cos_angle)
        
        return pos_err, rot_err
    
    def solve_tcp_pose(self, T_tcp: np.ndarray, q_init: Optional[np.ndarray] = None,
                      **kwargs) -> Tuple[Optional[np.ndarray], bool]:
        """
        Solve inverse kinematics for TCP pose (ignores tool if attached).
        
        Convenience method that calls solve() with use_tool_frame=False.
        """
        return self.solve(T_tcp, q_init, use_tool_frame=False, **kwargs)
    
    def solve_tool_pose(self, T_tool: np.ndarray, q_init: Optional[np.ndarray] = None,
                       **kwargs) -> Tuple[Optional[np.ndarray], bool]:
        """
        Solve inverse kinematics for tool pose.
        
        Convenience method that calls solve() with use_tool_frame=True.
        """
        return self.solve(T_tool, q_init, use_tool_frame=True, **kwargs)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.stats.copy()
        if stats['total_calls'] > 0:
            stats['success_rate'] = stats['successful_calls'] / stats['total_calls']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_calls']
        else:
            stats['success_rate'] = 0.0
            stats['cache_hit_rate'] = 0.0
        return stats
    
    def clear_cache(self):
        """Clear solution cache."""
        self.solution_cache.clear()
        logger.debug("Solution cache cleared")
    
    def reset_statistics(self):
        """Reset performance statistics."""
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'cache_hits': 0,
            'early_exits': 0
        }
        
    def update_parameters(self, **params):
        """Update solver parameters."""
        self.default_params.update(params)
        logger.info(f"FastIK parameters updated: {params}")
    
    def set_time_budget(self, time_budget: float):
        """
        Set time budget for real-time solving.
        
        Args:
            time_budget: Time budget in seconds
        """
        self.default_params['time_budget'] = time_budget
        logger.info(f"Time budget set to {time_budget*1000:.1f}ms")
        
    def warm_start_from_recent_poses(self, pose_sequence: List[np.ndarray], 
                                    q_sequence: List[np.ndarray]):
        """
        Warm start the solver with a sequence of recent poses and solutions.
        
        Useful for tracking paths or repeating similar pick-and-place operations.
        
        Args:
            pose_sequence: List of recent end-effector poses
            q_sequence: List of corresponding joint solutions
        """
        for pose, q in zip(pose_sequence, q_sequence):
            self._update_solution_cache(pose, q)