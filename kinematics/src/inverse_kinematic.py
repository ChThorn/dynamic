#!/usr/bin/env python3
"""
Inverse Kinematics Module for 6-DOF Robot Manipulator

This module implements robust inverse kinematics solving using damped least squares
method with multiple optimization strategies and fallback mechanisms.

Key Features:
- Damped Least Squares (DLS) method
- Multi-phase optimization strategy
- Random restart mechanisms
- Perturbation-based fallback
- Singularity handling
- Adaptive damping and step scaling
- Constraint checking integration

Author: Robot Control Team
"""

import numpy as np
from numpy.linalg import norm, inv
import logging
from typing import Tuple, Optional, Dict, Any
import time

logger = logging.getLogger(__name__)

class InverseKinematicsError(Exception):
    """Custom exception for inverse kinematics errors."""
    pass

class InverseKinematics:
    """Robust inverse kinematics solver using damped least squares."""
    
    def __init__(self, forward_kinematics, default_params: Optional[Dict[str, Any]] = None):
        """
        Initialize inverse kinematics solver.
        
        Args:
            forward_kinematics: ForwardKinematics instance
            default_params: Default IK parameters
        """
        self.fk = forward_kinematics
        self.S = forward_kinematics.S
        self.M = forward_kinematics.M
        self.joint_limits = forward_kinematics.joint_limits
        self.n_joints = forward_kinematics.n_joints
        
        # Default IK parameters
        self.default_params = default_params or {
            'pos_tol': 2e-3,           # 2mm position tolerance
            'rot_tol': 5e-3,           # ~0.3° rotation tolerance
            'max_iters': 300,          # Maximum iterations per attempt
            'damping': 5e-4,           # Initial damping factor
            'step_scale': 0.5,         # Step scaling factor
            'dq_max': 0.3,             # Maximum joint step size
            'num_attempts': 100,       # Number of random restart attempts
            'combined_tolerance': 3e-3, # Combined error tolerance
            'position_relaxation': 0.01, # Position relaxation for perturbation (1cm)
            'rotation_relaxation': 0.05  # Rotation relaxation for perturbation (~2.8°)
        }
        
        # Performance tracking
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'total_time': 0.0,
            'average_iterations': 0.0
        }
        
        logger.info("Inverse kinematics solver initialized")
    
    def solve(self, T_des: np.ndarray, q_init: Optional[np.ndarray] = None,
              **kwargs) -> Tuple[Optional[np.ndarray], bool]:
        """
        Solve inverse kinematics for desired end-effector pose.
        
        Args:
            T_des: Desired 4x4 homogeneous transformation matrix
            q_init: Initial joint configuration (optional)
            **kwargs: Override default IK parameters
            
        Returns:
            Tuple of (solution joint angles, convergence flag)
        """
        start_time = time.time()
        self.stats['total_calls'] += 1
        
        # Merge parameters
        params = self.default_params.copy()
        params.update(kwargs)
        
        try:
            # Validate inputs
            if T_des.shape != (4, 4):
                raise InverseKinematicsError("T_des must be a 4x4 transformation matrix")
            
            if q_init is not None and (q_init.shape != (self.n_joints,)):
                raise InverseKinematicsError(f"q_init must have shape ({self.n_joints},)")
            
            # Check if target is home position
            q_home, is_home = self._check_home_position(T_des, params)
            if is_home:
                self.stats['successful_calls'] += 1
                self.stats['total_time'] += time.time() - start_time
                return q_home, True
            
            # Main IK solving
            q_solution, converged = self._solve_with_multiple_attempts(T_des, q_init, params)
            
            # Fallback with pose perturbation if main method failed
            if not converged and q_solution is not None:
                logger.info("Main IK failed, attempting perturbation fallback")
                q_perturbed, perturb_converged = self._solve_with_perturbation(T_des, q_solution, params)
                if perturb_converged:
                    q_solution, converged = q_perturbed, True
                    logger.info("Perturbation fallback succeeded")
            
            # Update statistics
            if converged:
                self.stats['successful_calls'] += 1
            
            self.stats['total_time'] += time.time() - start_time
            
            return q_solution, converged
            
        except Exception as e:
            logger.error(f"Inverse kinematics failed: {e}")
            self.stats['total_time'] += time.time() - start_time
            return None, False
    
    def _check_home_position(self, T_des: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, bool]:
        """Check if desired pose is the home position."""
        q_home = np.zeros(self.n_joints)
        T_home = self.fk.compute_forward_kinematics(q_home, suppress_warnings=True)
        
        pos_err = norm(T_des[:3, 3] - T_home[:3, 3])
        rot_err = self._rotation_error(T_des[:3, :3], T_home[:3, :3])
        
        if pos_err < params['pos_tol'] and rot_err < params['rot_tol']:
            logger.info("Target is home position, returning zero joints")
            return q_home, True
        
        return q_home, False
    
    def _solve_with_multiple_attempts(self, T_des: np.ndarray, q_init: Optional[np.ndarray],
                                    params: Dict[str, Any]) -> Tuple[Optional[np.ndarray], bool]:
        """Solve IK with multiple random restart attempts."""
        limits_lower, limits_upper = self.joint_limits[0], self.joint_limits[1]
        
        best_q = None
        best_error = float('inf')
        best_converged = False
        
        # Generate initial configurations for attempts
        attempt_configs = self._generate_attempt_configurations(q_init, params['num_attempts'])
        
        for i, q0 in enumerate(attempt_configs):
            q_sol, converged = self._solve_dls(T_des, q0, params)
            
            if q_sol is not None:
                # Evaluate solution quality
                pos_err, rot_err = self._compute_pose_error(T_des, q_sol)
                total_err = pos_err + rot_err
                combined_err = pos_err + rot_err * 0.1  # Weight rotation error less
                
                if total_err < best_error:
                    best_error = total_err
                    best_q = q_sol.copy()
                    best_converged = converged
                    
                    # Early termination conditions
                    if converged:
                        logger.debug(f"IK converged on attempt {i+1}")
                        break
                    
                    # Accept good enough solutions
                    if combined_err < params['combined_tolerance']:
                        logger.debug(f"IK found acceptable solution on attempt {i+1}")
                        best_converged = True
                        break
        
        return best_q, best_converged
    
    def _generate_attempt_configurations(self, q_init: Optional[np.ndarray], 
                                       num_attempts: int) -> list:
        """Generate initial configurations for IK attempts."""
        limits_lower, limits_upper = self.joint_limits[0], self.joint_limits[1]
        configs = []
        
        # Add user-provided initial guess first
        if q_init is not None:
            configs.append(q_init.copy())
        
        # Add common configurations
        configs.extend([
            np.zeros(self.n_joints),                    # Home position
            (limits_lower + limits_upper) / 2,          # Middle position
        ])
        
        # Generate random configurations with different strategies
        remaining_attempts = num_attempts - len(configs)
        for i in range(remaining_attempts):
            if i % 3 == 0:
                # Uniform random
                q_rand = np.random.uniform(limits_lower, limits_upper, size=(self.n_joints,))
            elif i % 3 == 1:
                # Gaussian around middle
                mid = (limits_lower + limits_upper) / 2
                std = (limits_upper - limits_lower) / 6
                q_rand = np.random.normal(mid, std, size=(self.n_joints,))
                q_rand = np.clip(q_rand, limits_lower, limits_upper)
            else:
                # Perturbation around existing good config
                base = q_init if q_init is not None else np.zeros(self.n_joints)
                q_rand = base + np.random.normal(0, 0.2, self.n_joints)
                q_rand = np.clip(q_rand, limits_lower, limits_upper)
            
            configs.append(q_rand)
        
        return configs
    
    def _solve_dls(self, T_des: np.ndarray, q0: np.ndarray, 
                   params: Dict[str, Any]) -> Tuple[Optional[np.ndarray], bool]:
        """
        Solve IK using enhanced Damped Least Squares method.
        
        Uses a multi-phase optimization strategy with adaptive damping.
        """
        q = q0.copy()
        limits_lower, limits_upper = self.joint_limits[0], self.joint_limits[1]
        
        # Avoid starting exactly at home position (potential singularity)
        if np.allclose(q, 0, atol=1e-6):
            q += np.random.uniform(-0.1, 0.1, self.n_joints)
            logger.debug("Applied perturbation to avoid home position singularity")
        
        best_q = q.copy()
        best_error = float('inf')
        
        # Adaptive damping parameters
        min_damping = params['damping']
        max_damping = 1.0
        damping_factor = min_damping
        
        # Multi-phase optimization - handle both parameter key variations
        max_iters = params.get('max_iterations', params.get('max_iters', 300))
        phase_iters = [max_iters // 3, max_iters // 3, max_iters // 3 + max_iters % 3]
        phase_step_scales = [params['step_scale'] * 1.5, params['step_scale'], params['step_scale'] * 0.5]
        
        iteration = 0
        no_improvement_count = 0
        
        for phase, (phase_iter, phase_step) in enumerate(zip(phase_iters, phase_step_scales)):
            logger.debug(f"IK Phase {phase+1}: {phase_iter} iters, step_scale={phase_step}")
            
            for i in range(phase_iter):
                iteration += 1
                
                # Compute current pose error
                T_cur = self.fk.compute_forward_kinematics(q, suppress_warnings=True)
                error_twist = self._matrix_log6(inv(T_cur) @ T_des)
                
                rot_err = norm(error_twist[:3])
                pos_err = norm(error_twist[3:])
                total_error = pos_err + rot_err
                
                # Update best solution
                if total_error < best_error:
                    improvement = best_error - total_error
                    best_error = total_error
                    best_q = q.copy()
                    no_improvement_count = 0
                    
                    if improvement > 1e-4:
                        logger.debug(f"Iter {iteration}: error improved to {total_error:.6f}")
                else:
                    no_improvement_count += 1
                
                # Check convergence
                combined_error = pos_err + rot_err * 0.1
                if combined_error < (params['pos_tol'] + params['rot_tol'] * 0.1):
                    logger.debug(f"IK converged at iteration {iteration}")
                    return q, True
                
                # Compute body Jacobian
                Jb = self._compute_body_jacobian(q)
                
                # Adaptive damping based on manipulability
                manipulability = self._compute_manipulability(Jb)
                if manipulability < 1e-4:
                    damping_factor = min(max_damping, damping_factor * 1.2)
                elif no_improvement_count > 10:
                    damping_factor = min(max_damping, damping_factor * 1.1)
                    if no_improvement_count > 20:
                        # Apply escape perturbation
                        q += np.random.normal(0, 0.05, self.n_joints)
                        q = np.clip(q, limits_lower, limits_upper)
                        no_improvement_count = 0
                        logger.debug(f"Applied escape perturbation at iter {iteration}")
                else:
                    damping_factor = max(min_damping, damping_factor * 0.99)
                
                # Compute step using damped least squares
                dq = self._compute_dls_step(Jb, error_twist, damping_factor)
                
                # Limit step size
                dq_norm = norm(dq)
                if dq_norm > params['dq_max']:
                    dq = dq * (params['dq_max'] / dq_norm)
                
                # Adaptive step scaling
                adaptive_scale = phase_step
                if total_error > best_error * 1.5:
                    adaptive_scale *= 0.3
                
                # Update joint configuration
                q += adaptive_scale * dq
                q = np.clip(q, limits_lower, limits_upper)
                
                # Check for negligible progress
                if dq_norm < 1e-9 and no_improvement_count > 5:
                    logger.debug(f"Step size negligible at iteration {iteration}")
                    break
            
            # Inter-phase perturbation
            if phase < len(phase_iters) - 1 and best_error > params['pos_tol'] + params['rot_tol']:
                perturbation = np.random.normal(0, 0.1, self.n_joints) * (best_error / (params['pos_tol'] + params['rot_tol']))
                q = best_q + perturbation
                q = np.clip(q, limits_lower, limits_upper)
                logger.debug(f"Applied inter-phase perturbation")
        
        # Final convergence check
        pos_err_final, rot_err_final = self._compute_pose_error(T_des, best_q)
        combined_error_final = pos_err_final + rot_err_final * 0.1
        converged = combined_error_final < (params['pos_tol'] + params['rot_tol'] * 0.1)
        
        if not converged:
            logger.debug(f"IK completed {iteration} iterations, final error: {best_error:.6f}")
        
        return best_q, converged
    
    def _solve_with_perturbation(self, T_des: np.ndarray, q_init: np.ndarray,
                               params: Dict[str, Any]) -> Tuple[Optional[np.ndarray], bool]:
        """Fallback IK solver that perturbs the target pose."""
        pos_relax = params['position_relaxation']
        rot_relax = params['rotation_relaxation']
        
        for i in range(15):  # 15 perturbation attempts
            # Create random perturbation
            pos_offset = np.random.uniform(-pos_relax, pos_relax, 3)
            
            # Random rotation perturbation
            rot_vec = np.random.uniform(-rot_relax, rot_relax, 3)
            angle = norm(rot_vec)
            if angle > 1e-6:
                axis = rot_vec / angle
                K = self.fk.skew_symmetric(axis)
                R_offset = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            else:
                R_offset = np.eye(3)
            
            # Apply perturbation
            T_perturbed = T_des.copy()
            T_perturbed[:3, 3] += pos_offset
            T_perturbed[:3, :3] = T_perturbed[:3, :3] @ R_offset
            
            # Solve IK for perturbed target
            q_sol, converged = self._solve_dls(T_perturbed, q_init, params)
            
            if converged:
                # Check if solution is acceptable for original target
                pos_err, rot_err = self._compute_pose_error(T_des, q_sol)
                if (pos_err < params['pos_tol'] * 2.5 and 
                    rot_err < params['rot_tol'] * 2.5):
                    logger.info(f"Perturbation attempt {i+1} successful")
                    return q_sol, True
        
        return None, False
    
    def _compute_body_jacobian(self, q: np.ndarray) -> np.ndarray:
        """Compute body Jacobian at current configuration."""
        J_s = np.zeros((6, self.n_joints))
        T_temp = np.eye(4)
        
        for i in range(self.n_joints):
            if i == 0:
                J_s[:, i] = self.S[:, i]
            else:
                J_s[:, i] = self._adjoint_matrix(T_temp) @ self.S[:, i]
            T_temp = T_temp @ self.fk.matrix_exp6(self.S[:, i] * q[i])
        
        T_final = self.fk.compute_forward_kinematics(q, suppress_warnings=True)
        return self._adjoint_matrix(inv(T_final)) @ J_s
    
    def _compute_dls_step(self, Jb: np.ndarray, error_twist: np.ndarray, 
                         damping: float) -> np.ndarray:
        """Compute damped least squares step."""
        try:
            JtJ = Jb.T @ Jb
            reg_term = (damping ** 2) * np.eye(self.n_joints)
            damped_inv = inv(JtJ + reg_term)
            return damped_inv @ Jb.T @ error_twist
        except np.linalg.LinAlgError:
            logger.debug("Matrix inversion failed, using pseudoinverse")
            return np.linalg.pinv(Jb) @ error_twist
    
    def _compute_manipulability(self, Jb: np.ndarray) -> float:
        """Compute manipulability measure."""
        JJt = Jb @ Jb.T
        det_JJt = np.linalg.det(JJt)
        return np.sqrt(np.abs(det_JJt))
    
    def _compute_pose_error(self, T_des: np.ndarray, q: np.ndarray) -> Tuple[float, float]:
        """Compute position and rotation error."""
        T_actual = self.fk.compute_forward_kinematics(q, suppress_warnings=True)
        pos_err = norm(T_actual[:3, 3] - T_des[:3, 3])
        rot_err = self._rotation_error(T_actual[:3, :3], T_des[:3, :3])
        return pos_err, rot_err
    
    def _rotation_error(self, R1: np.ndarray, R2: np.ndarray) -> float:
        """Compute rotation error between two rotation matrices."""
        R_err = R1.T @ R2
        cos_angle = np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0)
        return np.arccos(cos_angle)
    
    @staticmethod
    def _adjoint_matrix(T: np.ndarray) -> np.ndarray:
        """Compute adjoint matrix for SE(3) transformation."""
        R, p = T[:3, :3], T[:3, 3]
        p_skew = InverseKinematics._skew_symmetric(p)
        return np.block([[R, np.zeros((3, 3))], [p_skew @ R, R]])
    
    @staticmethod
    def _skew_symmetric(v: np.ndarray) -> np.ndarray:
        """Compute skew-symmetric matrix from 3D vector."""
        return np.array([
            [0, -v[2], v[1]], 
            [v[2], 0, -v[0]], 
            [-v[1], v[0], 0]
        ])
    
    def _matrix_log6(self, T: np.ndarray) -> np.ndarray:
        """
        Compute matrix logarithm for SE(3) transformation.
        
        Returns the 6D screw vector [ω, v] corresponding to the transformation.
        """
        R, p = T[:3, :3], T[:3, 3]
        trace_R = np.trace(R)
        cos_th = np.clip((trace_R - 1.0) * 0.5, -1.0, 1.0)
        theta = np.arccos(cos_th)
        
        if theta < 1e-6:
            # Small angle approximation
            omega = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) * 0.5
            return np.hstack([omega, p])
        
        sin_th = np.sin(theta)
        if abs(sin_th) < 1e-6:
            # Near π rotation - use eigenvalue decomposition
            try:
                eigvals, eigvecs = np.linalg.eig(R)
                axis_idx = np.argmin(np.abs(eigvals - 1.0))
                omega_unit = np.real(eigvecs[:, axis_idx])
                omega_unit = omega_unit / (norm(omega_unit) + 1e-12)
                omega = omega_unit * theta
            except np.linalg.LinAlgError:
                logger.debug("Eigenvalue decomposition failed, using fallback")
                omega = np.array([0., 0., theta])
        else:
            # Standard case
            omega_hat = (R - R.T) * (0.5 / sin_th)
            omega = np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]]) * theta
        
        # Compute v using the V^(-1) matrix
        omega_norm = norm(omega) + 1e-12
        omega_unit = omega / omega_norm
        omega_hat = self._skew_symmetric(omega_unit)
        omega_hat2 = omega_hat @ omega_hat
        
        try:
            cot_half = 1.0 / np.tan(theta * 0.5)
            V_inv = (np.eye(3) / theta - 0.5 * omega_hat + 
                    (1.0 / theta - 0.5 * cot_half) * omega_hat2)
            v = V_inv @ p
        except (ZeroDivisionError, FloatingPointError):
            logger.debug("Numerical issue in V_inv computation, using fallback")
            v = p / (theta + 1e-12)
        
        return np.hstack([omega, v])
    
    def check_singularity(self, q: np.ndarray, threshold: float = 1e-6) -> bool:
        """Check if configuration is near a singularity."""
        Jb = self._compute_body_jacobian(q)
        manipulability = self._compute_manipulability(Jb)
        return manipulability < threshold
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.stats.copy()
        if stats['total_calls'] > 0:
            stats['success_rate'] = stats['successful_calls'] / stats['total_calls']
            stats['average_time'] = stats['total_time'] / stats['total_calls']
        else:
            stats['success_rate'] = 0.0
            stats['average_time'] = 0.0
        return stats
    
    def reset_statistics(self):
        """Reset performance statistics."""
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'total_time': 0.0,
            'average_iterations': 0.0
        }
    
    def update_parameters(self, **params):
        """Update default IK parameters."""
        self.default_params.update(params)
        logger.info(f"Updated IK parameters: {params}")