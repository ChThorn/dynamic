#!/usr/bin/env python3
"""
Motion Planning Coordinator Module

This module provides the high-level motion planning interface that coordinates
between path planning, trajectory planning, and constraint checking to provid            # Solve IK for start and goal poses with constraint validation
            q_start, start_valid = self.solve_constrained_ik(start_pose)
            if not start_valid:
                return MotionPlanningResult(
                    status=PlanningStatus.IK_CONSTRAINT_VIOLATION,
                    error_message="Failed to find constraint-satisfying IK solution for start pose",
                    planning_time=time.time() - start_time
                )
            
            q_goal, goal_valid = self.solve_constrained_ik(goal_pose)
            if not goal_valid:
                return MotionPlanningResult(
                    status=PlanningStatus.IK_CONSTRAINT_VIOLATION,
                    error_message="Failed to find constraint-satisfying IK solution for goal pose",
                    planning_time=time.time() - start_time
                )ion planning solution for robot applications.

Key Features:
- High-level motion planning interface
- Integration of path and trajectory planning
- Execution monitoring and safety validation
- Multiple planning strategies (joint space, Cartesian space)
- Fallback planning and recovery behaviors
- Real-time execution monitoring

The MotionPlanner serves as the main entry point for robot motion planning,
orchestrating all lower-level planning modules while ensuring safety and
performance requirements are met.

Author: Robot Control Team
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import time
import os

logger = logging.getLogger(__name__)

# Import enhanced collision checker
try:
    from .collision_checker import EnhancedCollisionChecker, CollisionResult, CollisionType
except ImportError:
    from collision_checker import EnhancedCollisionChecker, CollisionResult, CollisionType

class PlanningStrategy(Enum):
    """Available motion planning strategies."""
    JOINT_SPACE = "joint_space"
    CARTESIAN_SPACE = "cartesian_space"
    HYBRID = "hybrid"

class PlanningStatus(Enum):
    """Motion planning status codes."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    CONSTRAINT_VIOLATION = "constraint_violation"
    IK_FAILED = "ik_failed"
    IK_CONSTRAINT_VIOLATION = "ik_constraint_violation"
    COLLISION_DETECTED = "collision_detected"

@dataclass
class MotionPlan:
    """Complete motion plan with path and trajectory."""
    joint_waypoints: List[np.ndarray]
    trajectory: Optional[Any] = None  # Trajectory from trajectory_planner
    cartesian_waypoints: Optional[List[np.ndarray]] = None
    planning_time: float = 0.0
    strategy_used: Optional[PlanningStrategy] = None
    validation_results: Optional[Dict[str, Any]] = None
    
    @property
    def num_waypoints(self) -> int:
        """Number of joint waypoints in the plan."""
        return len(self.joint_waypoints) if self.joint_waypoints else 0
    
    @property
    def total_time(self) -> float:
        """Total execution time of the trajectory."""
        return self.trajectory.total_time if self.trajectory else 0.0

@dataclass
class MotionPlanningResult:
    """Result container for motion planning operations."""
    status: PlanningStatus
    plan: Optional[MotionPlan] = None
    error_message: Optional[str] = None
    planning_time: float = 0.0
    attempts_made: int = 1
    fallback_used: bool = False

class MotionPlanner:
    """Clean, minimal motion planning coordinator using AORRTC."""
    
    def __init__(self, kinematics_fk, kinematics_ik, 
                 path_planner=None, trajectory_planner=None):
        """
        Initialize clean motion planner with AORRTC system.
        
        Args:
            kinematics_fk: ForwardKinematics instance
            kinematics_ik: InverseKinematics instance  
            path_planner: PathPlanner instance (created if None)
            trajectory_planner: TrajectoryPlanner instance (created if None)
        """
        self.fk = kinematics_fk
        self.ik = kinematics_ik
        
        # Initialize clean planners
        if path_planner is None:
            try:
                from .path_planner import PathPlanner
                self.path_planner = PathPlanner(kinematics_fk, kinematics_ik)
            except ImportError:
                from path_planner import PathPlanner
                self.path_planner = PathPlanner(kinematics_fk, kinematics_ik)
        else:
            self.path_planner = path_planner
            
        if trajectory_planner is None:
            try:
                from .trajectory_planner import TrajectoryPlanner
                self.trajectory_planner = TrajectoryPlanner(self.path_planner)
            except ImportError:
                from trajectory_planner import TrajectoryPlanner
                self.trajectory_planner = TrajectoryPlanner(self.path_planner)
        else:
            self.trajectory_planner = trajectory_planner
            
        # Initialize enhanced collision checker
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'constraints.yaml')
        self.collision_checker = EnhancedCollisionChecker(config_path)
        
        # Minimal configuration for AORRTC system
        self.config = {
            'default_strategy': PlanningStrategy.JOINT_SPACE,
            'max_planning_time': 30.0,
            'default_waypoint_count': 10,
            'ik_max_attempts': 5,
            'ik_position_tolerance': 0.002,  # 2mm - more realistic tolerance
            'ik_rotation_tolerance': 2.0     # 2 degrees - more realistic tolerance
        }
        
        # Simple statistics
        self.stats = {
            'total_plans': 0,
            'successful_plans': 0,
            'failed_plans': 0
        }
        
        logger.info("Clean motion planner initialized: AORRTC + smart fallback")
    
    def plan_motion(self, start_config: np.ndarray, goal_config: np.ndarray,
                   strategy: Optional[PlanningStrategy] = None,
                   waypoint_count: Optional[int] = None,
                   **kwargs) -> MotionPlanningResult:
        """
        Plan motion between joint configurations.
        
        Args:
            start_config: Starting joint configuration
            goal_config: Goal joint configuration
            strategy: Planning strategy to use
            waypoint_count: Number of waypoints to generate
            **kwargs: Additional planning parameters
            
        Returns:
            MotionPlanningResult with planned motion
        """
        start_time = time.time()
        self.stats['total_plans'] += 1
        
        # Set default parameters
        strategy = strategy or self.config['default_strategy']
        waypoint_count = waypoint_count or self.config['default_waypoint_count']
        
        try:
            # Validate inputs
            validation_result = self._validate_planning_inputs(start_config, goal_config)
            if not validation_result['valid']:
                return MotionPlanningResult(
                    status=PlanningStatus.CONSTRAINT_VIOLATION,
                    error_message=validation_result['error'],
                    planning_time=time.time() - start_time
                )
            
            # Attempt planning with primary strategy
            result = self._attempt_planning(
                start_config, goal_config, strategy, waypoint_count, **kwargs
            )
            
            # Try fallback strategies if primary failed
            if (not result.status == PlanningStatus.SUCCESS and 
                self.config['enable_fallbacks'] and
                result.attempts_made < self.config['max_attempts']):
                
                result = self._try_fallback_strategies(
                    start_config, goal_config, strategy, waypoint_count, 
                    result, **kwargs
                )
            
            # Update statistics
            planning_time = time.time() - start_time
            self._update_statistics(result, planning_time)
            
            # Set final planning time
            result.planning_time = planning_time
            
            if result.plan:
                result.plan.planning_time = planning_time
            
            return result
            
        except Exception as e:
            logger.error(f"Motion planning failed with exception: {e}")
            return MotionPlanningResult(
                status=PlanningStatus.FAILED,
                error_message=f"Planning exception: {str(e)}",
                planning_time=time.time() - start_time
            )
    
    def plan_cartesian_motion(self, start_pose: np.ndarray, goal_pose: np.ndarray,
                            orientation_constraint: bool = True,
                            **kwargs) -> MotionPlanningResult:
        """
        Plan motion between Cartesian poses.
        
        Args:
            start_pose: Starting 4x4 transformation matrix
            goal_pose: Goal 4x4 transformation matrix  
            orientation_constraint: Whether to maintain orientation constraints
            **kwargs: Additional planning parameters
            
        Returns:
            MotionPlanningResult with planned motion
        """
        start_time = time.time()
        
        try:
            # Convert poses to joint configurations
            q_start, start_converged = self.ik.solve(start_pose)
            if not start_converged:
                return MotionPlanningResult(
                    status=PlanningStatus.IK_FAILED,
                    error_message="Failed to solve IK for start pose",
                    planning_time=time.time() - start_time
                )
            
            q_goal, goal_converged = self.ik.solve(goal_pose)
            if not goal_converged:
                return MotionPlanningResult(
                    status=PlanningStatus.IK_FAILED,
                    error_message="Failed to solve IK for goal pose",
                    planning_time=time.time() - start_time
                )
            
            # Plan in joint space between IK solutions
            result = self.plan_motion(
                q_start, q_goal, 
                strategy=PlanningStrategy.CARTESIAN_SPACE,
                **kwargs
            )
            
            # Add Cartesian waypoints if successful
            if result.status == PlanningStatus.SUCCESS and result.plan:
                cartesian_waypoints = []
                for q in result.plan.joint_waypoints:
                    T = self.fk.compute_forward_kinematics(q)
                    cartesian_waypoints.append(T)
                
                result.plan.cartesian_waypoints = cartesian_waypoints
            
            return result
            
        except Exception as e:
            return MotionPlanningResult(
                status=PlanningStatus.FAILED,
                error_message=f"Cartesian planning failed: {str(e)}",
                planning_time=time.time() - start_time
            )
    
    def plan_waypoint_motion(self, waypoints: List[np.ndarray],
                           strategy: Optional[PlanningStrategy] = None,
                           **kwargs) -> MotionPlanningResult:
        """
        Plan motion through a sequence of waypoints.
        
        Args:
            waypoints: List of joint configurations to visit
            strategy: Planning strategy to use
            **kwargs: Additional planning parameters
            
        Returns:
            MotionPlanningResult with planned motion
        """
        start_time = time.time()
        
        if len(waypoints) < 2:
            return MotionPlanningResult(
                status=PlanningStatus.FAILED,
                error_message="Need at least 2 waypoints for motion planning",
                planning_time=time.time() - start_time
            )
        
        try:
            # Validate all waypoints
            for i, waypoint in enumerate(waypoints):
                validation = self._validate_joint_config(waypoint)
                if not validation['valid']:
                    return MotionPlanningResult(
                        status=PlanningStatus.CONSTRAINT_VIOLATION,
                        error_message=f"Waypoint {i} invalid: {validation['error']}",
                        planning_time=time.time() - start_time
                    )
            
            # Plan path through waypoints
            path_result = self.path_planner.validate_joint_path(waypoints)
            if not path_result.success:
                return MotionPlanningResult(
                    status=PlanningStatus.CONSTRAINT_VIOLATION,
                    error_message=f"Waypoint path validation failed: {path_result.error_message}",
                    planning_time=time.time() - start_time
                )
            
            # Generate trajectory (filter out path-specific parameters)
            traj_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['use_aorrtc', 'max_iterations', 'step_size']}
            traj_result = self.trajectory_planner.plan_trajectory(waypoints, **traj_kwargs)
            if not traj_result.success:
                return MotionPlanningResult(
                    status=PlanningStatus.FAILED,
                    error_message=f"Trajectory planning failed: {traj_result.error_message}",
                    planning_time=time.time() - start_time
                )
            
            # Create motion plan
            plan = MotionPlan(
                joint_waypoints=waypoints,
                trajectory=traj_result.trajectory,
                strategy_used=strategy or self.config['default_strategy'],
                validation_results=path_result.validation_results
            )
            
            return MotionPlanningResult(
                status=PlanningStatus.SUCCESS,
                plan=plan,
                planning_time=time.time() - start_time
            )
            
        except Exception as e:
            return MotionPlanningResult(
                status=PlanningStatus.FAILED,
                error_message=f"Waypoint planning failed: {str(e)}",
                planning_time=time.time() - start_time
            )
    
    def _validate_planning_inputs(self, start_config: np.ndarray, 
                                goal_config: np.ndarray) -> Dict[str, Any]:
        """Validate planning inputs."""
        # Check joint configuration validity
        start_validation = self._validate_joint_config(start_config)
        if not start_validation['valid']:
            return {'valid': False, 'error': f"Start config invalid: {start_validation['error']}"}
        
        goal_validation = self._validate_joint_config(goal_config)
        if not goal_validation['valid']:
            return {'valid': False, 'error': f"Goal config invalid: {goal_validation['error']}"}
        
        return {'valid': True}
    
    def _validate_joint_config(self, q: np.ndarray) -> Dict[str, Any]:
        """Validate single joint configuration."""
        try:
            # Check joint limits
            joint_valid, joint_msg = self.path_planner.constraints_checker.check_joint_limits(q)
            if not joint_valid:
                return {'valid': False, 'error': joint_msg}
            
            # Check workspace via FK
            T = self.fk.compute_forward_kinematics(q)
            pose_valid, pose_msg = self.path_planner.constraints_checker.check_pose_constraints(T)
            if not pose_valid:
                return {'valid': False, 'error': pose_msg}
            
            return {'valid': True}
            
        except Exception as e:
            return {'valid': False, 'error': f"Validation failed: {str(e)}"}
    
    def _attempt_planning(self, start_config: np.ndarray, goal_config: np.ndarray,
                         strategy: PlanningStrategy, waypoint_count: int,
                         **kwargs) -> MotionPlanningResult:
        """Attempt planning with specified strategy."""
        try:
            if strategy == PlanningStrategy.JOINT_SPACE:
                return self._plan_joint_space(start_config, goal_config, waypoint_count, **kwargs)
            elif strategy == PlanningStrategy.CARTESIAN_SPACE:
                return self._plan_cartesian_space(start_config, goal_config, waypoint_count, **kwargs)
            elif strategy == PlanningStrategy.HYBRID:
                return self._plan_hybrid(start_config, goal_config, waypoint_count, **kwargs)
            else:
                return MotionPlanningResult(
                    status=PlanningStatus.FAILED,
                    error_message=f"Unknown planning strategy: {strategy}"
                )
                
        except Exception as e:
            return MotionPlanningResult(
                status=PlanningStatus.FAILED,
                error_message=f"Planning attempt failed: {str(e)}"
            )
    
    def _plan_joint_space(self, start_config: np.ndarray, goal_config: np.ndarray,
                         waypoint_count: int, **kwargs) -> MotionPlanningResult:
        """Plan motion in joint space using AORRTC with smart fallback."""
        # Clean interface: AORRTC primary + smart fallback
        use_fallback = kwargs.get('use_aorrtc', True)  # Default: use AORRTC with fallback
        max_iterations = kwargs.get('max_iterations', 3000)
        
        # Use clean path planner interface
        path_result = self.path_planner.plan_path(
            start_config, goal_config, 
            max_iterations=max_iterations,
            use_fallback=use_fallback
        )
        
        if not path_result.success:
            return MotionPlanningResult(
                status=PlanningStatus.CONSTRAINT_VIOLATION,
                error_message=path_result.error_message
            )
        
        # Generate trajectory (filter out path-specific parameters)
        traj_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['use_aorrtc', 'max_iterations', 'step_size']}
        traj_result = self.trajectory_planner.plan_trajectory(path_result.path, **traj_kwargs)
        
        if not traj_result.success:
            return MotionPlanningResult(
                status=PlanningStatus.FAILED,
                error_message=traj_result.error_message
            )
        
        # Create plan
        plan = MotionPlan(
            joint_waypoints=path_result.path,
            trajectory=traj_result.trajectory,
            strategy_used=PlanningStrategy.JOINT_SPACE,
            validation_results=path_result.validation_results
        )
        
        return MotionPlanningResult(
            status=PlanningStatus.SUCCESS,
            plan=plan
        )
    
    def _plan_cartesian_space(self, start_config: np.ndarray, goal_config: np.ndarray,
                            waypoint_count: int, **kwargs) -> MotionPlanningResult:
        """Plan motion in Cartesian space with IK solving."""
        try:
            # Get start and goal poses
            T_start = self.fk.compute_forward_kinematics(start_config)
            T_goal = self.fk.compute_forward_kinematics(goal_config)
            
            # Interpolate in Cartesian space
            cartesian_waypoints = self._interpolate_cartesian_path(T_start, T_goal, waypoint_count)
            
            # Solve IK for each waypoint
            joint_waypoints = [start_config]  # Start with given config
            
            for i, T_waypoint in enumerate(cartesian_waypoints[1:], 1):  # Skip first (same as start)
                q_solution, solution_valid = self.solve_constrained_ik(T_waypoint, max_attempts=3)
                
                if not solution_valid:
                    return MotionPlanningResult(
                        status=PlanningStatus.IK_CONSTRAINT_VIOLATION,
                        error_message=f"Failed to find constraint-satisfying IK solution for waypoint {i}"
                    )
                
                joint_waypoints.append(q_solution)
            
            # Validate joint path
            path_result = self.path_planner.validate_joint_path(joint_waypoints)
            
            if not path_result.success:
                return MotionPlanningResult(
                    status=PlanningStatus.CONSTRAINT_VIOLATION,
                    error_message=path_result.error_message
                )
            
            # Generate trajectory (filter out path-specific parameters)
            traj_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['use_aorrtc', 'max_iterations', 'step_size']}
            traj_result = self.trajectory_planner.plan_trajectory(joint_waypoints, **traj_kwargs)
            
            if not traj_result.success:
                return MotionPlanningResult(
                    status=PlanningStatus.FAILED,
                    error_message=traj_result.error_message
                )
            
            # Create plan
            plan = MotionPlan(
                joint_waypoints=joint_waypoints,
                trajectory=traj_result.trajectory,
                cartesian_waypoints=cartesian_waypoints,
                strategy_used=PlanningStrategy.CARTESIAN_SPACE,
                validation_results=path_result.validation_results
            )
            
            return MotionPlanningResult(
                status=PlanningStatus.SUCCESS,
                plan=plan
            )
            
        except Exception as e:
            return MotionPlanningResult(
                status=PlanningStatus.FAILED,
                error_message=f"Cartesian planning failed: {str(e)}"
            )
    
    def _plan_hybrid(self, start_config: np.ndarray, goal_config: np.ndarray,
                    waypoint_count: int, **kwargs) -> MotionPlanningResult:
        """Plan using hybrid approach (try both strategies)."""
        # Try joint space first
        joint_result = self._plan_joint_space(start_config, goal_config, waypoint_count, **kwargs)
        
        if joint_result.status == PlanningStatus.SUCCESS:
            joint_result.plan.strategy_used = PlanningStrategy.HYBRID
            return joint_result
        
        # Fall back to Cartesian space
        cartesian_result = self._plan_cartesian_space(start_config, goal_config, waypoint_count, **kwargs)
        
        if cartesian_result.status == PlanningStatus.SUCCESS:
            cartesian_result.plan.strategy_used = PlanningStrategy.HYBRID
            cartesian_result.fallback_used = True
            
        return cartesian_result
    
    def _interpolate_cartesian_path(self, T_start: np.ndarray, T_goal: np.ndarray,
                                  num_waypoints: int) -> List[np.ndarray]:
        """Interpolate between two transformation matrices."""
        waypoints = []
        
        # Extract positions and orientations
        pos_start = T_start[:3, 3]
        pos_goal = T_goal[:3, 3]
        
        # Linear interpolation for positions
        for i in range(num_waypoints):
            t = i / (num_waypoints - 1)
            pos_interp = (1 - t) * pos_start + t * pos_goal
            
            # Simple orientation interpolation (could be improved with SLERP)
            T_interp = T_start.copy()
            T_interp[:3, 3] = pos_interp
            waypoints.append(T_interp)
        
        return waypoints
    
    def _try_fallback_strategies(self, start_config: np.ndarray, goal_config: np.ndarray,
                               original_strategy: PlanningStrategy, waypoint_count: int,
                               previous_result: MotionPlanningResult,
                               **kwargs) -> MotionPlanningResult:
        """Try fallback planning strategies."""
        for fallback_strategy in self.config['fallback_strategies']:
            if fallback_strategy == original_strategy:
                continue  # Skip the strategy that already failed
            
            logger.info(f"Trying fallback strategy: {fallback_strategy}")
            
            fallback_result = self._attempt_planning(
                start_config, goal_config, fallback_strategy, waypoint_count, **kwargs
            )
            
            fallback_result.attempts_made = previous_result.attempts_made + 1
            
            if fallback_result.status == PlanningStatus.SUCCESS:
                fallback_result.fallback_used = True
                return fallback_result
        
        # All strategies failed
        previous_result.attempts_made += len(self.config['fallback_strategies'])
        return previous_result
    
    def _update_statistics(self, result: MotionPlanningResult, planning_time: float):
        """Update simple planning statistics."""
        if result.status == PlanningStatus.SUCCESS:
            self.stats['successful_plans'] += 1
        else:
            self.stats['failed_plans'] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get motion planning statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset planning statistics."""
        self.stats = {
            'total_plans': 0,
            'successful_plans': 0,
            'failed_plans': 0,
            'total_planning_time': 0.0,
            'average_planning_time': 0.0,
            'strategy_usage': {strategy: 0 for strategy in PlanningStrategy}
        }
    
    def solve_constrained_ik(self, target_pose: np.ndarray, 
                           max_attempts: Optional[int] = None,
                           use_different_seeds: bool = True) -> Tuple[Optional[np.ndarray], bool]:
        """
        Solve IK with constraint validation and intelligent retry.
        
        Tries multiple IK configurations to find a solution that satisfies both
        convergence and constraint requirements. Critical for robust planning
        when target poses have multiple reachable joint configurations.
        
        Args:
            target_pose: 4x4 target transformation matrix
            max_attempts: Maximum number of IK attempts with different seeds
            use_different_seeds: Whether to use different initial configurations
            
        Returns:
            Tuple of (joint solution, success flag)
        """
        max_attempts = max_attempts or self.config['ik_max_attempts']
        logger.debug(f"Attempting constraint-aware IK with {max_attempts} attempts")
        
        # First attempt with default parameters
        q_solution, converged = self.ik.solve(target_pose)
        if converged and self._validate_ik_solution(q_solution, target_pose):
            logger.debug("First IK attempt succeeded with valid constraints")
            return q_solution, True
        
        if not use_different_seeds or max_attempts <= 1:
            logger.warning("IK failed - no additional attempts configured")
            return None, False
            
        # Try with different initial configurations
        joint_limits = self.path_planner.get_joint_limits()
        limits_lower = np.array([joint_limits[f'j{i+1}']['min'] for i in range(6)])
        limits_upper = np.array([joint_limits[f'j{i+1}']['max'] for i in range(6)])
        
        for attempt in range(1, max_attempts):
            logger.debug(f"IK retry attempt {attempt + 1}/{max_attempts}")
            
            # Generate diverse initial configurations
            if attempt % 3 == 1:
                # Random configuration within joint limits
                q_init = np.random.uniform(limits_lower, limits_upper)
            elif attempt % 3 == 2:
                # Middle position with random perturbation
                q_init = (limits_lower + limits_upper) / 2
                q_init += np.random.normal(0, 0.3, size=q_init.shape)
                q_init = np.clip(q_init, limits_lower, limits_upper)
            else:
                # Configuration based on previous attempts
                q_init = np.random.uniform(limits_lower, limits_upper)
                # Bias toward successful regions if we had partial success
                if q_solution is not None:
                    weight = 0.3
                    q_init = weight * q_solution + (1 - weight) * q_init
                    q_init = np.clip(q_init, limits_lower, limits_upper)
            
            # Attempt IK with this initial configuration
            q_candidate, converged = self.ik.solve(target_pose, q_init=q_init)
            
            if converged and self._validate_ik_solution(q_candidate, target_pose):
                logger.info(f"Constraint-aware IK succeeded on attempt {attempt + 1}")
                return q_candidate, True
            
            # Keep best solution so far for potential fallback
            if converged and q_solution is None:
                q_solution = q_candidate
        
        logger.warning(f"All {max_attempts} IK attempts failed constraint validation")
        return None, False
    
    def _validate_ik_solution(self, q_solution: np.ndarray, 
                            target_pose: np.ndarray) -> bool:
        """
        Enhanced IK solution validation with comprehensive collision checking.
        
        Args:
            q_solution: Joint configuration to validate
            target_pose: Original target pose for accuracy checking
            
        Returns:
            True if solution is valid and collision-free
        """
        try:
            # Verify forward kinematics accuracy
            T_achieved = self.fk.compute_forward_kinematics(q_solution)
            tcp_position = T_achieved[:3, 3]
            
            # Position error check
            pos_error = np.linalg.norm(tcp_position - target_pose[:3, 3])
            pos_tolerance = self.config['ik_position_tolerance']
            if pos_error > pos_tolerance:
                logger.debug(f"IK solution has large position error: {pos_error:.6f}m > {pos_tolerance}m")
                return False
            
            # Orientation error check
            R_desired = target_pose[:3, :3]
            R_achieved = T_achieved[:3, :3]
            
            cos_angle = (np.trace(R_desired.T @ R_achieved) - 1) / 2
            cos_angle = np.clip(cos_angle, -1, 1)
            rot_error = np.arccos(cos_angle)
            
            rot_tolerance = np.radians(self.config['ik_rotation_tolerance'])
            if rot_error > rot_tolerance:
                logger.debug(f"IK solution has large rotation error: {np.degrees(rot_error):.3f}° > {self.config['ik_rotation_tolerance']}°")
                return False
            
            # Enhanced collision checking
            joint_positions = [tcp_position]  # Simplified - would need full link positions for complete check
            
            collision_result = self.collision_checker.check_configuration_collision(
                q_solution, tcp_position, joint_positions
            )
            
            if collision_result.is_collision:
                logger.debug(f"IK solution has collision: {collision_result.details}")
                return False
            
            logger.debug(f"IK solution validated - pos_err: {pos_error:.6f}m, rot_err: {np.degrees(rot_error):.3f}°")
            return True
            
        except Exception as e:
            logger.error(f"IK solution validation failed: {e}")
            return False
    
    def validate_motion_path(self, joint_path: List[np.ndarray]) -> Tuple[bool, str]:
        """
        Validate entire motion path for collisions including intermediate points.
        
        Args:
            joint_path: List of joint configurations defining the path
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check each waypoint for collision
            for i, q in enumerate(joint_path):
                T = self.fk.compute_forward_kinematics(q)
                tcp_pos = T[:3, 3]
                joint_positions = [tcp_pos]  # Simplified
                
                collision_result = self.collision_checker.check_configuration_collision(
                    q, tcp_pos, joint_positions
                )
                
                if collision_result.is_collision:
                    return False, f"Waypoint {i} collision: {collision_result.details}"
            
            # Check path between waypoints for intermediate collisions
            collision_result = self.collision_checker.check_path_collision(
                joint_path, self.fk.compute_forward_kinematics
            )
            
            if collision_result.is_collision:
                return False, f"Path collision: {collision_result.details}"
            
            return True, "Path validation successful"
            
        except Exception as e:
            return False, f"Path validation failed: {str(e)}"
    
    def get_collision_info(self) -> Dict[str, Any]:
        """Get collision checker configuration and status."""
        return self.collision_checker.get_collision_summary()

    def update_config(self, new_config: Dict[str, Any]):
        """Update motion planning configuration."""
        self.config.update(new_config)
        logger.info("Motion planner configuration updated")
