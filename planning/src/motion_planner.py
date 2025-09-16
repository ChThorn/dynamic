#!/usr/bin/env python3
"""
Motion Planning Module - Production Optimized

Clean motion planning with optional C-space integration for better IK performance.
Optimized for production use with essential functionality only.

Author: Robot Control Team  
"""

import numpy as np
import logging
import time
import os
import threading
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Import collision checker
try:
    from .collision_checker import EnhancedCollisionChecker, CollisionResult, CollisionType
except ImportError:
    from collision_checker import EnhancedCollisionChecker, CollisionResult, CollisionType

# Import configuration space analyzer
try:
    from .configuration_space_analyzer import ConfigurationSpaceAnalyzer
except ImportError:
    try:
        from configuration_space_analyzer import ConfigurationSpaceAnalyzer
    except ImportError:
        ConfigurationSpaceAnalyzer = None
        logger.debug("ConfigurationSpaceAnalyzer not available")

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
    """Container for a planned motion trajectory."""
    joint_waypoints: List[np.ndarray]
    cartesian_waypoints: Optional[List[np.ndarray]] = None
    velocity_profile: Optional[np.ndarray] = None
    acceleration_profile: Optional[np.ndarray] = None
    strategy_used: Optional[PlanningStrategy] = None
    planning_time: float = 0.0
    trajectory: Optional[Any] = None
    validation_results: Optional[Dict[str, Any]] = None
    
    @property
    def num_waypoints(self) -> int:
        """Get number of waypoints in the plan."""
        return len(self.joint_waypoints) if self.joint_waypoints else 0

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
    """Production motion planner with strategy selection and optional C-space optimization."""
    
    def __init__(self, kinematics_fk, kinematics_ik, 
                 path_planner=None, trajectory_planner=None):
        """
        Initialize motion planner.
        
        Args:
            kinematics_fk: ForwardKinematics instance
            kinematics_ik: InverseKinematics instance  
            path_planner: PathPlanner instance (created if None)
            trajectory_planner: TrajectoryPlanner instance (created if None)
        """
        self.fk = kinematics_fk
        self.ik = kinematics_ik
        
        # Initialize planners
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
            
        # Initialize collision checker
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'constraints.yaml')
        self.collision_checker = EnhancedCollisionChecker(config_path)
        
        # Configuration
        self.config = {
            'default_strategy': PlanningStrategy.JOINT_SPACE,
            'max_planning_time': 30.0,
            'default_waypoint_count': 10,
            'ik_max_attempts': 5,
            'ik_position_tolerance': 0.002,  # 2mm
            'ik_rotation_tolerance': 2.0,     # 2 degrees
            'enable_fallbacks': True,
            'max_attempts': 3,
            'fallback_strategies': [PlanningStrategy.CARTESIAN_SPACE, PlanningStrategy.HYBRID],
            'progress_feedback': False,  # Enable for progress updates
            'enable_timing_breakdown': False  # Enable for performance profiling
        }
        
        # Statistics
        self.stats = {
            'total_plans': 0,
            'successful_plans': 0,
            'failed_plans': 0
        }
        
        # Optional C-space analyzer
        self.config_analyzer = None
        self.cspace_analysis_enabled = False
        
        # Progress callback
        self.progress_callback = None
        
        # Thread safety
        self._planning_lock = threading.RLock()
        self._stats_lock = threading.RLock()
        self._config_lock = threading.RLock()
    
        logger.info("Motion planner initialized with thread safety")
    
    def enable_configuration_space_analysis(self, build_maps=False):
        """Enable C-space analysis for better IK performance."""
        if ConfigurationSpaceAnalyzer is None:
            logger.warning("ConfigurationSpaceAnalyzer not available")
            return
            
        logger.info("Enabling configuration space analysis")
        self.config_analyzer = ConfigurationSpaceAnalyzer(self.fk, self.ik)
        self.cspace_analysis_enabled = True
        
        # Load or build reachability map
        cache_path = os.path.join(os.path.dirname(__file__), '../cache/production_reachability_map.pkl')
        if os.path.exists(cache_path) and not build_maps:
            if self.config_analyzer.load_reachability_map(cache_path):
                logger.info("Loaded cached reachability map")
            else:
                build_maps = True
        
        if build_maps:
            logger.info("Building reachability maps...")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            self.config_analyzer.build_reachability_map(
                workspace_samples=500, c_space_samples=2000, save_path=cache_path)
            logger.info("Reachability maps built and cached")
    
    def solve_ik_with_cspace(self, target_pose: np.ndarray, q_current: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], bool]:
        """Solve IK with C-space optimization if available, with multiple initial guess fallbacks."""
        if self.cspace_analysis_enabled and self.config_analyzer:
            # Get optimal seed from C-space analysis
            target_position = target_pose[:3, 3]
            q_seed = self.config_analyzer.get_best_ik_region(target_position)
            if q_seed is not None:
                q_solution, converged = self.ik.solve(target_pose, q_init=q_seed)
                if converged:
                    return q_solution, True
        
        # Try multiple initial guesses for better success rate
        initial_guesses = [
            q_current if q_current is not None else np.zeros(6),  # Current or zero
            np.array([0, -np.pi/2, np.pi/2, 0, np.pi/2, 0]),     # Home position
            np.array([np.pi/4, -np.pi/3, np.pi/3, 0, np.pi/3, np.pi/4]),  # Alternative 1
            np.array([-np.pi/4, -np.pi/4, np.pi/4, 0, np.pi/4, -np.pi/4]), # Alternative 2
            np.array([0, 0, 0, 0, 0, 0]),                         # Zero position
            np.array([np.pi/6, -np.pi/4, np.pi/4, np.pi/6, np.pi/4, 0]),  # Conservative pose
            np.random.uniform(-np.pi, np.pi, 6),                  # Random guess 1
            np.random.uniform(-np.pi, np.pi, 6),                  # Random guess 2
            np.random.uniform(-np.pi, np.pi, 6),                  # Random guess 3
        ]
        
        for i, q_init in enumerate(initial_guesses):
            try:
                q_solution, converged = self.ik.solve(target_pose, q_init=q_init)
                if converged:
                    if i > 0:  # Log when fallback was needed
                        logger.info(f"IK converged using initial guess #{i+1}")
                    return q_solution, True
            except Exception as e:
                logger.debug(f"IK attempt {i+1} failed: {e}")
                continue
        
        # All attempts failed
        return None, False
    
    def plan_motion(self, start_config: np.ndarray, goal_config: np.ndarray,
                   strategy: Optional[PlanningStrategy] = None,
                   waypoint_count: Optional[int] = None,
                   **kwargs) -> MotionPlanningResult:
        """
        Plan motion between joint configurations. Thread-safe.
        
        Args:
            start_config: Starting joint configuration
            goal_config: Goal joint configuration
            strategy: Planning strategy to use
            waypoint_count: Number of waypoints to generate
            **kwargs: Additional planning parameters
            
        Returns:
            MotionPlanningResult with planned motion
        """
        with self._planning_lock:
            start_time = time.time()
            with self._stats_lock:
                self.stats['total_plans'] += 1
        
            # Set default parameters
            with self._config_lock:
                strategy = strategy or self.config['default_strategy']
                waypoint_count = waypoint_count or self.config['default_waypoint_count']
        
            try:
                # Validate inputs
                with self._config_lock:
                    progress_enabled = self.config['progress_feedback']
                    
                if self.progress_callback and progress_enabled:
                    self.progress_callback(10.0, "Validating configuration constraints")
            
                validation_result = self._validate_planning_inputs(start_config, goal_config)
                if not validation_result['valid']:
                    return MotionPlanningResult(
                        status=PlanningStatus.CONSTRAINT_VIOLATION,
                        error_message=validation_result['error'],
                        planning_time=time.time() - start_time
                    )
            
                # Attempt planning with primary strategy
                if self.progress_callback and progress_enabled:
                    self.progress_callback(30.0, f"Planning motion using {strategy.value} strategy")
            
                result = self._attempt_planning(
                    start_config, goal_config, strategy, waypoint_count, **kwargs
                )
            
                # Try fallback strategies if primary failed
                with self._config_lock:
                    enable_fallbacks = self.config['enable_fallbacks']
                    max_attempts = self.config['max_attempts']
                    
                if (not result.status == PlanningStatus.SUCCESS and 
                    enable_fallbacks and
                    result.attempts_made < max_attempts):
                
                    if self.progress_callback and progress_enabled:
                        self.progress_callback(60.0, "Trying fallback strategies")
                
                    result = self._try_fallback_strategies(
                        start_config, goal_config, strategy, waypoint_count, 
                        result, **kwargs
                    )
            
                # Update statistics
                if self.progress_callback and progress_enabled:
                    self.progress_callback(90.0, "Finalizing motion plan")
                
                planning_time = time.time() - start_time
                self._update_statistics(result)
                result.planning_time = planning_time
            
                if result.plan:
                    result.plan.planning_time = planning_time
            
                if self.progress_callback and progress_enabled:
                    status_msg = "Motion planning completed" if result.status == PlanningStatus.SUCCESS else "Motion planning failed"
                    self.progress_callback(100.0, status_msg)
            
                return result
                
            except Exception as e:
                logger.error(f"Motion planning failed with exception: {e}")
                return MotionPlanningResult(
                    status=PlanningStatus.FAILED,
                    error_message=f"Planning exception: {str(e)}",
                    planning_time=time.time() - start_time
                )
    
    def plan_cartesian_motion(self, start_pose: np.ndarray, goal_pose: np.ndarray,
                            current_joints: Optional[np.ndarray] = None,
                            orientation_constraint: bool = True,
                            **kwargs) -> MotionPlanningResult:
        """
        Plan motion between Cartesian poses following proper robot motion planning procedure.
        
        Args:
            start_pose: Starting 4x4 transformation matrix
            goal_pose: Goal 4x4 transformation matrix
            current_joints: Current robot joint configuration (if known)
            orientation_constraint: Whether to maintain orientation constraints
            **kwargs: Additional planning parameters
            
        Returns:
            MotionPlanningResult with planned motion
        """
        with self._planning_lock:
            start_time = time.time()
        
            try:
                # STEP 1: Determine initial guess based on robot state
                if current_joints is not None:
                    # Use current robot state as initial guess (proper robot procedure)
                    q_init_start = current_joints
                    logger.info("Using current robot joints as initial guess for start pose")
                else:
                    # Fallback: use home position for unknown robot state
                    q_init_start = np.array([0, -np.pi/2, np.pi/2, 0, np.pi/2, 0])  # Home position
                    logger.info("Using home position as initial guess (current robot state unknown)")
                
                # STEP 2: Solve IK for start pose from current/initial state
                q_start, start_converged = self.ik.solve(start_pose, q_init=q_init_start)
                if not start_converged:
                    # Try fallback initial guesses only if primary method fails
                    logger.warning("Primary IK for start pose failed, trying fallback initial guesses")
                    q_start, start_converged = self.solve_ik_with_cspace(start_pose)
                    if not start_converged:
                        return MotionPlanningResult(
                            status=PlanningStatus.IK_FAILED,
                            error_message="Failed to solve IK for start pose",
                            planning_time=time.time() - start_time
                        )
                
                # STEP 3: Solve IK for goal pose using start pose joints as initial guess (continuity!)
                q_goal, goal_converged = self.ik.solve(goal_pose, q_init=q_start)
                if not goal_converged:
                    # Try fallback initial guesses only if continuous method fails
                    logger.warning("Continuous IK for goal pose failed, trying fallback initial guesses")
                    q_goal, goal_converged = self.solve_ik_with_cspace(goal_pose, q_current=q_start)
                    if not goal_converged:
                        # Final fallback: try with multiple random seeds
                        logger.warning("All standard IK methods failed, trying aggressive fallback")
                        for attempt in range(10):
                            q_random = np.random.uniform(-np.pi, np.pi, 6)
                            q_try, converged = self.ik.solve(goal_pose, q_init=q_random)
                            if converged:
                                q_goal = q_try
                                goal_converged = True
                                logger.info(f"IK converged using random seed attempt #{attempt+1}")
                                break
                        
                        if not goal_converged:
                            return MotionPlanningResult(
                                status=PlanningStatus.IK_FAILED,
                                error_message="Failed to solve IK for goal pose after all attempts",
                                planning_time=time.time() - start_time
                            )
                
                # STEP 4: Validate joint space continuity
                joint_diff = np.linalg.norm(q_goal - q_start)
                if joint_diff > np.pi:  # Large joint space jump
                    logger.warning(f"Large joint space jump detected: {joint_diff:.3f} rad")
                
                # STEP 5: Plan smooth motion in joint space
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
    
    def plan_sequential_cartesian_motion(self, poses: List[np.ndarray], 
                                       current_joints: Optional[np.ndarray] = None,
                                       **kwargs) -> MotionPlanningResult:
        """
        Plan sequential motion through multiple Cartesian poses (proper robot procedure).
        
        This method follows real robot motion planning procedure:
        1. Uses current robot state as starting point
        2. Solves each IK using previous solution as initial guess (continuity)
        3. Plans smooth trajectories between sequential joint configurations
        
        Args:
            poses: List of 4x4 transformation matrices to visit sequentially
            current_joints: Current robot joint configuration (if known)
            **kwargs: Additional planning parameters
            
        Returns:
            MotionPlanningResult with complete sequential motion plan
        """
        if len(poses) < 2:
            return MotionPlanningResult(
                status=PlanningStatus.FAILED,
                error_message="Need at least 2 poses for sequential motion",
                planning_time=0.0
            )
        
        start_time = time.time()
        all_waypoints = []
        all_cartesian_waypoints = []
        
        try:
            # Initialize with current robot state or home position
            if current_joints is not None:
                q_current = current_joints.copy()
                logger.info("Starting sequential motion from current robot joints")
            else:
                q_current = np.array([0, -np.pi/2, np.pi/2, 0, np.pi/2, 0])  # Home position
                logger.info("Starting sequential motion from home position")
            
            # Process each pose sequentially
            for i, target_pose in enumerate(poses):
                logger.info(f"Planning to pose {i+1}/{len(poses)}")
                
                # Solve IK using previous joint configuration as initial guess
                q_target, converged = self.ik.solve(target_pose, q_init=q_current)
                
                if not converged:
                    # Fallback to multiple initial guesses only if needed
                    logger.warning(f"Sequential IK failed for pose {i+1}, trying fallback methods")
                    q_target, converged = self.solve_ik_with_cspace(target_pose)
                    
                    if not converged:
                        return MotionPlanningResult(
                            status=PlanningStatus.IK_FAILED,
                            error_message=f"Failed to solve IK for pose {i+1}",
                            planning_time=time.time() - start_time
                        )
                
                # Plan motion from current to target
                segment_result = self.plan_motion(q_current, q_target, **kwargs)
                
                if segment_result.status != PlanningStatus.SUCCESS:
                    return MotionPlanningResult(
                        status=segment_result.status,
                        error_message=f"Failed to plan segment {i+1}: {segment_result.error_message}",
                        planning_time=time.time() - start_time
                    )
                
                # Accumulate waypoints
                if segment_result.plan and segment_result.plan.joint_waypoints:
                    if i == 0:
                        # Include all waypoints for first segment
                        all_waypoints.extend(segment_result.plan.joint_waypoints)
                    else:
                        # Skip first waypoint for subsequent segments to avoid duplication
                        all_waypoints.extend(segment_result.plan.joint_waypoints[1:])
                
                # Generate Cartesian waypoints
                for q in (segment_result.plan.joint_waypoints if segment_result.plan else []):
                    T = self.fk.compute_forward_kinematics(q)
                    all_cartesian_waypoints.append(T)
                
                # Update current position for next iteration
                q_current = q_target.copy()
                
                logger.info(f"Pose {i+1} reached, joint difference: {np.linalg.norm(q_target - q_current):.3f} rad")
            
            # Create combined result
            combined_plan = MotionPlan(
                strategy_used=PlanningStrategy.CARTESIAN_SPACE,
                joint_waypoints=all_waypoints,
                cartesian_waypoints=all_cartesian_waypoints,
                planning_time=time.time() - start_time
            )
            
            return MotionPlanningResult(
                status=PlanningStatus.SUCCESS,
                plan=combined_plan,
                planning_time=time.time() - start_time,
                error_message=None
            )
            
        except Exception as e:
            return MotionPlanningResult(
                status=PlanningStatus.FAILED,
                error_message=f"Sequential motion planning failed: {str(e)}",
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
            
            # Generate trajectory
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
    
    def _validate_planning_inputs(self, start_config: np.ndarray, goal_config: np.ndarray) -> Dict[str, Any]:
        """Validate planning inputs."""
        # Validate start configuration
        start_validation = self._validate_joint_config(start_config)
        if not start_validation['valid']:
            return {'valid': False, 'error': f"Start config invalid: {start_validation['error']}"}
        
        # Validate goal configuration
        goal_validation = self._validate_joint_config(goal_config)
        if not goal_validation['valid']:
            return {'valid': False, 'error': f"Goal config invalid: {goal_validation['error']}"}
        
        return {'valid': True}
    
    def _validate_joint_config(self, q: np.ndarray) -> Dict[str, Any]:
        """Validate a joint configuration."""
        try:
            # Check joint limits
            joint_limits = self.path_planner.get_joint_limits()
            for i in range(len(q)):
                joint_key = f'j{i+1}'
                if joint_key in joint_limits:
                    min_limit = joint_limits[joint_key]['min']
                    max_limit = joint_limits[joint_key]['max']
                    
                    if q[i] < min_limit or q[i] > max_limit:
                        return {
                            'valid': False,
                            'error': f"Joint {i+1} value {q[i]:.3f} outside limits [{min_limit:.3f}, {max_limit:.3f}]"
                        }
            
            # Check for collisions
            T = self.fk.compute_forward_kinematics(q)
            tcp_position = T[:3, 3]
            
            collision_result = self.collision_checker.check_configuration_collision(
                q, tcp_position, self.fk.compute_forward_kinematics
            )
            
            if collision_result.is_collision:
                return {
                    'valid': False,
                    'error': f"Configuration has collision: {collision_result.details}"
                }
            
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
        """Plan motion in joint space with optional timing breakdown."""
        try:
            timing_breakdown = {}
            timing_enabled = self.config.get('enable_timing_breakdown', False)
            
            # Path planning phase
            path_start_time = time.time()
            path_result = self.path_planner.plan_path(
                start_config, goal_config, max_iterations=waypoint_count*50, **kwargs
            )
            if timing_enabled:
                timing_breakdown['path_planning'] = time.time() - path_start_time
            
            if not path_result.success:
                result = MotionPlanningResult(
                    status=PlanningStatus.FAILED,
                    error_message=path_result.error_message
                )
                if timing_enabled:
                    result.planning_time = timing_breakdown['path_planning']
                return result
            
            # Trajectory generation phase
            traj_start_time = time.time()
            traj_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['use_aorrtc', 'max_iterations', 'step_size']}
            traj_result = self.trajectory_planner.plan_trajectory(path_result.path, **traj_kwargs)
            if timing_enabled:
                timing_breakdown['trajectory_planning'] = time.time() - traj_start_time
            
            if not traj_result.success:
                result = MotionPlanningResult(
                    status=PlanningStatus.FAILED,
                    error_message=traj_result.error_message
                )
                if timing_enabled:
                    result.planning_time = sum(timing_breakdown.values())
                return result
            
            # Create plan
            plan = MotionPlan(
                joint_waypoints=path_result.path,
                trajectory=traj_result.trajectory,
                strategy_used=PlanningStrategy.JOINT_SPACE,
                validation_results=path_result.validation_results
            )
            
            # Add timing breakdown to validation results if enabled
            if timing_enabled:
                plan.validation_results = plan.validation_results or {}
                plan.validation_results['timing_breakdown'] = timing_breakdown
                logger.info(f"Joint space planning breakdown: {timing_breakdown}")
            
            return MotionPlanningResult(
                status=PlanningStatus.SUCCESS,
                plan=plan
            )
            
        except Exception as e:
            return MotionPlanningResult(
                status=PlanningStatus.FAILED,
                error_message=f"Joint space planning failed: {str(e)}"
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
            
            # Solve IK for each waypoint using C-space optimization
            joint_waypoints = [start_config]
            
            for i, T_waypoint in enumerate(cartesian_waypoints[1:], 1):
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
            
            # Generate trajectory
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
        
        # Extract positions
        pos_start = T_start[:3, 3]
        pos_goal = T_goal[:3, 3]
        
        # Linear interpolation for positions
        for i in range(num_waypoints):
            t = i / (num_waypoints - 1)
            pos_interp = (1 - t) * pos_start + t * pos_goal
            
            # Simple orientation interpolation
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
                continue
            
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
    
    def _update_statistics(self, result: MotionPlanningResult):
        """Update planning statistics. Thread-safe."""
        with self._stats_lock:
            if result.status == PlanningStatus.SUCCESS:
                self.stats['successful_plans'] += 1
            else:
                self.stats['failed_plans'] += 1
    
    def solve_constrained_ik(self, target_pose: np.ndarray, 
                           max_attempts: Optional[int] = None,
                           use_different_seeds: bool = True) -> Tuple[Optional[np.ndarray], bool]:
        """
        Solve IK with constraint validation and intelligent retry.
        
        Args:
            target_pose: 4x4 target transformation matrix
            max_attempts: Maximum number of IK attempts with different seeds
            use_different_seeds: Whether to use different initial configurations
            
        Returns:
            Tuple of (joint solution, success flag)
        """
        max_attempts = max_attempts or self.config['ik_max_attempts']
        
        # First attempt with C-space optimization
        q_solution, converged = self.solve_ik_with_cspace(target_pose)
        if converged and self._validate_ik_solution(q_solution, target_pose):
            return q_solution, True
        
        if not use_different_seeds or max_attempts <= 1:
            return None, False
            
        # Try with different initial configurations
        joint_limits = self.path_planner.get_joint_limits()
        limits_lower = np.array([joint_limits[f'j{i+1}']['min'] for i in range(6)])
        limits_upper = np.array([joint_limits[f'j{i+1}']['max'] for i in range(6)])
        
        for attempt in range(1, max_attempts):
            # Generate diverse initial configurations
            if attempt % 3 == 1:
                q_init = np.random.uniform(limits_lower, limits_upper)
            elif attempt % 3 == 2:
                q_init = (limits_lower + limits_upper) / 2
                q_init += np.random.normal(0, 0.3, size=q_init.shape)
                q_init = np.clip(q_init, limits_lower, limits_upper)
            else:
                q_init = np.random.uniform(limits_lower, limits_upper)
                if q_solution is not None:
                    weight = 0.3
                    q_init = weight * q_solution + (1 - weight) * q_init
                    q_init = np.clip(q_init, limits_lower, limits_upper)
            
            # Attempt IK with this initial configuration
            q_candidate, converged = self.ik.solve(target_pose, q_init=q_init)
            
            if converged and self._validate_ik_solution(q_candidate, target_pose):
                return q_candidate, True
            
            if converged and q_solution is None:
                q_solution = q_candidate
        
        return None, False
    
    def _validate_ik_solution(self, q_solution: np.ndarray, 
                            target_pose: np.ndarray) -> bool:
        """Validate IK solution for accuracy and collision-free status."""
        try:
            # Verify forward kinematics accuracy
            T_achieved = self.fk.compute_forward_kinematics(q_solution)
            tcp_position = T_achieved[:3, 3]
            
            # Position error check
            pos_error = np.linalg.norm(tcp_position - target_pose[:3, 3])
            if pos_error > self.config['ik_position_tolerance']:
                return False
            
            # Orientation error check
            R_desired = target_pose[:3, :3]
            R_achieved = T_achieved[:3, :3]
            
            cos_angle = (np.trace(R_desired.T @ R_achieved) - 1) / 2
            cos_angle = np.clip(cos_angle, -1, 1)
            rot_error = np.arccos(cos_angle)
            
            rot_tolerance = np.radians(self.config['ik_rotation_tolerance'])
            if rot_error > rot_tolerance:
                return False
            
            # Collision checking
            collision_result = self.collision_checker.check_configuration_collision(
                q_solution, tcp_position, self.fk.compute_forward_kinematics
            )
            
            if collision_result.is_collision:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"IK solution validation failed: {e}")
            return False
    
    def validate_motion_path(self, joint_path: List[np.ndarray]) -> Tuple[bool, str]:
        """Validate entire motion path for collisions."""
        try:
            # Check each waypoint for collision
            for i, q in enumerate(joint_path):
                T = self.fk.compute_forward_kinematics(q)
                tcp_pos = T[:3, 3]
                
                collision_result = self.collision_checker.check_configuration_collision(
                    q, tcp_pos, self.fk.compute_forward_kinematics
                )
                
                if collision_result.is_collision:
                    return False, f"Waypoint {i} collision: {collision_result.details}"
            
            # Check path between waypoints
            collision_result = self.collision_checker.check_path_collision(
                joint_path, self.fk.compute_forward_kinematics
            )
            
            if collision_result.is_collision:
                return False, f"Path collision: {collision_result.details}"
            
            return True, "Path validation successful"
            
        except Exception as e:
            return False, f"Path validation failed: {str(e)}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get motion planning statistics. Thread-safe."""
        with self._stats_lock:
            return self.stats.copy()
    
    def get_collision_info(self) -> Dict[str, Any]:
        """Get collision checker configuration and status."""
        return self.collision_checker.get_collision_summary()

    def update_config(self, new_config: Dict[str, Any]):
        """Update motion planning configuration. Thread-safe."""
        with self._config_lock:
            self.config.update(new_config)
        logger.info("Motion planner configuration updated")

    def set_progress_callback(self, callback):
        """Set progress callback for motion planning operations. Thread-safe."""
        with self._config_lock:
            self.progress_callback = callback
            self.config['progress_feedback'] = True
        logger.info("Progress feedback enabled")

    def enable_progress_feedback(self, enable: bool = True):
        """Enable or disable progress feedback. Thread-safe."""
        with self._config_lock:
            self.config['progress_feedback'] = enable
        if enable:
            logger.info("Progress feedback enabled")
        else:
            logger.info("Progress feedback disabled")
    
    def enable_fast_mode(self, enable: bool = True):
        """Enable fast mode for real-time robot operation."""
        self.path_planner.enable_fast_mode(enable)
        if enable:
            logger.info("Motion planner fast mode enabled for real-time operation")
        else:
            logger.info("Motion planner standard mode enabled")
    
    def enable_timing_breakdown(self, enable: bool = True):
        """Enable detailed timing breakdown for performance analysis."""
        with self._config_lock:
            self.config['enable_timing_breakdown'] = enable
        if enable:
            logger.info("Timing breakdown enabled for performance profiling")
        else:
            logger.info("Timing breakdown disabled")
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get current performance configuration."""
        path_stats = self.path_planner.get_planning_stats()
        with self._config_lock:
            config_snapshot = self.config.copy()
        
        return {
            'motion_planner': {
                'timing_breakdown_enabled': config_snapshot.get('enable_timing_breakdown', False),
                'progress_feedback_enabled': config_snapshot.get('progress_feedback', False),
                'max_planning_time': config_snapshot.get('max_planning_time', 30.0)
            },
            'path_planner': path_stats
        }

