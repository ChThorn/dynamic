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
        
        # Production features
        self.production_mode = self.config.get('production_mode', False)
        self.validation_mode = self.config.get('validation_mode', False)
        self.detailed_logging = self.config.get('detailed_logging', False)
        
        # Error reporting and diagnostics
        self.error_history = []
        self.max_error_history = 100
        self.diagnostic_data = {
            'last_successful_plan': None,
            'consecutive_failures': 0,
            'failure_patterns': {},
            'performance_metrics': []
        }
    
        logger.info("Motion planner initialized with thread safety")
    
    def _log_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None):
        """Log error with production features."""
        error_data = {
            'timestamp': time.time(),
            'type': error_type,
            'message': error_message,
            'context': context or {}
        }
        
        # Add to error history
        self.error_history.append(error_data)
        if len(self.error_history) > self.max_error_history:
            self.error_history.pop(0)
        
        # Update failure patterns
        self.diagnostic_data['consecutive_failures'] += 1
        if error_type not in self.diagnostic_data['failure_patterns']:
            self.diagnostic_data['failure_patterns'][error_type] = 0
        self.diagnostic_data['failure_patterns'][error_type] += 1
        
        if self.detailed_logging:
            logger.error(f"Planning error [{error_type}]: {error_message}")
            if context:
                logger.debug(f"Error context: {context}")
    
    def _log_success(self, plan: MotionPlan, planning_time: float):
        """Log successful planning with production features."""
        self.diagnostic_data['last_successful_plan'] = {
            'timestamp': time.time(),
            'planning_time': planning_time,
            'strategy': plan.strategy_used.value if plan.strategy_used else 'unknown',
            'waypoints': plan.num_waypoints
        }
        
        # Reset consecutive failures
        self.diagnostic_data['consecutive_failures'] = 0
        
        # Add performance metric
        self.diagnostic_data['performance_metrics'].append({
            'timestamp': time.time(),
            'planning_time': planning_time,
            'strategy': plan.strategy_used.value if plan.strategy_used else 'unknown'
        })
        
        # Keep only recent metrics
        if len(self.diagnostic_data['performance_metrics']) > 100:
            self.diagnostic_data['performance_metrics'].pop(0)
        
        if self.detailed_logging:
            logger.info(f"Planning successful in {planning_time:.3f}s using {plan.strategy_used.value if plan.strategy_used else 'unknown'}")
    
    def get_diagnostic_report(self) -> Dict[str, Any]:
        """Get comprehensive diagnostic report for production monitoring."""
        with self._stats_lock:
            # Calculate success rate
            total_plans = self.stats['total_plans']
            success_rate = (self.stats['successful_plans'] / total_plans) if total_plans > 0 else 0.0
            
            # Calculate average planning time
            if self.diagnostic_data['performance_metrics']:
                recent_times = [m['planning_time'] for m in self.diagnostic_data['performance_metrics'][-20:]]
                avg_planning_time = sum(recent_times) / len(recent_times)
            else:
                avg_planning_time = 0.0
            
            # Get failure patterns
            total_failures = sum(self.diagnostic_data['failure_patterns'].values())
            failure_rates = {}
            if total_failures > 0:
                for error_type, count in self.diagnostic_data['failure_patterns'].items():
                    failure_rates[error_type] = count / total_failures
            
            return {
                'production_mode': self.production_mode,
                'validation_mode': self.validation_mode,
                'planning_statistics': self.stats.copy(),
                'success_rate': success_rate,
                'average_planning_time': avg_planning_time,
                'consecutive_failures': self.diagnostic_data['consecutive_failures'],
                'failure_patterns': self.diagnostic_data['failure_patterns'].copy(),
                'failure_rates': failure_rates,
                'last_successful_plan': self.diagnostic_data['last_successful_plan'],
                'recent_errors': self.error_history[-5:] if self.error_history else [],
                'error_history_count': len(self.error_history),
                'cspace_analysis_enabled': self.cspace_analysis_enabled,
                'timestamp': time.time()
            }
    
    def enable_production_mode(self, detailed_logging: bool = True, validation_mode: bool = False):
        """Enable production mode with enhanced monitoring and error reporting."""
        with self._config_lock:
            self.production_mode = True
            self.detailed_logging = detailed_logging
            self.validation_mode = validation_mode
            
            # Update configuration
            self.config.update({
                'production_mode': True,
                'detailed_logging': detailed_logging,
                'validation_mode': validation_mode,
                'enable_fallbacks': True,  # Always enable fallbacks in production
                'progress_feedback': True,  # Enable progress feedback
                'enable_timing_breakdown': True  # Enable performance profiling
            })
        
        logger.info(f"Production mode enabled: logging={detailed_logging}, validation={validation_mode}")
    
    def disable_production_mode(self):
        """Disable production mode and return to standard operation."""
        with self._config_lock:
            self.production_mode = False
            self.detailed_logging = False
            self.validation_mode = False
            
            # Update configuration
            self.config.update({
                'production_mode': False,
                'detailed_logging': False,
                'validation_mode': False
            })
        
        logger.info("Production mode disabled")
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate current configuration for production readiness."""
        issues = []
        warnings = []
        
        # Check critical configuration values
        if self.config.get('max_planning_time', 0) <= 0:
            issues.append("max_planning_time must be positive")
        elif self.config.get('max_planning_time', 0) > 60:
            warnings.append("max_planning_time is very high (>60s)")
        
        if self.config.get('ik_position_tolerance', 0) <= 0:
            issues.append("ik_position_tolerance must be positive")
        elif self.config.get('ik_position_tolerance', 0) > 0.01:
            warnings.append("ik_position_tolerance is high (>10mm)")
        
        if not self.config.get('enable_fallbacks', False) and self.production_mode:
            warnings.append("Fallback strategies disabled in production mode")
        
        # Check collision checker
        collision_issues = []
        try:
            test_config = np.zeros(6)
            test_tcp = np.array([0, 0, 0.8])
            result = self.collision_checker.check_configuration_collision(test_config, test_tcp)
            if result.is_collision:
                collision_issues.append("Home position incorrectly flagged as collision")
        except Exception as e:
            collision_issues.append(f"Collision checker error: {e}")
        
        # Check kinematic solvers
        ik_issues = []
        try:
            test_pose = np.eye(4)
            test_pose[:3, 3] = [0.4, 0.0, 0.5]
            q_solution, ik_success = self.ik.solve(test_pose)
            if not ik_success:
                ik_issues.append("Basic IK test failed")
        except Exception as e:
            ik_issues.append(f"IK solver error: {e}")
        
        return {
            'configuration_valid': len(issues) == 0,
            'production_ready': len(issues) == 0 and len(collision_issues) == 0 and len(ik_issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'collision_issues': collision_issues,
            'ik_issues': ik_issues,
            'recommendations': self._get_configuration_recommendations(issues, warnings, collision_issues, ik_issues)
        }
    
    def _get_configuration_recommendations(self, issues: List[str], warnings: List[str], 
                                         collision_issues: List[str], ik_issues: List[str]) -> List[str]:
        """Get configuration recommendations based on validation results."""
        recommendations = []
        
        if issues:
            recommendations.append("Fix configuration issues before production deployment")
        
        if collision_issues:
            recommendations.append("Recalibrate collision detection thresholds in constraints.yaml")
            recommendations.append("Verify robot model and workspace configuration")
        
        if ik_issues:
            recommendations.append("Review IK solver configuration and tolerance settings")
            recommendations.append("Consider enabling C-space analysis for better convergence")
        
        if warnings:
            recommendations.append("Review configuration warnings for optimal performance")
        
        if self.production_mode and not self.config.get('enable_fallbacks', False):
            recommendations.append("Enable fallback strategies for production robustness")
        
        return recommendations

    def enable_cartesian_optimization(self):
        """Enable optimizations specifically for Cartesian planning performance."""
        with self._config_lock:
            # Reduce planning timeouts for faster feedback
            self.config['max_planning_time'] = 10.0  # Reduced from 30s
            
            # Optimize IK tolerances for Cartesian planning
            self.config['ik_position_tolerance'] = 0.003  # 3mm (relaxed from 2mm)
            self.config['ik_rotation_tolerance'] = 3.0     # 3 degrees (relaxed from 2°)
            
            # Reduce maximum planning attempts
            self.config['max_attempts'] = 2  # Reduced from 3
            
            # Enable C-space analysis if available
            if not self.cspace_analysis_enabled:
                self.enable_configuration_space_analysis(build_maps=False)
        
        logger.info("Cartesian planning optimizations enabled")
    
    def disable_cartesian_optimization(self):
        """Disable Cartesian-specific optimizations and return to standard settings."""
        with self._config_lock:
            # Restore standard timeouts
            self.config['max_planning_time'] = 30.0
            
            # Restore strict IK tolerances
            self.config['ik_position_tolerance'] = 0.002  # 2mm
            self.config['ik_rotation_tolerance'] = 2.0     # 2 degrees
            
            # Restore maximum planning attempts
            self.config['max_attempts'] = 3
        
        logger.info("Cartesian planning optimizations disabled")

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
        """Solve IK with C-space optimization if available, with optimized parameters for Cartesian planning."""
        if self.cspace_analysis_enabled and self.config_analyzer:
            # Get optimal seed from C-space analysis
            target_position = target_pose[:3, 3]
            q_seed = self.config_analyzer.get_best_ik_region(target_position)
            if q_seed is not None:
                # Use fast IK parameters for C-space seeded attempts
                fast_params = {
                    'max_iters': 150,      # Reduced from 600
                    'num_attempts': 1,     # Single attempt with good seed
                    'pos_tol': 2e-3,       # Slightly relaxed tolerance
                    'rot_tol': 5e-3        # Slightly relaxed tolerance
                }
                q_solution, converged = self.ik.solve(target_pose, q_init=q_seed, **fast_params)
                if converged:
                    return q_solution, True
        
        # Fast IK parameters for Cartesian planning
        fast_ik_params = {
            'max_iters': 200,          # Reduced from 600
            'num_attempts': 12,        # Reduced from 100  
            'pos_tol': 2e-3,           # Relaxed from 8e-4 (2mm tolerance)
            'rot_tol': 5e-3,           # Relaxed from 2e-3 (0.3° tolerance)
            'damping': 5e-4,           # Increased damping for stability
            'step_scale': 0.7,         # Larger steps for faster convergence
            'combined_tolerance': 3e-3, # More permissive combined tolerance
            'smart_seeding': True,
            'adaptive_tolerance': True
        }
        
        # Try multiple strategic initial guesses with fast parameters
        initial_guesses = [
            q_current if q_current is not None else np.zeros(6),  # Current or zero
            np.array([0, -np.pi/2, np.pi/2, 0, np.pi/2, 0]),     # Home position
            np.array([np.pi/4, -np.pi/3, np.pi/3, 0, np.pi/3, np.pi/4]),  # Alternative 1
            np.array([-np.pi/4, -np.pi/4, np.pi/4, 0, np.pi/4, -np.pi/4]), # Alternative 2
            np.array([0, 0, 0, 0, 0, 0]),                         # Zero position
            np.array([np.pi/6, -np.pi/4, np.pi/4, np.pi/6, np.pi/4, 0]),  # Conservative pose
        ]
        
        for i, q_init in enumerate(initial_guesses):
            try:
                q_solution, converged = self.ik.solve(target_pose, q_init=q_init, **fast_ik_params)
                if converged:
                    if i > 0:  # Log when fallback was needed
                        logger.info(f"IK converged using initial guess #{i+1}")
                    return q_solution, True
            except Exception as e:
                logger.debug(f"IK attempt {i+1} failed: {e}")
                continue
        
        # If all strategic guesses fail, try a few random attempts with even faster parameters
        emergency_params = {
            'max_iters': 100,          # Very fast
            'num_attempts': 3,         # Only 3 random attempts
            'pos_tol': 5e-3,           # More relaxed (5mm)
            'rot_tol': 1e-2,           # More relaxed (0.6°)
            'damping': 1e-3,           # Higher damping
            'step_scale': 0.8          # Larger steps
        }
        
        for attempt in range(3):
            try:
                q_random = np.random.uniform(-np.pi, np.pi, 6)
                q_solution, converged = self.ik.solve(target_pose, q_init=q_random, **emergency_params)
                if converged:
                    logger.info(f"IK converged using emergency random seed #{attempt+1}")
                    return q_solution, True
            except Exception as e:
                logger.debug(f"Emergency IK attempt {attempt+1} failed: {e}")
                continue
        
        # All attempts failed
        return None, False
    
    def _optimize_joint_continuity(self, q_start: np.ndarray, q_goal: np.ndarray, 
                                 goal_pose: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Optimize joint continuity by finding alternative IK solutions with smaller joint jumps.
        
        Args:
            q_start: Starting joint configuration
            q_goal: Current goal joint configuration
            goal_pose: Target Cartesian pose
            
        Returns:
            Optimized goal configuration and success flag
        """
        initial_jump = np.linalg.norm(q_goal - q_start)
        
        # If jump is reasonable, return as-is
        if initial_jump <= np.pi:
            return q_goal, True
        
        logger.info(f"Large joint jump detected ({initial_jump:.3f} rad), optimizing continuity...")
        
        # Try to find alternative IK solutions with better continuity
        best_q = q_goal
        best_jump = initial_jump
        
        # Generate IK seeds based on the start configuration
        continuity_seeds = [
            q_start,  # Start from current position
            q_start + 0.1 * np.random.randn(6),  # Small perturbation 1
            q_start + 0.2 * np.random.randn(6),  # Small perturbation 2
            q_start + 0.3 * np.random.randn(6),  # Medium perturbation
        ]
        
        # Add strategic joint modifications (handling joint wrapping)
        for i in range(6):
            # Try adding/subtracting 2π to each joint
            q_mod = q_goal.copy()
            if q_mod[i] - q_start[i] > np.pi:
                q_mod[i] -= 2*np.pi
                continuity_seeds.append(q_mod.copy())
            elif q_start[i] - q_mod[i] > np.pi:
                q_mod[i] += 2*np.pi
                continuity_seeds.append(q_mod.copy())
        
        # Fast IK parameters for continuity optimization
        continuity_params = {
            'max_iters': 100,      # Fast iterations
            'num_attempts': 3,     # Few attempts per seed
            'pos_tol': 3e-3,       # Slightly relaxed (3mm)
            'rot_tol': 8e-3,       # Slightly relaxed (0.5°)
        }
        
        for seed in continuity_seeds:
            try:
                q_candidate, converged = self.ik.solve(goal_pose, q_init=seed, **continuity_params)
                if converged:
                    jump = np.linalg.norm(q_candidate - q_start)
                    if jump < best_jump:
                        best_q = q_candidate
                        best_jump = jump
                        
                        # Early exit if we find a good solution
                        if jump <= np.pi:
                            logger.info(f"Continuity optimized: {initial_jump:.3f} → {jump:.3f} rad")
                            return best_q, True
            except Exception as e:
                logger.debug(f"Continuity optimization attempt failed: {e}")
                continue
        
        # Return best found solution
        if best_jump < initial_jump:
            logger.info(f"Continuity improved: {initial_jump:.3f} → {best_jump:.3f} rad")
            return best_q, True
        else:
            logger.warning(f"Could not improve continuity, keeping original solution")
            return q_goal, False
    
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
                
                # Production logging
                planning_time = time.time() - start_time
                if result.status == PlanningStatus.SUCCESS and result.plan:
                    self._log_success(result.plan, planning_time)
                    with self._stats_lock:
                        self.stats['successful_plans'] += 1
                else:
                    error_context = {
                        'strategy': strategy.value if strategy else 'unknown',
                        'waypoint_count': waypoint_count,
                        'start_config_shape': start_config.shape,
                        'goal_config_shape': goal_config.shape
                    }
                    self._log_error(result.status.value, result.error_message or "Unknown planning failure", error_context)
                    with self._stats_lock:
                        self.stats['failed_plans'] += 1
            
                return result
                
            except Exception as e:
                planning_time = time.time() - start_time
                error_message = f"Planning exception: {str(e)}"
                
                # Production error logging
                error_context = {
                    'strategy': strategy.value if strategy else 'unknown',
                    'waypoint_count': waypoint_count,
                    'exception_type': type(e).__name__,
                    'planning_time': planning_time
                }
                self._log_error("EXCEPTION", error_message, error_context)
                
                with self._stats_lock:
                    self.stats['failed_plans'] += 1
                
                logger.error(f"Motion planning failed with exception: {e}")
                return MotionPlanningResult(
                    status=PlanningStatus.FAILED,
                    error_message=error_message,
                    planning_time=planning_time
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
            
            # Enable Cartesian-specific optimizations
            self.enable_cartesian_optimization()
        
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
                
                # STEP 2: Solve IK for start pose from current/initial state with fast parameters
                fast_ik_params = {
                    'max_iters': 200,      # Reduced iterations
                    'num_attempts': 8,     # Reduced attempts
                    'pos_tol': 2e-3,       # 2mm tolerance
                    'rot_tol': 5e-3,       # 0.3° tolerance
                }
                
                q_start, start_converged = self.ik.solve(start_pose, q_init=q_init_start, **fast_ik_params)
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
                q_goal, goal_converged = self.ik.solve(goal_pose, q_init=q_start, **fast_ik_params)
                if not goal_converged:
                    # Try fallback initial guesses only if continuous method fails
                    logger.warning("Continuous IK for goal pose failed, trying fallback initial guesses")
                    q_goal, goal_converged = self.solve_ik_with_cspace(goal_pose, q_current=q_start)
                    if not goal_converged:
                        # Final fallback: try with multiple random seeds using fast parameters
                        logger.warning("All standard IK methods failed, trying aggressive fallback")
                        emergency_params = {
                            'max_iters': 100,  # Very fast
                            'num_attempts': 5, # Only 5 random attempts
                            'pos_tol': 5e-3,   # More relaxed (5mm)
                            'rot_tol': 1e-2,   # More relaxed (0.6°)
                        }
                        
                        for attempt in range(5):
                            q_random = np.random.uniform(-np.pi, np.pi, 6)
                            q_try, converged = self.ik.solve(goal_pose, q_init=q_random, **emergency_params)
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
                
                # STEP 4: Optimize joint space continuity
                q_goal, continuity_improved = self._optimize_joint_continuity(q_start, q_goal, goal_pose)
                
                # STEP 5: Validate final joint space continuity
                joint_diff = np.linalg.norm(q_goal - q_start)
                if joint_diff > np.pi:  # Large joint space jump
                    logger.warning(f"Large joint space jump detected: {joint_diff:.3f} rad")
                    
                    # If still a large jump, try one more optimization round
                    if joint_diff > 2*np.pi:
                        logger.warning("Extremely large jump detected, attempting final optimization")
                        q_goal_opt, _ = self._optimize_joint_continuity(q_start, q_goal, goal_pose)
                        final_diff = np.linalg.norm(q_goal_opt - q_start)
                        if final_diff < joint_diff:
                            q_goal = q_goal_opt
                            joint_diff = final_diff
                            logger.info(f"Final optimization reduced jump to {joint_diff:.3f} rad")
                
                # STEP 6: Plan smooth motion in joint space
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
                
                # Disable Cartesian optimizations before returning
                self.disable_cartesian_optimization()
            
                return result
            
            except Exception as e:
                # Disable Cartesian optimizations before returning error
                self.disable_cartesian_optimization()
                
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
        Enhanced IK solver with constraint validation and intelligent retry strategies.
        
        Args:
            target_pose: 4x4 target transformation matrix
            max_attempts: Maximum number of IK attempts with different seeds
            use_different_seeds: Whether to use different initial configurations
            
        Returns:
            Tuple of (joint solution, success flag)
        """
        max_attempts = max_attempts or self.config['ik_max_attempts']
        
        # Phase 1: Try with C-space optimization
        q_solution, converged = self.solve_ik_with_cspace(target_pose)
        if converged and self._validate_ik_solution(q_solution, target_pose):
            return q_solution, True
        
        # Phase 2: Store best solution even if not perfect
        best_solution = q_solution if converged else None
        best_error = float('inf')
        
        if not use_different_seeds or max_attempts <= 1:
            return best_solution, converged
            
        # Phase 3: Enhanced multi-strategy approach
        joint_limits = self.path_planner.get_joint_limits()
        limits_lower = np.array([joint_limits[f'j{i+1}']['min'] for i in range(6)])
        limits_upper = np.array([joint_limits[f'j{i+1}']['max'] for i in range(6)])
        
        # Convert degrees to radians for limits
        limits_lower = np.radians(limits_lower)
        limits_upper = np.radians(limits_upper)
        
        for attempt in range(1, max_attempts):
            # Strategy selection based on attempt number
            if attempt % 5 == 1:
                # Random uniform sampling
                q_init = np.random.uniform(limits_lower, limits_upper)
            elif attempt % 5 == 2:
                # Gaussian around middle with variable spread
                q_init = (limits_lower + limits_upper) / 2
                spread = 0.2 + (attempt % 3) * 0.1
                q_init += np.random.normal(0, spread, size=q_init.shape)
                q_init = np.clip(q_init, limits_lower, limits_upper)
            elif attempt % 5 == 3:
                # Strategic configurations for RB3-730ES-U
                strategic_configs = [
                    np.array([0.0, -0.5, 1.0, 0.0, 0.5, 0.0]),
                    np.array([1.57, -1.0, 0.5, 0.0, 1.0, 0.0]),
                    np.array([-1.57, -1.0, 0.5, 0.0, 1.0, 0.0]),
                    np.array([0.78, -0.78, 0.78, 1.57, 0.0, 0.0])
                ]
                config_idx = (attempt // 5) % len(strategic_configs)
                q_init = strategic_configs[config_idx].copy()
                # Add small perturbation
                q_init += np.random.normal(0, 0.1, size=q_init.shape)
                q_init = np.clip(q_init, limits_lower, limits_upper)
            elif attempt % 5 == 4:
                # Perturbation around best solution found so far
                if best_solution is not None:
                    perturbation_scale = 0.15 + (attempt % 3) * 0.05
                    q_init = best_solution + np.random.normal(0, perturbation_scale, best_solution.shape)
                    q_init = np.clip(q_init, limits_lower, limits_upper)
                else:
                    q_init = np.random.uniform(limits_lower, limits_upper)
            else:
                # Workspace-biased sampling (favor reachable configurations)
                # Target position-based initial guess
                target_pos = target_pose[:3, 3]
                if np.linalg.norm(target_pos) > 0.1:  # Non-origin target
                    # Simple heuristic for initial shoulder and elbow angles
                    reach_distance = np.linalg.norm(target_pos[:2])  # XY distance
                    q_init = np.zeros(6)
                    q_init[0] = np.arctan2(target_pos[1], target_pos[0])  # Base rotation
                    q_init[1] = -0.3 - reach_distance * 0.2  # Shoulder
                    q_init[2] = 0.6 + reach_distance * 0.3   # Elbow
                    # Add randomization to other joints
                    q_init[3:] = np.random.uniform(limits_lower[3:], limits_upper[3:])
                    q_init = np.clip(q_init, limits_lower, limits_upper)
                else:
                    q_init = np.random.uniform(limits_lower, limits_upper)
            
            # Enhanced IK parameters for difficult poses
            ik_params = {
                'num_attempts': 50 + attempt * 2,  # Increase attempts for later iterations
                'max_iters': 400 + attempt * 10,   # More iterations for difficult cases
                'pos_tol': 1e-3 if attempt < 5 else 1.5e-3,  # Relax tolerance gradually
                'rot_tol': 2e-3 if attempt < 5 else 3e-3
            }
            
            # Attempt IK with enhanced parameters
            q_candidate, converged = self.ik.solve(target_pose, q_init=q_init, **ik_params)
            
            if q_candidate is not None:
                # Evaluate solution quality even if not fully converged
                error = self._compute_solution_error(q_candidate, target_pose)
                
                if converged and self._validate_ik_solution(q_candidate, target_pose):
                    return q_candidate, True
                
                # Track best solution
                if error < best_error:
                    best_error = error
                    best_solution = q_candidate
                    
                # Accept "good enough" solutions for complex poses
                if error < self.config.get('ik_relaxed_tolerance', 2e-3):
                    # Additional validation for relaxed solutions
                    if self._validate_ik_solution_relaxed(q_candidate, target_pose):
                        logger.debug(f"Accepted relaxed IK solution on attempt {attempt}")
                        return q_candidate, True
        
        # Return best solution found, even if not perfect
        if best_solution is not None and best_error < 5e-3:  # 5mm tolerance
            logger.debug(f"Returning best IK solution with error {best_error:.6f}")
            return best_solution, False
        
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
    
    def _validate_ik_solution_relaxed(self, q_solution: np.ndarray, 
                                    target_pose: np.ndarray) -> bool:
        """Relaxed validation for complex poses that are close enough."""
        try:
            # More permissive accuracy checks
            T_achieved = self.fk.compute_forward_kinematics(q_solution)
            tcp_position = T_achieved[:3, 3]
            
            # Relaxed position error check (2x normal tolerance)
            pos_error = np.linalg.norm(tcp_position - target_pose[:3, 3])
            if pos_error > self.config['ik_position_tolerance'] * 2:
                return False
            
            # Relaxed orientation error check (2x normal tolerance)
            R_desired = target_pose[:3, :3]
            R_achieved = T_achieved[:3, :3]
            
            cos_angle = (np.trace(R_desired.T @ R_achieved) - 1) / 2
            cos_angle = np.clip(cos_angle, -1, 1)
            rot_error = np.arccos(cos_angle)
            
            rot_tolerance = np.radians(self.config['ik_rotation_tolerance'] * 2)
            if rot_error > rot_tolerance:
                return False
            
            # Still require collision-free
            collision_result = self.collision_checker.check_configuration_collision(
                q_solution, tcp_position, self.fk.compute_forward_kinematics
            )
            
            if collision_result.is_collision:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Relaxed IK solution validation failed: {e}")
            return False
    
    def _compute_solution_error(self, q_solution: np.ndarray, 
                              target_pose: np.ndarray) -> float:
        """Compute combined error metric for IK solution quality assessment."""
        try:
            T_achieved = self.fk.compute_forward_kinematics(q_solution)
            
            # Position error
            pos_error = np.linalg.norm(T_achieved[:3, 3] - target_pose[:3, 3])
            
            # Orientation error
            R_desired = target_pose[:3, :3]
            R_achieved = T_achieved[:3, :3]
            cos_angle = (np.trace(R_desired.T @ R_achieved) - 1) / 2
            cos_angle = np.clip(cos_angle, -1, 1)
            rot_error = np.arccos(cos_angle)
            
            # Combined error with position weighted more heavily
            return pos_error + rot_error * 0.1
            
        except Exception as e:
            logger.error(f"Error computation failed: {e}")
            return float('inf')
    
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

