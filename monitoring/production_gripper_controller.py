#!/usr/bin/env python3
"""
Production-Ready Gripper Controller
==================================

This module provides production-ready gripper control with:
- Robot state continuity
- Advanced error handling
- Safety monitoring
- Real robot interface ready

Author: GitHub Copilot
Date: September 2025
"""

import sys
import os
import numpy as np
import time
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Add module paths
sys.path.append('../kinematics/src')
sys.path.append('../planning/src')

from forward_kinematic import ForwardKinematics
from inverse_kinematic import InverseKinematics

try:
    from motion_planner import MotionPlanner, PlanningStrategy, PlanningStatus
    from trajectory_planner import TrajectoryPlanner
    PLANNING_AVAILABLE = True
except ImportError:
    PLANNING_AVAILABLE = False

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('gripper_production.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Task execution status"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    ABORTED = "aborted"

class SafetyLevel(Enum):
    """Safety levels for operation"""
    SAFE = "safe"
    CAUTION = "caution"
    DANGER = "danger"
    EMERGENCY = "emergency"

@dataclass
class RobotState:
    """Current robot state"""
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    gripper_position: np.ndarray  # 3D position in mm
    gripper_is_open: bool
    last_update_time: float
    safety_level: SafetyLevel

@dataclass
class TaskResult:
    """Result of task execution"""
    status: TaskStatus
    completed_waypoints: int
    total_waypoints: int
    execution_time: float
    error_message: Optional[str] = None
    trajectory_points: Optional[List[np.ndarray]] = None

class ProductionGripperController:
    """
    Production-ready gripper controller with advanced error handling,
    safety monitoring, and robot state continuity.
    """
    
    def __init__(self, enable_planning: bool = True):
        """Initialize production gripper controller"""
        
        # Initialize kinematics with gripper mode
        self.fk = ForwardKinematics(tool_name='default_gripper')
        self.ik = InverseKinematics(self.fk)
        
        # Initialize planning if available
        self.planning_enabled = enable_planning and PLANNING_AVAILABLE
        if self.planning_enabled:
            self.motion_planner = MotionPlanner(self.fk, self.ik)
            self.trajectory_planner = TrajectoryPlanner()
            logger.info("Production controller initialized with trajectory planning")
        else:
            logger.info("Production controller initialized in basic mode")
        
        # Robot state tracking
        self.robot_state: Optional[RobotState] = None
        self.last_successful_position: Optional[np.ndarray] = None
        
        # Safety parameters
        self.safety_params = {
            'max_joint_velocity': np.deg2rad([90, 90, 90, 180, 180, 180]),  # deg/s
            'max_joint_acceleration': np.deg2rad([45, 45, 45, 90, 90, 90]),  # deg/s²
            'max_cartesian_velocity': 0.5,  # m/s
            'emergency_stop_distance': 0.05,  # 50mm emergency stop distance
            'workspace_safety_margin': 0.02,  # 20mm safety margin
        }
        
        # Performance tracking
        self.stats = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_ik_attempts': 0,
            'emergency_stops': 0,
            'average_execution_time': 0.0
        }
        
        logger.info("Production gripper controller ready")
    
    def update_robot_state(self, joint_positions: np.ndarray, 
                          joint_velocities: np.ndarray = None,
                          gripper_is_open: bool = True) -> None:
        """
        Update current robot state (called by robot interface)
        
        Args:
            joint_positions: Current joint angles (6,)
            joint_velocities: Current joint velocities (6,) 
            gripper_is_open: Current gripper state
        """
        
        if joint_velocities is None:
            joint_velocities = np.zeros(6)
        
        # Calculate current gripper position
        T_current = self.fk.compute_forward_kinematics(joint_positions)
        gripper_position = T_current[:3, 3] * 1000  # Convert to mm
        
        # Assess safety level
        safety_level = self._assess_safety_level(joint_positions, joint_velocities)
        
        self.robot_state = RobotState(
            joint_positions=joint_positions.copy(),
            joint_velocities=joint_velocities.copy(),
            gripper_position=gripper_position.copy(),
            gripper_is_open=gripper_is_open,
            last_update_time=time.time(),
            safety_level=safety_level
        )
        
        self.last_successful_position = joint_positions.copy()
        logger.debug(f"Robot state updated: {gripper_position.round(1)} mm")
    
    def solve_ik_with_continuity(self, T_target: np.ndarray, 
                                max_attempts: int = 3) -> Tuple[Optional[np.ndarray], bool]:
        """
        Production IK solver with state continuity and multiple fallback strategies
        
        Args:
            T_target: Target transformation matrix
            max_attempts: Maximum number of different approaches to try
            
        Returns:
            (q_solution, success): Joint solution and success flag
        """
        
        attempts = []
        
        # Strategy 1: Use current robot state (highest priority)
        if self.robot_state is not None:
            attempts.append(("current_state", self.robot_state.joint_positions))
            logger.debug("Using current robot state as initial guess")
        
        # Strategy 2: Use last successful position
        if self.last_successful_position is not None:
            attempts.append(("last_successful", self.last_successful_position))
        
        # Strategy 3: Use proven demo guess (fallback)
        attempts.append(("proven_demo", np.deg2rad([0, -30, 60, 0, 45, 0])))
        
        # Strategy 4: Use home position (last resort)
        attempts.append(("home_position", np.zeros(6)))
        
        # Try each strategy
        for i, (strategy_name, q_init) in enumerate(attempts[:max_attempts]):
            try:
                q_solution, converged = self.ik.solve(T_target, q_init=q_init)
                
                if converged:
                    # Validate solution quality
                    if self._validate_solution_safety(q_solution, T_target):
                        logger.debug(f"IK converged using {strategy_name} strategy")
                        return q_solution, True
                    else:
                        logger.warning(f"IK solution from {strategy_name} failed safety validation")
                
            except Exception as e:
                logger.warning(f"IK attempt {i+1} ({strategy_name}) failed: {e}")
                self.stats['failed_ik_attempts'] += 1
        
        # All strategies failed
        logger.error("All IK strategies failed")
        return None, False
    
    def execute_pick_and_place_task(self, pick_position: List[float], 
                                  place_position: List[float],
                                  approach_height: float = 200,
                                  grasp_height: float = 120) -> TaskResult:
        """
        Execute complete pick-and-place task with production-level reliability
        
        Args:
            pick_position: [x, y] coordinates for pick location (mm)
            place_position: [x, y] coordinates for place location (mm)
            approach_height: Safe approach height (mm)
            grasp_height: Grasping height (mm)
            
        Returns:
            TaskResult with execution details
        """
        
        start_time = time.time()
        self.stats['total_tasks'] += 1
        
        logger.info(f"Starting pick-and-place task: {pick_position} → {place_position}")
        
        # Generate task waypoints
        waypoints = self._generate_pick_place_waypoints(
            pick_position, place_position, approach_height, grasp_height
        )
        
        # Execute with trajectory planning if available
        if self.planning_enabled:
            result = self._execute_planned_trajectory(waypoints)
        else:
            result = self._execute_basic_trajectory(waypoints)
        
        # Update statistics
        execution_time = time.time() - start_time
        result.execution_time = execution_time
        
        if result.status == TaskStatus.SUCCESS:
            self.stats['successful_tasks'] += 1
            logger.info(f"Task completed successfully in {execution_time:.2f}s")
        else:
            logger.error(f"Task failed: {result.error_message}")
        
        # Update average execution time
        self.stats['average_execution_time'] = (
            (self.stats['average_execution_time'] * (self.stats['total_tasks'] - 1) + execution_time) 
            / self.stats['total_tasks']
        )
        
        return result
    
    def _generate_pick_place_waypoints(self, pick_pos: List[float], place_pos: List[float],
                                     approach_height: float, grasp_height: float) -> List[Tuple[List[float], str]]:
        """Generate optimized waypoints for pick-and-place task"""
        
        # Start from current position if available
        if self.robot_state is not None:
            current_pos = self.robot_state.gripper_position.tolist()
            home_position = current_pos
        else:
            # Conservative home position
            home_position = [200, 0, approach_height]
        
        waypoints = [
            (home_position, "Current/Home position"),
            ([pick_pos[0], pick_pos[1], approach_height], "Approach pick location"),
            ([pick_pos[0], pick_pos[1], grasp_height], "Grasp object"),
            ([pick_pos[0], pick_pos[1], approach_height], "Lift object"),
            ([place_pos[0], place_pos[1], approach_height], "Transfer to place"),
            ([place_pos[0], place_pos[1], grasp_height], "Place object"),
            ([place_pos[0], place_pos[1], approach_height], "Retract from place"),
            (home_position, "Return to home")
        ]
        
        return waypoints
    
    def _execute_planned_trajectory(self, waypoints: List[Tuple[List[float], str]]) -> TaskResult:
        """Execute waypoints using advanced trajectory planning"""
        
        logger.info("Executing with advanced trajectory planning")
        
        # Convert waypoints to joint space
        joint_waypoints = []
        for i, (position, description) in enumerate(waypoints):
            T_target = np.eye(4)
            T_target[:3, 3] = np.array(position) / 1000.0
            
            q_solution, success = self.solve_ik_with_continuity(T_target)
            
            if success:
                joint_waypoints.append(q_solution)
                logger.debug(f"Waypoint {i+1}: {description} - IK success")
            else:
                error_msg = f"IK failed for waypoint {i+1}: {description}"
                return TaskResult(
                    status=TaskStatus.FAILED,
                    completed_waypoints=i,
                    total_waypoints=len(waypoints),
                    execution_time=0,
                    error_message=error_msg
                )
        
        # Plan trajectory with motion planner
        try:
            planning_result = self.motion_planner.plan_waypoint_motion(
                joint_waypoints,
                strategy=PlanningStrategy.JOINT_SPACE
            )
            
            if planning_result.status == PlanningStatus.SUCCESS:
                # Generate smooth trajectory
                trajectory_result = self.trajectory_planner.plan_trajectory(
                    planning_result.plan.joint_waypoints,
                    time_scaling=1.0,
                    optimize=True
                )
                
                if trajectory_result.success:
                    logger.info(f"Trajectory generated: {len(trajectory_result.trajectory.points)} interpolated points")
                    return TaskResult(
                        status=TaskStatus.SUCCESS,
                        completed_waypoints=len(waypoints),
                        total_waypoints=len(waypoints),
                        execution_time=0,  # Will be set by caller
                        trajectory_points=trajectory_result.trajectory.points
                    )
                else:
                    return TaskResult(
                        status=TaskStatus.FAILED,
                        completed_waypoints=len(joint_waypoints),
                        total_waypoints=len(waypoints),
                        execution_time=0,
                        error_message=f"Trajectory generation failed: {trajectory_result.error_message}"
                    )
            else:
                return TaskResult(
                    status=TaskStatus.FAILED,
                    completed_waypoints=len(joint_waypoints),
                    total_waypoints=len(waypoints),
                    execution_time=0,
                    error_message=f"Motion planning failed: {planning_result.error_message}"
                )
                
        except Exception as e:
            return TaskResult(
                status=TaskStatus.FAILED,
                completed_waypoints=len(joint_waypoints),
                total_waypoints=len(waypoints),
                execution_time=0,
                error_message=f"Planning system error: {e}"
            )
    
    def _execute_basic_trajectory(self, waypoints: List[Tuple[List[float], str]]) -> TaskResult:
        """Execute waypoints using basic point-to-point motion"""
        
        logger.info("Executing with basic point-to-point motion")
        
        trajectory_points = []
        
        for i, (position, description) in enumerate(waypoints):
            T_target = np.eye(4)
            T_target[:3, 3] = np.array(position) / 1000.0
            
            q_solution, success = self.solve_ik_with_continuity(T_target)
            
            if success:
                trajectory_points.append(q_solution)
                # Update robot state for next iteration
                if self.robot_state is not None:
                    self.update_robot_state(q_solution)
                logger.debug(f"Waypoint {i+1}: {description} - Success")
            else:
                error_msg = f"IK failed for waypoint {i+1}: {description}"
                return TaskResult(
                    status=TaskStatus.PARTIAL if i > 0 else TaskStatus.FAILED,
                    completed_waypoints=i,
                    total_waypoints=len(waypoints),
                    execution_time=0,
                    error_message=error_msg,
                    trajectory_points=trajectory_points
                )
        
        return TaskResult(
            status=TaskStatus.SUCCESS,
            completed_waypoints=len(waypoints),
            total_waypoints=len(waypoints),
            execution_time=0,
            trajectory_points=trajectory_points
        )
    
    def _assess_safety_level(self, joint_positions: np.ndarray, 
                           joint_velocities: np.ndarray) -> SafetyLevel:
        """Assess current safety level based on robot state"""
        
        # Check joint limits
        joint_limits = self.fk.joint_limits
        limits_lower, limits_upper = joint_limits[0], joint_limits[1]
        
        margin_lower = joint_positions - limits_lower
        margin_upper = limits_upper - joint_positions
        min_margin = np.min(np.minimum(margin_lower, margin_upper))
        
        # Check velocities
        max_velocity = np.max(np.abs(joint_velocities))
        velocity_limit = np.max(self.safety_params['max_joint_velocity'])
        
        # Determine safety level
        if min_margin < np.deg2rad(5) or max_velocity > velocity_limit * 0.9:
            return SafetyLevel.DANGER
        elif min_margin < np.deg2rad(15) or max_velocity > velocity_limit * 0.7:
            return SafetyLevel.CAUTION
        else:
            return SafetyLevel.SAFE
    
    def _validate_solution_safety(self, q_solution: np.ndarray, 
                                T_target: np.ndarray, tolerance: float = 0.002) -> bool:
        """Validate IK solution for safety and accuracy"""
        
        # Check accuracy
        T_achieved = self.fk.compute_forward_kinematics(q_solution)
        position_error = np.linalg.norm(T_achieved[:3, 3] - T_target[:3, 3])
        
        if position_error > tolerance:
            logger.warning(f"Solution accuracy too low: {position_error*1000:.1f}mm error")
            return False
        
        # Check joint limits
        joint_limits = self.fk.joint_limits
        limits_lower, limits_upper = joint_limits[0], joint_limits[1]
        
        if not np.all((q_solution >= limits_lower) & (q_solution <= limits_upper)):
            logger.warning("Solution violates joint limits")
            return False
        
        return True
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get controller performance statistics"""
        
        success_rate = (self.stats['successful_tasks'] / self.stats['total_tasks'] 
                       if self.stats['total_tasks'] > 0 else 0)
        
        return {
            'total_tasks': self.stats['total_tasks'],
            'successful_tasks': self.stats['successful_tasks'],
            'success_rate': f"{success_rate*100:.1f}%",
            'failed_ik_attempts': self.stats['failed_ik_attempts'],
            'emergency_stops': self.stats['emergency_stops'],
            'average_execution_time': f"{self.stats['average_execution_time']:.2f}s",
            'planning_enabled': self.planning_enabled,
            'current_safety_level': self.robot_state.safety_level.value if self.robot_state else "unknown"
        }

# Example usage and testing
if __name__ == "__main__":
    
    # Initialize production controller
    controller = ProductionGripperController(enable_planning=True)
    
    # Simulate robot state update (would come from real robot interface)
    initial_joints = np.deg2rad([0, -30, 60, 0, 45, 0])
    controller.update_robot_state(initial_joints, gripper_is_open=True)
    
    # Execute pick-and-place task
    pick_location = [250, 100]  # mm
    place_location = [300, -80]  # mm
    
    result = controller.execute_pick_and_place_task(
        pick_position=pick_location,
        place_position=place_location,
        approach_height=200,
        grasp_height=120
    )
    
    # Display results
    print("\n🏭 PRODUCTION CONTROLLER TEST RESULTS")
    print("=" * 50)
    print(f"Task Status: {result.status.value.upper()}")
    print(f"Completed Waypoints: {result.completed_waypoints}/{result.total_waypoints}")
    print(f"Execution Time: {result.execution_time:.2f}s")
    
    if result.trajectory_points:
        print(f"Trajectory Points: {len(result.trajectory_points)}")
    
    if result.error_message:
        print(f"Error: {result.error_message}")
    
    # Show performance statistics
    stats = controller.get_performance_stats()
    print("\n📊 PERFORMANCE STATISTICS:")
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")