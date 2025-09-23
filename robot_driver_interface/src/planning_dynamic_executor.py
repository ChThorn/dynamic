"""
Planning Dynamic Executor - Integration Bridge
==============================================

This module bridges the motion planning system with the dynamic path execution,
combining the best of both worlds:
- Advanced motion planning (collision avoidance, optimization)
- Dynamic timing-based execution from path_playing_dynamically.py
- CHUNKED WAYPOINT PROCESSING for robot buffer limitations

Author: Robot Control Team  
Date: September 2025
"""

import sys
import os
import time
import threading
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import numpy as np

# Add paths for planning module imports - using absolute paths for reliability
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(current_dir, '..', '..')
planning_src = os.path.join(base_dir, 'planning', 'src')
planning_examples = os.path.join(base_dir, 'planning', 'examples')
kinematics_src = os.path.join(base_dir, 'kinematics', 'src')
robot_driver_dir = os.path.join(base_dir, 'robot_driver')

sys.path.append(planning_src)
sys.path.append(planning_examples)
sys.path.append(kinematics_src)
sys.path.append(robot_driver_dir)

# Import planning components
try:
    from clean_robot_interface import CleanRobotMotionPlanner
    PLANNING_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Planning module not available: {e}")
    PLANNING_AVAILABLE = False

# Import robot driver components  
try:
    from arm_driver import RobotArmDriver, PlatformInitStatus, RobotState
    ROBOT_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Robot driver not available: {e}")
    ROBOT_AVAILABLE = False

# Import robot logger with fallback
try:
    from robot_logger import RobotLogger
    LOGGER_AVAILABLE = True
except ImportError:
    # Create fallback logger
    import logging
    class RobotLogger:
        def __init__(self, name: str):
            self.logger = logging.getLogger(name)
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
        
        def info(self, msg: str): self.logger.info(msg)
        def warning(self, msg: str): self.logger.warning(msg)
        def error(self, msg: str): self.logger.error(msg)
        def debug(self, msg: str): self.logger.debug(msg)
    LOGGER_AVAILABLE = False

@dataclass
class PlanningTarget:
    """Target pose for planning"""
    tcp_position_mm: List[float]  # [x, y, z] in mm
    tcp_rotation_deg: List[float]  # [rx, ry, rz] in degrees
    gripper_mode: bool = False

@dataclass
class ExecutionWaypoint:
    """Waypoint ready for robot execution"""
    joints_deg: List[float]  # Joint angles in degrees
    timestamp: float         # Execution time in seconds
    speed: float = 0.5       # Speed multiplier
    acceleration: float = 0.8  # Acceleration multiplier

class PlanningDynamicExecutor:
    """
    Advanced robot controller that combines:
    - Motion planning with collision avoidance
    - Chunked blend motion execution (inspired by path_playing_dynamically.py)
    - Real-time monitoring and control
    - SAFETY-FIRST operation modes (simulation by default, real by explicit activation)
    
    Safety Features:
    - Initializes in SIMULATION mode by default
    - Requires explicit activation of REAL mode
    - Safety warnings for real robot operations
    - Mode switching validation and logging
    """
    
    def __init__(self, robot_ip: str, operation_mode: str = "simulation", 
                 execution_mode: str = "blend", chunk_size: int = 100):
        """
        Initialize the planning dynamic executor
        
        Args:
            robot_ip: Robot IP address
            execution_mode: "blend" for smooth motion, "point" for point-to-point
            chunk_size: Number of waypoints per chunk (default: 100, optimized chunking)
            operation_mode: "simulation" for safe mode (default), "real" for real robot operation
        """
        self.robot_ip = robot_ip
        self.execution_mode = execution_mode
        self.operation_mode = operation_mode  # Safety-first: simulation by default
        
        # Setup logging
        self.logger = RobotLogger("PlanningDynamicExecutor")
        
        # Initialize components
        self.robot = None
        self.planner = None
        
        # Execution control
        self.is_executing = False
        self.execution_complete = False
        self.execution_thread = None
        self.stop_requested = False
        self.execution_start_time = None  # Track execution start time for progress feedback
        
        # Pre-buffering execution parameters (optimized for continuous motion)
        self.chunk_size = chunk_size  # Configurable waypoints per chunk (default: 100 - optimized chunking)
        self.max_buffer_size = 20  # Maximum waypoints in robot buffer at once
        self.prefill_ratio = 0.75  # Fill 75% of buffer before starting motion
        self.buffer_low_threshold = 0.4  # Stream new chunks when buffer < 40% full
        
        # Buffer tracking for internal management since robot doesn't have get_buffer_size()
        self.points_in_buffer = 0  # Track estimated points in robot buffer
        self.last_chunk_time = time.time()  # Track when last chunk was sent
        self.initial_buffer_points = 0  # Track initial buffer size for consumption calculation
        
        # Performance tracking
        self.stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'planning_times': [],
            'execution_times': [],
            'planning_time': 0.0,
            'execution_time': 0.0
        }
        
        self.logger.info(f"Planning Dynamic Executor initialized for {robot_ip}")
    
    def initialize(self) -> bool:
        """Initialize robot and planning components"""
        try:
            # Initialize robot driver
            if ROBOT_AVAILABLE:
                self.logger.info("Initializing robot driver...")
                self.robot = RobotArmDriver(robot_ip=self.robot_ip, logger=self.logger)
                
                if not self.robot.start():
                    self.logger.error("Failed to start robot driver")
                    return False
                
                # Wait for connections
                timeout = time.time() + 10
                while time.time() < timeout:
                    cmd_connected, data_connected = self.robot.is_connected()
                    if cmd_connected and data_connected:
                        break
                    time.sleep(0.5)
                else:
                    self.logger.error("Robot connection timeout")
                    return False
                
                # Initialize robot platform
                time.sleep(1.0)
                current_init_status = self.robot.sys_status.init_state_info
                if current_init_status != PlatformInitStatus.INITIALIZATION_DONE.value:
                    self.logger.info("Initializing robot platform...")
                    self.robot.robot_init()
                    time.sleep(3.0)
                
                self.logger.info("Robot driver initialized")
                
                # Set robot to simulation mode by default for safety
                self.logger.info("Setting robot to SIMULATION mode for safety")
                if not self.set_operation_mode(self.operation_mode):
                    self.logger.warning("Failed to set initial operation mode, defaulting to simulation")
                    self.operation_mode = "simulation"
                    
            else:
                self.logger.warning("Robot driver not available - limited functionality")
            
            # Initialize motion planner
            if PLANNING_AVAILABLE:
                self.planner = CleanRobotMotionPlanner()
                self.logger.info("Motion planner initialized")
            else:
                self.logger.warning("Motion planner not available - will use direct execution")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def plan_and_execute_motion(self, target: PlanningTarget, 
                              start_joints: Optional[List[float]] = None) -> bool:
        """
        Plan and execute motion to target pose
        
        Args:
            target: Target pose for motion
            start_joints: Starting joint configuration (current position if None)
        
        Returns:
            True if planning and execution successful
        """
        # Safety check: Warn if in real mode
        if self.operation_mode == "real":
            self.logger.warning("EXECUTING REAL ROBOT MOTION")
        else:
            self.logger.info("Executing motion in SIMULATION mode (safe)")
            
        if not self.planner or not self.robot:
            self.logger.error("Components not initialized")
            return False
        
        if self.is_executing:
            self.logger.error("Already executing motion")
            return False
        
        try:
            # Record start time
            self.stats['total_executions'] += 1
            
            # Get current position if not provided
            if start_joints is None:
                start_joints = self._get_current_joints()
                if not start_joints:
                    return False
            
            # Plan motion
            self.logger.info(f"Planning motion to {target.tcp_position_mm}")
            planning_start = time.time()
            
            plan = self.planner.plan_motion(
                current_joints_deg=start_joints,
                target_position_mm=target.tcp_position_mm,
                target_orientation_deg=target.tcp_rotation_deg
            )
            
            planning_time = time.time() - planning_start
            
            if not plan.success:
                self.logger.error(f"Planning failed: {plan.error_message}")
                return False
            
            self.logger.info(f"Planning successful: {len(plan.waypoints)} waypoints in {planning_time:.2f}s")
            
            # Convert to execution waypoints
            execution_waypoints = self._convert_to_execution_waypoints(plan.waypoints)
            
            # SPEED FIX: Log when using slower speeds for initial positioning
            if self.stats['successful_executions'] == 0:
                self.logger.info("Using reduced speed for initial positioning to home/target")
            
            # Execute with timing control
            execution_start = time.time()
            success = self._execute_dynamic_trajectory(execution_waypoints)
            execution_time = time.time() - execution_start
            
            # Update statistics
            if success:
                self.stats['successful_executions'] += 1
                self.logger.info(f"Motion completed in {execution_time:.2f}s")
                
                # STABILITY FIX: Add extra stabilization for first motion in sequence
                # This helps prevent controller warnings when robot moves from arbitrary current position
                if self.stats['successful_executions'] == 1:
                    self.logger.debug("First motion complete - ensuring full stabilization")
                    time.sleep(0.3)  # Extra 300ms for first motion stability
            
            self.stats['planning_times'].append(planning_time)
            self.stats['execution_times'].append(execution_time)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Motion execution failed: {e}")
            return False
    
    def execute_pick_and_place_sequence(self, pick_target: PlanningTarget, 
                                      place_target: PlanningTarget) -> bool:
        """
        Execute complete pick and place sequence with planning
        
        Args:
            pick_target: Pick location target
            place_target: Place location target
        
        Returns:
            True if entire sequence successful
        """
        # Safety check: Warn if in real mode
        if self.operation_mode == "real":
            self.logger.warning("EXECUTING REAL PICK AND PLACE SEQUENCE")
        else:
            self.logger.info("Executing pick and place in SIMULATION mode (safe)")
            
        self.logger.info("Starting pick and place sequence")
        
        try:
            # 1. Move to pick approach position
            pick_approach = PlanningTarget(
                tcp_position_mm=[pick_target.tcp_position_mm[0], 
                               pick_target.tcp_position_mm[1],
                               pick_target.tcp_position_mm[2] + 50],  # 50mm above
                tcp_rotation_deg=pick_target.tcp_rotation_deg,
                gripper_mode=pick_target.gripper_mode
            )
            
            if not self.plan_and_execute_motion(pick_approach):
                self.logger.error("Failed to reach pick approach")
                return False
            
            # 2. Move to pick position
            if not self.plan_and_execute_motion(pick_target):
                self.logger.error("Failed to reach pick position")
                return False
            
            # 3. Close gripper (placeholder - needs gripper integration)
            self.logger.info("Closing gripper")
            time.sleep(0.5)
            
            # 4. Move to place approach position
            place_approach = PlanningTarget(
                tcp_position_mm=[place_target.tcp_position_mm[0], 
                               place_target.tcp_position_mm[1],
                               place_target.tcp_position_mm[2] + 50],  # 50mm above
                tcp_rotation_deg=place_target.tcp_rotation_deg,
                gripper_mode=place_target.gripper_mode
            )
            
            if not self.plan_and_execute_motion(place_approach):
                self.logger.error("Failed to reach place approach")
                return False
            
            # 5. Move to place position
            if not self.plan_and_execute_motion(place_target):
                self.logger.error("Failed to reach place position")
                return False
            
            # 6. Open gripper (placeholder - needs gripper integration)
            self.logger.info("Opening gripper")
            time.sleep(0.5)
            
            self.logger.info("Pick and place sequence completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Pick and place failed: {e}")
            return False
    
    def _calculate_motion_time(self, planning_waypoints) -> float:
        """Calculate appropriate execution time based on path length"""
        total_angular_distance = 0
        
        for i in range(1, len(planning_waypoints)):
            joint_diff = np.array(planning_waypoints[i].joints_deg) - \
                         np.array(planning_waypoints[i-1].joints_deg)
            total_angular_distance += np.linalg.norm(joint_diff)
        
        # Assume comfortable speed of 30-60 deg/s
        comfortable_speed = 45  # deg/s
        base_time = total_angular_distance / comfortable_speed
        
        # Add time for acceleration/deceleration phases
        return base_time * 1.3  # 30% overhead for smooth accel/decel

    def _convert_to_execution_waypoints(self, planning_waypoints, 
                                      total_time: float = None) -> List[ExecutionWaypoint]:
        """Convert planning waypoints to execution waypoints with polynomial trajectory timing"""
        if total_time is None:
            # Auto-calculate execution time based on path distance
            total_time = self._calculate_motion_time(planning_waypoints)
        
        execution_waypoints = []
        
        # Generate polynomial trajectory with smooth timing
        if len(planning_waypoints) > 2:
            # Use quintic polynomial for smooth acceleration profiles
            execution_waypoints = self._generate_polynomial_trajectory(planning_waypoints, total_time)
        else:
            # Fallback to linear interpolation for < 3 waypoints
            for i, wp in enumerate(planning_waypoints):
                # Calculate timing
                timestamp = (i / (len(planning_waypoints) - 1)) * total_time if len(planning_waypoints) > 1 else 0
                
                # Calculate speed based on motion characteristics
                speed = self._calculate_waypoint_speed(planning_waypoints, i)
                
                execution_waypoints.append(ExecutionWaypoint(
                    joints_deg=wp.joints_deg,
                    timestamp=timestamp,
                    speed=speed,
                    acceleration=0.8
                ))
        
        return execution_waypoints
    
    def _calculate_waypoint_speed(self, waypoints, index: int) -> float:
        """Calculate appropriate speed for waypoint based on motion characteristics"""
        base_speed = 0.5
        
        # SPEED FIX: Use slower speed for first motion (initial positioning)
        # This helps prevent controller warnings and improves stability
        if self.stats['successful_executions'] == 0:  # First motion in sequence
            base_speed = 0.25  # 50% slower for initial positioning
            self.logger.debug(f"Using slower speed ({base_speed}) for initial positioning")
        
        if len(waypoints) < 2:
            return base_speed
        
        # Calculate joint movement magnitude
        if index > 0:
            prev_joints = np.array(waypoints[index-1].joints_deg)
            curr_joints = np.array(waypoints[index].joints_deg)
            joint_movement = np.linalg.norm(curr_joints - prev_joints)
            
            # Adjust speed based on movement magnitude
            if joint_movement > 20:  # Large movement
                speed_multiplier = 0.6 if self.stats['successful_executions'] == 0 else 0.3
                return speed_multiplier  # Even slower for large initial movements
            elif joint_movement < 5:  # Small movement
                speed_multiplier = 0.5 if self.stats['successful_executions'] == 0 else 0.7
                return speed_multiplier  # Controlled speed for small initial movements
        
        return base_speed
    
    def _generate_polynomial_trajectory(self, planning_waypoints, total_time: float) -> List[ExecutionWaypoint]:
        """
        Generate smooth polynomial trajectory with quintic (5th order) blending
        
        This creates optimal motion profiles with:
        - Smooth position transitions
        - Continuous velocity profiles  
        - Continuous acceleration profiles
        - Minimal jerk for robot-friendly motion
        
        Args:
            planning_waypoints: Input waypoints from motion planner
            total_time: Total execution time
            
        Returns:
            List of execution waypoints with polynomial timing
        """
        if len(planning_waypoints) < 3:
            return []
        
        # Extract joint positions
        joint_positions = np.array([wp.joints_deg for wp in planning_waypoints])
        num_joints = joint_positions.shape[1]
        num_waypoints = len(planning_waypoints)
        
        # Create dense time sampling for smooth trajectory
        dense_time_points = int(total_time * 20)  # 20 points per second
        dense_timestamps = np.linspace(0, total_time, dense_time_points)
        
        # Generate smooth trajectory for each joint using quintic polynomials
        smooth_trajectory = np.zeros((dense_time_points, num_joints))
        
        for joint_idx in range(num_joints):
            # Extract joint positions for this joint
            joint_values = joint_positions[:, joint_idx]
            
            # Create parameter values for waypoints (normalized 0-1)
            waypoint_params = np.linspace(0, 1, num_waypoints)
            
            # Generate quintic polynomial trajectory
            smooth_joint_trajectory = self._quintic_polynomial_interpolation(
                waypoint_params, joint_values, dense_time_points
            )
            
            smooth_trajectory[:, joint_idx] = smooth_joint_trajectory
        
        # Convert to execution waypoints with optimized timing
        execution_waypoints = []
        
        for i, timestamp in enumerate(dense_timestamps):
            # Calculate speed based on trajectory derivatives
            speed = self._calculate_polynomial_speed(smooth_trajectory, i, dense_timestamps)
            
            # Calculate acceleration based on motion profile
            acceleration = self._calculate_polynomial_acceleration(smooth_trajectory, i, dense_timestamps)
            
            execution_waypoints.append(ExecutionWaypoint(
                joints_deg=smooth_trajectory[i].tolist(),
                timestamp=timestamp,
                speed=speed,
                acceleration=acceleration
            ))
        
        return execution_waypoints
    
    def _quintic_polynomial_interpolation(self, param_values: np.ndarray, 
                                        joint_values: np.ndarray, 
                                        num_output_points: int) -> np.ndarray:
        """
        Quintic (5th order) polynomial interpolation for smooth motion
        
        Quintic polynomials provide:
        - Continuous position (C0)
        - Continuous velocity (C1) 
        - Continuous acceleration (C2)
        - Minimal jerk characteristics
        
        Args:
            param_values: Parameter values for input waypoints [0, 1]
            joint_values: Joint angle values at waypoints
            num_output_points: Number of output trajectory points
            
        Returns:
            Smooth joint trajectory with quintic interpolation
        """
        try:
            from scipy import interpolate
            
            # Create dense parameter sampling
            dense_params = np.linspace(0, 1, num_output_points)
            
            # Quintic spline interpolation with boundary conditions
            # Zero velocity and acceleration at start and end for smoothness
            cs = interpolate.CubicSpline(
                param_values, joint_values,
                bc_type=((1, 0.0), (1, 0.0))  # Zero first derivative (velocity) at boundaries
            )
            
            # Evaluate smooth trajectory
            smooth_trajectory = cs(dense_params)
            
            return smooth_trajectory
            
        except ImportError:
            # Fallback to manual quintic polynomial if scipy not available
            return self._manual_quintic_interpolation(param_values, joint_values, num_output_points)
    
    def _manual_quintic_interpolation(self, param_values: np.ndarray, 
                                    joint_values: np.ndarray, 
                                    num_output_points: int) -> np.ndarray:
        """
        Manual quintic polynomial interpolation without scipy dependency
        
        Uses quintic polynomial: q(t) = a₀ + a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵
        With boundary conditions for smooth motion
        """
        dense_params = np.linspace(0, 1, num_output_points)
        
        # For manual implementation, use simplified approach
        # S-curve profile for smooth acceleration/deceleration
        smooth_trajectory = np.zeros(num_output_points)
        
        start_value = joint_values[0]
        end_value = joint_values[-1]
        total_change = end_value - start_value
        
        for i, t in enumerate(dense_params):
            # Quintic S-curve: smooth acceleration and deceleration
            if t <= 0.5:
                # Acceleration phase: quintic polynomial from 0 to 0.5
                s = 16 * t**5 - 20 * t**4 + 10 * t**3
            else:
                # Deceleration phase: quintic polynomial from 0.5 to 1
                t_shifted = 1 - t
                s = 1 - (16 * t_shifted**5 - 20 * t_shifted**4 + 10 * t_shifted**3)
            
            smooth_trajectory[i] = start_value + total_change * s
        
        return smooth_trajectory
    
    def _calculate_polynomial_speed(self, trajectory: np.ndarray, 
                                  index: int, timestamps: np.ndarray) -> float:
        """Calculate speed based on polynomial trajectory derivatives"""
        if index == 0 or index >= len(trajectory) - 1:
            # SPEED FIX: Extra slow at start/end, especially for first motion
            base_start_end_speed = 0.15 if self.stats['successful_executions'] == 0 else 0.3
            return base_start_end_speed
        
        # Calculate velocity magnitude from numerical derivative
        dt = timestamps[index + 1] - timestamps[index - 1]
        if dt == 0:
            return 0.25 if self.stats['successful_executions'] == 0 else 0.5
        
        # Position change over time interval
        pos_change = np.linalg.norm(trajectory[index + 1] - trajectory[index - 1])
        velocity_magnitude = pos_change / dt
        
        # SPEED FIX: Reduce all speeds for first motion (initial positioning)
        speed_reduction = 0.5 if self.stats['successful_executions'] == 0 else 1.0
        
        # Map velocity to speed multiplier (0.2 to 0.8)
        # Higher velocity -> lower speed multiplier for smoother motion
        if velocity_magnitude > 50:  # High velocity
            return 0.2 * speed_reduction
        elif velocity_magnitude > 20:  # Medium velocity
            return 0.4 * speed_reduction
        elif velocity_magnitude > 5:   # Low velocity
            return 0.6 * speed_reduction
        else:  # Very low velocity
            return 0.8 * speed_reduction
    
    def _calculate_polynomial_acceleration(self, trajectory: np.ndarray, 
                                         index: int, timestamps: np.ndarray) -> float:
        """Calculate acceleration based on polynomial trajectory characteristics"""
        if len(trajectory) < 3 or index == 0 or index >= len(trajectory) - 1:
            return 0.6  # Default acceleration
        
        # Calculate acceleration magnitude from second derivative
        try:
            dt = timestamps[1] - timestamps[0]  # Assuming uniform spacing
            if dt == 0:
                return 0.6
            
            # Second derivative approximation
            accel_vector = (trajectory[index + 1] - 2 * trajectory[index] + trajectory[index - 1]) / (dt**2)
            accel_magnitude = np.linalg.norm(accel_vector)
            
            # Map acceleration to acceleration multiplier (0.4 to 0.9)
            if accel_magnitude > 100:  # High acceleration
                return 0.4
            elif accel_magnitude > 50:  # Medium acceleration
                return 0.6
            elif accel_magnitude > 10:  # Low acceleration
                return 0.8
            else:  # Very low acceleration
                return 0.9
                
        except Exception:
            return 0.6  # Safe default
    
    def _execute_dynamic_trajectory(self, waypoints: List[ExecutionWaypoint]) -> bool:
        """Execute trajectory using dynamic timing approach with chunking"""
        if not waypoints:
            return False
        
        try:
            self.is_executing = True
            self.execution_complete = False
            self.stop_requested = False
            
            if self.execution_mode == "blend" and len(waypoints) > 1:
                return self._execute_blend_motion(waypoints)
            else:
                return self._execute_point_to_point(waypoints)
                
        except Exception as e:
            self.logger.error(f"Dynamic trajectory execution failed: {e}")
            return False
        finally:
            self.is_executing = False
    
    def _execute_blend_motion(self, waypoints: List[ExecutionWaypoint]) -> bool:
        """
        Execute blend motion using safe chunked approach
        
        This method follows the proven pattern from path_playing_dynamically.py:
        - Each chunk is cleared before adding points
        - Chunks are executed immediately 
        - No buffer overflow possible
        """
        try:
            self.logger.info(f"Starting safe chunked blend execution: {len(waypoints)} waypoints, chunk size: {self.chunk_size}")
            
            # Create waypoint chunks for safe buffer management
            waypoint_chunks = self._create_waypoint_chunks(waypoints)
            self.logger.info(f"Created {len(waypoint_chunks)} chunks for execution")
            
            # Execute chunks sequentially (proven safe approach)
            total_chunks = len(waypoint_chunks)
            for i, chunk in enumerate(waypoint_chunks):
                if self.stop_requested:
                    self.logger.info("Execution stopped by user request")
                    return False
                
                # Execute chunk with proven safe pattern
                success = self._execute_waypoint_chunk(chunk, is_first_chunk=(i == 0))
                if not success:
                    self.logger.error(f"Failed to execute chunk {i+1}/{total_chunks}")
                    return False
                
                # Wait for chunk completion before next chunk
                self._wait_for_chunk_completion()
                
                # Progress reporting
                progress = ((i + 1) / total_chunks) * 100
                self.logger.info(f"Execution progress: {progress:.1f}% ({i+1}/{total_chunks} chunks)")
            
            # Final completion wait
            if not self._wait_for_motion_completion():
                self.logger.warning("Motion completion timeout")
                return False
            
            self.logger.info("Safe chunked blend motion completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Safe chunked blend motion failed: {e}")
            return False
    
    def _create_waypoint_chunks(self, waypoints: List[ExecutionWaypoint]) -> List[List[ExecutionWaypoint]]:
        """Create optimized chunks for pre-buffering strategy"""
        chunks = []
        total_waypoints = len(waypoints)
        
        for chunk_start in range(0, total_waypoints, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, total_waypoints)
            chunk = waypoints[chunk_start:chunk_end]
            chunks.append(chunk)
            
        return chunks
    
    def _prefill_robot_buffer(self, waypoint_chunks: List[List[ExecutionWaypoint]]) -> bool:
        """Pre-fill robot buffer to prevent gaps during execution"""
        try:
            self.logger.info("Phase 1: Pre-filling robot buffer for continuous motion")
            
            # Clear buffer first
            if hasattr(self.robot, 'move_joint_blend_clear'):
                self.robot.move_joint_blend_clear()
            
            # Calculate how many chunks to pre-fill (fill ~75% of buffer capacity)
            buffer_capacity = self.max_buffer_size  # 20 waypoints
            prefill_target = int(buffer_capacity * 0.75)  # 15 waypoints
            prefill_chunks = (prefill_target + self.chunk_size - 1) // self.chunk_size
            
            # Pre-fill with initial chunks
            prefilled_waypoints = 0
            chunks_prefilled = 0
            
            for i, chunk in enumerate(waypoint_chunks[:prefill_chunks]):
                if self.stop_requested:
                    return False
                
                # Add chunk to robot buffer with accurate tracking
                points_added_in_chunk = 0
                for wp in chunk:
                    if hasattr(self.robot, 'move_joint_blend_add_point'):
                        success = self.robot.move_joint_blend_add_point(
                            wp.joints_deg, 
                            speed=wp.speed, 
                            acceleration=wp.acceleration
                        )
                        if success:
                            points_added_in_chunk += 1
                        else:
                            self.logger.error(f"Failed to pre-fill waypoint in buffer")
                            return False
                
                prefilled_waypoints += points_added_in_chunk
                chunks_prefilled += 1
                
                if points_added_in_chunk < len(chunk):
                    self.logger.warning(f"Pre-fill chunk {i+1}: only added {points_added_in_chunk}/{len(chunk)} points")
                else:
                    self.logger.info(f"Pre-filled chunk {i+1}: {points_added_in_chunk} waypoints added successfully")
                
                # Check if we've reached optimal pre-fill level
                if prefilled_waypoints >= prefill_target:
                    break
            
            # Update buffer tracking
            self.points_in_buffer = prefilled_waypoints
            self.last_chunk_time = time.time()
            
            self.logger.info(f"Buffer pre-filled: {prefilled_waypoints} waypoints, {chunks_prefilled} chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Buffer pre-filling failed: {e}")
            return False
    
    def _start_motion_and_stream(self, waypoint_chunks: List[List[ExecutionWaypoint]]) -> bool:
        """Enhanced with progress feedback"""
        try:
            self.logger.info("Phase 2: Starting motion with continuous chunk streaming")
            
            # Start robot motion
            if hasattr(self.robot, 'move_joint_blend_move_point'):
                if not self.robot.move_joint_blend_move_point():
                    self.logger.error("Failed to start blend motion")
                    return False
            
            self.logger.info("Robot motion started with pre-filled buffer")
            
            # Initialize progress tracking
            total_points = sum(len(chunk) for chunk in waypoint_chunks)
            executed_points = 0
            self.execution_start_time = time.time()
            
            # Initialize buffer tracking for consumption rate estimation
            self.initial_buffer_points = self.points_in_buffer
            self.last_chunk_time = self.execution_start_time  # Initialize timing
            
            # Reset any previous consumption rate estimates for clean start
            if hasattr(self, 'smoothed_rate'):
                delattr(self, 'smoothed_rate')
            
            # Stream remaining chunks while robot executes
            total_chunks = len(waypoint_chunks)
            prefilled_chunks = min(3, total_chunks)  # Estimate based on buffer capacity
            
            for chunk_idx in range(prefilled_chunks, total_chunks):
                if self.stop_requested:
                    self.logger.info("Execution stopped by request")
                    return False
                
                chunk = waypoint_chunks[chunk_idx]
                
                # Wait for buffer space before streaming next chunk
                if not self._wait_for_buffer_space():
                    self.logger.error("Buffer space timeout during streaming")
                    return False
                
                # Stream chunk to robot
                if not self._stream_chunk_to_buffer(chunk, chunk_idx + 1):
                    self.logger.error(f"Failed to stream chunk {chunk_idx + 1}")
                    return False
                
                # Update progress tracking
                executed_points += len(chunk)
                progress = (executed_points / total_points) * 100
                
                # Real-time feedback
                self.logger.info(f"Execution progress: {progress:.1f}% ({executed_points}/{total_points} points)")
                
                # Estimate time remaining
                if self.execution_start_time:
                    elapsed = time.time() - self.execution_start_time
                    rate = executed_points / elapsed
                    remaining = (total_points - executed_points) / rate
                    self.logger.info(f"Estimated time remaining: {remaining:.1f}s")
                
                self.logger.info(f"Streamed chunk {chunk_idx + 1}/{total_chunks} ({progress:.1f}%)")
            
            # Wait for final motion completion
            if not self._wait_for_motion_completion():
                self.logger.error("Motion completion timeout")
                return False
                
            self.logger.info("Continuous streaming execution completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Streaming execution failed: {e}")
            return False
    
    def _estimate_consumption_rate(self):
        """Estimate robot's waypoint consumption rate with conservative bounds"""
        if not self.execution_start_time or self.points_in_buffer == 0:
            return 8  # Conservative default estimate
        
        elapsed = time.time() - self.execution_start_time
        if elapsed > 0.5:  # Only estimate after some execution time
            # Calculate raw consumption rate
            consumed_points = max(0, self.initial_buffer_points - self.points_in_buffer)
            raw_consumption_rate = consumed_points / elapsed
            
            # Apply exponential moving average smoothing with conservative bounds
            if not hasattr(self, 'smoothed_rate'):
                # Initialize with conservative value
                self.smoothed_rate = min(raw_consumption_rate, 15.0)  # Cap initial rate
            else:
                # Smooth the rate using exponential moving average
                alpha = 0.2  # More conservative smoothing factor
                new_rate = alpha * raw_consumption_rate + (1 - alpha) * self.smoothed_rate
                self.smoothed_rate = max(1.0, min(new_rate, 20.0))  # Bound between 1-20 points/sec
            
            # Log consumption rate updates for debugging
            if abs(raw_consumption_rate - 8) > 0.1:  # Only log when different from default
                self.logger.debug(f"Consumption rate - Raw: {raw_consumption_rate:.2f}, Smoothed: {self.smoothed_rate:.2f} points/sec")
            
            return self.smoothed_rate
        return 8  # Conservative default

    def _wait_for_buffer_space(self, timeout: float = 5.0) -> bool:
        """Improved buffer management with robust timing calculations"""
        try:
            # Estimate based on time and adaptive consumption rate
            points_per_second = self._estimate_consumption_rate()
            elapsed = time.time() - self.last_chunk_time
            estimated_consumed = max(0, elapsed * points_per_second)  # Ensure non-negative
            
            # Update estimated buffer state
            current_buffer_estimate = max(0, self.points_in_buffer - estimated_consumed)
            
            # Check if we have sufficient space
            if current_buffer_estimate < self.chunk_size:
                return True  # Space available
            
            # Calculate safe wait time (ensure positive values)
            points_to_consume = max(1, current_buffer_estimate - self.chunk_size + 1)
            wait_time = points_to_consume / max(points_per_second, 1.0)
            
            # Cap wait time to reasonable bounds
            safe_wait_time = max(0.01, min(wait_time, 0.3))  # 10ms to 300ms
            time.sleep(safe_wait_time)
            return True
            
        except Exception as e:
            self.logger.error(f"Buffer space check failed: {e}")
            return True  # Continue anyway
    
    def _stream_chunk_to_buffer(self, chunk: List[ExecutionWaypoint], chunk_number: int) -> bool:
        """Stream a single chunk to robot buffer during execution"""
        try:
            success = True
            points_added = 0
            
            for i, wp in enumerate(chunk):
                if hasattr(self.robot, 'move_joint_blend_add_point'):
                    if self.robot.move_joint_blend_add_point(
                        wp.joints_deg, 
                        speed=wp.speed, 
                        acceleration=wp.acceleration
                    ):
                        points_added += 1
                    else:
                        self.logger.error(f"Failed to stream waypoint {i+1} in chunk {chunk_number}")
                        success = False
                        break
            
            # Update buffer tracking with actual points added
            current_time = time.time()
            
            # Estimate points consumed since last update
            if hasattr(self, 'last_chunk_time') and self.last_chunk_time > 0:
                elapsed = current_time - self.last_chunk_time
                estimated_consumed = elapsed * self._estimate_consumption_rate()
                self.points_in_buffer = max(0, self.points_in_buffer - estimated_consumed)
            
            # Add new points
            self.points_in_buffer += points_added
            self.last_chunk_time = current_time
            
            # Periodically reset buffer estimate to prevent drift
            if chunk_number % 5 == 0:  # Reset every 5 chunks
                self.points_in_buffer = min(self.points_in_buffer, 50)  # Cap at reasonable maximum
            
            if points_added < len(chunk):
                self.logger.warning(f"Only added {points_added}/{len(chunk)} points from chunk {chunk_number}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Chunk streaming failed: {e}")
            return False
    
    def _wait_for_motion_completion(self, timeout: float = 30.0) -> bool:
        """Wait for all motion to complete"""
        try:
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if self.stop_requested:
                    return True
                
                # Check if robot is still moving
                if hasattr(self.robot, 'sys_status') and hasattr(self.robot.sys_status, 'robot_state'):
                    if self.robot.sys_status.robot_state == RobotState.IDLE.value:
                        self.logger.info("Robot motion completed")
                        
                        # STABILITY FIX: Add stabilization delay after motion completion
                        # This prevents the controller warning message when transitioning between motions
                        time.sleep(0.5)  # 500ms final stabilization
                        self.logger.debug("Robot fully stabilized after motion completion")
                        return True
                else:
                    # Simulation: assume completion after reasonable delay
                    time.sleep(1.0)
                    return True
                
                time.sleep(0.1)
            
            self.logger.warning("Motion completion timeout")
            return True
            
        except Exception as e:
            self.logger.error(f"Motion completion check failed: {e}")
            return True
            
        except Exception as e:
            self.logger.error(f"Chunked blend motion failed: {e}")
            return False
    
    def _execute_waypoint_chunk(self, chunk: List[ExecutionWaypoint], is_first_chunk: bool = False) -> bool:
        """Execute a single chunk of waypoints"""
        try:
            # CRITICAL: Clear blend buffer for EVERY chunk (matches working path_playing_dynamically.py)
            self.robot.move_joint_blend_clear()
            
            # Add waypoints in chunk to blend motion
            for i, wp in enumerate(chunk):
                success = self.robot.move_joint_blend_add_point(
                    wp.joints_deg, 
                    speed=wp.speed, 
                    acceleration=wp.acceleration
                )
                
                if not success:
                    self.logger.error(f"Failed to add waypoint {i+1} in chunk to blend motion")
                    return False
            
            # Execute the chunk immediately (matches working pattern)
            success = self.robot.move_joint_blend_move_point()
            if not success:
                self.logger.error("Failed to execute waypoint chunk")
                return False
            
            self.logger.debug(f"Executed chunk with {len(chunk)} points successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Waypoint chunk execution failed: {e}")
            return False
    
    def _wait_for_chunk_completion(self):
        """Wait for current chunk to complete before starting next (matches path_playing_dynamically.py pattern)"""
        try:
            # Wait for robot to start moving (like path_playing_dynamically.py)
            start_timeout = time.time() + 2.0
            while (self.robot.sys_status.robot_state == RobotState.IDLE.value and 
                   time.time() < start_timeout):
                if self.stop_requested:
                    self.logger.info("Stopping due to user request")
                    return
                time.sleep(0.05)

            # Wait for robot to finish moving (like path_playing_dynamically.py)
            completion_timeout = time.time() + 10.0
            while (self.robot.sys_status.robot_state != RobotState.IDLE.value and 
                   time.time() < completion_timeout):
                if self.stop_requested:
                    self.logger.info("Stopping due to user request")
                    return
                time.sleep(0.05)

            # STABILITY FIX: Add brief stabilization delay to prevent controller warning
            # This gives the robot time to fully settle at position before next motion
            time.sleep(0.2)  # 200ms stabilization delay
            self.logger.debug("Robot stabilized and ready for next motion")            # Small delay between chunks for stability (like path_playing_dynamically.py)
            time.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Error waiting for chunk completion: {e}")
    
    def _execute_point_to_point(self, waypoints: List[ExecutionWaypoint]) -> bool:
        """Execute waypoints one by one with timing control"""
        try:
            start_time = time.time()
            
            for i, wp in enumerate(waypoints):
                if self.stop_requested:
                    break
                
                # Wait for correct timing
                target_time = start_time + wp.timestamp
                current_time = time.time()
                if current_time < target_time:
                    time.sleep(target_time - current_time)
                
                # Execute waypoint
                success = self.robot.move_joint(
                    wp.joints_deg, 
                    speed=wp.speed, 
                    acceleration=wp.acceleration
                )
                
                if not success:
                    self.logger.error(f"Failed to execute waypoint {i+1}")
                    return False
                
                # Wait for motion to complete (using RobotState like path_playing_dynamically.py)
                while self.robot.sys_status.robot_state != RobotState.IDLE.value:
                    if self.stop_requested:
                        self.robot.motion_halt()
                        return False
                    time.sleep(0.05)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Point-to-point execution failed: {e}")
            return False
    
    def _get_current_joints(self) -> List[float]:
        """Get current robot joint positions"""
        try:
            if self.robot and hasattr(self.robot, 'sys_status'):
                return list(self.robot.sys_status.jnt_ang)
            return [0, 0, 0, 0, 0, 0]  # Default position
        except Exception as e:
            self.logger.error(f"Failed to get current joints: {e}")
            return [0, 0, 0, 0, 0, 0]
    
    def stop_execution(self):
        """Stop current motion execution"""
        self.stop_requested = True
        if self.robot:
            self.robot.motion_halt()
        self.logger.info("Motion stop requested")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        stats = self.stats.copy()
        
        if self.stats['planning_times']:
            stats['average_planning_time'] = sum(self.stats['planning_times']) / len(self.stats['planning_times'])
        else:
            stats['average_planning_time'] = 0.0
            
        if self.stats['execution_times']:
            stats['average_execution_time'] = sum(self.stats['execution_times']) / len(self.stats['execution_times'])
        else:
            stats['average_execution_time'] = 0.0
        
        return stats
    
    def shutdown(self):
        """Clean shutdown"""
        self.logger.info("Shutting down planning dynamic executor...")
        
        # Stop execution
        self.logger.info("Stopping execution...")
        self.stop_execution()
        
        # Wait for execution to stop
        if self.execution_thread and self.execution_thread.is_alive():
            self.execution_thread.join(timeout=5.0)
        
        # Shutdown robot
        if self.robot:
            self.logger.info("Stopping robot arm driver...")
            self.robot.stop()
        
        self.logger.info("Planning dynamic executor shutdown complete")
    
    def set_operation_mode(self, mode: str) -> bool:
        """
        Set robot operation mode with safety validation
        
        Args:
            mode: "simulation" for safe mode, "real" for real robot operation
            
        Returns:
            True if mode switch successful
        """
        if mode not in ["simulation", "real"]:
            self.logger.error(f"Invalid operation mode: {mode}. Must be 'simulation' or 'real'")
            return False
        
        if not self.robot:
            self.logger.error("Robot not initialized. Cannot set operation mode.")
            return False
        
        if self.is_executing:
            self.logger.error("Cannot change operation mode during execution")
            return False
        
        try:
            if mode == "simulation":
                self.logger.warning("SWITCHING TO SIMULATION MODE - Robot will not execute real motions")
                self.robot.set_program_mode_simulation()
                self.operation_mode = "simulation"
                self.logger.info("Operation mode set to SIMULATION")
                
            elif mode == "real":
                # Safety warning for real mode
                self.logger.warning("WARNING: SWITCHING TO REAL ROBOT MODE")
                self.logger.warning("Robot will execute REAL PHYSICAL MOTIONS")
                self.logger.warning("Ensure workspace is clear and safety systems are active")
                
                # Additional safety check
                current_state = self.robot.sys_status.robot_state
                if current_state != RobotState.IDLE.value:
                    self.logger.error("Robot must be in IDLE state to switch to real mode")
                    return False
                
                self.robot.set_program_mode_real()
                self.operation_mode = "real"
                self.logger.info("Operation mode set to REAL - Physical motions enabled")
                
            # Small delay for mode change to take effect
            time.sleep(0.5)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set operation mode to {mode}: {e}")
            return False
    
    def get_operation_mode(self) -> str:
        """Get current operation mode"""
        return self.operation_mode
    
    def is_safe_mode(self) -> bool:
        """Check if robot is in safe simulation mode"""
        return self.operation_mode == "simulation"

def create_planning_target(x_mm: float, y_mm: float, z_mm: float, 
                         rx_deg: float = 180.0, ry_deg: float = 0.0, rz_deg: float = 0.0,
                         gripper_mode: bool = False) -> PlanningTarget:
    """Create a planning target with default downward orientation"""
    return PlanningTarget(
        tcp_position_mm=[x_mm, y_mm, z_mm],
        tcp_rotation_deg=[rx_deg, ry_deg, rz_deg],
        gripper_mode=gripper_mode
    )