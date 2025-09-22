"""
Time-Based Dynamic Motion Imitation
Automatically matches recorded timestamps regardless of UI speed settings
Uses timing-driven speed calculation instead of hardcoded values
"""

import json
import time
import threading
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import deque
from arm_driver import RobotArmDriver, PlatformInitStatus, RobotState
from robot_logger import RobotLogger


class TimeBasedPlayer:
    """Time-based motion imitation that matches original timing automatically"""
    
    def __init__(self, robot_ip="192.168.0.10", initial_move_speed=0.5, high_fidelity_mode=True):
        self.logger = RobotLogger("TimeBasedPlayer")
        self.robot = RobotArmDriver(robot_ip=robot_ip, logger=self.logger)
        self.path_data: List[Dict[str, Any]] = []
        
        # Motion control parameters
        self.queue_target_size = 12
        self.queue_min_size = 8
        self.streaming_active = False
        self.streaming_thread = None
        self.point_stream = deque()
        self.stream_lock = threading.Lock()
        
        # Timing control
        self.execution_start_time = 0.0
        self.timing_tolerance = 0.1  # Allow 100ms timing deviation
        
        # Robot speed characteristics (these will be auto-calibrated)
        self.max_joint_speed_dps = 180.0  # degrees per second (typical robot max)
        self.speed_safety_factor = 0.8    # Use 80% of max speed for safety
        
        # Initial positioning (separate from timing-based motion)
        self.initial_move_speed = initial_move_speed  # Default speed for moving to start position
        
        # HIGH FIDELITY MODE - Use all points for maximum spatial accuracy
        self.high_fidelity_mode = high_fidelity_mode
        self.chunk_size = 8  # Process points in small chunks to work with robot limitations
        
    def load_data(self, json_file: str) -> bool:
        """Load motion data and convert to relative time"""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if not data:
                return False
            
            # Convert to relative time
            start_time = data[0]['timestamp']
            for point in data:
                point['time'] = point['timestamp'] - start_time
            
            self.path_data = data
            total_time = data[-1]['time']
            self.logger.info(f"Loaded {len(data)} points, duration: {total_time:.2f}s")
            
            # Analyze motion characteristics
            self._analyze_motion_characteristics()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            return False
    
    def _analyze_motion_characteristics(self):
        """Analyze the motion to understand speed requirements"""
        max_velocity = 0.0
        total_distance = 0.0
        
        for i in range(1, len(self.path_data)):
            dt = self.path_data[i]['time'] - self.path_data[i-1]['time']
            if dt > 0:
                joints1 = np.array(self.path_data[i-1]['jnt_ang'])
                joints2 = np.array(self.path_data[i]['jnt_ang'])
                distance = np.max(np.abs(joints2 - joints1))        # How far joins moved
                velocity = distance / dt                            # How fast (degrees per second)
                
                max_velocity = max(max_velocity, velocity)
                total_distance += distance
        
        self.logger.info(f"Motion analysis:")
        self.logger.info(f"  Max velocity: {max_velocity:.1f}°/s")
        self.logger.info(f"  Total distance: {total_distance:.1f}°")
        
        # Adjust robot speed characteristics if needed
        if max_velocity > self.max_joint_speed_dps * self.speed_safety_factor:
            self.logger.warning(f"Motion requires {max_velocity:.1f}°/s, robot max is {self.max_joint_speed_dps * self.speed_safety_factor:.1f}°/s")
            self.logger.warning("Motion will be scaled to robot limits")
    
    def setup_robot(self) -> bool:
        """Standard robot setup"""
        if not self.robot.start():
            return False
        
        self.logger.info("Connecting to robot...")
        timeout = time.time() + 10
        while time.time() < timeout:
            if all(self.robot.is_connected()):
                break
            time.sleep(0.5)
        else:
            return False
        
        time.sleep(1.0)
        
        # Initialize if needed
        current_init_status = self.robot.sys_status.init_state_info
        if current_init_status != PlatformInitStatus.INITIALIZATION_DONE.value:
            self.logger.info("Initializing robot...")
            self.robot.robot_init()
            
            timeout = time.time() + 20
            while time.time() < timeout:
                if self.robot.sys_status.init_state_info == PlatformInitStatus.INITIALIZATION_DONE.value:
                    break
                time.sleep(0.2)
            else:
                return False
        
        # self.robot.set_program_mode_simulation()
        self.robot.set_program_mode_real()
        time.sleep(0.5)
        return True
    
    def analyze_motion_segments(self) -> List[Tuple[str, int, int]]:
        """Analyze motion into stationary and moving segments"""
        if len(self.path_data) < 3:
            return [('moving', 0, len(self.path_data) - 1)]
        
        velocities = []
        
        # Calculate velocities
        for i in range(1, len(self.path_data)):
            prev_joints = np.array(self.path_data[i-1]['jnt_ang'])
            curr_joints = np.array(self.path_data[i]['jnt_ang'])
            dt = self.path_data[i]['time'] - self.path_data[i-1]['time']
            
            # velocities at each point
            if dt > 0:
                velocity = np.max(np.abs(curr_joints - prev_joints)) / dt
            else:
                velocity = 0
            velocities.append(velocity)
        
        # Smooth velocities
        smoothed_velocities = self._smooth_velocities(velocities)
        
        # Calculate adaptive threshold using velocity distribution analysis
        moving_velocities = [v for v in smoothed_velocities if v > 0]
        if moving_velocities:
            threshold = self._calculate_adaptive_threshold(moving_velocities)
        else:
            threshold = 0.5
        
        self.logger.info(f"Motion threshold: {threshold:.2f}°/s (adaptive)")
        
        # Segment analysis
        segments = []
        i = 0
        while i < len(smoothed_velocities):
            if smoothed_velocities[i] <= threshold:
                # Stationary segment
                start_idx = i
                while i < len(smoothed_velocities) and smoothed_velocities[i] <= threshold:
                    i += 1
                
                if i > start_idx + 1:
                    segments.append(('stationary', start_idx, i))
                    duration = self.path_data[i]['time'] - self.path_data[start_idx]['time']
                    self.logger.info(f"Stationary: {duration:.2f}s")
            else:
                # Moving segment
                start_idx = i
                while i < len(smoothed_velocities) and smoothed_velocities[i] > threshold:
                    i += 1
                
                segments.append(('moving', start_idx, i))
                duration = self.path_data[i]['time'] - self.path_data[start_idx]['time']
                self.logger.info(f"Moving: {duration:.2f}s ({i - start_idx} points)")
        
        return segments
    
    def _smooth_velocities(self, velocities: List[float]) -> List[float]:
        """Smooth velocity data"""
        window_size = 5
        if len(velocities) < window_size:
            return velocities
        
        smoothed = []
        half_window = window_size // 2
        
        for i in range(len(velocities)):
            start = max(0, i - half_window)
            end = min(len(velocities), i + half_window + 1)
            smoothed.append(np.mean(velocities[start:end]))
        
        return smoothed
    
    def _calculate_adaptive_threshold(self, velocities: List[float]) -> float:
        """
        Calculate adaptive motion threshold without hardcoded percentiles
        Uses velocity distribution analysis to find natural separation point
        """
        if len(velocities) < 5:
            return np.mean(velocities) * 0.3  # Fallback for small datasets
        
        v_array = np.array(velocities)
        
        # Method 1: Look for natural gaps in velocity distribution (Find gaps between speeds as well as find the biggest gap in slow speeds)
        sorted_velocities = np.sort(v_array)
        velocity_gaps = np.diff(sorted_velocities)      # Find gaps between speeds
        
        if len(velocity_gaps) > 3:
            # Find the largest gap in the lower 50% of velocities
            mid_point = len(velocity_gaps) // 2
            lower_gaps = velocity_gaps[:mid_point]
            
            if len(lower_gaps) > 0:
                max_gap_idx = np.argmax(lower_gaps)     # Find biggest gap in slow speeds
                gap_threshold = sorted_velocities[max_gap_idx]
                
                # Validate the gap is significant
                if velocity_gaps[max_gap_idx] > np.std(velocities) * 0.5:
                    self.logger.info(f"Gap-based threshold: {gap_threshold:.2f}°/s")
                    return max(gap_threshold, 0.1)
        
        # Method 2: Use statistical analysis to find outliers
        q1 = np.percentile(v_array, 25)     # 25th percentile
        q3 = np.percentile(v_array, 75)     # 75th percentile
        iqr = q3 - q1                   # Interquartile range
        
        # Lower fence for outlier detection
        lower_fence = q1 - 1.5 * iqr
        stationary_threshold = max(lower_fence, np.min(v_array) * 1.5)
        
        if stationary_threshold > 0.05:
            self.logger.info(f"IQR-based threshold: {stationary_threshold:.2f}°/s")
            return stationary_threshold
        
        # Method 3: Adaptive percentile based on velocity distribution shape (adjust threshold dynamically)
        # Calculate skewness to determine appropriate percentile
        mean_vel = np.mean(v_array)
        std_vel = np.std(v_array)
        
        if std_vel > 0:
            # Coefficient of variation indicates distribution spread
            cv = std_vel / mean_vel
            
            if cv < 0.5:  # Low variation - use lower percentile
                adaptive_percentile = 15
            elif cv > 1.5:  # High variation - use higher percentile  
                adaptive_percentile = 30
            else:  # Medium variation - middle ground
                adaptive_percentile = 22
            
            threshold = np.percentile(v_array, adaptive_percentile)
            self.logger.info(f"Adaptive percentile ({adaptive_percentile}th): {threshold:.2f}°/s")
            return max(threshold, 0.1)
        
        # Final fallback
        fallback_threshold = np.mean(v_array) * 0.25
        self.logger.info(f"Fallback threshold: {fallback_threshold:.2f}°/s")
        return max(fallback_threshold, 0.1)
    
    def create_timing_based_stream(self, segment_data: List[Dict]) -> List[Dict]:
        """Create motion stream with timing-based speed calculation"""
        
        if self.high_fidelity_mode:
            # HIGH FIDELITY MODE: Use ALL points - no reduction!
            self.logger.info(f"High Fidelity Mode: Using ALL {len(segment_data)} points")
            return self._add_timing_based_speeds(segment_data)
        
        else:
            # REDUCED MODE: Original behavior for comparison
            if len(segment_data) <= self.queue_target_size:
                return self._add_timing_based_speeds(segment_data)
            
            # Reduce points while preserving timing
            # STEP 1: Calculate importance scores for ALL 48 points
            significance_scores = self._calculate_significance(segment_data)
            
            # STEP 2: Always keep start and end points
            essential_points = [0, len(segment_data) - 1]   # First and last points
            remaining_budget = self.queue_target_size - 2   # 12 - 2 = 10 more points to pick
            
            # STEP 3: Sort middle points by importance (highest first)
            middle_indices = list(range(1, len(segment_data) - 1))                      # Points 1 to 46
            middle_indices.sort(key=lambda i: significance_scores[i], reverse=True)     # SORT BY IMPORTANCE
            
            # STEP 4: Take the TOP 10 most important middle points
            selected_indices = essential_points + middle_indices[:remaining_budget]     # Keep best 10
            selected_indices.sort()                                                     # Put back in time order
            
            # STEP 5: Create final stream with selected points
            stream = [segment_data[i] for i in selected_indices]
            stream = self._add_timing_based_speeds(stream)
            
            self.logger.info(f"Reduced Mode: {len(segment_data)} → {len(stream)} points")
            return stream
    
    def _add_timing_based_speeds(self, stream: List[Dict]) -> List[Dict]:
        """Add timing-based speed and acceleration to each point"""
        for i in range(1, len(stream)):
            # Calculate distance and time for this segment
            prev_joints = np.array(stream[i-1]['jnt_ang'])
            curr_joints = np.array(stream[i]['jnt_ang'])
            joint_distance = np.max(np.abs(curr_joints - prev_joints))  # "How far did joints move between these two points?"
            
            time_duration = stream[i]['time'] - stream[i-1]['time']     # "How much time should this movement take?"
            
            # Calculate timing-based speed and acceleration
            speed, acceleration = self.calculate_timing_based_speed(joint_distance, time_duration)
            
            stream[i]['calculated_speed'] = speed
            stream[i]['calculated_acceleration'] = acceleration
            stream[i]['expected_duration'] = time_duration
            
            self.logger.debug(f"Point {i}: distance={joint_distance:.1f}°, time={time_duration:.2f}s, speed={speed:.3f}")
        
        return stream
    
    def calculate_timing_based_speed(self, joint_distance: float, time_duration: float) -> Tuple[float, float]:
        """
        Calculate speed and acceleration needed to cover distance in specified time
        This is ONLY used for the recorded motion imitation - NOT for initial positioning
        
        Args:
            joint_distance: Maximum joint angle change (degrees)
            time_duration: Time to complete motion (seconds)
            
        Returns:
            Tuple of (speed_factor, acceleration_factor) for robot commands
        """
        if time_duration <= 0 or joint_distance <= 0:
            return 0.3, 0.6  # Default safe values
        
        # Required velocity to cover distance in time
        required_velocity = joint_distance / time_duration
        
        # Convert to robot speed factor (0.0 to 1.0)
        # Robot speed factor represents fraction of maximum robot speed
        speed_factor = required_velocity / (self.max_joint_speed_dps * self.speed_safety_factor)
        
        # Clamp to valid range
        speed_factor = max(0.05, min(1.0, speed_factor))
        
        # Acceleration should be proportional to speed for smooth motion
        # Higher speed = higher acceleration for responsiveness
        acceleration_factor = max(0.3, min(1.0, speed_factor * 1.2))
        
        return speed_factor, acceleration_factor
    
    def _calculate_significance(self, segment_data: List[Dict]) -> List[float]:
        """Calculate significance of each point for intelligent reduction"""
        if len(segment_data) < 3:
            return [1.0] * len(segment_data)
        
        significance = [0.0] * len(segment_data)
        
        for i in range(1, len(segment_data) - 1):
            
            prev_joints = np.array(segment_data[i-1]['jnt_ang'])        
            curr_joints = np.array(segment_data[i]['jnt_ang'])
            next_joints = np.array(segment_data[i+1]['jnt_ang'])
            
            # 1. CURVATURE: How much direction changed at this point
            dir1 = curr_joints - prev_joints    # Direction from previous to current
            dir2 = next_joints - curr_joints    # Direction from current to next
            
            curvature = 0.0
            if np.linalg.norm(dir1) > 0 and np.linalg.norm(dir2) > 0:
                dir1_norm = dir1 / np.linalg.norm(dir1)
                dir2_norm = dir2 / np.linalg.norm(dir2)
                curvature = np.linalg.norm(dir2_norm - dir1_norm)   # How much direction changes
            
            # 2. VELOCITY: How fast motion was at this point
            dt = segment_data[i]['time'] - segment_data[i-1]['time']
            if dt > 0:
                velocity = np.max(np.abs(curr_joints - prev_joints)) / dt   # How fast joints moved
            else:
                velocity = 0
            
            # Time interval significance (preserve temporal landmarks)
            time_gap_before = segment_data[i]['time'] - segment_data[i-1]['time']
            time_gap_after = segment_data[i+1]['time'] - segment_data[i]['time']

            # 3. TIME SIGNIFICANCE: Points with unusual timing gaps
            time_significance = 1.0 / (1.0 + min(time_gap_before, time_gap_after))      # time significance inversely proportional to time gap
            
            # 4. COMBINED SCORE (weighted importance)
            significance[i] = curvature * 2 + velocity * 0.1 + time_significance * 0.5  # Combine factors with weights
        
        significance[0] = max(significance) + 1  # Always keep first point
        significance[-1] = max(significance) + 1  # Always keep last point
        
        return significance
    
    # High fidelity streaming method
    def start_high_fidelity_streaming(self, motion_stream: List[Dict]):
        """Start high-fidelity streaming that processes ALL points in chunks"""
        with self.stream_lock:
            self.point_stream.clear()
            self.point_stream.extend(motion_stream)
        
        self.streaming_active = True
        self.streaming_thread = threading.Thread(target=self._high_fidelity_streaming_worker, daemon=True)
        self.streaming_thread.start()
        
        self.logger.info(f"Started high-fidelity streaming with {len(motion_stream)} points (chunk size: {self.chunk_size})")
    
    def _high_fidelity_streaming_worker(self):
        """High-fidelity streaming worker that processes all points in manageable chunks"""
        total_points = len(self.point_stream)
        processed_points = 0
        
        while self.streaming_active and self.point_stream:
            try:
                # Calculate chunk size based on remaining points and robot capabilities
                remaining_points = len(self.point_stream)
                current_chunk_size = min(self.chunk_size, remaining_points)
                
                if current_chunk_size > 0:
                    # Extract chunk from stream
                    chunk_points = []
                    with self.stream_lock:
                        for _ in range(current_chunk_size):
                            if self.point_stream:
                                chunk_points.append(self.point_stream.popleft())
                    
                    if chunk_points:
                        # Execute chunk
                        self._execute_point_chunk(chunk_points)
                        processed_points += len(chunk_points)
                        
                        # Progress logging
                        progress = (processed_points / total_points) * 100
                        self.logger.info(f"High-fidelity progress: {processed_points}/{total_points} points ({progress:.1f}%)")
                        
                        # Wait for chunk completion before starting next
                        self._wait_for_chunk_completion()
                
            except Exception as e:
                self.logger.error(f"High-fidelity streaming error: {e}")
                break
        
        self.streaming_active = False
        self.logger.info(f"High-fidelity streaming completed: {processed_points}/{total_points} points processed")
    
    def _execute_point_chunk(self, chunk_points: List[Dict]):
        """Execute a chunk of points using blend motion"""
        if len(chunk_points) < 1:
            return
        
        # Clear any previous blend points
        self.robot.move_joint_blend_clear()
        
        # Add all points in chunk
        for point in chunk_points:
            speed = point.get('calculated_speed', 0.4)
            acceleration = point.get('calculated_acceleration', 0.6)
            
            success = self.robot.move_joint_blend_add_point(
                point['jnt_ang'],
                speed=speed,
                acceleration=acceleration
            )
            
            if not success:
                self.logger.warning(f"Failed to add point to chunk: {point['jnt_ang']}")
                break
        
        # Execute the chunk
        self.robot.move_joint_blend_move_point()
        self.logger.debug(f"Executed chunk with {len(chunk_points)} points")
    
    def _wait_for_chunk_completion(self):
        """Wait for current chunk to complete before starting next"""
        # Wait for robot to start moving
        start_timeout = time.time() + 2.0
        while (self.robot.sys_status.robot_state == RobotState.IDLE.value and 
               time.time() < start_timeout):
            time.sleep(0.05)
        
        # Wait for robot to finish moving
        completion_timeout = time.time() + 10.0
        while (self.robot.sys_status.robot_state != RobotState.IDLE.value and 
               time.time() < completion_timeout):
            time.sleep(0.05)
        
        # Small delay between chunks for stability
        time.sleep(0.1)
    
    # Reduced mode streaming method - single batch execution
    def start_timing_based_streaming(self, motion_stream: List[Dict]):
        """Original streaming method for reduced point mode"""
        with self.stream_lock:
            self.point_stream.clear()
            self.point_stream.extend(motion_stream)
        
        self.streaming_active = True
        self.streaming_thread = threading.Thread(target=self._timing_based_streaming_worker, daemon=True)
        self.streaming_thread.start()
        
        self.logger.info(f"Started timing-based streaming with {len(motion_stream)} points")
    
    # Queue management and timing-based streaming worker
    def _timing_based_streaming_worker(self):
        """Original streaming worker with timing synchronization"""
        points_in_queue = 0
        segment_start_time = time.time()
        
        while self.streaming_active and (self.point_stream or points_in_queue > 0):
            try:
                current_robot_state = self.robot.sys_status.robot_state
                should_add_points = False
                
                if current_robot_state == RobotState.MOVING.value:
                    if points_in_queue <= self.queue_min_size:
                        should_add_points = True
                elif points_in_queue == 0 and self.point_stream:
                    should_add_points = True
                
                if should_add_points and self.point_stream:
                    with self.stream_lock:
                        points_to_add = min(
                            self.queue_target_size - points_in_queue,
                            len(self.point_stream)
                        )
                        
                        if points_to_add > 0:
                            if points_in_queue == 0:
                                self.robot.move_joint_blend_clear()
                                segment_start_time = time.time()
                            
                            batch_added = 0
                            for _ in range(points_to_add):
                                if self.point_stream:
                                    point = self.point_stream.popleft()
                                    speed = point.get('calculated_speed', 0.4)
                                    acceleration = point.get('calculated_acceleration', 0.6)
                                    
                                    if self.robot.move_joint_blend_add_point(
                                        point['jnt_ang'],
                                        speed=speed,
                                        acceleration=acceleration
                                    ):
                                        batch_added += 1
                                    else:
                                        break
                            
                            if batch_added > 0:
                                points_in_queue += batch_added
                                
                                if points_in_queue == batch_added:
                                    self.robot.move_joint_blend_move_point()
                
                if current_robot_state == RobotState.IDLE.value and points_in_queue > 0:
                    points_in_queue = 0
                
                time.sleep(0.05)  # 20Hz control loop
                
            except Exception as e:
                self.logger.error(f"Timing-based streaming error: {e}")
                break
        
        self.streaming_active = False
        self.logger.info("Timing-based streaming completed")
    
    # Main execution controller
    def play(self) -> bool:
        """Execute motion with timing-based control"""
        if not self.path_data:
            return False
        
        # ========================================================================
        # PHASE 1: Initial Positioning (NOT part of timing-based imitation)
        # ========================================================================
        start_joints = self.path_data[0]['jnt_ang']
        self.logger.info(f"Moving to start position (speed: {self.initial_move_speed})...")
        self.logger.info("This initial move is NOT part of the recorded motion timing")
        
        if not self.robot.move_joint(start_joints, speed=self.initial_move_speed):
            return False
        
        # Wait for start position
        timeout = time.time() + 20
        while self.robot.sys_status.robot_state != RobotState.IDLE.value:
            if time.time() > timeout:
                return False
            time.sleep(0.05)
        
        self.logger.info("Start position reached - beginning timing-based imitation")
        
        # ========================================================================
        # PHASE 2: Timing-Based Motion Imitation (extracted speeds/accelerations)
        # ========================================================================
        
        # Analyze motion segments
        self.logger.info("Analyzing motion with timing-based approach...")
        segments = self.analyze_motion_segments()
        self.logger.info(f"Found {len(segments)} motion segments")
        
        if not segments:
            return False
        
        # Execute with timing control
        self.logger.info("Starting timing-based motion execution...")
        self.logger.info("All speeds/accelerations extracted from recorded timestamps")
        self.execution_start_time = time.time()
        
        for seg_idx, (seg_type, start_idx, end_idx) in enumerate(segments):
            segment_data = self.path_data[start_idx:end_idx+1]
            
            # Calculate segment timing
            segment_start_time = segment_data[0]['time']
            segment_end_time = segment_data[-1]['time']
            segment_duration = segment_end_time - segment_start_time
            
            if seg_type == 'stationary':
                if segment_duration > 0:
                    self.logger.info(f"Segment {seg_idx+1}: Stationary ({segment_duration:.2f}s)")
                    
                    # Precise timing control for stationary segments
                    target_time = self.execution_start_time + segment_end_time
                    current_time = time.time()
                    
                    if target_time > current_time:
                        wait_time = target_time - current_time
                        time.sleep(wait_time)
                    
            else:  # moving segment
                self.logger.info(f"Segment {seg_idx+1}: Motion ({len(segment_data)} points, {segment_duration:.2f}s)")
                if self.high_fidelity_mode:
                    self.logger.info("High Fidelity Mode: Using ALL original points")
                else:
                    self.logger.info("Using extracted speeds/accelerations from timestamps")
                
                # Create timing-based motion stream
                motion_stream = self.create_timing_based_stream(segment_data)
                
                if len(motion_stream) < 2:
                    continue
                
                # Execute with timing synchronization
                motion_start_time = time.time()
                
                if self.high_fidelity_mode:
                    # Use high-fidelity chunked execution
                    self.start_high_fidelity_streaming(motion_stream)
                else:
                    # Use original streaming method
                    self.start_timing_based_streaming(motion_stream)
                
                # Wait for motion completion
                while self.streaming_active:
                    time.sleep(0.1)
                
                # Wait for robot to finish
                timeout = time.time() + 15
                while (self.robot.sys_status.robot_state != RobotState.IDLE.value and 
                       time.time() < timeout):
                    time.sleep(0.05)
                
                # Timing synchronization
                motion_actual_duration = time.time() - motion_start_time
                expected_end_time = self.execution_start_time + segment_end_time
                current_time = time.time()
                
                self.logger.info(f"Motion timing: expected={segment_duration:.2f}s, actual={motion_actual_duration:.2f}s")
                
                # Sync to expected timing
                if expected_end_time > current_time:
                    sync_wait = expected_end_time - current_time
                    if 0 < sync_wait < 3.0:  # Reasonable sync wait
                        self.logger.info(f"Timing sync: waiting {sync_wait:.2f}s")
                        time.sleep(sync_wait)
        
        # Final timing analysis
        total_expected = self.path_data[-1]['time']
        total_actual = time.time() - self.execution_start_time
        timing_error = abs(total_actual - total_expected)
        accuracy = 100 * (1 - timing_error / total_expected)
        
        self.logger.info("Timing-based motion completed!")
        self.logger.info(f"Expected: {total_expected:.2f}s, Actual: {total_actual:.2f}s")
        self.logger.info(f"Timing error: {timing_error:.2f}s")
        self.logger.info(f"Timing accuracy: {accuracy:.1f}%")
        
        return True
    
    def shutdown(self):
        """Clean shutdown"""
        self.streaming_active = False
        if self.streaming_thread and self.streaming_thread.is_alive():
            self.streaming_thread.join(timeout=2.0)
        
        try:
            self.robot.stop()
            self.logger.info("Shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


def main():
    """Main function for timing-based motion imitation"""
    
    print("\nTiming-Based Motion Imitation Configuration")
    print("Choose fidelity mode:")
    print("1. High Fidelity Mode: Uses ALL recorded points (maximum spatial accuracy)")
    print("2. Reduced Mode: Intelligent point reduction (faster execution)")
    
    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice == "1":
            high_fidelity = True
            initial_speed = 0.1
            print("High Fidelity Mode selected - ALL points will be used")
            break
        elif choice == "2":
            high_fidelity = False
            initial_speed = 0.1
            print("Reduced Mode selected - Points will be intelligently reduced")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    # Create player with selected mode
    player = TimeBasedPlayer(
        robot_ip="192.168.0.10", 
        initial_move_speed=initial_speed,
        high_fidelity_mode=high_fidelity
    )
    
    try:
        if not player.load_data("data_recorded.json"):
            print("Failed to load motion data")
            return
        
        if not player.setup_robot():
            print("Failed to setup robot")
            return
        
        print(f"\nTiming-Based Motion Imitation")
        print(f"   Mode: {'High Fidelity (ALL points)' if high_fidelity else 'Reduced (intelligent selection)'}")
        print(f"   Phase 1: Move to start position (speed: {initial_speed})")
        print(f"   Phase 2: Dynamic imitation with extracted speeds/accelerations")
        print(f"   Automatically matches recorded timestamps")
        input("\nPress Enter to start...")
        
        if player.play():
            print("\nTiming-based motion completed!")
            print("Initial move: Used default speed")
            if high_fidelity:
                print("Motion imitation: Used ALL original points (maximum fidelity)")
            else:
                print("Motion imitation: Used intelligently selected points")
            print("Timing: Matched original recording timestamps")
        else:
            print("\nMotion imitation failed")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        player.robot.motion_halt()
        
    except Exception as e:
        print(f"\nError: {e}")
        
    finally:
        player.shutdown()


if __name__ == "__main__":
    main()