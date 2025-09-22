import json
import time
import threading
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import deque
from arm_driver import RobotArmDriver, PlatformInitStatus, RobotState
from robot_logger import RobotLogger


class SeamlessDynamicPlayer:
    """
    Seamless human motion imitation using real-time blend queue streaming.
    INCLUDES an improved method for identical, time-synchronized playback.
    """

    def __init__(self, robot_ip="192.168.0.10", queue_target_size=12):
        self.logger = RobotLogger("SeamlessDynamicPlayer")
        self.robot = RobotArmDriver(robot_ip=robot_ip, logger=self.logger)
        self.path_data: List[Dict[str, Any]] = []
        # Flag to track normal identical playback completion
        self.play_completed = False

        # --- Unused in new method ---
        self.queue_target_size = queue_target_size
        self.streaming_active = False
        self.streaming_thread = None
        self.point_stream = deque()
        self.stream_lock = threading.Lock()
        self.execution_start_time = None

    def load_data(self, json_file: str) -> bool:
        """Load and preprocess recorded path data."""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            if not data:
                self.logger.error("Empty data file")
                return False

            # Convert absolute timestamps to relative time from the start
            start_time = data[0]['timestamp']
            for point in data:
                point['relative_time'] = point['timestamp'] - start_time

            self.path_data = data
            total_time = data[-1]['relative_time']
            self.logger.info(
                f"Loaded {len(data)} points, total duration: {total_time:.2f}s")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            return False

    def setup_robot(self) -> bool:
        """Initialize robot with intelligent state checking."""
        if not self.robot.start():
            return False

        # Wait for connection
        self.logger.info("Connecting to robot...")
        timeout = time.time() + 10
        while time.time() < timeout:
            if all(self.robot.is_connected()):
                break
            time.sleep(0.5)
        else:
            self.logger.error("Connection timeout")
            return False

        time.sleep(1.0)

        # Check and set robot state if needed
        if self.robot.sys_status.init_state_info != PlatformInitStatus.INITIALIZATION_DONE.value:
            self.logger.info("Initializing robot...")
            if not self.robot.robot_init():
                return False
            # Wait for initialization to complete
            init_timeout = time.time() + 20
            while time.time() < init_timeout:
                if self.robot.sys_status.init_state_info == PlatformInitStatus.INITIALIZATION_DONE.value:
                    self.logger.info("Robot initialization completed")
                    break
                time.sleep(0.2)
            else:
                self.logger.error("Initialization timeout")
                return False

        # Set to simulation mode
        self.logger.info("Setting robot to simulation mode")
        self.robot.set_program_mode_simulation()
        time.sleep(0.5)

        return True

    def play_identically(self) -> bool:
        """
        Executes motion with identical timing and movement based on the file.
        Ensures each movement completes before sending the next command.
        """
        if not self.path_data:
            self.logger.error("No data loaded, cannot play.")
            return False

        # 1. Move to the starting position
        start_joints = self.path_data[0]['jnt_ang']
        self.logger.info(f"Moving to start position: {start_joints}")
        # Use high speed to get to the start position quickly
        if not self.robot.move_joint(start_joints, speed=1.0, acceleration=1.0):
            self.logger.error("Failed to move to start position.")
            return False

        # Wait until the robot is idle after reaching the start
        timeout = time.time() + 20
        while self.robot.sys_status.robot_state != RobotState.IDLE.value:
            if time.time() > timeout:
                self.logger.error("Timeout while moving to start position.")
                return False
            time.sleep(0.05)
        
        input("\nRobot at start position. Press Enter to begin identical playback...")

        # 2. Execute the timed motion loop with strict completion checking
        self.logger.info("Starting motion playback with completion tracking...")
        
        # Use a monotonic clock for precise timekeeping
        playback_start_time = time.monotonic()
        
        # Filter similar positions to reduce number of commands
        # Higher tolerance means fewer points and less UI flickering
        tolerance_degrees = 1.0
        
        # Pre-filter points to eliminate small movements
        filtered_points = []
        last_significant_joints = start_joints
        
        # Create filtered list of points with significant joint changes
        for i in range(1, len(self.path_data)):
            point = self.path_data[i]
            if point.get('relative_time', 0) == 0:
                continue
                
            current_joints = point['jnt_ang']
            
            # Check if any joint has moved significantly
            significant_change = False
            for j in range(len(current_joints)):
                if abs(current_joints[j] - last_significant_joints[j]) > tolerance_degrees:
                    significant_change = True
                    break
            
            # Only include points with significant movement
            if significant_change:
                filtered_points.append({
                    'joints': current_joints,
                    'time': point['relative_time']
                })
                last_significant_joints = current_joints
        
        self.logger.info(f"Filtered {len(self.path_data)} points down to {len(filtered_points)} significant waypoints")
        
        # Execute each filtered point
        for i, point in enumerate(filtered_points):
            # Calculate when this point should be executed according to the original timeline
            target_time = point['time']
            current_time = time.monotonic() - playback_start_time
            
            # Wait if we're ahead of schedule
            if current_time < target_time:
                wait_time = target_time - current_time
                time.sleep(wait_time)
            
            # Send the motion command
            self.robot.move_joint(point['joints'], speed=1.0, acceleration=1.0)
            
            # CRITICAL: Wait for the robot to COMPLETE this motion before continuing
            # This prevents command flooding and UI flickering
            motion_timeout = 10.0  # Maximum time to wait for motion completion
            
            # First wait for robot to start moving (it might take a moment)
            start_moving_timeout = time.time() + 2.0
            while self.robot.sys_status.robot_state != RobotState.MOVING.value:
                if time.time() > start_moving_timeout:
                    # If it doesn't start moving, it might already be at the position
                    break
                time.sleep(0.05)
            
            # Then wait for the robot to return to idle (movement complete)
            completion_timeout = time.time() + motion_timeout
            while self.robot.sys_status.robot_state == RobotState.MOVING.value:
                if time.time() > completion_timeout:
                    self.logger.warning(f"Motion timeout for point {i+1}/{len(filtered_points)}. Continuing anyway.")
                    break
                time.sleep(0.05)
            
            # Log progress periodically
            if (i+1) % 5 == 0 or i == len(filtered_points)-1:
                self.logger.info(f"Completed {i+1}/{len(filtered_points)} movements")

        # Final timing report
        total_expected = self.path_data[-1]['relative_time']
        total_actual = time.monotonic() - playback_start_time
        accuracy = 100 * (1 - abs(total_actual - total_expected) / total_expected) if total_expected > 0 else 100

        self.logger.info("Identical playback complete!")
        self.logger.info(f"Expected Duration: {total_expected:.2f}s, Actual Duration: {total_actual:.2f}s")
        self.logger.info(f"Timing Accuracy: {accuracy:.1f}%")

        # mark successful playback to avoid redundant halt
        self.play_completed = True
        return True
        
    def shutdown(self):
        """Clean shutdown."""
        try:
            # Only halt on abnormal shutdown (not after normal playback)
            if not self.play_completed and not self.robot.is_motion_idle():
                self.robot.motion_halt()
            time.sleep(0.5)
            self.robot.stop()
            self.logger.info("Seamless player shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# --- You also need to modify the main() function to call the new method ---

def main():
    """Main function for seamless human motion imitation"""
    player = SeamlessDynamicPlayer(robot_ip="192.168.0.10")

    try:
        if not player.load_data("dynamic_model.json"):
            return

        if not player.setup_robot():
            return
            
        # Execute the new, identical playback method
        if player.play_identically():
            print("\nIdentical motion playback completed successfully!")
        else:
            print("\nMotion playback failed.")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        player.robot.motion_halt()

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

    finally:
        player.shutdown()


if __name__ == "__main__":
    main()